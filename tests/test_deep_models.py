from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from app.deep_artifacts import DeepArtifactStore
from recsys.deep import (
    DeepFM,
    DeepFMConfig,
    EarlyStopper,
    _apply_cold_start_feature_dropout,
    _load_checkpoint,
    _sampled_protocol_gate,
    _save_checkpoint,
    resolve_torch_device,
)
from recsys.pipeline import build_parser
from recsys.two_stage import (
    DEFAULT_RETRIEVAL_LIMITS,
    DEFAULT_SOURCE_CAPS_AT_10,
    UNIFIED_SOURCE_ORDER,
    UnifiedCandidate,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_deep_artifact(
    root: Path,
    version: str = "deep-test-v1",
    *,
    unified: bool = False,
    unified_v3: bool = False,
) -> Path:
    artifact = root / version
    artifact.mkdir(parents=True)
    categorical_sizes = (2, 2, len(UNIFIED_SOURCE_ORDER) + 1) if unified else (2, 2, 2, 2, 1)
    continuous_dim = (
        len(UNIFIED_SOURCE_ORDER) * (3 if unified_v3 else 2) + 3
        if unified
        else 5
    )
    config = DeepFMConfig(
        categorical_sizes=categorical_sizes, continuous_dim=continuous_dim
    )
    model = DeepFM(config)
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()},
        artifact / "deepfm.safetensors",
    )
    save_file({"placeholder": torch.ones(1)}, artifact / "dssm.safetensors")
    (artifact / "dssm_config.json").write_text("{}", encoding="utf-8")
    (artifact / "deepfm_config.json").write_text(
        json.dumps(
            {
                "categorical_sizes": list(categorical_sizes),
                "continuous_dim": continuous_dim,
                "embedding_dim": 8,
                "hidden_dims": [64, 32],
                "dropout": 0.1,
            }
        ),
        encoding="utf-8",
    )
    (artifact / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact / "training.json").write_text("{}", encoding="utf-8")
    arrays = {
        "deep_user_ids.npy": np.asarray([1], dtype=np.int64),
        "deep_item_ids.npy": np.asarray([10, 11], dtype=np.int64),
        "deep_item_model_indices.npy": np.asarray([1, 0], dtype=np.int64),
        "deep_user_embeddings.npy": np.ones((1, 64), dtype=np.float32),
        "deep_item_embeddings.npy": np.ones((2, 64), dtype=np.float32),
        "deep_user_profiles.npy": np.ones((1, 4), dtype=np.float32),
        "deep_item_content.npy": np.ones((2, 4), dtype=np.float32),
        "deep_item_popularity.npy": np.asarray([1.0, 0.0], dtype=np.float32),
        "deep_item_pop_bucket.npy": np.asarray([1, 0], dtype=np.int64),
        "deep_user_history_bucket.npy": np.asarray([1], dtype=np.int64),
    }
    if unified:
        arrays["deep_user_history_density.npy"] = np.asarray([0.5], dtype=np.float32)
    for name, value in arrays.items():
        np.save(artifact / name, value, allow_pickle=False)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _hash(path)}
        for path in artifact.iterdir()
        if path.is_file()
    }
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "dssm_deepfm_experiment",
                "model_version": version,
                "base_model_version": "stable-v1",
                "data_version": "data-v1",
                "files": files,
                "metrics": {},
                "serving": (
                    {
                        "mode": (
                            "unified_multisource_v3"
                            if unified_v3
                            else "unified_multisource_v2"
                        ),
                        "retrieval_top_n": 2,
                        "rank_strategy": "unified_deepfm",
                        "stable_rank_weight": 0.0,
                        "required_multimodal_version": "multimodal-v1",
                        "source_limits": DEFAULT_RETRIEVAL_LIMITS,
                        "source_caps_at_10": DEFAULT_SOURCE_CAPS_AT_10,
                    }
                    if unified
                    else {
                        "retrieval_top_n": 2,
                        "rank_strategy": "protected_top10_rerank",
                        "stable_rank_weight": 0.25,
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    pointer = root / "experiment-current.json"
    pointer.write_text(
        json.dumps({"manifest": f"{version}/manifest.json", "model_version": version}),
        encoding="utf-8",
    )
    return pointer


def test_early_stopping_and_checkpoint_restore_optimizer(tmp_path: Path) -> None:
    stopper = EarlyStopper(patience=2, min_delta=0.01)
    assert stopper.update(0.5, 1) == (True, False)
    assert stopper.update(0.505, 2) == (False, False)
    assert stopper.update(0.504, 3) == (False, True)
    assert stopper.best_epoch == 1

    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones((2, 3))).sum()
    loss.backward()
    optimizer.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    root = tmp_path / "checkpoints"
    _save_checkpoint(
        root,
        epoch=1,
        model=model,
        optimizer=optimizer,
        metadata={"seed": 7, "best_metric": 0.5, "best_epoch": 1},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    loaded_epoch, state = _load_checkpoint(root, model, optimizer)
    assert loaded_epoch == 1
    assert state["seed"] == 7
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_deepfm_forward_is_finite_and_uses_five_feature_groups() -> None:
    model = DeepFM(DeepFMConfig(categorical_sizes=(3, 4, 5, 6, 2)))
    categorical = torch.tensor([[1, 2, 3, 4, 1], [2, 3, 4, 5, 0]])
    continuous = torch.tensor(
        [[0.8, 0.2, 0.4, 0.0, 1.0], [0.3, -0.1, 0.1, 0.2, 0.0]]
    )
    values = model(categorical, continuous)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()


def test_device_auto_falls_back_to_cpu_and_explicit_cuda_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_torch_device("auto") == torch.device("cpu")
    assert resolve_torch_device("cpu") == torch.device("cpu")
    with pytest.raises(Exception, match="CUDA was requested"):
        resolve_torch_device("cuda")


def test_train_deep_cli_exposes_device_and_validation_mode() -> None:
    args = build_parser().parse_args(
        [
            "train-deep",
            "--device",
            "cuda",
            "--validation-mode",
            "sampled",
        ]
    )

    assert args.device == "cuda"
    assert args.validation_mode == "sampled"

    defaults = build_parser().parse_args(["train-deep"])
    assert defaults.device == "auto"
    assert defaults.validation_mode == "sampled"


def test_sampled_protocol_gate_rejects_full_catalog_diagnostics() -> None:
    cohort = {"protocol": "deterministic_sampled_negatives_v1"}
    assert _sampled_protocol_gate("sampled", cohort) is True
    assert _sampled_protocol_gate(
        "full", {"protocol": "complete_eligible_catalog_v1"}
    ) is False


def test_cold_start_dropout_masks_identity_and_collaborative_sources() -> None:
    categorical = np.ones((100, 3), dtype=np.int64)
    categorical[:, 0] = 3
    continuous = np.ones((100, len(UNIFIED_SOURCE_ORDER) * 3 + 3), dtype=np.float32)

    masked = _apply_cold_start_feature_dropout(
        categorical, continuous, seed=7, probability=0.2
    )

    assert 0 < masked.sum() < len(masked)
    svd = UNIFIED_SOURCE_ORDER.index("svd")
    item_cf = UNIFIED_SOURCE_ORDER.index("item_cf")
    assert np.all(continuous[masked, svd] == 0)
    assert np.all(continuous[masked, item_cf] == 0)
    assert np.all(continuous[masked, len(UNIFIED_SOURCE_ORDER) + svd] == 0)
    assert np.all(continuous[masked, len(UNIFIED_SOURCE_ORDER) + item_cf] == 0)
    assert np.all(continuous[masked, 2 * len(UNIFIED_SOURCE_ORDER) + svd] == 0)
    assert np.all(continuous[masked, 2 * len(UNIFIED_SOURCE_ORDER) + item_cf] == 0)
    assert np.all(continuous[masked, -2] == 1)
    assert np.all(categorical[masked, 2] >= 1)
    assert np.all(categorical[~masked, 0] == 3)


def test_deep_artifact_hash_validation_warmup_and_previous_good_fallback(
    tmp_path: Path,
) -> None:
    pointer = _write_deep_artifact(tmp_path)
    store = DeepArtifactStore(pointer)
    loaded = store.get()
    assert loaded is not None
    assert loaded.model_version == "deep-test-v1"
    assert loaded.rank_strategy == "protected_top10_rerank"
    assert loaded.stable_rank_weight == 0.25
    assert store.last_error is None

    bad_pointer = {"manifest": "missing/manifest.json", "model_version": "deep-bad"}
    pointer.write_text(json.dumps(bad_pointer) + "\n", encoding="utf-8")
    os.utime(pointer, None)
    assert store.get() is loaded
    assert store.last_error is not None


def test_corrupt_deep_artifact_never_loads_as_available(tmp_path: Path) -> None:
    pointer = _write_deep_artifact(tmp_path)
    manifest = json.loads((tmp_path / "deep-test-v1" / "manifest.json").read_text())
    manifest["files"]["deepfm.safetensors"]["sha256"] = "0" * 64
    (tmp_path / "deep-test-v1" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    store = DeepArtifactStore(pointer)
    assert store.get() is None
    assert "invalid deep artifact" in str(store.last_error)


def test_unified_artifact_loads_pinned_multimodal_contract_and_ranks(
    tmp_path: Path,
) -> None:
    pointer = _write_deep_artifact(tmp_path, "deep-unified-v2", unified=True)

    loaded = DeepArtifactStore(pointer).get()

    assert loaded is not None
    assert loaded.serving_mode == "unified_multisource_v2"
    assert loaded.rank_strategy == "unified_deepfm"
    assert loaded.required_multimodal_version == "multimodal-v1"
    candidate = UnifiedCandidate(
        item_id="10",
        source_scores=(1.0, 0.8, 0.0, 0.6, 0.0, 0.0, 0.0),
        source_mask=(1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        primary_source="svd",
        source_raw_scores=(0.9, 0.7, None, 0.5, None, None, None),
    )
    ranked = loaded.rank_unified(dataset_user_id="1", candidates=[candidate], limit=1)
    assert [value.item_id for value in ranked] == ["10"]
    assert np.isfinite(ranked[0].deepfm_score)


def test_unified_v3_loads_without_a_stable_ranking_prior(tmp_path: Path) -> None:
    pointer = _write_deep_artifact(
        tmp_path, "deep-unified-v3", unified=True, unified_v3=True
    )

    loaded = DeepArtifactStore(pointer).get()

    assert loaded is not None
    assert loaded.serving_mode == "unified_multisource_v3"
    assert loaded.stable_rank_weight == 0.0
