from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .exposure import load_exposure_training_rows
from .model import (
    CatalogEvaluationQuery,
    EvaluationQuery,
    SAMPLED_NEGATIVE_PROTOCOL,
    _rank_normalize,
    _read_processed_interactions,
    _select_users,
    build_catalog_evaluation_queries,
    build_evaluation_queries,
    build_sampled_all_items_queries,
    evaluate_catalog_queries,
    evaluate_queries,
    sampled_all_items_metrics,
)
from .two_stage import (
    DEFAULT_RETRIEVAL_LIMITS,
    DEFAULT_SOURCE_CAPS_AT_10,
    UNIFIED_FEATURE_SCHEMA_VERSION,
    UNIFIED_SOURCE_ORDER,
    apply_source_caps,
    calibrate_source_score,
)


class DeepTrainingError(RuntimeError):
    pass


def _sampled_protocol_gate(
    validation_mode: str,
    cohort: dict[str, Any],
) -> bool:
    return (
        validation_mode == "sampled"
        and cohort.get("protocol") == SAMPLED_NEGATIVE_PROTOCOL
    )


@dataclass(frozen=True)
class DSSMConfig:
    user_count: int
    item_id_count: int
    content_dim: int
    id_embedding_dim: int = 32
    hidden_dim: int = 128
    output_dim: int = 64
    dropout: float = 0.1
    temperature: float = 10.0


@dataclass(frozen=True)
class DeepFMConfig:
    categorical_sizes: tuple[int, ...]
    continuous_dim: int = 5
    embedding_dim: int = 8
    hidden_dims: tuple[int, int] = (64, 32)
    dropout: float = 0.1


@dataclass(frozen=True)
class VisualFeatureData:
    model_version: str
    item_embeddings: np.ndarray
    available: np.ndarray
    user_profiles: np.ndarray


@dataclass(frozen=True)
class _OfflineRankedCandidate:
    item_id: str
    source: str
    row: int


class DSSM(torch.nn.Module):
    def __init__(self, config: DSSMConfig):
        super().__init__()
        self.config = config
        self.user_id = torch.nn.Embedding(config.user_count + 1, config.id_embedding_dim)
        self.item_id = torch.nn.Embedding(config.item_id_count + 1, config.id_embedding_dim)
        self.user_tower = torch.nn.Sequential(
            torch.nn.Linear(config.id_embedding_dim + config.content_dim + 1, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, config.output_dim),
        )
        self.item_tower = torch.nn.Sequential(
            torch.nn.Linear(config.id_embedding_dim + config.content_dim + 2, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, config.output_dim),
        )
        torch.nn.init.normal_(self.user_id.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.item_id.weight, mean=0.0, std=0.02)

    def encode_user(
        self, user_ids: torch.Tensor, profiles: torch.Tensor, history_density: torch.Tensor
    ) -> torch.Tensor:
        values = torch.cat((self.user_id(user_ids), profiles, history_density[:, None]), dim=1)
        return torch.nn.functional.normalize(self.user_tower(values), dim=1)

    def encode_item(
        self,
        item_ids: torch.Tensor,
        content: torch.Tensor,
        popularity: torch.Tensor,
        cold: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat(
            (self.item_id(item_ids), content, popularity[:, None], cold[:, None]), dim=1
        )
        return torch.nn.functional.normalize(self.item_tower(values), dim=1)

    def forward(
        self,
        user_ids: torch.Tensor,
        profiles: torch.Tensor,
        history_density: torch.Tensor,
        item_ids: torch.Tensor,
        item_content: torch.Tensor,
        item_popularity: torch.Tensor,
        item_cold: torch.Tensor,
    ) -> torch.Tensor:
        users = self.encode_user(user_ids, profiles, history_density)
        batch, candidates = item_ids.shape
        items = self.encode_item(
            item_ids.reshape(-1),
            item_content.reshape(batch * candidates, -1),
            item_popularity.reshape(-1),
            item_cold.reshape(-1),
        ).reshape(batch, candidates, -1)
        return self.config.temperature * torch.einsum("bd,bcd->bc", users, items)


class DeepFM(torch.nn.Module):
    def __init__(self, config: DeepFMConfig):
        super().__init__()
        self.config = config
        self.linear_embeddings = torch.nn.ModuleList(
            torch.nn.Embedding(size, 1) for size in config.categorical_sizes
        )
        self.fm_embeddings = torch.nn.ModuleList(
            torch.nn.Embedding(size, config.embedding_dim) for size in config.categorical_sizes
        )
        self.continuous_linear = torch.nn.Linear(config.continuous_dim, 1)
        dnn_input = len(config.categorical_sizes) * config.embedding_dim + config.continuous_dim
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(dnn_input, config.hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_dims[1], 1),
        )
        for embedding in self.linear_embeddings:
            torch.nn.init.zeros_(embedding.weight)
        for embedding in self.fm_embeddings:
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        torch.nn.init.xavier_uniform_(self.continuous_linear.weight)
        torch.nn.init.zeros_(self.continuous_linear.bias)
        for layer in self.dnn:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        linear = self.continuous_linear(continuous)
        fields = []
        for index, (linear_embedding, fm_embedding) in enumerate(
            zip(self.linear_embeddings, self.fm_embeddings)
        ):
            field = categorical[:, index]
            linear = linear + linear_embedding(field)
            fields.append(fm_embedding(field))
        stacked = torch.stack(fields, dim=1)
        summed = stacked.sum(dim=1)
        fm = 0.5 * ((summed * summed) - (stacked * stacked).sum(dim=1)).sum(
            dim=1, keepdim=True
        )
        dense_input = torch.cat((*fields, continuous), dim=1)
        return (linear + fm + self.dnn(dense_input)).squeeze(1)


@dataclass
class EarlyStopper:
    patience: int = 2
    min_delta: float = 1e-4
    best: float = -math.inf
    best_epoch: int = 0
    stale_epochs: int = 0

    def update(self, metric: float, epoch: int) -> tuple[bool, bool]:
        improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.best_epoch = epoch
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_checkpoint(
    root: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> Path:
    destination = root / f"epoch-{epoch:03d}"
    temporary = root / f".epoch-{epoch:03d}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
        temporary / "model.safetensors",
    )
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    _atomic_json(temporary / "state.json", {**metadata, "epoch": epoch})
    os.replace(temporary, destination)
    _atomic_json(root / "latest.json", {"checkpoint": destination.name, "epoch": epoch})
    return destination


def _load_checkpoint(
    root: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict[str, Any]]:
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        return 0, {}
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = (root / str(latest["checkpoint"])).resolve()
    try:
        checkpoint.relative_to(root.resolve())
    except ValueError as exc:
        raise DeepTrainingError("checkpoint path escapes checkpoint root") from exc
    model.load_state_dict(load_file(checkpoint / "model.safetensors"))
    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    return int(state["epoch"]), state


def _read_pointer(pointer: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest = (pointer.parent / str(payload["manifest"])).resolve()
    try:
        manifest.relative_to(pointer.parent.resolve())
    except ValueError as exc:
        raise DeepTrainingError("base manifest escapes artifact root") from exc
    return manifest, json.loads(manifest.read_text(encoding="utf-8"))


def _read_items(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    item_ids: list[int] = []
    titles: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_id = int(row["item_id"])
            item_ids.append(item_id)
            titles[item_id] = str(row.get("title") or "")
    return np.asarray(item_ids, dtype=np.int64), titles


def _buckets(values: np.ndarray, boundaries: Sequence[int]) -> np.ndarray:
    return np.searchsorted(np.asarray(boundaries), values, side="right").astype(np.int64)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_torch_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise DeepTrainingError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise DeepTrainingError("CUDA was requested but is not available")
    use_cuda = requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    )
    return torch.device("cuda" if use_cuda else "cpu")


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _negative_samples(
    positives: Sequence[tuple[int, int, int]],
    *,
    user_position: dict[int, int],
    catalog_position: dict[int, int],
    train_item_positions: np.ndarray,
    popularity_probabilities: np.ndarray,
    known_by_user: dict[int, set[int]],
    seed: int,
    negatives_per_positive: int,
    max_interactions: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in positives if row[0] in user_position]
    if max_interactions is not None and len(selected) > max_interactions:
        selected = sorted(
            selected,
            key=lambda row: hashlib.sha256(f"{seed}:deep-row:{row}".encode()).digest(),
        )[:max_interactions]
    users = np.empty(len(selected), dtype=np.int64)
    items = np.empty((len(selected), negatives_per_positive + 1), dtype=np.int64)
    split = negatives_per_positive // 2
    for row_index, (user_id, item_id, _) in enumerate(selected):
        rng = np.random.default_rng(seed + row_index * 104729)
        users[row_index] = user_position[user_id]
        items[row_index, 0] = catalog_position[item_id]
        excluded = known_by_user[user_id]
        chosen: set[int] = set()
        while len(chosen) < negatives_per_positive:
            if len(chosen) < split:
                candidate = int(train_item_positions[int(rng.integers(0, len(train_item_positions)))])
            else:
                candidate = int(rng.choice(train_item_positions, p=popularity_probabilities))
            if candidate not in excluded:
                chosen.add(candidate)
        items[row_index, 1:] = sorted(chosen)
    return users, items


def _encode_dssm(
    model: DSSM,
    user_profiles: np.ndarray,
    history_density: np.ndarray,
    item_model_indices: np.ndarray,
    item_dense: np.ndarray,
    item_popularity: np.ndarray,
    item_cold: np.ndarray,
    *,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    users: list[np.ndarray] = []
    items: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(user_profiles), batch_size):
            stop = min(len(user_profiles), start + batch_size)
            encoded = model.encode_user(
                torch.arange(start + 1, stop + 1, dtype=torch.long, device=device),
                torch.from_numpy(user_profiles[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(history_density[start:stop]).to(device=device, dtype=torch.float32),
            )
            users.append(encoded.cpu().numpy())
        for start in range(0, len(item_dense), batch_size):
            stop = min(len(item_dense), start + batch_size)
            encoded = model.encode_item(
                torch.from_numpy(item_model_indices[start:stop]).to(device=device, dtype=torch.long),
                torch.from_numpy(item_dense[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(item_popularity[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(item_cold[start:stop]).to(device=device, dtype=torch.float32),
            )
            items.append(encoded.cpu().numpy())
    return np.vstack(users), np.vstack(items)


def _train_dssm_epoch(
    model: DSSM,
    optimizer: torch.optim.Optimizer,
    users: np.ndarray,
    candidate_items: np.ndarray,
    user_profiles: np.ndarray,
    history_density: np.ndarray,
    item_model_indices: np.ndarray,
    item_dense: np.ndarray,
    item_popularity: np.ndarray,
    item_cold: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> float:
    device = next(model.parameters()).device
    model.train()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(users), generator=generator).numpy()
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        user_rows = users[rows]
        item_rows = candidate_items[rows]
        logits = model(
            torch.from_numpy(user_rows + 1).to(device=device, dtype=torch.long),
            torch.from_numpy(user_profiles[user_rows]).to(device=device, dtype=torch.float32),
            torch.from_numpy(history_density[user_rows]).to(device=device, dtype=torch.float32),
            torch.from_numpy(item_model_indices[item_rows]).to(device=device, dtype=torch.long),
            torch.from_numpy(item_dense[item_rows]).to(device=device, dtype=torch.float32),
            torch.from_numpy(item_popularity[item_rows]).to(device=device, dtype=torch.float32),
            torch.from_numpy(item_cold[item_rows]).to(device=device, dtype=torch.float32),
        )
        loss = torch.nn.functional.cross_entropy(
            logits, torch.zeros(len(rows), dtype=torch.long, device=device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _deepfm_features(
    users: np.ndarray,
    items: np.ndarray,
    *,
    dssm_users: np.ndarray,
    dssm_items: np.ndarray,
    user_ids: np.ndarray,
    catalog_ids: np.ndarray,
    item_model_indices: np.ndarray,
    item_popularity: np.ndarray,
    item_pop_bucket: np.ndarray,
    user_history_bucket: np.ndarray,
    user_profiles: np.ndarray,
    item_dense: np.ndarray,
    base_user_index: dict[int, int],
    base_item_index: dict[int, int],
    base_user_factors: np.ndarray,
    base_item_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_users = users.reshape(-1)
    flat_items = items.reshape(-1)
    user_model_ids = flat_users + 1
    cats = np.column_stack(
        (
            user_model_ids,
            item_model_indices[flat_items],
            item_pop_bucket[flat_items],
            user_history_bucket[flat_users],
            np.zeros(len(flat_users), dtype=np.int64),
        )
    ).astype(np.int64)
    dssm_score = np.sum(dssm_users[flat_users] * dssm_items[flat_items], axis=1)
    content_score = np.sum(user_profiles[flat_users] * item_dense[flat_items], axis=1)
    svd_score = np.zeros(len(flat_users), dtype=np.float32)
    for index, (user_row, item_row) in enumerate(zip(flat_users, flat_items)):
        base_user = base_user_index.get(int(user_ids[user_row]))
        base_item = base_item_index.get(int(catalog_ids[item_row]))
        if base_user is not None and base_item is not None:
            svd_score[index] = float(base_user_factors[base_user] @ base_item_factors[base_item])
    continuous = np.column_stack(
        (
            dssm_score,
            svd_score,
            content_score,
            np.zeros(len(flat_users), dtype=np.float32),
            item_popularity[flat_items],
        )
    ).astype(np.float32)
    return cats, continuous


def _load_visual_features(
    pointer: Path,
    *,
    base_model_version: str,
    data_version: str,
    catalog_ids: np.ndarray,
    user_ids: np.ndarray,
) -> VisualFeatureData:
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest_path = (pointer.parent / str(pointer_payload["manifest"])).resolve()
    try:
        manifest_path.relative_to(pointer.parent.resolve())
    except ValueError as exc:
        raise DeepTrainingError("multimodal manifest escapes artifact root") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_type") != "mobilenet_text_fusion_experiment"
        or str(manifest.get("base_model_version")) != base_model_version
        or str(manifest.get("data_version")) != data_version
    ):
        raise DeepTrainingError("multimodal artifact is incompatible with deep training")
    names = (
        "visual_item_ids.npy",
        "visual_item_embeddings.npy",
        "visual_available.npy",
        "visual_user_ids.npy",
        "visual_user_profiles.npy",
    )
    files = dict(manifest.get("files") or {})
    for name in names:
        path = manifest_path.parent / name
        expected = str((files.get(name) or {}).get("sha256") or "")
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise DeepTrainingError(f"invalid multimodal feature file: {name}")
    visual_item_ids = np.load(
        manifest_path.parent / "visual_item_ids.npy", allow_pickle=False
    )
    visual_embeddings = np.load(
        manifest_path.parent / "visual_item_embeddings.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    visual_available = np.load(
        manifest_path.parent / "visual_available.npy", allow_pickle=False
    ).astype(bool, copy=False)
    visual_user_ids = np.load(
        manifest_path.parent / "visual_user_ids.npy", allow_pickle=False
    )
    visual_profiles = np.load(
        manifest_path.parent / "visual_user_profiles.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    item_lookup = {int(value): index for index, value in enumerate(visual_item_ids)}
    user_lookup = {int(value): index for index, value in enumerate(visual_user_ids)}
    if not set(map(int, catalog_ids)) <= set(item_lookup):
        raise DeepTrainingError("multimodal artifact does not cover the catalog")
    if not set(map(int, user_ids)) <= set(user_lookup):
        raise DeepTrainingError("multimodal artifact does not cover deep users")
    item_rows = np.asarray([item_lookup[int(value)] for value in catalog_ids], dtype=np.int64)
    user_rows = np.asarray([user_lookup[int(value)] for value in user_ids], dtype=np.int64)
    aligned_embeddings = visual_embeddings[item_rows]
    aligned_available = visual_available[item_rows]
    aligned_profiles = visual_profiles[user_rows]
    if (
        not np.isfinite(aligned_embeddings).all()
        or not np.isfinite(aligned_profiles).all()
    ):
        raise DeepTrainingError("multimodal features contain non-finite values")
    return VisualFeatureData(
        model_version=str(manifest["model_version"]),
        item_embeddings=aligned_embeddings,
        available=aligned_available,
        user_profiles=aligned_profiles,
    )


def _pair_cf_scores(
    flat_users: np.ndarray,
    flat_items: np.ndarray,
    *,
    user_ids: np.ndarray,
    catalog_ids: np.ndarray,
    base_user_index: dict[int, int],
    base_item_index: dict[int, int],
    item_cf_history: Any,
    item_cf_neighbors: Any,
) -> np.ndarray:
    result = np.zeros(len(flat_users), dtype=np.float32)
    mapped_users = np.asarray(
        [base_user_index.get(int(user_ids[row]), -1) for row in flat_users],
        dtype=np.int64,
    )
    mapped_items = np.asarray(
        [base_item_index.get(int(catalog_ids[row]), -1) for row in flat_items],
        dtype=np.int64,
    )
    valid_positions = np.flatnonzero((mapped_users >= 0) & (mapped_items >= 0))
    if valid_positions.size == 0:
        return result
    order = valid_positions[np.argsort(mapped_users[valid_positions], kind="stable")]
    start = 0
    while start < len(order):
        user_row = mapped_users[order[start]]
        stop = start + 1
        while stop < len(order) and mapped_users[order[stop]] == user_row:
            stop += 1
        positions = order[start:stop]
        targets = mapped_items[positions]
        result[positions] = (
            item_cf_history[user_row] @ item_cf_neighbors[:, targets]
        ).toarray().ravel()
        start = stop
    return result


def _unified_deepfm_features(
    users: np.ndarray,
    items: np.ndarray,
    *,
    dssm_users: np.ndarray,
    dssm_items: np.ndarray,
    user_ids: np.ndarray,
    catalog_ids: np.ndarray,
    item_model_indices: np.ndarray,
    item_popularity: np.ndarray,
    item_pop_bucket: np.ndarray,
    user_history_bucket: np.ndarray,
    history_density: np.ndarray,
    user_profiles: np.ndarray,
    item_dense: np.ndarray,
    visual: VisualFeatureData,
    base_user_index: dict[int, int],
    base_item_index: dict[int, int],
    base_user_factors: np.ndarray,
    base_item_factors: np.ndarray,
    item_cf_history: Any,
    item_cf_neighbors: Any,
    seed: int,
    source_limits: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original_shape = items.shape
    flat_users = users.reshape(-1)
    flat_items = items.reshape(-1)
    group_count, candidate_count = original_shape
    dssm_score = np.sum(dssm_users[flat_users] * dssm_items[flat_items], axis=1)
    content_score = np.sum(user_profiles[flat_users] * item_dense[flat_items], axis=1)
    visual_score = np.sum(
        visual.user_profiles[flat_users] * visual.item_embeddings[flat_items], axis=1
    )
    visual_available = visual.available[flat_items]
    mapped_users = np.asarray(
        [base_user_index.get(int(user_ids[row]), -1) for row in flat_users],
        dtype=np.int64,
    )
    mapped_items = np.asarray(
        [base_item_index.get(int(catalog_ids[row]), -1) for row in flat_items],
        dtype=np.int64,
    )
    svd_score = np.zeros(len(flat_users), dtype=np.float32)
    svd_valid = (mapped_users >= 0) & (mapped_items >= 0)
    svd_score[svd_valid] = np.sum(
        base_user_factors[mapped_users[svd_valid]]
        * base_item_factors[mapped_items[svd_valid]],
        axis=1,
    )
    cf_score = _pair_cf_scores(
        flat_users,
        flat_items,
        user_ids=user_ids,
        catalog_ids=catalog_ids,
        base_user_index=base_user_index,
        base_item_index=base_item_index,
        item_cf_history=item_cf_history,
        item_cf_neighbors=item_cf_neighbors,
    )
    user_values = user_ids[flat_users].astype(np.uint64)
    item_values = catalog_ids[flat_items].astype(np.uint64)
    explore_score = (
        (
            user_values * np.uint64(11400714819323198485)
            + item_values * np.uint64(14029467366897019727)
            + np.uint64(seed)
        )
        % np.uint64(1_000_003)
    ).astype(np.float64) / 1_000_003.0
    raw = np.column_stack(
        (
            svd_score,
            dssm_score,
            content_score,
            visual_score,
            cf_score,
            item_popularity[flat_items],
            explore_score,
        )
    ).reshape(group_count, candidate_count, len(UNIFIED_SOURCE_ORDER))
    eligible = np.column_stack(
        (
            svd_valid,
            np.isfinite(dssm_score),
            content_score > 0,
            visual_available & np.isfinite(visual_score),
            cf_score > 0,
            item_popularity[flat_items] > 0,
            np.ones(len(flat_users), dtype=bool),
        )
    ).reshape(group_count, candidate_count, len(UNIFIED_SOURCE_ORDER))
    selected = np.zeros_like(eligible)
    normalized = np.zeros_like(raw, dtype=np.float32)
    for source_index, source_name in enumerate(UNIFIED_SOURCE_ORDER):
        limit = max(0, int(source_limits[source_name]))
        if limit == 0:
            continue
        values = np.where(eligible[:, :, source_index], raw[:, :, source_index], -np.inf)
        order = np.argsort(-values, axis=1, kind="stable")
        # The caller supplies the complete eligible catalog. Select the exact same
        # per-source Top-N upper bound used by online candidate generation.
        take = min(candidate_count, limit)
        rows = np.repeat(np.arange(group_count), take)
        columns = order[:, :take].reshape(-1)
        allowed = eligible[rows, columns, source_index]
        selected[rows[allowed], columns[allowed], source_index] = True
        for row in range(group_count):
            chosen = np.flatnonzero(selected[row, :, source_index])
            if chosen.size == 0:
                continue
            chosen = chosen[
                np.lexsort(
                    (
                        catalog_ids[items[row, chosen]],
                        -raw[row, chosen, source_index],
                    )
                )
            ]
            denominator = max(1, len(chosen) - 1)
            normalized[row, chosen, source_index] = (
                1.0 - np.arange(len(chosen), dtype=np.float32) / denominator
            )
    included = selected.any(axis=2)
    primary = np.argmax(np.where(selected, normalized, -1.0), axis=2) + 1
    calibrated = np.zeros_like(raw, dtype=np.float32)
    for source_index, source_name in enumerate(UNIFIED_SOURCE_ORDER):
        positions = np.argwhere(selected[:, :, source_index])
        for group_row, candidate_row in positions:
            calibrated[group_row, candidate_row, source_index] = calibrate_source_score(
                source_name,
                float(raw[group_row, candidate_row, source_index]),
            )
    cats = np.column_stack(
        (
            item_pop_bucket[flat_items],
            user_history_bucket[flat_users],
            primary.reshape(-1),
        )
    ).astype(np.int64)
    continuous = np.column_stack(
        (
            normalized.reshape(-1, len(UNIFIED_SOURCE_ORDER)),
            calibrated.reshape(-1, len(UNIFIED_SOURCE_ORDER)),
            selected.astype(np.float32).reshape(-1, len(UNIFIED_SOURCE_ORDER)),
            history_density[flat_users],
            (item_model_indices[flat_items] == 0).astype(np.float32),
            visual_available.astype(np.float32),
        )
    ).astype(np.float32)
    return cats, continuous, included.reshape(-1)


def _build_replay_training_features(
    sample_users: np.ndarray,
    sample_items: np.ndarray,
    *,
    catalog_ids: np.ndarray,
    user_ids: np.ndarray,
    known_by_user: dict[int, set[int]],
    query_features: Callable[
        [int, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    negatives_per_positive: int,
    seed: int,
    cold_simulation_probability: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay each user's seven sources over the catalog before sampling ranker rows."""
    positives_by_user: dict[int, list[int]] = defaultdict(list)
    for user_row, item_rows in zip(sample_users, sample_items):
        positives_by_user[int(user_row)].append(int(item_rows[0]))
    categorical_rows: list[np.ndarray] = []
    continuous_rows: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    skipped_not_retrieved = 0
    replayed_union_sizes: list[int] = []
    source_memberships = {name: 0 for name in UNIFIED_SOURCE_ORDER}
    simulated_cold_groups = 0
    for user_row in sorted(positives_by_user):
        dataset_user_id = int(user_ids[user_row])
        cats, continuous, included = query_features(dataset_user_id, catalog_ids)
        included_rows = np.flatnonzero(included)
        replayed_union_sizes.append(int(included_rows.size))
        mask_offset = 2 * len(UNIFIED_SOURCE_ORDER)
        memberships = continuous[
            included_rows,
            mask_offset : mask_offset + len(UNIFIED_SOURCE_ORDER),
        ].sum(axis=0)
        for source_name, count in zip(UNIFIED_SOURCE_ORDER, memberships):
            source_memberships[source_name] += int(count)
        known = known_by_user[dataset_user_id]
        hard_pool = np.asarray(
            [row for row in included_rows if int(row) not in known], dtype=np.int64
        )
        if hard_pool.size:
            relevance = np.max(
                continuous[hard_pool, : len(UNIFIED_SOURCE_ORDER)], axis=1
            )
            hard_pool = hard_pool[
                np.lexsort((catalog_ids[hard_pool], -relevance))
            ]
        for positive_index, positive_row in enumerate(positives_by_user[user_row]):
            if not bool(included[positive_row]) or hard_pool.size < negatives_per_positive:
                skipped_not_retrieved += 1
                continue
            start = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:replay-hard:{dataset_user_id}:{positive_row}:{positive_index}".encode()
                ).digest()[:8],
                "little",
            ) % len(hard_pool)
            negative_rows = np.asarray(
                [hard_pool[(start + offset) % len(hard_pool)] for offset in range(negatives_per_positive)],
                dtype=np.int64,
            )
            selected = np.concatenate(
                (np.asarray([positive_row], dtype=np.int64), negative_rows)
            )
            selected_cats = cats[selected].copy()
            selected_continuous = continuous[selected].copy()
            simulate_cold = (
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}:simulate-cold:{dataset_user_id}:{positive_row}".encode()
                    ).digest()[:8],
                    "little",
                )
                / float(2**64)
                < cold_simulation_probability
            )
            semantic_indices = [
                UNIFIED_SOURCE_ORDER.index(name)
                for name in ("dssm", "content", "visual")
            ]
            mask_offset = 2 * len(UNIFIED_SOURCE_ORDER)
            semantic_available = any(
                selected_continuous[0, mask_offset + index] > 0
                for index in semantic_indices
            )
            positive_weight = 1.0
            if simulate_cold and semantic_available:
                for source_name in ("svd", "item_cf"):
                    source_row = UNIFIED_SOURCE_ORDER.index(source_name)
                    selected_continuous[0, source_row] = 0.0
                    selected_continuous[
                        0, len(UNIFIED_SOURCE_ORDER) + source_row
                    ] = 0.0
                    selected_continuous[0, mask_offset + source_row] = 0.0
                selected_continuous[0, -2] = 1.0
                remaining_mask = selected_continuous[
                    0, mask_offset : mask_offset + len(UNIFIED_SOURCE_ORDER)
                ]
                selected_cats[0, 2] = int(
                    np.argmax(
                        np.where(
                            remaining_mask > 0,
                            selected_continuous[0, : len(UNIFIED_SOURCE_ORDER)],
                            -1.0,
                        )
                    )
                ) + 1
                simulated_cold_groups += 1
                positive_weight = 2.0
            categorical_rows.append(selected_cats)
            continuous_rows.append(selected_continuous)
            labels.extend([1.0, *([0.0] * negatives_per_positive)])
            weights.extend([positive_weight, *([1.0] * negatives_per_positive)])
    if not categorical_rows:
        raise DeepTrainingError("full-catalog replay did not retrieve any training positives")
    return (
        np.vstack(categorical_rows),
        np.vstack(continuous_rows),
        np.asarray(labels, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        {
            "protocol": "complete-catalog-seven-source-replay-v1",
            "users": len(positives_by_user),
            "catalog_items": len(catalog_ids),
            "retrieved_positive_groups": len(categorical_rows),
            "skipped_positive_groups": skipped_not_retrieved,
            "mean_union_size": float(np.mean(replayed_union_sizes)),
            "min_union_size": min(replayed_union_sizes),
            "max_union_size": max(replayed_union_sizes),
            "source_memberships": source_memberships,
            "simulated_cold_positive_groups": simulated_cold_groups,
        },
    )


def _apply_cold_start_feature_dropout(
    categorical: np.ndarray,
    continuous: np.ndarray,
    *,
    seed: int,
    probability: float,
) -> np.ndarray:
    """Mask train-known item identity and collaborative sources without using holdout labels."""
    if not 0.0 <= probability < 1.0:
        raise DeepTrainingError("cold-start dropout probability must be in [0, 1)")
    mask = np.random.default_rng(seed).random(len(categorical)) < probability
    if not mask.any():
        return mask
    svd_index = UNIFIED_SOURCE_ORDER.index("svd")
    cf_index = UNIFIED_SOURCE_ORDER.index("item_cf")
    continuous[mask, svd_index] = 0.0
    continuous[mask, cf_index] = 0.0
    calibrated_offset = len(UNIFIED_SOURCE_ORDER)
    mask_offset = 2 * len(UNIFIED_SOURCE_ORDER)
    continuous[mask, calibrated_offset + svd_index] = 0.0
    continuous[mask, calibrated_offset + cf_index] = 0.0
    continuous[mask, mask_offset + svd_index] = 0.0
    continuous[mask, mask_offset + cf_index] = 0.0
    continuous[mask, -2] = 1.0
    source_scores = continuous[mask, : len(UNIFIED_SOURCE_ORDER)]
    source_masks = continuous[
        mask, mask_offset : 3 * len(UNIFIED_SOURCE_ORDER)
    ]
    categorical[mask, 2] = np.argmax(
        np.where(source_masks > 0, source_scores, -1.0), axis=1
    ) + 1
    return mask


def _deepfm_scores(
    model: DeepFM, categorical: np.ndarray, continuous: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(categorical), batch_size):
            stop = min(len(categorical), start + batch_size)
            values.append(
                model(
                    torch.from_numpy(categorical[start:stop]).to(device=device, dtype=torch.long),
                    torch.from_numpy(continuous[start:stop]).to(device=device, dtype=torch.float32),
                ).cpu().numpy()
            )
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _safe_baseline_scorer(
    *,
    base_user_ids: np.ndarray,
    base_item_ids: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    content_user_ids: np.ndarray,
    content_item_ids: np.ndarray,
    content_user_vectors: Any,
    content_item_vectors: Any,
    popularity_scores: dict[int, float],
) -> Any:
    user_index = {int(value): index for index, value in enumerate(base_user_ids)}
    item_index = {int(value): index for index, value in enumerate(base_item_ids)}
    content_user_index = {int(value): index for index, value in enumerate(content_user_ids)}
    content_item_index = {int(value): index for index, value in enumerate(content_item_ids)}

    def score(user_id: int, candidate_ids: np.ndarray) -> np.ndarray:
        user_row = user_index.get(user_id)
        content_user_row = content_user_index.get(user_id)
        svd_values = np.full(len(candidate_ids), -1e12, dtype=np.float64)
        content_values = np.zeros(len(candidate_ids), dtype=np.float64)
        available = np.zeros(len(candidate_ids), dtype=bool)
        for index, item_id in enumerate(map(int, candidate_ids)):
            item_row = item_index.get(item_id)
            if user_row is not None and item_row is not None:
                available[index] = True
                svd_values[index] = float(user_factors[user_row] @ item_factors[item_row])
            content_item_row = content_item_index.get(item_id)
            if content_user_row is not None and content_item_row is not None:
                content_values[index] = float(
                    content_user_vectors[content_user_row]
                    .multiply(content_item_vectors[content_item_row])
                    .sum()
                )
        cold_usable = (~available) & (content_values > 0)
        if not cold_usable.any():
            return svd_values
        popular_values = np.asarray(
            [popularity_scores.get(int(item_id), 0.0) for item_id in candidate_ids],
            dtype=np.float64,
        )
        blended = (
            0.55 * _rank_normalize(svd_values, candidate_ids)
            + 0.35 * _rank_normalize(content_values, candidate_ids)
            + 0.10 * _rank_normalize(popular_values, candidate_ids)
        )
        warm_order = [
            int(index)
            for index in np.lexsort((candidate_ids, -svd_values))
            if available[int(index)]
        ]
        cold_order = [
            int(index)
            for index in np.lexsort((candidate_ids, -content_values))
            if cold_usable[int(index)]
        ]
        selected = [*warm_order[:7], *cold_order[:3]]
        selected_set = set(selected)
        remainder = [
            int(index)
            for index in np.lexsort((candidate_ids, -blended))
            if int(index) not in selected_set
        ]
        scores = np.empty(len(candidate_ids), dtype=np.float64)
        for rank, candidate_index in enumerate([*selected, *remainder]):
            scores[candidate_index] = float(len(candidate_ids) - rank)
        return scores

    return score


def _deep_evaluation_markdown(
    metrics: dict[str, object], training_summary: dict[str, object]
) -> str:
    """Render a human-readable evaluation report for the deep two-stage experiment.

    The stable SVD path writes ``evaluation.md`` via ``_evaluation_markdown``; this is
    the equivalent for the DSSM recall + DeepFM rerank pipeline, so both training paths
    satisfy the "提交评估报告" acceptance criterion with a readable artifact.
    """
    protocol = dict(metrics.get("evaluation_protocol") or {})
    selected = dict(metrics.get("selected_rank_strategy") or {})
    lines = [
        "# Deep Two-Stage Offline Evaluation",
        "",
        "DSSM dual-tower recall over the complete eligible catalog, followed by DeepFM",
        "reranking of a seven-source candidate union (svd/dssm/content/visual/item_cf/"
        "popular/explore). Selection uses validation only; the test split is run once after",
        "the model is locked.",
        "",
        f"- Validation mode: `{protocol.get('validation_mode', 'n/a')}`",
        f"- Protocol: {protocol.get('protocol', 'n/a')}",
        f"- Candidate universe: {protocol.get('candidate_universe', 'n/a')}",
        f"- Sampled-negative usage: {protocol.get('sampled_negative_usage', 'n/a')}",
        f"- Cohort matches stable artifact: {protocol.get('cohort_matches_stable_artifact', 'n/a')}",
        f"- Selected rank strategy: {selected.get('name', 'n/a')}",
        f"- Publishable: `{bool(metrics.get('publishable'))}`",
        "",
    ]

    def _model_rows(models: object) -> list[str]:
        rows = ["| Model | Recall@10 | NDCG@10 | HitRate@10 |", "|---|---:|---:|---:|"]
        if isinstance(models, dict):
            for name, values in models.items():
                if not isinstance(values, dict):
                    continue
                rows.append(
                    f"| {name} | {float(values.get('recall@10', 0.0)):.6f} | "
                    f"{float(values.get('ndcg@10', 0.0)):.6f} | "
                    f"{float(values.get('hitrate@10', 0.0)):.6f} |"
                )
        return rows

    def _emit_split(split_name: str, split: object) -> None:
        if not isinstance(split, dict):
            return
        lines.append(f"## {split_name.title()}")
        lines.append("")
        if "status" in split and not any(isinstance(value, dict) for value in split.values()):
            lines.append(str(split["status"]))
            lines.append("")
            return
        model_names = [
            name for name in ("stable_safe", "dssm", "unified_multisource_deepfm")
            if name in split
        ]
        if model_names:
            lines.extend(_model_rows({name: split[name] for name in model_names}))
            lines.append("")
        warm = split.get("warm")
        if isinstance(warm, dict):
            lines.append("### Warm users only")
            lines.append("")
            lines.extend(_model_rows(warm))
            lines.append("")
        ablation = split.get("source_ablation")
        if isinstance(ablation, dict):
            lines.append("### Source ablation")
            lines.append("")
            ablated = {
                key: value
                for key, value in ablation.items()
                if key in {"without_dssm", "without_visual"} and isinstance(value, dict)
            }
            if ablated:
                lines.extend(_model_rows(ablated))
                lines.append("")

    _emit_split("validation", metrics.get("validation"))
    _emit_split("test", metrics.get("test"))

    validation = metrics.get("validation")
    if isinstance(validation, dict):
        auc = validation.get("deepfm_auc")
        linear = validation.get("linear_auc")
        if auc is not None or linear is not None:
            lines.extend(
                [
                    "## Ranking AUC (DeepFM vs linear baseline)",
                    "",
                    f"- DeepFM AUC: `{auc}`",
                    f"- Linear baseline AUC: `{linear}`",
                    "",
                ]
            )

    quality_gate = metrics.get("quality_gate")
    if isinstance(quality_gate, dict):
        lines.extend(["## Quality gate", "", "| Gate | Passed |", "|---|---:|"])
        for name, value in quality_gate.items():
            lines.append(f"| {name} | `{bool(value)}` |")
        lines.append("")

    cov = metrics.get("candidate_coverage")
    cold_cov = metrics.get("cold_candidate_coverage")
    if cov is not None or cold_cov is not None:
        lines.append("## Candidate coverage")
        lines.append("")
        if cov is not None:
            lines.append(f"- Candidate coverage: {float(cov):.6f}")
        if cold_cov is not None:
            lines.append(f"- Cold candidate coverage: {float(cold_cov):.6f}")
        lines.append("")

    dssm_summary = training_summary.get("dssm")
    deepfm_summary = training_summary.get("deepfm")
    if isinstance(dssm_summary, dict) or isinstance(deepfm_summary, dict):
        lines.append("## Training summary")
        lines.append("")
        if isinstance(dssm_summary, dict):
            lines.append(
                f"- DSSM: best epoch {dssm_summary.get('best_epoch')}, "
                f"validation recall@50 `{dssm_summary.get('best_validation_recall@50')}`, "
                f"early-stopped `{dssm_summary.get('early_stopped')}`"
            )
        if isinstance(deepfm_summary, dict):
            lines.append(
                f"- DeepFM: best epoch {deepfm_summary.get('best_epoch')}, "
                f"validation NDCG@10 `{deepfm_summary.get('best_validation_ndcg@10')}`, "
                f"early-stopped `{deepfm_summary.get('early_stopped')}`"
            )
        lines.append(
            f"- Train users: `{training_summary.get('train_users')}`, "
            f"train interactions: `{training_summary.get('train_interactions')}`"
        )
        lines.append(
            f"- Catalog items: `{training_summary.get('catalog_items')}`, "
            f"cold items: `{training_summary.get('cold_items')}`, "
            f"elapsed: `{training_summary.get('elapsed_seconds')}s`"
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "The unified DeepFM reranker scores a seven-source candidate union. Its value is",
            "measured against the stable SVD+content baseline (`stable_safe`) and the raw DSSM",
            "retriever (`dssm`) on the same locked cohort. `publishable=false` indicates the",
            "quality gate did not pass; the artifact remains eligible for serving via the",
            "experiment pointer.",
            "",
        ]
    )
    return "\n".join(lines)


def train_deep_experiment(
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    base_pointer: Path | None = None,
    multimodal_pointer: Path | None = None,
    mode: str = "smoke",
    max_users: int | None = None,
    max_train_interactions: int | None = None,
    max_eval_users: int | None = None,
    epochs: int = 8,
    patience: int = 2,
    retrieval_eval_top_n: int = 50,
    validation_mode: str = "sampled",
    device: str = "auto",
    run_test: bool = True,
    seed: int = 20260901,
    resume: bool = False,
    exposure_database: Path | None = None,
    publish: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in {"smoke", "full"}:
        raise DeepTrainingError("mode must be smoke or full")
    if epochs < 1 or patience < 1:
        raise DeepTrainingError("epochs and patience must be positive")
    if retrieval_eval_top_n < 10:
        raise DeepTrainingError("retrieval_eval_top_n must be at least 10")
    if validation_mode not in {"full", "sampled"}:
        raise DeepTrainingError("validation_mode must be full or sampled")
    if mode == "smoke":
        max_users = max_users or 5000
        max_train_interactions = max_train_interactions or 30000
        max_eval_users = max_eval_users or 1000
    _set_seed(seed)
    torch_device = resolve_torch_device(device)
    processed_dir = processed_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    base_pointer = (base_pointer or artifacts_dir / "current.json").resolve()
    multimodal_pointer = (
        multimodal_pointer or artifacts_dir / "multimodal-current.json"
    ).resolve()
    observation_end = datetime.now(timezone.utc).isoformat()
    if exposure_database is not None:
        exposure_rows, exposure_audit = load_exposure_training_rows(
            exposure_database.resolve(), observation_end=observation_end
        )
    else:
        exposure_rows, exposure_audit = [], {
            "status": "not_configured",
            "labeled_rows": 0,
        }
    base_manifest_path, base_manifest = _read_pointer(base_pointer)
    base_dir = base_manifest_path.parent
    summary = json.loads((processed_dir / "summary.json").read_text(encoding="utf-8"))
    train_rows = _read_processed_interactions(processed_dir / "train.csv")
    validation_rows = _read_processed_interactions(processed_dir / "validation.csv")
    test_rows = _read_processed_interactions(processed_dir / "test.csv")
    catalog_ids, _ = _read_items(processed_dir / "items.csv")
    selected_users = _select_users(
        (row[0] for row in train_rows), seed=seed, limit=max_users, purpose="deep-train"
    )
    user_ids = np.asarray(selected_users, dtype=np.int64)
    user_position = {int(value): index for index, value in enumerate(user_ids)}
    catalog_position = {int(value): index for index, value in enumerate(catalog_ids)}
    selected_train = [row for row in train_rows if row[0] in user_position]
    train_item_ids = np.asarray(sorted({row[1] for row in selected_train}), dtype=np.int64)
    train_item_position = {int(value): index + 1 for index, value in enumerate(train_item_ids)}
    catalog_model_indices = np.asarray(
        [train_item_position.get(int(value), 0) for value in catalog_ids], dtype=np.int64
    )
    cold = (catalog_model_indices == 0).astype(np.float32)
    visual = _load_visual_features(
        multimodal_pointer,
        base_model_version=str(base_manifest["model_version"]),
        data_version=str(summary["data_version"]),
        catalog_ids=catalog_ids,
        user_ids=user_ids,
    )

    content_item_ids = np.load(base_dir / "content_item_ids.npy", allow_pickle=False)
    content_item_vectors = load_npz(base_dir / "content_item_vectors.npz").tocsr()
    content_user_ids = np.load(base_dir / "content_user_ids.npy", allow_pickle=False)
    content_user_vectors = load_npz(base_dir / "content_user_vectors.npz").tocsr()
    content_position = {int(value): index for index, value in enumerate(content_item_ids)}
    train_content_rows = np.asarray(
        [content_position[int(value)] for value in train_item_ids if int(value) in content_position],
        dtype=np.int64,
    )
    content_dim = min(32, max(2, content_item_vectors.shape[1] - 1))
    reducer = TruncatedSVD(n_components=content_dim, n_iter=5, random_state=seed)
    reducer.fit(content_item_vectors[train_content_rows])
    item_dense = np.zeros((len(catalog_ids), content_dim), dtype=np.float32)
    present_catalog = np.asarray(
        [index for index, value in enumerate(catalog_ids) if int(value) in content_position],
        dtype=np.int64,
    )
    item_dense[present_catalog] = reducer.transform(
        content_item_vectors[
            np.asarray([content_position[int(catalog_ids[index])] for index in present_catalog])
        ]
    ).astype(np.float32)
    item_norms = np.linalg.norm(item_dense, axis=1, keepdims=True)
    item_dense /= np.maximum(item_norms, 1e-8)

    known_by_user: dict[int, set[int]] = defaultdict(set)
    history_count = np.zeros(len(user_ids), dtype=np.int64)
    user_profiles = np.zeros((len(user_ids), content_dim), dtype=np.float32)
    popularity_counts = np.zeros(len(catalog_ids), dtype=np.int64)
    for user_id, item_id, _ in selected_train:
        item_row = catalog_position[item_id]
        known_by_user[user_id].add(item_row)
        history_count[user_position[user_id]] += 1
        user_profiles[user_position[user_id]] += item_dense[item_row]
        popularity_counts[item_row] += 1
    profile_norms = np.linalg.norm(user_profiles, axis=1, keepdims=True)
    user_profiles /= np.maximum(profile_norms, 1e-8)
    history_density = (
        np.log1p(history_count).astype(np.float32) / max(1.0, float(np.log1p(history_count.max())))
    )
    user_history_bucket = _buckets(history_count, (5, 10, 20, 50))
    item_pop_bucket = _buckets(popularity_counts, (0, 1, 2, 4, 8, 16, 32, 64))
    item_popularity = (
        np.log1p(popularity_counts).astype(np.float32)
        / max(1.0, float(np.log1p(popularity_counts.max())))
    )
    train_catalog_positions = np.asarray(
        [catalog_position[int(value)] for value in train_item_ids], dtype=np.int64
    )
    popularity_weights = popularity_counts[train_catalog_positions].astype(np.float64)
    popularity_weights = np.maximum(popularity_weights, 1.0)
    popularity_probabilities = popularity_weights / popularity_weights.sum()
    sample_users, sample_items = _negative_samples(
        selected_train,
        user_position=user_position,
        catalog_position=catalog_position,
        train_item_positions=train_catalog_positions,
        popularity_probabilities=popularity_probabilities,
        known_by_user=known_by_user,
        seed=seed,
        negatives_per_positive=4,
        max_interactions=max_train_interactions,
    )

    validation_sampled_queries, validation_sampled_cohort = build_sampled_all_items_queries(
        split_name="validation",
        target_rows=validation_rows,
        known_rows=selected_train,
        model_user_ids=user_ids,
        catalog_item_ids=catalog_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    test_sampled_queries, test_sampled_cohort = build_sampled_all_items_queries(
        split_name="test",
        target_rows=test_rows,
        known_rows=[*selected_train, *validation_rows],
        model_user_ids=user_ids,
        catalog_item_ids=catalog_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    validation_sampled_warm_queries, validation_sampled_warm_cohort = build_evaluation_queries(
        split_name="validation",
        target_rows=validation_rows,
        known_rows=selected_train,
        model_user_ids=user_ids,
        model_item_ids=train_item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    test_sampled_warm_queries, test_sampled_warm_cohort = build_evaluation_queries(
        split_name="test",
        target_rows=test_rows,
        known_rows=[*selected_train, *validation_rows],
        model_user_ids=user_ids,
        model_item_ids=train_item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    validation_queries, validation_cohort = build_catalog_evaluation_queries(
        split_name="validation",
        target_rows=validation_rows,
        known_rows=selected_train,
        model_user_ids=user_ids,
        catalog_item_ids=catalog_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    test_queries, test_cohort = build_catalog_evaluation_queries(
        split_name="test",
        target_rows=test_rows,
        known_rows=[*selected_train, *validation_rows],
        model_user_ids=user_ids,
        catalog_item_ids=catalog_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    validation_warm_queries, validation_warm_cohort = build_catalog_evaluation_queries(
        split_name="validation-warm",
        target_rows=validation_rows,
        known_rows=selected_train,
        model_user_ids=user_ids,
        catalog_item_ids=catalog_ids,
        allowed_positive_item_ids=train_item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    test_warm_queries, test_warm_cohort = build_catalog_evaluation_queries(
        split_name="test-warm",
        target_rows=test_rows,
        known_rows=[*selected_train, *validation_rows],
        model_user_ids=user_ids,
        catalog_item_ids=catalog_ids,
        allowed_positive_item_ids=train_item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    if not validation_queries or not test_queries or not validation_warm_queries or not test_warm_queries:
        raise DeepTrainingError("deep evaluation cohort is empty")

    def catalog_eval(
        queries: Sequence[CatalogEvaluationQuery],
        scorer: Callable[[int, np.ndarray], np.ndarray],
        *,
        k: int = 10,
    ) -> dict[str, float]:
        return evaluate_catalog_queries(queries, catalog_ids, scorer, k=k)
    validation_queries_for_selection = (
        validation_queries if validation_mode == "full" else validation_sampled_queries
    )
    validation_warm_queries_for_selection = (
        validation_warm_queries
        if validation_mode == "full"
        else validation_sampled_warm_queries
    )
    test_queries_for_report = test_queries if validation_mode == "full" else test_sampled_queries
    test_warm_queries_for_report = (
        test_warm_queries if validation_mode == "full" else test_sampled_warm_queries
    )
    if not validation_queries_for_selection or not validation_warm_queries_for_selection:
        raise DeepTrainingError("selected deep evaluation cohort is empty")

    def validation_eval(
        queries: Sequence[Any],
        scorer: Callable[[int, np.ndarray], np.ndarray],
        *,
        k: int = 10,
    ) -> dict[str, float]:
        if validation_mode == "full":
            return catalog_eval(queries, scorer, k=k)
        return evaluate_queries(queries, scorer, k=k)
    base_metrics_for_cohort = dict(base_manifest.get("metrics") or {})
    base_validation_for_cohort = dict(base_metrics_for_cohort.get("validation") or {})
    base_test_for_cohort = dict(base_metrics_for_cohort.get("test") or {})
    expected_hashes = {
        "validation_full": (
            sampled_all_items_metrics(base_validation_for_cohort).get("cohort") or {}
        ).get("query_set_sha256"),
        "test_full": (
            sampled_all_items_metrics(base_test_for_cohort).get("cohort") or {}
        ).get("query_set_sha256"),
        "validation_warm": (
            base_validation_for_cohort.get("cohort") or {}
        ).get("query_set_sha256"),
        "test_warm": (base_test_for_cohort.get("cohort") or {}).get("query_set_sha256"),
    }
    actual_hashes = {
        "validation_full": validation_sampled_cohort["query_set_sha256"],
        "test_full": test_sampled_cohort["query_set_sha256"],
        "validation_warm": validation_sampled_warm_cohort["query_set_sha256"],
        "test_warm": test_sampled_warm_cohort["query_set_sha256"],
    }
    cohort_matches_stable_artifact = expected_hashes == actual_hashes

    run_config = {
        "algorithm": "pytorch_dssm_deepfm",
        "implementation_version": "unified-multisource-v3",
        "mode": mode,
        "seed": seed,
        "max_users": max_users,
        "max_train_interactions": max_train_interactions,
        "max_eval_users": max_eval_users,
        "epochs": epochs,
        "patience": patience,
        "retrieval_eval_top_n": retrieval_eval_top_n,
        "validation_mode": validation_mode,
        "requested_device": device,
        "resolved_device": str(torch_device),
        "dssm_negative_sampling": "4 per positive: 50% uniform + 50% train-only popularity",
        "deepfm_training_samples": "complete-catalog seven-source replay plus mature exposure labels",
        "exposure_database": str(exposure_database.resolve()) if exposure_database else None,
        "feature_cutoff_ms": int(summary["cutoffs"]["train_cutoff_ms"]),
        "data_version": summary["data_version"],
        "required_multimodal_version": visual.model_version,
        "source_limits": dict(DEFAULT_RETRIEVAL_LIMITS),
        "source_caps_at_10": dict(DEFAULT_SOURCE_CAPS_AT_10),
        "cold_start_feature_dropout": 0.0,
        "replay_cold_simulation_probability": 0.5,
        "deepfm_objective": "request_group_listwise_cross_entropy_plus_exposure_bce",
    }
    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    run_name = f"deep-{summary['data_version'][-8:]}-{config_hash}"
    checkpoint_root = artifacts_dir / "checkpoints" / run_name
    if checkpoint_root.exists() and not resume:
        checkpoint_root = checkpoint_root.with_name(
            f"{checkpoint_root.name}-{uuid.uuid4().hex[:8]}"
        )
    dssm_checkpoint_root = checkpoint_root / "dssm"
    deepfm_checkpoint_root = checkpoint_root / "deepfm"
    dssm_checkpoint_root.mkdir(parents=True, exist_ok=True)
    deepfm_checkpoint_root.mkdir(parents=True, exist_ok=True)

    dssm_config = DSSMConfig(
        user_count=len(user_ids), item_id_count=len(train_item_ids), content_dim=content_dim
    )
    dssm = DSSM(dssm_config).to(torch_device)
    dssm_optimizer = torch.optim.AdamW(dssm.parameters(), lr=1e-3, weight_decay=1e-5)
    start_epoch, resume_state = (
        _load_checkpoint(dssm_checkpoint_root, dssm, dssm_optimizer) if resume else (0, {})
    )
    _optimizer_to_device(dssm_optimizer, torch_device)
    dssm_stopper = EarlyStopper(
        patience=patience,
        best=float(resume_state.get("best_metric", -math.inf)),
        best_epoch=int(resume_state.get("best_epoch", 0)),
        stale_epochs=max(
            0, start_epoch - int(resume_state.get("best_epoch", start_epoch))
        ),
    )
    dssm_history: list[dict[str, Any]] = []
    best_dssm_path: Path | None = None
    catalog_lookup = {int(value): index for index, value in enumerate(catalog_ids)}
    user_lookup = {int(value): index for index, value in enumerate(user_ids)}
    dssm_epochs = (
        range(start_epoch + 1, epochs + 1)
        if dssm_stopper.stale_epochs < patience
        else ()
    )
    for epoch in dssm_epochs:
        loss = _train_dssm_epoch(
            dssm,
            dssm_optimizer,
            sample_users,
            sample_items,
            user_profiles,
            history_density,
            catalog_model_indices,
            item_dense,
            item_popularity,
            cold,
            batch_size=512,
            seed=seed + epoch,
        )
        encoded_users, encoded_items = _encode_dssm(
            dssm,
            user_profiles,
            history_density,
            catalog_model_indices,
            item_dense,
            item_popularity,
            cold,
        )

        def dssm_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
            user_row = user_lookup.get(int(user_id))
            if user_row is None:
                return np.full(len(candidates), -1e6, dtype=np.float64)
            rows = np.asarray([catalog_lookup[int(value)] for value in candidates])
            return encoded_items[rows] @ encoded_users[user_row]

        validation_recall50 = evaluate_queries(
            validation_sampled_queries, dssm_score, k=50
        )["recall@50"]
        improved, stop = dssm_stopper.update(validation_recall50, epoch)
        checkpoint = _save_checkpoint(
            dssm_checkpoint_root,
            epoch=epoch,
            model=dssm,
            optimizer=dssm_optimizer,
            metadata={
                "seed": seed,
                "cutoff": int(summary["cutoffs"]["train_cutoff_ms"]),
                "data_version": summary["data_version"],
                "feature_version": "title-svd32-v1",
                "config_hash": config_hash,
                "train_loss": loss,
                "validation_recall@50": validation_recall50,
                "best_metric": dssm_stopper.best,
                "best_epoch": dssm_stopper.best_epoch,
            },
        )
        if improved:
            best_dssm_path = checkpoint
            _atomic_json(
                dssm_checkpoint_root / "best.json",
                {"checkpoint": checkpoint.name, "epoch": epoch, "metric": validation_recall50},
            )
        dssm_history.append(
            {"epoch": epoch, "train_loss": loss, "validation_recall@50": validation_recall50}
        )
        if stop:
            break
    if best_dssm_path is None:
        best_pointer = json.loads((dssm_checkpoint_root / "best.json").read_text(encoding="utf-8"))
        best_dssm_path = dssm_checkpoint_root / str(best_pointer["checkpoint"])
    dssm.load_state_dict(load_file(best_dssm_path / "model.safetensors"))
    encoded_users, encoded_items = _encode_dssm(
        dssm,
        user_profiles,
        history_density,
        catalog_model_indices,
        item_dense,
        item_popularity,
        cold,
    )

    base_user_ids = np.load(base_dir / "user_ids.npy", allow_pickle=False)
    base_item_ids = np.load(base_dir / "item_ids.npy", allow_pickle=False)
    base_user_factors = np.load(base_dir / "user_factors.npy", allow_pickle=False)
    base_item_factors = np.load(base_dir / "item_factors.npy", allow_pickle=False)
    item_cf_history = load_npz(base_dir / "item_cf_user_history.npz").tocsr()
    item_cf_neighbors = load_npz(base_dir / "item_cf_neighbors.npz").tocsr()
    popularity_payload = json.loads((base_dir / "popularity.json").read_text(encoding="utf-8"))
    stable_popularity_scores = {
        int(row["item_id"]): float(row["score"])
        for row in popularity_payload.get("items", [])
    }
    base_user_index = {int(value): index for index, value in enumerate(base_user_ids)}
    base_item_index = {int(value): index for index, value in enumerate(base_item_ids)}
    source_limits = dict(DEFAULT_RETRIEVAL_LIMITS)
    source_caps_at_10 = dict(DEFAULT_SOURCE_CAPS_AT_10)
    deepfm_config = DeepFMConfig(
        categorical_sizes=(
            int(item_pop_bucket.max()) + 1,
            int(user_history_bucket.max()) + 1,
            len(UNIFIED_SOURCE_ORDER) + 1,
        ),
        continuous_dim=len(UNIFIED_SOURCE_ORDER) * 3 + 3,
    )
    deepfm = DeepFM(deepfm_config).to(torch_device)
    deepfm_optimizer = torch.optim.AdamW(deepfm.parameters(), lr=8e-4, weight_decay=1e-5)
    deepfm_start, deepfm_resume_state = (
        _load_checkpoint(deepfm_checkpoint_root, deepfm, deepfm_optimizer)
        if resume
        else (0, {})
    )
    _optimizer_to_device(deepfm_optimizer, torch_device)
    deepfm_stopper = EarlyStopper(
        patience=patience,
        best=float(deepfm_resume_state.get("best_metric", -math.inf)),
        best_epoch=int(deepfm_resume_state.get("best_epoch", 0)),
        stale_epochs=max(
            0, deepfm_start - int(deepfm_resume_state.get("best_epoch", deepfm_start))
        ),
    )

    def query_features(
        user_id: int, candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        user_row = user_lookup[int(user_id)]
        item_rows = np.asarray([catalog_lookup[int(value)] for value in candidates])
        return _unified_deepfm_features(
            np.full((1, len(item_rows)), user_row, dtype=np.int64),
            item_rows[None, :],
            dssm_users=encoded_users,
            dssm_items=encoded_items,
            user_ids=user_ids,
            catalog_ids=catalog_ids,
            item_model_indices=catalog_model_indices,
            item_popularity=item_popularity,
            item_pop_bucket=item_pop_bucket,
            user_history_bucket=user_history_bucket,
            history_density=history_density,
            user_profiles=user_profiles,
            item_dense=item_dense,
            visual=visual,
            base_user_index=base_user_index,
            base_item_index=base_item_index,
            base_user_factors=base_user_factors,
            base_item_factors=base_item_factors,
            item_cf_history=item_cf_history,
            item_cf_neighbors=item_cf_neighbors,
            seed=seed,
            source_limits=source_limits,
        )

    (
        train_cats,
        train_continuous,
        train_labels,
        train_weights,
        replay_training_summary,
    ) = _build_replay_training_features(
        sample_users,
        sample_items,
        catalog_ids=catalog_ids,
        user_ids=user_ids,
        known_by_user=known_by_user,
        query_features=query_features,
        negatives_per_positive=4,
        seed=seed,
        cold_simulation_probability=float(
            run_config["replay_cold_simulation_probability"]
        ),
    )
    exposure_cats: list[list[int]] = []
    exposure_continuous: list[list[float]] = []
    exposure_labels: list[float] = []
    exposure_weights: list[float] = []
    source_index = {name: index + 1 for index, name in enumerate(UNIFIED_SOURCE_ORDER)}
    if exposure_audit.get("status") == "usable":
        for sample in exposure_rows:
            user_row = user_lookup.get(sample.dataset_user_id)
            item_row = catalog_lookup.get(sample.item_id)
            if (
                user_row is None
                or item_row is None
                or sample.primary_source not in source_index
                or sample.feature_schema_version != UNIFIED_FEATURE_SCHEMA_VERSION
                or len(sample.source_scores) != len(UNIFIED_SOURCE_ORDER)
                or len(sample.source_calibrated_scores) != len(UNIFIED_SOURCE_ORDER)
                or len(sample.source_mask) != len(UNIFIED_SOURCE_ORDER)
            ):
                continue
            exposure_cats.append(
                [
                    user_row + 1,
                    int(catalog_model_indices[item_row]),
                    int(item_pop_bucket[item_row]),
                    int(user_history_bucket[user_row]),
                    source_index[sample.primary_source],
                ]
            )
            exposure_continuous.append(
                [
                    *sample.source_scores,
                    *sample.source_calibrated_scores,
                    *sample.source_mask,
                    float(history_density[user_row]),
                    float(catalog_model_indices[item_row] == 0),
                    float(visual.available[item_row]),
                ]
            )
            exposure_labels.append(sample.label)
            exposure_weights.append(sample.sample_weight)
    exposure_audit = {
        **exposure_audit,
        "used_rows": len(exposure_labels),
        "training_mode": (
            "replay_plus_exposure"
            if exposure_labels
            else "replay_supervised_exposure_insufficient"
        ),
    }
    if exposure_labels:
        train_cats = np.vstack(
            (train_cats, np.asarray(exposure_cats, dtype=np.int64))
        )
        train_continuous = np.vstack(
            (train_continuous, np.asarray(exposure_continuous, dtype=np.float32))
        )
        train_labels = np.concatenate(
            (train_labels, np.asarray(exposure_labels, dtype=np.float32))
        )
        train_weights = np.concatenate(
            (train_weights, np.asarray(exposure_weights, dtype=np.float32))
        )
    cold_dropout_mask = _apply_cold_start_feature_dropout(
        train_cats,
        train_continuous,
        seed=seed + 70_001,
        probability=float(run_config["cold_start_feature_dropout"]),
    )
    if not train_labels.any() or train_labels.all():
        raise DeepTrainingError("replayed candidate training set must contain both classes")

    def two_stage_score(
        user_id: int,
        candidates: np.ndarray,
        disabled_sources: frozenset[str] = frozenset(),
    ) -> np.ndarray:
        categorical, continuous, included = query_features(user_id, candidates)
        if disabled_sources:
            mask_offset = 2 * len(UNIFIED_SOURCE_ORDER)
            for source in disabled_sources:
                source_row = UNIFIED_SOURCE_ORDER.index(source)
                continuous[:, source_row] = 0.0
                continuous[:, len(UNIFIED_SOURCE_ORDER) + source_row] = 0.0
                continuous[:, mask_offset + source_row] = 0.0
            source_scores = continuous[:, : len(UNIFIED_SOURCE_ORDER)]
            source_masks = continuous[
                :, mask_offset : 3 * len(UNIFIED_SOURCE_ORDER)
            ]
            included = included & source_masks.any(axis=1)
            categorical[:, 2] = np.argmax(
                np.where(source_masks > 0, source_scores, -1.0), axis=1
            ) + 1
        result = np.full(len(candidates), -1e6, dtype=np.float64)
        included_rows = np.flatnonzero(included)
        if included_rows.size == 0:
            return result
        deep_scores = _deepfm_scores(deepfm, categorical[included], continuous[included])
        primary_sources = [
            UNIFIED_SOURCE_ORDER[int(value) - 1]
            for value in categorical[included, 2]
        ]
        item_ids = candidates[included_rows]
        ranked_rows = np.lexsort((item_ids, -deep_scores))
        ranked = [
            _OfflineRankedCandidate(
                item_id=str(candidates[included_rows[position]]),
                source=primary_sources[position],
                row=int(included_rows[position]),
            )
            for position in ranked_rows
        ]
        capped, _ = apply_source_caps(
            ranked, limit=min(10, len(ranked)), caps_at_10=source_caps_at_10
        )
        for rank, candidate in enumerate(capped):
            result[candidate.row] = float(len(capped) - rank)
        return result

    deepfm_history: list[dict[str, Any]] = []
    best_deepfm_path: Path | None = None
    deepfm_epochs = (
        range(deepfm_start + 1, epochs + 1)
        if deepfm_stopper.stale_epochs < patience
        else ()
    )
    for epoch in deepfm_epochs:
        deepfm.train()
        deepfm_device = next(deepfm.parameters()).device
        generator = torch.Generator().manual_seed(seed + 10_000 + epoch)
        replay_group_count = int(replay_training_summary["retrieved_positive_groups"])
        replay_group_width = 5
        replay_row_count = replay_group_count * replay_group_width
        group_order = torch.randperm(replay_group_count, generator=generator).numpy()
        epoch_losses: list[float] = []
        for start in range(0, len(group_order), 256):
            groups = group_order[start : start + 256]
            rows = (
                groups[:, None] * replay_group_width
                + np.arange(replay_group_width, dtype=np.int64)[None, :]
            ).reshape(-1)
            logits = deepfm(
                torch.from_numpy(train_cats[rows]).to(device=deepfm_device, dtype=torch.long),
                torch.from_numpy(train_continuous[rows]).to(device=deepfm_device, dtype=torch.float32),
            ).reshape(-1, replay_group_width)
            per_group_loss = torch.nn.functional.cross_entropy(
                logits,
                torch.zeros(len(groups), dtype=torch.long, device=deepfm_device),
                reduction="none",
            )
            positive_rows = groups * replay_group_width
            group_weights = torch.from_numpy(train_weights[positive_rows]).to(
                device=deepfm_device, dtype=torch.float32
            )
            loss = (per_group_loss * group_weights).sum() / group_weights.sum()
            deepfm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deepfm.parameters(), 5.0)
            deepfm_optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if replay_row_count < len(train_labels):
            exposure_rows = np.arange(replay_row_count, len(train_labels), dtype=np.int64)
            logits = deepfm(
                torch.from_numpy(train_cats[exposure_rows]).to(device=deepfm_device, dtype=torch.long),
                torch.from_numpy(train_continuous[exposure_rows]).to(device=deepfm_device, dtype=torch.float32),
            )
            per_row_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                torch.from_numpy(train_labels[exposure_rows]).to(
                    device=deepfm_device, dtype=torch.float32
                ),
                reduction="none",
            )
            row_weights = torch.from_numpy(train_weights[exposure_rows]).to(
                device=deepfm_device, dtype=torch.float32
            )
            exposure_loss = (
                (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(1.0)
            )
            deepfm_optimizer.zero_grad(set_to_none=True)
            exposure_loss.backward()
            torch.nn.utils.clip_grad_norm_(deepfm.parameters(), 5.0)
            deepfm_optimizer.step()
            epoch_losses.append(float(exposure_loss.detach().cpu()))
        validation_ndcg = evaluate_queries(
            validation_sampled_queries, two_stage_score
        )["ndcg@10"]
        improved, stop = deepfm_stopper.update(validation_ndcg, epoch)
        checkpoint = _save_checkpoint(
            deepfm_checkpoint_root,
            epoch=epoch,
            model=deepfm,
            optimizer=deepfm_optimizer,
            metadata={
                "seed": seed,
                "cutoff": int(summary["cutoffs"]["train_cutoff_ms"]),
                "data_version": summary["data_version"],
                "feature_version": "unified-seven-source-visual-v4",
                "objective": run_config["deepfm_objective"],
                "config_hash": config_hash,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_ndcg@10": validation_ndcg,
                "best_metric": deepfm_stopper.best,
                "best_epoch": deepfm_stopper.best_epoch,
            },
        )
        if improved:
            best_deepfm_path = checkpoint
            _atomic_json(
                deepfm_checkpoint_root / "best.json",
                {"checkpoint": checkpoint.name, "epoch": epoch, "metric": validation_ndcg},
            )
        deepfm_history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_ndcg@10": validation_ndcg,
            }
        )
        if stop:
            break
    if best_deepfm_path is None:
        best_pointer = json.loads((deepfm_checkpoint_root / "best.json").read_text(encoding="utf-8"))
        best_deepfm_path = deepfm_checkpoint_root / str(best_pointer["checkpoint"])
    deepfm.load_state_dict(load_file(best_deepfm_path / "model.safetensors"))

    stable_score = _safe_baseline_scorer(
        base_user_ids=base_user_ids,
        base_item_ids=base_item_ids,
        user_factors=base_user_factors,
        item_factors=base_item_factors,
        content_user_ids=content_user_ids,
        content_item_ids=content_item_ids,
        content_user_vectors=content_user_vectors,
        content_item_vectors=content_item_vectors,
        popularity_scores=stable_popularity_scores,
    )
    validation_pairs_cats: list[np.ndarray] = []
    validation_pairs_cont: list[np.ndarray] = []
    validation_labels: list[float] = []
    for query in validation_sampled_queries:
        candidates = np.asarray([*query.positive_item_ids, *query.negative_item_ids])
        categorical, continuous, included = query_features(query.user_id, candidates)
        validation_pairs_cats.append(categorical[included])
        validation_pairs_cont.append(continuous[included])
        positives = set(query.positive_item_ids)
        validation_labels.extend(
            float(int(value) in positives) for value in candidates[included]
        )
    validation_cats = np.vstack(validation_pairs_cats)
    validation_cont = np.vstack(validation_pairs_cont)
    validation_labels_array = np.asarray(validation_labels, dtype=np.int64)
    deepfm_auc = float(
        roc_auc_score(
            validation_labels_array,
            _deepfm_scores(deepfm, validation_cats, validation_cont),
        )
    )
    linear = LogisticRegression(max_iter=200, random_state=seed)
    linear.fit(train_continuous, train_labels.astype(np.int64))
    linear_auc = float(
        roc_auc_score(validation_labels_array, linear.predict_proba(validation_cont)[:, 1])
    )
    base_metrics = dict(base_manifest.get("metrics") or {})
    base_validation = dict(base_metrics.get("validation") or {})
    base_test = dict(base_metrics.get("test") or {})
    base_validation_full = sampled_all_items_metrics(base_validation)
    base_test_full = sampled_all_items_metrics(base_test)
    artifact_stable_validation = dict(
        (base_validation_full.get("models") or {})["svd_content_fallback"]
    )
    artifact_stable_test = dict(
        (base_test_full.get("models") or {})["svd_content_fallback"]
    )
    artifact_stable_validation_warm = dict(
        (base_validation.get("models") or {})["svd_content_fallback"]
    )
    artifact_stable_test_warm = dict(
        (base_test.get("models") or {})["svd_content_fallback"]
    )
    reproduced_sampled_validation = evaluate_queries(
        validation_sampled_queries, stable_score
    )
    reproduced_sampled_warm = evaluate_queries(
        validation_sampled_warm_queries, stable_score
    )
    reproduced_sampled_test = evaluate_queries(test_sampled_queries, stable_score)
    reproduced_sampled_test_warm = evaluate_queries(
        test_sampled_warm_queries, stable_score
    )
    if cohort_matches_stable_artifact:
        for expected, actual in (
            (artifact_stable_validation, reproduced_sampled_validation),
            (artifact_stable_validation_warm, reproduced_sampled_warm),
            (artifact_stable_test, reproduced_sampled_test),
            (artifact_stable_test_warm, reproduced_sampled_test_warm),
        ):
            for metric_name in ("recall@10", "ndcg@10", "hitrate@10"):
                if not math.isclose(
                    float(expected[metric_name]),
                    float(actual[metric_name]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise DeepTrainingError(
                        f"stable scorer mismatch for {metric_name}: "
                        f"{actual[metric_name]} != {expected[metric_name]}"
                    )
    stable_validation_metrics = validation_eval(
        validation_queries_for_selection, stable_score
    )
    stable_validation_warm = validation_eval(
        validation_warm_queries_for_selection, stable_score
    )
    stable_test_metrics = (
        validation_eval(test_queries_for_report, stable_score) if run_test else None
    )
    stable_test_warm = (
        validation_eval(test_warm_queries_for_report, stable_score) if run_test else None
    )

    selected_strategy = "unified_multisource_v3"
    unified_validation = validation_eval(
        validation_queries_for_selection, two_stage_score
    )
    unified_validation_warm = validation_eval(
        validation_warm_queries_for_selection, two_stage_score
    )
    selected_score = two_stage_score
    validation_metrics = {
        "stable_safe": stable_validation_metrics,
        "dssm": validation_eval(
            validation_queries_for_selection,
            lambda user_id, candidates: encoded_items[
                np.asarray([catalog_lookup[int(value)] for value in candidates])
            ]
            @ encoded_users[user_lookup[int(user_id)]],
        ),
        "unified_multisource_deepfm": unified_validation,
        "selected_rank_strategy": selected_strategy,
        "stable_quality_prior_weight": 0.0,
        "source_order": list(UNIFIED_SOURCE_ORDER),
        "source_limits": source_limits,
        "source_caps_at_10": source_caps_at_10,
        "deepfm_auc": deepfm_auc,
        "linear_auc": linear_auc,
        "source_ablation": {
            "protocol": SAMPLED_NEGATIVE_PROTOCOL,
            "without_dssm": evaluate_queries(
                validation_sampled_queries,
                lambda user_id, candidates: two_stage_score(
                    user_id, candidates, frozenset({"dssm"})
                ),
            ),
            "without_visual": evaluate_queries(
                validation_sampled_queries,
                lambda user_id, candidates: two_stage_score(
                    user_id, candidates, frozenset({"visual"})
                ),
            ),
        },
        "warm": {
            "stable_safe": stable_validation_warm,
            "dssm": validation_eval(
                validation_warm_queries_for_selection,
                lambda user_id, candidates: encoded_items[
                    np.asarray([catalog_lookup[int(value)] for value in candidates])
                ]
                @ encoded_users[user_lookup[int(user_id)]],
            ),
            "unified_multisource_deepfm": unified_validation_warm,
        },
    }
    test_metrics = (
        {
            "stable_safe": stable_test_metrics,
            "dssm": validation_eval(
                test_queries_for_report,
                lambda user_id, candidates: encoded_items[
                    np.asarray([catalog_lookup[int(value)] for value in candidates])
                ]
                @ encoded_users[user_lookup[int(user_id)]],
            ),
            "unified_multisource_deepfm": validation_eval(
                test_queries_for_report, selected_score
            ),
            "warm": {
                "stable_safe": stable_test_warm,
                "dssm": validation_eval(
                    test_warm_queries_for_report,
                    lambda user_id, candidates: encoded_items[
                        np.asarray([catalog_lookup[int(value)] for value in candidates])
                    ]
                    @ encoded_users[user_lookup[int(user_id)]],
                ),
                "unified_multisource_deepfm": validation_eval(
                    test_warm_queries_for_report, selected_score
                ),
            },
        }
        if run_test
        else {"status": "not_run_validation_only"}
    )
    stable_validation = validation_metrics["stable_safe"]
    deep_validation = validation_metrics["unified_multisource_deepfm"]
    stable_validation_warm = validation_metrics["warm"]["stable_safe"]
    deep_validation_warm = validation_metrics["warm"]["unified_multisource_deepfm"]
    without_visual = validation_metrics["source_ablation"]["without_visual"]
    sampled_queries_are_unique = all(
        len((*query.positive_item_ids, *query.negative_item_ids))
        == len(set((*query.positive_item_ids, *query.negative_item_ids)))
        for query in validation_sampled_queries
    )
    quality_gate = {
        "unified_sampled_protocol": _sampled_protocol_gate(
            validation_mode,
            validation_sampled_cohort,
        ),
        "cohort_matches_stable_artifact": cohort_matches_stable_artifact,
        "replay_supervised_training": replay_training_summary["retrieved_positive_groups"] > 0,
        "dssm_contributes_candidates": replay_training_summary["source_memberships"]["dssm"] > 0,
        "visual_contributes_candidates": replay_training_summary["source_memberships"]["visual"] > 0,
        "test_run_once_after_validation_lock": run_test,
        "sampled_all_items_recall_not_below_stable": deep_validation["recall@10"]
        >= stable_validation["recall@10"],
        "sampled_all_items_ndcg_not_below_stable": deep_validation["ndcg@10"]
        >= stable_validation["ndcg@10"],
        "core_metric_improves_at_least_1pct": (
            deep_validation["recall@10"] >= 1.01 * stable_validation["recall@10"]
            or deep_validation["ndcg@10"] >= 1.01 * stable_validation["ndcg@10"]
        ),
        "warm_recall_within_1pct_of_stable": deep_validation_warm["recall@10"]
        >= 0.99 * stable_validation_warm["recall@10"],
        "warm_ndcg_within_1pct_of_stable": deep_validation_warm["ndcg@10"]
        >= 0.99 * stable_validation_warm["ndcg@10"],
        "deepfm_auc_beats_linear_baseline": math.isfinite(deepfm_auc)
        and math.isfinite(linear_auc)
        and deepfm_auc > linear_auc,
        "visual_ablation_has_incremental_value": (
            deep_validation["recall@10"] > without_visual["recall@10"]
            or deep_validation["ndcg@10"] > without_visual["ndcg@10"]
        ),
        "evaluation_candidates_have_zero_duplicates": sampled_queries_are_unique,
        "multimodal_features_loaded": bool(visual.available.any()),
        "cold_coverage_at_least_0_99": float(
            np.mean(np.linalg.norm(encoded_items, axis=1) > 0)
        )
        >= 0.99,
    }
    publishable = all(quality_gate.values())
    optimization_gate = {
        "unified_functional_release_gate": publishable,
        "quality_requires_stable_non_regression_and_core_improvement": True,
        "production_activation_requires_online_p95_and_official_e2e": True,
    }
    now = datetime.now(timezone.utc)
    model_version = f"deep-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{config_hash}"
    staging = artifacts_dir / f".staging-{model_version}-{uuid.uuid4().hex}"
    destination = artifacts_dir / model_version
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(best_dssm_path / "model.safetensors", staging / "dssm.safetensors")
        shutil.copy2(best_deepfm_path / "model.safetensors", staging / "deepfm.safetensors")
        arrays = {
            "deep_user_ids.npy": user_ids,
            "deep_item_ids.npy": catalog_ids,
            "deep_item_model_indices.npy": catalog_model_indices,
            "deep_user_embeddings.npy": encoded_users.astype(np.float32),
            "deep_item_embeddings.npy": encoded_items.astype(np.float32),
            "deep_user_profiles.npy": user_profiles,
            "deep_item_content.npy": item_dense,
            "deep_item_popularity.npy": item_popularity,
            "deep_item_pop_bucket.npy": item_pop_bucket,
            "deep_user_history_bucket.npy": user_history_bucket,
            "deep_user_history_density.npy": history_density,
        }
        for name, value in arrays.items():
            np.save(staging / name, value, allow_pickle=False)
        _atomic_json(staging / "dssm_config.json", asdict(dssm_config))
        _atomic_json(
            staging / "deepfm_config.json",
            {**asdict(deepfm_config), "categorical_sizes": list(deepfm_config.categorical_sizes), "hidden_dims": list(deepfm_config.hidden_dims)},
        )
        metrics = {
            "schema_version": 1,
            "evaluation_protocol": {
                "validation_only_selection": True,
                "validation_mode": validation_mode,
                "early_stopping": "deterministic_sampled_negatives_v1",
                "test_run_after_lock": run_test,
                "protocol": (
                    SAMPLED_NEGATIVE_PROTOCOL
                    if validation_mode == "sampled"
                    else "complete_eligible_catalog_v1"
                ),
                "candidate_universe": (
                    "complete eligible catalog"
                    if validation_mode == "full"
                    else "positives plus deterministic sampled negatives"
                ),
                "sampled_negative_usage": (
                    "standard validation, test, AUC and ablation protocol"
                    if validation_mode == "sampled"
                    else "AUC and ablation diagnostics only"
                ),
                "cohort_matches_stable_artifact": cohort_matches_stable_artifact,
                "validation_query_set_sha256": (
                    validation_cohort
                    if validation_mode == "full"
                    else validation_sampled_cohort
                )["query_set_sha256"],
                "test_query_set_sha256": (
                    test_cohort if validation_mode == "full" else test_sampled_cohort
                )["query_set_sha256"],
                "validation_warm_query_set_sha256": (
                    validation_warm_cohort if validation_mode == "full" else validation_sampled_warm_cohort
                )[
                    "query_set_sha256"
                ],
                "test_warm_query_set_sha256": (
                    test_warm_cohort if validation_mode == "full" else test_sampled_warm_cohort
                )["query_set_sha256"],
                "unscorable_positive_rule": "all positives retained; absent scores are misses",
            },
            "validation": validation_metrics,
            "test": test_metrics,
            "validation_cohort": (
                validation_cohort
                if validation_mode == "full"
                else validation_sampled_cohort
            ),
            "test_cohort": (
                test_cohort if validation_mode == "full" else test_sampled_cohort
            ),
            "validation_warm_cohort": (
                validation_warm_cohort
                if validation_mode == "full"
                else validation_sampled_warm_cohort
            ),
            "test_warm_cohort": (
                test_warm_cohort if validation_mode == "full" else test_sampled_warm_cohort
            ),
            "quality_gate": quality_gate,
            "optimization_gate": optimization_gate,
            "selected_rank_strategy": {
                "name": selected_strategy,
                "quality_prior_weight": 0.0,
                "deepfm_weight": 1.0,
                "selection_split": "validation",
            },
            "publishable": publishable,
            "candidate_coverage": float(np.mean(np.linalg.norm(encoded_items, axis=1) > 0)),
            "cold_candidate_coverage": float(
                np.mean(np.linalg.norm(encoded_items[cold.astype(bool)], axis=1) > 0)
            ),
        }
        training_summary = {
            "dssm": {
                "best_epoch": dssm_stopper.best_epoch,
                "best_validation_recall@50": dssm_stopper.best,
                "early_stopped": len(dssm_history) < epochs,
                "epochs": dssm_history,
            },
            "deepfm": {
                "best_epoch": deepfm_stopper.best_epoch,
                "best_validation_ndcg@10": deepfm_stopper.best,
                "early_stopped": len(deepfm_history) < epochs,
                "epochs": deepfm_history,
                "auc": deepfm_auc,
                "linear_baseline_auc": linear_auc,
                "training_samples": replay_training_summary,
            },
            "train_users": len(user_ids),
            "train_interactions": len(sample_users),
            "exposure_training": exposure_audit,
            "cold_start_dropout_rows": int(cold_dropout_mask.sum()),
            "catalog_items": len(catalog_ids),
            "cold_items": int(cold.sum()),
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(torch_device),
            "cuda_device_name": (
                torch.cuda.get_device_name(torch_device)
                if torch_device.type == "cuda"
                else None
            ),
        }
        _atomic_json(staging / "metrics.json", metrics)
        _atomic_json(staging / "training.json", training_summary)
        (staging / "evaluation.md").write_text(
            _deep_evaluation_markdown(metrics, training_summary), encoding="utf-8"
        )
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in staging.iterdir()
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "artifact_type": "dssm_deepfm_model",
            "model_version": model_version,
            "base_model_version": base_manifest["model_version"],
            "base_manifest": os.path.relpath(base_manifest_path, staging),
            "data_version": summary["data_version"],
            "created_at": now.isoformat(),
            "algorithm": "pytorch_dssm_recall_deepfm_rank",
            "feature_cutoff_ms": int(summary["cutoffs"]["train_cutoff_ms"]),
            "feature_version": "unified-seven-source-visual-v3",
            "feature_schema": {
                "categorical": [
                    "popularity_bucket",
                    "history_density_bucket",
                    "primary_source",
                ],
                "continuous": [
                    *[f"{name}_normalized_score" for name in UNIFIED_SOURCE_ORDER],
                    *[f"{name}_calibrated_score" for name in UNIFIED_SOURCE_ORDER],
                    *[f"{name}_present" for name in UNIFIED_SOURCE_ORDER],
                    "history_density",
                    "cold_item",
                    "visual_available",
                ],
            },
            "config": run_config,
            "files": files,
            "metrics": metrics,
            "training_summary": training_summary,
            "serving": {
                "experimental_only": not publishable,
                "mode": "unified_multisource_v3",
                "retrieval_top_n": max(200, retrieval_eval_top_n),
                "rank_strategy": "unified_deepfm",
                "stable_rank_weight": 0.0,
                "required_multimodal_version": visual.model_version,
                "source_limits": source_limits,
                "source_caps_at_10": source_caps_at_10,
                "fallback": "base_model",
                "online_image_inference": False,
            },
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        pointer_payload = {
            "schema_version": 1,
            "model_version": model_version,
            "manifest": f"{model_version}/manifest.json",
            "published_at": now.isoformat(),
            "status": "experiment" if not publishable else "eligible_experiment",
        }
        _atomic_json(artifacts_dir / "experiment-current.json", pointer_payload)
        published = bool(publish and publishable)
        if published:
            _atomic_json(
                artifacts_dir / "deep-current.json",
                {**pointer_payload, "status": "active"},
            )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "artifact_dir": str(destination),
        "model_version": model_version,
        "base_model_version": base_manifest["model_version"],
        "publishable": publishable,
        "published": bool(publish and publishable),
        "quality_gate": quality_gate,
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "training_summary": training_summary,
    }
