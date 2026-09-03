from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import time
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .model import (
    SAMPLED_NEGATIVE_PROTOCOL,
    _rank_normalize,
    _read_processed_interactions,
    _queries_for_items,
    build_sampled_all_items_queries,
    evaluate_queries,
    sampled_all_items_metrics,
)


class VisionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _item_ids(items_path: Path) -> np.ndarray:
    values: list[int] = []
    with items_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(int(row["item_id"]))
    return np.asarray(values, dtype=np.int64)


def _safe_member(member: zipfile.ZipInfo) -> tuple[int, str]:
    path = PurePosixPath(member.filename)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "MicroLens-50k_covers"
        or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
    ):
        raise VisionError(f"unsafe or unexpected cover archive member: {member.filename}")
    try:
        item_id = int(path.stem)
    except ValueError as exc:
        raise VisionError(f"cover filename is not an item id: {member.filename}") from exc
    return item_id, path.name


def audit_and_extract_covers(
    archive_path: Path,
    destination: Path,
    items_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    archive_path = archive_path.resolve()
    destination = destination.resolve()
    expected_ids = set(map(int, _item_ids(items_path.resolve())))
    staging = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    if destination.exists():
        report_path = destination / "audit.json"
        if not report_path.is_file():
            raise VisionError("cover destination exists without audit.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("archive_sha256") != _sha256(archive_path):
            raise VisionError("existing covers were extracted from a different archive")
        return report
    staging.mkdir(parents=True, exist_ok=False)
    image_hashes: dict[str, str] = {}
    mapped_ids: set[int] = set()
    damaged: list[int] = []
    duplicates: dict[str, list[int]] = defaultdict(list)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                item_id, filename = _safe_member(member)
                if item_id in mapped_ids:
                    raise VisionError(f"duplicate item id in cover archive: {item_id}")
                mapped_ids.add(item_id)
                payload = archive.read(member)
                digest = hashlib.sha256(payload).hexdigest()
                image_hashes[str(item_id)] = digest
                duplicates[digest].append(item_id)
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError, ValueError):
                    damaged.append(item_id)
                    continue
                target = (staging / filename).resolve()
                try:
                    target.relative_to(staging)
                except ValueError as exc:
                    raise VisionError("cover extraction path escapes destination") from exc
                target.write_bytes(payload)
        missing = sorted(expected_ids - mapped_ids)
        unexpected = sorted(mapped_ids - expected_ids)
        duplicate_groups = [sorted(values) for values in duplicates.values() if len(values) > 1]
        report = {
            "schema_version": 1,
            "archive": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256(archive_path),
            "expected_items": len(expected_ids),
            "archive_images": len(mapped_ids),
            "mapped_images": len(mapped_ids & expected_ids),
            "mapping_success_rate": len(mapped_ids & expected_ids) / max(1, len(expected_ids)),
            "missing_items": missing,
            "missing_rate": len(missing) / max(1, len(expected_ids)),
            "unexpected_items": unexpected,
            "damaged_items": sorted(damaged),
            "damaged_rate": len(damaged) / max(1, len(mapped_ids)),
            "duplicate_image_groups": duplicate_groups,
            "duplicate_image_count": sum(len(values) - 1 for values in duplicate_groups),
            "image_sha256": image_hashes,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(staging / "audit.json", report)
        os.replace(staging, destination)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _read_pointer(pointer: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest_path = (pointer.parent / str(payload["manifest"])).resolve()
    try:
        manifest_path.relative_to(pointer.parent.resolve())
    except ValueError as exc:
        raise VisionError("model manifest escapes artifacts directory") from exc
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _weight_file(weights: MobileNet_V3_Small_Weights) -> Path:
    filename = Path(weights.url).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def extract_visual_embeddings(
    *,
    item_ids: np.ndarray,
    covers_dir: Path,
    train_item_ids: set[int],
    seed: int,
    batch_size: int,
    pca_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    weight_path = _weight_file(weights)
    if not weight_path.is_file():
        raise VisionError(f"pretrained weight is unavailable: {weight_path}")
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    transform = weights.transforms()
    raw_embeddings = np.zeros((len(item_ids), 576), dtype=np.float32)
    available = np.zeros(len(item_ids), dtype=bool)
    damaged: list[int] = []
    started = time.perf_counter()
    batch_tensors: list[torch.Tensor] = []
    batch_rows: list[int] = []

    def flush() -> None:
        if not batch_rows:
            return
        inputs = torch.stack(batch_tensors)
        with torch.inference_mode():
            features = model.features(inputs)
            pooled = model.avgpool(features).flatten(1)
        raw_embeddings[np.asarray(batch_rows)] = pooled.cpu().numpy().astype(np.float32)
        available[np.asarray(batch_rows)] = True
        batch_tensors.clear()
        batch_rows.clear()

    for row, item_id in enumerate(item_ids):
        path = covers_dir / f"{int(item_id)}.jpg"
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                batch_tensors.append(transform(image.convert("RGB")))
            batch_rows.append(row)
        except (UnidentifiedImageError, OSError, ValueError):
            damaged.append(int(item_id))
            continue
        if len(batch_rows) >= batch_size:
            flush()
    flush()
    train_rows = np.asarray(
        [row for row, value in enumerate(item_ids) if int(value) in train_item_ids and available[row]],
        dtype=np.int64,
    )
    if len(train_rows) < pca_dim:
        raise VisionError("not enough train-visible cover embeddings for PCA")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
    pca.fit(raw_embeddings[train_rows])
    reduced = np.zeros((len(item_ids), pca_dim), dtype=np.float32)
    reduced[available] = pca.transform(raw_embeddings[available]).astype(np.float32)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced /= np.maximum(norms, 1e-8)
    return reduced, available, {
        "encoder": "torchvision.mobilenet_v3_small",
        "weights": str(weights),
        "weights_url": weights.url,
        "weights_bytes": weight_path.stat().st_size,
        "weights_sha256": _sha256(weight_path),
        "raw_dimension": 576,
        "pca_dimension": pca_dim,
        "pca_fit_scope": "train-visible covers only",
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "available_items": int(available.sum()),
        "damaged_during_extraction": damaged,
        "batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "online_image_inference": False,
    }


def _profile_vectors(
    train_rows: Sequence[tuple[int, int, int]],
    *,
    item_position: dict[int, int],
    embeddings: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    user_ids = np.asarray(sorted({row[0] for row in train_rows}), dtype=np.int64)
    user_position = {int(value): index for index, value in enumerate(user_ids)}
    profiles = np.zeros((len(user_ids), embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(len(user_ids), dtype=np.int64)
    for user_id, item_id, _ in train_rows:
        item_row = item_position.get(item_id)
        if item_row is None or not available[item_row]:
            continue
        profiles[user_position[user_id]] += embeddings[item_row]
        counts[user_position[user_id]] += 1
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-8)
    return user_ids, profiles


def _visual_scorer(
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    user_profiles: np.ndarray,
    item_embeddings: np.ndarray,
    available: np.ndarray,
) -> Callable[[int, np.ndarray], np.ndarray]:
    user_position = {int(value): index for index, value in enumerate(user_ids)}
    item_position = {int(value): index for index, value in enumerate(item_ids)}

    def score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        user_row = user_position.get(int(user_id))
        if user_row is None:
            return np.zeros(len(candidates), dtype=np.float64)
        rows = np.asarray([item_position[int(value)] for value in candidates])
        values = item_embeddings[rows] @ user_profiles[user_row]
        return np.where(available[rows], values, -1.0).astype(np.float64)

    return score


def _text_scorer(base_dir: Path) -> Callable[[int, np.ndarray], np.ndarray]:
    from scipy.sparse import load_npz

    user_ids = np.load(base_dir / "content_user_ids.npy", allow_pickle=False)
    item_ids = np.load(base_dir / "content_item_ids.npy", allow_pickle=False)
    users = load_npz(base_dir / "content_user_vectors.npz").tocsr()
    items = load_npz(base_dir / "content_item_vectors.npz").tocsr()
    user_position = {int(value): index for index, value in enumerate(user_ids)}
    item_position = {int(value): index for index, value in enumerate(item_ids)}

    def score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        user_row = user_position.get(int(user_id))
        if user_row is None:
            return np.zeros(len(candidates), dtype=np.float64)
        rows = np.asarray([item_position[int(value)] for value in candidates])
        return np.asarray((items[rows] @ users[user_row].T).toarray().ravel())

    return score


def _fusion_scorer(
    text_score: Callable[[int, np.ndarray], np.ndarray],
    visual_score: Callable[[int, np.ndarray], np.ndarray],
    *,
    train_item_ids: set[int],
    warm_visual_weight: float,
    cold_visual_weight: float,
) -> Callable[[int, np.ndarray], np.ndarray]:
    def score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        text = _rank_normalize(text_score(user_id, candidates), candidates)
        visual = _rank_normalize(visual_score(user_id, candidates), candidates)
        weights = np.asarray(
            [
                warm_visual_weight if int(item_id) in train_item_ids else cold_visual_weight
                for item_id in candidates
            ],
            dtype=np.float64,
        )
        return (1.0 - weights) * text + weights * visual

    return score


def build_multimodal_experiment(
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    covers_archive: Path,
    covers_dir: Path,
    base_pointer: Path | None = None,
    batch_size: int = 64,
    pca_dim: int = 128,
    max_eval_users: int = 5000,
    run_test: bool = True,
    locked_visual_weight: float | None = None,
    seed: int = 20260901,
) -> dict[str, Any]:
    started = time.perf_counter()
    processed_dir = processed_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    base_pointer = (base_pointer or artifacts_dir / "current.json").resolve()
    base_manifest_path, base_manifest = _read_pointer(base_pointer)
    base_dir = base_manifest_path.parent
    summary = json.loads((processed_dir / "summary.json").read_text(encoding="utf-8"))
    audit = audit_and_extract_covers(
        covers_archive.resolve(), covers_dir.resolve(), processed_dir / "items.csv"
    )
    item_ids = _item_ids(processed_dir / "items.csv")
    train_rows = _read_processed_interactions(processed_dir / "train.csv")
    validation_rows = _read_processed_interactions(processed_dir / "validation.csv")
    test_rows = _read_processed_interactions(processed_dir / "test.csv")
    train_item_ids = {row[1] for row in train_rows}
    item_embeddings, available, extraction = extract_visual_embeddings(
        item_ids=item_ids,
        covers_dir=covers_dir.resolve(),
        train_item_ids=train_item_ids,
        seed=seed,
        batch_size=batch_size,
        pca_dim=pca_dim,
    )
    item_position = {int(value): index for index, value in enumerate(item_ids)}
    visual_user_ids, visual_profiles = _profile_vectors(
        train_rows,
        item_position=item_position,
        embeddings=item_embeddings,
        available=available,
    )
    validation_queries, validation_cohort = build_sampled_all_items_queries(
        split_name="validation",
        target_rows=validation_rows,
        known_rows=train_rows,
        model_user_ids=visual_user_ids,
        catalog_item_ids=item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    test_queries, test_cohort = build_sampled_all_items_queries(
        split_name="test",
        target_rows=test_rows,
        known_rows=[*train_rows, *validation_rows],
        model_user_ids=visual_user_ids,
        catalog_item_ids=item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    base_validation_hash = sampled_all_items_metrics(
        (base_manifest.get("metrics") or {}).get("validation") or {}
    ).get("cohort", {}).get("query_set_sha256")
    base_test_hash = sampled_all_items_metrics(
        (base_manifest.get("metrics") or {}).get("test") or {}
    ).get("cohort", {}).get("query_set_sha256")
    if (
        validation_cohort["query_set_sha256"] != base_validation_hash
        or test_cohort["query_set_sha256"] != base_test_hash
    ):
        raise VisionError("multimodal and stable evaluation query sets do not match")
    visual_score = _visual_scorer(
        user_ids=visual_user_ids,
        item_ids=item_ids,
        user_profiles=visual_profiles,
        item_embeddings=item_embeddings,
        available=available,
    )
    text_score = _text_scorer(base_dir)
    score_cache: dict[tuple[int, bytes], tuple[np.ndarray, np.ndarray]] = {}

    def cached_scores(
        user_id: int, candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (int(user_id), np.asarray(candidates, dtype=np.int64).tobytes())
        cached = score_cache.get(key)
        if cached is None:
            cached = (
                text_score(user_id, candidates),
                visual_score(user_id, candidates),
            )
            score_cache[key] = cached
        return cached

    def cached_text_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        return cached_scores(user_id, candidates)[0]

    def cached_visual_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        return cached_scores(user_id, candidates)[1]

    warm_queries = _queries_for_items(validation_queries, train_item_ids)
    cold_item_ids = set(map(int, item_ids)) - train_item_ids
    cold_queries = _queries_for_items(validation_queries, cold_item_ids)
    if locked_visual_weight is not None and not 0.0 <= locked_visual_weight <= 1.0:
        raise VisionError("locked visual weight must be between 0 and 1")
    candidate_policies = (
        {
            f"locked_static_{locked_visual_weight:g}": (
                locked_visual_weight,
                locked_visual_weight,
            )
        }
        if locked_visual_weight is not None
        else {
            "static_0.16": (0.16, 0.16),
            "static_0.17": (0.17, 0.17),
            "static_0.175": (0.175, 0.175),
            "static_0.18": (0.18, 0.18),
            "static_0.19": (0.19, 0.19),
        }
    )
    baseline_policies = {"text_only": (0.0, 0.0), "visual_only": (1.0, 1.0)}
    validation_candidates: dict[str, dict[str, Any]] = {}
    for policy_name, (warm_weight, cold_weight) in {
        **baseline_policies,
        **candidate_policies,
    }.items():
        scorer = _fusion_scorer(
            cached_text_score,
            cached_visual_score,
            train_item_ids=train_item_ids,
            warm_visual_weight=warm_weight,
            cold_visual_weight=cold_weight,
        )
        validation_candidates[policy_name] = {
            "warm_visual_weight": warm_weight,
            "cold_visual_weight": cold_weight,
            "sampled_all_items": evaluate_queries(validation_queries, scorer),
            "warm": evaluate_queries(warm_queries, scorer),
            "cold": evaluate_queries(cold_queries, scorer),
        }
    text_metrics = validation_candidates["text_only"]
    eligible_policies = []
    for policy_name in candidate_policies:
        candidate = validation_candidates[policy_name]
        sampled_all_items = candidate["sampled_all_items"]
        warm = candidate["warm"]
        cold = candidate["cold"]
        candidate["quality_gate"] = {
            "sampled_all_items_recall_not_down_0_5pct": sampled_all_items["recall@10"]
            >= 0.995 * text_metrics["sampled_all_items"]["recall@10"],
            "sampled_all_items_ndcg_not_down_0_5pct": sampled_all_items["ndcg@10"]
            >= 0.995 * text_metrics["sampled_all_items"]["ndcg@10"],
            "warm_recall_not_down_0_5pct": warm["recall@10"]
            >= 0.995 * text_metrics["warm"]["recall@10"],
            "warm_ndcg_not_down_0_5pct": warm["ndcg@10"]
            >= 0.995 * text_metrics["warm"]["ndcg@10"],
            "sampled_all_items_recall_improves_1pct": sampled_all_items["recall@10"]
            >= 1.01 * text_metrics["sampled_all_items"]["recall@10"],
            "sampled_all_items_ndcg_improves_1pct": sampled_all_items["ndcg@10"]
            >= 1.01 * text_metrics["sampled_all_items"]["ndcg@10"],
            "cold_recall_improves_3pct": cold["recall@10"]
            >= 1.03 * text_metrics["cold"]["recall@10"],
            "cold_ndcg_improves_3pct": cold["ndcg@10"]
            >= 1.03 * text_metrics["cold"]["ndcg@10"],
        }
        safety = all(
            candidate["quality_gate"][name]
            for name in (
                "sampled_all_items_recall_not_down_0_5pct",
                "sampled_all_items_ndcg_not_down_0_5pct",
                "warm_recall_not_down_0_5pct",
                "warm_ndcg_not_down_0_5pct",
            )
        )
        improvement = any(
            candidate["quality_gate"][name]
            for name in (
                "sampled_all_items_recall_improves_1pct",
                "sampled_all_items_ndcg_improves_1pct",
                "cold_recall_improves_3pct",
                "cold_ndcg_improves_3pct",
            )
        )
        if safety and improvement:
            eligible_policies.append(policy_name)
    selectable_policies = eligible_policies or list(candidate_policies)
    selected_policy = max(
        selectable_policies,
        key=lambda policy_name: (
            validation_candidates[policy_name]["sampled_all_items"]["ndcg@10"]
            + validation_candidates[policy_name]["sampled_all_items"]["recall@10"]
            + 0.25
            * (
                validation_candidates[policy_name]["cold"]["ndcg@10"]
                + validation_candidates[policy_name]["cold"]["recall@10"]
            ),
            -validation_candidates[policy_name]["cold_visual_weight"],
        ),
    )
    selected_warm_weight = float(
        validation_candidates[selected_policy]["warm_visual_weight"]
    )
    selected_cold_weight = float(
        validation_candidates[selected_policy]["cold_visual_weight"]
    )
    locked_score = _fusion_scorer(
        cached_text_score,
        cached_visual_score,
        train_item_ids=train_item_ids,
        warm_visual_weight=selected_warm_weight,
        cold_visual_weight=selected_cold_weight,
    )
    test_metrics = (
        evaluate_queries(test_queries, locked_score)
        if run_test
        else {"status": "not_run_validation_only"}
    )
    cold_rows = np.asarray(
        [index for index, value in enumerate(item_ids) if int(value) not in train_item_ids],
        dtype=np.int64,
    )
    metrics = {
        "schema_version": 1,
        "evaluation_protocol": {
            "protocol": SAMPLED_NEGATIVE_PROTOCOL,
            "validation_only_selection": True,
            "test_run_after_weight_lock": run_test,
            "validation_query_set_sha256": validation_cohort["query_set_sha256"],
            "test_query_set_sha256": test_cohort["query_set_sha256"],
            "score_calibration": "per-query deterministic rank normalization",
        },
        "validation": {
            "candidate_policies": validation_candidates,
            "text_only": validation_candidates["text_only"]["sampled_all_items"],
            "visual_only": validation_candidates["visual_only"]["sampled_all_items"],
            "fusion": validation_candidates[selected_policy]["sampled_all_items"],
            "warm": validation_candidates[selected_policy]["warm"],
            "cold": validation_candidates[selected_policy]["cold"],
            "selected_policy": selected_policy,
            "selected_warm_visual_weight": selected_warm_weight,
            "selected_cold_visual_weight": selected_cold_weight,
        },
        "test": {
            "fusion": test_metrics,
            "selected_policy": selected_policy,
            "selected_warm_visual_weight": selected_warm_weight,
            "selected_cold_visual_weight": selected_cold_weight,
        },
        "validation_cohort": validation_cohort,
        "test_cohort": test_cohort,
        "item_visual_coverage": float(available.mean()),
        "cold_item_visual_coverage": float(available[cold_rows].mean()),
    }
    text_validation = metrics["validation"]["text_only"]
    fusion_validation = metrics["validation"]["fusion"]
    quality_gate = {
        "coverage_at_least_0_99": metrics["item_visual_coverage"] >= 0.99,
        "fusion_recall_improves_1pct": fusion_validation["recall@10"]
        >= 1.01 * text_validation["recall@10"],
        "fusion_ndcg_improves_1pct": fusion_validation["ndcg@10"]
        >= 1.01 * text_validation["ndcg@10"],
    }
    quality_gate["quality_improvement"] = (
        quality_gate["fusion_recall_improves_1pct"]
        or quality_gate["fusion_ndcg_improves_1pct"]
    )
    now = datetime.now(timezone.utc)
    config = {
        "encoder": extraction["encoder"],
        "weights_sha256": extraction["weights_sha256"],
        "batch_size": batch_size,
        "pca_dim": pca_dim,
        "seed": seed,
        "feature_cutoff_ms": int(summary["cutoffs"]["train_cutoff_ms"]),
        "candidate_visual_policies": {
            name: {"warm": weights[0], "cold": weights[1]}
            for name, weights in candidate_policies.items()
        },
        "locked_visual_weight": locked_visual_weight,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    version = f"multimodal-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{config_hash}"
    staging = artifacts_dir / f".staging-{version}-{uuid.uuid4().hex}"
    destination = artifacts_dir / version
    staging.mkdir(parents=True, exist_ok=False)
    try:
        arrays = {
            "visual_item_ids.npy": item_ids,
            "visual_item_embeddings.npy": item_embeddings,
            "visual_available.npy": available,
            "visual_user_ids.npy": visual_user_ids,
            "visual_user_profiles.npy": visual_profiles,
        }
        for name, value in arrays.items():
            np.save(staging / name, value, allow_pickle=False)
        _atomic_json(staging / "metrics.json", metrics)
        _atomic_json(
            staging / "extraction.json",
            {
                **extraction,
                "archive_sha256": audit["archive_sha256"],
                "mapping_success_rate": audit["mapping_success_rate"],
                "damaged_rate": audit["damaged_rate"],
                "duplicate_image_count": audit["duplicate_image_count"],
            },
        )
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in staging.iterdir()
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "artifact_type": "mobilenet_text_fusion_experiment",
            "model_version": version,
            "base_model_version": base_manifest["model_version"],
            "data_version": summary["data_version"],
            "created_at": now.isoformat(),
            "feature_cutoff_ms": int(summary["cutoffs"]["train_cutoff_ms"]),
            "feature_version": "mobilenetv3-small-pca128-text-late-fusion-v1",
            "config": config,
            "files": files,
            "metrics": metrics,
            "quality_gate": quality_gate,
            "publishable": bool(
                quality_gate["coverage_at_least_0_99"]
                and quality_gate["quality_improvement"]
            ),
            "serving": {
                "selected_policy": selected_policy,
                "selected_visual_weight": selected_cold_weight,
                "selected_warm_visual_weight": selected_warm_weight,
                "selected_cold_visual_weight": selected_cold_weight,
                "fallback": "text_only",
                "online_image_inference": False,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        _atomic_json(
            artifacts_dir / "multimodal-current.json",
            {
                "schema_version": 1,
                "model_version": version,
                "manifest": f"{version}/manifest.json",
                "status": "eligible_experiment" if manifest["publishable"] else "experiment",
                "published_at": now.isoformat(),
            },
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "artifact_dir": str(destination),
        "model_version": version,
        "publishable": manifest["publishable"],
        "quality_gate": quality_gate,
        "metrics": metrics,
        "audit": {
            key: value for key, value in audit.items() if key != "image_sha256"
        },
        "extraction": extraction,
        "elapsed_seconds": manifest["elapsed_seconds"],
    }
