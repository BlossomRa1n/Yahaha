"""回归测试：数据准备与产物暂存必须使用 Path.mkdir 暂存目录，而非目录级 tempfile。

背景（方案 C）：目录级 ``tempfile.mkdtemp`` / ``tempfile.TemporaryDirectory`` 在
Windows 沙箱/受限 DACL 环境下会生成带受限 ACL 的临时目录，随后经 ``os.replace``
一并带入最终产物，导致产物目录无法被其它会话（或其它机器）读取/删除。

本测试把 ``tempfile.mkdtemp`` 与 ``tempfile.TemporaryDirectory`` 打桩为抛错，
断言 ``prepare_data`` 与 ``write_staged_artifact`` 在完全不依赖目录级 tempfile
的情况下仍能成功完成，防止未来回归到旧的实现。
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from recsys.artifacts import write_staged_artifact
from recsys.data import prepare_data


def _forbidden(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise AssertionError("directory-level tempfile must not be used")


def _write_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    interactions: list[tuple[int, int, int]] = []
    for timestamp in range(1, 31):
        user_id = ((timestamp - 1) % 5) + 1
        interactions.append((user_id, timestamp, timestamp // 3 + 1))
    with (raw_dir / "MicroLens-50k_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user", "item", "timestamp"])
        writer.writerows(interactions)
    with (raw_dir / "MicroLens-50k_titles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", "title"])
        for item_id in range(1, 31):
            writer.writerow([item_id, f'Title, "{item_id}"'])
    with (raw_dir / "MicroLens-50k_likes_and_views.txt").open("w", encoding="utf-8", newline="") as handle:
        for item_id in range(1, 31):
            handle.write(f"{item_id}\t{item_id}\t{item_id * 10}\n")


def test_prepare_data_avoids_directory_level_tempfile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tempfile, "mkdtemp", _forbidden)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", _forbidden)

    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir)
    summary = prepare_data(raw_dir, tmp_path / "processed", seed=7)

    assert summary["counts"]["interactions"] == 30
    assert (tmp_path / "processed" / "train.csv").is_file()


def test_write_staged_artifact_avoids_directory_level_tempfile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tempfile, "mkdtemp", _forbidden)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", _forbidden)

    staging = write_staged_artifact(
        tmp_path / "artifacts",
        manifest_base={
            "algorithm": "toy",
            "data_version": "toy-data",
            "model_version": "model-one",
            "schema_version": 1,
            "training_config": {"seed": 1},
            "training_summary": {},
        },
        user_ids=np.asarray([1, 2]),
        item_ids=np.asarray([10, 20, 30]),
        user_factors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        item_factors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        popularity={"items": [], "schema_version": 1},
        metrics={"schema_version": 1},
        evaluation_markdown="# Toy\n",
    )

    assert staging.is_dir()
    assert (staging / "manifest.json").is_file()

    second = write_staged_artifact(
        tmp_path / "artifacts",
        manifest_base={
            "algorithm": "toy",
            "data_version": "toy-data",
            "model_version": "model-one",
            "schema_version": 1,
            "training_config": {"seed": 1},
            "training_summary": {},
        },
        user_ids=np.asarray([1, 2]),
        item_ids=np.asarray([10, 20, 30]),
        user_factors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        item_factors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        popularity={"items": [], "schema_version": 1},
        metrics={"schema_version": 1},
        evaluation_markdown="# Toy\n",
    )
    assert second != staging
    assert second.is_dir()
    assert staging.is_dir()
