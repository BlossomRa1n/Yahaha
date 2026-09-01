from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from recsys.data import DataValidationError, prepare_data


def _write_raw(raw_dir: Path, *, missing_title: bool = False) -> None:
    raw_dir.mkdir(parents=True)
    interactions: list[tuple[int, int, int]] = []
    for timestamp in range(1, 31):
        user_id = ((timestamp - 1) % 5) + 1
        interactions.append((user_id, timestamp, timestamp // 3 + 1))
    with (raw_dir / "MicroLens-50k_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user", "item", "timestamp"])
        writer.writerows(interactions)
    with (raw_dir / "MicroLens-50k_titles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", "title"])
        for item_id in range(1, 31 - int(missing_title)):
            writer.writerow([item_id, f'Title, "{item_id}"'])
    with (raw_dir / "MicroLens-50k_likes_and_views.txt").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for item_id in range(1, 31):
            handle.write(f"{item_id}\t{item_id}\t{item_id * 10}\n")


def _read_timestamps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [int(row["timestamp_ms"]) for row in csv.DictReader(handle)]


def test_prepare_is_deterministic_and_has_strict_time_boundaries(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir)
    first = tmp_path / "first"
    second = tmp_path / "second"

    summary = prepare_data(raw_dir, first, seed=7)
    repeated = prepare_data(raw_dir, second, seed=7)

    assert summary == repeated
    assert summary["counts"]["interactions"] == 30
    train = _read_timestamps(first / "train.csv")
    validation = _read_timestamps(first / "validation.csv")
    test = _read_timestamps(first / "test.csv")
    assert max(train) < min(validation)
    assert max(validation) < min(test)
    for name in (
        "train.csv",
        "validation.csv",
        "test.csv",
        "items.csv",
        "user_history.jsonl",
        "summary.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    histories = [json.loads(line) for line in (first / "user_history.jsonl").read_text().splitlines()]
    assert len(histories) == 5
    assert all(max(history["timestamps_ms"]) <= max(train) for history in histories)


def test_prepare_rejects_item_metadata_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, missing_title=True)
    with pytest.raises(DataValidationError, match="item metadata does not match"):
        prepare_data(raw_dir, tmp_path / "processed")

