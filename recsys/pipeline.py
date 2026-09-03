from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import prepare_data


DEFAULT_SEED = 20260901


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recsys.pipeline",
        description="Prepare, train, evaluate and publish the MicroLens-50K recommender.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="validate and time-split official data")
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument(
        "--stats-available-at",
        default=None,
        help="ISO-8601 time when the cumulative likes/views snapshot became available",
    )

    train = subparsers.add_parser("train", help="train, evaluate and atomically publish SVD")
    train.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    _add_train_arguments(train)

    all_command = subparsers.add_parser("all", help="run prepare followed by train")
    all_command.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    all_command.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    all_command.add_argument(
        "--stats-available-at",
        default=None,
        help="ISO-8601 time when the cumulative likes/views snapshot became available",
    )
    _add_train_arguments(all_command)

    deep = subparsers.add_parser("train-deep", help="train isolated PyTorch DSSM + DeepFM")
    deep.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    deep.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    deep.add_argument("--base-pointer", type=Path, default=Path("artifacts/current.json"))
    deep.add_argument(
        "--multimodal-pointer",
        type=Path,
        default=Path("artifacts/multimodal-current.json"),
    )
    deep.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    deep.add_argument("--max-users", type=int)
    deep.add_argument("--max-train-interactions", type=int)
    deep.add_argument("--max-eval-users", type=int)
    deep.add_argument("--epochs", type=int, default=8)
    deep.add_argument("--patience", type=int, default=2)
    deep.add_argument("--retrieval-top-n", type=int, default=50)
    deep.add_argument(
        "--validation-mode",
        choices=("full", "sampled"),
        default="sampled",
        help="use the unified sampled protocol or optional complete-catalog diagnostics",
    )
    deep.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch device; auto uses CUDA when available and otherwise CPU",
    )
    deep.add_argument("--validation-only", action="store_true")
    deep.add_argument("--seed", type=int, default=DEFAULT_SEED)
    deep.add_argument("--resume", action="store_true")
    deep.add_argument(
        "--exposure-database", type=Path, default=Path("data/app.db")
    )
    deep.add_argument(
        "--publish",
        action="store_true",
        help="atomically activate the model locally after functional and quality guards pass",
    )

    vision_audit = subparsers.add_parser(
        "vision-audit", help="safely extract and audit the official MicroLens cover archive"
    )
    vision_audit.add_argument(
        "--archive", type=Path, default=Path("data/raw/MicroLens-50k_covers.zip")
    )
    vision_audit.add_argument(
        "--covers-dir", type=Path, default=Path("data/raw/MicroLens-50k_covers")
    )
    vision_audit.add_argument(
        "--items", type=Path, default=Path("data/processed/items.csv")
    )

    multimodal = subparsers.add_parser(
        "train-multimodal",
        help="extract MobileNet cover features and train validation-locked text fusion",
    )
    multimodal.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    multimodal.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    multimodal.add_argument("--base-pointer", type=Path, default=Path("artifacts/current.json"))
    multimodal.add_argument(
        "--archive", type=Path, default=Path("data/raw/MicroLens-50k_covers.zip")
    )
    multimodal.add_argument(
        "--covers-dir", type=Path, default=Path("data/raw/MicroLens-50k_covers")
    )
    multimodal.add_argument("--batch-size", type=int, default=64)
    multimodal.add_argument("--pca-dim", type=int, default=128)
    multimodal.add_argument("--max-eval-users", type=int, default=5000)
    multimodal.add_argument("--validation-only", action="store_true")
    multimodal.add_argument("--locked-visual-weight", type=float)
    multimodal.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_data(
            args.raw_dir,
            args.out_dir,
            args.seed,
            stats_available_at=args.stats_available_at,
        )
    elif args.command == "train":
        from .model import train_model

        result = train_model(
            args.processed_dir,
            args.artifacts_dir,
            mode=args.mode,
            max_users=args.max_users,
            max_eval_users=args.max_eval_users,
            rank=args.rank,
            seed=args.seed,
        )
    elif args.command == "train-deep":
        from .deep import train_deep_experiment

        result = train_deep_experiment(
            args.processed_dir, args.artifacts_dir,
            base_pointer=args.base_pointer,
            multimodal_pointer=args.multimodal_pointer,
            mode=args.mode,
            max_users=args.max_users,
            max_train_interactions=args.max_train_interactions,
            max_eval_users=args.max_eval_users,
            epochs=args.epochs, patience=args.patience,
            retrieval_eval_top_n=args.retrieval_top_n,
            validation_mode=args.validation_mode,
            device=args.device,
            run_test=not args.validation_only,
            seed=args.seed,
            resume=args.resume,
            exposure_database=args.exposure_database,
            publish=args.publish,
        )
    elif args.command == "vision-audit":
        from .vision import audit_and_extract_covers

        result = audit_and_extract_covers(args.archive, args.covers_dir, args.items)
    elif args.command == "train-multimodal":
        from .vision import build_multimodal_experiment

        result = build_multimodal_experiment(
            args.processed_dir,
            args.artifacts_dir,
            covers_archive=args.archive,
            covers_dir=args.covers_dir,
            base_pointer=args.base_pointer,
            batch_size=args.batch_size,
            pca_dim=args.pca_dim,
            max_eval_users=args.max_eval_users,
            run_test=not args.validation_only,
            locked_visual_weight=args.locked_visual_weight,
            seed=args.seed,
        )
    else:
        from .model import train_model

        prepared = prepare_data(
            args.raw_dir,
            args.out_dir,
            args.seed,
            stats_available_at=args.stats_available_at,
        )
        trained = train_model(
            args.out_dir,
            args.artifacts_dir,
            mode=args.mode,
            max_users=args.max_users,
            max_eval_users=args.max_eval_users,
            rank=args.rank,
            seed=args.seed,
        )
        result = {"prepare": prepared, "train": trained}
    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
