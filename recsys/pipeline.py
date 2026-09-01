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

    train = subparsers.add_parser("train", help="train, evaluate and atomically publish SVD")
    train.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    _add_train_arguments(train)

    all_command = subparsers.add_parser("all", help="run prepare followed by train")
    all_command.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    all_command.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    _add_train_arguments(all_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_data(args.raw_dir, args.out_dir, args.seed)
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
    else:
        from .model import train_model

        prepared = prepare_data(args.raw_dir, args.out_dir, args.seed)
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
