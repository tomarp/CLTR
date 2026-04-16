from __future__ import annotations

import argparse
import os
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
(WORK_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(WORK_ROOT / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(WORK_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(WORK_ROOT / ".cache"))

from .config import default_config
from .pipeline import CLTRPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cltr_framework")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-root", type=Path, required=True)
    common.add_argument("--outdir", type=Path, required=True)
    common.add_argument("--session-limit", type=int, default=None)
    common.add_argument("--session-ids", nargs="*", default=None)
    common.add_argument("--participants", nargs="*", default=None)
    common.add_argument("--conditions", nargs="*", default=None)
    common.add_argument("--modalities", nargs="*", default=None)

    sub.add_parser("validate", parents=[common])
    sub.add_parser("preprocess", parents=[common])
    sub.add_parser("analyze", parents=[common])
    sub.add_parser("report", parents=[common])
    sub.add_parser("run-all", parents=[common])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = CLTRPipeline(args.dataset_root, args.outdir, default_config())
    if args.command == "validate":
        pipeline.validate()
    elif args.command == "preprocess":
        pipeline.preprocess(
            session_limit=args.session_limit,
            session_ids=args.session_ids,
            participants=args.participants,
            conditions=args.conditions,
        )
    elif args.command == "analyze":
        pipeline.analyze(
            session_limit=args.session_limit,
            session_ids=args.session_ids,
            participants=args.participants,
            conditions=args.conditions,
        )
    elif args.command == "report":
        pipeline.report(
            session_limit=args.session_limit,
            session_ids=args.session_ids,
            participants=args.participants,
            conditions=args.conditions,
            modalities=args.modalities,
        )
    elif args.command == "run-all":
        pipeline.run_all(
            session_limit=args.session_limit,
            session_ids=args.session_ids,
            participants=args.participants,
            conditions=args.conditions,
            modalities=args.modalities,
        )


if __name__ == "__main__":
    main()
