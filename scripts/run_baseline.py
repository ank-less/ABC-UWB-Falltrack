"""Run the baseline pipeline, or one of its stages.

Stages: inspect, manifest, train (new run folder, tuned on train only),
evaluate (score the holdout split of an existing run folder), all (train + evaluate).
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap
from abc_uwb_falltrack.config import load_config
from abc_uwb_falltrack.experiment import evaluate_existing, run_all, run_manifest, setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _bootstrap.add_common_arguments(parser)
    parser.add_argument("--stage", choices=["inspect", "manifest", "train", "evaluate", "all"], default="all")
    parser.add_argument("--run-dir", default=None, help="existing run folder (required for --stage evaluate)")
    args = parser.parse_args(argv)

    if args.stage == "evaluate":
        if not args.run_dir:
            parser.error("--stage evaluate needs --run-dir (it reuses that run's resolved_config.yaml)")
        final = evaluate_existing(args.run_dir)
        print(json.dumps({m: v["holdout"]["event"]["ranking_score"] for m, v in final["models"].items()}, indent=2))
        return 0
    if args.stage == "inspect":
        import inspect_data

        forwarded = ["--config", args.config, *sum((["--override", o] for o in args.override), [])]
        return inspect_data.main(forwarded + (["--smoke"] if args.smoke else []))

    cfg = load_config(args.config, args.override, args.smoke)
    if args.stage == "manifest":
        setup_logging()
        manifest, _segments = run_manifest(cfg)
        print(f"manifest: {len(manifest)} streams")
        return 0
    run_dir = run_all(cfg, args.config, evaluate=args.stage == "all")
    print(f"run folder: {run_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
