"""Inspect the dataset without loading whole streams: files, HDF5 schema, timestamps,
NaN/Inf, labels and composition. Writes JSON and a short Markdown summary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap
from abc_uwb_falltrack.config import load_config
from abc_uwb_falltrack.experiment import setup_logging, write_json
from abc_uwb_falltrack.inspection import inspect_dataset, summary_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _bootstrap.add_common_arguments(parser)
    parser.add_argument("--no-scan", action="store_true", help="skip the full NaN/Inf and value scan (much faster)")
    parser.add_argument("--out", default=None, help="output folder (default: outputs/inspection/<timestamp>)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override, args.smoke)
    setup_logging()
    out = Path(args.out or Path("outputs/inspection") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    report = inspect_dataset(cfg, scan=not args.no_scan)
    write_json(out / "inspection.json", report)
    (out / "inspection.md").write_text(summary_markdown(report), encoding="utf-8")
    print(json.dumps({"out": out.as_posix(), "streams": len(report["streams"]), "issues": report["issues"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
