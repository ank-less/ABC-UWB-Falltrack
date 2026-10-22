"""Score a submission CSV (stream_id, t_start, t_end, score[, t_alarm]) on one split.

Uses the official evaluator of the dataset package: its format checks first, then
its event-level scoring. Without the evaluator the result is marked provisional.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import _bootstrap
from abc_uwb_falltrack.config import data_path, get, load_config
from abc_uwb_falltrack.data_io import load_events
from abc_uwb_falltrack.experiment import setup_logging, write_json
from abc_uwb_falltrack.metrics import event_report, ground_truth_records, load_official_evaluator, prediction_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _bootstrap.add_common_arguments(parser)
    parser.add_argument("--predictions", required=True, help="submission CSV to score")
    parser.add_argument("--split", default="val", help="streams to score against (default: %(default)s)")
    parser.add_argument("--out", default=None, help="report JSON (default: <predictions>.report.json)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override, args.smoke)
    setup_logging()
    predictions_path = Path(args.predictions)
    streams = data_path(cfg, "streams_dir") / args.split
    evaluator = load_official_evaluator(get(cfg, "evaluation.official_evaluator"))
    ignore_flagged = bool(get(cfg, "evaluation.ignore_flagged", True))
    events = load_events(data_path(cfg, "events_csv"))
    if evaluator is not None:
        durations = evaluator.stream_durations(streams)
        records = evaluator.validate_predictions(predictions_path, durations)  # raises on format errors
    else:
        import h5py

        rate = float(get(cfg, "data.rate_hz"))
        durations = {}
        for path in sorted(streams.glob("stream_*.h5")):
            with h5py.File(path, "r") as handle:
                durations[f"{args.split}_{int(path.stem.split('_')[1]):02d}"] = handle["cir"].shape[1] / rate
        frame = pd.read_csv(predictions_path)
        missing = {"stream_id", "t_start", "t_end", "score"} - set(frame.columns)
        if missing:
            raise SystemExit(f"{predictions_path}: missing columns {sorted(missing)}")
        records = prediction_records(frame)
    report = event_report(ground_truth_records(events, durations, ignore_flagged), records, durations, evaluator)
    report["ignore_flagged"] = ignore_flagged
    out = Path(args.out) if args.out else predictions_path.with_suffix(".report.json")
    if out.exists():
        raise SystemExit(f"{out} exists; pass another --out")
    write_json(out, report)
    print(json.dumps({"source": report["source"], "ranking_score": report["ranking_score"], "report": out.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
