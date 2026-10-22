"""Pipeline 00: Baseline gốc — EMA background + energy threshold."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import energy_events
from experiments.utils import load_val, make_rows, run_pipeline


def run(streams):
    rows = []
    for sid, cir in streams.items():
        events = energy_events(cir, percentile=98.0, min_duration=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(0, "Baseline gốc: EMA background + energy threshold", rows, None)


if __name__ == "__main__":
    main()
