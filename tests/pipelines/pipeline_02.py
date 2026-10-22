"""Pipeline 02: Baseline + smoothing + duration filter (≥1s)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.background import subtract_background
from abc2027.baseline import frame_energy
from experiments.utils import load_val, make_rows, run_pipeline, energy_events_raw
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter


def run(streams):
    rows = []
    for sid, cir in streams.items():
        residual = subtract_background(cir, 3.0)
        energy = smooth(frame_energy(residual), sigma_s=0.3)
        events = energy_events_raw(energy, pct=97.0, min_dur=0.3, merge_gap=0.8)
        events = dur_filter(events, min_dur=1.0)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(2, "baseline + smoothing + duration_filter(>=1.0s)", rows, 0.1117)


if __name__ == "__main__":
    main()
