"""Pipeline 03: Baseline + smoothing + duration filter + gap merger."""
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
from experiments.postprocessing.gap_merger import gap_merge


def run(streams):
    rows = []
    for sid, cir in streams.items():
        residual = subtract_background(cir, 3.0)
        energy = smooth(frame_energy(residual), sigma_s=0.3)
        events = energy_events_raw(energy, pct=97.0, min_dur=0.3, merge_gap=0.5)
        events = dur_filter(events, min_dur=1.0)
        events = gap_merge(events, max_gap=1.5)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(3, "baseline + smoothing + duration_filter + gap_merger(1.5s)", rows, 0.1705)


if __name__ == "__main__":
    main()
