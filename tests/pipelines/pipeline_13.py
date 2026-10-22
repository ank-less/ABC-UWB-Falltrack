"""Pipeline 13: p04 + window shrinker to 1.6s around energy peak."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import frame_energy
from experiments.utils import load_val, make_rows, run_pipeline, energy_events_raw, RATE_HZ
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.window_shrinker import shrink_to_peak


def run(streams):
    rows = []
    for sid, cir in streams.items():
        stream_dur_s = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        events = shrink_to_peak(events, energy, stream_dur_s, target_dur=1.6)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(13, "p04_base + window_shrinker(target=1.6s around energy peak)", rows, 0.2622)


if __name__ == "__main__":
    main()
