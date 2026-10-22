"""Pipeline 14: p04 + Gaussian heatmap boundary refinement (k=1.2)."""
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
from experiments.postprocessing.boundary_refiner import gaussian_boundary


def run(streams):
    rows = []
    for sid, cir in streams.items():
        stream_dur_s = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        events = gaussian_boundary(events, energy, stream_dur_s, k=1.2)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(14, "p04_base + Gaussian_heatmap_boundary(k=1.2)", rows, 0.4676)


if __name__ == "__main__":
    main()
