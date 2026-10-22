"""Pipeline 05: Chebyshev + IQR normalization + energy threshold + smoothing."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import frame_energy
from experiments.utils import load_val, make_rows, run_pipeline, energy_events_raw
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.preprocessing.iqr_normalization import iqr_norm
from experiments.postprocessing.temporal_smoothing import smooth


def run(streams):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        energy = iqr_norm(frame_energy(filtered).astype(np.float64))
        energy = smooth(energy, sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    rows = run(streams)
    run_pipeline(5, "Chebyshev + IQR normalization + energy threshold + smoothing", rows, 0.4676)


if __name__ == "__main__":
    main()
