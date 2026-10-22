"""Pipeline 16: p09 with train-CV-optimised LGB threshold."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import frame_energy
from experiments.utils import load_val, make_rows, run_pipeline, energy_events_raw, RATE_HZ
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter
from experiments.postprocessing.gap_merger import gap_merge
from experiments.postprocessing.boundary_refiner import refine_boundary
from experiments.models.lightgbm_classifier import (
    load_classifier_base, lgb_frame_scores, find_best_threshold_on_train,
)


def run(streams, model, threshold=0.10):
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores   = lgb_frame_scores(filtered, model)
        T        = filtered.shape[1]
        events   = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            if mx > threshold:
                ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                events.append({"t_start": ns, "t_end": ne,
                               "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    model = load_classifier_base()
    opt_thr = find_best_threshold_on_train(model)
    rows = run(streams, model, threshold=opt_thr)
    run_pipeline(16, f"p09_base + train_opt_threshold={opt_thr:.2f} + boundary_refiner", rows, 0.5758)


if __name__ == "__main__":
    main()
