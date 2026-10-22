"""Pipeline 19: Rich features + CV threshold + p14 dual gate + boundary refiner."""
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
from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
from experiments.models.lightgbm_classifier import (
    load_classifier_rich_aug, load_classifier_base,
    lgb_frame_scores, window_features_rich,
    find_best_threshold_on_train,
)


def run(streams, model, threshold=0.25):
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores   = lgb_frame_scores(filtered, model, feat_fn=window_features_rich)

        p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        T = filtered.shape[1]
        high_thr = min(threshold * 2.5, 0.50)

        events = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            if mx <= threshold * 0.6:
                continue
            p14_ok = any(
                min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                for pev in p14_events
            )
            if (p14_ok and mx > threshold) or mx > high_thr:
                ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                events.append({"t_start": ns, "t_end": ne,
                               "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    model_base = load_classifier_base()
    opt_thr    = find_best_threshold_on_train(model_base)
    p19_thr    = float(np.clip(opt_thr, 0.10, 0.25))
    model      = load_classifier_rich_aug()
    rows = run(streams, model, threshold=p19_thr)
    run_pipeline(19, f"rich+aug(27f) + CV_thr={p19_thr:.2f} + p14_dual_gate + boundary_refiner", rows, 0.6422)


if __name__ == "__main__":
    main()
