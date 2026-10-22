"""Pipeline 06: Chebyshev + IQR + LightGBM/RF classifier."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import iter_windows, window_features, runs_to_events
from experiments.utils import load_val, make_rows, run_pipeline, RATE_HZ
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.models.lightgbm_classifier import load_classifier


def _windows_to_events(cir, model, threshold=0.35, min_dur=0.6, merge_gap=0.8):
    wins = iter_windows(cir.shape[1])
    if not wins:
        return []
    feats = np.array([window_features(cir[:, s:e, :]) for s, e in wins])
    scores = model.predict_proba(feats)[:, 1]
    scores_s = gaussian_filter1d(scores, sigma=2.0)
    xs = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
    full = np.interp(np.arange(cir.shape[1], dtype=np.float64), xs, scores_s)
    full = gaussian_filter1d(full, sigma=5.0)
    level = float(np.percentile(full, 99.0))
    level = max(level, threshold)
    if level <= 0:
        return []
    return runs_to_events(full >= level, full / (level + 1e-9), min_dur, merge_gap)


def run(streams, model):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        events = _windows_to_events(filtered, model, threshold=0.35, min_dur=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    model = load_classifier()
    rows = run(streams, model)
    run_pipeline(6, "Chebyshev + IQR + LightGBM/RF classifier + smoothing", rows, 0.4676)


if __name__ == "__main__":
    main()
