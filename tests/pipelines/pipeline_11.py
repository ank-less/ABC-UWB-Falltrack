"""Pipeline 11: Two-stage — energy(pct=95) → LightGBM confirm (thresh=0.2)."""
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
from experiments.models.lightgbm_classifier import load_classifier, lgb_frame_scores


def run(streams, model):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        candidates = energy_events_raw(energy, pct=95.0, min_dur=0.3, merge_gap=1.0)
        scores = lgb_frame_scores(filtered, model)
        T = filtered.shape[1]
        events = []
        for ev in candidates:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            max_lgb = float(np.max(scores[s:e]))
            if max_lgb > 0.2:
                ev["score"] = max_lgb
                events.append(ev)
        events = dur_filter(events, min_dur=0.5)
        events = gap_merge(events, max_gap=1.5)
        rows.extend(make_rows(sid, events))
    return rows


def main():
    streams = load_val()
    model = load_classifier()
    rows = run(streams, model)
    run_pipeline(11, "two_stage: energy(pct=95) → LightGBM_confirm(thresh=0.2) + gap_merge(1.5s)", rows, 0.0190)


if __name__ == "__main__":
    main()
