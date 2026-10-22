"""Generate OOF predictions for pipeline_19c at given threshold.

Usage: python experiments/pipelines/run_p19c_oof.py --threshold 0.475
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import frame_energy
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions,
)
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter
from experiments.postprocessing.gap_merger import gap_merge
from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
from experiments.models.lightgbm_classifier import lgb_frame_scores, window_features_rich
from experiments.audit.oof_harness import build_folds, _train_fold
from experiments.utils import energy_events_raw, STREAMS_TRAIN, RATE_HZ, append_log

EVENTS_CSV = ROOT / "streams_events.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.475)
    args = parser.parse_args()
    thr = args.threshold
    high_thr = min(thr * 2.5, 0.50)

    print(f"Running OOF for pipeline_19c at threshold={thr:.3f}")
    folds = build_folds()
    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)

    oof_rows = []
    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"Fold {k:02d}: training on {len(train_sids)} streams...", flush=True)
        model = _train_fold(train_sids, gt_all)

        test_path = STREAMS_TRAIN / f"stream_{int(test_sid.split('_')[1]):02d}.h5"
        with h5py.File(test_path) as f:
            cir = f["cir"][:]
        T = cir.shape[1]
        dur_s = T / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        cands = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores = lgb_frame_scores(filtered, model, feat_fn=window_features_rich)

        p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        events = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            if mx <= thr * 0.6:
                continue
            p14_ok = any(
                min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                for pev in p14_events
            )
            if (p14_ok and mx > thr) or mx > high_thr:
                ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)

        for ev in events:
            oof_rows.append({
                "stream_id": test_sid,
                "t_start":  f"{ev['t_start']:.3f}",
                "t_end":    f"{ev['t_end']:.3f}",
                "score":    f"{ev['score']:.6f}",
                "t_alarm":  f"{ev['t_alarm']:.3f}",
            })
        print(f"  → {len(events)} detections", flush=True)

    # Save OOF CSV
    oof_csv = ROOT / "oof_pipeline_19c.csv"
    with open(oof_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
        w.writeheader()
        w.writerows(oof_rows)
    print(f"\nOOF CSV saved: {oof_csv}")

    # Evaluate OOF
    durs = stream_durations(STREAMS_TRAIN)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    preds = validate_predictions(oof_csv, durs)
    result = evaluate(gt_train, preds, durs)
    rs = result["ranking_score"]
    bt = result["by_theta"]
    oof_metrics = {
        "f1_mean": round(rs["f1_mean"], 4),
        "f1_01":   round(bt["0.1"]["f1"], 4),
        "f1_03":   round(bt["0.3"]["f1"], 4),
        "f1_05":   round(bt["0.5"]["f1"], 4),
        "tp":      bt["0.3"]["tp"],
        "fp":      bt["0.3"]["fp"],
        "fn":      bt["0.3"]["fn"],
        "false_alarms_per_hour": round(rs["false_alarms_per_hour"], 1),
    }
    print(f"\n=== OOF Metrics (p19c, thr={thr:.3f}) ===")
    for key, val in oof_metrics.items():
        print(f"  {key}: {val}")

    # Save to log
    entry_oof = oof_metrics.copy()
    entry_oof["experiment_id"] = "pipeline_19c_oof_metrics"
    entry_oof["threshold"] = thr

    out = ROOT / "experiments" / "audit" / "reports" / "p19c_oof_metrics.json"
    out.write_text(json.dumps({"threshold": thr, "oof_metrics": oof_metrics}, indent=2))
    print(f"OOF metrics saved: {out}")
    return oof_metrics, thr


if __name__ == "__main__":
    main()
