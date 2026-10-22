"""Pipeline 19c: p19 với ngưỡng chọn từ OOF (không clip tay).

Baseline sạch cho GĐ4. Đây là điểm tham chiếu R cho tất cả pipeline 21-27.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EVENTS_CSV = ROOT / "streams_events.csv"

from abc2027.baseline import frame_energy
from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    energy_events_raw, RATE_HZ, STREAMS_TRAIN,
)
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter
from experiments.postprocessing.gap_merger import gap_merge
from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
from experiments.models.lightgbm_classifier import (
    load_classifier_rich_aug, lgb_frame_scores, window_features_rich,
)
from experiments.audit.oof_harness import (
    oof_threshold_sweep, run_oof, _train_fold, build_folds,
)


def run(streams, model, threshold=0.25):
    """Pipeline 19 logic — same as p19 but threshold is passed in."""
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-oof", action="store_true",
                        help="Skip OOF sweep, use val only with threshold=0.25")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Use this threshold instead of sweeping OOF")
    args = parser.parse_args()

    if args.threshold is not None:
        oof_thr = args.threshold
        oof_metrics = None
        print(f"Using provided threshold: {oof_thr:.3f}")
    elif args.skip_oof:
        oof_thr = 0.25
        oof_metrics = None
        print("Skipping OOF sweep, using threshold=0.25")
    else:
        # Step 1+2: OOF threshold sweep — caches scores so no double training
        print("=== Step 0.2: pipeline_19c OOF threshold sweep ===")
        thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
        oof_thr, thr_results, fold_cache = oof_threshold_sweep(
            pipeline_fn=run,
            thresholds=thresholds,
            label="p19c",
        )
        print(f"\nOOF best threshold: {oof_thr:.3f}")

        # Compute OOF metrics using cached fold data (no retraining)
        print("\n=== Computing OOF metrics from cached fold data ===")
        from experiments.postprocessing.duration_filter import dur_filter as _df
        from experiments.postprocessing.gap_merger import gap_merge as _gm
        import csv as _csv
        from abc2027.evaluate import evaluate as _ev, stream_durations, validate_predictions
        from abc2027.evaluate import load_ground_truth as _lgt

        gt_all2 = _lgt(EVENTS_CSV, ignore_flagged=True)
        gt_train2 = [e for e in gt_all2 if e["stream_id"].startswith("train_")]
        durs = stream_durations(STREAMS_TRAIN)
        high_thr = min(oof_thr * 2.5, 0.50)

        oof_rows = []
        for test_sid, cand_data in fold_cache:
            events = []
            for cand in cand_data:
                mx = cand["score"]
                if mx <= oof_thr * 0.6:
                    continue
                if (cand["p14_ok"] and mx > oof_thr) or mx > high_thr:
                    events.append({"t_start": cand["t_start"], "t_end": cand["t_end"],
                                   "score": mx, "t_alarm": cand["t_start"]})
            events = _df(events, min_dur=0.6)
            events = _gm(events, max_gap=1.0)
            for ev in events:
                oof_rows.append({
                    "stream_id": test_sid,
                    "t_start":  f"{ev['t_start']:.3f}",
                    "t_end":    f"{ev['t_end']:.3f}",
                    "score":    f"{ev['score']:.6f}",
                    "t_alarm":  f"{ev['t_alarm']:.3f}",
                })

        oof_csv = ROOT / "oof_pipeline_19c.csv"
        with open(oof_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
            w.writeheader()
            w.writerows(oof_rows)

        preds = validate_predictions(oof_csv, durs)
        result = _ev(gt_train2, preds, durs)
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
        print(f"\n=== OOF Metrics (p19c, thr={oof_thr:.3f}) ===")
        for k, v in oof_metrics.items():
            print(f"  {k}: {v}")

    # Step 3: Val evaluation (1 time only)
    print(f"\n=== Val evaluation (1 run, thr={oof_thr:.3f}) ===")
    print("Training full model on all 13 train streams...")
    model_full = load_classifier_rich_aug()
    val_streams = load_val()
    val_rows = run(val_streams, model_full, threshold=oof_thr)
    val_path = save_sub(val_rows, "pipeline_19c")
    val_metrics = run_eval(val_path)

    print(f"\n=== Val Metrics (p19c) ===")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    # Log to experiments_log.json
    delta = round(val_metrics["f1_mean"] - 0.6819, 4)
    entry = {
        "experiment_id": "pipeline_19c",
        "phase": 4,
        "base": "pipeline_19",
        "description": f"p19 + OOF threshold selection (thr={oof_thr:.3f}, no manual clip)",
        "components": ["oof_threshold_selection"],
        "oof_f1_mean":  oof_metrics["f1_mean"]  if oof_metrics else None,
        "oof_f1_01":    oof_metrics["f1_01"]    if oof_metrics else None,
        "oof_f1_03":    oof_metrics["f1_03"]    if oof_metrics else None,
        "oof_f1_05":    oof_metrics["f1_05"]    if oof_metrics else None,
        "oof_tp":       oof_metrics["tp"]       if oof_metrics else None,
        "oof_fp":       oof_metrics["fp"]       if oof_metrics else None,
        "oof_fn":       oof_metrics["fn"]       if oof_metrics else None,
        "delta_oof_vs_base": None,
        "p_delta_gt0":  None,
        "f1_mean": val_metrics["f1_mean"],
        "f1_01":   val_metrics["f1_01"],
        "f1_03":   val_metrics["f1_03"],
        "f1_05":   val_metrics["f1_05"],
        "tp":      val_metrics["tp"],
        "fp":      val_metrics["fp"],
        "fn":      val_metrics["fn"],
        "false_alarms_per_hour": val_metrics["false_alarms_per_hour"],
        "delta_f1": delta,
        "probe_auc": None,
        "probe_macro_f1": None,
        "gate": None,
        "decision": "BASE",
        "notes": f"OOF threshold={oof_thr:.3f} (no manual clip). "
                 f"R = OOF F1_mean={oof_metrics['f1_mean'] if oof_metrics else 'N/A'}",
    }
    append_log(entry)
    print(f"\n[pipeline_19c] Val F1_mean={val_metrics['f1_mean']:.4f} Δ={delta:+.4f}")
    print(f"  OOF F1_mean={oof_metrics['f1_mean'] if oof_metrics else 'N/A'} "
          f"(this is R for GĐ4)")


if __name__ == "__main__":
    main()
