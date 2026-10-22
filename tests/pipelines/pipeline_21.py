"""Pipeline 21: p19c + interval calibrator (nested OOF boundary shift/scale).

GĐ4 — detection-neutral: keeps stream_id, score, number of predictions.
Only shifts t_start / t_end.

Decision criterion:
  KEEP if OOF F1_mean >= R + 0.005 AND OOF F1_05 >= R_05 + 0.02
  DROP  if OOF F1_mean < R
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EVENTS_CSV    = ROOT / "streams_events.csv"
OOF_P19C_CSV  = ROOT / "oof_pipeline_19c.csv"
REPORTS_DIR   = ROOT / "experiments" / "audit" / "reports"
CAL_PARAMS_F  = ROOT / "experiments" / "audit" / "reports" / "calibration_params.json"

from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    energy_events_raw, RATE_HZ, STREAMS_TRAIN, STREAMS_VAL,
)
from experiments.postprocessing.interval_calibrator import (
    calibrate_nested_oof, apply_calibration,
)
from experiments.audit.oof_harness import build_folds
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions,
)


def _assert_detection_neutral(orig_csv: Path, cal_rows: list):
    """Assert stream_ids and prediction count are unchanged."""
    orig_by = {}
    with open(orig_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            orig_by.setdefault(row["stream_id"], []).append(row)
    cal_by = {}
    for row in cal_rows:
        cal_by.setdefault(row["stream_id"], []).append(row)
    assert set(orig_by) == set(cal_by), "stream_id mismatch after calibration"
    for sid in orig_by:
        assert len(orig_by[sid]) == len(cal_by[sid]), \
            f"prediction count changed for {sid}: {len(orig_by[sid])} → {len(cal_by[sid])}"
    print("  ✓ detection-neutral assertion passed")


def _load_log_entry(experiment_id: str) -> dict:
    log_file = ROOT / "experiments_log.json"
    if not log_file.exists():
        return {}
    log = json.loads(log_file.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == experiment_id), {})


def main():
    print("=" * 70)
    print("Pipeline 21 — Interval Calibration (nested OOF, detection-neutral)")
    print("=" * 70)

    # ── Sanity: check OOF CSV exists ─────────────────────────────────────────
    if not OOF_P19C_CSV.exists():
        print(f"ERROR: OOF CSV not found at {OOF_P19C_CSV}")
        print("  Run pipeline_19c first.")
        return

    # ── Load R from log ───────────────────────────────────────────────────────
    p19c = _load_log_entry("pipeline_19c")
    R     = p19c.get("oof_f1_mean")
    R_05  = p19c.get("oof_f1_05")
    if R is None:
        print("WARNING: pipeline_19c OOF F1_mean not in log — using val as proxy (biased).")
        R    = p19c.get("f1_mean", 0.0)
        R_05 = p19c.get("f1_05",  0.0)
    print(f"\nBaseline R = {R:.4f}  R_05 = {R_05:.4f}")

    # ── Load ground truth + durations ─────────────────────────────────────────
    gt_all    = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by     = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)
    gt_train  = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs      = stream_durations(STREAMS_TRAIN)

    # ── Run nested OOF calibration ────────────────────────────────────────────
    folds  = build_folds()
    print(f"\nRunning nested OOF calibration ({len(folds)} folds)…")
    form, global_params, cal_rows = calibrate_nested_oof(
        oof_csv=OOF_P19C_CSV,
        folds=folds,
        gt_by=gt_by,
        durs=durs,
        verbose=True,
    )

    # Stability abort
    if global_params.get("unstable"):
        print("\n⚠️  Calibration params are unstable (max std > 0.15 s) → DROP")
        entry = _make_log_entry(R, R_05, None, None, None, None,
                                global_params, "DROP",
                                "Unstable calibration params across folds")
        append_log(entry)
        return

    # ── Detection-neutral assertion ───────────────────────────────────────────
    _assert_detection_neutral(OOF_P19C_CSV, cal_rows)

    # ── Evaluate calibrated OOF ───────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_cal_csv = ROOT / "oof_pipeline_21.csv"
    with open(oof_cal_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        w.writeheader()
        w.writerows(cal_rows)

    preds   = validate_predictions(oof_cal_csv, durs)
    result  = evaluate(gt_train, preds, durs)
    rs      = result["ranking_score"]
    bt      = result["by_theta"]
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

    print(f"\n=== OOF Results (p21) ===")
    print(f"  F1_mean = {oof_metrics['f1_mean']:.4f}  (R={R:.4f}, Δ={oof_metrics['f1_mean']-R:+.4f})")
    print(f"  F1_05   = {oof_metrics['f1_05']:.4f}  (R_05={R_05:.4f}, Δ={oof_metrics['f1_05']-R_05:+.4f})")
    print(f"  F1_01   = {oof_metrics['f1_01']:.4f}  F1_03 = {oof_metrics['f1_03']:.4f}")
    print(f"  TP={oof_metrics['tp']}  FP={oof_metrics['fp']}  FN={oof_metrics['fn']}")

    # ── Decision ──────────────────────────────────────────────────────────────
    delta_mean = oof_metrics["f1_mean"] - R
    delta_05   = oof_metrics["f1_05"]   - (R_05 or 0.0)
    if delta_mean < 0:
        decision = "DROP"
        notes    = f"OOF F1_mean < R ({oof_metrics['f1_mean']:.4f} < {R:.4f})"
    elif oof_metrics["f1_mean"] >= R + 0.005 and oof_metrics["f1_05"] >= (R_05 or 0) + 0.02:
        decision = "KEEP"
        notes    = f"OOF F1_mean≥R+0.005 and F1_05≥R_05+0.02 → KEEP"
    else:
        decision = "DROP"
        notes    = (f"OOF Δ_mean={delta_mean:+.4f} (need≥+0.005), "
                    f"Δ_05={delta_05:+.4f} (need≥+0.020)")
    print(f"\nDecision: {decision}  — {notes}")

    if decision == "DROP":
        entry = _make_log_entry(R, R_05, oof_metrics, None, None, global_params,
                                global_params, decision, notes)
        append_log(entry)
        print("\nNot saving calibration params (DROP).")
        return

    # ── Val evaluation (exactly once) ─────────────────────────────────────────
    print("\n=== Val evaluation (1 run) ===")

    # Save global calibration params
    CAL_PARAMS_F.parent.mkdir(parents=True, exist_ok=True)
    CAL_PARAMS_F.write_text(json.dumps(global_params, indent=2), encoding="utf-8")
    print(f"Calibration params saved: {CAL_PARAMS_F}")
    print(f"  {global_params}")

    # Load val streams and apply p19c then calibrate
    from experiments.pipelines.pipeline_19c import run as p19c_run
    from experiments.models.lightgbm_classifier import load_classifier_rich_aug
    from abc2027.evaluate import stream_durations as sd_val

    model_full = load_classifier_rich_aug()
    val_streams = load_val()
    val_durs    = stream_durations(STREAMS_VAL)

    # p19c threshold from log
    thr = 0.475  # default from p19c log
    p19c_entry = _load_log_entry("pipeline_19c")
    notes_str  = p19c_entry.get("notes", "")
    import re
    m = re.search(r"thr(?:eshold)?=([0-9.]+)", notes_str, re.IGNORECASE)
    if m:
        thr = float(m.group(1))

    val_rows_uncal = p19c_run(val_streams, model_full, threshold=thr)

    # Group and calibrate
    preds_by_val: dict = {}
    for row in val_rows_uncal:
        preds_by_val.setdefault(row["stream_id"], []).append({
            "t_start": float(row["t_start"]),
            "t_end":   float(row["t_end"]),
            "score":   float(row["score"]),
        })

    cal_val_rows = []
    for sid, evs in preds_by_val.items():
        dur_s = val_durs.get(sid, 1800.0)
        caled = apply_calibration(evs, global_params, dur_s)
        for ev in caled:
            cal_val_rows.append({
                "stream_id": sid,
                "t_start":   f"{ev['t_start']:.3f}",
                "t_end":     f"{ev['t_end']:.3f}",
                "score":     f"{ev['score']:.6f}",
                "t_alarm":   f"{ev['t_start']:.3f}",
            })

    val_path   = save_sub(cal_val_rows, "pipeline_21")
    val_metrics = run_eval(val_path)

    print(f"\n=== Val Metrics (p21) ===")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    # ── Log ───────────────────────────────────────────────────────────────────
    delta_val = round(val_metrics["f1_mean"] - (p19c_entry.get("f1_mean") or 0.0), 4)
    entry = {
        "experiment_id": "pipeline_21",
        "phase": 4,
        "base": "pipeline_19c",
        "description": (f"p19c + interval_calibrator Form {form} "
                        f"(ds={global_params.get('ds',global_params.get('dc',0)):+.2f}, "
                        f"nested OOF)"),
        "components": ["interval_calibrator"],
        "oof_f1_mean":  oof_metrics["f1_mean"],
        "oof_f1_01":    oof_metrics["f1_01"],
        "oof_f1_03":    oof_metrics["f1_03"],
        "oof_f1_05":    oof_metrics["f1_05"],
        "oof_tp":       oof_metrics["tp"],
        "oof_fp":       oof_metrics["fp"],
        "oof_fn":       oof_metrics["fn"],
        "delta_oof_vs_base": round(delta_mean, 4),
        "p_delta_gt0": None,
        "f1_mean": val_metrics["f1_mean"],
        "f1_01":   val_metrics["f1_01"],
        "f1_03":   val_metrics["f1_03"],
        "f1_05":   val_metrics["f1_05"],
        "tp":      val_metrics["tp"],
        "fp":      val_metrics["fp"],
        "fn":      val_metrics["fn"],
        "false_alarms_per_hour": val_metrics["false_alarms_per_hour"],
        "delta_f1": delta_val,
        "probe_auc": None,
        "probe_macro_f1": None,
        "gate": None,
        "decision": decision,
        "notes": (f"Form {form}: {global_params}. "
                  f"Δ_oof_mean={delta_mean:+.4f}, Δ_oof_05={delta_05:+.4f}. {notes}"),
    }
    append_log(entry)
    print(f"\nLogged: {entry['experiment_id']}  decision={decision}")


def _make_log_entry(R, R_05, oof_metrics, val_metrics, delta_val, global_params, params, decision, notes):
    return {
        "experiment_id": "pipeline_21",
        "phase": 4,
        "base": "pipeline_19c",
        "description": "p19c + interval_calibrator (nested OOF)",
        "components": ["interval_calibrator"],
        "oof_f1_mean":  oof_metrics["f1_mean"] if oof_metrics else None,
        "oof_f1_01":    oof_metrics["f1_01"]   if oof_metrics else None,
        "oof_f1_03":    oof_metrics["f1_03"]   if oof_metrics else None,
        "oof_f1_05":    oof_metrics["f1_05"]   if oof_metrics else None,
        "oof_tp":       oof_metrics["tp"]       if oof_metrics else None,
        "oof_fp":       oof_metrics["fp"]       if oof_metrics else None,
        "oof_fn":       oof_metrics["fn"]       if oof_metrics else None,
        "delta_oof_vs_base": round((oof_metrics["f1_mean"] - R), 4) if oof_metrics and R else None,
        "p_delta_gt0":  None,
        "f1_mean": val_metrics["f1_mean"] if val_metrics else None,
        "f1_01":   val_metrics["f1_01"]   if val_metrics else None,
        "f1_03":   val_metrics["f1_03"]   if val_metrics else None,
        "f1_05":   val_metrics["f1_05"]   if val_metrics else None,
        "tp":      val_metrics["tp"]       if val_metrics else None,
        "fp":      val_metrics["fp"]       if val_metrics else None,
        "fn":      val_metrics["fn"]       if val_metrics else None,
        "false_alarms_per_hour": val_metrics["false_alarms_per_hour"] if val_metrics else None,
        "delta_f1": delta_val,
        "probe_auc": None,
        "probe_macro_f1": None,
        "gate": None,
        "decision": decision,
        "notes": notes,
    }


if __name__ == "__main__":
    main()
