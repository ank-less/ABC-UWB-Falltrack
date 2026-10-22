"""Pipeline 22: p21 (base after p21 if KEEP, else p19c) + temporal NMS.

GĐ4 — Suppresses FP fragments clustered after a fall detection.
Only runs if gate_p22 = True (from error_audit.py).

Decision:
  KEEP if OOF FP_pure decreases >= 30% AND OOF TP_0.1 unchanged
       AND OOF F1_mean >= R_cur + 0.005
  DROP if TP_0.1 drops or OOF F1_mean < R_cur
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

EVENTS_CSV   = ROOT / "streams_events.csv"
REPORTS_DIR  = ROOT / "experiments" / "audit" / "reports"
CAL_PARAMS_F = REPORTS_DIR / "calibration_params.json"

from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    RATE_HZ, STREAMS_TRAIN, STREAMS_VAL,
)
from experiments.postprocessing.temporal_nms import temporal_nms
from experiments.postprocessing.gap_merger import gap_merge
from experiments.audit.oof_harness import build_folds
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions,
)


def _temporal_iou(s1, e1, s2, e2) -> float:
    ov = min(e1, e2) - max(s1, s2)
    if ov <= 0:
        return 0.0
    un = (e1 - s1) + (e2 - s2) - ov
    return ov / un if un > 0 else 0.0


def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def _load_oof_csv(path: Path) -> dict:
    preds = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            preds.setdefault(row["stream_id"], []).append({
                "t_start": float(row["t_start"]),
                "t_end":   float(row["t_end"]),
                "score":   float(row["score"]),
                "t_alarm": float(row.get("t_alarm", row["t_start"])),
            })
    return preds


def _apply_nms_merge(preds_by: dict, K: float, gap_g: float,
                     durs: dict) -> list:
    """Apply temporal NMS + optional gap-merge."""
    rows = []
    for sid, evs in preds_by.items():
        evs2 = temporal_nms(evs, K)
        if gap_g > 0:
            evs2 = gap_merge(evs2, max_gap=gap_g)
        for ev in evs2:
            rows.append({
                "stream_id": sid,
                "t_start":  f"{ev['t_start']:.3f}",
                "t_end":    f"{ev['t_end']:.3f}",
                "score":    f"{ev['score']:.6f}",
                "t_alarm":  f"{ev.get('t_alarm', ev['t_start']):.3f}",
            })
    return rows


def _count_pure_fp_and_tp(oof_csv: Path, gt_by: dict, durs: dict,
                            theta: float = 0.1):
    """Count pure FP (tIoU<0.1) and TP at theta."""
    preds_by = _load_oof_csv(oof_csv)
    tp_total, fp_total = 0, 0
    for sid in sorted(set(list(gt_by.keys()) + list(preds_by.keys()))):
        if not sid.startswith("train_"):
            continue
        gt = gt_by.get(sid, [])
        pr = sorted(preds_by.get(sid, []), key=lambda x: -x["score"])
        used = set()
        tp = 0
        for pred in pr:
            best_iou, best_j = 0.0, -1
            for j, ev in enumerate(gt):
                if j in used:
                    continue
                iou = _temporal_iou(pred["t_start"], pred["t_end"], ev["t_start"], ev["t_end"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= theta:
                used.add(best_j)
                tp += 1
        tp_total += tp
        fp_total += len(pr) - tp
    return tp_total, fp_total


def main():
    print("=" * 70)
    print("Pipeline 22 — Temporal NMS (FP fragment suppression)")
    print("=" * 70)

    # ── Gate check ────────────────────────────────────────────────────────────
    gates_f = REPORTS_DIR / "gates.json"
    if not gates_f.exists():
        print(f"ERROR: gates.json not found at {gates_f}")
        print("  Run error_audit.py first.")
        return

    gates = json.loads(gates_f.read_text(encoding="utf-8"))
    gate_p22 = gates.get("gate_p22", False)
    if not gate_p22:
        print("Gate p22 = False → SKIPPED_GATE")
        entry = _make_log_entry(None, None, None, None, "SKIPPED_GATE",
                                "gate_p22 = False")
        append_log(entry)
        return
    print("Gate p22 = True → proceeding")

    # ── Load base OOF (p21 if KEEP else p19c) ────────────────────────────────
    p21_entry = _load_log_entry("pipeline_21")
    if p21_entry.get("decision") == "KEEP":
        base_oof = ROOT / "oof_pipeline_21.csv"
        base_id  = "pipeline_21"
        R     = p21_entry.get("oof_f1_mean")
        R_05  = p21_entry.get("oof_f1_05")
    else:
        base_oof = ROOT / "oof_pipeline_19c.csv"
        base_id  = "pipeline_19c"
        p19c = _load_log_entry("pipeline_19c")
        R    = p19c.get("oof_f1_mean")
        R_05 = p19c.get("oof_f1_05")
    print(f"Base: {base_id}  R={R}  R_05={R_05}")

    if not base_oof.exists():
        print(f"ERROR: base OOF CSV not found at {base_oof}")
        return

    # ── Load GT ───────────────────────────────────────────────────────────────
    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by    = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    # Min centre gap from audit (to cap K)
    audit_f = REPORTS_DIR / "oof_errors.csv"
    min_gap  = 999.0
    if audit_f.exists():
        import csv as _csv
        with open(audit_f, encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                pass  # just check it's readable
        # Load from gates which has min_center_gap
        min_gap = gates.get("min_center_gap_s", 999.0)
    K_max = min(8.0, min_gap - 0.5) if min_gap < 999 else 8.0
    print(f"min_center_gap={min_gap:.1f}s → K_max={K_max:.1f}s")

    # Base TP/FP counts
    base_tp, base_fp = _count_pure_fp_and_tp(base_oof, gt_by, durs)
    print(f"Base OOF: TP(θ=0.1)={base_tp}  pure_FP={base_fp}")

    # ── Grid search K × gap_g ─────────────────────────────────────────────────
    K_vals   = [k for k in (3, 5, 8) if k <= K_max]
    gap_vals = [0.0, 1.0, 1.5, 2.0]

    preds_by = _load_oof_csv(base_oof)
    best_f1, best_params = -1.0, None
    tmp_csv  = ROOT / "_nms_tmp.csv"

    for K in K_vals:
        for gap_g in gap_vals:
            rows = _apply_nms_merge(preds_by, K, gap_g, durs)
            with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
                w.writeheader()
                w.writerows(rows)
            try:
                preds = validate_predictions(tmp_csv, durs)
                res   = evaluate(gt_train, preds, durs)
                f1    = round(res["ranking_score"]["f1_mean"], 4)
            except Exception as ex:
                f1 = 0.0
                print(f"  K={K} gap={gap_g}: eval error {ex}")
                continue
            print(f"  K={K}s  gap={gap_g}s  F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_params = {"K": K, "gap_g": gap_g}

    tmp_csv.unlink(missing_ok=True)

    if best_params is None:
        print("No valid params found → DROP")
        entry = _make_log_entry(None, None, None, None, "DROP", "No valid NMS params")
        append_log(entry)
        return

    # ── Evaluate best params ───────────────────────────────────────────────────
    K, gap_g = best_params["K"], best_params["gap_g"]
    rows = _apply_nms_merge(preds_by, K, gap_g, durs)
    oof_nms_csv = ROOT / "oof_pipeline_22.csv"
    with open(oof_nms_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        w.writeheader()
        w.writerows(rows)

    preds   = validate_predictions(oof_nms_csv, durs)
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
    nms_tp, nms_fp = _count_pure_fp_and_tp(oof_nms_csv, gt_by, durs)
    fp_reduction = (base_fp - nms_fp) / max(base_fp, 1)
    tp_change    = nms_tp - base_tp

    print(f"\n=== OOF NMS Results (K={K}, gap={gap_g}) ===")
    print(f"  F1_mean = {oof_metrics['f1_mean']:.4f}  (R={R}, Δ={oof_metrics['f1_mean']-(R or 0):+.4f})")
    print(f"  TP(θ=0.1): {base_tp} → {nms_tp}  (Δ={tp_change:+d})")
    print(f"  pure_FP:   {base_fp} → {nms_fp}  (reduction={fp_reduction:.1%})")

    delta_mean = oof_metrics["f1_mean"] - (R or 0)
    if tp_change < 0:
        decision = "DROP"
        notes    = f"TP(θ=0.1) dropped by {-tp_change}"
    elif fp_reduction < 0.30:
        decision = "DROP"
        notes    = f"FP reduction {fp_reduction:.1%} < 30%"
    elif delta_mean < 0.005:
        decision = "DROP"
        notes    = f"OOF Δ_mean={delta_mean:+.4f} < 0.005"
    else:
        decision = "KEEP"
        notes    = (f"FP↓{fp_reduction:.1%}, TP unchanged, "
                    f"Δ_mean={delta_mean:+.4f} ≥ 0.005")

    print(f"Decision: {decision}  — {notes}")

    # Val evaluation if KEEP
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        from experiments.pipelines.pipeline_19c import run as p19c_run
        from experiments.models.lightgbm_classifier import load_classifier_rich_aug
        from abc2027.evaluate import stream_durations as sv

        model_full = load_classifier_rich_aug()
        val_streams = load_val()
        val_durs    = sv(STREAMS_VAL)

        # Get p19c threshold
        p19c_entry = _load_log_entry("pipeline_19c")
        import re
        m = re.search(r"thr(?:eshold)?=([0-9.]+)", p19c_entry.get("notes", ""), re.IGNORECASE)
        thr = float(m.group(1)) if m else 0.475

        val_uncal = p19c_run(val_streams, model_full, threshold=thr)

        # Apply p21 calibration if KEEP
        from experiments.postprocessing.interval_calibrator import apply_calibration
        if CAL_PARAMS_F.exists() and p21_entry.get("decision") == "KEEP":
            cal_params = json.loads(CAL_PARAMS_F.read_text(encoding="utf-8"))
            evs_by = {}
            for row in val_uncal:
                evs_by.setdefault(row["stream_id"], []).append({
                    "t_start": float(row["t_start"]),
                    "t_end":   float(row["t_end"]),
                    "score":   float(row["score"]),
                })
            val_uncal = []
            for sid, evs in evs_by.items():
                dur_s = val_durs.get(sid, 1800.0)
                for ev in apply_calibration(evs, cal_params, dur_s):
                    val_uncal.append({"stream_id": sid, **{k: str(v) for k, v in ev.items()}})

        # Apply NMS
        preds_val = {}
        for row in val_uncal:
            preds_val.setdefault(row["stream_id"], []).append({
                "t_start": float(row["t_start"]),
                "t_end":   float(row["t_end"]),
                "score":   float(row["score"]),
                "t_alarm": float(row.get("t_alarm", row["t_start"])),
            })
        val_rows = []
        for sid, evs in preds_val.items():
            evs2 = temporal_nms(evs, K)
            if gap_g > 0:
                evs2 = gap_merge(evs2, max_gap=gap_g)
            for ev in evs2:
                val_rows.append({
                    "stream_id": sid,
                    "t_start":  f"{ev['t_start']:.3f}",
                    "t_end":    f"{ev['t_end']:.3f}",
                    "score":    f"{ev['score']:.6f}",
                    "t_alarm":  f"{ev.get('t_alarm', ev['t_start']):.3f}",
                })

        val_path    = save_sub(val_rows, "pipeline_22")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p22) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    entry = _make_log_entry(oof_metrics, val_metrics,
                            best_params, fp_reduction, tp_change, R,
                            decision, notes)
    append_log(entry)
    print(f"\nLogged: pipeline_22  decision={decision}")


def _make_log_entry(oof_metrics, val_metrics, best_params, fp_red, tp_ch,
                    R, decision, notes):
    p19c = _load_log_entry("pipeline_19c")
    base_val = p19c.get("f1_mean", 0.0)
    delta_val = round((val_metrics["f1_mean"] - base_val), 4) if val_metrics else None
    return {
        "experiment_id": "pipeline_22",
        "phase": 4,
        "base": "pipeline_21" if _load_log_entry("pipeline_21").get("decision") == "KEEP" else "pipeline_19c",
        "description": (f"NMS K={best_params['K']}s gap={best_params['gap_g']}s"
                        if best_params else "NMS — SKIPPED/DROPPED"),
        "components": ["temporal_nms"],
        "oof_f1_mean":  oof_metrics["f1_mean"] if oof_metrics else None,
        "oof_f1_01":    oof_metrics["f1_01"]   if oof_metrics else None,
        "oof_f1_03":    oof_metrics["f1_03"]   if oof_metrics else None,
        "oof_f1_05":    oof_metrics["f1_05"]   if oof_metrics else None,
        "oof_tp":       oof_metrics["tp"]       if oof_metrics else None,
        "oof_fp":       oof_metrics["fp"]       if oof_metrics else None,
        "oof_fn":       oof_metrics["fn"]       if oof_metrics else None,
        "delta_oof_vs_base": round((oof_metrics["f1_mean"] - (R or 0)), 4) if oof_metrics else None,
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
        "gate": "gate_p22",
        "decision": decision,
        "notes": (f"K={best_params['K']}s gap={best_params['gap_g']}s, "
                  f"FP_red={fp_red:.1%}, TP_Δ={tp_ch:+d}. "
                  if best_params and fp_red is not None else "") + (notes or ""),
    }


if __name__ == "__main__":
    main()
