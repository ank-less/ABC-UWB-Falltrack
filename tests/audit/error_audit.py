"""Error audit on OOF predictions of pipeline_19c.

Step 0.6: Analyze FN/FP/low-tIoU detections, compute gates for p22/p26/p27.
Outputs: oof_errors.csv, gates.json
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV    = ROOT / "streams_events.csv"
COMP_CSV      = ROOT / "streams_composition.csv"
REPORTS_DIR   = ROOT / "experiments" / "audit" / "reports"
RATE_HZ       = 50.0


def load_oof_predictions(oof_csv: Path) -> Dict[str, list]:
    """Load OOF predictions grouped by stream."""
    preds = {}
    with open(oof_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["stream_id"]
            preds.setdefault(sid, []).append({
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "score": float(row["score"]),
            })
    return preds


def load_join_times() -> Dict[str, List[float]]:
    """Load join times from join_detector report."""
    p = REPORTS_DIR / "join_detector.json"
    if not p.exists():
        return {}
    report = json.loads(p.read_text(encoding="utf-8"))
    return report.get("join_times_by_stream", {})


def temporal_iou(s1, e1, s2, e2) -> float:
    ov = min(e1, e2) - max(s1, s2)
    if ov <= 0:
        return 0.0
    un = (e1 - s1) + (e2 - s2) - ov
    return ov / un if un > 0 else 0.0


def run_error_audit(oof_csv: Optional[Path] = None) -> dict:
    """Analyze OOF predictions, compute error stats and gates.

    Args:
        oof_csv: path to OOF predictions CSV (default: oof_pipeline_19c.csv in ROOT)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if oof_csv is None:
        oof_csv = ROOT / "oof_pipeline_19c.csv"
    if not oof_csv.exists():
        print(f"ERROR: OOF CSV not found at {oof_csv}")
        print("  Run pipeline_19c first to generate OOF predictions.")
        return {"error": f"OOF CSV not found: {oof_csv}"}

    from abc2027.evaluate import load_ground_truth
    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by  = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)

    preds_by = load_oof_predictions(oof_csv)
    join_times = load_join_times()

    # Load composition for segment type info
    comp_by: Dict[str, list] = {}
    with open(COMP_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            comp_by.setdefault(row["stream_id"], []).append(row)

    error_rows = []
    fn_list    = []   # FN events (θ=0.1)
    fp_list    = []   # pure FP (tIoU < 0.1 with any GT)
    low_iou    = []   # matches with tIoU 0.1–0.5

    # Track statistics for gates
    fall_durations = []
    fall_margins   = []  # (pre_margin, post_margin) for each fall
    center_gaps    = []  # min gaps between fall centers in same stream
    flagged_positives = 0

    stage1_pooling = "sum"  # default; will be verified from p19 logic

    for sid in sorted(set(list(gt_by.keys()) + list(preds_by.keys()))):
        if not sid.startswith("train_"):
            continue

        gt_events = [e for e in gt_by.get(sid, []) if not e["ignore"]]
        ignored   = [e for e in gt_by.get(sid, []) if e["ignore"]]
        all_gt    = gt_by.get(sid, [])
        preds     = sorted(preds_by.get(sid, []), key=lambda x: -x["score"])
        joins     = join_times.get(sid, [])

        # Load stream duration
        p = STREAMS_TRAIN / f"stream_{int(sid.split('_')[1]):02d}.h5"
        if p.exists():
            with h5py.File(p) as f:
                T = f["cir"].shape[1]
            dur_s = T / RATE_HZ
        else:
            dur_s = 1800.0

        # Count flagged positives
        for ev in all_gt:
            if ev["ignore"]:
                flagged_positives += 1

        # Fall duration stats
        for ev in gt_events:
            fall_durations.append(ev["t_end"] - ev["t_start"])

        # Min center gaps
        if len(gt_events) >= 2:
            centers = sorted([(e["t_start"] + e["t_end"]) / 2 for e in gt_events])
            for i in range(len(centers) - 1):
                center_gaps.append(centers[i + 1] - centers[i])

        # Match predictions to GT
        used_gt = set()
        matched_preds = set()
        for pred_i, pred in enumerate(preds):
            best_iou, best_j = 0.0, -1
            for j, ev in enumerate(gt_events):
                if j in used_gt:
                    continue
                iou = temporal_iou(pred["t_start"], pred["t_end"], ev["t_start"], ev["t_end"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.1:
                used_gt.add(best_j)
                matched_preds.add(pred_i)

        # FNs (GT not matched at θ=0.1)
        for j, ev in enumerate(gt_events):
            if j not in used_gt:
                # Find best prediction overlap
                best_iou, best_score = 0.0, 0.0
                for pred in preds:
                    iou = temporal_iou(pred["t_start"], pred["t_end"], ev["t_start"], ev["t_end"])
                    if iou > best_iou:
                        best_iou = iou
                        best_score = pred["score"]

                # Distance to nearest join
                nearest_join = min([abs(ev["t_start"] - j) for j in joins] + [999.0])

                row = {
                    "stream_id": sid,
                    "event_id": ev.get("event_id", ""),
                    "type": "FN",
                    "gt_start": round(ev["t_start"], 3),
                    "gt_end": round(ev["t_end"], 3),
                    "pred_start": None,
                    "pred_end": None,
                    "best_iou": round(best_iou, 3),
                    "pred_score": round(best_score, 4),
                    "has_stage1_overlap": best_iou > 0,
                    "fall_duration": round(ev["t_end"] - ev["t_start"], 3),
                    "nearest_join_s": round(nearest_join, 3),
                }
                error_rows.append(row)
                fn_list.append(row)

        # Pure FPs (no GT overlap at θ=0.1)
        for pred_i, pred in enumerate(preds):
            if pred_i in matched_preds:
                continue
            # Check if it overlaps any ignored event (with ±2s margin)
            overlaps_ignored = any(
                min(pred["t_end"], ev["t_end"] + 2) - max(pred["t_start"], ev["t_start"] - 2) > 0
                for ev in ignored
            )
            if overlaps_ignored:
                continue

            # Find nearest GT fall end
            nearest_fall_after = min(
                [pred["t_start"] - ev["t_end"] for ev in all_gt
                 if pred["t_start"] > ev["t_end"]] + [999.0]
            )
            nearest_fall_before = min(
                [ev["t_start"] - pred["t_end"] for ev in all_gt
                 if ev["t_start"] > pred["t_end"]] + [999.0]
            )

            # Distance to nearest join
            nearest_join = min([abs(pred["t_start"] - j) for j in joins] + [999.0])

            # Segment kind at this time
            seg_kind = "unknown"
            for seg in comp_by.get(sid, []):
                if float(seg["t_start"]) <= pred["t_start"] <= float(seg["t_end"]):
                    seg_kind = seg["kind"]
                    break

            row = {
                "stream_id": sid,
                "event_id": "",
                "type": "FP",
                "gt_start": None,
                "gt_end": None,
                "pred_start": round(pred["t_start"], 3),
                "pred_end": round(pred["t_end"], 3),
                "best_iou": 0.0,
                "pred_score": round(pred["score"], 4),
                "has_stage1_overlap": False,
                "fall_duration": round(pred["t_end"] - pred["t_start"], 3),
                "nearest_join_s": round(nearest_join, 3),
                "time_after_gt_fall_s": round(nearest_fall_after, 3),
                "time_before_gt_fall_s": round(nearest_fall_before, 3),
                "segment_kind": seg_kind,
            }
            error_rows.append(row)
            fp_list.append(row)

        # Low-tIoU matches (0.1 ≤ tIoU < 0.5) — boundary errors
        for pred_i, pred in enumerate(preds):
            if pred_i not in matched_preds:
                continue
            for j, ev in enumerate(gt_events):
                iou = temporal_iou(pred["t_start"], pred["t_end"], ev["t_start"], ev["t_end"])
                if 0.1 <= iou < 0.5:
                    row = {
                        "stream_id": sid,
                        "event_id": ev.get("event_id", ""),
                        "type": "LOW_IOU",
                        "gt_start": round(ev["t_start"], 3),
                        "gt_end": round(ev["t_end"], 3),
                        "pred_start": round(pred["t_start"], 3),
                        "pred_end": round(pred["t_end"], 3),
                        "best_iou": round(iou, 3),
                        "pred_score": round(pred["score"], 4),
                        "delta_start": round(pred["t_start"] - ev["t_start"], 3),
                        "delta_end": round(pred["t_end"] - ev["t_end"], 3),
                        "len_ratio": round((pred["t_end"] - pred["t_start"]) / max(0.01, ev["t_end"] - ev["t_start"]), 3),
                    }
                    error_rows.append(row)
                    low_iou.append(row)

    # Save error CSV
    oof_errors_path = REPORTS_DIR / "oof_errors.csv"
    if error_rows:
        fieldnames = sorted(set().union(*[set(r.keys()) for r in error_rows]))
        with open(oof_errors_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(error_rows)
    print(f"Error audit CSV saved: {oof_errors_path}")
    print(f"  FN={len(fn_list)} FP={len(fp_list)} LOW_IOU={len(low_iou)}")

    # ── Compute gates ─────────────────────────────────────────────────────────
    # gate_p22: ≥ 25% FP in 1–15s after t_end of a real fall
    n_fp = len(fp_list)
    n_fp_after_fall = sum(
        1 for fp in fp_list
        if fp.get("time_after_gt_fall_s", 999) is not None
        and 1.0 <= fp.get("time_after_gt_fall_s", 999) <= 15.0
    )
    gate_p22_frac = n_fp_after_fall / max(1, n_fp)
    gate_p22 = gate_p22_frac >= 0.25
    print(f"\ngate_p22: {n_fp_after_fall}/{n_fp} FP within 1-15s after fall "
          f"({gate_p22_frac:.2%}) → {gate_p22}")

    # gate_p26: ≥ 25% FP within ±1s of a join
    n_fp_near_join = sum(
        1 for fp in fp_list
        if fp.get("nearest_join_s", 999) is not None
        and fp.get("nearest_join_s", 999) <= 1.0
    )
    gate_p26_frac = n_fp_near_join / max(1, n_fp)
    gate_p26 = gate_p26_frac >= 0.25
    print(f"gate_p26: {n_fp_near_join}/{n_fp} FP within ±1s of join "
          f"({gate_p26_frac:.2%}) → {gate_p26}")

    # gate_p27: ≥ 1/3 FN with no Stage-1 overlap
    n_fn = len(fn_list)
    n_fn_no_s1 = sum(1 for fn in fn_list if not fn.get("has_stage1_overlap", True))
    gate_p27_frac = n_fn_no_s1 / max(1, n_fn)
    gate_p27 = gate_p27_frac >= (1.0 / 3.0)
    print(f"gate_p27: {n_fn_no_s1}/{n_fn} FN with no Stage-1 overlap "
          f"({gate_p27_frac:.2%}) → {gate_p27}")

    # Additional stats
    stats = {}
    if fall_durations:
        stats["fall_dur_p5"] = round(float(np.percentile(fall_durations, 5)), 3)
        stats["fall_dur_p50"] = round(float(np.percentile(fall_durations, 50)), 3)
        stats["fall_dur_p95"] = round(float(np.percentile(fall_durations, 95)), 3)
    if center_gaps:
        stats["min_center_gap_s"] = round(float(np.min(center_gaps)), 3)
        stats["median_center_gap_s"] = round(float(np.median(center_gaps)), 3)
    stats["flagged_positive_count"] = flagged_positives
    stats["stage1_pooling"] = stage1_pooling

    print(f"\nAdditional stats:")
    print(f"  Fall duration p5/p50/p95: {stats.get('fall_dur_p5')}/{stats.get('fall_dur_p50')}/{stats.get('fall_dur_p95')} s")
    print(f"  Min center gap between falls: {stats.get('min_center_gap_s')} s")
    print(f"  Flagged positives: {stats.get('flagged_positive_count')}")

    gates = {
        "gate_p22": gate_p22,
        "gate_p22_frac": round(gate_p22_frac, 3),
        "gate_p26": gate_p26,
        "gate_p26_frac": round(gate_p26_frac, 3),
        "gate_p27": gate_p27,
        "gate_p27_frac": round(gate_p27_frac, 3),
        "n_fn": n_fn,
        "n_fp": n_fp,
        "n_low_iou": len(low_iou),
        "stats": stats,
    }

    gates_path = REPORTS_DIR / "gates.json"
    gates_path.write_text(json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGates saved to {gates_path}")

    return gates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-csv", default=None)
    args = parser.parse_args()

    oof_csv = Path(args.oof_csv) if args.oof_csv else None
    gates = run_error_audit(oof_csv=oof_csv)
    print("\nGates summary:")
    for k, v in gates.items():
        if k != "stats":
            print(f"  {k}: {v}")
