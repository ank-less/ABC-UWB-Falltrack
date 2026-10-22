"""Pipeline 29: Static CIR boundary refinement.

Apply lightweight static hinge refiner to OOF predictions from base.
Detection-neutral: stream_id, score, event count are unchanged.
Only t_start / t_end are modified.

Base: pipeline_28 (if KEEP) else pipeline_24
KEEP if OOF F1_05 >= R_05 + 0.02 AND OOF F1_mean >= R_mean - 0.01
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EVENTS_CSV  = ROOT / "streams_events.csv"
STREAMS_TRAIN = ROOT / "streams" / "train"
STREAMS_VAL   = ROOT / "streams" / "val"

from experiments.utils import (
    run_eval, save_sub, append_log, make_rows, RATE_HZ,
)
from experiments.postprocessing.static_hinge_refiner import apply_static_refiner
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def _pick_base() -> tuple[str, float, float]:
    """Return (base_id, R_mean, R_05) from latest KEEP'd pipeline."""
    for eid in ("pipeline_28", "pipeline_24"):
        e = _load_log_entry(eid)
        if e.get("decision") in ("KEEP", "BASE"):
            return eid, e.get("oof_f1_mean", 0.0), e.get("oof_f1_05", 0.0)
    # Fallback: use p24 fixed values
    return "pipeline_24", 0.5559, 0.3920


def _load_predictions_csv(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "stream_id": row["stream_id"],
                "t_start":   float(row["t_start"]),
                "t_end":     float(row["t_end"]),
                "score":     float(row["score"]),
                "t_alarm":   float(row["t_alarm"]),
            })
    return rows


def _stream_path(stream_id: str) -> Path:
    """Derive h5 path from stream_id (train_XX or val_XX)."""
    idx = int(stream_id.split("_")[1])
    if stream_id.startswith("train"):
        return STREAMS_TRAIN / f"stream_{idx:02d}.h5"
    else:
        return STREAMS_VAL / f"stream_{idx:02d}.h5"


def _get_stream_dur(stream_id: str) -> float:
    p = _stream_path(stream_id)
    with h5py.File(p, "r") as f:
        T = f["cir"].shape[1]
    return T / RATE_HZ


def apply_refiner_to_csv(
    pred_rows: List[dict],
    ctx_s: float = 3.0,
) -> List[dict]:
    """Apply static hinge refiner to all predictions, grouped by stream."""
    from collections import defaultdict
    by_stream = defaultdict(list)
    for row in pred_rows:
        by_stream[row["stream_id"]].append(row)

    out = []
    for sid, events in sorted(by_stream.items()):
        stream_p = _stream_path(sid)
        if not stream_p.exists():
            out.extend(events)
            continue
        dur_s = _get_stream_dur(sid)
        events_simple = [
            {"t_start": e["t_start"], "t_end": e["t_end"],
             "score": e["score"], "t_alarm": e["t_alarm"]}
            for e in events
        ]
        refined = apply_static_refiner(events_simple, stream_p, dur_s, ctx_s=ctx_s)
        for r in refined:
            out.append({"stream_id": sid, **r})
    return out


def _write_csv(rows: List[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
        w.writeheader()
        for row in rows:
            w.writerow({
                "stream_id": row["stream_id"],
                "t_start":   f"{row['t_start']:.3f}",
                "t_end":     f"{row['t_end']:.3f}",
                "score":     f"{row['score']:.6f}",
                "t_alarm":   f"{row['t_alarm']:.3f}",
            })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Pipeline 29 — Static CIR boundary refinement")
    print("=" * 70)

    base_id, R, R_05 = _pick_base()
    print(f"Base: {base_id}  OOF F1_mean={R:.4f}  OOF F1_05={R_05:.4f}")

    # Load base OOF predictions
    oof_base_csv = ROOT / f"oof_{base_id}.csv"
    if not oof_base_csv.exists():
        print(f"ERROR: {oof_base_csv} not found — run base pipeline first")
        return

    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    # Apply static refiner to OOF predictions
    print(f"\nApplying static hinge refiner to OOF predictions from {oof_base_csv.name}…")
    base_preds = _load_predictions_csv(oof_base_csv)
    print(f"  {len(base_preds)} predictions in {len(set(r['stream_id'] for r in base_preds))} streams")
    refined_preds = apply_refiner_to_csv(base_preds, ctx_s=3.0)

    oof_csv = ROOT / "oof_pipeline_29.csv"
    _write_csv(refined_preds, oof_csv)

    # Evaluate OOF
    preds  = validate_predictions(oof_csv, durs)
    result = evaluate(gt_train, preds, durs)
    rs, bt = result["ranking_score"], result["by_theta"]
    oof = {
        "f1_mean": round(rs["f1_mean"], 4),
        "f1_01":   round(bt["0.1"]["f1"], 4),
        "f1_03":   round(bt["0.3"]["f1"], 4),
        "f1_05":   round(bt["0.5"]["f1"], 4),
        "tp":      bt["0.3"]["tp"],
        "fp":      bt["0.3"]["fp"],
        "fn":      bt["0.3"]["fn"],
        "false_alarms_per_hour": round(rs["false_alarms_per_hour"], 1),
    }
    delta_mean = round(oof["f1_mean"] - R,    4)
    delta_05   = round(oof["f1_05"]   - R_05, 4)

    print(f"\n=== OOF Results (p29) ===")
    print(f"  F1_mean={oof['f1_mean']:.4f}  (R={R:.4f}, Δ={delta_mean:+.4f})")
    print(f"  F1_01  ={oof['f1_01']:.4f}")
    print(f"  F1_03  ={oof['f1_03']:.4f}")
    print(f"  F1_05  ={oof['f1_05']:.4f}  (R_05={R_05:.4f}, Δ_05={delta_05:+.4f})")
    print(f"  TP={oof['tp']}  FP={oof['fp']}  FN={oof['fn']}")

    # Decision
    if delta_05 >= 0.02 and delta_mean >= -0.01:
        decision = "KEEP"
        notes    = f"Δ_05={delta_05:+.4f}≥0.02 AND Δ_mean={delta_mean:+.4f}≥-0.01"
    elif delta_05 >= 0.01 and delta_mean >= 0.0:
        decision = "KEEP"
        notes    = f"Δ_05={delta_05:+.4f}≥0.01 AND Δ_mean={delta_mean:+.4f}≥0 — marginal KEEP"
    else:
        decision = "DROP"
        notes    = f"Δ_05={delta_05:+.4f}<0.01 or Δ_mean={delta_mean:+.4f}<-0.01 — refiner not effective"
    print(f"Decision: {decision}  — {notes}")

    # Val (1 run if KEEP)
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        # Load base val submission
        base_val_csv = ROOT / f"submission_{base_id}.csv"
        if not base_val_csv.exists():
            # Check my_submission folder
            base_val_csv = ROOT / "my_submission" / f"submission_{base_id}.csv"
        if not base_val_csv.exists():
            print(f"WARNING: base val submission not found, skipping val")
        else:
            val_preds = _load_predictions_csv(base_val_csv)
            print(f"  Loaded {len(val_preds)} val predictions")
            refined_val = apply_refiner_to_csv(val_preds, ctx_s=3.0)

            val_path = ROOT / "submission_pipeline_29.csv"
            _write_csv(refined_val, val_path)
            val_metrics = run_eval(val_path)
            print(f"\n=== Val Metrics (p29) ===")
            for k, v in val_metrics.items():
                print(f"  {k}: {v}")

    # Log
    base_val_f1 = _load_log_entry(base_id).get("f1_mean") or 0.0
    delta_val   = round((val_metrics["f1_mean"] - base_val_f1) if val_metrics else 0.0, 4)
    entry = {
        "experiment_id": "pipeline_29",
        "phase": 4,
        "base": base_id,
        "description": f"p29: static CIR hinge refiner on {base_id} OOF (ctx=3s, max_shift=2s)",
        "components": ["static_hinge_refiner"],
        "oof_f1_mean": oof["f1_mean"],
        "oof_f1_01":   oof["f1_01"],
        "oof_f1_03":   oof["f1_03"],
        "oof_f1_05":   oof["f1_05"],
        "oof_tp":      oof["tp"],
        "oof_fp":      oof["fp"],
        "oof_fn":      oof["fn"],
        "delta_oof_vs_base": delta_mean,
        "p_delta_gt0":       None,
        "f1_mean":  val_metrics["f1_mean"] if val_metrics else None,
        "f1_01":    val_metrics["f1_01"]   if val_metrics else None,
        "f1_03":    val_metrics["f1_03"]   if val_metrics else None,
        "f1_05":    val_metrics["f1_05"]   if val_metrics else None,
        "tp":       val_metrics["tp"]       if val_metrics else None,
        "fp":       val_metrics["fp"]       if val_metrics else None,
        "fn":       val_metrics["fn"]       if val_metrics else None,
        "false_alarms_per_hour": val_metrics["false_alarms_per_hour"] if val_metrics else None,
        "delta_f1": delta_val if val_metrics else None,
        "probe_auc":      None,
        "probe_macro_f1": None,
        "gate":     None,
        "decision": decision,
        "notes":    notes,
    }
    append_log(entry)
    print(f"\nLogged: pipeline_29  decision={decision}")
    return decision, oof, val_metrics


if __name__ == "__main__":
    main()
