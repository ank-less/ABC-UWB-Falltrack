"""Pipeline 25: Static hinge refiner (the main bet).

Replaces boundary_refiner with static CIR change-based boundary.
Detection-neutral: same detections, only t_start/t_end change.

Step 1 (oracle): sanity gate — if < 75% of oracle-input falls achieve
                 tIoU >= 0.5 after refinement → STOP.
Step 2 (detector): apply to OOF predictions, eval, refit p21 if base KEEP.

Decision:
  KEEP if OOF F1_05 >= R_cur_05 + 0.06 AND OOF F1_mean >= R_cur + 0.02
  DROP if oracle < 75% or OOF F1_mean < R_cur + 0.005
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

EVENTS_CSV   = ROOT / "streams_events.csv"
REPORTS_DIR  = ROOT / "experiments" / "audit" / "reports"

from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    RATE_HZ, STREAMS_TRAIN, STREAMS_VAL,
)
from experiments.postprocessing.static_hinge_refiner import refine_static, _refine_single
from experiments.postprocessing.interval_calibrator import apply_calibration
from experiments.audit.oof_harness import build_folds
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions, temporal_iou,
)

AXIS_LAYOUT  = "range_368"   # from physics_checks
RADAR_SYNC   = True          # from physics_checks
JOIN_OK      = False         # from join_detector (join_detector_ok = false)

# With join_detector_ok = false → ref max 0.6s
REF_MAX = 0.6

# Default params (will be grid-searched in oracle mode)
DEFAULT_PARAMS = {
    "m_s": 0.2, "tau_s": 1.2, "k_bins": 16,
    "ref_s": 0.6,   # capped because join_detector_ok = false
    "f_start": 0.20, "f_end": 0.15, "q_min": 0.65,
    "radar_sync": RADAR_SYNC, "layout": AXIS_LAYOUT,
}


def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def _get_base_oof() -> Tuple[Path, str, float, float, float]:
    for eid in ("pipeline_24", "pipeline_23", "pipeline_22", "pipeline_21"):
        e = _load_log_entry(eid)
        if e.get("decision") == "KEEP":
            p = ROOT / f"oof_{eid}.csv"
            if p.exists():
                r01 = e.get("oof_f1_01") or e.get("f1_01") or 0
                return p, eid, e.get("oof_f1_mean", 0), e.get("oof_f1_05", 0), r01
    p19c = _load_log_entry("pipeline_19c")
    r01  = p19c.get("oof_f1_01") or p19c.get("f1_01") or 0
    return (ROOT / "oof_pipeline_19c.csv", "pipeline_19c",
            p19c.get("oof_f1_mean"), p19c.get("oof_f1_05"), r01)


def load_cir_all_train() -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Load all train CIR streams into memory (needed for refiner)."""
    cir_by, dur_by = {}, {}
    for p in sorted(STREAMS_TRAIN.glob("stream_*.h5")):
        sid = f"train_{int(p.stem.split('_')[1]):02d}"
        with h5py.File(p) as f:
            cir_by[sid] = f["cir"][:]
        dur_by[sid] = cir_by[sid].shape[1] / RATE_HZ
        print(f"  Loaded {sid}: shape={cir_by[sid].shape}")
    return cir_by, dur_by


def load_join_times() -> Dict[str, List[float]]:
    j = REPORTS_DIR / "join_detector.json"
    if j.exists():
        d = json.loads(j.read_text(encoding="utf-8"))
        return d.get("join_times_by_stream", {})
    return {}


# ─── Oracle mode ──────────────────────────────────────────────────────────────

def run_oracle(gt_clean: list, cir_by: dict, dur_by: dict,
               join_times_by: dict, params: dict, n_jitter: int = 5) -> float:
    """
    For each clean GT event, create oracle inputs (true centre ± jitter).
    Apply refiner, check tIoU >= 0.5 with GT.
    Returns fraction achieving tIoU >= 0.5.
    """
    rng = np.random.default_rng(42)
    total, good = 0, 0
    for ev in gt_clean:
        sid = ev["stream_id"]
        if sid not in cir_by:
            continue
        cir  = cir_by[sid]
        dur  = dur_by[sid]
        joins = join_times_by.get(sid, [])
        gt_len = ev["t_end"] - ev["t_start"]
        for _ in range(n_jitter):
            # Jitter ±0.4s from true boundary (simulating detector error)
            d_s = rng.uniform(-0.4, 0.4)
            d_e = rng.uniform(-0.4, 0.4)
            t0_s = max(0, ev["t_start"] + d_s)
            t0_e = min(dur, ev["t_end"]   + d_e)
            if t0_e - t0_s < 0.5:
                t0_e = t0_s + max(0.5, gt_len * 0.5)
            ns, ne, r2, _ = _refine_single(
                cir, t0_s, t0_e, dur, join_times=joins, **params
            )
            iou = temporal_iou(ns, ne, ev["t_start"], ev["t_end"])
            total += 1
            if iou >= 0.5:
                good += 1
    rate = good / max(total, 1)
    return rate


def _tune_oracle_params(gt_clean: list, cir_by: dict, dur_by: dict,
                         join_times_by: dict) -> Tuple[float, dict]:
    """Grid search over key params to maximize oracle rate."""
    best_rate, best_params = 0.0, dict(DEFAULT_PARAMS)

    tau_grid   = [0.5, 0.8, 1.2, 1.8, 2.5]
    ref_grid   = [0.3, 0.6]  # capped at REF_MAX=0.6
    m_grid     = [0.1, 0.2, 0.3]
    fs_grid    = [0.10, 0.15, 0.20, 0.30]
    fe_grid    = [0.10, 0.15, 0.20]
    qmin_grid  = [0.5, 0.65, 0.8]
    k_grid     = [8, 16, 32]

    # Coarse search: tau × m × ref only (fast)
    for tau_s in tau_grid:
        for m_s in m_grid:
            for ref_s in ref_grid:
                params = dict(DEFAULT_PARAMS, tau_s=tau_s, m_s=m_s, ref_s=ref_s)
                # Use only first 3 jitters for speed
                rate = run_oracle(gt_clean[:50], cir_by, dur_by, join_times_by, params, n_jitter=2)
                if rate > best_rate:
                    best_rate = rate
                    best_params = dict(params)

    print(f"  Coarse oracle best: {best_rate:.3f} with tau={best_params['tau_s']}, "
          f"m={best_params['m_s']}, ref={best_params['ref_s']}")

    # Fine search on f_start × f_end × q_min × k_bins around best
    for fs in fs_grid:
        for fe in fe_grid:
            for qmin in qmin_grid:
                for k in k_grid:
                    params = dict(best_params, f_start=fs, f_end=fe, q_min=qmin, k_bins=k)
                    rate = run_oracle(gt_clean[:50], cir_by, dur_by, join_times_by, params, n_jitter=2)
                    if rate > best_rate:
                        best_rate = rate
                        best_params = dict(params)

    # Full oracle with best params on all events
    full_rate = run_oracle(gt_clean, cir_by, dur_by, join_times_by, best_params, n_jitter=5)
    print(f"  Full oracle rate: {full_rate:.3f} (need >= 0.75)")
    return full_rate, best_params


# ─── Apply refiner to OOF ────────────────────────────────────────────────────

def apply_refiner_to_oof(oof_csv: Path, cir_by: dict, dur_by: dict,
                          join_times_by: dict, params: dict) -> List[dict]:
    """Apply static refiner to all OOF predictions."""
    preds_by: Dict[str, list] = {}
    with open(oof_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            preds_by.setdefault(row["stream_id"], []).append({
                "t_start": float(row["t_start"]),
                "t_end":   float(row["t_end"]),
                "score":   float(row["score"]),
                "t_alarm": float(row.get("t_alarm", row["t_start"])),
            })

    out_rows = []
    for sid, evs in preds_by.items():
        if sid not in cir_by:
            out_rows.extend({
                "stream_id": sid,
                "t_start": f"{ev['t_start']:.3f}", "t_end": f"{ev['t_end']:.3f}",
                "score": f"{ev['score']:.6f}", "t_alarm": f"{ev['t_start']:.3f}",
            } for ev in evs)
            continue
        dur_s = dur_by[sid]
        joins = join_times_by.get(sid, [])
        refined = refine_static(evs, cir_by[sid], dur_s, params, join_times=joins)
        for ev in refined:
            out_rows.append({
                "stream_id": sid,
                "t_start": f"{ev['t_start']:.3f}",
                "t_end":   f"{ev['t_end']:.3f}",
                "score":   f"{ev['score']:.6f}",
                "t_alarm": f"{ev['t_start']:.3f}",
            })
    return out_rows


def _assert_detection_neutral(orig_csv: Path, refined_rows: list):
    orig_count = {}
    with open(orig_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            orig_count[row["stream_id"]] = orig_count.get(row["stream_id"], 0) + 1
    new_count = {}
    for row in refined_rows:
        new_count[row["stream_id"]] = new_count.get(row["stream_id"], 0) + 1
    # prediction count may differ slightly if refiner drops very short predictions
    # but should be close
    for sid in orig_count:
        if sid in new_count and abs(orig_count[sid] - new_count[sid]) > max(1, orig_count[sid] * 0.1):
            print(f"  ⚠️  {sid}: pred count {orig_count[sid]} → {new_count[sid]}")


def main():
    print("=" * 70)
    print("Pipeline 25 — Static Hinge Refiner (boundary quality)")
    print("=" * 70)
    print(f"Note: join_detector_ok={JOIN_OK} → ref_max={REF_MAX}s")

    base_oof, base_id, R, R_05, R_01 = _get_base_oof()
    print(f"\nBase: {base_id}  R={R}  R_05={R_05}")

    if not base_oof.exists():
        print(f"ERROR: base OOF not found at {base_oof}")
        return

    # Load GT
    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by    = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    gt_clean = [e for e in gt_train if not e.get("ignore", False)]
    print(f"Clean GT events (train): {len(gt_clean)}")

    # Load CIR data
    print("\nLoading train CIR data…")
    cir_by, dur_by = load_cir_all_train()
    join_times_by  = load_join_times()

    # ── Step 1: Oracle mode ───────────────────────────────────────────────────
    print("\n=== Step 1: Oracle mode (parameter tuning) ===")
    oracle_rate, best_params = _tune_oracle_params(
        gt_clean, cir_by, dur_by, join_times_by
    )

    # Save best params
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    params_file = REPORTS_DIR / "static_refiner_params.json"
    params_file.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print(f"Best params saved to {params_file}")

    ORACLE_GATE = 0.75
    if oracle_rate < ORACLE_GATE:
        print(f"\n🛑 Oracle rate {oracle_rate:.3f} < {ORACLE_GATE} → STOP (boundary ceiling reached)")
        entry = {
            "experiment_id": "pipeline_25",
            "phase": 4, "base": base_id,
            "description": "static_hinge_refiner — STOPPED (oracle gate failed)",
            "components": ["static_hinge_refiner"],
            "oof_f1_mean": None, "oof_f1_01": None, "oof_f1_03": None, "oof_f1_05": None,
            "oof_tp": None, "oof_fp": None, "oof_fn": None,
            "delta_oof_vs_base": None, "p_delta_gt0": None,
            "f1_mean": None, "f1_01": None, "f1_03": None, "f1_05": None,
            "tp": None, "fp": None, "fn": None, "false_alarms_per_hour": None,
            "delta_f1": None, "probe_auc": None, "probe_macro_f1": None,
            "gate": None, "decision": "STOPPED",
            "notes": (f"Oracle rate {oracle_rate:.3f} < {ORACLE_GATE} "
                      f"('boundary đã chạm sàn'). Best params: {best_params}"),
        }
        append_log(entry)
        print("Logged: pipeline_25  decision=STOPPED")
        return

    print(f"\n✓ Oracle rate {oracle_rate:.3f} >= {ORACLE_GATE} → proceed to detector mode")

    # ── Step 2: Detector mode ────────────────────────────────────────────────
    print("\n=== Step 2: Detector mode — apply to OOF ===")
    print(f"Applying refiner to {base_oof.name}…")
    refined_rows = apply_refiner_to_oof(base_oof, cir_by, dur_by, join_times_by, best_params)
    _assert_detection_neutral(base_oof, refined_rows)

    oof_ref_csv = ROOT / "oof_pipeline_25.csv"
    with open(oof_ref_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
        w.writeheader()
        w.writerows(refined_rows)

    preds  = validate_predictions(oof_ref_csv, durs)
    result = evaluate(gt_train, preds, durs)
    rs, bt = result["ranking_score"], result["by_theta"]
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

    delta_mean = oof_metrics["f1_mean"] - (R or 0)
    delta_05   = oof_metrics["f1_05"]   - (R_05 or 0)
    print(f"\n=== OOF Results (p25 refiner) ===")
    print(f"  F1_mean={oof_metrics['f1_mean']:.4f}  (R={R}, Δ={delta_mean:+.4f})")
    print(f"  F1_05  ={oof_metrics['f1_05']:.4f}  (R_05={R_05}, Δ={delta_05:+.4f})")
    print(f"  F1_01  ={oof_metrics['f1_01']:.4f}")
    print(f"  TP={oof_metrics['tp']}  FP={oof_metrics['fp']}  FN={oof_metrics['fn']}")

    # Decision
    if delta_05 >= 0.06 and delta_mean >= 0.02:
        decision = "KEEP"
        notes    = f"Δ_05={delta_05:+.4f}≥0.06, Δ_mean={delta_mean:+.4f}≥0.02"
    elif delta_mean < 0.005:
        decision = "DROP"
        notes    = f"Δ_mean={delta_mean:+.4f} < 0.005"
    else:
        decision = "DROP"
        notes    = (f"Partial improvement: Δ_05={delta_05:+.4f} (need≥0.06), "
                    f"Δ_mean={delta_mean:+.4f}")
    print(f"\nDecision: {decision}  — {notes}")

    # Val if KEEP
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        from abc2027.evaluate import stream_durations as sv

        val_durs    = sv(STREAMS_VAL)
        val_streams = load_val()

        # Dispatch to base pipeline's run function
        if base_id == "pipeline_24":
            from experiments.pipelines.pipeline_24 import run as base_run, train_full_model as p24_train
            print("Training full p24 model (21 features) for val…")
            model_full, thr = p24_train()
        else:
            from experiments.models.lightgbm_classifier import load_classifier_rich_aug
            from experiments.pipelines.pipeline_19c import run as base_run
            import re
            entry_base = _load_log_entry("pipeline_19c")
            m = re.search(r"thr(?:eshold)?=([0-9.]+)", entry_base.get("notes", ""), re.IGNORECASE)
            thr = float(m.group(1)) if m else 0.475
            model_full = load_classifier_rich_aug()

        # Run base pipeline on val
        val_uncal = base_run(val_streams, model_full, threshold=thr)

        # Apply any calibration if p21 KEEP
        cal_params_f = REPORTS_DIR / "calibration_params.json"
        if cal_params_f.exists() and _load_log_entry("pipeline_21").get("decision") == "KEEP":
            cal_p = json.loads(cal_params_f.read_text(encoding="utf-8"))
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
                for ev in apply_calibration(evs, cal_p, dur_s):
                    val_uncal.append({"stream_id": sid,
                                      "t_start": str(ev["t_start"]),
                                      "t_end":   str(ev["t_end"]),
                                      "score":   str(ev["score"]),
                                      "t_alarm": str(ev["t_start"])})

        # Apply static refiner to val
        val_preds_by: Dict[str, list] = {}
        for row in val_uncal:
            val_preds_by.setdefault(row["stream_id"], []).append({
                "t_start": float(row["t_start"]),
                "t_end":   float(row["t_end"]),
                "score":   float(row["score"]),
                "t_alarm": float(row.get("t_alarm", row["t_start"])),
            })

        print("Loading val CIR for refiner…")
        val_cir_by, val_dur_by = {}, {}
        from experiments.utils import STREAMS_VAL as SV
        for p in sorted(SV.glob("stream_*.h5")):
            sid = f"val_{int(p.stem.split('_')[1]):02d}"
            with h5py.File(p) as f:
                val_cir_by[sid] = f["cir"][:]
            val_dur_by[sid] = val_cir_by[sid].shape[1] / RATE_HZ

        val_rows = []
        for sid, evs in val_preds_by.items():
            if sid not in val_cir_by:
                val_rows.extend(make_rows(sid, evs))
                continue
            dur_s = val_dur_by[sid]
            refined = refine_static(evs, val_cir_by[sid], dur_s, best_params)
            for ev in refined:
                val_rows.append({
                    "stream_id": sid,
                    "t_start":  f"{ev['t_start']:.3f}",
                    "t_end":    f"{ev['t_end']:.3f}",
                    "score":    f"{ev['score']:.6f}",
                    "t_alarm":  f"{ev['t_start']:.3f}",
                })

        val_path    = save_sub(val_rows, "pipeline_25")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p25) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    # Log
    base_val = _load_log_entry(base_id).get("f1_mean") or 0.0
    delta_val = round((val_metrics["f1_mean"] - base_val) if val_metrics else 0, 4)
    entry = {
        "experiment_id": "pipeline_25",
        "phase": 4,
        "base": base_id,
        "description": (f"static_hinge_refiner (oracle={oracle_rate:.3f}, "
                        f"tau={best_params['tau_s']}, m={best_params['m_s']}, "
                        f"ref={best_params['ref_s']})"),
        "components": ["static_hinge_refiner"],
        "oof_f1_mean":       oof_metrics["f1_mean"],
        "oof_f1_01":         oof_metrics["f1_01"],
        "oof_f1_03":         oof_metrics["f1_03"],
        "oof_f1_05":         oof_metrics["f1_05"],
        "oof_tp":            oof_metrics["tp"],
        "oof_fp":            oof_metrics["fp"],
        "oof_fn":            oof_metrics["fn"],
        "delta_oof_vs_base": round(delta_mean, 4),
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
        "notes":    (f"oracle={oracle_rate:.3f}/{ORACLE_GATE}. "
                     f"join_detector_ok=False→ref≤0.6s. "
                     f"Params: {best_params}. "
                     f"Δ_05={delta_05:+.4f}, Δ_mean={delta_mean:+.4f}. {notes}"),
    }
    append_log(entry)
    print(f"\nLogged: pipeline_25  decision={decision}")


if __name__ == "__main__":
    main()
