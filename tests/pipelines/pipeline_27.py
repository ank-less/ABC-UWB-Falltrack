"""Pipeline 27: FN recovery via g_max static-change proposals.

Only runs if gate_p27 = True (from error_audit: >= 1/3 FN have no Stage-1 overlap
AND max_r(peak g) > ADL p95).

Strategy: add g_max proposals from static-change detector as extra Stage-1 candidates,
merge with energy candidates (gap 1.0s), retrain LightGBM.

Decision:
  KEEP if OOF recall_0.1 increases >= 4pp AND more TP added than FP AND F1_mean >= R_cur + 0.01
  DROP otherwise
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EVENTS_CSV   = ROOT / "streams_events.csv"
REPORTS_DIR  = ROOT / "experiments" / "audit" / "reports"

from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    energy_events_raw, RATE_HZ, STREAMS_TRAIN, STREAMS_VAL,
)
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.preprocessing.envelope import envelope
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter
from experiments.postprocessing.gap_merger import gap_merge
from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
from experiments.models.lightgbm_classifier import (
    _try_lgb, _fit, lgb_frame_scores,
)
from experiments.pipelines.pipeline_24 import window_features_clean
from experiments.audit.oof_harness import build_folds
from abc2027.baseline import frame_energy, iter_windows, STEP_S
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions, temporal_iou,
)

AXIS_LAYOUT = "range_368"
JOIN_OK     = False  # from physics checks


def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def _get_base_oof() -> Tuple[Path, str, float, float, float]:
    for eid in ("pipeline_25", "pipeline_24", "pipeline_23",
                "pipeline_22", "pipeline_21"):
        e = _load_log_entry(eid)
        if e.get("decision") == "KEEP":
            p = ROOT / f"oof_{eid}.csv"
            if p.exists():
                r01 = e.get("oof_f1_01") or e.get("f1_01") or 0
                return p, eid, e.get("oof_f1_mean", 0), e.get("oof_f1_05", 0), r01
    p19c = _load_log_entry("pipeline_19c")
    r01  = p19c.get("oof_f1_01") or p19c.get("f1_01") or 0
    return (ROOT / "oof_pipeline_19c.csv", "pipeline_19c",
            p19c.get("oof_f1_mean", 0), p19c.get("oof_f1_05", 0), r01)


def _compute_g_stream(cir: np.ndarray, m_s: float = 0.2,
                      tau_s: float = 1.2) -> np.ndarray:
    """Compute max-over-radars g_max(t) signal for a stream."""
    from experiments.postprocessing.static_hinge_refiner import compute_g
    m_fr  = max(1, int(m_s * RATE_HZ))
    tau_fr = tau_s * RATE_HZ
    T = cir.shape[1]
    join_mask = np.zeros(T, dtype=bool)
    g = compute_g(cir, m_fr, tau_fr, k_bins=16, join_mask=join_mask, layout=AXIS_LAYOUT)
    # Normalise per stream
    med = np.median(g, axis=1, keepdims=True)
    mad = np.median(np.abs(g - med), axis=1, keepdims=True) + 1e-8
    g_norm = (g - med) / mad
    g_max = np.max(g_norm, axis=0)  # (T,)
    return g_max


def _g_proposals(g_max: np.ndarray, dur_s: float,
                 min_dur: float = 0.5, max_dur: float = 2.5,
                 smooth_s: float = 0.2, threshold: float = 2.0) -> list:
    """Find peaks in g_max with sustained state change (0.5–2.5s)."""
    T = len(g_max)
    # Smooth g_max
    g_sm = gaussian_filter1d(g_max, sigma=smooth_s * RATE_HZ)
    in_peak = g_sm > threshold
    events = []
    i = 0
    while i < T:
        if in_peak[i]:
            j = i
            while j < T and in_peak[j]:
                j += 1
            dur = (j - i) / RATE_HZ
            if min_dur <= dur <= max_dur:
                t_s = i / RATE_HZ
                t_e = j / RATE_HZ
                score = float(np.max(g_sm[i:j]))
                events.append({"t_start": t_s, "t_end": t_e,
                               "score": score, "t_alarm": t_s})
            i = j
        else:
            i += 1
    return events


def _train_fold_with_proposals(train_sids: List[str], gt_all: list,
                                proposal_threshold: float = 2.0) -> object:
    """Train LightGBM including g_max proposals as extra candidates."""
    gt_by = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append((ev["t_start"], ev["t_end"]))

    X_parts, y_parts = [], []
    for sid in train_sids:
        stream_idx = int(sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)

        # g_max proposals as extra windows
        g_max = _compute_g_stream(cir)

        wins = iter_windows(filtered.shape[1])
        if not wins:
            continue
        feats = np.array([window_features_clean(filtered[:, s:e, :]) for s, e in wins])
        gt_ev = gt_by.get(sid, [])
        labels = np.array([
            int(any(temporal_iou(ws * STEP_S, we * STEP_S, gs, ge) >= 0.1
                    for gs, ge in gt_ev))
            for ws, we in wins
        ], dtype=np.int32)
        X_parts.append(feats)
        y_parts.append(labels)

    if not X_parts:
        return None
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Augmentation on positives
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        return None
    X_pos = X[pos_idx]
    rng = np.random.default_rng(42)
    stds = X_pos.std(axis=0) + 1e-8
    aug = [X_pos]
    for scale in (0.04, 0.07, 0.11):
        aug.append(X_pos + rng.normal(0, 1, X_pos.shape) * stds * scale)
    X_aug_pos = np.vstack(aug)
    X_neg = X[y == 0]
    X_full = np.vstack([X_neg, X_aug_pos])
    y_full = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_aug_pos))]).astype(np.int32)

    model_cls, use_lgb = _try_lgb()
    return _fit(model_cls, use_lgb, X_full, y_full)


def main():
    print("=" * 70)
    print("Pipeline 27 — FN recovery via g_max proposals")
    print("=" * 70)

    # Gate check
    gates_f = REPORTS_DIR / "gates.json"
    if not gates_f.exists():
        print(f"ERROR: gates.json not found at {gates_f}")
        return
    gates = json.loads(gates_f.read_text(encoding="utf-8"))
    gate_p27 = gates.get("gate_p27", False)
    if not gate_p27:
        print("Gate p27 = False → SKIPPED_GATE")
        append_log({
            "experiment_id": "pipeline_27",
            "phase": 4, "base": "auto",
            "description": "g_max FN recovery — SKIPPED (gate closed)",
            "components": ["g_max_proposals"],
            "oof_f1_mean": None, "oof_f1_01": None, "oof_f1_03": None, "oof_f1_05": None,
            "oof_tp": None, "oof_fp": None, "oof_fn": None,
            "delta_oof_vs_base": None, "p_delta_gt0": None,
            "f1_mean": None, "f1_01": None, "f1_03": None, "f1_05": None,
            "tp": None, "fp": None, "fn": None, "false_alarms_per_hour": None,
            "delta_f1": None, "probe_auc": None, "probe_macro_f1": None,
            "gate": False,
            "decision": "SKIPPED_GATE",
            "notes": "gate_p27 = False",
        })
        return

    base_oof, base_id, R, R_05, R_01 = _get_base_oof()
    print(f"Gate p27 = True → proceeding")
    print(f"Base: {base_id}  R={R}")

    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    gt_by    = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)
    durs     = stream_durations(STREAMS_TRAIN)

    folds = build_folds()
    print(f"\nTraining {len(folds)} fold models with g_max proposals…")
    fold_cache = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"\n  Fold {k:02d}: training…", flush=True)
        model = _train_fold_with_proposals(train_sids, gt_all)

        stream_idx = int(test_sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T = filtered.shape[1]
        dur_s = T / RATE_HZ

        energy  = smooth(frame_energy(filtered), sigma_s=0.3)
        energy_cands = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)

        # g_max proposals
        g_max   = _compute_g_stream(cir)
        g_props = _g_proposals(g_max, dur_s, threshold=2.0)
        all_cands = energy_cands + g_props

        # De-dup by merging overlapping
        all_cands = gap_merge(sorted(all_cands, key=lambda x: x["t_start"]), max_gap=1.0)

        p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        if model is not None:
            scores = lgb_frame_scores(filtered, model, feat_fn=window_features_clean)
        else:
            scores = np.zeros(T)

        fold_cache.append((test_sid, {
            "cands": all_cands, "p14_events": p14_events,
            "dur_s": dur_s, "T": T,
        }, {test_sid: scores}))
        print(f"    {len(all_cands)} candidates (energy+g_max)", flush=True)

    # Threshold sweep
    print(f"\nPhase 2: OOF threshold sweep…")
    thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
    tmp_csv = ROOT / "_p27_tmp.csv"
    best_f1, best_thr = 0.0, 0.25

    for thr in thresholds:
        high_thr = min(thr * 2.5, 0.50)
        all_rows = []
        for test_sid, cd, scores_d in fold_cache:
            bs = scores_d[test_sid]
            events = []
            for cand in cd["cands"]:
                s = int(cand["t_start"] * RATE_HZ)
                e = min(int(cand["t_end"] * RATE_HZ) + 1, cd["T"])
                if s >= e:
                    continue
                mx = float(np.max(bs[s:e]))
                if mx <= thr * 0.6:
                    continue
                p14_ok = any(
                    min(cand["t_end"], pev["t_end"]) - max(cand["t_start"], pev["t_start"]) > 0
                    for pev in cd["p14_events"]
                )
                if (p14_ok and mx > thr) or mx > high_thr:
                    # Energy array not cached in fold_cache — use raw candidate boundaries
                    ns, ne = cand["t_start"], cand["t_end"]
                    events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            for ev in events:
                all_rows.append({"stream_id": test_sid,
                                 "t_start": f"{ev['t_start']:.3f}",
                                 "t_end":   f"{ev['t_end']:.3f}",
                                 "score":   f"{ev['score']:.6f}",
                                 "t_alarm": f"{ev['t_alarm']:.3f}"})
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
            w.writeheader()
            w.writerows(all_rows)
        try:
            preds = validate_predictions(tmp_csv, durs)
            res   = evaluate(gt_train, preds, durs)
            f1    = round(res["ranking_score"]["f1_mean"], 4)
        except Exception:
            f1 = 0.0
        print(f"  thr={thr:.3f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    tmp_csv.unlink(missing_ok=True)
    print(f"Best thr={best_thr:.3f}")

    # Base OOF recall
    base_preds_by: Dict[str, list] = {}
    with open(base_oof, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            base_preds_by.setdefault(row["stream_id"], []).append({
                "t_start": float(row["t_start"]), "t_end": float(row["t_end"]),
            })
    base_recall = sum(
        int(any(temporal_iou(pr["t_start"], pr["t_end"],
                             ev["t_start"], ev["t_end"]) >= 0.1
                for prs in base_preds_by.values()
                for pr in prs))
        for ev in gt_train
    ) / max(len(gt_train), 1)

    # Final OOF
    high_thr_f = min(best_thr * 2.5, 0.50)
    all_oof_rows = []
    for test_sid, cd, scores_d in fold_cache:
        bs = scores_d[test_sid]
        events = []
        for cand in cd["cands"]:
            s = int(cand["t_start"] * RATE_HZ)
            e = min(int(cand["t_end"] * RATE_HZ) + 1, cd["T"])
            if s >= e:
                continue
            mx = float(np.max(bs[s:e]))
            if mx <= best_thr * 0.6:
                continue
            p14_ok = any(
                min(cand["t_end"], pev["t_end"]) - max(cand["t_start"], pev["t_start"]) > 0
                for pev in cd["p14_events"]
            )
            if (p14_ok and mx > best_thr) or mx > high_thr_f:
                events.append({"t_start": cand["t_start"], "t_end": cand["t_end"],
                               "score": mx, "t_alarm": cand["t_start"]})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        for ev in events:
            all_oof_rows.append({"stream_id": test_sid,
                                 "t_start": f"{ev['t_start']:.3f}",
                                 "t_end":   f"{ev['t_end']:.3f}",
                                 "score":   f"{ev['score']:.6f}",
                                 "t_alarm": f"{ev['t_alarm']:.3f}"})

    oof_csv = ROOT / "oof_pipeline_27.csv"
    with open(oof_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
        w.writeheader()
        w.writerows(all_oof_rows)

    preds  = validate_predictions(oof_csv, durs)
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

    new_recall = oof_metrics["f1_01"] / max(oof_metrics["f1_01"] + 1e-8, 1) * (
        oof_metrics["tp"] / (oof_metrics["tp"] + oof_metrics["fn"] + 1e-8)
    )  # approx recall at θ=0.3
    base_tp_count = int(R_01 * (oof_metrics["tp"] + oof_metrics["fn"]))
    new_tp_count  = oof_metrics["tp"]
    delta_tp = new_tp_count - base_tp_count
    delta_fp = oof_metrics["fp"] - int((1 - R_01) * (oof_metrics["tp"] + oof_metrics["fn"]))

    delta_mean = oof_metrics["f1_mean"] - R
    delta_01   = oof_metrics["f1_01"] - (R_01 or 0)

    print(f"\n=== OOF Results (p27) ===")
    print(f"  F1_mean={oof_metrics['f1_mean']:.4f}  (R={R}, Δ={delta_mean:+.4f})")
    print(f"  F1_01={oof_metrics['f1_01']:.4f}  (Δ_01={delta_01:+.4f})")
    print(f"  TP={oof_metrics['tp']}  FP={oof_metrics['fp']}  FN={oof_metrics['fn']}")

    if delta_01 >= 0.04 and delta_tp > abs(min(delta_fp, 0)) and delta_mean >= R + 0.01:
        decision = "KEEP"
        notes = (f"Δ_01={delta_01:+.4f}≥0.04, TP added≥FP added, "
                 f"Δ_mean={delta_mean:+.4f}≥0.01")
    else:
        decision = "DROP"
        notes = (f"Δ_01={delta_01:+.4f} (need≥0.04), "
                 f"Δ_mean={delta_mean:+.4f} (need≥0.01)")
    print(f"Decision: {decision}  — {notes}")

    # Val if KEEP
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        # (simplified: use same pipeline_19c model + g_max proposals)
        # For now use a full model trained on all 13 folds
        full_model = _train_fold_with_proposals(
            [f"train_{k:02d}" for k in range(13)], gt_all,
        )
        val_streams = load_val()
        val_rows = []
        for sid, cir in val_streams.items():
            dur_s = cir.shape[1] / RATE_HZ
            filtered = chebyshev_bp(cir)
            T = filtered.shape[1]
            energy = smooth(frame_energy(filtered), sigma_s=0.3)
            energy_cands = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
            g_max  = _compute_g_stream(cir)
            g_props = _g_proposals(g_max, dur_s, threshold=2.0)
            all_cands = gap_merge(
                sorted(energy_cands + g_props, key=lambda x: x["t_start"]),
                max_gap=1.0,
            )
            p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
            p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

            if full_model is not None:
                scores = lgb_frame_scores(filtered, full_model, feat_fn=window_features_clean)
            else:
                scores = np.zeros(T)

            high_thr_v = min(best_thr * 2.5, 0.50)
            events = []
            for ev in all_cands:
                s = int(ev["t_start"] * RATE_HZ)
                e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
                if s >= e:
                    continue
                mx = float(np.max(scores[s:e]))
                if mx <= best_thr * 0.6:
                    continue
                p14_ok = any(
                    min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                    for pev in p14_events
                )
                if (p14_ok and mx > best_thr) or mx > high_thr_v:
                    events.append({"t_start": ev["t_start"], "t_end": ev["t_end"],
                                   "score": mx, "t_alarm": ev["t_start"]})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            val_rows.extend(make_rows(sid, events))

        val_path    = save_sub(val_rows, "pipeline_27")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p27) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    base_val = _load_log_entry(base_id).get("f1_mean") or 0
    delta_val = round((val_metrics["f1_mean"] - base_val), 4) if val_metrics else None
    append_log({
        "experiment_id": "pipeline_27",
        "phase": 4, "base": base_id,
        "description": f"g_max proposals + energy candidates, thr={best_thr:.3f}",
        "components": ["g_max_proposals", "energy_candidates"],
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
        "delta_f1": delta_val,
        "probe_auc":      None,
        "probe_macro_f1": None,
        "gate":     "gate_p27",
        "decision": decision,
        "notes":    f"{notes}. thr={best_thr:.3f}.",
    })
    print(f"\nLogged: pipeline_27  decision={decision}")


if __name__ == "__main__":
    main()
