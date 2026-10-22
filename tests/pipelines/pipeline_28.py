"""Pipeline 28: Stage-1 max-pooling across radars (root cause fix).

ROOT CAUSE: Stage-1 frame_energy = mean(sum(cir²,bins), radars).
When a fall is visible on only 1-2 radars, mean dilutes the signal
below the pct=97 threshold → missed entirely by Stage-1 (98.8% FN).

FIX: Replace mean with max across radars.
Falls on 1 radar contribute at full strength; threshold self-adjusts
via percentile.

Base: pipeline_24 (OOF F1_mean=0.5559, F1_05=0.3920)
KEEP if OOF F1_mean >= R_cur + 0.005
DROP if OOF F1_mean < R_cur
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

EVENTS_CSV  = ROOT / "streams_events.csv"
COMP_CSV    = ROOT / "streams_composition.csv"
REPORTS_DIR = ROOT / "experiments" / "audit" / "reports"

from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    energy_events_raw, RATE_HZ, STREAMS_TRAIN, STREAMS_VAL,
)
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter
from experiments.postprocessing.gap_merger import gap_merge
from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
from experiments.models.lightgbm_classifier import (
    window_features_rich, _try_lgb, _fit, lgb_frame_scores,
)
from experiments.audit.oof_harness import build_folds
from abc2027.baseline import frame_energy, iter_windows, runs_to_events, STEP_S
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions, temporal_iou,
)


# ── P28 core: max-pooling Stage-1 energy ─────────────────────────────────────

def frame_energy_max(cir: np.ndarray) -> np.ndarray:
    """Per-frame energy using MAX across radars (not mean).
    Falls visible on 1 radar no longer diluted by inactive radars.
    Shape: (T,)
    """
    per_radar = np.sum(cir.astype(np.float64) ** 2, axis=2)  # (3, T)
    return np.max(per_radar, axis=0)                          # (T,)


# ── 21 clean features (same as p24) ──────────────────────────────────────────

LEAKY_INDICES = list(range(21, 27))
CLEAN_INDICES = [i for i in range(27) if i not in LEAKY_INDICES]


def window_features_clean(block: np.ndarray) -> np.ndarray:
    full = window_features_rich(block)
    return full[CLEAN_INDICES]


# ── Composition helpers ───────────────────────────────────────────────────────

def load_composition() -> List[dict]:
    with open(COMP_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── Hard negative mining (same as p24) ───────────────────────────────────────

def collect_hard_negatives(
    train_sids: List[str], gt_by: dict,
    feat_fn, weight: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_hard, w_hard = [], []
    for sid in train_sids:
        stream_idx = int(sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T = filtered.shape[1]
        wins = iter_windows(T)
        if not wins:
            continue
        gt_ev = [(ev["t_start"], ev["t_end"]) for ev in gt_by.get(sid, [])]
        for ws, we in wins:
            t_s, t_e = ws * STEP_S, we * STEP_S
            if any(temporal_iou(t_s, t_e, gs, ge) > 0 for gs, ge in gt_ev):
                continue
            is_pre  = any(t_e <= gs - 1.0 for gs, ge in gt_ev)
            is_post = any(t_s >= ge + 3.0  for gs, ge in gt_ev)
            if is_pre or is_post:
                feat = feat_fn(filtered[:, ws:we, :])
                X_hard.append(feat)
                w_hard.append(weight)
    if not X_hard:
        return np.empty((0, len(CLEAN_INDICES))), np.empty(0), np.empty(0)
    return np.array(X_hard), np.zeros(len(X_hard), dtype=np.int32), np.array(w_hard)


# ── Fold training (same as p24 clean) ────────────────────────────────────────

def _train_fold_clean(
    train_sids: List[str], gt_all: list,
    feat_fn, hard_neg_weight: float = 2.0,
) -> object:
    gt_by = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)

    X_parts, y_parts, w_parts = [], [], []
    for sid in train_sids:
        stream_idx = int(sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        wins = iter_windows(filtered.shape[1])
        if not wins:
            continue
        feats = np.array([feat_fn(filtered[:, s:e, :]) for s, e in wins])
        gt_ev = [(ev["t_start"], ev["t_end"]) for ev in gt_by.get(sid, [])]
        labels = np.array([
            int(any(temporal_iou(ws * STEP_S, we * STEP_S, gs, ge) >= 0.1
                    for gs, ge in gt_ev))
            for ws, we in wins
        ], dtype=np.int32)
        X_parts.append(feats)
        y_parts.append(labels)
        w_parts.append(np.ones(len(labels)))

    if not X_parts:
        return None
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)

    # Augmentation on positives
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        return None
    X_pos = X[pos_idx]
    rng   = np.random.default_rng(42)
    stds  = X_pos.std(axis=0) + 1e-8
    aug   = [X_pos]
    for scale in (0.04, 0.07, 0.11):
        aug.append(X_pos + rng.normal(0, 1, X_pos.shape) * stds * scale)
    for lo, hi in ((0.78, 1.22), (0.85, 1.15)):
        s = rng.uniform(lo, hi, (len(X_pos), 1))
        aug.append(X_pos * s)
    n    = len(X_pos)
    half = X_pos.shape[1] // 2
    perm = rng.permutation(n)
    aug.append(np.concatenate([X_pos[:, :half], X_pos[perm, half:]], axis=1))

    X_aug  = np.vstack(aug)
    X_neg  = X[y == 0]
    w_neg  = w[y == 0]
    X_full = np.vstack([X_neg, X_aug])
    y_full = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_aug))]).astype(np.int32)
    w_full = np.concatenate([w_neg, np.ones(len(X_aug))])

    # Hard negatives (p24b)
    X_hard, y_hard, w_hard = collect_hard_negatives(train_sids, gt_by, feat_fn, hard_neg_weight)
    if len(X_hard) > 0:
        X_full = np.vstack([X_full, X_hard])
        y_full = np.concatenate([y_full, y_hard])
        w_full = np.concatenate([w_full, w_hard])

    model_cls, use_lgb = _try_lgb()
    if use_lgb:
        m = model_cls(n_estimators=500, num_leaves=63, min_child_samples=5,
                      n_jobs=-1, verbose=-1)
        m.fit(X_full, y_full, sample_weight=w_full)
    else:
        m = model_cls(n_estimators=400, n_jobs=-1, random_state=42)
        m.fit(X_full, y_full, sample_weight=w_full)
    return m


def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Pipeline 28 — Stage-1 max-pooling (root cause fix)")
    print("=" * 70)

    # Base: pipeline_24
    p24 = _load_log_entry("pipeline_24")
    R      = p24.get("oof_f1_mean", 0.5559)
    R_05   = p24.get("oof_f1_05",   0.3920)
    print(f"Base: pipeline_24  OOF F1_mean={R}  OOF F1_05={R_05}")

    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    folds = build_folds()
    print(f"\nPhase 1: Training {len(folds)} fold models (p28: max-pool Stage-1)…")
    fold_cache = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"\n  Fold {k:02d}: training on {len(train_sids)} streams…", flush=True)
        model = _train_fold_clean(train_sids, gt_all, window_features_clean, hard_neg_weight=2.0)

        stream_idx = int(test_sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T     = filtered.shape[1]
        dur_s = T / RATE_HZ

        # P28 KEY CHANGE: max pooling for Stage-1 candidates
        energy_max  = smooth(frame_energy_max(filtered), sigma_s=0.3)
        energy_mean = smooth(frame_energy(filtered),     sigma_s=0.3)

        cands      = energy_events_raw(energy_max,  pct=97.0, min_dur=0.4, merge_gap=0.8)
        p14_cands  = energy_events_raw(energy_mean, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy_mean, dur_s, k=1.2)

        # LGB frame scores
        if model is not None:
            wins  = iter_windows(T)
            feats = np.array([window_features_clean(filtered[:, s:e, :]) for s, e in wins]) if wins else None
            if feats is not None:
                raw = model.predict_proba(feats)[:, 1]
                raw = gaussian_filter1d(raw, sigma=2.0)
                xs  = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
                fs  = np.interp(np.arange(T, dtype=np.float64), xs, raw)
                fs  = gaussian_filter1d(fs, sigma=5.0)
            else:
                fs = np.zeros(T)
        else:
            fs = np.zeros(T)

        fold_cache.append((
            test_sid,
            {"cands": cands, "p14_events": p14_events,
             "energy_max": energy_max, "dur_s": dur_s, "T": T},
            {test_sid: fs},
        ))
        print(f"    Stage-1 candidates: {len(cands)}", flush=True)

    # Threshold sweep (OOF)
    print(f"\nPhase 2: OOF threshold sweep…")
    thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
    tmp_csv = ROOT / "_p28_tmp.csv"
    best_f1, best_thr = 0.0, 0.25

    for thr in thresholds:
        high_thr  = min(thr * 2.5, 0.50)
        all_rows  = []
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
                    events.append({"t_start": cand["t_start"], "t_end": cand["t_end"],
                                   "score": mx, "t_alarm": cand["t_start"]})
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
        print(f"  thr={thr:.3f}  OOF F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    tmp_csv.unlink(missing_ok=True)
    print(f"Best OOF thr={best_thr:.3f}  F1={best_f1:.4f}")

    # Final OOF with best threshold (with boundary refinement)
    all_oof_rows = []
    high_thr = min(best_thr * 2.5, 0.50)
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
            if (p14_ok and mx > best_thr) or mx > high_thr:
                ns, ne = refine_boundary(cd["energy_max"], cand["t_start"],
                                         cand["t_end"], cd["dur_s"])
                events.append({"t_start": ns, "t_end": ne,
                               "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        for ev in events:
            all_oof_rows.append({"stream_id": test_sid,
                                 "t_start": f"{ev['t_start']:.3f}",
                                 "t_end":   f"{ev['t_end']:.3f}",
                                 "score":   f"{ev['score']:.6f}",
                                 "t_alarm": f"{ev['t_alarm']:.3f}"})

    oof_csv = ROOT / "oof_pipeline_28.csv"
    with open(oof_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
        w.writeheader()
        w.writerows(all_oof_rows)

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

    print(f"\n=== OOF Results (p28) ===")
    print(f"  F1_mean={oof['f1_mean']:.4f}  (R={R}, Δ={delta_mean:+.4f})")
    print(f"  F1_01  ={oof['f1_01']:.4f}")
    print(f"  F1_03  ={oof['f1_03']:.4f}")
    print(f"  F1_05  ={oof['f1_05']:.4f}  Δ_05={delta_05:+.4f}")
    print(f"  TP={oof['tp']}  FP={oof['fp']}  FN={oof['fn']}")

    # Decision
    if delta_mean >= 0.005:
        decision = "KEEP"
        notes    = f"Δ_mean={delta_mean:+.4f}≥0.005 — max-pool Stage-1 effective"
    elif delta_mean >= 0.0:
        decision = "KEEP"
        notes    = f"Δ_mean={delta_mean:+.4f}≥0 — marginal but neutral, keep"
    else:
        decision = "DROP"
        notes    = f"Δ_mean={delta_mean:+.4f}<0 — max-pool Stage-1 did not improve"
    print(f"Decision: {decision}  — {notes}")

    # Val (exactly 1 run)
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        val_durs    = stream_durations(STREAMS_VAL)
        val_streams = load_val()

        print("Training full model on all 13 train streams…")
        full_model = _train_fold_clean(
            [f"train_{k:02d}" for k in range(13)],
            gt_all, window_features_clean, hard_neg_weight=2.0,
        )
        val_rows = []
        for sid, cir in val_streams.items():
            dur_s    = cir.shape[1] / RATE_HZ
            filtered = chebyshev_bp(cir)
            T        = filtered.shape[1]

            # P28: max-pool Stage-1 on val too
            energy_max  = smooth(frame_energy_max(filtered), sigma_s=0.3)
            energy_mean = smooth(frame_energy(filtered),     sigma_s=0.3)
            cands      = energy_events_raw(energy_max,  pct=97.0, min_dur=0.4, merge_gap=0.8)
            p14_cands  = energy_events_raw(energy_mean, pct=98.0, min_dur=0.6, merge_gap=0.8)
            p14_events = gaussian_boundary(p14_cands, energy_mean, dur_s, k=1.2)

            wins  = iter_windows(T)
            feats = np.array([window_features_clean(filtered[:, s:e, :]) for s, e in wins]) if wins else None
            if feats is None or full_model is None:
                continue
            raw = full_model.predict_proba(feats)[:, 1]
            raw = gaussian_filter1d(raw, sigma=2.0)
            xs  = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
            fs  = np.interp(np.arange(T, dtype=np.float64), xs, raw)
            fs  = gaussian_filter1d(fs, sigma=5.0)

            high_thr_v = min(best_thr * 2.5, 0.50)
            events = []
            for ev in cands:
                s = int(ev["t_start"] * RATE_HZ)
                e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
                if s >= e:
                    continue
                mx = float(np.max(fs[s:e]))
                if mx <= best_thr * 0.6:
                    continue
                p14_ok = any(
                    min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                    for pev in p14_events
                )
                if (p14_ok and mx > best_thr) or mx > high_thr_v:
                    ns, ne = refine_boundary(energy_max, ev["t_start"], ev["t_end"], dur_s)
                    events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            val_rows.extend(make_rows(sid, events))

        val_path    = save_sub(val_rows, "pipeline_28")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p28) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    # Log
    p24_val  = p24.get("f1_mean") or 0.0
    delta_val = round((val_metrics["f1_mean"] - p24_val) if val_metrics else 0.0, 4)
    entry = {
        "experiment_id": "pipeline_28",
        "phase": 4,
        "base": "pipeline_24",
        "description": f"p24 + Stage-1 max-pooling across radars (pct=97, thr={best_thr:.3f})",
        "components": ["stage1_max_pooling", "feature_pruning_21f", "hard_negative_mining"],
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
    print(f"\nLogged: pipeline_28  decision={decision}")
    return decision, oof, val_metrics


if __name__ == "__main__":
    main()
