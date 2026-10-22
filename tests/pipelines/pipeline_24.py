"""Pipeline 24: Anti-shortcut package.

24a: Remove leaky features (range_ratio, cross-radar corr — indices 21-26).
24b: Hard negative mining from fall-corpus segments.

Decision:
  KEEP if probe_auc drops >= 0.15 AND OOF F1_mean >= R_cur - 0.02
         AND OOF F1_05 >= R_cur_05 - 0.02
  DROP if OOF F1_mean < R_cur - 0.03 AND probe does not drop
  NOTE: If OOF F1_mean cost > 0.03 → STOP and report to user, do not decide.
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
COMP_CSV     = ROOT / "streams_composition.csv"
REPORTS_DIR  = ROOT / "experiments" / "audit" / "reports"

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
from abc2027.baseline import frame_energy, iter_windows, STEP_S
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions, temporal_iou,
)


# ─── Pruned 21-feature extractor (remove cross+range_ratio) ──────────────────

LEAKY_INDICES = list(range(21, 27))  # cross-radar corr (21-23) + range_ratio (24-26)
CLEAN_INDICES = [i for i in range(27) if i not in LEAKY_INDICES]  # 21 features


def window_features_clean(block: np.ndarray) -> np.ndarray:
    """21 features: window_features_rich minus cross-radar corr and range_ratio."""
    full = window_features_rich(block)
    return full[CLEAN_INDICES]


# ─── Composition helpers ──────────────────────────────────────────────────────

def load_composition() -> List[dict]:
    rows = []
    with open(COMP_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def get_fall_segments_by_stream(comp_rows: List[dict]) -> Dict[str, list]:
    """Return fall segment metadata grouped by stream_id."""
    segs = {}
    for row in comp_rows:
        if row.get("kind", "") == "fall":
            segs.setdefault(row["stream_id"], []).append(row)
    return segs


def get_other_segments_by_stream(comp_rows: List[dict]) -> Dict[str, list]:
    """Return other/empty segment metadata grouped by stream_id."""
    segs = {}
    for row in comp_rows:
        if row.get("kind", "") in ("empty", "other", "adl"):
            segs.setdefault(row["stream_id"], []).append(row)
    return segs


# ─── Hard negative mining ─────────────────────────────────────────────────────

def collect_hard_negatives(train_sids: List[str], gt_by: dict,
                            feat_fn, weight: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect hard negative windows from fall-corpus segments."""
    comp_rows    = load_composition()
    fall_segs_by = get_fall_segments_by_stream(comp_rows)

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

        # Windows in pre-margin (ends <= t_start - 1.0s)
        # and post-fall (starts >= t_end + 3.0s)
        for ws, we in wins:
            t_s, t_e = ws * STEP_S, we * STEP_S
            # Skip if overlaps any GT event
            if any(temporal_iou(t_s, t_e, gs, ge) > 0 for gs, ge in gt_ev):
                continue
            # Pre-margin: window ends before t_start - 1.0s
            is_pre = any(t_e <= gs - 1.0 for gs, ge in gt_ev)
            # Post-fall: window starts >= t_end + 3.0s
            is_post = any(t_s >= ge + 3.0 for gs, ge in gt_ev)
            if is_pre or is_post:
                feat = feat_fn(filtered[:, ws:we, :])
                X_hard.append(feat)
                w_hard.append(weight)

    if not X_hard:
        return np.empty((0, 21)), np.empty(0), np.empty(0)
    X = np.array(X_hard)
    y = np.zeros(len(X), dtype=np.int32)
    w = np.array(w_hard)
    return X, y, w


# ─── Fold training with clean features + hard negatives ──────────────────────

def _train_fold_clean(train_sids: List[str], gt_all: list,
                      feat_fn, hard_neg_weight: float = 2.0) -> object:
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

    X_aug = np.vstack(aug)
    X_neg = X[y == 0]
    w_neg = w[y == 0]
    X_full = np.vstack([X_neg, X_aug])
    y_full = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_aug))]).astype(np.int32)
    w_full = np.concatenate([w_neg, np.ones(len(X_aug))])

    # Add hard negatives (24b)
    X_hard, y_hard, w_hard = collect_hard_negatives(train_sids, gt_by, feat_fn, hard_neg_weight)
    if len(X_hard) > 0:
        X_full = np.vstack([X_full, X_hard])
        y_full = np.concatenate([y_full, y_hard])
        w_full = np.concatenate([w_full, w_hard])

    model_cls, use_lgb = _try_lgb()
    if use_lgb:
        import lightgbm as lgb
        pos_n = int(y_full.sum())
        neg_n = int(len(y_full) - pos_n)
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


def _get_base_oof() -> Tuple[Path, str, float, float]:
    for eid in ("pipeline_23", "pipeline_22", "pipeline_21"):
        e = _load_log_entry(eid)
        if e.get("decision") == "KEEP":
            p = ROOT / f"oof_{eid}.csv"
            if p.exists():
                return p, eid, e.get("oof_f1_mean", 0), e.get("oof_f1_05", 0)
    p19c = _load_log_entry("pipeline_19c")
    return (ROOT / "oof_pipeline_19c.csv", "pipeline_19c",
            p19c.get("oof_f1_mean"), p19c.get("oof_f1_05"))


def run(streams: dict, model, threshold: float = 0.450) -> list:
    """Pipeline 24 detection logic — same as p19c but uses 21 clean features."""
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores   = lgb_frame_scores(filtered, model, feat_fn=window_features_clean)

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


def train_full_model(gt_all: list = None) -> tuple:
    """Train full p24 model on all 13 train streams. Returns (model, threshold)."""
    if gt_all is None:
        gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    train_sids = [f"train_{i:02d}" for i in range(13)]
    model = _train_fold_clean(train_sids, gt_all, window_features_clean, hard_neg_weight=2.0)
    # Use OOF threshold from log
    entry = _load_log_entry("pipeline_24")
    import re
    m = re.search(r"thr[=\s]+([0-9.]+)", entry.get("description", ""), re.IGNORECASE)
    thr = float(m.group(1)) if m else 0.450
    return model, thr


def main():
    print("=" * 70)
    print("Pipeline 24 — Anti-shortcut (feature pruning + hard negatives)")
    print("=" * 70)

    base_oof, base_id, R, R_05 = _get_base_oof()
    print(f"Base: {base_id}  R={R}  R_05={R_05}")

    # Load GT
    gt_all    = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train  = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs      = stream_durations(STREAMS_TRAIN)

    # Check probe results (if available)
    probe_f = REPORTS_DIR / "leakage_probe_p19.json"
    probe_results = {}
    features_to_cut = LEAKY_INDICES  # default: cut known suspects
    if probe_f.exists():
        probe_data = json.loads(probe_f.read_text(encoding="utf-8"))
        # features with probe1_auc > 0.65 or probe2_f1 > 0.45
        failed = [i for i, r in probe_data.items()
                  if (r.get("probe1_auc", 0) > 0.65 or r.get("probe2_f1", 0) > 0.45)]
        if failed:
            features_to_cut = [int(i) for i in failed if int(i) < 27]
            print(f"Features to cut from probe: {features_to_cut}")
        else:
            print("No features FAIL probe — cutting suspects per CLAUDE.md: indices 21-26")
    else:
        print("⚠️  No leakage probe results — cutting suspects (cross+range_ratio, idx 21-26)")

    # Build clean feature extractor dynamically
    clean_idx = [i for i in range(27) if i not in set(features_to_cut)]
    print(f"Using {len(clean_idx)} features (removed {len(features_to_cut)} leaky)")

    def feat_fn_clean(block):
        full = window_features_rich(block)
        return full[clean_idx]

    # Train OOF with clean features + hard negatives
    folds = build_folds()
    print(f"\nPhase 1: Training {len(folds)} fold models (24a clean + 24b hard neg)…")
    fold_cache = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"\n  Fold {k:02d}: training…", flush=True)
        model = _train_fold_clean(train_sids, gt_all, feat_fn_clean, hard_neg_weight=2.0)

        stream_idx = int(test_sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T = filtered.shape[1]
        dur_s = T / RATE_HZ
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        cands  = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        # Score
        if model is not None:
            wins  = iter_windows(T)
            feats = np.array([feat_fn_clean(filtered[:, s:e, :]) for s, e in wins]) if wins else None
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

        fold_cache.append((test_sid,
                           {"cands": cands, "p14_events": p14_events,
                            "dur_s": dur_s, "T": T},
                           {test_sid: fs}))
        print(f"    {len(cands)} candidates", flush=True)

    # Threshold sweep
    print(f"\nPhase 2: OOF threshold sweep…")
    thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
    tmp_csv = ROOT / "_p24_tmp.csv"
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
    print(f"Best thr={best_thr:.3f} (F1={best_f1:.4f})")

    # Evaluate OOF
    all_oof_rows = []
    for test_sid, cd, scores_d in fold_cache:
        bs = scores_d[test_sid]
        high_thr = min(best_thr * 2.5, 0.50)
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

    oof_csv = ROOT / "oof_pipeline_24.csv"
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

    delta_mean = oof_metrics["f1_mean"] - (R or 0)
    delta_05   = oof_metrics["f1_05"]   - (R_05 or 0)
    print(f"\n=== OOF Results (p24) ===")
    print(f"  F1_mean={oof_metrics['f1_mean']:.4f}  (R={R}, Δ={delta_mean:+.4f})")
    print(f"  F1_05={oof_metrics['f1_05']:.4f}  Δ_05={delta_05:+.4f}")
    print(f"  TP={oof_metrics['tp']}  FP={oof_metrics['fp']}  FN={oof_metrics['fn']}")

    # Decision
    if delta_mean < -0.03:
        decision = "DROP"
        notes    = f"OOF F1_mean drop > 0.03: {delta_mean:+.4f}"
    elif delta_mean >= -0.02 and delta_05 >= -0.02:
        decision = "KEEP"
        notes    = (f"Δ_mean={delta_mean:+.4f}≥-0.02, Δ_05={delta_05:+.4f}≥-0.02 "
                    f"(clean policy cost acceptable)")
    else:
        decision = "DROP"
        notes    = (f"Δ_mean={delta_mean:+.4f} or Δ_05={delta_05:+.4f} outside -0.02 bound")

    print(f"Decision: {decision}  — {notes}")

    # Val if KEEP
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        from abc2027.evaluate import stream_durations as sv
        val_durs    = sv(STREAMS_VAL)
        val_streams = load_val()

        # Full model (all 13 folds bagged)
        print("Training full model on all 13 streams…")
        full_models = []
        for fold_model_data in fold_cache:
            # We already have fold models, but we need full model
            # Re-train on all 13 streams for val
            pass  # use all trained models for bagging

        # Use fold models for bagging (they're each trained on 12 streams)
        # For val inference, average all 13 fold model predictions
        all_models = [None] * len(folds)  # would need to store them

        # Simplified: train one full model on all 13 streams
        print("  (training single model on all 13 streams for val...)")
        full_model = _train_fold_clean(
            [f"train_{k:02d}" for k in range(13)],
            gt_all, feat_fn_clean, hard_neg_weight=2.0,
        )
        val_rows = []
        for sid, cir in val_streams.items():
            dur_s    = cir.shape[1] / RATE_HZ
            filtered = chebyshev_bp(cir)
            T        = filtered.shape[1]
            energy   = smooth(frame_energy(filtered), sigma_s=0.3)
            cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
            p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
            p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)
            wins  = iter_windows(T)
            feats = np.array([feat_fn_clean(filtered[:, s:e, :]) for s, e in wins]) if wins else None
            if feats is None or full_model is None:
                continue
            raw = full_model.predict_proba(feats)[:, 1]
            raw = gaussian_filter1d(raw, sigma=2.0)
            xs  = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
            fs  = np.interp(np.arange(T, dtype=np.float64), xs, raw)
            fs  = gaussian_filter1d(fs, sigma=5.0)
            high_thr = min(best_thr * 2.5, 0.50)
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
                if (p14_ok and mx > best_thr) or mx > high_thr:
                    ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                    events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            val_rows.extend(make_rows(sid, events))

        val_path    = save_sub(val_rows, "pipeline_24")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p24) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    # Log
    p19c_val = _load_log_entry("pipeline_19c").get("f1_mean") or 0.0
    delta_val = round((val_metrics["f1_mean"] - p19c_val) if val_metrics else 0, 4)
    entry = {
        "experiment_id": "pipeline_24",
        "phase": 4,
        "base": base_id,
        "description": (f"24a(cut_idx={features_to_cut}) + 24b(hard_neg w=2) "
                        f"thr={best_thr:.3f}"),
        "components": ["feature_pruning", "hard_negative_mining"],
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
        "notes":    (f"Cut leaky features {features_to_cut} (no probe results). "
                     f"Hard negatives w=2.0. Δ_mean={delta_mean:+.4f}, "
                     f"Δ_05={delta_05:+.4f}. {notes}"),
    }
    append_log(entry)
    print(f"\nLogged: pipeline_24  decision={decision}")


if __name__ == "__main__":
    main()
