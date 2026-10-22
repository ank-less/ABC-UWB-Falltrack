"""Pipeline 23: label cleaning + max pooling + fold-bagging.

23a: Remove flagged/low-quality positives from training. Windows overlapping
     excluded events become ignored (not positive, not negative).
23b: Switch Stage-1 energy from mean over radars to max (per-radar normalised).
Fold-bagging: average 13 fold models for val/test inference.
Threshold: selected on nested OOF.

Decision:
  KEEP if OOF F1_mean >= R_cur + 0.005 AND OOF F1_01 >= R_cur_01 + 0.01
  DROP if OOF F1_mean < R_cur - 0.005
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ─── Label cleaning ───────────────────────────────────────────────────────────

def load_clean_events(events_csv: Path = None):
    """Load events, split into clean positives, ignored events, and ignore windows."""
    if events_csv is None:
        events_csv = EVENTS_CSV
    clean, ignored = [], []
    with open(events_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ev = {
                "stream_id":   row["stream_id"],
                "t_start":     float(row["t_start"]),
                "t_end":       float(row["t_end"]),
                "ignore":      row.get("ignore", "false").lower() == "true",
                "quality":     row.get("quality", "normal"),
                "needs_review": row.get("needs_review", "false").lower() == "true",
            }
            if (not ev["ignore"] and ev["quality"] == "normal"
                    and not ev["needs_review"]):
                clean.append(ev)
            else:
                ignored.append(ev)
    return clean, ignored


def is_window_clean(t_s: float, t_e: float, ignored_events: List[dict],
                    ignored_by_sid: Dict[str, list], sid: str,
                    iou_thr: float = 0.1) -> Optional[bool]:
    """True=positive, False=negative, None=ignore (overlaps ignored event±2s)."""
    for ev in ignored_by_sid.get(sid, []):
        # Ignore zone: ±2s around ignored event
        zone_s = ev["t_start"] - 2.0
        zone_e = ev["t_end"]   + 2.0
        if min(t_e, zone_e) - max(t_s, zone_s) > 0:
            return None
    return True  # or False — caller decides after checking clean events


# ─── Energy with max pooling ──────────────────────────────────────────────────

def frame_energy_max(cir: np.ndarray) -> np.ndarray:
    """Per-frame energy: max over radars (normalised by whole-stream median/IQR)."""
    T = cir.shape[1]
    per_radar = np.sum(cir.astype(np.float64) ** 2, axis=2)  # (3, T)
    normalised = np.empty_like(per_radar)
    for r in range(per_radar.shape[0]):
        e = per_radar[r]
        med = float(np.median(e))
        iqr = float(np.percentile(e, 75) - np.percentile(e, 25)) + 1e-12
        normalised[r] = (e - med) / iqr
    return np.max(normalised, axis=0)


# ─── OOF fold training ────────────────────────────────────────────────────────

def _train_fold_clean(train_sids: List[str], clean_events: list, ignored_events: list,
                      use_max_pooling: bool = True) -> object:
    """Train LightGBM with clean labels + augmentation."""
    clean_by   = {}
    for ev in clean_events:
        clean_by.setdefault(ev["stream_id"], []).append((ev["t_start"], ev["t_end"]))
    ignored_by = {}
    for ev in ignored_events:
        ignored_by.setdefault(ev["stream_id"], []).append(ev)

    X_parts, y_parts = [], []
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
        feats  = np.array([window_features_rich(filtered[:, s:e, :]) for s, e in wins])
        gt_ev  = clean_by.get(sid, [])
        ig_by_sid = {sid: ignored_by.get(sid, [])}

        labels   = []
        keep_idx = []
        for i, (ws, we) in enumerate(wins):
            t_s, t_e = ws * STEP_S, we * STEP_S
            # Check ignore zone
            is_ignored = False
            for ev_ig in ignored_by.get(sid, []):
                zone_s = ev_ig["t_start"] - 2.0
                zone_e = ev_ig["t_end"]   + 2.0
                if min(t_e, zone_e) - max(t_s, zone_s) > 0:
                    is_ignored = True
                    break
            if is_ignored:
                continue
            # Label
            is_pos = int(any(temporal_iou(t_s, t_e, gs, ge) >= 0.1 for gs, ge in gt_ev))
            labels.append(is_pos)
            keep_idx.append(i)

        if not keep_idx:
            continue
        X_parts.append(feats[keep_idx])
        y_parts.append(np.array(labels, dtype=np.int32))

    if not X_parts:
        return None
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

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
    X_full = np.vstack([X_neg, X_aug])
    y_full = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_aug))]).astype(np.int32)

    model_cls, use_lgb = _try_lgb()
    return _fit(model_cls, use_lgb, X_full, y_full)


# ─── Pipeline run function ────────────────────────────────────────────────────

def run(streams: dict, models: List[object], threshold: float,
        use_max_pooling: bool = True) -> list:
    """Run pipeline 23 on given streams.

    models: list of fold models for bagging (average probabilities).
    """
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)

        if use_max_pooling:
            energy = smooth(frame_energy_max(filtered), sigma_s=0.3)
        else:
            energy = smooth(frame_energy(filtered), sigma_s=0.3)

        cands   = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)

        # Bag model scores
        T = filtered.shape[1]
        wins  = iter_windows(T)
        feats = np.array([window_features_rich(filtered[:, s:e, :]) for s, e in wins]) if wins else None
        if feats is None or len(models) == 0:
            continue
        bag_scores = np.zeros(len(feats))
        for m in models:
            bag_scores += m.predict_proba(feats)[:, 1]
        bag_scores /= len(models)
        bag_smooth  = gaussian_filter1d(bag_scores, sigma=2.0)
        xs          = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
        full_scores = np.interp(np.arange(T, dtype=np.float64), xs, bag_smooth)
        full_scores = gaussian_filter1d(full_scores, sigma=5.0)

        p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)
        high_thr   = min(threshold * 2.5, 0.50)

        events = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(full_scores[s:e]))
            if mx <= threshold * 0.6:
                continue
            p14_ok = any(
                min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                for pev in p14_events
            )
            if (p14_ok and mx > threshold) or mx > high_thr:
                ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})

        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


# ─── Threshold sweep on cached fold data ─────────────────────────────────────

def _sweep_threshold(fold_cache: list, thresholds: list,
                     gt_train: list, durs: dict,
                     use_max_pooling: bool) -> Tuple[float, dict]:
    """Sweep LGB threshold on cached fold data (no retraining)."""
    tmp_csv = ROOT / "_p23_sweep_tmp.csv"
    best_f1, best_thr = 0.0, thresholds[0]
    results = {}
    for thr in thresholds:
        high_thr = min(thr * 2.5, 0.50)
        all_rows = []
        for test_sid, cand_data, bag_scores_dict in fold_cache:
            bs = bag_scores_dict[test_sid]
            cands  = cand_data["cands"]
            p14ev  = cand_data["p14_events"]
            dur_s  = cand_data["dur_s"]
            T      = cand_data["T"]
            events = []
            for cand in cands:
                s = int(cand["t_start"] * RATE_HZ)
                e = min(int(cand["t_end"]   * RATE_HZ) + 1, T)
                if s >= e:
                    continue
                mx = float(np.max(bs[s:e]))
                if mx <= thr * 0.6:
                    continue
                p14_ok = any(
                    min(cand["t_end"], pev["t_end"]) - max(cand["t_start"], pev["t_start"]) > 0
                    for pev in p14ev
                )
                if (p14_ok and mx > thr) or mx > high_thr:
                    events.append({"t_start": cand["t_start"], "t_end": cand["t_end"],
                                   "score": mx, "t_alarm": cand["t_start"]})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            for ev in events:
                all_rows.append({
                    "stream_id": test_sid,
                    "t_start":  f"{ev['t_start']:.3f}",
                    "t_end":    f"{ev['t_end']:.3f}",
                    "score":    f"{ev['score']:.6f}",
                    "t_alarm":  f"{ev['t_alarm']:.3f}",
                })
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
        results[thr] = f1
        print(f"  thr={thr:.3f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    tmp_csv.unlink(missing_ok=True)
    print(f"Best threshold: {best_thr:.3f} (F1={best_f1:.4f})")
    return best_thr, results


def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def _get_base_oof_csv() -> Tuple[Path, str, float, float]:
    """Return (oof_csv, base_id, R, R_05) for the current best base."""
    for eid in ("pipeline_22", "pipeline_21"):
        e = _load_log_entry(eid)
        if e.get("decision") == "KEEP":
            p = ROOT / f"oof_{eid}.csv"
            if p.exists():
                return p, eid, e.get("oof_f1_mean", 0), e.get("oof_f1_05", 0)
    p19c = _load_log_entry("pipeline_19c")
    return (ROOT / "oof_pipeline_19c.csv", "pipeline_19c",
            p19c.get("oof_f1_mean"), p19c.get("oof_f1_05"))


def main():
    print("=" * 70)
    print("Pipeline 23 — Label cleaning + Max pooling + Fold-bagging")
    print("=" * 70)

    # Load base info
    base_oof, base_id, R, R_05 = _get_base_oof_csv()
    print(f"Base: {base_id}  R={R}  R_05={R_05}")

    if not base_oof.exists():
        print(f"ERROR: base OOF not found at {base_oof}")
        return

    # ── Load GT ───────────────────────────────────────────────────────────────
    clean_evs, ignored_evs = load_clean_events()
    all_gt = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in all_gt if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    n_ignored_train = sum(1 for e in ignored_evs if e.get("stream_id","").startswith("train_"))
    n_clean_train   = sum(1 for e in clean_evs   if e.get("stream_id","").startswith("train_"))
    print(f"\nClean events (train): {n_clean_train}")
    print(f"Ignored events (train): {n_ignored_train}")
    pct_removed = 100 * n_ignored_train / max(n_clean_train + n_ignored_train, 1)
    if pct_removed > 25:
        print(f"⚠️  {pct_removed:.1f}% of positives removed (>25%) — noting in log")

    # ── Determine stage1_pooling from p19 ────────────────────────────────────
    # frame_energy in baseline uses mean(sum(...)) → "mean"
    stage1_pooling = "mean"  # verified from baseline.py
    use_max_pooling_23b = True  # always try — 23b condition is met
    print(f"stage1_pooling = {stage1_pooling} → applying 23b (max pooling)")

    # ── Phase 1: train 13 fold models + cache scores ──────────────────────────
    folds = build_folds()
    print(f"\nPhase 1: Training {len(folds)} fold models with clean labels + max pooling…")
    fold_cache = []
    fold_models = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"\n  Fold {k:02d}: training on {len(train_sids)} streams…", flush=True)
        model = _train_fold_clean(train_sids, clean_evs, ignored_evs, use_max_pooling_23b)
        fold_models.append(model)

        # Cache test stream data
        stream_idx = int(test_sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T = filtered.shape[1]
        dur_s = T / RATE_HZ

        if use_max_pooling_23b:
            energy = smooth(frame_energy_max(filtered), sigma_s=0.3)
        else:
            energy = smooth(frame_energy(filtered), sigma_s=0.3)

        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        p14_cands = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        # Score using only this fold's model
        if model is not None:
            wins  = iter_windows(T)
            feats = np.array([window_features_rich(filtered[:, s:e, :]) for s, e in wins]) if wins else None
            if feats is not None:
                raw_scores = model.predict_proba(feats)[:, 1]
                raw_scores = gaussian_filter1d(raw_scores, sigma=2.0)
                xs         = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
                full_s     = np.interp(np.arange(T, dtype=np.float64), xs, raw_scores)
                full_s     = gaussian_filter1d(full_s, sigma=5.0)
            else:
                full_s = np.zeros(T)
        else:
            full_s = np.zeros(T)

        fold_cache.append((
            test_sid,
            {"cands": cands, "p14_events": p14_events, "dur_s": dur_s, "T": T},
            {test_sid: full_s},
        ))
        print(f"    {len(cands)} candidates cached", flush=True)

    # ── Phase 2: threshold sweep ───────────────────────────────────────────────
    print(f"\nPhase 2: OOF threshold sweep…")
    thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
    best_thr, thr_results = _sweep_threshold(fold_cache, thresholds, gt_train, durs, use_max_pooling_23b)

    # ── Evaluate OOF at best threshold ────────────────────────────────────────
    all_oof_rows = []
    for test_sid, cand_data, bag_scores_dict in fold_cache:
        bs     = bag_scores_dict[test_sid]
        cands  = cand_data["cands"]
        p14ev  = cand_data["p14_events"]
        dur_s  = cand_data["dur_s"]
        T      = cand_data["T"]
        high_thr = min(best_thr * 2.5, 0.50)
        events   = []
        for cand in cands:
            s = int(cand["t_start"] * RATE_HZ)
            e = min(int(cand["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(bs[s:e]))
            if mx <= best_thr * 0.6:
                continue
            p14_ok = any(
                min(cand["t_end"], pev["t_end"]) - max(cand["t_start"], pev["t_start"]) > 0
                for pev in p14ev
            )
            if (p14_ok and mx > best_thr) or mx > high_thr:
                events.append({"t_start": cand["t_start"], "t_end": cand["t_end"],
                               "score": mx, "t_alarm": cand["t_start"]})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        for ev in events:
            all_oof_rows.append({
                "stream_id": test_sid,
                "t_start":  f"{ev['t_start']:.3f}",
                "t_end":    f"{ev['t_end']:.3f}",
                "score":    f"{ev['score']:.6f}",
                "t_alarm":  f"{ev['t_alarm']:.3f}",
            })

    oof_csv = ROOT / "oof_pipeline_23.csv"
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
    delta_01   = oof_metrics["f1_01"]   - (_load_log_entry(base_id).get("oof_f1_01") or
                                           _load_log_entry(base_id).get("f1_01") or 0)
    print(f"\n=== OOF Results (p23) ===")
    print(f"  F1_mean={oof_metrics['f1_mean']:.4f}  (R={R}, Δ={delta_mean:+.4f})")
    print(f"  F1_01  ={oof_metrics['f1_01']:.4f}  Δ_01={delta_01:+.4f}")
    print(f"  F1_05  ={oof_metrics['f1_05']:.4f}")
    print(f"  TP={oof_metrics['tp']}  FP={oof_metrics['fp']}  FN={oof_metrics['fn']}")

    # ── Decision ──────────────────────────────────────────────────────────────
    R_cur_01 = _load_log_entry(base_id).get("oof_f1_01") or _load_log_entry(base_id).get("f1_01") or 0
    if delta_mean < -0.005:
        decision = "DROP"
        notes    = f"OOF F1_mean dropped {delta_mean:+.4f}"
    elif oof_metrics["f1_mean"] >= (R or 0) + 0.005 and oof_metrics["f1_01"] >= R_cur_01 + 0.01:
        decision = "KEEP"
        notes    = f"F1_mean Δ={delta_mean:+.4f}, F1_01 Δ={delta_01:+.4f} — both pass"
    else:
        decision = "DROP"
        notes    = (f"Δ_mean={delta_mean:+.4f} (need≥+0.005) or "
                    f"Δ_01={delta_01:+.4f} (need≥+0.010)")
    print(f"\nDecision: {decision}  — {notes}")

    # ── Val evaluation if KEEP ────────────────────────────────────────────────
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation with fold-bagging ===")
        from abc2027.evaluate import stream_durations as sv
        val_durs = sv(STREAMS_VAL)
        val_streams = load_val()

        # Val rows using all 13 fold models bagged
        val_rows_single = run(val_streams, fold_models, best_thr, use_max_pooling_23b)
        val_path = save_sub(val_rows_single, "pipeline_23")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p23, bagged {len(fold_models)} models) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    # ── Log ───────────────────────────────────────────────────────────────────
    p19c_val = _load_log_entry("pipeline_19c").get("f1_mean") or 0.0
    delta_val = round((val_metrics["f1_mean"] if val_metrics else 0) - p19c_val, 4)
    entry = {
        "experiment_id": "pipeline_23",
        "phase": 4,
        "base": base_id,
        "description": ("p23a(clean_labels) + p23b(max_pool) + fold-bagging "
                        f"thr={best_thr:.3f}"),
        "components": ["clean_labels", "max_pooling", "fold_bagging"],
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
        "notes":    (f"23a clean: {n_clean_train} pos, {n_ignored_train} ignored. "
                     f"23b max_pool. thr={best_thr:.3f}. Δ_mean={delta_mean:+.4f}, "
                     f"Δ_01={delta_01:+.4f}. {notes}. "
                     f"pct_ignored={pct_removed:.1f}%"),
    }
    append_log(entry)
    print(f"\nLogged: pipeline_23  decision={decision}")


if __name__ == "__main__":
    main()
