"""Pipeline 41: Multi-scale Stage-1 pct=95 UNION adaptive rolling 30s pct=97.

Combines best of p32 (multi-scale top-K energy) and p39 (union Stage-1).

  p32 showed multi-scale Stage-1 gave +0.0291 OOF vs global pct=97 (p30→p32).
  p39 showed union stage-1 gave +0.0379 OOF vs single adaptive (p37→p39).
  Hypothesis: replace the 'global pct=97' branch in p39 with the better
  'multi-scale pct=95' branch from p32 → union of higher-quality candidates.

Stage-1 options:
  Branch A: multi-scale top-K=32 bins, sigma=[0.15, 0.30, 0.60]s, pct=95  (p32)
  Branch B: adaptive rolling 30s window pct=97 + global 30th-pct floor    (p37)
  Union of A+B with gap-merge 0.5s.

Stage-2: LightGBM 27f (same as p30/p39).

Base: pipeline_39 (OOF=0.7219)
Decision: KEEP if OOF F1_mean >= R_cur - 0.005
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EVENTS_CSV  = ROOT / "streams_events.csv"
COMP_CSV    = ROOT / "streams_composition.csv"
RATE_HZ     = 50.0

from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    energy_events_raw, STREAMS_TRAIN, STREAMS_VAL,
)
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter
from experiments.postprocessing.gap_merger import gap_merge
from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
from experiments.models.lightgbm_classifier import (
    window_features_rich, _try_lgb,
)
from experiments.audit.oof_harness import build_folds
from abc2027.baseline import frame_energy, iter_windows, STEP_S
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions, temporal_iou,
)
from abc2027.baseline import runs_to_events


# ── Features: same as p30/p39 ────────────────────────────────────────────────

def range_centroid_features(block: np.ndarray) -> np.ndarray:
    residual = block.astype(np.float64)
    residual -= np.mean(residual, axis=1, keepdims=True)
    e = residual ** 2
    B = e.shape[2]
    bins = np.arange(B, dtype=np.float64)
    T = e.shape[1]
    third = T // 3
    if third == 0:
        return np.zeros(6)

    def centroid_seg(seg):
        row_e = seg.mean(axis=1)
        row_s = row_e.sum(axis=1) + 1e-12
        return (row_e @ bins) / row_s

    c1 = centroid_seg(e[:, :third, :])
    c2 = centroid_seg(e[:, third:2*third, :])
    c3 = centroid_seg(e[:, 2*third:, :])
    return np.concatenate([c3 - c1, c2 - (c1 + c3) / 2])


LEAKY_INDICES = list(range(21, 27))
CLEAN_INDICES = [i for i in range(27) if i not in LEAKY_INDICES]


def window_features_p41(block: np.ndarray) -> np.ndarray:
    """27 features: 21 clean + 6 range centroid dynamics (same as p30/p39)."""
    clean = window_features_rich(block)[CLEAN_INDICES]
    dynf  = range_centroid_features(block)
    return np.concatenate([clean, dynf])


# ── Stage-1 methods ───────────────────────────────────────────────────────────

def multiscale_energy(filtered: np.ndarray) -> np.ndarray:
    """Multi-scale energy profile (p32 style).
    Top-K=32 active bins, max across radars, 3 smoothing scales, element-wise max.
    """
    K = 32
    energy_per_bin = (filtered.astype(np.float64) ** 2).mean(axis=(0, 1))  # (368,)
    top_bins = np.argsort(energy_per_bin)[-K:]

    per_radar_topk = np.sum(
        filtered[:, :, top_bins].astype(np.float64) ** 2, axis=2
    )  # (3, T)
    base_energy = np.max(per_radar_topk, axis=0)  # (T,)

    scales_s = [0.15, 0.30, 0.60]
    result = np.zeros_like(base_energy)
    for sigma_s in scales_s:
        smoothed = smooth(base_energy, sigma_s=sigma_s)
        result = np.maximum(result, smoothed)
    return result


def stage1_multiscale(energy_ms: np.ndarray, pct: float = 95.0,
                      min_dur: float = 0.4, merge_gap: float = 0.8) -> List[dict]:
    """p32-style multi-scale Stage-1 with global percentile threshold."""
    return energy_events_raw(energy_ms, pct=pct, min_dur=min_dur, merge_gap=merge_gap)


def stage1_adaptive(energy_s1: np.ndarray, window_s: float = 30.0,
                    pct: float = 97.0, min_dur: float = 0.4,
                    merge_gap: float = 0.8) -> List[dict]:
    """Adaptive rolling local percentile threshold (p37 style)."""
    window_frames = max(3, int(window_s * RATE_HZ) | 1)
    local_thr = percentile_filter(energy_s1, pct, size=window_frames)
    global_floor = float(np.percentile(energy_s1, 30))
    threshold = np.maximum(local_thr, global_floor)
    mask = energy_s1 >= threshold
    if not np.any(mask):
        return []
    score = energy_s1 / (threshold + 1e-12)
    return runs_to_events(mask, score, min_dur, merge_gap)


def _frame_energy_simple(filtered: np.ndarray) -> np.ndarray:
    """Simple max-pool energy for adaptive Stage-1 (p30/p37 style)."""
    per_radar = np.sum(filtered.astype(np.float64) ** 2, axis=2)
    return np.max(per_radar, axis=0)


def union_candidates(
    cands_a: List[dict],
    cands_b: List[dict],
    merge_gap: float = 0.5,
) -> List[dict]:
    """Union two Stage-1 candidate lists with gap-merge."""
    all_cands = cands_a + cands_b
    if not all_cands:
        return []
    all_cands = sorted(all_cands, key=lambda c: c["t_start"])
    merged = [dict(all_cands[0])]
    for c in all_cands[1:]:
        prev = merged[-1]
        if c["t_start"] <= prev["t_end"] + merge_gap:
            prev["t_end"]   = max(prev["t_end"], c["t_end"])
            prev["score"]   = max(prev["score"], c["score"])
            prev["t_alarm"] = prev["t_start"]
        else:
            merged.append(dict(c))
    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def collect_hard_negatives(
    train_sids: List[str], gt_by: dict, feat_fn, weight: float = 2.0,
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
                X_hard.append(feat_fn(filtered[:, ws:we, :]))
                w_hard.append(weight)
    if not X_hard:
        return np.empty((0, 27)), np.empty(0), np.empty(0)
    return np.array(X_hard), np.zeros(len(X_hard), dtype=np.int32), np.array(w_hard)


def _train_fold(
    train_sids: List[str], gt_all: list, feat_fn, hard_neg_weight: float = 2.0,
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
        T = filtered.shape[1]
        wins = iter_windows(T)
        if not wins:
            continue
        feats  = np.array([feat_fn(filtered[:, s:e, :]) for s, e in wins])
        gt_ev  = [(ev["t_start"], ev["t_end"]) for ev in gt_by.get(sid, [])]
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
    n = len(X_pos); half = X_pos.shape[1] // 2
    perm = rng.permutation(n)
    aug.append(np.concatenate([X_pos[:, :half], X_pos[perm, half:]], axis=1))

    X_aug  = np.vstack(aug)
    X_neg  = X[y == 0]; w_neg = w[y == 0]
    X_full = np.vstack([X_neg, X_aug])
    y_full = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_aug))]).astype(np.int32)
    w_full = np.concatenate([w_neg, np.ones(len(X_aug))])

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Pipeline 41 — Multi-scale Stage-1 pct=95 UNION adaptive rolling pct=97")
    print("=" * 70)

    for eid in ("pipeline_40", "pipeline_39", "pipeline_37", "pipeline_30",
                "pipeline_28", "pipeline_24"):
        e = _load_log_entry(eid)
        if e.get("decision") in ("KEEP", "BASE"):
            base_id = eid
            R    = e.get("oof_f1_mean", 0.7219)
            R_05 = e.get("oof_f1_05",  0.6135)
            break
    else:
        base_id, R, R_05 = "pipeline_39", 0.7219, 0.6135

    print(f"Base: {base_id}  OOF F1_mean={R:.4f}  OOF F1_05={R_05:.4f}")
    print(f"Stage-1: multi-scale top-K=32 pct=95 UNION adaptive 30s pct=97")
    print(f"Stage-2: LightGBM 27f (same as p30)\n")

    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    MS_PCT      = 95.0
    ADAPT_PCT   = 97.0
    ADAPT_WIN_S = 30.0

    folds = build_folds()
    print(f"Running {len(folds)} fold OOF…\n")
    fold_cache = []
    fold_models = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"  Fold {k:02d}: training…", flush=True)
        model = _train_fold(train_sids, gt_all, window_features_p41, hard_neg_weight=2.0)
        fold_models.append(model)

        stream_idx = int(test_sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T        = filtered.shape[1]
        dur_s    = T / RATE_HZ

        # Multi-scale energy for Branch A
        energy_ms = multiscale_energy(filtered)
        # Simple max-pool energy for Branch B adaptive
        energy_simple = smooth(_frame_energy_simple(filtered), sigma_s=0.3)
        # energy_p14 for dual-gate
        energy_p14 = smooth(frame_energy(filtered), sigma_s=0.3)

        cands_ms = stage1_multiscale(energy_ms, pct=MS_PCT, min_dur=0.4, merge_gap=0.8)
        cands_ad = stage1_adaptive(energy_simple, window_s=ADAPT_WIN_S, pct=ADAPT_PCT,
                                    min_dur=0.4, merge_gap=0.8)
        cands    = union_candidates(cands_ms, cands_ad, merge_gap=0.5)

        p14_cands  = energy_events_raw(energy_p14, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy_p14, dur_s, k=1.2)

        if model is not None:
            wins  = iter_windows(T)
            feats = np.array([window_features_p41(filtered[:, s:e, :]) for s, e in wins]) if wins else None
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
            {"cands": cands,
             "cands_ms": len(cands_ms), "cands_ad": len(cands_ad), "cands_union": len(cands),
             "p14_events": p14_events,
             "energy_ms": energy_ms, "energy_simple": energy_simple,
             "dur_s": dur_s, "T": T},
            {test_sid: fs},
        ))
        print(f"    ms={len(cands_ms)} + adaptive={len(cands_ad)} → union={len(cands)}", flush=True)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print(f"\nPhase 2: OOF threshold sweep…")
    thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
    tmp_csv = ROOT / "_p41_tmp.csv"
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

    # Final OOF
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
                # Use multi-scale energy for boundary refinement
                ns, ne = refine_boundary(cd["energy_ms"], cand["t_start"],
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

    oof_csv = ROOT / "oof_pipeline_41.csv"
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

    print(f"\n=== OOF Results (p41) ===")
    print(f"  F1_mean={oof['f1_mean']:.4f}  Delta={delta_mean:+.4f}")
    print(f"  F1_05  ={oof['f1_05']:.4f}  Delta_05={delta_05:+.4f}")
    print(f"  TP={oof['tp']}  FP={oof['fp']}  FN={oof['fn']}")

    total_ms = sum(cd["cands_ms"] for _, cd, _ in fold_cache)
    total_ad = sum(cd["cands_ad"] for _, cd, _ in fold_cache)
    total_u  = sum(cd["cands_union"] for _, cd, _ in fold_cache)
    print(f"  Candidates: ms={total_ms} adaptive={total_ad} union={total_u}")

    decision = "KEEP" if delta_mean >= -0.005 else "DROP"
    notes = (
        f"ms_pct={MS_PCT:.0f}+adaptive_30s_pct={ADAPT_PCT:.0f} union; "
        f"total_candidates={total_u}; Δ_mean={delta_mean:+.4f}; Δ_05={delta_05:+.4f}"
    )
    print(f"Decision: {decision}  — {notes}")

    val_metrics = None
    if decision == "KEEP":
        valid_fold_models = [m for m in fold_models if m is not None]
        print(f"\n=== Val evaluation (fold-bagging: average {len(valid_fold_models)} fold models) ===")
        val_streams = load_val()
        val_rows = []
        for sid, cir in val_streams.items():
            filtered = chebyshev_bp(cir)
            T        = filtered.shape[1]
            dur_s    = T / RATE_HZ
            energy_ms     = multiscale_energy(filtered)
            energy_simple = smooth(_frame_energy_simple(filtered), sigma_s=0.3)
            energy_p14    = smooth(frame_energy(filtered), sigma_s=0.3)

            cands_ms = stage1_multiscale(energy_ms, pct=MS_PCT, min_dur=0.4, merge_gap=0.8)
            cands_ad = stage1_adaptive(energy_simple, window_s=ADAPT_WIN_S, pct=ADAPT_PCT,
                                        min_dur=0.4, merge_gap=0.8)
            cands    = union_candidates(cands_ms, cands_ad, merge_gap=0.5)

            p14_cands  = energy_events_raw(energy_p14, pct=98.0, min_dur=0.6, merge_gap=0.8)
            p14_events = gaussian_boundary(p14_cands, energy_p14, dur_s, k=1.2)

            wins  = iter_windows(T)
            if not wins:
                continue
            feats = np.array([window_features_p41(filtered[:, s:e, :]) for s, e in wins])

            all_probs = []
            for fold_m in valid_fold_models:
                raw = fold_m.predict_proba(feats)[:, 1]
                raw = gaussian_filter1d(raw, sigma=2.0)
                xs  = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
                fs_fold = np.interp(np.arange(T, dtype=np.float64), xs, raw)
                fs_fold = gaussian_filter1d(fs_fold, sigma=5.0)
                all_probs.append(fs_fold)
            fs = np.mean(all_probs, axis=0)

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
                    ns, ne = refine_boundary(energy_ms, ev["t_start"], ev["t_end"], dur_s)
                    events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            val_rows.extend(make_rows(sid, events))

        val_path    = save_sub(val_rows, "pipeline_41")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p41) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    base_val_f1 = _load_log_entry(base_id).get("f1_mean") or 0.0
    delta_val   = round((val_metrics["f1_mean"] - base_val_f1) if val_metrics else 0.0, 4)
    entry = {
        "experiment_id": "pipeline_41",
        "phase": 5, "base": base_id,
        "description": (
            f"p41: multi-scale top-K=32 pct={MS_PCT:.0f} UNION adaptive {ADAPT_WIN_S:.0f}s pct={ADAPT_PCT:.0f}; "
            f"27f LightGBM"
        ),
        "components": [
            "dual_stage1_union", "multiscale_stage1_topk32_pct95",
            "adaptive_stage1_rolling30s",
            "range_centroid_dynamics", "feature_pruning_21f", "hard_negative_mining",
        ],
        "oof_f1_mean": oof["f1_mean"], "oof_f1_01": oof["f1_01"],
        "oof_f1_03":   oof["f1_03"],   "oof_f1_05": oof["f1_05"],
        "oof_tp": oof["tp"], "oof_fp": oof["fp"], "oof_fn": oof["fn"],
        "delta_oof_vs_base": delta_mean, "p_delta_gt0": None,
        "f1_mean":  val_metrics["f1_mean"] if val_metrics else None,
        "f1_01":    val_metrics["f1_01"]   if val_metrics else None,
        "f1_03":    val_metrics["f1_03"]   if val_metrics else None,
        "f1_05":    val_metrics["f1_05"]   if val_metrics else None,
        "tp":       val_metrics["tp"]       if val_metrics else None,
        "fp":       val_metrics["fp"]       if val_metrics else None,
        "fn":       val_metrics["fn"]       if val_metrics else None,
        "false_alarms_per_hour": val_metrics["false_alarms_per_hour"] if val_metrics else None,
        "delta_f1": delta_val if val_metrics else None,
        "probe_auc": None, "probe_macro_f1": None,
        "gate": None, "decision": decision, "notes": notes,
    }
    append_log(entry)
    print(f"\nLogged: pipeline_41  decision={decision}")
    return decision, oof, val_metrics


if __name__ == "__main__":
    main()
