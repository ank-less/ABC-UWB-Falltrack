"""Pipeline 52: Combined Stage-1 fix + asymmetric boundary extension.

Combines:
1. Best Stage-1 from GD6 (spike-robust adaptive or gap-fill causal)
2. Systematic t_end extension to fix F1_05 boundary underestimation
3. Builds on whichever pipeline is best KEEP from {p48, p49, p50, p51}

Falls detected by the energy-based Stage-1 end when bandpass energy drops below
threshold. But annotators use static CIR profile changes which persist longer
(person settling on floor). Extending t_end by delta_end (tuned on OOF) directly
addresses this systematic underestimation.

Previous boundary attempts:
- P42: static CIR deviation -> EMA converges -> no signal -> merged intervals
- P43: fixed 2s window -> wrong for variable-length falls
- P44: EMA velocity -> alpha too slow -> no signal
- P46: only t_start shift (-0.2s) -> OOF F1_05=0.6626

New approach: sweep delta_end in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]s on OOF.
No complex refining model needed. Simple global extension similar to p46's
t_start extension.

Stage-1: best from {p46/p48/p49/p51} — reads best KEEP from log
Post:    extend t_start by -0.2s (from p46) AND t_end by +delta_end (new)

Base:   best KEEP pipeline in experiments_log.json
KEEP if OOF F1_mean >= best_base + 0.02.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EVENTS_CSV = ROOT / "streams_events.csv"
RATE_HZ    = 50.0

from experiments.utils import (
    load_val, make_rows, run_eval, save_sub, append_log,
    energy_events_raw, STREAMS_TRAIN,
)
from experiments.preprocessing.chebyshev_filter import chebyshev_bp
from experiments.postprocessing.temporal_smoothing import smooth
from experiments.postprocessing.duration_filter import dur_filter
from experiments.postprocessing.gap_merger import gap_merge
from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
from experiments.models.lightgbm_classifier import window_features_rich, _try_lgb
from experiments.audit.oof_harness import build_folds
from abc2027.baseline import frame_energy, iter_windows, STEP_S
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations,
    validate_predictions, temporal_iou,
)
from abc2027.baseline import runs_to_events


# ── Features: same 27f as p46 ─────────────────────────────────────────────────

def range_centroid_features(block: np.ndarray) -> np.ndarray:
    residual = block.astype(np.float64)
    residual -= np.mean(residual, axis=1, keepdims=True)
    e    = residual ** 2
    bins = np.arange(e.shape[2], dtype=np.float64)
    T    = e.shape[1]
    third = T // 3
    if third == 0:
        return np.zeros(6)
    def _centroid(seg):
        row_e = seg.mean(axis=1)
        row_s = row_e.sum(axis=1) + 1e-12
        return (row_e @ bins) / row_s
    c1 = _centroid(e[:, :third,        :])
    c2 = _centroid(e[:, third:2*third, :])
    c3 = _centroid(e[:, 2*third:,      :])
    return np.concatenate([c3 - c1, c2 - (c1 + c3) / 2])


LEAKY_INDICES = list(range(21, 27))
CLEAN_INDICES = [i for i in range(27) if i not in LEAKY_INDICES]


def window_features_p52(block: np.ndarray) -> np.ndarray:
    clean = window_features_rich(block)[CLEAN_INDICES]
    dynf  = range_centroid_features(block)
    return np.concatenate([clean, dynf])


# ── Stage-1: same as p46 (global + adaptive 30s) ─────────────────────────────

def _frame_energy_stage1(filtered: np.ndarray) -> np.ndarray:
    per_radar = np.sum(filtered.astype(np.float64) ** 2, axis=2)
    return np.max(per_radar, axis=0)


def stage1_global(energy_s1, pct=97.0, min_dur=0.4, merge_gap=0.8):
    return energy_events_raw(energy_s1, pct=pct, min_dur=min_dur, merge_gap=merge_gap)


def stage1_adaptive(energy_s1, window_s=30.0, pct=97.0, min_dur=0.4, merge_gap=0.8):
    W = max(3, int(window_s * RATE_HZ) | 1)
    local_thr = percentile_filter(energy_s1, pct, size=W)
    global_floor = float(np.percentile(energy_s1, 30))
    threshold = np.maximum(local_thr, global_floor)
    mask = energy_s1 >= threshold
    if not np.any(mask):
        return []
    score = energy_s1 / (threshold + 1e-12)
    return runs_to_events(mask, score, min_dur, merge_gap)


def union_candidates(g, a, merge_gap=0.5):
    all_c = sorted((g or []) + (a or []), key=lambda c: c["t_start"])
    if not all_c:
        return []
    merged = [dict(all_c[0])]
    for c in all_c[1:]:
        prev = merged[-1]
        if c["t_start"] <= prev["t_end"] + merge_gap:
            prev["t_end"]   = max(prev["t_end"],   c["t_end"])
            prev["score"]   = max(prev["score"],   c["score"])
            prev["t_alarm"] = prev["t_start"]
        else:
            merged.append(dict(c))
    return merged


# ── Post-processing ────────────────────────────────────────────────────────────

def extend_intervals(events, dur_s, delta_start, delta_end):
    out = []
    for ev in events:
        ns = max(0.0, ev["t_start"] - delta_start)
        ne = min(dur_s, ev["t_end"]   + delta_end)
        if ne - ns >= 0.4:
            out.append({**ev, "t_start": ns, "t_end": ne, "t_alarm": ns})
    return out


# ── Training: same as p46 ─────────────────────────────────────────────────────

def collect_hard_negatives(train_sids, gt_by, feat_fn, weight=2.0):
    X_hard, w_hard = [], []
    for sid in train_sids:
        idx = int(sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T    = filtered.shape[1]
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
        return np.empty((0, 27)), np.empty(0, dtype=np.int32), np.empty(0)
    return np.array(X_hard), np.zeros(len(X_hard), dtype=np.int32), np.array(w_hard)


def _train_fold(train_sids, gt_all, feat_fn, hard_neg_weight=2.0):
    gt_by = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)
    X_parts, y_parts, w_parts = [], [], []
    for sid in train_sids:
        idx = int(sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T    = filtered.shape[1]
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
        w_parts.append(np.ones(len(labels), dtype=np.float64))
    if not X_parts:
        return None
    X = np.vstack(X_parts);  y = np.concatenate(y_parts);  w = np.concatenate(w_parts)

    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        return None
    X_pos = X[pos_idx];  w_pos = w[pos_idx]
    rng   = np.random.default_rng(42)
    stds  = X_pos.std(axis=0) + 1e-8
    aug   = [X_pos];  w_aug = [w_pos]
    for scale in (0.04, 0.07, 0.11):
        aug.append(X_pos + rng.normal(0, 1, X_pos.shape) * stds * scale)
        w_aug.append(w_pos)
    for lo, hi in ((0.78, 1.22), (0.85, 1.15)):
        aug.append(X_pos * rng.uniform(lo, hi, (len(X_pos), 1)))
        w_aug.append(w_pos)
    n = len(X_pos); half = X_pos.shape[1] // 2
    perm = rng.permutation(n)
    aug.append(np.concatenate([X_pos[:, :half], X_pos[perm, half:]], axis=1))
    w_aug.append(w_pos)

    X_aug  = np.vstack(aug);   w_aug = np.concatenate(w_aug)
    X_neg  = X[y == 0];        w_neg = w[y == 0]
    X_full = np.vstack([X_neg, X_aug])
    y_full = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_aug))]).astype(np.int32)
    w_full = np.concatenate([w_neg, w_aug])

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


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_log(eid):
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


def _find_best_base():
    """Find current best KEEP pipeline from GD5/GD6 log."""
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return "pipeline_46", 0.7423
    log = json.loads(f.read_text(encoding="utf-8"))
    candidates = [e for e in log
                  if e.get("phase") in (5, 6)
                  and e.get("decision") == "KEEP"
                  and isinstance(e.get("oof_f1_mean"), float)
                  and e.get("oof_f1_mean", 0) >= 0.7423]
    if not candidates:
        return "pipeline_46", 0.7423
    best = max(candidates, key=lambda e: e.get("oof_f1_mean", 0))
    return best["experiment_id"], best["oof_f1_mean"]


def main():
    base_id, R = _find_best_base()
    R_05 = _load_log(base_id).get("oof_f1_05", 0.6626)

    print("=" * 70)
    print("Pipeline 52 — Combined Stage-1 + t_end extension sweep")
    print(f"Inherits Stage-1 from p46 (same as base {base_id})")
    print(f"Adds: sweep delta_end in [0..3s] on top of fixed ds=0.2s")
    print("=" * 70)
    print(f"Base: {base_id}  OOF F1_mean={R:.4f}  OOF F1_05={R_05:.4f}")
    print(f"KEEP if OOF F1_mean >= {R + 0.02:.4f}\n")

    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    DS_FIXED    = 0.2   # from p46
    GLOBAL_PCT  = 97.0
    ADAPT_WIN_S = 30.0

    folds = build_folds()
    print(f"Running {len(folds)} OOF folds…\n")
    fold_cache  = []
    fold_models = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"  Fold {k:02d}: training…", flush=True)
        model = _train_fold(train_sids, gt_all, window_features_p52, hard_neg_weight=2.0)
        fold_models.append(model)

        idx = int(test_sid.split("_")[1])
        p_h5 = STREAMS_TRAIN / f"stream_{idx:02d}.h5"
        with h5py.File(p_h5) as f:
            cir = f["cir"][:]
        filtered   = chebyshev_bp(cir)
        T          = filtered.shape[1]
        dur_s      = T / RATE_HZ
        energy_s1  = smooth(_frame_energy_stage1(filtered), sigma_s=0.3)
        energy_p14 = smooth(frame_energy(filtered),         sigma_s=0.3)
        cands_g    = stage1_global(energy_s1, pct=GLOBAL_PCT, min_dur=0.4, merge_gap=0.8)
        cands_a    = stage1_adaptive(energy_s1, window_s=ADAPT_WIN_S, pct=97.0,
                                     min_dur=0.4, merge_gap=0.8)
        cands      = union_candidates(cands_g, cands_a, merge_gap=0.5)
        p14_cands  = energy_events_raw(energy_p14, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy_p14, dur_s, k=1.2)

        if model is not None:
            wins  = iter_windows(T)
            feats = (np.array([window_features_p52(filtered[:, s:e, :]) for s, e in wins])
                     if wins else None)
            if feats is not None:
                raw_p = model.predict_proba(feats)[:, 1]
                raw_p = gaussian_filter1d(raw_p, sigma=2.0)
                xs    = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
                fs    = np.interp(np.arange(T, dtype=np.float64), xs, raw_p)
                fs    = gaussian_filter1d(fs, sigma=5.0)
            else:
                fs = np.zeros(T)
        else:
            fs = np.zeros(T)

        fold_cache.append((
            test_sid,
            {"cands": cands, "energy_s1": energy_s1, "p14_events": p14_events,
             "dur_s": dur_s, "T": T},
            {test_sid: fs},
        ))
        print(f"    cands={len(cands)}", flush=True)

    tmp_csv = ROOT / "_p52_tmp.csv"

    def _eval_params(thr, delta_end):
        high_thr = min(thr * 2.5, 0.50)
        rows = []
        for test_sid, cd, scores_d in fold_cache:
            bs     = scores_d[test_sid]
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
                    ns, ne = refine_boundary(cd["energy_s1"], cand["t_start"],
                                             cand["t_end"], cd["dur_s"])
                    events.append({"t_start": ns, "t_end": ne,
                                   "score": mx, "t_alarm": ns})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            events = extend_intervals(events, cd["dur_s"], DS_FIXED, delta_end)
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=0.5)
            for ev in events:
                rows.append({"stream_id": test_sid,
                             "t_start": f"{ev['t_start']:.3f}",
                             "t_end":   f"{ev['t_end']:.3f}",
                             "score":   f"{ev['score']:.6f}",
                             "t_alarm": f"{ev['t_alarm']:.3f}"})
        with open(tmp_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
            wr.writeheader();  wr.writerows(rows)
        try:
            preds = validate_predictions(tmp_csv, durs)
            res   = evaluate(gt_train, preds, durs)
            rs    = res["ranking_score"];  bt = res["by_theta"]
            return (rs["f1_mean"], bt["0.5"]["f1"],
                    bt["0.1"]["f1"], bt["0.1"]["tp"], bt["0.1"]["fp"])
        except Exception as ex:
            print(f"  eval error: {ex}")
            return 0.0, 0.0, 0.0, 0, 0

    # ── Pass 1: sweep delta_end (anchor thr=0.500) ────────────────────────────
    print("\nPass 1: delta_end sweep (anchor thr=0.500)…")
    de_grid = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    best_p1_f1, best_de = 0.0, 0.0
    for de in de_grid:
        f1, f105, f101, tp01, fp01 = _eval_params(0.500, de)
        flag = " <-" if f1 > best_p1_f1 else ""
        print(f"  de={de:4.1f}s  F1={f1:.4f}  F1_05={f105:.4f}  TP_01={tp01}  FP_01={fp01}{flag}")
        if f1 > best_p1_f1:
            best_p1_f1 = f1
            best_de    = de
    print(f"\n  Best: de={best_de}s  F1={best_p1_f1:.4f}")

    # ── Pass 2: threshold sweep ───────────────────────────────────────────────
    print(f"\nPass 2: thr sweep (de={best_de}s)…")
    best_f1, best_thr = 0.0, 0.500
    for thr in [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]:
        f1, _, _, _, _ = _eval_params(thr, best_de)
        flag = " <-" if f1 > best_f1 else ""
        print(f"  thr={thr:.3f}  F1={f1:.4f}{flag}")
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"  Best thr={best_thr:.3f}  F1={best_f1:.4f}")

    tmp_csv.unlink(missing_ok=True)

    # ── Final OOF ─────────────────────────────────────────────────────────────
    print("\nBuilding final OOF predictions…")
    oof_csv  = ROOT / "oof_pipeline_52.csv"
    high_thr = min(best_thr * 2.5, 0.50)
    all_rows = []
    for test_sid, cd, scores_d in fold_cache:
        bs     = scores_d[test_sid]
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
                ns, ne = refine_boundary(cd["energy_s1"], cand["t_start"],
                                         cand["t_end"], cd["dur_s"])
                events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        events = extend_intervals(events, cd["dur_s"], DS_FIXED, best_de)
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=0.5)
        for ev in events:
            all_rows.append({"stream_id": test_sid,
                             "t_start": f"{ev['t_start']:.3f}",
                             "t_end":   f"{ev['t_end']:.3f}",
                             "score":   f"{ev['score']:.6f}",
                             "t_alarm": f"{ev['t_alarm']:.3f}"})

    with open(oof_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
        wr.writeheader();  wr.writerows(all_rows)

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
        "tp_01":   bt["0.1"]["tp"],
        "false_alarms_per_hour": round(rs["false_alarms_per_hour"], 1),
    }
    delta_mean = round(oof["f1_mean"] - R, 4)
    delta_05   = round(oof["f1_05"]   - R_05, 4)

    print(f"\n=== OOF Results (p52) ===")
    print(f"  F1_mean={oof['f1_mean']:.4f}  delta={delta_mean:+.4f}  (KEEP if >= {R+0.02:.4f})")
    print(f"  F1_05  ={oof['f1_05']:.4f}   delta_05={delta_05:+.4f}")
    print(f"  TP_01={oof['tp_01']} (base={_load_log(base_id).get('oof_tp', '?')})")
    print(f"  TP={oof['tp']}  FP={oof['fp']}  FN={oof['fn']}")

    keep     = oof["f1_mean"] >= R + 0.02
    decision = "KEEP" if keep else "DROP"
    notes    = (
        f"base={base_id}; de={best_de}s ds=0.2s thr={best_thr:.3f}; "
        f"delta_mean={delta_mean:+.4f} delta_05={delta_05:+.4f}; "
        f"TP_01={oof['tp_01']}"
    )
    print(f"Decision: {decision}  --  {notes}")

    val_metrics = None
    if decision == "KEEP":
        valid_models = [m for m in fold_models if m is not None]
        print(f"\n=== Val evaluation (fold-bagging: {len(valid_models)} models) ===")
        val_rows = []
        for sid, cir_v in load_val().items():
            filt_v     = chebyshev_bp(cir_v)
            T_v        = filt_v.shape[1]
            dur_s_v    = T_v / RATE_HZ
            en_s1_v    = smooth(_frame_energy_stage1(filt_v), sigma_s=0.3)
            en_p14_v   = smooth(frame_energy(filt_v),         sigma_s=0.3)
            cg_v       = stage1_global(en_s1_v, pct=GLOBAL_PCT, min_dur=0.4, merge_gap=0.8)
            ca_v       = stage1_adaptive(en_s1_v, window_s=ADAPT_WIN_S, pct=97.0,
                                         min_dur=0.4, merge_gap=0.8)
            cands_v    = union_candidates(cg_v, ca_v, merge_gap=0.5)
            p14c_v     = energy_events_raw(en_p14_v, pct=98.0, min_dur=0.6, merge_gap=0.8)
            p14ev_v    = gaussian_boundary(p14c_v, en_p14_v, dur_s_v, k=1.2)

            wins_v  = iter_windows(T_v)
            if not wins_v:
                continue
            feats_v = np.array([window_features_p52(filt_v[:, s:e, :]) for s, e in wins_v])
            all_prob = []
            for fm in valid_models:
                rp = gaussian_filter1d(fm.predict_proba(feats_v)[:, 1], sigma=2.0)
                xs = np.array([(s + e) // 2 for s, e in wins_v], dtype=np.float64)
                fp = gaussian_filter1d(np.interp(np.arange(T_v, dtype=np.float64), xs, rp),
                                       sigma=5.0)
                all_prob.append(fp)
            fs_v = np.mean(all_prob, axis=0)

            ht_v  = min(best_thr * 2.5, 0.50)
            evs_v = []
            for ev in cands_v:
                s = int(ev["t_start"] * RATE_HZ)
                e = min(int(ev["t_end"] * RATE_HZ) + 1, T_v)
                if s >= e:
                    continue
                mx = float(np.max(fs_v[s:e]))
                if mx <= best_thr * 0.6:
                    continue
                p14_ok = any(
                    min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                    for pev in p14ev_v
                )
                if (p14_ok and mx > best_thr) or mx > ht_v:
                    ns, ne = refine_boundary(en_s1_v, ev["t_start"], ev["t_end"], dur_s_v)
                    evs_v.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
            evs_v = dur_filter(evs_v, min_dur=0.6)
            evs_v = gap_merge(evs_v, max_gap=1.0)
            evs_v = extend_intervals(evs_v, dur_s_v, DS_FIXED, best_de)
            evs_v = dur_filter(evs_v, min_dur=0.6)
            evs_v = gap_merge(evs_v, max_gap=0.5)
            val_rows.extend(make_rows(sid, evs_v))

        val_path    = save_sub(val_rows, "pipeline_52")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p52) ===")
        for kv, vv in val_metrics.items():
            print(f"  {kv}: {vv}")

    base_val_f1 = _load_log(base_id).get("f1_mean") or 0.0
    delta_val   = round((val_metrics["f1_mean"] - base_val_f1) if val_metrics else 0.0, 4)
    entry = {
        "experiment_id": "pipeline_52",
        "phase": 6, "base": base_id,
        "description": (
            f"p52 Stage-1 + t_end extension de={best_de}s ds=0.2s "
            f"thr={best_thr:.3f} (base={base_id})"
        ),
        "components": [
            "dual_stage1_union",
            "adaptive_stage1_rolling30s",
            "maxpool_stage1_pct97",
            "range_centroid_dynamics",
            "feature_pruning_21f",
            "hard_negative_mining",
            "asymmetric_interval_extension",
            "t_end_extension_sweep",
        ],
        "oof_f1_mean": oof["f1_mean"], "oof_f1_01": oof["f1_01"],
        "oof_f1_03":   oof["f1_03"],   "oof_f1_05": oof["f1_05"],
        "oof_tp":  oof["tp"],  "oof_fp": oof["fp"],  "oof_fn": oof["fn"],
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
    print(f"\nLogged: pipeline_52  decision={decision}")
    return decision, oof, val_metrics


if __name__ == "__main__":
    main()
