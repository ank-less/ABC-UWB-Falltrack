"""Pipeline 49 — Room-invariant clean features.

Root cause từ probe: f_mean/f_std/f_max/t_spread encode biên độ tuyệt đối
của phòng (radar coverage khác nhau) → leakage AUC=0.99 trên p46.

Fix: thay toàn bộ amplitude features bằng shape/ratio features không phụ thuộc
vào mức năng lượng tuyệt đối của phòng.

Feature set (21f, tất cả room-invariant):
  [0:3]   temporal_asym   — fraction of energy in 2nd half minus 1st half   (ratio)
  [3:6]   peak_t          — when peak occurs in window (0–1)                 (timing)
  [6:9]   coef_var        — std/mean of energy frames (shape)                (ratio)
  [9:12]  peak_mean_ratio — max/mean frame energy (spike sharpness)          (ratio)
  [12:15] delta_norm      — (last_third - first_third) / mean               (ratio)
  [15:18] centroid_shift  — (centroid_last_third - centroid_first_third) / B (ratio)
  [18:21] centroid_curve  — centroid_mid - avg(centroid_first, centroid_last) (ratio)

Stage-1: union global pct=97 + adaptive 30s pct=97 + short-window 5s pct=90 (same p48)
Post: asymmetric extension ds=0.2s de=0.0s (OOF-tuned in p46)

Base: pipeline_46 (OOF=0.7423)
KEEP if probe passes AND OOF F1_mean >= 0.7423 - 0.01 (allow small drop for cleanliness)
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
from experiments.audit.oof_harness import build_folds
from abc2027.baseline import frame_energy, iter_windows, STEP_S
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations,
    validate_predictions, temporal_iou,
)
from abc2027.baseline import runs_to_events


# ── Room-invariant feature extractor ─────────────────────────────────────────

def window_features_clean(block: np.ndarray) -> np.ndarray:
    """21 shape/ratio features — no absolute amplitude.

    All features are invariant to room-level energy scaling.
    block: (R, T, B) filtered CIR
    """
    R, T, B = block.shape
    e = block.astype(np.float64) ** 2  # energy

    # Per-radar per-frame energy (sum over range bins)
    e_frame = e.sum(axis=2)  # (R, T)

    bins = np.arange(B, dtype=np.float64)
    t3 = max(1, T // 3)

    feats = []
    for r in range(R):
        ef = e_frame[r]  # (T,)
        e_mean = ef.mean() + 1e-12

        # 1. Temporal asymmetry: fraction energy 2nd half - 1st half  [-1, 1]
        t_half = T // 2
        e_total = ef.sum() + 1e-12
        asym = float((ef[t_half:].sum() - ef[:t_half].sum()) / e_total)

        # 2. Normalised peak timing  [0, 1]
        peak_t = float(np.argmax(ef)) / T

        # 3. Coefficient of variation: std/mean  [0, ∞)
        coef_var = float(ef.std()) / e_mean

        # 4. Peak-to-mean ratio  [1, ∞)
        peak_mean = float(ef.max()) / e_mean

        # 5. Normalised delta: (last_third_mean - first_third_mean) / mean
        e_first = ef[:t3].mean()
        e_last  = ef[-t3:].mean()
        delta_n = (e_last - e_first) / e_mean

        feats.extend([asym, peak_t, coef_var, peak_mean, delta_n])

    # Range centroid dynamics — 6f (room-invariant: relative shift in bin space)
    centroid_shift = np.zeros(R)
    centroid_curve = np.zeros(R)
    for r in range(R):
        segs = [e[r, i*t3:(i+1)*t3, :] for i in range(3)]
        cs = []
        for seg in segs:
            eb = seg.sum(axis=0) + 1e-12
            cs.append(float(eb @ bins) / float(eb.sum()) / B)  # normalised 0-1
        centroid_shift[r] = cs[2] - cs[0]
        centroid_curve[r] = cs[1] - (cs[0] + cs[2]) / 2.0

    return np.concatenate([np.array(feats), centroid_shift, centroid_curve])
    # 5*3 + 3 + 3 = 21f


# ── Stage-1 (same as p48) ─────────────────────────────────────────────────────

def _frame_energy_stage1(filtered: np.ndarray) -> np.ndarray:
    per_radar = np.sum(filtered.astype(np.float64) ** 2, axis=2)
    return np.max(per_radar, axis=0)


def stage1_global(energy_s1, pct=97.0, min_dur=0.4, merge_gap=0.8):
    return energy_events_raw(energy_s1, pct=pct, min_dur=min_dur, merge_gap=merge_gap)


def stage1_adaptive(energy_s1, window_s=30.0, pct=97.0, min_dur=0.4, merge_gap=0.8):
    window_frames = max(3, int(window_s * RATE_HZ) | 1)
    local_thr = percentile_filter(energy_s1, pct, size=window_frames)
    global_floor = float(np.percentile(energy_s1, 30))
    threshold = np.maximum(local_thr, global_floor)
    mask = energy_s1 >= threshold
    if not np.any(mask):
        return []
    score = energy_s1 / (threshold + 1e-12)
    return runs_to_events(mask, score, min_dur, merge_gap)


def stage1_short(energy_s1, window_s=5.0, pct=90.0, min_dur=0.4, merge_gap=0.8):
    window_frames = max(3, int(window_s * RATE_HZ) | 1)
    local_thr = percentile_filter(energy_s1, pct, size=window_frames)
    global_floor = float(np.percentile(energy_s1, 30))
    threshold = np.maximum(local_thr, global_floor)
    mask = energy_s1 >= threshold
    if not np.any(mask):
        return []
    score = energy_s1 / (threshold + 1e-12)
    return runs_to_events(mask, score, min_dur, merge_gap)


def union_candidates(g, a, b, merge_gap=0.5):
    all_c = sorted(g + a + b, key=lambda c: c["t_start"])
    if not all_c:
        return []
    merged = [dict(all_c[0])]
    for c in all_c[1:]:
        prev = merged[-1]
        if c["t_start"] <= prev["t_end"] + merge_gap:
            prev["t_end"]   = max(prev["t_end"], c["t_end"])
            prev["score"]   = max(prev["score"], c["score"])
            prev["t_alarm"] = prev["t_start"]
        else:
            merged.append(dict(c))
    return merged


def extend_intervals(events, dur_s, delta_start=0.2, delta_end=0.0):
    out = []
    for ev in events:
        ns = max(0.0, ev["t_start"] - delta_start)
        ne = min(dur_s, ev["t_end"] + delta_end)
        if ne - ns >= 0.4:
            out.append({**ev, "t_start": ns, "t_end": ne, "t_alarm": ns})
    return out


# ── LightGBM (same pattern as p46) ───────────────────────────────────────────

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def _build_windows(filtered: np.ndarray, candidates: list) -> tuple[np.ndarray, list]:
    """Extract feature windows centred on each Stage-1 candidate."""
    T = filtered.shape[1]
    wins = iter_windows(T)
    if not wins:
        return np.empty((0, 21)), []

    win_size = wins[0][1] - wins[0][0]
    half     = win_size // 2

    X, meta = [], []
    for c in candidates:
        mid  = int((c["t_start"] + c["t_end"]) / 2 * RATE_HZ)
        s    = max(0, mid - half)
        e    = min(T, s + win_size)
        s    = max(0, e - win_size)
        if e - s < win_size // 2:
            continue
        block = filtered[:, s:e, :]
        if block.shape[1] < 10:
            continue
        X.append(window_features_clean(block))
        meta.append(c)

    return (np.array(X) if X else np.empty((0, 21))), meta


def _train_lgb(X: np.ndarray, y: np.ndarray, thr: float = 0.5):
    if not HAS_LGB or X.shape[0] == 0:
        return None, thr
    pos = y.sum()
    neg = len(y) - pos
    spw = max(1.0, neg / max(1, pos))
    m   = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, min_child_samples=5,
        n_jobs=-1, verbose=-1, scale_pos_weight=spw,
    )
    m.fit(X, y)
    return m, thr


def _predict(model, X: np.ndarray, thr: float):
    if model is None or X.shape[0] == 0:
        return np.zeros(X.shape[0])
    p = model.predict_proba(X)[:, 1]
    return (p >= thr).astype(int)


# ── OOF harness ───────────────────────────────────────────────────────────────

def run_stream_p49(
    cir: np.ndarray,
    model,
    thr: float,
    dur_s: float,
    ds: float = 0.2,
    de: float = 0.0,
) -> list:
    filtered   = chebyshev_bp(cir)
    e_s1       = _frame_energy_stage1(filtered)
    g_cands    = stage1_global(e_s1, pct=97.0)
    a_cands    = stage1_adaptive(e_s1, window_s=30.0, pct=97.0)
    b_cands    = stage1_short(e_s1, window_s=5.0, pct=90.0)
    candidates = union_candidates(g_cands, a_cands, b_cands)
    if not candidates:
        return []
    X, meta = _build_windows(filtered, candidates)
    if X.shape[0] == 0:
        return []
    preds = _predict(model, X, thr)
    kept  = [m for m, p in zip(meta, preds) if p == 1]
    kept  = extend_intervals(kept, dur_s, ds, de)
    kept  = dur_filter(kept, min_dur=0.4)
    kept  = gap_merge(kept, max_gap=1.5)
    return kept


def run_oof_p49(ds: float = 0.2, de: float = 0.0, thr: float = 0.5) -> list:
    from experiments.utils import STREAMS_TRAIN as ST
    gt_all = load_ground_truth(str(EVENTS_CSV), ignore_flagged=True)
    gt_by  = {}
    for ev in gt_all:
        if not ev["ignore"]:
            gt_by.setdefault(ev["stream_id"], []).append(ev)

    folds = build_folds()  # list of (train_ids, test_id)
    oof_rows = []

    for fold_idx, (train_ids, test_id) in enumerate(folds):
        k = int(test_id.split("_")[1])
        print(f"  Fold {fold_idx+1:02d}/13 test={test_id} ...", end=" ", flush=True)

        # Collect train windows
        X_tr, y_tr = [], []
        for sid in train_ids:
            ki  = int(sid.split("_")[1])
            p   = ST / f"stream_{ki:02d}.h5"
            if not p.exists():
                continue
            with h5py.File(p) as f:
                cir = f["cir"][:]
            T         = cir.shape[1]
            filtered  = chebyshev_bp(cir)
            e_s1      = _frame_energy_stage1(filtered)
            g_c       = stage1_global(e_s1)
            a_c       = stage1_adaptive(e_s1)
            b_c       = stage1_short(e_s1)
            cands     = union_candidates(g_c, a_c, b_c)
            if not cands:
                continue
            X_w, meta = _build_windows(filtered, cands)
            if X_w.shape[0] == 0:
                continue
            falls = gt_by.get(sid, [])
            for xi, (feat, m) in enumerate(zip(X_w, meta)):
                best_iou = max((temporal_iou(m["t_start"], m["t_end"],
                                             ev["t_start"], ev["t_end"])
                                for ev in falls), default=0.0)
                X_tr.append(feat)
                y_tr.append(1 if best_iou >= 0.1 else 0)

        if not X_tr:
            print("skip (no train windows)")
            continue

        X_tr_arr = np.array(X_tr)
        y_tr_arr = np.array(y_tr)
        model, _ = _train_lgb(X_tr_arr, y_tr_arr, thr)

        # Evaluate on test fold
        p_test = ST / f"stream_{k:02d}.h5"
        if not p_test.exists():
            print("skip (no test file)")
            continue
        with h5py.File(p_test) as f:
            cir_test = f["cir"][:]
        dur_s = cir_test.shape[1] / RATE_HZ
        evts  = run_stream_p49(cir_test, model, thr, dur_s, ds, de)
        for ev in evts:
            oof_rows.append({
                "stream_id": test_id,
                "t_start": round(ev["t_start"], 3),
                "t_end":   round(ev["t_end"], 3),
                "score":   round(ev["score"], 4),
                "t_alarm": round(ev["t_alarm"], 3),
            })
        print(f"done ({len(evts)} events)")

    return oof_rows


def sweep_threshold_on_oof(oof_rows: list, thrs=None) -> float:
    """Find threshold maximising OOF F1_mean."""
    if thrs is None:
        thrs = [round(x, 3) for x in np.arange(0.10, 0.55, 0.025)]
    from experiments.utils import STREAMS_TRAIN as ST
    durs = {}
    for k in range(13):
        p = ST / f"stream_{k:02d}.h5"
        if p.exists():
            with h5py.File(p) as f:
                T = f["cir"].shape[1]
            durs[f"train_{k:02d}"] = T / RATE_HZ

    gt = load_ground_truth(str(EVENTS_CSV), ignore_flagged=True)

    best_f1, best_thr = -1.0, 0.5
    for thr in thrs:
        rows = [r for r in oof_rows if r["score"] >= thr]
        if not rows:
            continue
        try:
            res = evaluate(gt, rows, durs)
            f1  = res["ranking_score"]
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        except Exception:
            pass
    print(f"OOF sweep: best_thr={best_thr:.3f} F1={best_f1:.4f}")
    return best_thr


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true",
                        help="Only run leakage probe, skip OOF")
    args = parser.parse_args()

    # ── Step 0: Leakage probe on new features ────────────────────────────────
    print("="*60)
    print("LEAKAGE PROBE — window_features_clean (21f, room-invariant)")
    print("="*60)

    import csv as csv_mod
    from abc2027.evaluate import load_ground_truth
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    COMP_CSV      = ROOT / "streams_composition.csv"
    STREAMS_TRAIN_P = ROOT / "streams" / "train"

    gt_all = load_ground_truth(str(EVENTS_CSV), ignore_flagged=True)
    gt_by  = {}
    for ev in gt_all:
        if not ev["ignore"]:
            gt_by.setdefault(ev["stream_id"], []).append(ev)

    with open(COMP_CSV, encoding="utf-8", newline="") as f:
        comp_rows = list(csv_mod.DictReader(f))

    rng = np.random.default_rng(42)
    pre_feats, pre_groups = [], []
    adl_feats, adl_groups = [], []

    print("Extracting windows for probe...")
    for k in range(13):
        sid = f"train_{k:02d}"
        p   = STREAMS_TRAIN_P / f"stream_{k:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        T        = cir.shape[1]
        filtered = chebyshev_bp(cir)
        wins     = iter_windows(T)
        if not wins:
            continue
        win_size = wins[0][1] - wins[0][0]

        for fall in gt_by.get(sid, []):
            t_end_pre   = fall["t_start"] - 0.5
            t_start_pre = t_end_pre - win_size / RATE_HZ
            if t_start_pre < 0:
                continue
            ps, pe = int(t_start_pre * RATE_HZ), int(t_end_pre * RATE_HZ)
            if pe > T or pe - ps < 10:
                continue
            pre_feats.append(window_features_clean(filtered[:, ps:pe, :]))
            pre_groups.append(k)

        for seg in [r for r in comp_rows if r["stream_id"] == sid and r["kind"] == "adl"]:
            ss, se = float(seg["t_start"]), float(seg["t_end"])
            if se - ss < win_size / RATE_HZ * 1.5:
                continue
            for _ in range(3):
                s = rng.integers(int(ss * RATE_HZ),
                                 max(int(ss * RATE_HZ) + 1, int(se * RATE_HZ) - win_size))
                e = s + win_size
                if e > T:
                    continue
                adl_feats.append(window_features_clean(filtered[:, s:e, :]))
                adl_groups.append(k)

    pre_feats  = np.array(pre_feats)
    adl_feats  = np.array(adl_feats)
    pre_groups = np.array(pre_groups)
    adl_groups = np.array(adl_groups)
    print(f"  Pre-fall: {len(pre_feats)} | ADL: {len(adl_feats)}")

    X = np.vstack([adl_feats, pre_feats])
    y = np.concatenate([np.zeros(len(adl_feats)), np.ones(len(pre_feats))]).astype(int)
    groups = np.concatenate([adl_groups, pre_groups])

    # Probe 1
    unique_g = np.unique(groups)
    oof_sc   = np.zeros(len(y))
    for g in unique_g:
        tr, te = groups != g, groups == g
        if tr.sum() < 5:
            continue
        m = lgb.LGBMClassifier(
            n_estimators=100, num_leaves=31, n_jobs=-1, verbose=-1,
            scale_pos_weight=max(1.0, (y[tr]==0).sum()/max(1,(y[tr]==1).sum())),
        )
        m.fit(X[tr], y[tr])
        oof_sc[te] = m.predict_proba(X[te])[:, 1]

    probe1_auc = float(roc_auc_score(y, oof_sc))
    p1_status  = "✅ PASS" if probe1_auc <= 0.65 else "❌ FAIL"
    print(f"\nProbe 1 AUC (all 21f): {probe1_auc:.4f}  {p1_status}")

    FGROUPS = {
        "temporal_asym":   slice(0, 3),
        "peak_t":          slice(3, 6),
        "coef_var":        slice(6, 9),
        "peak_mean_ratio": slice(9, 12),
        "delta_norm":      slice(12, 15),
        "centroid_shift":  slice(15, 18),
        "centroid_curve":  slice(18, 21),
    }
    group_aucs = {}
    for gname, gslice in FGROUPS.items():
        oof_g = np.zeros(len(y))
        for g in unique_g:
            tr, te = groups != g, groups == g
            if tr.sum() < 5:
                continue
            m = lgb.LGBMClassifier(
                n_estimators=100, num_leaves=31, n_jobs=-1, verbose=-1,
                scale_pos_weight=max(1.0, (y[tr]==0).sum()/max(1,(y[tr]==1).sum())),
            )
            m.fit(X[tr, gslice], y[tr])
            oof_g[te] = m.predict_proba(X[te, gslice])[:, 1]
        auc_g = float(roc_auc_score(y, oof_g))
        tag   = "✅" if auc_g <= 0.65 else ("⚠️" if auc_g <= 0.90 else "❌")
        print(f"  {tag} {gname:20s}: {auc_g:.4f}")
        group_aucs[gname] = round(auc_g, 4)

    # Probe 2
    km  = KMeans(n_clusters=3, random_state=42, n_init=10)
    cl  = km.fit_predict(pre_feats)
    sil = float(silhouette_score(pre_feats, cl))
    oof_pred = np.zeros(len(cl), dtype=int)
    for g in np.unique(pre_groups):
        tr, te = pre_groups != g, pre_groups == g
        if tr.sum() < 5:
            continue
        m2 = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1, verbose=-1)
        m2.fit(pre_feats[tr], cl[tr])
        oof_pred[te] = m2.predict(pre_feats[te])
    macro_f1 = float(f1_score(cl, oof_pred, average="macro", zero_division=0))
    p2_status = "✅ PASS" if macro_f1 <= 0.45 else "❌ FAIL"
    print(f"\nProbe 2 macro-F1: {macro_f1:.4f}  sil={sil:.4f}  {p2_status}")

    probe_pass = probe1_auc <= 0.65 and macro_f1 <= 0.45
    print(f"\n{'✅ PROBE PASSED — proceed to OOF' if probe_pass else '❌ PROBE FAILED — review features'}")

    probe_result = {
        "probe1_auc_all": round(probe1_auc, 4),
        "probe1_by_group": group_aucs,
        "probe2_silhouette": round(sil, 4),
        "probe2_macro_f1": round(macro_f1, 4),
        "verdict": "PASS" if probe_pass else "FAIL",
    }

    if args.probe_only or not probe_pass:
        print(json.dumps(probe_result, indent=2))
        sys.exit(0 if probe_pass else 1)

    # ── OOF ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("OOF EVALUATION (13-fold LOSO)")
    print("="*60)

    TMP_OOF = ROOT / "oof_pipeline_49_raw.csv"

    # Phase 1: collect raw OOF predictions at thr=0.0 (all candidates)
    raw_rows = run_oof_p49(ds=0.2, de=0.0, thr=0.0)

    # Phase 2: sweep threshold
    best_thr = sweep_threshold_on_oof(raw_rows)

    # Phase 3: filter at best threshold
    oof_rows = [r for r in raw_rows if r["score"] >= best_thr]

    # Save OOF
    oof_path = ROOT / "oof_pipeline_49.csv"
    with open(oof_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        w.writeheader()
        w.writerows(oof_rows)
    print(f"OOF saved: {oof_path}")

    # Score OOF
    from experiments.utils import STREAMS_TRAIN as ST
    durs = {}
    for k2 in range(13):
        p2 = ST / f"stream_{k2:02d}.h5"
        if p2.exists():
            with h5py.File(p2) as f2:
                durs[f"train_{k2:02d}"] = f2["cir"].shape[1] / RATE_HZ

    gt = load_ground_truth(str(EVENTS_CSV), ignore_flagged=True)
    res = evaluate(gt, oof_rows, durs)
    oof_f1  = res["ranking_score"]
    by_t    = res["by_theta"]
    oof_tp  = sum(v["tp"] for v in by_t.values()) // len(by_t)
    oof_fp  = sum(v["fp"] for v in by_t.values()) // len(by_t)
    oof_fn  = sum(v["fn"] for v in by_t.values()) // len(by_t)

    print(f"\nOOF F1_mean={oof_f1:.4f}")
    for t, v in by_t.items():
        print(f"  θ={t}: F1={v['f1']:.4f} TP={v['tp']} FP={v['fp']} FN={v['fn']}")

    # Val
    val_cirs = load_val()
    val_model_X, val_model_y = [], []
    for k2 in range(13):
        sid2 = f"train_{k2:02d}"
        p2   = ST / f"stream_{k2:02d}.h5"
        if not p2.exists():
            continue
        with h5py.File(p2) as f2:
            cir2 = f2["cir"][:]
        filtered2 = chebyshev_bp(cir2)
        e_s2      = _frame_energy_stage1(filtered2)
        cands2    = union_candidates(stage1_global(e_s2), stage1_adaptive(e_s2), stage1_short(e_s2))
        if not cands2:
            continue
        X_w, meta = _build_windows(filtered2, cands2)
        if X_w.shape[0] == 0:
            continue
        falls2 = gt_by.get(sid2, [])
        for feat, m in zip(X_w, meta):
            best = max((temporal_iou(m["t_start"], m["t_end"],
                                     ev["t_start"], ev["t_end"]) for ev in falls2), default=0.0)
            val_model_X.append(feat)
            val_model_y.append(1 if best >= 0.1 else 0)

    val_model, _ = _train_lgb(np.array(val_model_X), np.array(val_model_y), best_thr)

    val_rows = []
    for sid_v, cir_v in val_cirs.items():
        dur_v = cir_v.shape[1] / RATE_HZ
        evts  = run_stream_p49(cir_v, val_model, best_thr, dur_v, ds=0.2, de=0.0)
        for ev in evts:
            val_rows.append({
                "stream_id": sid_v,
                "t_start":   round(ev["t_start"], 3),
                "t_end":     round(ev["t_end"], 3),
                "score":     round(ev["score"], 4),
                "t_alarm":   round(ev["t_alarm"], 3),
            })

    val_path = save_sub(val_rows, "pipeline_49")
    val_res  = run_eval(val_path)
    val_f1   = val_res["f1_mean"]
    val_bt   = val_res["by_theta"]
    val_tp   = val_bt[0.1]["tp"]
    val_fp   = val_bt[0.1]["fp"]
    val_fn   = val_bt[0.1]["fn"]
    val_fa   = val_res["false_alarms_per_hour"]

    print(f"\nVal F1_mean={val_f1:.4f} TP={val_tp} FP={val_fp} FN={val_fn} FA/h={val_fa:.1f}")

    base_oof = 0.7423  # p46
    delta    = round(oof_f1 - base_oof, 4)
    decision = "KEEP" if oof_f1 >= base_oof - 0.01 else "DROP"

    log_entry = {
        "experiment_id": "pipeline_49",
        "phase": 6,
        "base": "pipeline_46",
        "description": f"Room-invariant 21f (shape/ratio only) + triple-union Stage-1; thr={best_thr:.3f} ds=0.2s de=0.0s",
        "components": [
            "room_invariant_21f",
            "dual_stage1_union",
            "adaptive_stage1_rolling30s",
            "adaptive_stage1_short5s",
            "asymmetric_interval_extension",
        ],
        "oof_f1_mean": round(oof_f1, 4),
        "oof_f1_01":   round(by_t[0.1]["f1"], 4),
        "oof_f1_03":   round(by_t[0.3]["f1"], 4),
        "oof_f1_05":   round(by_t[0.5]["f1"], 4),
        "oof_tp":  oof_tp,
        "oof_fp":  oof_fp,
        "oof_fn":  oof_fn,
        "delta_oof_vs_base": delta,
        "p_delta_gt0": None,
        "f1_mean":    round(val_f1, 4),
        "f1_01":      round(val_bt[0.1]["f1"], 4),
        "f1_03":      round(val_bt[0.3]["f1"], 4),
        "f1_05":      round(val_bt[0.5]["f1"], 4),
        "tp":  val_tp,
        "fp":  val_fp,
        "fn":  val_fn,
        "false_alarms_per_hour": val_fa,
        "delta_f1": round(val_f1 - 0.7278, 4),
        "probe_auc":      round(probe1_auc, 4),
        "probe_macro_f1": round(macro_f1, 4),
        "gate": None,
        "decision": decision,
        "notes": (
            f"room_invariant_21f; thr={best_thr:.3f}; "
            f"Δ_mean={delta:+.4f}; probe1_auc={probe1_auc:.4f}; probe2_f1={macro_f1:.4f}"
        ),
    }

    append_log(log_entry)
    print(f"\nDecision: {decision} (OOF={oof_f1:.4f} vs base={base_oof})")
    print("Log updated.")
