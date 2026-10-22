"""LightGBM fall classifier: training, feature extraction, and scoring."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.baseline import (
    STEP_S, frame_energy, iter_windows, runs_to_events, window_features,
)
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations,
    temporal_iou, validate_predictions,
)
from experiments.preprocessing.chebyshev_filter import chebyshev_bp

STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV    = ROOT / "streams_events.csv"
SUB_DIR       = ROOT
RATE_HZ       = 50.0


# ── Rich 27-feature extractor ──────────────────────────────────────────────────

def window_features_rich(block: np.ndarray) -> np.ndarray:
    """27 features per window (3, W, 368):
    original 9 | temporal asymmetry 3 | peak timing 3 |
    temporal centroid 3 | centroid spread 3 | cross-radar corr 3 | range ratio 3
    """
    residual = block.astype(np.float64)
    residual -= np.mean(residual, axis=1, keepdims=True)
    frame = np.mean(residual * residual, axis=2)  # (3, T) per-frame energy
    T    = frame.shape[1]
    half = T // 2

    f_mean  = frame.mean(axis=1)
    f_std   = frame.std(axis=1)
    f_max   = frame.max(axis=1)
    f_delta = frame[:, half:].mean(axis=1) - frame[:, :half].mean(axis=1)
    f_peak_t = np.argmax(frame, axis=1).astype(np.float64) / max(T - 1, 1)

    t_idx   = np.arange(T, dtype=np.float64)
    row_sum = frame.sum(axis=1) + 1e-12
    t_mean  = (frame @ t_idx) / row_sum / T
    t_var   = (frame @ (t_idx / T) ** 2) / row_sum - t_mean ** 2
    t_spread = np.sqrt(np.maximum(t_var, 0.0))

    def safe_corr(a, b):
        sa, sb = a.std(), b.std()
        if sa < 1e-10 or sb < 1e-10:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    cross = np.array([
        safe_corr(frame[0], frame[1]),
        safe_corr(frame[0], frame[2]),
        safe_corr(frame[1], frame[2]),
    ])

    bins = block.shape[2]
    near_e = np.mean(residual[:, :, :bins // 3] ** 2, axis=(1, 2))
    far_e  = np.mean(residual[:, :, 2 * bins // 3:] ** 2, axis=(1, 2))
    range_ratio = near_e / (far_e + 1e-12)

    return np.concatenate([
        f_mean, f_std, f_max,
        f_delta, f_peak_t,
        t_mean, t_spread,
        cross,
        range_ratio,
    ])


# ── Internal helpers ───────────────────────────────────────────────────────────

def _try_lgb():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier, True
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier, False


def _fit(model_cls, use_lgb, X, y, extra=None):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if use_lgb:
        kw = dict(n_estimators=500, num_leaves=63, min_child_samples=5,
                  scale_pos_weight=neg / max(pos, 1), n_jobs=-1, verbose=-1)
        if extra:
            kw.update(extra)
        m = model_cls(**kw)
    else:
        m = model_cls(n_estimators=400, class_weight="balanced",
                      n_jobs=-1, random_state=42)
    m.fit(X, y)
    return m


def _collect_windows(feat_fn):
    """Load all train streams and extract features + labels."""
    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by: dict[str, list] = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append((ev["t_start"], ev["t_end"]))

    X_parts, y_parts = [], []
    for p in sorted(STREAMS_TRAIN.glob("stream_*.h5")):
        sid = f"train_{int(p.stem.split('_')[1]):02d}"
        print(f"  [{sid}]...", end=" ", flush=True)
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        wins = iter_windows(filtered.shape[1])
        if not wins:
            print("skip")
            continue
        feats = np.array([feat_fn(filtered[:, s:e, :]) for s, e in wins])
        gt_ev = gt_by.get(sid, [])
        labels = np.array(
            [int(any(temporal_iou(ws * STEP_S, we * STEP_S, gs, ge) >= 0.1
                     for gs, ge in gt_ev))
             for ws, we in wins],
            dtype=np.int32,
        )
        X_parts.append(feats)
        y_parts.append(labels)
        print(f"pos={int(labels.sum())}/{len(labels)}", flush=True)

    return np.vstack(X_parts), np.concatenate(y_parts)


# ── Classifier factory functions ───────────────────────────────────────────────

def load_classifier():
    """Train base LGB (9 features, 400 estimators) — used in p06–p11."""
    try:
        import lightgbm as lgb
        model_cls = lgb.LGBMClassifier
        use_lgb = True
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model_cls = RandomForestClassifier
        use_lgb = False

    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by: dict[str, list] = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append((ev["t_start"], ev["t_end"]))

    X_parts, y_parts = [], []
    for p in sorted(STREAMS_TRAIN.glob("stream_*.h5")):
        sid = f"train_{int(p.stem.split('_')[1]):02d}"
        print(f"  [{sid}] ...", flush=True, end=" ")
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        wins = iter_windows(filtered.shape[1])
        if not wins:
            print("skip (short)")
            continue
        feats = np.array([window_features(filtered[:, s:e, :]) for s, e in wins])
        gt_ev = gt_by.get(sid, [])
        labels = np.zeros(len(wins), dtype=np.int32)
        for i, (ws, we) in enumerate(wins):
            t_s, t_e = ws * STEP_S, we * STEP_S
            if any(temporal_iou(t_s, t_e, gs, ge) >= 0.1 for gs, ge in gt_ev):
                labels[i] = 1
        X_parts.append(feats)
        y_parts.append(labels)
        print(f"pos={int(labels.sum())}/{len(labels)}", flush=True)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    print(f"  Total: {len(y)} samples (pos={pos}, neg={neg}, ratio={neg//max(pos,1)}:1)")

    if use_lgb:
        model = model_cls(
            n_estimators=400,
            num_leaves=31,
            scale_pos_weight=neg / max(pos, 1),
            n_jobs=-1,
            verbose=-1,
        )
    else:
        model = model_cls(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    model.fit(X, y)
    return model


def load_classifier_base():
    """Train high-capacity LGB (9 features, 500 estimators) — used in p15, p16."""
    print("Training base LGB (9 features, window_features)...")
    model_cls, use_lgb = _try_lgb()
    X, y = _collect_windows(window_features)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    print(f"  {len(y)} samples (pos={pos} neg={neg} ratio={neg//max(pos,1)}:1)")
    return _fit(model_cls, use_lgb, X, y)


def load_classifier_augmented(feat_fn=window_features, label="base+aug"):
    """Train with feature-level augmentation for positive samples."""
    print(f"Training augmented LGB ({label})...")
    model_cls, use_lgb = _try_lgb()
    X, y = _collect_windows(feat_fn)

    pos_idx = np.where(y == 1)[0]
    X_pos   = X[pos_idx]
    rng     = np.random.default_rng(42)
    stds    = X_pos.std(axis=0) + 1e-8

    aug = [X_pos]
    for scale in (0.04, 0.07, 0.11):
        aug.append(X_pos + rng.normal(0, 1, X_pos.shape) * stds * scale)
    for lo, hi in ((0.78, 1.22), (0.85, 1.15)):
        s = rng.uniform(lo, hi, (len(X_pos), 1))
        aug.append(X_pos * s)
    n = len(X_pos)
    half = X_pos.shape[1] // 2
    perm = rng.permutation(n)
    aug.append(np.concatenate([X_pos[:, :half], X_pos[perm, half:]], axis=1))

    X_aug_pos = np.vstack(aug)
    X_neg = X[y == 0]
    y_aug = np.ones(len(X_aug_pos), dtype=np.int32)
    y_neg = np.zeros(len(X_neg), dtype=np.int32)
    X_full = np.vstack([X_neg, X_aug_pos])
    y_full = np.concatenate([y_neg, y_aug])
    pos_f, neg_f = int(y_full.sum()), int(len(y_full) - y_full.sum())
    print(f"  {len(y_full)} samples after aug (pos={pos_f} neg={neg_f})")
    return _fit(model_cls, use_lgb, X_full, y_full)


def load_classifier_rich():
    """Train LGB on 27-feature rich set."""
    print("Training rich LGB (27 features)...")
    model_cls, use_lgb = _try_lgb()
    X, y = _collect_windows(window_features_rich)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    print(f"  {len(y)} samples (pos={pos} neg={neg})")
    return _fit(model_cls, use_lgb, X, y)


def load_classifier_rich_aug():
    """Train LGB on 27-feature rich set with augmentation."""
    return load_classifier_augmented(
        feat_fn=window_features_rich, label="rich+aug (27 features)"
    )


# ── Per-frame score interpolation ─────────────────────────────────────────────

def lgb_frame_scores(cir, model, feat_fn=window_features):
    """Compute smoothed per-frame LGB probability, interpolated from sliding windows."""
    T = cir.shape[1]
    wins = iter_windows(T)
    if not wins:
        return np.zeros(T)
    feats  = np.array([feat_fn(cir[:, s:e, :]) for s, e in wins])
    scores = model.predict_proba(feats)[:, 1]
    scores = gaussian_filter1d(scores, sigma=2.0)
    xs     = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
    full   = np.interp(np.arange(T, dtype=np.float64), xs, scores)
    return gaussian_filter1d(full, sigma=5.0)


# ── Train-set threshold sweep ──────────────────────────────────────────────────

def find_best_threshold_on_train(model, feat_fn=window_features, thresholds=None):
    """Sweep threshold on TRAIN streams, return threshold with best F1.

    Note: model trained on these streams → scores biased upward for positives.
    """
    from experiments.utils import energy_events_raw, RATE_HZ as R
    from experiments.postprocessing.duration_filter import dur_filter
    from experiments.postprocessing.gap_merger import gap_merge
    from experiments.postprocessing.boundary_refiner import refine_boundary
    from experiments.postprocessing.temporal_smoothing import smooth

    if thresholds is None:
        thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    print("  Pre-computing LGB scores on train streams...")
    cache = {}
    for p in sorted(STREAMS_TRAIN.glob("stream_*.h5")):
        sid = f"train_{int(p.stem.split('_')[1]):02d}"
        print(f"    {sid}...", flush=True)
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        scores   = lgb_frame_scores(filtered, model, feat_fn)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        cache[sid] = (energy, scores, cands, cir.shape[1])

    durs_train = stream_durations(STREAMS_TRAIN)
    gt_train   = [e for e in load_ground_truth(EVENTS_CSV, ignore_flagged=True)
                  if e["stream_id"].startswith("train_")]
    tmp = SUB_DIR / "_tmp_thr_sweep.csv"

    best_f1, best_t = 0.0, 0.10
    print("  Sweeping threshold on train:")
    for t in thresholds:
        all_rows = []
        for sid, (energy, scores, cands, T) in cache.items():
            dur_s = T / R
            events = []
            for ev in cands:
                s = int(ev["t_start"] * R)
                e = min(int(ev["t_end"] * R) + 1, T)
                if s >= e:
                    continue
                mx = float(np.max(scores[s:e]))
                if mx > t:
                    ev2 = dict(ev)
                    ev2["score"] = mx
                    ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                    ev2["t_start"] = ns
                    ev2["t_end"]   = ne
                    ev2["t_alarm"] = ns
                    events.append(ev2)
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            for ev in events:
                all_rows.append({
                    "stream_id": sid,
                    "t_start":  f"{ev['t_start']:.3f}",
                    "t_end":    f"{ev['t_end']:.3f}",
                    "score":    f"{ev['score']:.6f}",
                    "t_alarm":  f"{ev['t_alarm']:.3f}",
                })

        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"]
            )
            w.writeheader()
            w.writerows(all_rows)

        try:
            preds  = validate_predictions(tmp, durs_train)
            result = evaluate(gt_train, preds, durs_train)
            f1     = round(result["ranking_score"]["f1_mean"], 4)
        except Exception as ex:
            f1 = 0.0
            print(f"      eval error at t={t}: {ex}")

        print(f"    t={t:.2f} → train_F1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    tmp.unlink(missing_ok=True)
    print(f"  Best train threshold: t={best_t:.2f} (train_F1={best_f1:.4f})")
    return best_t
