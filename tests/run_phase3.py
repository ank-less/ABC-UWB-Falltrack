"""Phase 3: Tối ưu & Ensemble (pipeline 15–20).
Base: pipeline_09  F1=0.5230  TP=48 FP=52 FN=19
Mục tiêu: F1_mean > 0.6

p15: p09 + p14 dual-model validation (intersection gate)
p16: p09 + train-CV-optimised LGB threshold
p17: p09 + CutMix/noise augmentation on training features
p18: p09 + rich feature set (27 features vs 9)
p19: rich features + CV threshold + dual-model gate + boundary refiner
p20: p19 + final val-set threshold sweep
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import cheby1, sosfilt, sosfilt_zi

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abc2027.baseline import (
    STEP_S, WINDOW, HOP,
    frame_energy, iter_windows, runs_to_events, window_features,
)
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations,
    temporal_iou, validate_predictions,
)

STREAMS_VAL   = ROOT / "streams" / "val"
STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV    = ROOT / "streams_events.csv"
LOG_FILE      = ROOT / "experiments_log.json"
SUB_DIR       = ROOT
RATE_HZ       = 50.0

# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_val():
    out = {}
    for p in sorted(STREAMS_VAL.glob("stream_*.h5")):
        sid = f"val_{int(p.stem.split('_')[1]):02d}"
        with h5py.File(p) as f:
            out[sid] = f["cir"][:]
    return out


def save_sub(rows, name):
    path = SUB_DIR / f"submission_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"]
        )
        w.writeheader()
        w.writerows(rows)
    return path


def run_eval(pred_path):
    durs  = stream_durations(STREAMS_VAL)
    preds = validate_predictions(pred_path, durs)
    gt    = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    result = evaluate(gt, preds, durs)
    rs = result["ranking_score"]
    bt = result["by_theta"]
    return {
        "f1_mean": round(rs["f1_mean"], 4),
        "f1_01":  round(bt["0.1"]["f1"], 4),
        "f1_03":  round(bt["0.3"]["f1"], 4),
        "f1_05":  round(bt["0.5"]["f1"], 4),
        "tp":     bt["0.3"]["tp"],
        "fp":     bt["0.3"]["fp"],
        "fn":     bt["0.3"]["fn"],
        "false_alarms_per_hour": round(rs["false_alarms_per_hour"], 1),
    }


def append_log(entry):
    log: list = []
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    ids = [e["experiment_id"] for e in log]
    if entry["experiment_id"] in ids:
        log[ids.index(entry["experiment_id"])] = entry
    else:
        log.append(entry)
    LOG_FILE.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def make_rows(sid, events):
    return [
        {
            "stream_id": sid,
            "t_start":  f"{e['t_start']:.3f}",
            "t_end":    f"{e['t_end']:.3f}",
            "score":    f"{e['score']:.6f}",
            "t_alarm":  f"{e['t_alarm']:.3f}",
        }
        for e in events
    ]


def run_pipeline(pid, desc, rows, prev_f1):
    name = f"pipeline_{pid:02d}"
    path = save_sub(rows, name)
    m = run_eval(path)
    delta = round(m["f1_mean"] - prev_f1, 4)
    entry = {
        "experiment_id": name,
        "description": desc,
        "components": [],
        **m,
        "delta_f1": delta,
        "notes": "",
    }
    append_log(entry)
    print(
        f"[{name}] F1_mean={m['f1_mean']:.4f}  "
        f"TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
        f"FA/h={m['false_alarms_per_hour']:.1f}  Δ={delta:+.4f}"
    )
    return m["f1_mean"]


# ─── Preprocessing ────────────────────────────────────────────────────────────

def chebyshev_bp(cir, low=0.5, high=15.0, order=4):
    nyq = RATE_HZ / 2.0
    sos = cheby1(order, 1.0, [low / nyq, high / nyq], btype="band", output="sos")
    radars, T, bins = cir.shape
    flat = cir.astype(np.float64).transpose(0, 2, 1).reshape(-1, T)
    zi_tmpl = sosfilt_zi(sos)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        zi = zi_tmpl * flat[i, 0]
        out[i], _ = sosfilt(sos, flat[i], zi=zi)
    return out.reshape(radars, bins, T).transpose(0, 2, 1).astype(np.float32)


def smooth(energy, sigma_s=0.3):
    return gaussian_filter1d(energy.astype(np.float64), sigma=sigma_s * RATE_HZ)


def dur_filter(events, min_dur=1.0):
    return [e for e in events if e["t_end"] - e["t_start"] >= min_dur]


def gap_merge(events, max_gap=1.5):
    if not events:
        return []
    evs = sorted(events, key=lambda e: e["t_start"])
    m = [dict(evs[0])]
    for e in evs[1:]:
        if e["t_start"] - m[-1]["t_end"] <= max_gap:
            m[-1]["t_end"] = max(m[-1]["t_end"], e["t_end"])
            m[-1]["score"] = max(m[-1]["score"], e["score"])
        else:
            m.append(dict(e))
    return m


def energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8):
    level = float(np.percentile(energy, pct))
    if level <= 0:
        return []
    return runs_to_events(energy >= level, energy / level, min_dur, merge_gap)


def refine_boundary(energy, t_start, t_end, stream_dur_s, context_s=2.0, frac=0.4):
    T = len(energy)
    s = max(0, int((t_start - context_s) * RATE_HZ))
    e = min(T, int((t_end + context_s) * RATE_HZ) + 1)
    seg = energy[s:e]
    if len(seg) == 0:
        return t_start, t_end
    peak_val = float(np.max(seg))
    if peak_val <= 0:
        return t_start, t_end
    idxs = np.flatnonzero(seg >= frac * peak_val)
    if len(idxs) == 0:
        return t_start, t_end
    new_start = max(0.0, (s + int(idxs[0])) / RATE_HZ)
    new_end   = min(stream_dur_s, (s + int(idxs[-1]) + 1) / RATE_HZ)
    if new_start >= new_end:
        return t_start, t_end
    return new_start, new_end


def gaussian_boundary(events, energy, stream_dur_s, k=1.2):
    T = len(energy)
    MIN_SIGMA = 0.3 * RATE_HZ
    refined = []
    for ev in events:
        ctx = 1.0
        s = max(0, int((ev["t_start"] - ctx) * RATE_HZ))
        e = min(T, int((ev["t_end"] + ctx) * RATE_HZ) + 1)
        seg = energy[s:e].copy()
        seg -= float(np.min(seg))
        tot = float(np.sum(seg))
        if tot <= 0:
            refined.append(ev)
            continue
        idx = np.arange(len(seg), dtype=np.float64)
        mu  = float(np.dot(idx, seg)) / tot
        var = float(np.dot((idx - mu) ** 2, seg)) / tot
        sig = float(np.sqrt(max(var, MIN_SIGMA ** 2)))
        ns  = max(0.0, (s + mu - k * sig) / RATE_HZ)
        ne  = min(stream_dur_s, (s + mu + k * sig) / RATE_HZ)
        if ne - ns < 0.5:
            refined.append(ev)
        else:
            refined.append({"t_start": ns, "t_end": ne,
                            "score": ev["score"], "t_alarm": ns})
    return refined


# ─── Rich feature set (27 features) ─────────────────────────────────────────

def window_features_rich(block: np.ndarray) -> np.ndarray:
    """27 features per window (3, W, 368):
    original 9 | temporal asymmetry 3 | peak timing 3 |
    temporal centroid 3 | centroid spread 3 | cross-radar corr 3 | range ratio 3
    """
    residual = block.astype(np.float64)
    residual -= np.mean(residual, axis=1, keepdims=True)   # remove static background
    frame = np.mean(residual * residual, axis=2)           # (3, T) per-frame energy
    T = frame.shape[1]
    half = T // 2

    # Original 9
    f_mean = frame.mean(axis=1)
    f_std  = frame.std(axis=1)
    f_max  = frame.max(axis=1)

    # Temporal asymmetry: second-half mean - first-half mean (3)
    f_delta = frame[:, half:].mean(axis=1) - frame[:, :half].mean(axis=1)

    # Normalised peak position (3)
    f_peak_t = np.argmax(frame, axis=1).astype(np.float64) / max(T - 1, 1)

    # Temporal centroid and spread (3 each)
    t_idx   = np.arange(T, dtype=np.float64)
    row_sum = frame.sum(axis=1) + 1e-12
    t_mean  = (frame @ t_idx) / row_sum / T            # normalised centroid (3,)
    t_var   = (frame @ (t_idx / T) ** 2) / row_sum - t_mean ** 2
    t_spread = np.sqrt(np.maximum(t_var, 0.0))         # (3,)

    # Cross-radar correlations (3)
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

    # Near-range vs far-range energy ratio (3)
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


# ─── Classifier builders ─────────────────────────────────────────────────────

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
    """Load all train windows, return X (N, F) and y (N,)."""
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


def load_classifier_base():
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
    # Gaussian noise at 3 intensities
    for scale in (0.04, 0.07, 0.11):
        aug.append(X_pos + rng.normal(0, 1, X_pos.shape) * stds * scale)
    # Multiplicative scale augmentation
    for lo, hi in ((0.78, 1.22), (0.85, 1.15)):
        s = rng.uniform(lo, hi, (len(X_pos), 1))
        aug.append(X_pos * s)
    # Feature-block mixup (first half from A, second half from B)
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
    print("Training rich LGB (27 features)...")
    model_cls, use_lgb = _try_lgb()
    X, y = _collect_windows(window_features_rich)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    print(f"  {len(y)} samples (pos={pos} neg={neg})")
    return _fit(model_cls, use_lgb, X, y)


def load_classifier_rich_aug():
    return load_classifier_augmented(
        feat_fn=window_features_rich, label="rich+aug (27 features)"
    )


# ─── Per-frame LGB score ──────────────────────────────────────────────────────

def lgb_frame_scores(cir, model, feat_fn=window_features):
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


# ─── Train-set threshold optimisation ────────────────────────────────────────

def find_best_threshold_on_train(model, feat_fn=window_features,
                                  thresholds=None):
    """Sweep threshold on TRAIN streams, return threshold with best F1.
    Note: model was trained on these streams → scores are biased upward for
    positives, but the relative ordering still reveals a useful operating point.
    """
    if thresholds is None:
        thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    print("  Pre-computing LGB scores on train streams...")
    cache = {}
    for p in sorted(STREAMS_TRAIN.glob("stream_*.h5")):
        sid = f"train_{int(p.stem.split('_')[1]):02d}"
        print(f"    {sid}...", flush=True)
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered  = chebyshev_bp(cir)
        energy    = smooth(frame_energy(filtered), sigma_s=0.3)
        scores    = lgb_frame_scores(filtered, model, feat_fn)
        cands     = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
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
            dur_s = T / RATE_HZ
            events = []
            for ev in cands:
                s = int(ev["t_start"] * RATE_HZ)
                e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
                if s >= e:
                    continue
                mx = float(np.max(scores[s:e]))
                if mx > t:
                    ev2 = dict(ev)
                    ev2["score"] = mx
                    ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                    ev2["t_start"] = ns; ev2["t_end"] = ne; ev2["t_alarm"] = ns
                    events.append(ev2)
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            all_rows.extend(make_rows(sid, events))

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


# ─── Pipeline 15: dual-model validation ──────────────────────────────────────

def p15(streams, model_base):
    """p09 events gated by p14-style Gaussian detection OR high LGB score."""
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores   = lgb_frame_scores(filtered, model_base)
        T        = filtered.shape[1]

        # p14-style Gaussian boundary events (stricter energy threshold)
        p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        events = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            if mx <= 0.1:
                continue
            p14_ok = any(
                min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                for pev in p14_events
            )
            if p14_ok or mx > 0.30:
                ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                events.append({"t_start": ns, "t_end": ne,
                               "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


# ─── Pipeline 16: train-CV-optimised threshold ───────────────────────────────

def p16(streams, model_base, threshold):
    """p09 pipeline with CV-tuned LGB threshold."""
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores   = lgb_frame_scores(filtered, model_base)
        T        = filtered.shape[1]
        events   = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            if mx > threshold:
                ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                events.append({"t_start": ns, "t_end": ne,
                               "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


# ─── Pipeline 17: augmented training ─────────────────────────────────────────

def p17(streams, model_aug):
    """Same p09 structure, CutMix+noise augmented model."""
    return p16(streams, model_aug, threshold=0.10)


# ─── Pipeline 18: rich features ──────────────────────────────────────────────

def p18(streams, model_rich):
    """27-feature LGB model, same p09 pipeline structure."""
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores   = lgb_frame_scores(filtered, model_rich, feat_fn=window_features_rich)
        T        = filtered.shape[1]
        events   = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            if mx > 0.10:
                ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
                events.append({"t_start": ns, "t_end": ne,
                               "score": mx, "t_alarm": ns})
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


# ─── Pipeline 19: full combination ───────────────────────────────────────────

def p19(streams, model_rich_aug, threshold):
    """Rich features + CV threshold + p14 dual gate + boundary refiner."""
    rows = []
    for sid, cir in streams.items():
        dur_s    = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy   = smooth(frame_energy(filtered), sigma_s=0.3)
        cands    = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores   = lgb_frame_scores(filtered, model_rich_aug, feat_fn=window_features_rich)

        p14_cands  = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        T = filtered.shape[1]
        # High-confidence threshold: events with score > 2.5× threshold kept regardless
        high_thr = min(threshold * 2.5, 0.50)

        events = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            if mx <= threshold * 0.6:   # hard reject very low scores
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


# ─── Pipeline 20: val-set threshold sweep ────────────────────────────────────

def p20_sweep(streams, model_rich_aug, sweep_thresholds):
    """Sweep threshold on val, return (best_threshold, best_rows, best_metrics)."""
    best_f1 = 0.0
    best_thr = sweep_thresholds[0]
    best_rows = []
    print("  Val threshold sweep (p20):")
    for t in sweep_thresholds:
        rows = p19(streams, model_rich_aug, threshold=t)
        path = save_sub(rows, "p20_sweep_tmp")
        m    = run_eval(path)
        print(f"    t={t:.2f} → F1={m['f1_mean']:.4f} "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']}", flush=True)
        if m["f1_mean"] > best_f1:
            best_f1  = m["f1_mean"]
            best_thr = t
            best_rows = rows
    import os
    try:
        os.remove(SUB_DIR / "submission_p20_sweep_tmp.csv")
    except Exception:
        pass
    print(f"  Best val threshold: t={best_thr:.2f} (F1={best_f1:.4f})")
    return best_thr, best_rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PHASE 3: Tối ưu & Ensemble  (pipeline 15–20)")
    print("Base: pipeline_09  F1=0.5230  TP=48 FP=52 FN=19")
    print("Mục tiêu: F1_mean > 0.6")
    print("=" * 80)

    streams = load_val()
    print(f"Loaded {len(streams)} val streams\n")

    BASE_F1 = 0.5230
    prev    = BASE_F1
    best_f1 = BASE_F1
    best_id = "pipeline_09"
    no_improve = 0

    # ── Train base classifier (used for p15, p16) ────────────────────────────
    print("\n── Training base classifier (p15, p16) ──")
    model_base = load_classifier_base()
    print()

    # ── Pipeline 15: dual-model validation ──────────────────────────────────
    print("── Pipeline 15: dual-model validation (p09 + p14 Gaussian gate) ──")
    rows = p15(streams, model_base)
    prev = run_pipeline(15,
        "p09_base + p14_Gaussian_gate(pct=98,k=1.2): keep if p14_confirms OR lgb>0.30",
        rows, prev)
    if prev > best_f1:
        best_f1 = prev; best_id = "pipeline_15"; no_improve = 0
    else:
        no_improve += 1
    if best_f1 > 0.6:
        print(f"\nEARLY STOP: F1 > 0.6 tại {best_id} ({best_f1:.4f})!")

    # ── Pipeline 16: CV-optimised threshold ─────────────────────────────────
    print("\n── Pipeline 16: train-optimised LGB threshold ──")
    print("  Finding optimal threshold on train streams (biased but fast)...")
    opt_thr = find_best_threshold_on_train(model_base)
    print(f"  Applying threshold={opt_thr:.2f} to val...")
    rows = p16(streams, model_base, threshold=opt_thr)
    prev = run_pipeline(16,
        f"p09_base + train_opt_threshold={opt_thr:.2f} + boundary_refiner",
        rows, prev)
    if prev > best_f1:
        best_f1 = prev; best_id = "pipeline_16"; no_improve = 0
    else:
        no_improve += 1
    if best_f1 > 0.6:
        print(f"\nEARLY STOP: F1 > 0.6 tại {best_id} ({best_f1:.4f})!")

    if no_improve >= 3:
        print("3 liên tiếp không cải thiện — vẫn chạy tiếp theo kế hoạch.")
        no_improve = 0

    # ── Train augmented classifier (p17) ─────────────────────────────────────
    print("\n── Training augmented LGB (9 features + noise/scale/mixup aug) ──")
    model_aug = load_classifier_augmented(feat_fn=window_features, label="base+aug")
    print()

    print("── Pipeline 17: CutMix/noise-augmented model ──")
    rows = p17(streams, model_aug)
    prev = run_pipeline(17,
        "p09 + feature-aug training (noise+scale+mixup, 9 features, thr=0.10)",
        rows, prev)
    if prev > best_f1:
        best_f1 = prev; best_id = "pipeline_17"; no_improve = 0
    else:
        no_improve += 1
    if best_f1 > 0.6:
        print(f"\nEARLY STOP: F1 > 0.6 tại {best_id} ({best_f1:.4f})!")

    if no_improve >= 3:
        print("3 liên tiếp không cải thiện — vẫn chạy tiếp theo kế hoạch.")
        no_improve = 0

    # ── Train rich classifier (p18, p19, p20) ────────────────────────────────
    print("\n── Training rich LGB (27 features) ──")
    model_rich = load_classifier_rich()
    print()

    print("── Pipeline 18: rich features (27) ──")
    rows = p18(streams, model_rich)
    prev = run_pipeline(18,
        "p09 + window_features_rich(27 features): temporal asymmetry, peak timing, cross-radar corr, range ratio",
        rows, prev)
    if prev > best_f1:
        best_f1 = prev; best_id = "pipeline_18"; no_improve = 0
    else:
        no_improve += 1
    if best_f1 > 0.6:
        print(f"\nEARLY STOP: F1 > 0.6 tại {best_id} ({best_f1:.4f})!")

    # ── Pipeline 19: best combination ────────────────────────────────────────
    print("\n── Training rich+aug LGB (27 features + augmentation) ──")
    model_rich_aug = load_classifier_rich_aug()
    print()

    # Use CV threshold from p16 but apply to p19 rich model; clamp to [0.10, 0.25]
    p19_thr = float(np.clip(opt_thr, 0.10, 0.25))
    print(f"── Pipeline 19: full combination (rich+aug + CV_thr={p19_thr:.2f} + dual gate) ──")
    rows = p19(streams, model_rich_aug, threshold=p19_thr)
    prev = run_pipeline(19,
        f"rich+aug(27f) + CV_thr={p19_thr:.2f} + p14_dual_gate + boundary_refiner",
        rows, prev)
    if prev > best_f1:
        best_f1 = prev; best_id = "pipeline_19"; no_improve = 0
    else:
        no_improve += 1
    if best_f1 > 0.6:
        print(f"\nEARLY STOP: F1 > 0.6 tại {best_id} ({best_f1:.4f})!")

    # ── Pipeline 20: final val-set threshold sweep ───────────────────────────
    print("\n── Pipeline 20: final val-set threshold sweep ──")
    sweep_thrs = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    best_sweep_thr, best_sweep_rows = p20_sweep(streams, model_rich_aug, sweep_thrs)
    prev = run_pipeline(20,
        f"p19 + val_sweep_best_thr={best_sweep_thr:.2f} (rich+aug+dual_gate)",
        best_sweep_rows, prev)
    if prev > best_f1:
        best_f1 = prev; best_id = "pipeline_20"
    if best_f1 > 0.6:
        print(f"\nEARLY STOP: F1 > 0.6 tại {best_id} ({best_f1:.4f})!")

    # ─── Final report ────────────────────────────────────────────────────────
    log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    all_ids = {f"pipeline_{i:02d}" for i in range(21)}
    all_entries = [e for e in log if e["experiment_id"] in all_ids]
    all_entries.sort(key=lambda e: int(e["experiment_id"].split("_")[1]))

    phase3_ids = {f"pipeline_{i:02d}" for i in range(15, 21)}
    phase3 = [e for e in all_entries if e["experiment_id"] in phase3_ids]

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP 21 PIPELINE")
    print("=" * 90)
    print(f"{'ID':<16} {'F1_mean':>8} {'F1_01':>7} {'F1_03':>7} {'F1_05':>7} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'FA/h':>6} {'Δ':>8}  Giai đoạn")
    print("-" * 90)
    overall_best_f1 = max(e["f1_mean"] for e in all_entries)
    for e in all_entries:
        pid = int(e["experiment_id"].split("_")[1])
        phase = "GĐ1" if pid <= 7 else ("GĐ2" if pid <= 14 else "GĐ3")
        marker = " ◄ BEST" if e["f1_mean"] == overall_best_f1 else ""
        print(
            f"{e['experiment_id']:<16} {e['f1_mean']:>8.4f} {e['f1_01']:>7.4f} "
            f"{e['f1_03']:>7.4f} {e['f1_05']:>7.4f} {e['tp']:>4} {e['fp']:>4} "
            f"{e['fn']:>4} {e['false_alarms_per_hour']:>6.1f} "
            f"{e.get('delta_f1', 0):>+8.4f}  {phase}{marker}"
        )
    print("=" * 90)

    # Contribution analysis
    print("\nTHÀNH PHẦN ĐÓNG GÓP:")
    contributions = [
        ("Chebyshev bandpass (p04 vs p00)", 0.4676 - 0.3151),
        ("LightGBM gate (p08 vs p04)", 0.5190 - 0.4676),
        ("Boundary refiner (p09 vs p08)", 0.5230 - 0.5190),
    ]
    phase3_best = max(phase3, key=lambda e: e["f1_mean"]) if phase3 else None
    if phase3_best:
        contributions.append((
            f"Phase 3 best ({phase3_best['experiment_id']}  vs p09)",
            phase3_best["f1_mean"] - 0.5230,
        ))
    for name, delta in contributions:
        bar = "+" * max(0, int(delta * 80))
        print(f"  {delta:+.4f}  {bar}  {name}")

    print("\n" + "=" * 90)
    global_best_id = max(all_entries, key=lambda e: e["f1_mean"])["experiment_id"]
    global_best_f1 = max(all_entries, key=lambda e: e["f1_mean"])["f1_mean"]
    print(f"Pipeline tốt nhất để nộp bài: {global_best_id}  (F1_mean = {global_best_f1:.4f})")
    if global_best_f1 >= 0.6:
        print("✓ ĐẠT MỤC TIÊU GĐ3: F1 mean ≥ 0.6")
    elif global_best_f1 >= 0.5:
        print("✓ Duy trì mục tiêu GĐ2 (F1 > 0.5), chưa đạt GĐ3 (F1 > 0.6)")
    else:
        print("✗ Chưa đạt F1 > 0.5")
        print("→ Xem báo cáo dưới đây để thảo luận hướng mới")
    print("=" * 90)

    # Final recommendations if F1 < 0.5
    if global_best_f1 < 0.5:
        print("\nBÁO CÁO KHI F1 < 0.5 — Gợi ý hướng mới:")
        print("  1. Neural network: CNN/LSTM trên raw CIR — học spatial-temporal patterns")
        print("  2. Contrastive learning: few-shot — học embedding phân biệt fall vs ADL")
        print("  3. Cross-stream normalization per-stream để loại bỏ environment bias")
        print("  4. Spectrogram features (STFT) thay vì time-domain energy features")
        print("  5. End-to-end model thay vì pipeline modular")


if __name__ == "__main__":
    main()
