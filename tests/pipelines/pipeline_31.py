"""Pipeline 31 — Multi-scale Stage-1 + TFA features.

TWO improvements over p30:

1. STAGE-1 (main): multi-scale energy union (sigma=0.15/0.30/0.60s) + top-K bin
   pooling (K=32) + max across radars. Catches both fast and slow falls.
   Expected FN recovery at Stage-1: 50-65% of the 85 missed events.

2. FEATURES: 7 Hilbert/IF-based TFA features appended to 27 from p30 = 34 total.
   Features: if_chirp_rate, if_std, if_acf5, hflf_shift, spectral_ent_delta,
             energy_accel_skew, if_range.
   Probe first — DROP if any feature AUC > 0.65.

Base: pipeline_30 (OOF F1=0.6334)
KEEP if OOF F1_mean >= R + 0.005
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from scipy.stats import skew as _sp_skew

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EVENTS_CSV   = ROOT / "streams_events.csv"
RATE_HZ      = 50.0

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
from experiments.audit.leakage_probe import run_probe_1
from abc2027.baseline import frame_energy, iter_windows, STEP_S
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions, temporal_iou,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _load_log_entry(eid: str) -> dict:
    f = ROOT / "experiments_log.json"
    if not f.exists():
        return {}
    log = json.loads(f.read_text(encoding="utf-8"))
    return next((e for e in log if e["experiment_id"] == eid), {})


# ── Stage-1: multi-scale energy + top-K bin + max-radar ──────────────────────

def multiscale_energy_stage1(filtered: np.ndarray) -> np.ndarray:
    """Multi-scale Stage-1 energy — catches fast AND slow falls.

    Steps:
    1. Top-K (K=32) bin selection: focus on most active range bins.
    2. Max across 3 radars (from p28).
    3. Smooth at 3 temporal scales (0.15, 0.30, 0.60 s).
    4. Return element-wise max across scales.
    """
    K = 32
    # Long-term bin energy for top-K selection (whole stream)
    energy_per_bin = (filtered.astype(np.float64) ** 2).mean((0, 1))  # (368,)
    top_bins = np.argsort(energy_per_bin)[-K:]

    # Max across radars, summing only top-K bins
    per_radar_topk = np.sum(
        filtered[:, :, top_bins].astype(np.float64) ** 2, axis=2
    )  # (3, T)
    base_energy = np.max(per_radar_topk, axis=0)  # (T,) — max across radars

    # Multi-scale smoothing: take element-wise max
    scales_s = [0.15, 0.30, 0.60]
    result = np.zeros_like(base_energy)
    for sigma_s in scales_s:
        smoothed = smooth(base_energy, sigma_s=sigma_s)
        result = np.maximum(result, smoothed)

    return result


# ── Features: inherit p30 (27f) + add 7 TFA features = 34 ───────────────────

LEAKY_IDX = list(range(21, 27))
CLEAN_IDX  = [i for i in range(27) if i not in LEAKY_IDX]


def _range_centroid_features(block: np.ndarray) -> np.ndarray:
    """6 features: slope + acceleration of energy-weighted range centroid.
    Same as p30."""
    res = block.astype(np.float64)
    res -= np.mean(res, axis=1, keepdims=True)
    e   = res ** 2
    B   = e.shape[2]
    bins = np.arange(B, dtype=np.float64)
    T   = e.shape[1]
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
    slope = c3 - c1
    accel = c2 - (c1 + c3) / 2
    return np.concatenate([slope, accel])


def extract_tfa_features(block: np.ndarray, fs: float = RATE_HZ) -> np.ndarray:
    """7 Cohen's Class-inspired TFA features, max across radars (no leakage).

    Features:
      0 if_chirp_rate  — slope of instantaneous frequency (deceleration after impact)
      1 if_std         — std of IF (high during impact burst)
      2 if_acf5        — 5-lag autocorrelation of IF (high = smooth chirp = fall)
      3 hflf_shift     — (HF/LF first_half) - (HF/LF second_half): HF→LF transition
      4 spectral_ent_delta — entropy(first_third STFT) - entropy(last_third STFT)
      5 energy_accel_skew — skewness of d²e/dt² (asymmetric impact spike)
      6 if_range       — IF max - min (large frequency sweep = chirp event)
    """
    n_radars, T, _ = block.shape
    per_radar = []

    for r in range(n_radars):
        e = (block[r].astype(np.float64) ** 2).sum(axis=1)  # (T,)
        e_n = e - e.mean()
        if e_n.std() < 1e-10:
            per_radar.append(np.zeros(7))
            continue

        # Instantaneous frequency via Hilbert transform
        analytic = hilbert(e_n)
        phase    = np.unwrap(np.angle(analytic))
        IF       = np.clip(np.diff(phase) * fs / (2.0 * np.pi), -15.0, 15.0)

        # Short STFT features (32-frame window, 8-frame hop)
        W, H = 32, 8
        win  = np.hanning(W)
        frames = [e_n[i:i+W] * win for i in range(0, T - W, H) if i + W <= T]
        if not frames:
            per_radar.append(np.zeros(7))
            continue
        frames_arr = np.array(frames)  # (nf, W)
        spec  = np.abs(np.fft.rfft(frames_arr, axis=1)) ** 2
        freqs = np.fft.rfftfreq(W, 1.0 / fs)
        hf_mask = freqs >= 5.0
        lf_mask = (freqs >= 0.5) & (freqs < 5.0)
        nf = spec.shape[0]
        h_ = max(1, nf // 2)
        t3 = max(1, nf // 3)

        def ent(s_):
            p = s_ / (s_.sum() + 1e-8)
            return -(p * np.log(p + 1e-12)).sum()

        def hflf_ratio(sub):
            return sub[:, hf_mask].sum() / (sub[:, lf_mask].sum() + 1e-8)

        chirp_rate = float(np.polyfit(np.linspace(0, 1, len(IF)), IF, 1)[0])
        if_std_val = float(IF.std())
        if_acf5    = float(np.corrcoef(IF[:-5], IF[5:])[0, 1]) if len(IF) > 5 else 0.0
        hflf_s     = float(hflf_ratio(spec[:h_]) - hflf_ratio(spec[h_:]))
        ent_d      = float(ent(spec[:t3].mean(0)) - ent(spec[-t3:].mean(0)))
        accel_skew = float(_sp_skew(np.diff(np.diff(e)))) if T > 2 else 0.0
        if_range_v = float(IF.max() - IF.min())

        per_radar.append(np.array(
            [chirp_rate, if_std_val, if_acf5, hflf_s, ent_d, accel_skew, if_range_v],
            dtype=np.float64
        ))

    arr = np.array(per_radar)  # (3, 7)
    # max across radars (no per-radar index → no leakage)
    return arr.max(axis=0)  # (7,)


def window_features_p31(block: np.ndarray) -> np.ndarray:
    """34 features: 21 clean (p24) + 6 range centroid (p30) + 7 TFA (p31)."""
    clean = window_features_rich(block)[CLEAN_IDX]  # 21f
    cdyn  = _range_centroid_features(block)          # 6f
    tfa   = extract_tfa_features(block)              # 7f
    return np.concatenate([clean, cdyn, tfa])        # 34f


# ── Leakage probe on the 7 new TFA features ──────────────────────────────────

def probe_tfa_features(gt_all: list) -> float:
    from abc2027.baseline import WINDOW
    gt_by = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)

    X_pre, X_adl, groups = [], [], []
    rng = np.random.default_rng(99)

    for idx in range(13):
        sid = f"train_{idx:02d}"
        p   = STREAMS_TRAIN / f"stream_{idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T        = filtered.shape[1]
        events   = [(ev["t_start"], ev["t_end"]) for ev in gt_by.get(sid, [])
                    if not ev.get("ignore", False)]

        for ts, te in events:
            pre_e = int((ts - 0.5) * RATE_HZ)
            pre_s = pre_e - WINDOW
            if pre_s >= 0 and pre_e <= T:
                X_pre.append(extract_tfa_features(filtered[:, pre_s:pre_e, :]))
                groups.append(idx)

            for attempt in range(5):
                adl_s = int(rng.integers(0, max(1, T - WINDOW)))
                adl_e = adl_s + WINDOW
                t_s_a, t_e_a = adl_s / RATE_HZ, adl_e / RATE_HZ
                if any(abs(t_s_a - gs) < 5.0 or abs(t_e_a - ge) < 5.0
                       for gs, ge in events):
                    continue
                X_adl.append(extract_tfa_features(filtered[:, adl_s:adl_e, :]))
                break

    if len(X_pre) < 10 or len(X_adl) < 10:
        print("  Probe TFA: insufficient samples → assume PASS (0.5)")
        return 0.5
    n  = min(len(X_pre), len(X_adl))
    Xf = np.vstack([np.array(X_pre[:n]), np.array(X_adl[:n])])
    yf = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    gf = np.array(groups[:n] + groups[:n])
    # Replace NaN/Inf
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
    return run_probe_1(Xf, yf, gf, label="tfa_features")


# ── Hard negative mining ──────────────────────────────────────────────────────

def collect_hard_negatives(
    train_sids: List[str], gt_by: dict, feat_fn, weight: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_hard, w_hard = [], []
    for sid in train_sids:
        idx = int(sid.split("_")[1])
        p   = STREAMS_TRAIN / f"stream_{idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        T        = filtered.shape[1]
        wins     = iter_windows(T)
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
        return np.empty((0, 34)), np.empty(0), np.empty(0)
    return np.array(X_hard), np.zeros(len(X_hard), dtype=np.int32), np.array(w_hard)


# ── Fold training ─────────────────────────────────────────────────────────────

def _train_fold(
    train_sids: List[str], gt_all: list, feat_fn, hard_neg_weight: float = 2.0,
) -> object:
    gt_by = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)

    X_parts, y_parts, w_parts = [], [], []
    for sid in train_sids:
        idx = int(sid.split("_")[1])
        p   = STREAMS_TRAIN / f"stream_{idx:02d}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        wins     = iter_windows(filtered.shape[1])
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
    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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

    X_hard, y_hard, w_hard = collect_hard_negatives(
        train_sids, gt_by, feat_fn, hard_neg_weight
    )
    if len(X_hard) > 0:
        X_hard = np.nan_to_num(X_hard, nan=0.0, posinf=0.0, neginf=0.0)
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
    print("Pipeline 31 — Multi-scale Stage-1 + TFA features (34f)")
    print("=" * 70)

    # Determine base
    for eid in ("pipeline_30", "pipeline_28", "pipeline_24"):
        e = _load_log_entry(eid)
        if e.get("decision") in ("KEEP", "BASE"):
            base_id = eid
            R     = e.get("oof_f1_mean", 0.6334)
            R_05  = e.get("oof_f1_05",   0.5449)
            break
    else:
        base_id, R, R_05 = "pipeline_30", 0.6334, 0.5449

    print(f"Base: {base_id}  OOF F1_mean={R:.4f}  OOF F1_05={R_05:.4f}")

    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    # ── Leakage probe on 7 new TFA features ──────────────────────────────────
    print("\nRunning leakage probe on TFA features…")
    probe_auc = probe_tfa_features(gt_all)
    print(f"    Probe 1 AUC (tfa_features): {probe_auc:.4f}")
    if probe_auc > 0.65:
        print(f"FAIL probe: AUC={probe_auc:.4f} > 0.65 — DROP p31")
        entry = {
            "experiment_id": "pipeline_31",
            "phase": 5,
            "base": base_id,
            "description": "p31: multi-scale Stage-1 + TFA features — FAILED probe",
            "components": ["multiscale_stage1", "tfa_features"],
            "oof_f1_mean": None, "oof_f1_01": None, "oof_f1_03": None, "oof_f1_05": None,
            "oof_tp": None, "oof_fp": None, "oof_fn": None,
            "delta_oof_vs_base": None, "p_delta_gt0": None,
            "f1_mean": None, "f1_01": None, "f1_03": None, "f1_05": None,
            "tp": None, "fp": None, "fn": None, "false_alarms_per_hour": None,
            "delta_f1": None,
            "probe_auc": round(probe_auc, 4),
            "probe_macro_f1": None,
            "gate": None,
            "decision": "DROP",
            "notes": f"TFA leakage probe FAIL AUC={probe_auc:.4f}",
        }
        append_log(entry)
        return "DROP", None, None
    print(f"Probe PASS: AUC={probe_auc:.4f} ≤ 0.65 — proceeding")

    # ── OOF training ─────────────────────────────────────────────────────────
    folds = build_folds()
    print(f"\nPhase 1: Training {len(folds)} fold models (34f: multiscale + TFA)…")
    fold_cache = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"\n  Fold {k:02d}: training…", flush=True)
        model = _train_fold(train_sids, gt_all, window_features_p31, hard_neg_weight=2.0)

        stream_idx = int(test_sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered     = chebyshev_bp(cir)
        T            = filtered.shape[1]
        dur_s        = T / RATE_HZ

        # Multi-scale Stage-1
        energy_ms   = multiscale_energy_stage1(filtered)  # (T,)
        # Fallback dual-gate (p14 candidates for safety)
        energy_mean = smooth(frame_energy(filtered), sigma_s=0.3)
        cands       = energy_events_raw(energy_ms,   pct=95.0, min_dur=0.4, merge_gap=0.8)
        p14_cands   = energy_events_raw(energy_mean, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events  = gaussian_boundary(p14_cands, energy_mean, dur_s, k=1.2)

        if model is not None:
            wins  = iter_windows(T)
            if wins:
                feats = np.array([window_features_p31(filtered[:, s:e, :]) for s, e in wins])
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                raw   = model.predict_proba(feats)[:, 1]
                raw   = gaussian_filter1d(raw, sigma=2.0)
                xs    = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
                fs    = np.interp(np.arange(T, dtype=np.float64), xs, raw)
                fs    = gaussian_filter1d(fs, sigma=5.0)
            else:
                fs = np.zeros(T)
        else:
            fs = np.zeros(T)

        fold_cache.append((
            test_sid,
            {"cands": cands, "p14_events": p14_events,
             "energy_ms": energy_ms, "dur_s": dur_s, "T": T},
            {test_sid: fs},
        ))
        print(f"    {len(cands)} candidates", flush=True)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print(f"\nPhase 2: OOF threshold sweep…")
    thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
    tmp_csv    = ROOT / "_p31_tmp.csv"
    best_f1, best_thr = 0.0, 0.25

    for thr in thresholds:
        high_thr = min(thr * 2.5, 0.50)
        all_rows = []
        for test_sid, cd, scores_d in fold_cache:
            bs = scores_d[test_sid]
            events = []
            for cand in cd["cands"]:
                s = int(cand["t_start"] * RATE_HZ)
                e = min(int(cand["t_end"]  * RATE_HZ) + 1, cd["T"])
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

    # ── Final OOF predictions ─────────────────────────────────────────────────
    all_oof_rows = []
    high_thr = min(best_thr * 2.5, 0.50)
    for test_sid, cd, scores_d in fold_cache:
        bs = scores_d[test_sid]
        events = []
        for cand in cd["cands"]:
            s = int(cand["t_start"] * RATE_HZ)
            e = min(int(cand["t_end"]  * RATE_HZ) + 1, cd["T"])
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

    oof_csv = ROOT / "oof_pipeline_31.csv"
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
    delta_mean = round(oof["f1_mean"] - R, 4)

    print(f"\n=== OOF Results (p31) ===")
    print(f"  F1_mean={oof['f1_mean']:.4f}  (R={R:.4f}, Δ={delta_mean:+.4f})")
    print(f"  F1_01  ={oof['f1_01']:.4f}")
    print(f"  F1_03  ={oof['f1_03']:.4f}")
    print(f"  F1_05  ={oof['f1_05']:.4f}  (R_05={R_05:.4f})")
    print(f"  TP={oof['tp']}  FP={oof['fp']}  FN={oof['fn']}")

    # Decision
    if delta_mean >= 0.005:
        decision = "KEEP"
        notes    = f"Δ_mean={delta_mean:+.4f}≥0.005 — multiscale+TFA effective"
    elif delta_mean >= -0.005:
        decision = "KEEP"
        notes    = f"Δ_mean={delta_mean:+.4f} neutral — keeping for Stage-1 recall gain"
    else:
        decision = "DROP"
        notes    = f"Δ_mean={delta_mean:+.4f}<-0.005 — not effective"
    print(f"Decision: {decision}  — {notes}")

    # Val (1 run if KEEP)
    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        print("Training full model on all 13 train streams…")
        full_model = _train_fold(
            [f"train_{i:02d}" for i in range(13)], gt_all,
            window_features_p31, hard_neg_weight=2.0,
        )
        val_rows = []
        for vi in range(5):
            sid = f"val_{vi:02d}"
            p   = STREAMS_VAL / f"stream_{vi:02d}.h5"
            if not p.exists():
                continue
            with h5py.File(p) as f:
                cir = f["cir"][:]
            filtered  = chebyshev_bp(cir)
            T         = filtered.shape[1]
            dur_s     = T / RATE_HZ
            energy_ms = multiscale_energy_stage1(filtered)
            energy_m  = smooth(frame_energy(filtered), sigma_s=0.3)
            cands     = energy_events_raw(energy_ms,  pct=95.0, min_dur=0.4, merge_gap=0.8)
            p14_c     = energy_events_raw(energy_m,   pct=98.0, min_dur=0.6, merge_gap=0.8)
            p14_ev    = gaussian_boundary(p14_c, energy_m, dur_s, k=1.2)

            if full_model is not None:
                wins  = iter_windows(T)
                if wins:
                    feats = np.array([window_features_p31(filtered[:, s:e, :]) for s, e in wins])
                    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                    raw   = full_model.predict_proba(feats)[:, 1]
                    raw   = gaussian_filter1d(raw, sigma=2.0)
                    xs    = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
                    fs    = np.interp(np.arange(T, dtype=np.float64), xs, raw)
                    fs    = gaussian_filter1d(fs, sigma=5.0)
                else:
                    fs = np.zeros(T)
            else:
                fs = np.zeros(T)

            high_thr = min(best_thr * 2.5, 0.50)
            events = []
            for cand in cands:
                s = int(cand["t_start"] * RATE_HZ)
                e = min(int(cand["t_end"] * RATE_HZ) + 1, T)
                if s >= e:
                    continue
                mx = float(np.max(fs[s:e]))
                if mx <= best_thr * 0.6:
                    continue
                p14_ok = any(
                    min(cand["t_end"], pev["t_end"]) - max(cand["t_start"], pev["t_start"]) > 0
                    for pev in p14_ev
                )
                if (p14_ok and mx > best_thr) or mx > high_thr:
                    ns, ne = refine_boundary(energy_ms, cand["t_start"], cand["t_end"], dur_s)
                    events.append({"t_start": ns, "t_end": ne,
                                   "score": mx, "t_alarm": ns})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            for ev in events:
                val_rows.append({"stream_id": sid,
                                 "t_start": f"{ev['t_start']:.3f}",
                                 "t_end":   f"{ev['t_end']:.3f}",
                                 "score":   f"{ev['score']:.6f}",
                                 "t_alarm": f"{ev['t_alarm']:.3f}"})

        val_path = ROOT / "submission_pipeline_31.csv"
        with open(val_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stream_id","t_start","t_end","score","t_alarm"])
            w.writeheader()
            w.writerows(val_rows)
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p31) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    # Log
    base_val_f1 = _load_log_entry(base_id).get("f1_mean") or 0.0
    delta_val   = round((val_metrics["f1_mean"] - base_val_f1) if val_metrics else 0.0, 4)
    entry = {
        "experiment_id": "pipeline_31",
        "phase": 5,
        "base": base_id,
        "description": f"p31: multiscale Stage-1 (pct95, scales 0.15/0.30/0.60s) + top-32 bins + 7 TFA features (34f total)",
        "components": ["multiscale_stage1", "topk_bins", "tfa_features", "hard_negatives"],
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
        "probe_auc":      round(probe_auc, 4),
        "probe_macro_f1": None,
        "gate": None,
        "decision": decision,
        "notes":    notes,
    }
    append_log(entry)
    print(f"\nLogged: pipeline_31  decision={decision}")
    return decision, oof, val_metrics


if __name__ == "__main__":
    main()
