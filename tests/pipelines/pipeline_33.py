"""Pipeline 33: Tucker-mode features via numpy SVD (no tensorly).

Tucker decomposition of (3, T, 368) block:
  - Mode-1 factor (radar, r=2): unfolding → SVD
  - Mode-3 factor (range, s=8): unfolding → SVD
  - Core tensor G = U1.T @ block_unfolded @ U3 (r × q × s projected)
Features (10):
  1-2:  radar singular values (energy per component)
  3-10: range factor 8 loadings (dominant range pattern)

Not leaky because:
  - Structural patterns, not time-series dynamics
  - No absolute range positions (factor loadings are relative)
  - No cross-event or cross-stream normalization

Probe: AUC must be <= 0.65 before OOF.
Base: p32 (if KEEP) else p30
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
from experiments.audit.leakage_probe import run_probe_1
from abc2027.baseline import frame_energy, iter_windows, STEP_S, WINDOW
from abc2027.evaluate import (
    evaluate, load_ground_truth, stream_durations, validate_predictions, temporal_iou,
)


# ── Range centroid dynamics (same as p30/p32) ─────────────────────────────────

def range_centroid_features(block: np.ndarray) -> np.ndarray:
    """6 features: slope + acceleration of energy-weighted range centroid per radar."""
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


# ── Multi-scale Stage-1 (same as p32) ────────────────────────────────────────

def multiscale_energy_stage1(filtered: np.ndarray) -> np.ndarray:
    K = 32
    energy_per_bin = (filtered.astype(np.float64) ** 2).mean(axis=(0, 1))
    top_bins = np.argsort(energy_per_bin)[-K:]
    per_radar_topk = np.sum(filtered[:, :, top_bins].astype(np.float64) ** 2, axis=2)
    base_energy = np.max(per_radar_topk, axis=0)
    scales_s = [0.15, 0.30, 0.60]
    result = np.zeros_like(base_energy)
    for sigma_s in scales_s:
        result = np.maximum(result, smooth(base_energy, sigma_s=sigma_s))
    return result


# ── Tucker-mode features via numpy SVD ───────────────────────────────────────

def tucker_features(block: np.ndarray, r_radar: int = 2, s_range: int = 8) -> np.ndarray:
    """10 Tucker-mode features from SVD of tensor unfoldings.

    Block shape: (3, T, B).
    Mode-1 unfolding: (3, T*B) → SVD → top-r_radar singular values + radar factor
    Mode-3 unfolding: (B, 3*T) → SVD → top-s_range singular values

    Features (10):
      [0:r_radar]  — radar singular values (normalised by total energy)
      [r_radar:r_radar+s_range] — range singular values (normalised)

    Structural, not temporal-dynamic → lower leakage risk than IF/chirp features.
    """
    X = block.astype(np.float64)
    R, T, B = X.shape
    total_energy = np.sqrt(np.sum(X ** 2)) + 1e-12

    # Mode-1 unfolding: (R, T*B)
    X1 = X.reshape(R, T * B)
    try:
        U1, s1, _ = np.linalg.svd(X1, full_matrices=False)
    except np.linalg.LinAlgError:
        s1 = np.zeros(R)
    s1_norm = s1[:r_radar] / total_energy

    # Mode-3 unfolding: (B, R*T)
    X3 = X.transpose(2, 0, 1).reshape(B, R * T)
    try:
        _, s3, _ = np.linalg.svd(X3, full_matrices=False)
    except np.linalg.LinAlgError:
        s3 = np.zeros(B)
    s3_norm = s3[:s_range] / total_energy

    # Pad if SVD returned fewer values
    if len(s1_norm) < r_radar:
        s1_norm = np.pad(s1_norm, (0, r_radar - len(s1_norm)))
    if len(s3_norm) < s_range:
        s3_norm = np.pad(s3_norm, (0, s_range - len(s3_norm)))

    return np.concatenate([s1_norm, s3_norm])  # (r_radar + s_range,) = 10f


def window_features_p33(block: np.ndarray) -> np.ndarray:
    """37 features: 21 clean + 6 range_centroid_dynamics + 10 Tucker."""
    clean  = window_features_rich(block)[CLEAN_INDICES]  # 21f
    dynf   = range_centroid_features(block)              # 6f
    tucker = tucker_features(block)                      # 10f
    return np.concatenate([clean, dynf, tucker])         # 37f


# ── Leakage probe on Tucker features ─────────────────────────────────────────

def probe_tucker_features(gt_all: list) -> float:
    """Probe 1 on Tucker features only. Returns AUC."""
    gt_by = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append(ev)

    X_pre, X_adl, groups = [], [], []
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
                feat = tucker_features(filtered[:, pre_s:pre_e, :])
                X_pre.append(feat)
                groups.append(idx)

            for attempt in range(5):
                rng_t = np.random.default_rng(idx * 1000 + int(ts) + attempt)
                adl_s = int(rng_t.integers(0, max(1, T - WINDOW)))
                adl_e = adl_s + WINDOW
                t_s_adl, t_e_adl = adl_s / RATE_HZ, adl_e / RATE_HZ
                if any(abs(t_s_adl - gs) < 5.0 or abs(t_e_adl - ge) < 5.0
                       for gs, ge in events):
                    continue
                X_adl.append(tucker_features(filtered[:, adl_s:adl_e, :]))
                break

    if not X_pre or not X_adl:
        print("  Probe: insufficient samples, skipping")
        return 0.5

    n = min(len(X_pre), len(X_adl))
    Xf = np.vstack([X_pre[:n], X_adl[:n]])
    yf = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    gf = np.concatenate([groups[:n], groups[:n]])
    return run_probe_1(Xf, yf, gf, label="tucker_features")


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
    n_feat = 37
    if not X_hard:
        return np.empty((0, n_feat)), np.empty(0), np.empty(0)
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
        wins = iter_windows(filtered.shape[1])
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
    print("Pipeline 33 — Tucker-mode SVD features (37f: 21+6+10)")
    print("=" * 70)

    # Base: p32 if KEEP, else p30
    for eid in ("pipeline_32", "pipeline_30", "pipeline_28", "pipeline_24"):
        e = _load_log_entry(eid)
        if e.get("decision") in ("KEEP", "BASE"):
            base_id = eid
            R    = e.get("oof_f1_mean", 0.6334)
            R_05 = e.get("oof_f1_05",  0.5449)
            break
    else:
        base_id, R, R_05 = "pipeline_30", 0.6334, 0.5449

    print(f"Base: {base_id}  OOF F1_mean={R:.4f}  OOF F1_05={R_05:.4f}")
    print("Features: 37f = 21 clean + 6 centroid_dyn + 10 Tucker (SVD r=2 radar, s=8 range)")

    gt_all   = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs     = stream_durations(STREAMS_TRAIN)

    # ── Leakage probe on Tucker features ─────────────────────────────────────
    print("\nRunning leakage probe on Tucker features…")
    probe_auc = probe_tucker_features(gt_all)
    if probe_auc > 0.65:
        print(f"FAIL probe: AUC={probe_auc:.4f} > 0.65 — Tucker features are leaky, DROP p33")
        entry = {
            "experiment_id": "pipeline_33",
            "phase": 5,
            "base": base_id,
            "description": "Tucker-mode SVD features — FAILED leakage probe",
            "components": ["tucker_svd_features"],
            "oof_f1_mean": None, "oof_f1_01": None, "oof_f1_03": None, "oof_f1_05": None,
            "oof_tp": None, "oof_fp": None, "oof_fn": None,
            "delta_oof_vs_base": None, "p_delta_gt0": None,
            "f1_mean": None, "f1_01": None, "f1_03": None, "f1_05": None,
            "tp": None, "fp": None, "fn": None, "false_alarms_per_hour": None,
            "delta_f1": None,
            "probe_auc": round(probe_auc, 4), "probe_macro_f1": None,
            "gate": None, "decision": "DROP",
            "notes": f"Leakage probe FAIL: AUC={probe_auc:.4f}>0.65",
        }
        append_log(entry)
        return "DROP", None, None
    print(f"Probe PASS: AUC={probe_auc:.4f} ≤ 0.65 — proceeding")

    # ── OOF training ─────────────────────────────────────────────────────────
    STAGE1_PCT = 95.0
    folds = build_folds()
    print(f"\nPhase 1: Training {len(folds)} fold models (37f)…")
    fold_cache = []

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"\n  Fold {k:02d}: training…", flush=True)
        model = _train_fold(train_sids, gt_all, window_features_p33, hard_neg_weight=2.0)

        stream_idx = int(test_sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered   = chebyshev_bp(cir)
        T          = filtered.shape[1]
        dur_s      = T / RATE_HZ
        energy_s1  = multiscale_energy_stage1(filtered)
        energy_p14 = smooth(frame_energy(filtered), sigma_s=0.3)
        cands      = energy_events_raw(energy_s1,  pct=STAGE1_PCT, min_dur=0.4, merge_gap=0.8)
        p14_cands  = energy_events_raw(energy_p14, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy_p14, dur_s, k=1.2)

        if model is not None:
            wins  = iter_windows(T)
            feats = np.array([window_features_p33(filtered[:, s:e, :]) for s, e in wins]) if wins else None
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
             "energy_s1": energy_s1, "dur_s": dur_s, "T": T},
            {test_sid: fs},
        ))
        print(f"    {len(cands)} candidates", flush=True)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print(f"\nPhase 2: OOF threshold sweep…")
    thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]
    tmp_csv = ROOT / "_p33_tmp.csv"
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
                ns, ne = refine_boundary(cd["energy_s1"], cand["t_start"],
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

    oof_csv = ROOT / "oof_pipeline_33.csv"
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

    print(f"\n=== OOF Results (p33) ===")
    print(f"  F1_mean={oof['f1_mean']:.4f}  Δ={delta_mean:+.4f}")
    print(f"  F1_05  ={oof['f1_05']:.4f}  Δ_05={delta_05:+.4f}")
    print(f"  TP={oof['tp']}  FP={oof['fp']}  FN={oof['fn']}")

    if delta_mean >= 0.005:
        decision = "KEEP"
        notes = f"Tucker SVD probe PASS AUC={probe_auc:.4f}; Δ_mean={delta_mean:+.4f}≥0.005"
    elif delta_mean >= -0.005:
        decision = "KEEP"
        notes = f"Tucker SVD probe PASS AUC={probe_auc:.4f}; Δ_mean={delta_mean:+.4f} neutral"
    else:
        decision = "DROP"
        notes = f"Tucker SVD probe PASS AUC={probe_auc:.4f}; Δ_mean={delta_mean:+.4f}<-0.005"
    print(f"Decision: {decision}  — {notes}")

    val_metrics = None
    if decision == "KEEP":
        print("\n=== Val evaluation (1 run) ===")
        val_streams = load_val()
        val_rows = []
        full_model = _train_fold(
            [f"train_{k:02d}" for k in range(13)],
            gt_all, window_features_p33, hard_neg_weight=2.0,
        )
        for sid, cir in val_streams.items():
            filtered   = chebyshev_bp(cir)
            T          = filtered.shape[1]
            dur_s      = T / RATE_HZ
            energy_s1  = multiscale_energy_stage1(filtered)
            energy_p14 = smooth(frame_energy(filtered), sigma_s=0.3)
            cands      = energy_events_raw(energy_s1,  pct=STAGE1_PCT, min_dur=0.4, merge_gap=0.8)
            p14_cands  = energy_events_raw(energy_p14, pct=98.0, min_dur=0.6, merge_gap=0.8)
            p14_events = gaussian_boundary(p14_cands, energy_p14, dur_s, k=1.2)
            wins  = iter_windows(T)
            feats = np.array([window_features_p33(filtered[:, s:e, :]) for s, e in wins]) if wins else None
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
                    ns, ne = refine_boundary(energy_s1, ev["t_start"], ev["t_end"], dur_s)
                    events.append({"t_start": ns, "t_end": ne, "score": mx, "t_alarm": ns})
            events = dur_filter(events, min_dur=0.6)
            events = gap_merge(events, max_gap=1.0)
            val_rows.extend(make_rows(sid, events))

        val_path    = save_sub(val_rows, "pipeline_33")
        val_metrics = run_eval(val_path)
        print(f"\n=== Val Metrics (p33) ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    base_val_f1 = _load_log_entry(base_id).get("f1_mean") or 0.0
    delta_val   = round((val_metrics["f1_mean"] - base_val_f1) if val_metrics else 0.0, 4)
    entry = {
        "experiment_id": "pipeline_33",
        "phase": 5,
        "base": base_id,
        "description": (
            f"p33: 37f (21 clean + 6 centroid_dyn + 10 Tucker SVD r=2/s=8) "
            f"+ multi-scale Stage-1 pct=95"
        ),
        "components": [
            "tucker_svd_r2s8", "multiscale_stage1", "topK32_bins",
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
        "probe_auc": round(probe_auc, 4), "probe_macro_f1": None,
        "gate": None, "decision": decision, "notes": notes,
    }
    append_log(entry)
    print(f"\nLogged: pipeline_33  decision={decision}")
    return decision, oof, val_metrics


if __name__ == "__main__":
    main()
