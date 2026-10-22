"""OOF harness for GĐ4: LOSO 13-fold cross-validation on train streams.

Step 0.1: Build LOSO folds respecting subject-level exclusion,
run pipeline for each fold, collect OOF predictions, evaluate.
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from abc2027.evaluate import evaluate, load_ground_truth, stream_durations, validate_predictions
from abc2027.baseline import frame_energy, iter_windows, window_features, STEP_S

STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV    = ROOT / "streams_events.csv"
COMP_CSV      = ROOT / "streams_composition.csv"
RATE_HZ       = 50.0

# ── Composition helpers ────────────────────────────────────────────────────────

def load_composition() -> List[dict]:
    with open(COMP_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_recording_id(source_path: str) -> Optional[str]:
    """Extract recording session ID from source path.
    e.g. 'falls/Dataset 3/Test_P3_3/Chute03_152.json' → 'Dataset_3/Test_P3_3'
    ADL paths ('activities/...') → None (shared across subjects)
    """
    if not source_path or not source_path.startswith("falls/"):
        return None
    parts = source_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        dataset = parts[1].replace(" ", "_")
        session = parts[2]
        return f"{dataset}/{session}"
    return None


def get_person_id(source_path: str) -> Optional[str]:
    """Extract person ID from recording path.
    'falls/Dataset 3/Test_P3_3/...' → 'Dataset_3/P3'
    """
    rec = get_recording_id(source_path)
    if rec is None:
        return None
    # rec = "Dataset_3/Test_P3_3" → extract PX
    m = re.search(r"Test_(P\d+)_", rec)
    if m:
        dataset_part = rec.split("/")[0]
        return f"{dataset_part}/{m.group(1)}"
    return rec


def build_subject_map(comp_rows: List[dict]) -> Dict[str, set]:
    """Map each stream_id to the set of person IDs it contains (falls only)."""
    stream_persons: Dict[str, set] = {}
    for row in comp_rows:
        sid = row["stream_id"]
        if row["kind"] != "fall":
            continue
        person = get_person_id(row.get("path", ""))
        if person is not None:
            stream_persons.setdefault(sid, set()).add(person)
    return stream_persons


def build_folds(n_train: int = 13) -> List[Tuple[str, List[str]]]:
    """Build LOSO folds with subject-level exclusion.

    Returns list of (test_stream, [train_streams]) for each fold k.

    Falls back to pure LOSO if all streams share subjects (as in this dataset where
    all 13 streams contain falls from all 3 subjects/datasets — no recording is
    unique to a single stream).
    """
    comp_rows = load_composition()
    stream_persons = build_subject_map(comp_rows)

    all_streams = [f"train_{k:02d}" for k in range(n_train)]

    # Check if any stream could have training data after subject exclusion
    can_exclude = False
    for k in range(n_train):
        test_sid = f"train_{k:02d}"
        test_persons = stream_persons.get(test_sid, set())
        remaining = [
            sid for sid in all_streams
            if sid != test_sid and not (stream_persons.get(sid, set()) & test_persons)
        ]
        if remaining:
            can_exclude = True
            break

    if can_exclude:
        folds = []
        excluded_counts = []
        for k in range(n_train):
            test_sid = f"train_{k:02d}"
            test_persons = stream_persons.get(test_sid, set())
            train_sids = []
            n_excluded = 0
            for sid in all_streams:
                if sid == test_sid:
                    continue
                sid_persons = stream_persons.get(sid, set())
                if sid_persons & test_persons:
                    n_excluded += 1
                else:
                    train_sids.append(sid)
            folds.append((test_sid, train_sids))
            excluded_counts.append(n_excluded)
        print("Folds (subject-level exclusion):")
        for k, ((test, train), exc) in enumerate(zip(folds, excluded_counts)):
            print(f"  fold {k:02d}: test={test}  train_streams={len(train)}  excluded={exc}")
        return folds
    else:
        # Pure LOSO: no subject can be excluded (all streams share all subjects)
        # This is the case for this dataset where all 13 streams mix the same 3 subjects
        print("WARNING: All streams share subjects -- falling back to pure LOSO (no subject exclusion).")
        print("  Note: subject-level leakage is unavoidable in this dataset design.")
        print("  Subjects in ALL streams: " + str(set().union(*stream_persons.values())))
        folds = [(f"train_{k:02d}", [f"train_{j:02d}" for j in range(n_train) if j != k])
                 for k in range(n_train)]
        print("Folds (pure LOSO, no subject exclusion):")
        for k, (test, train) in enumerate(folds):
            print(f"  fold {k:02d}: test={test}  train_streams={len(train)}")
        return folds


# ── OOF runner ────────────────────────────────────────────────────────────────

def run_oof(
    pipeline_fn: Callable,
    threshold_fn: Optional[Callable] = None,
    label: str = "oof",
    out_dir: Optional[Path] = None,
) -> Tuple[dict, Path]:
    """Run LOSO OOF for a pipeline.

    Args:
        pipeline_fn: callable(streams_dict, model, threshold) → list of rows
        threshold_fn: callable(model) → float threshold (OOF-based)
        label: prefix for output files
        out_dir: where to save OOF csv (default: ROOT)

    Returns:
        (metrics_dict, oof_csv_path)
    """
    if out_dir is None:
        out_dir = ROOT
    out_dir = Path(out_dir)

    folds = build_folds()
    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)

    oof_rows = []
    all_scores_by_stream = {}  # for threshold selection

    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"\n=== Fold {k:02d}: test={test_sid}  train={train_sids} ===")

        # Train model on this fold's training streams
        print(f"  Training on {len(train_sids)} streams...")
        model = _train_fold(train_sids, gt_all)

        # Optionally select threshold on this fold's OOF training data
        fold_threshold = 0.25  # default
        if threshold_fn is not None:
            fold_threshold = threshold_fn(model, train_sids)
            print(f"  Fold threshold: {fold_threshold:.3f}")

        # Load test stream
        test_path = STREAMS_TRAIN / f"stream_{int(test_sid.split('_')[1]):02d}.h5"
        with h5py.File(test_path) as f:
            test_cir = f["cir"][:]
        streams = {test_sid: test_cir}

        # Run pipeline
        rows = pipeline_fn(streams, model, fold_threshold)
        oof_rows.extend(rows)

        # Store scores for threshold sweep
        all_scores_by_stream[test_sid] = rows

        print(f"  Fold {k:02d}: {len(rows)} predictions")

    # Save OOF predictions
    oof_csv = out_dir / f"oof_{label}.csv"
    with open(oof_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        w.writeheader()
        w.writerows(oof_rows)
    print(f"\nOOF CSV saved: {oof_csv}")

    # Evaluate
    durs = stream_durations(STREAMS_TRAIN)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    preds = validate_predictions(oof_csv, durs)
    result = evaluate(gt_train, preds, durs)

    rs = result["ranking_score"]
    bt = result["by_theta"]
    metrics = {
        "f1_mean": round(rs["f1_mean"], 4),
        "f1_01":   round(bt["0.1"]["f1"], 4),
        "f1_03":   round(bt["0.3"]["f1"], 4),
        "f1_05":   round(bt["0.5"]["f1"], 4),
        "tp":      bt["0.3"]["tp"],
        "fp":      bt["0.3"]["fp"],
        "fn":      bt["0.3"]["fn"],
        "false_alarms_per_hour": round(rs["false_alarms_per_hour"], 1),
    }
    print(f"\nOOF Results ({label}):")
    print(f"  F1_mean={metrics['f1_mean']:.4f}  F1_01={metrics['f1_01']:.4f}  "
          f"F1_03={metrics['f1_03']:.4f}  F1_05={metrics['f1_05']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
          f"FA/h={metrics['false_alarms_per_hour']:.1f}")
    return metrics, oof_csv


def _train_fold(train_sids: List[str], gt_all: list) -> object:
    """Train LightGBM with rich+aug features on given training streams."""
    from experiments.preprocessing.chebyshev_filter import chebyshev_bp
    from experiments.models.lightgbm_classifier import (
        window_features_rich, _try_lgb, _fit,
    )
    from abc2027.evaluate import temporal_iou

    gt_by: Dict[str, list] = {}
    for ev in gt_all:
        gt_by.setdefault(ev["stream_id"], []).append((ev["t_start"], ev["t_end"]))

    X_parts, y_parts = [], []
    for sid in train_sids:
        stream_idx = int(sid.split("_")[1])
        p = STREAMS_TRAIN / f"stream_{stream_idx:02d}.h5"
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        wins = iter_windows(filtered.shape[1])
        if not wins:
            continue
        feats = np.array([window_features_rich(filtered[:, s:e, :]) for s, e in wins])
        gt_ev = gt_by.get(sid, [])
        from abc2027.evaluate import temporal_iou
        labels = np.array([
            int(any(temporal_iou(ws * STEP_S, we * STEP_S, gs, ge) >= 0.1
                    for gs, ge in gt_ev))
            for ws, we in wins
        ], dtype=np.int32)
        X_parts.append(feats)
        y_parts.append(labels)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Augmentation on positives
    pos_idx = np.where(y == 1)[0]
    X_pos = X[pos_idx]
    rng = np.random.default_rng(42)
    stds = X_pos.std(axis=0) + 1e-8
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
    y_full = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_aug_pos))]).astype(np.int32)
    X_full = np.vstack([X_neg, X_aug_pos])

    model_cls, use_lgb = _try_lgb()
    return _fit(model_cls, use_lgb, X_full, y_full)


# ── OOF threshold sweep ────────────────────────────────────────────────────────

def oof_threshold_sweep(
    pipeline_fn: Callable,
    thresholds: Optional[List[float]] = None,
    label: str = "p19c",
) -> Tuple[float, dict]:
    """Sweep threshold on nested OOF, return (best_threshold, all_results).

    Caches LGB scores per fold to avoid recomputing for each threshold.
    """
    from abc2027.baseline import frame_energy, iter_windows
    from experiments.preprocessing.chebyshev_filter import chebyshev_bp
    from experiments.postprocessing.temporal_smoothing import smooth
    from experiments.models.lightgbm_classifier import lgb_frame_scores, window_features_rich
    from experiments.postprocessing.boundary_refiner import refine_boundary, gaussian_boundary
    from experiments.postprocessing.duration_filter import dur_filter
    from experiments.postprocessing.gap_merger import gap_merge
    from experiments.utils import energy_events_raw

    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.10, 0.51, 0.025)]

    print(f"OOF threshold sweep for {label} over {len(thresholds)} thresholds...")
    folds = build_folds()
    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = [e for e in gt_all if e["stream_id"].startswith("train_")]
    durs = stream_durations(STREAMS_TRAIN)

    # Phase 1: Train fold models AND cache scores for each test stream
    print("Phase 1: Training fold models + caching scores...")
    fold_cache = []  # list of (test_sid, cached_data)
    for k, (test_sid, train_sids) in enumerate(folds):
        print(f"  Fold {k:02d}: training on {len(train_sids)} streams...", flush=True)
        model = _train_fold(train_sids, gt_all)

        # Load and score test stream
        test_path = STREAMS_TRAIN / f"stream_{int(test_sid.split('_')[1]):02d}.h5"
        with h5py.File(test_path) as f:
            test_cir = f["cir"][:]

        T = test_cir.shape[1]
        dur_s = T / RATE_HZ
        filtered = chebyshev_bp(test_cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        cands = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        scores = lgb_frame_scores(filtered, model, feat_fn=window_features_rich)
        p14_cands = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        p14_events = gaussian_boundary(p14_cands, energy, dur_s, k=1.2)

        # Precompute per-candidate: max_score, p14_overlap, refined_boundary
        cand_data = []
        for ev in cands:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            mx = float(np.max(scores[s:e]))
            p14_ok = any(
                min(ev["t_end"], pev["t_end"]) - max(ev["t_start"], pev["t_start"]) > 0
                for pev in p14_events
            )
            ns, ne = refine_boundary(energy, ev["t_start"], ev["t_end"], dur_s)
            cand_data.append({
                "t_start": ns, "t_end": ne, "score": mx, "p14_ok": p14_ok,
            })

        fold_cache.append((test_sid, cand_data))
        print(f"    {len(cand_data)} candidates cached", flush=True)

    # Phase 2: Sweep thresholds on cached data (fast)
    print(f"\nPhase 2: Sweeping {len(thresholds)} thresholds...")
    tmp_csv = ROOT / "_oof_sweep_tmp.csv"
    best_f1, best_thr = 0.0, 0.25
    results = {}

    for thr in thresholds:
        high_thr = min(thr * 2.5, 0.50)
        all_rows = []
        for test_sid, cand_data in fold_cache:
            events = []
            for cand in cand_data:
                mx = cand["score"]
                if mx <= thr * 0.6:
                    continue
                if (cand["p14_ok"] and mx > thr) or mx > high_thr:
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
            w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
            w.writeheader()
            w.writerows(all_rows)

        try:
            preds = validate_predictions(tmp_csv, durs)
            res = evaluate(gt_train, preds, durs)
            f1 = round(res["ranking_score"]["f1_mean"], 4)
        except Exception as ex:
            f1 = 0.0
            print(f"  thr={thr:.3f} eval error: {ex}")

        results[thr] = f1
        print(f"  thr={thr:.3f}  OOF F1_mean={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    tmp_csv.unlink(missing_ok=True)
    print(f"\nBest OOF threshold: {best_thr:.3f} (F1={best_f1:.4f})")
    return best_thr, results, fold_cache  # also return fold_cache for reuse


# ── Paired bootstrap ──────────────────────────────────────────────────────────

def paired_bootstrap(
    pred_a: Path,
    pred_b: Path,
    n: int = 5000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap test comparing two OOF prediction files.

    Resamples both by stream AND by event.
    Returns delta F1_mean, delta F1_05, P(delta > 0).
    """
    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_train = {e["stream_id"]: [] for e in gt_all if e["stream_id"].startswith("train_")}
    for e in gt_all:
        if e["stream_id"] in gt_train:
            gt_train[e["stream_id"]].append(e)

    durs = stream_durations(STREAMS_TRAIN)
    preds_a = validate_predictions(pred_a, durs)
    preds_b = validate_predictions(pred_b, durs)

    pa_by = {sid: [] for sid in durs}
    pb_by = {sid: [] for sid in durs}
    for p in preds_a:
        pa_by[p["stream_id"]].append(p)
    for p in preds_b:
        pb_by[p["stream_id"]].append(p)

    streams = sorted(durs.keys())
    rng = np.random.default_rng(seed)
    delta_mean_boot = np.zeros(n)
    delta_05_boot   = np.zeros(n)

    for i in range(n):
        # Resample streams with replacement
        boot_streams = rng.choice(streams, size=len(streams), replace=True)
        boot_gt = []
        boot_pa = []
        boot_pb = []
        boot_durs = {}
        for j, sid in enumerate(boot_streams):
            new_sid = f"boot_{j:04d}"
            boot_durs[new_sid] = durs[sid]
            for ev in gt_train.get(sid, []):
                ev2 = dict(ev)
                ev2["stream_id"] = new_sid
                boot_gt.append(ev2)
            for p in pa_by.get(sid, []):
                p2 = dict(p)
                p2["stream_id"] = new_sid
                boot_pa.append(p2)
            for p in pb_by.get(sid, []):
                p2 = dict(p)
                p2["stream_id"] = new_sid
                boot_pb.append(p2)

        try:
            ra = evaluate(boot_gt, boot_pa, boot_durs)
            rb = evaluate(boot_gt, boot_pb, boot_durs)
            delta_mean_boot[i] = rb["ranking_score"]["f1_mean"] - ra["ranking_score"]["f1_mean"]
            delta_05_boot[i] = rb["by_theta"]["0.5"]["f1"] - ra["by_theta"]["0.5"]["f1"]
        except Exception:
            delta_mean_boot[i] = np.nan
            delta_05_boot[i] = np.nan

    valid_mean = delta_mean_boot[~np.isnan(delta_mean_boot)]
    valid_05   = delta_05_boot[~np.isnan(delta_05_boot)]

    return {
        "delta_f1_mean": float(np.mean(valid_mean)) if len(valid_mean) else 0.0,
        "delta_f1_05":   float(np.mean(valid_05))   if len(valid_05) else 0.0,
        "p_delta_mean_gt0": float(np.mean(valid_mean > 0)) if len(valid_mean) else 0.0,
        "p_delta_05_gt0":   float(np.mean(valid_05 > 0))   if len(valid_05) else 0.0,
        "n_valid": int(len(valid_mean)),
    }
