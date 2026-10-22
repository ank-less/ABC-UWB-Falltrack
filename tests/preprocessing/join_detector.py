"""Join detector for ramp crossfade artifacts in composed streams.

Step 0.4: Detects 10-frame ramp crossfades from the signal itself.
Uses second-order difference spikes to find join boundaries.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

STREAMS_TRAIN = ROOT / "streams" / "train"
STREAMS_VAL   = ROOT / "streams" / "val"
COMP_CSV      = ROOT / "streams_composition.csv"
REPORTS_DIR   = ROOT / "experiments" / "audit" / "reports"
RATE_HZ       = 50.0
RAMP_FRAMES   = 10
TOLERANCE_FRAMES = 3


def detect_joins(cir: np.ndarray, threshold: float = 3.0) -> List[int]:
    """Detect join locations (frame indices) from ramp crossfade artifacts.

    Strategy: compute mean absolute second-order difference across (radars, bins).
    A ramp crossfade produces a spike followed ~10 frames later by an opposite spike.
    """
    # cir shape: (3, T, 368)
    T = cir.shape[1]

    # Use mean absolute second difference across all radars and bins
    # shape: (T-2,)
    d2 = np.diff(cir.astype(np.float64), n=2, axis=1)  # (3, T-2, 368)
    d2_score = np.mean(np.abs(d2), axis=(0, 2))  # (T-2,)

    # Normalize
    med = np.median(d2_score)
    mad = np.median(np.abs(d2_score - med)) + 1e-10
    d2_norm = (d2_score - med) / mad

    # First-order diff for paired spike detection
    d1 = np.diff(cir.astype(np.float64), n=1, axis=1)  # (3, T-1, 368)
    d1_score = np.mean(np.abs(d1), axis=(0, 2))  # (T-1,)
    d1_med = np.median(d1_score)
    d1_mad = np.median(np.abs(d1_score - d1_med)) + 1e-10
    d1_norm = (d1_score - d1_med) / d1_mad

    # Find peaks in d2_norm that exceed threshold
    peaks = []
    for t in range(len(d2_norm)):
        if d2_norm[t] > threshold:
            # Check if this is a local maximum
            lo = max(0, t - 3)
            hi = min(len(d2_norm), t + 4)
            if d2_norm[t] == d2_norm[lo:hi].max():
                peaks.append(t + 1)  # +1 offset for second diff

    # Pair-detection: look for opposing spikes ~10 frames apart
    join_candidates = []

    # Also look for large single spikes in d1 (crossfade boundary)
    d1_peaks = []
    for t in range(len(d1_norm)):
        if d1_norm[t] > threshold:
            lo = max(0, t - 3)
            hi = min(len(d1_norm), t + 4)
            if d1_norm[t] == d1_norm[lo:hi].max():
                d1_peaks.append(t + 1)

    # Combine: any large first-diff spike is a join candidate
    all_candidates = sorted(set(peaks + d1_peaks))

    # Cluster nearby candidates
    merged = []
    for c in all_candidates:
        if merged and c - merged[-1] < 5:
            merged[-1] = (merged[-1] + c) // 2  # take midpoint
        else:
            merged.append(c)

    return merged


def load_gt_joins() -> dict:
    """Load ground truth join times from composition CSV.

    Returns {stream_id: [join_frame_indices]}
    """
    gt_joins = {}
    try:
        with open(COMP_CSV, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return {}

    prev_stream = None
    for row in rows:
        sid = row["stream_id"]
        t_start = float(row["t_start"])
        if sid != prev_stream and prev_stream is not None:
            pass
        if sid not in gt_joins:
            gt_joins[sid] = []
        if t_start > 0.0:
            # This is a join point
            frame = int(round(t_start * RATE_HZ))
            if frame > 0:
                # Avoid duplicates
                if not gt_joins[sid] or abs(frame - gt_joins[sid][-1]) > RAMP_FRAMES:
                    gt_joins[sid].append(frame)
        prev_stream = sid

    return gt_joins


def evaluate_join_detector(threshold: float = 3.0, n_streams: int = 13) -> dict:
    """Evaluate join detector on train (and optionally val) streams.

    Returns recall, precision, join_detector_ok.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    gt_joins = load_gt_joins()

    results = []
    all_tp, all_fp, all_fn = 0, 0, 0
    join_times_by_stream = {}

    for k in range(n_streams):
        sid = f"train_{k:02d}"
        p = STREAMS_TRAIN / f"stream_{k:02d}.h5"
        if not p.exists():
            continue

        with h5py.File(p) as f:
            cir = f["cir"][:]

        detected = detect_joins(cir, threshold=threshold)
        gt = gt_joins.get(sid, [])
        join_times_by_stream[sid] = [d / RATE_HZ for d in detected]

        # Match detected to GT (within tolerance)
        matched_gt = set()
        tp = 0
        for d in detected:
            for gi, g in enumerate(gt):
                if gi not in matched_gt and abs(d - g) <= TOLERANCE_FRAMES:
                    tp += 1
                    matched_gt.add(gi)
                    break
        fp = len(detected) - tp
        fn = len(gt) - len(matched_gt)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        results.append({
            "stream": sid, "detected": len(detected), "gt": len(gt),
            "tp": tp, "fp": fp, "fn": fn, "prec": round(prec, 3), "rec": round(rec, 3),
        })
        print(f"  {sid}: detected={len(detected)} GT={len(gt)} TP={tp} FP={fp} FN={fn}")

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 1.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 1.0
    join_detector_ok = recall >= 0.98 and precision >= 0.95
    print(f"\nOverall: TP={all_tp} FP={all_fp} FN={all_fn}")
    print(f"  Recall={recall:.4f}  Precision={precision:.4f}")
    print(f"  join_detector_ok = {join_detector_ok}")

    summary = {
        "threshold": threshold,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "join_detector_ok": join_detector_ok,
        "join_times_by_stream": join_times_by_stream,
        "per_stream": results,
    }

    out_path = REPORTS_DIR / "join_detector.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Join detector report saved to {out_path}")
    return summary


def get_join_times(sid: str, cir: Optional[np.ndarray] = None, threshold: float = 3.0) -> List[float]:
    """Get join times (in seconds) for a stream.

    Loads from cached report if available, else detects from cir.
    """
    report_path = REPORTS_DIR / "join_detector.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cached = report.get("join_times_by_stream", {})
        if sid in cached:
            return cached[sid]

    if cir is None:
        return []

    detected = detect_joins(cir, threshold=threshold)
    return [d / RATE_HZ for d in detected]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--n-streams", type=int, default=13)
    args = parser.parse_args()

    print(f"=== Join detector evaluation (threshold={args.threshold}) ===")
    summary = evaluate_join_detector(threshold=args.threshold, n_streams=args.n_streams)
    print(f"\nResult: join_detector_ok={summary['join_detector_ok']}")
