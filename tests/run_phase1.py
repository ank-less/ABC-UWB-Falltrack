"""Phase 1: Giảm FP — pipeline 00–07.

Chạy: python experiments/run_phase1.py
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

# ── Setup paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abc2027.background import subtract_background
from abc2027.baseline import (
    STEP_S,
    frame_energy,
    iter_windows,
    runs_to_events,
    window_features,
)
from abc2027.evaluate import (
    evaluate,
    load_ground_truth,
    stream_durations,
    temporal_iou,
    validate_predictions,
)

STREAMS_VAL = ROOT / "streams" / "val"
STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV = ROOT / "streams_events.csv"
LOG_FILE = ROOT / "experiments_log.json"
SUB_DIR = ROOT
RATE_HZ = 50.0

# ── Helpers ────────────────────────────────────────────────────────────────

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
        w = csv.DictWriter(f, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        w.writeheader()
        w.writerows(rows)
    return path


def run_eval(pred_path):
    durs = stream_durations(STREAMS_VAL)
    preds = validate_predictions(pred_path, durs)
    gt = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    result = evaluate(gt, preds, durs)
    rs = result["ranking_score"]
    bt = result["by_theta"]
    return {
        "f1_mean": round(rs["f1_mean"], 4),
        "f1_01": round(bt["0.1"]["f1"], 4),
        "f1_03": round(bt["0.3"]["f1"], 4),
        "f1_05": round(bt["0.5"]["f1"], 4),
        "tp": bt["0.3"]["tp"],
        "fp": bt["0.3"]["fp"],
        "fn": bt["0.3"]["fn"],
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
            "t_start": f"{e['t_start']:.3f}",
            "t_end": f"{e['t_end']:.3f}",
            "score": f"{e['score']:.6f}",
            "t_alarm": f"{e['t_alarm']:.3f}",
        }
        for e in events
    ]


def run_pipeline(pid, desc, rows, prev_f1=None):
    name = f"pipeline_{pid:02d}"
    path = save_sub(rows, name)
    m = run_eval(path)
    delta = round(m["f1_mean"] - prev_f1, 4) if prev_f1 is not None else 0.0
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
        f"FA/h={m['false_alarms_per_hour']:.1f}  "
        f"Δ={delta:+.4f}"
    )
    return m["f1_mean"]


# ── Preprocessing ──────────────────────────────────────────────────────────

def chebyshev_bp(cir, low=0.5, high=15.0, order=4):
    nyq = RATE_HZ / 2.0
    sos = cheby1(order, 1.0, [low / nyq, high / nyq], btype="band", output="sos")
    radars, T, bins = cir.shape
    flat = cir.astype(np.float64).transpose(0, 2, 1).reshape(-1, T)  # (R*B, T)
    zi_tmpl = sosfilt_zi(sos)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        zi = zi_tmpl * flat[i, 0]
        out[i], _ = sosfilt(sos, flat[i], zi=zi)
    return out.reshape(radars, bins, T).transpose(0, 2, 1).astype(np.float32)


def iqr_norm(energy, eps=1e-9):
    q25, q75 = np.percentile(energy, [25, 75])
    iqr = q75 - q25
    return (energy - q25) / (iqr + eps)


# ── Postprocessing ─────────────────────────────────────────────────────────

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


# ── LightGBM training ──────────────────────────────────────────────────────

def train_classifier():
    """Train on all 13 train streams. Returns fitted model."""
    try:
        import lightgbm as lgb
        clf_cls = lgb.LGBMClassifier
        use_lgb = True
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        clf_cls = RandomForestClassifier
        use_lgb = False

    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by_stream: dict[str, list] = {}
    for ev in gt_all:
        gt_by_stream.setdefault(ev["stream_id"], []).append((ev["t_start"], ev["t_end"]))

    X_parts, y_parts = [], []
    for p in sorted(STREAMS_TRAIN.glob("stream_*.h5")):
        sid = f"train_{int(p.stem.split('_')[1]):02d}"
        print(f"  Loading {sid} for training …", flush=True)
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        wins = iter_windows(filtered.shape[1])
        if not wins:
            continue
        feats = np.array([window_features(filtered[:, s:e, :]) for s, e in wins])
        gt_ev = gt_by_stream.get(sid, [])
        labels = np.zeros(len(wins), dtype=np.int32)
        for i, (ws, we) in enumerate(wins):
            t_s, t_e = ws * STEP_S, we * STEP_S
            if any(temporal_iou(t_s, t_e, gs, ge) >= 0.1 for gs, ge in gt_ev):
                labels[i] = 1
        X_parts.append(feats)
        y_parts.append(labels)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    print(f"  Train samples: {len(y)} (pos={pos}, neg={neg}), ratio={neg//max(pos,1)}:1")

    if use_lgb:
        model = clf_cls(
            n_estimators=400,
            num_leaves=31,
            scale_pos_weight=neg / max(pos, 1),
            n_jobs=-1,
            verbose=-1,
        )
    else:
        model = clf_cls(
            n_estimators=400,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    model.fit(X, y)
    return model


def windows_to_events(cir, model, threshold=0.35, min_dur=0.6, merge_gap=0.8):
    wins = iter_windows(cir.shape[1])
    if not wins:
        return []
    feats = np.array([window_features(cir[:, s:e, :]) for s, e in wins])
    scores = model.predict_proba(feats)[:, 1]
    scores_s = gaussian_filter1d(scores, sigma=2.0)

    # Interpolate to full time resolution
    xs = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
    full = np.interp(np.arange(cir.shape[1], dtype=np.float64), xs, scores_s)
    full = gaussian_filter1d(full, sigma=5.0)

    # Dynamic threshold
    level = float(np.percentile(full, 99.0))
    level = max(level, threshold)
    if level <= 0:
        return []
    return runs_to_events(full >= level, full / (level + 1e-9), min_dur, merge_gap)


# ── Pipeline 00 – Baseline ─────────────────────────────────────────────────

def p00(streams):
    from abc2027.baseline import energy_events
    rows = []
    for sid, cir in streams.items():
        events = energy_events(cir, percentile=98.0, min_duration=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 01 – Temporal smoothing ──────────────────────────────────────

def p01(streams):
    rows = []
    for sid, cir in streams.items():
        residual = subtract_background(cir, 3.0)
        energy = smooth(frame_energy(residual), sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 02 – Smoothing + duration filter ──────────────────────────────

def p02(streams):
    rows = []
    for sid, cir in streams.items():
        residual = subtract_background(cir, 3.0)
        energy = smooth(frame_energy(residual), sigma_s=0.3)
        # Lower pct to capture more, then filter
        events = energy_events_raw(energy, pct=97.0, min_dur=0.3, merge_gap=0.8)
        events = dur_filter(events, min_dur=1.0)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 03 – Smoothing + duration filter + gap merger ────────────────

def p03(streams):
    rows = []
    for sid, cir in streams.items():
        residual = subtract_background(cir, 3.0)
        energy = smooth(frame_energy(residual), sigma_s=0.3)
        events = energy_events_raw(energy, pct=97.0, min_dur=0.3, merge_gap=0.5)
        events = dur_filter(events, min_dur=1.0)
        events = gap_merge(events, max_gap=1.5)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 04 – Chebyshev + smoothing ───────────────────────────────────

def p04(streams):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 05 – Chebyshev + IQR + smoothing ─────────────────────────────

def p05(streams):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        energy = iqr_norm(frame_energy(filtered).astype(np.float64))
        energy = smooth(energy, sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 06 – Chebyshev + IQR + classifier ────────────────────────────

def p06(streams, model):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        events = windows_to_events(filtered, model, threshold=0.35, min_dur=0.6, merge_gap=0.8)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 07 – Chebyshev + IQR + classifier + filter + merger ──────────

def p07(streams, model):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        events = windows_to_events(filtered, model, threshold=0.35, min_dur=0.3, merge_gap=0.5)
        events = dur_filter(events, min_dur=0.8)
        events = gap_merge(events, max_gap=1.5)
        rows.extend(make_rows(sid, events))
    return rows


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=== PHASE 1: Giảm FP (pipeline 00–07) ===\n")
    streams = load_val()
    print(f"Loaded {len(streams)} val streams\n")

    prev = None

    # pipeline_00
    print("── Pipeline 00: Baseline ──")
    prev = run_pipeline(0, "Baseline gốc: EMA background + energy threshold", p00(streams), None)

    # pipeline_01
    print("── Pipeline 01: Baseline + temporal smoothing ──")
    prev = run_pipeline(1, "baseline + Gaussian temporal smoothing (σ=0.3s)", p01(streams), prev)

    # pipeline_02
    print("── Pipeline 02: Baseline + smoothing + duration filter ──")
    prev = run_pipeline(2, "baseline + smoothing + duration_filter(>=1.0s)", p02(streams), prev)

    # pipeline_03
    print("── Pipeline 03: Smoothing + dur_filter + gap_merger ──")
    prev = run_pipeline(3, "baseline + smoothing + duration_filter + gap_merger(1.5s)", p03(streams), prev)

    # pipeline_04
    print("── Pipeline 04: Chebyshev + smoothing ──")
    prev = run_pipeline(4, "Chebyshev bandpass(0.5–15 Hz) + energy threshold + smoothing", p04(streams), prev)

    # pipeline_05
    print("── Pipeline 05: Chebyshev + IQR + smoothing ──")
    prev = run_pipeline(5, "Chebyshev + IQR normalization + energy threshold + smoothing", p05(streams), prev)

    # Train classifier for pipelines 06–07
    print("\n── Training classifier on 13 train streams (pipelines 06–07) ──")
    model = train_classifier()
    print("  Classifier trained.\n")

    # pipeline_06
    print("── Pipeline 06: Chebyshev + IQR + classifier ──")
    prev = run_pipeline(6, "Chebyshev + IQR + LightGBM/RF classifier + smoothing", p06(streams, model), prev)

    # pipeline_07
    print("── Pipeline 07: Chebyshev + IQR + classifier + dur + gap ──")
    prev = run_pipeline(7, "Chebyshev + IQR + classifier + duration_filter + gap_merger", p07(streams, model), prev)

    # Summary
    log: list = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    phase1 = [e for e in log if e["experiment_id"].startswith("pipeline_0")]

    print("\n" + "=" * 75)
    print("TỔNG HỢP GIAI ĐOẠN 1")
    print("=" * 75)
    print(f"{'ID':<14} {'F1_mean':>8} {'F1_01':>7} {'F1_03':>7} {'F1_05':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'FA/h':>6} {'Δ':>7}")
    print("-" * 75)
    best_id, best_f1 = "", 0.0
    for e in phase1:
        print(
            f"{e['experiment_id']:<14} {e['f1_mean']:>8.4f} {e['f1_01']:>7.4f} "
            f"{e['f1_03']:>7.4f} {e['f1_05']:>7.4f} {e['tp']:>4} {e['fp']:>4} "
            f"{e['fn']:>4} {e['false_alarms_per_hour']:>6.1f} {e['delta_f1']:>+7.4f}"
        )
        if e["f1_mean"] > best_f1:
            best_f1 = e["f1_mean"]
            best_id = e["experiment_id"]
    print("=" * 75)
    print(f"\nPipeline tốt nhất: {best_id}  (F1_mean = {best_f1:.4f})")
    if best_f1 >= 0.4:
        print("✓ Đạt mục tiêu Giai đoạn 1: F1 mean > 0.4")
    else:
        print(f"✗ Chưa đạt F1 > 0.4 (hiện tại {best_f1:.4f}) — xem xét tuning tiếp")


if __name__ == "__main__":
    main()
