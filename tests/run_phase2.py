"""Phase 2: Cải thiện model — pipeline 08–14.
Base: pipeline_04 (Chebyshev bandpass + energy threshold + smoothing, F1=0.4676).
Mục tiêu: F1_mean > 0.5
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

STREAMS_VAL = ROOT / "streams" / "val"
STREAMS_TRAIN = ROOT / "streams" / "train"
EVENTS_CSV = ROOT / "streams_events.csv"
LOG_FILE = ROOT / "experiments_log.json"
SUB_DIR = ROOT
RATE_HZ = 50.0

# ── Helpers ─────────────────────────────────────────────────────────────────

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
        f"FA/h={m['false_alarms_per_hour']:.1f}  "
        f"Δ={delta:+.4f}"
    )
    return m["f1_mean"]


# ── Preprocessing ────────────────────────────────────────────────────────────

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


# ── LightGBM classifier ──────────────────────────────────────────────────────

def load_classifier():
    """Train on all 13 train streams using Chebyshev + window_features."""
    try:
        import lightgbm as lgb
        model_cls = lgb.LGBMClassifier
        use_lgb = True
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model_cls = RandomForestClassifier
        use_lgb = False

    gt_all = load_ground_truth(EVENTS_CSV, ignore_flagged=True)
    gt_by_stream: dict[str, list] = {}
    for ev in gt_all:
        gt_by_stream.setdefault(ev["stream_id"], []).append((ev["t_start"], ev["t_end"]))

    X_parts, y_parts = [], []
    for p in sorted(STREAMS_TRAIN.glob("stream_*.h5")):
        sid = f"train_{int(p.stem.split('_')[1]):02d}"
        print(f"  [{sid}] …", flush=True, end=" ")
        with h5py.File(p) as f:
            cir = f["cir"][:]
        filtered = chebyshev_bp(cir)
        wins = iter_windows(filtered.shape[1])
        if not wins:
            print("skip (short)")
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


def get_lgb_frame_scores(cir, model):
    """Smooth per-frame LightGBM probability, interpolated from sliding windows."""
    T = cir.shape[1]
    wins = iter_windows(T)
    if not wins:
        return np.zeros(T)
    feats = np.array([window_features(cir[:, s:e, :]) for s, e in wins])
    scores = model.predict_proba(feats)[:, 1]
    scores = gaussian_filter1d(scores, sigma=2.0)
    xs = np.array([(s + e) // 2 for s, e in wins], dtype=np.float64)
    full = np.interp(np.arange(T, dtype=np.float64), xs, scores)
    return gaussian_filter1d(full, sigma=5.0)


# ── Pipeline 08 ──────────────────────────────────────────────────────────────
# Chebyshev + energy candidates → LightGBM gate (thresh=0.1) + dur_filter + gap_merge

def p08(streams, model):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        # Slightly lower pct for more candidates than p04
        candidates = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        lgb_scores = get_lgb_frame_scores(filtered, model)
        T = filtered.shape[1]
        events = []
        for ev in candidates:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            max_lgb = float(np.max(lgb_scores[s:e]))
            if max_lgb > 0.1:
                ev["score"] = max_lgb
                events.append(ev)
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 09 ──────────────────────────────────────────────────────────────
# p08 + boundary_refiner: half-max energy within extended event window

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
    thresh = frac * peak_val
    idxs = np.flatnonzero(seg >= thresh)
    if len(idxs) == 0:
        return t_start, t_end
    new_start = max(0.0, (s + int(idxs[0])) / RATE_HZ)
    new_end = min(stream_dur_s, (s + int(idxs[-1]) + 1) / RATE_HZ)
    if new_start >= new_end:
        return t_start, t_end
    return new_start, new_end


def p09(streams, model):
    rows = []
    for sid, cir in streams.items():
        stream_dur_s = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        candidates = energy_events_raw(energy, pct=97.0, min_dur=0.4, merge_gap=0.8)
        lgb_scores = get_lgb_frame_scores(filtered, model)
        T = filtered.shape[1]
        events = []
        for ev in candidates:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            max_lgb = float(np.max(lgb_scores[s:e]))
            if max_lgb > 0.1:
                ev["score"] = max_lgb
                new_start, new_end = refine_boundary(
                    energy, ev["t_start"], ev["t_end"], stream_dur_s
                )
                ev["t_start"] = new_start
                ev["t_end"] = new_end
                ev["t_alarm"] = new_start
                events.append(ev)
        events = dur_filter(events, min_dur=0.6)
        events = gap_merge(events, max_gap=1.0)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 10 ──────────────────────────────────────────────────────────────
# Per-radar independent p04 + majority-vote fusion (2 of 3 radars must agree)

def p04_single_radar(radar_cir):
    """Run p04 on a single-radar CIR (1, T, 368)."""
    filtered = chebyshev_bp(radar_cir)
    energy = smooth(frame_energy(filtered), sigma_s=0.3)
    return energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)


def late_fusion_majority(events_r, min_support=2):
    """Keep events detected by at least min_support radars; deduplicate overlaps."""
    n_radars = len(events_r)
    confirmed = []
    for r in range(n_radars):
        for ev in events_r[r]:
            support = 1
            for r2 in range(n_radars):
                if r2 == r:
                    continue
                for ev2 in events_r[r2]:
                    overlap = min(ev["t_end"], ev2["t_end"]) - max(ev["t_start"], ev2["t_start"])
                    if overlap > 0:
                        support += 1
                        break
            if support >= min_support:
                confirmed.append(dict(ev))
    if not confirmed:
        return []
    confirmed.sort(key=lambda e: e["t_start"])
    merged = [confirmed[0]]
    for ev in confirmed[1:]:
        if ev["t_start"] < merged[-1]["t_end"]:
            merged[-1]["t_end"] = max(merged[-1]["t_end"], ev["t_end"])
            merged[-1]["score"] = max(merged[-1]["score"], ev["score"])
        else:
            merged.append(ev)
    return merged


def p10(streams):
    rows = []
    for sid, cir in streams.items():
        n_radars = cir.shape[0]
        events_r = [p04_single_radar(cir[r:r+1, :, :]) for r in range(n_radars)]
        events = late_fusion_majority(events_r, min_support=2)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 11 ──────────────────────────────────────────────────────────────
# Two-stage: low energy threshold (pct=95) → LightGBM confirmation (thresh=0.2)

def p11(streams, model):
    rows = []
    for sid, cir in streams.items():
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        # Stage 1: very permissive energy candidates
        candidates = energy_events_raw(energy, pct=95.0, min_dur=0.3, merge_gap=1.0)
        # Stage 2: LightGBM confirmation
        lgb_scores = get_lgb_frame_scores(filtered, model)
        T = filtered.shape[1]
        events = []
        for ev in candidates:
            s = int(ev["t_start"] * RATE_HZ)
            e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
            if s >= e:
                continue
            max_lgb = float(np.max(lgb_scores[s:e]))
            if max_lgb > 0.2:
                ev["score"] = max_lgb
                events.append(ev)
        events = dur_filter(events, min_dur=0.5)
        events = gap_merge(events, max_gap=1.5)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 12 ──────────────────────────────────────────────────────────────
# Chebyshev + 3-state HSMM: forward-dilate fall mask to include lying phase

def forward_dilate(mask, frames):
    """Extend each True run forward by 'frames' frames (causal)."""
    T = len(mask)
    result = mask.copy()
    active_until = -1
    for i in range(T):
        if mask[i]:
            active_until = max(active_until, i + frames)
        if i <= active_until:
            result[i] = True
    return result


def p12(streams):
    rows = []
    for sid, cir in streams.items():
        stream_dur_s = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)

        # State 1: fall (high energy burst)
        fall_q = float(np.percentile(energy, 95))
        fall_mask = energy > fall_q

        # State 2: lying (extend each fall segment forward by 2s)
        lying_frames = int(2.0 * RATE_HZ)
        extended = forward_dilate(fall_mask, lying_frames)

        # Convert extended mask to events
        events = runs_to_events(
            extended,
            energy / (fall_q + 1e-9),
            min_duration=0.6,
            merge_gap=1.0,
        )
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 13 ──────────────────────────────────────────────────────────────
# p04 events → shrink to ~1.6s window around energy peak

def shrink_to_peak(events, energy, stream_dur_s, target_dur=1.6):
    T = len(energy)
    shrunken = []
    for ev in events:
        s = int(ev["t_start"] * RATE_HZ)
        e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
        if s >= e:
            shrunken.append(ev)
            continue
        peak_idx = s + int(np.argmax(energy[s:e]))
        half = target_dur / 2.0
        new_start = max(0.0, peak_idx / RATE_HZ - half)
        new_end = min(stream_dur_s, peak_idx / RATE_HZ + half)
        if new_end - new_start < 0.5:
            shrunken.append(ev)
            continue
        shrunken.append({
            "t_start": new_start,
            "t_end": new_end,
            "score": ev["score"],
            "t_alarm": new_start,
        })
    return shrunken


def p13(streams):
    rows = []
    for sid, cir in streams.items():
        stream_dur_s = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        events = shrink_to_peak(events, energy, stream_dur_s, target_dur=1.6)
        rows.extend(make_rows(sid, events))
    return rows


# ── Pipeline 14 ──────────────────────────────────────────────────────────────
# p04 events → Gaussian heatmap boundary (weighted mean ± k*σ)

def gaussian_boundary(events, energy, stream_dur_s, k=1.2):
    T = len(energy)
    MIN_SIGMA_FRAMES = 0.3 * RATE_HZ  # minimum 0.3s half-width
    refined = []
    for ev in events:
        context_s = 1.0
        s = max(0, int((ev["t_start"] - context_s) * RATE_HZ))
        e = min(T, int((ev["t_end"] + context_s) * RATE_HZ) + 1)
        seg = energy[s:e].copy()
        seg_min = float(np.min(seg))
        seg = seg - seg_min
        total = float(np.sum(seg))
        if total <= 0:
            refined.append(ev)
            continue
        idx = np.arange(len(seg), dtype=np.float64)
        mu = float(np.dot(idx, seg)) / total
        var = float(np.dot((idx - mu) ** 2, seg)) / total
        sigma = float(np.sqrt(max(var, MIN_SIGMA_FRAMES ** 2)))
        new_start = max(0.0, (s + mu - k * sigma) / RATE_HZ)
        new_end = min(stream_dur_s, (s + mu + k * sigma) / RATE_HZ)
        if new_end - new_start < 0.5:
            refined.append(ev)
        else:
            refined.append({
                "t_start": new_start,
                "t_end": new_end,
                "score": ev["score"],
                "t_alarm": new_start,
            })
    return refined


def p14(streams):
    rows = []
    for sid, cir in streams.items():
        stream_dur_s = cir.shape[1] / RATE_HZ
        filtered = chebyshev_bp(cir)
        energy = smooth(frame_energy(filtered), sigma_s=0.3)
        events = energy_events_raw(energy, pct=98.0, min_dur=0.6, merge_gap=0.8)
        events = gaussian_boundary(events, energy, stream_dur_s, k=1.2)
        rows.extend(make_rows(sid, events))
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== PHASE 2: Cải thiện model (pipeline 08–14) ===")
    print("Base: pipeline_04  F1_mean=0.4676  TP=38 FP=39 FN=29\n")

    streams = load_val()
    print(f"Loaded {len(streams)} val streams\n")

    BASE_F1 = 0.4676
    prev = BASE_F1
    best_f1 = BASE_F1
    best_id = "pipeline_04"
    no_improve = 0

    # Train classifier for p08, p09, p11 (do this once)
    print("── Training LightGBM on 13 train streams ──")
    model = load_classifier()
    print()

    plan = [
        (8,  "p04_base + LightGBM_gate(thresh=0.1) + dur_filter(0.6s) + gap_merge(1.0s)",
             lambda: p08(streams, model)),
        (9,  "p04_base + LightGBM_gate + boundary_refiner(frac=0.4,ctx=2s)",
             lambda: p09(streams, model)),
        (10, "3_radar_independent_p04 + majority_vote_fusion(2/3)",
             lambda: p10(streams)),
        (11, "two_stage: energy(pct=95) → LightGBM_confirm(thresh=0.2) + gap_merge(1.5s)",
             lambda: p11(streams, model)),
        (12, "Chebyshev + 3state_HSMM: fall_mask + forward_lying_dilation(2s)",
             lambda: p12(streams)),
        (13, "p04_base + window_shrinker(target=1.6s around energy peak)",
             lambda: p13(streams)),
        (14, "p04_base + Gaussian_heatmap_boundary(k=1.2)",
             lambda: p14(streams)),
    ]

    for pid, desc, fn in plan:
        label_map = {
            8:  "Pipeline 08: energy candidates → LightGBM gate + postproc",
            9:  "Pipeline 09: pipeline_08 + boundary refiner",
            10: "Pipeline 10: per-radar detection + majority-vote fusion",
            11: "Pipeline 11: two-stage (low threshold + LightGBM confirm)",
            12: "Pipeline 12: 3-state HSMM (fall + lying phase extension)",
            13: "Pipeline 13: p04 + window shrinker to 1.6s",
            14: "Pipeline 14: p04 + Gaussian heatmap boundary",
        }
        print(f"── {label_map[pid]} ──")
        rows = fn()
        f1 = run_pipeline(pid, desc, rows, prev)
        prev = f1

        if f1 > best_f1:
            best_f1 = f1
            best_id = f"pipeline_{pid:02d}"
            no_improve = 0
        else:
            no_improve += 1

        if best_f1 > 0.6:
            print(f"\nEARLY STOP: F1 > 0.6 đạt được tại {best_id} ({best_f1:.4f})!")
            break

        if no_improve >= 3 and pid < 14:
            print(f"\n3 pipeline liên tiếp không cải thiện so với best ({best_f1:.4f}). Chạy tiếp.")
            no_improve = 0  # Reset and continue per spec

    # Summary
    log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    phase2 = [e for e in log if e["experiment_id"] in {f"pipeline_{i:02d}" for i in range(8, 15)}]

    print("\n" + "=" * 82)
    print("TỔNG HỢP GIAI ĐOẠN 2  (base: pipeline_04, F1=0.4676)")
    print("=" * 82)
    print(f"{'ID':<16} {'F1_mean':>8} {'F1_01':>7} {'F1_03':>7} {'F1_05':>7} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'FA/h':>6} {'Δ':>8}")
    print("-" * 82)
    best_phase2_id, best_phase2_f1 = "", 0.0
    for e in phase2:
        marker = " ◄" if e["f1_mean"] == max(x["f1_mean"] for x in phase2) else ""
        print(
            f"{e['experiment_id']:<16} {e['f1_mean']:>8.4f} {e['f1_01']:>7.4f} "
            f"{e['f1_03']:>7.4f} {e['f1_05']:>7.4f} {e['tp']:>4} {e['fp']:>4} "
            f"{e['fn']:>4} {e['false_alarms_per_hour']:>6.1f} {e['delta_f1']:>+8.4f}{marker}"
        )
        if e["f1_mean"] > best_phase2_f1:
            best_phase2_f1 = e["f1_mean"]
            best_phase2_id = e["experiment_id"]
    print("=" * 82)
    overall_best = max(best_phase2_f1, BASE_F1)
    overall_best_id = best_phase2_id if best_phase2_f1 > BASE_F1 else "pipeline_04"
    print(f"\nPipeline tốt nhất GĐ2 : {best_phase2_id}  (F1_mean = {best_phase2_f1:.4f})")
    print(f"Pipeline tốt nhất toàn cục: {overall_best_id}  (F1_mean = {overall_best:.4f})")
    if overall_best >= 0.5:
        print(f"✓ ĐẠT MỤC TIÊU GĐ2: F1 mean > 0.5")
    else:
        print(f"✗ Chưa đạt F1 > 0.5 — best = {overall_best:.4f}")
        print("  → Xem xét pipeline_13/14 (boundary refine) cho GĐ3")


if __name__ == "__main__":
    main()
