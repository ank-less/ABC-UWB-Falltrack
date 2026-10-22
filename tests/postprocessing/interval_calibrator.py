"""Interval calibrator for pipeline_21.

Globally shift/scale event boundaries to maximise OOF F1_mean.
Two forms:
  A: shift t_start by δ_start, t_end by δ_end (independent)
  B: shift centre by δ_c, scale length by k

Fits with nested OOF: for fold k, fit params on OOF of the other 12 folds.
For val/test: fit on full OOF.
"""
from __future__ import annotations

import csv
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RATE_HZ = 50.0

# ──────────────────────────────────────────────────────────────────────────────
# Grid definitions (0.1 s step for nested-OOF fitting speed;
# fine 0.05 s step used only for the single global-param fit on full OOF)
# ──────────────────────────────────────────────────────────────────────────────
_DS_COARSE = [round(x, 2) for x in np.arange(-0.6, 0.61, 0.1)]   # 13 vals
_DE_COARSE = [round(x, 2) for x in np.arange(-0.6, 0.61, 0.1)]
_DC_COARSE = [round(x, 2) for x in np.arange(-0.4, 0.41, 0.1)]   # 9 vals
_K_COARSE  = [round(x, 2) for x in np.arange(0.8, 1.81, 0.1)]    # 11 vals
_DMIN_GRID = [0.0, 0.8, 1.0, 1.2]

_DS_FINE = [round(x, 2) for x in np.arange(-0.6, 0.61, 0.05)]
_DE_FINE = [round(x, 2) for x in np.arange(-0.6, 0.61, 0.05)]
_DC_FINE = [round(x, 2) for x in np.arange(-0.4, 0.41, 0.05)]
_K_FINE  = [round(x, 2) for x in np.arange(0.8, 1.81, 0.1)]


# ──────────────────────────────────────────────────────────────────────────────
# Stream-data helper
# ──────────────────────────────────────────────────────────────────────────────

class _StreamData:
    """Precomputed data for fast calibration evaluation."""
    __slots__ = ("pred_s", "pred_e", "pred_scores", "gt_s", "gt_e", "dur_s")

    def __init__(self, preds: List[dict], gt: List[dict], dur_s: float):
        self.pred_s = np.array([p["t_start"] for p in preds], dtype=np.float64)
        self.pred_e = np.array([p["t_end"]   for p in preds], dtype=np.float64)
        self.pred_scores = np.array([p.get("score", 1.0) for p in preds], dtype=np.float64)
        self.gt_s   = np.array([g["t_start"] for g in gt], dtype=np.float64)
        self.gt_e   = np.array([g["t_end"]   for g in gt], dtype=np.float64)
        self.dur_s  = dur_s


def _iou_matrix(ps: np.ndarray, pe: np.ndarray,
                gs: np.ndarray, ge: np.ndarray) -> np.ndarray:
    """Vectorised IoU: shape (N_pred, N_gt)."""
    ov = np.maximum(0.0,
                    np.minimum(pe[:, None], ge[None, :]) -
                    np.maximum(ps[:, None], gs[None, :]))
    un = (pe - ps)[:, None] + (ge - gs)[None, :] - ov
    return np.where(un > 0, ov / un, 0.0)


def _count_matches(ps: np.ndarray, pe: np.ndarray, scores: np.ndarray,
                   gs: np.ndarray, ge: np.ndarray, theta: float) -> Tuple[int, int, int]:
    """Greedy TP/FP/FN at threshold theta (highest-score first)."""
    if len(ps) == 0 and len(gs) == 0:
        return 0, 0, 0
    if len(ps) == 0:
        return 0, 0, len(gs)
    if len(gs) == 0:
        return 0, len(ps), 0
    iou = _iou_matrix(ps, pe, gs, ge)
    order = np.argsort(-scores)
    used = np.zeros(len(gs), dtype=bool)
    tp = 0
    for i in order:
        best_j = -1
        best_v = theta - 1e-9
        for j in range(len(gs)):
            if not used[j] and iou[i, j] > best_v:
                best_v = iou[i, j]
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = len(ps) - tp
    fn = int(used.sum() == 0 and len(gs) > 0) * len(gs)  # not used
    # correct fn count
    fn = int(np.sum(~used))
    return tp, fp, fn


def _eval_streams(stream_data: List[_StreamData],
                  ps_adj: List[np.ndarray], pe_adj: List[np.ndarray],
                  thetas=(0.1, 0.3, 0.5)) -> float:
    """Mean F1 across thetas over all streams."""
    agg = {th: [0, 0, 0] for th in thetas}  # tp, fp, fn
    for sd, ps, pe in zip(stream_data, ps_adj, pe_adj):
        scores = sd.pred_scores
        for th in thetas:
            tp, fp, fn = _count_matches(ps, pe, scores, sd.gt_s, sd.gt_e, th)
            agg[th][0] += tp
            agg[th][1] += fp
            agg[th][2] += fn
    f1s = []
    for th in thetas:
        tp, fp, fn = agg[th]
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(f1s))


def _adjust_form_a(sd: _StreamData, ds: float, de: float, d_min: float):
    ps = np.maximum(0.0, sd.pred_s + ds)
    pe = np.minimum(sd.dur_s, sd.pred_e + de)
    short = pe - ps < d_min
    if short.any():
        c = (sd.pred_s[short] + sd.pred_e[short]) / 2.0
        ps[short] = np.maximum(0.0, c - d_min / 2)
        pe[short] = np.minimum(sd.dur_s, c + d_min / 2)
    valid = pe > ps
    return ps[valid], pe[valid], valid


def _adjust_form_b(sd: _StreamData, dc: float, k: float, d_min: float):
    c  = (sd.pred_s + sd.pred_e) / 2.0 + dc
    hl = (sd.pred_e - sd.pred_s) / 2.0 * k
    ps = np.maximum(0.0, c - hl)
    pe = np.minimum(sd.dur_s, c + hl)
    short = pe - ps < d_min
    if short.any():
        ps[short] = np.maximum(0.0, c[short] - d_min / 2)
        pe[short] = np.minimum(sd.dur_s, c[short] + d_min / 2)
    valid = pe > ps
    return ps[valid], pe[valid], valid


def _grid_search(stream_data: List[_StreamData], form: str, fine: bool = False):
    """Grid search over calibration params. Returns (best_f1, best_params)."""
    ds_vals = _DS_FINE if fine else _DS_COARSE
    de_vals = _DE_FINE if fine else _DE_COARSE
    dc_vals = _DC_FINE if fine else _DC_COARSE
    k_vals  = _K_FINE  if fine else _K_COARSE

    best_f1, best_params = -1.0, None

    if form == "A":
        for d_min in _DMIN_GRID:
            for ds, de in product(ds_vals, de_vals):
                adj = [_adjust_form_a(sd, ds, de, d_min) for sd in stream_data]
                ps_list = [a[0] for a in adj]
                pe_list = [a[1] for a in adj]
                # rebuild scores removing invalid
                sd_list2, ps2, pe2 = [], [], []
                for sd, (ps, pe, valid) in zip(stream_data, adj):
                    sd2 = _StreamData.__new__(_StreamData)
                    sd2.pred_s = ps
                    sd2.pred_e = pe
                    sd2.pred_scores = sd.pred_scores[valid]
                    sd2.gt_s = sd.gt_s
                    sd2.gt_e = sd.gt_e
                    sd2.dur_s = sd.dur_s
                    sd_list2.append(sd2)
                    ps2.append(ps)
                    pe2.append(pe)
                f1 = _eval_streams(sd_list2, ps2, pe2)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"form": "A", "ds": ds, "de": de, "d_min": d_min}
    else:  # form == "B"
        for d_min in _DMIN_GRID:
            for dc, k in product(dc_vals, k_vals):
                adj = [_adjust_form_b(sd, dc, k, d_min) for sd in stream_data]
                sd_list2, ps2, pe2 = [], [], []
                for sd, (ps, pe, valid) in zip(stream_data, adj):
                    sd2 = _StreamData.__new__(_StreamData)
                    sd2.pred_s = ps
                    sd2.pred_e = pe
                    sd2.pred_scores = sd.pred_scores[valid]
                    sd2.gt_s = sd.gt_s
                    sd2.gt_e = sd.gt_e
                    sd2.dur_s = sd.dur_s
                    sd_list2.append(sd2)
                    ps2.append(ps)
                    pe2.append(pe)
                f1 = _eval_streams(sd_list2, ps2, pe2)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"form": "B", "dc": dc, "k": k, "d_min": d_min}

    return best_f1, best_params


# ──────────────────────────────────────────────────────────────────────────────
# Main calibration API
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_nested_oof(
    oof_csv: Path,
    folds: list,
    gt_by: Dict[str, list],
    durs: Dict[str, float],
    verbose: bool = True,
) -> Tuple[str, dict, List[dict]]:
    """Nested OOF calibration.

    Returns (form_name, global_params, calibrated_oof_rows).
    global_params is fitted on all OOF for val/test inference.
    """
    # Load raw OOF predictions
    oof_preds_by: Dict[str, list] = {}
    with open(oof_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["stream_id"]
            oof_preds_by.setdefault(sid, []).append({
                "t_start": float(row["t_start"]),
                "t_end":   float(row["t_end"]),
                "score":   float(row["score"]),
                "t_alarm": float(row.get("t_alarm", row["t_start"])),
            })

    # Build stream-data objects for all OOF streams
    all_sids = sorted(oof_preds_by.keys())
    all_sd   = {sid: _StreamData(oof_preds_by[sid], gt_by.get(sid, []), durs.get(sid, 1800.0))
                for sid in all_sids}

    # Nested OOF: for each fold k, fit on the 12-other-folds OOF
    form_f1_a, form_f1_b = [], []
    fold_params_a, fold_params_b = {}, {}
    stability_a, stability_b = [], []

    if verbose:
        print("Nested OOF calibration (coarse grid)…")

    for k, (test_sid, _) in enumerate(folds):
        train_sids = [sid for sid in all_sids if sid != test_sid]
        train_sd   = [all_sd[sid] for sid in train_sids]

        f1a, pa = _grid_search(train_sd, "A", fine=False)
        f1b, pb = _grid_search(train_sd, "B", fine=False)

        form_f1_a.append(f1a)
        form_f1_b.append(f1b)
        fold_params_a[test_sid] = pa
        fold_params_b[test_sid] = pb
        stability_a.append((pa["ds"], pa["de"]))
        stability_b.append((pb["dc"], pb["k"]))

        if verbose:
            print(f"  fold {k:02d} ({test_sid}): "
                  f"A(ds={pa['ds']:+.2f},de={pa['de']:+.2f},d_min={pa['d_min']}) F1={f1a:.4f} | "
                  f"B(dc={pb['dc']:+.2f},k={pb['k']:.1f},d_min={pb['d_min']}) F1={f1b:.4f}")

    mean_a = float(np.mean(form_f1_a))
    mean_b = float(np.mean(form_f1_b))
    form   = "A" if mean_a >= mean_b else "B"
    if verbose:
        print(f"\nForm A mean={mean_a:.4f}  Form B mean={mean_b:.4f}  → Use Form {form}")

    # Stability check
    sa = np.array(stability_a)
    sb = np.array(stability_b)
    std_a = float(np.std(sa, axis=0).max())
    std_b = float(np.std(sb, axis=0).max())
    chosen_std = std_a if form == "A" else std_b
    unstable   = chosen_std > 0.15
    if verbose:
        print(f"Stability: max_std(A)={std_a:.3f}  max_std(B)={std_b:.3f}"
              + ("  ⚠️ UNSTABLE → DROP" if unstable else "  ✓ stable"))

    # Apply nested params to produce calibrated OOF
    fold_params = fold_params_a if form == "A" else fold_params_b
    cal_rows = []
    for test_sid, _ in folds:
        dur_s = durs.get(test_sid, 1800.0)
        preds = oof_preds_by.get(test_sid, [])
        p = fold_params[test_sid]
        if form == "A":
            for pred in preds:
                ps = max(0.0, pred["t_start"] + p["ds"])
                pe = min(dur_s, pred["t_end"]   + p["de"])
                if pe - ps < p["d_min"]:
                    c = (pred["t_start"] + pred["t_end"]) / 2.0
                    ps = max(0.0, c - p["d_min"] / 2)
                    pe = min(dur_s, c + p["d_min"] / 2)
                if pe > ps:
                    cal_rows.append({"stream_id": test_sid,
                                     "t_start": f"{ps:.3f}", "t_end": f"{pe:.3f}",
                                     "score": f"{pred['score']:.6f}",
                                     "t_alarm": f"{ps:.3f}"})
        else:
            for pred in preds:
                c  = (pred["t_start"] + pred["t_end"]) / 2.0 + p["dc"]
                hl = (pred["t_end"] - pred["t_start"]) / 2.0 * p["k"]
                ps = max(0.0, c - hl)
                pe = min(dur_s, c + hl)
                if pe - ps < p["d_min"]:
                    ps = max(0.0, c - p["d_min"] / 2)
                    pe = min(dur_s, c + p["d_min"] / 2)
                if pe > ps:
                    cal_rows.append({"stream_id": test_sid,
                                     "t_start": f"{ps:.3f}", "t_end": f"{pe:.3f}",
                                     "score": f"{pred['score']:.6f}",
                                     "t_alarm": f"{ps:.3f}"})

    # Global params (fine grid on full OOF)
    if verbose:
        print("\nFitting global params on full OOF (fine grid) for val/test…")
    full_sd = [all_sd[sid] for sid in all_sids]
    gf1a, gpa = _grid_search(full_sd, "A", fine=True)
    gf1b, gpb = _grid_search(full_sd, "B", fine=True)
    global_params = (gpa if form == "A" else gpb)
    global_params["oof_f1"] = gf1a if form == "A" else gf1b
    global_params["unstable"] = unstable
    if verbose:
        print(f"  Global {form}: {global_params}")

    return form, global_params, cal_rows


def apply_calibration(events: List[dict], params: dict, dur_s: float) -> List[dict]:
    """Apply saved calibration params to a list of events."""
    form  = params["form"]
    d_min = params.get("d_min", 0.0)
    out   = []
    for ev in events:
        if form == "A":
            ps = max(0.0, ev["t_start"] + params["ds"])
            pe = min(dur_s, ev["t_end"]  + params["de"])
        else:
            c  = (ev["t_start"] + ev["t_end"]) / 2.0 + params["dc"]
            hl = (ev["t_end"] - ev["t_start"]) / 2.0 * params["k"]
            ps = max(0.0, c - hl)
            pe = min(dur_s, c + hl)
        if pe - ps < d_min:
            c = (ev["t_start"] + ev["t_end"]) / 2.0
            ps = max(0.0, c - d_min / 2)
            pe = min(dur_s, c + d_min / 2)
        if pe > ps:
            out.append(dict(ev, t_start=ps, t_end=pe, t_alarm=ps))
    return out
