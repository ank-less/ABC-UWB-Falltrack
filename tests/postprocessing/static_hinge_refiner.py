"""Lightweight static CIR boundary refiner (pipeline 29).

Ground truth annotation method (from README):
  "Bounds placed from the change in the static CIR profile,
   on the radar where that change is largest, after a short
   high-pass of the background."

Memory-efficient: loads only [t_start-ctx_s, t_end+ctx_s] slice per candidate.
~1.3 MB per candidate (6s × 50fps × 3 radars × 368 bins × 4 bytes).
No OOM risk for typical 20-30 candidates per stream.

Detection-neutral: does NOT change stream_id, score, or detection count.
Only modifies t_start / t_end.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

RATE_HZ = 50.0
N_BINS  = 368
TOP_K   = 16  # top-K active bins for g_r computation
RISE_THR = 0.15  # g_norm threshold for onset detection
FALL_THR = 0.20  # g_norm threshold for offset detection
MAX_SHIFT = 2.0  # discard refinement if shift > 2s (unreliable)
MIN_DUR   = 0.3  # minimum refined duration


def _static_refine_single(
    stream_path: Path,
    t_start0: float,
    t_end0: float,
    stream_dur_s: float,
    ctx_s: float = 3.0,
    median_w_s: float = 0.2,
    ema_tau_s: float = 1.0,
) -> Tuple[float, float]:
    """Refine one event boundary using static CIR profile change.

    Loads only a slice of the h5 file to avoid OOM.
    Returns (new_t_start, new_t_end), falls back to original on failure.
    """
    ctx_frames = int(ctx_s * RATE_HZ)
    ws = max(0, int((t_start0 - ctx_s) * RATE_HZ))
    we = min(int(stream_dur_s * RATE_HZ), int((t_end0 + ctx_s) * RATE_HZ) + 1)

    if we - ws < 20:
        return t_start0, t_end0

    try:
        with h5py.File(stream_path, "r") as f:
            cir_slice = f["cir"][:, ws:we, :].astype(np.float64)  # (3, T, 368)
    except Exception:
        return t_start0, t_end0

    slice_T = cir_slice.shape[1]
    if slice_T < 20:
        return t_start0, t_end0

    # Envelope (abs — layout is range_368, not IQ)
    env = np.abs(cir_slice)  # (3, T, 368)

    # Moving average ≈ static profile (faster than median, close enough)
    median_w = max(1, int(median_w_s * RATE_HZ))
    s_r = uniform_filter1d(env, size=median_w, axis=1)  # (3, T, 368)

    # High-pass: subtract causal EMA with τ = ema_tau_s
    tau_frames = max(1, int(ema_tau_s * RATE_HZ))
    alpha = 1.0 - np.exp(-1.0 / tau_frames)
    ema = np.empty_like(s_r)
    ema[:, 0, :] = s_r[:, 0, :]
    for t in range(1, slice_T):
        ema[:, t, :] = alpha * s_r[:, t, :] + (1.0 - alpha) * ema[:, t - 1, :]
    h_r = s_r - ema  # (3, T, 368)

    # g_r(t) = L2 norm over top-K active bins per radar
    energy_per_bin = np.mean(h_r ** 2, axis=1)  # (3, 368)
    g_r = np.zeros((3, slice_T))
    for r in range(3):
        top_k_idx = np.argsort(energy_per_bin[r])[-TOP_K:]
        g_r[r] = np.sqrt(np.mean(h_r[r][:, top_k_idx] ** 2, axis=1))

    # r* = radar where g is largest within event region
    ev_s = max(0, int((t_start0 - ws / RATE_HZ) * RATE_HZ))
    ev_e = min(slice_T, int((t_end0   - ws / RATE_HZ) * RATE_HZ) + 1)
    ev_s = max(0, min(ev_s, slice_T - 1))
    ev_e = max(ev_s + 1, min(ev_e, slice_T))
    r_star = int(np.argmax(g_r[:, ev_s:ev_e].sum(axis=1)))
    g = gaussian_filter1d(g_r[r_star], sigma=3.0)

    # Normalize to [0, 1] relative to pre-event baseline
    pre_end = max(1, ev_s)
    g_base = float(np.median(g[:pre_end]))
    g_peak = float(g.max())
    denom  = g_peak - g_base
    if denom < 1e-12:
        return t_start0, t_end0
    g_norm = (g - g_base) / denom

    # t_start_new: first frame where g_norm >= RISE_THR, searching backwards from ev_s
    search_s = max(0, ev_s - int(1.5 * RATE_HZ))
    seg_rise = g_norm[search_s: ev_s + int(0.4 * RATE_HZ)]
    idxs_rise = np.where(seg_rise >= RISE_THR)[0]
    if len(idxs_rise) > 0:
        new_start_frame = ws + search_s + int(idxs_rise[0])
    else:
        new_start_frame = ws + ev_s

    # t_end_new: first frame after ev_e where g_norm drops below FALL_THR
    search_e = min(slice_T, ev_e + int(2.0 * RATE_HZ))
    seg_fall = g_norm[ev_e:search_e]
    idxs_fall = np.where(seg_fall < FALL_THR)[0]
    if len(idxs_fall) > 0:
        new_end_frame = ws + ev_e + int(idxs_fall[0])
    else:
        new_end_frame = ws + ev_e

    new_t_start = new_start_frame / RATE_HZ
    new_t_end   = new_end_frame   / RATE_HZ

    # Quality / sanity checks
    if new_t_end <= new_t_start:
        return t_start0, t_end0
    if new_t_end - new_t_start < MIN_DUR:
        return t_start0, t_end0
    if abs(new_t_start - t_start0) > MAX_SHIFT:
        return t_start0, t_end0
    if abs(new_t_end - t_end0) > MAX_SHIFT:
        return t_start0, t_end0

    new_t_start = max(0.0, min(stream_dur_s, new_t_start))
    new_t_end   = max(0.0, min(stream_dur_s, new_t_end))
    return new_t_start, new_t_end


def apply_static_refiner(
    events: List[dict],
    stream_path: Path,
    stream_dur_s: float,
    ctx_s: float = 3.0,
) -> List[dict]:
    """Apply static hinge refinement to a list of detected events.

    Detection-neutral: stream_id, score, and event count are unchanged.
    Only t_start, t_end, t_alarm may change.
    """
    refined = []
    for ev in events:
        ns, ne = _static_refine_single(
            stream_path, ev["t_start"], ev["t_end"], stream_dur_s, ctx_s=ctx_s,
        )
        refined.append({
            "t_start": ns,
            "t_end":   ne,
            "score":   ev["score"],
            "t_alarm": ns,
        })
    # Assertion: detection count unchanged
    assert len(refined) == len(events), "static_refiner changed event count"
    return refined
