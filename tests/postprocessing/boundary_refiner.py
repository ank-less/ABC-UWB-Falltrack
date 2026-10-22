"""Boundary refinement using energy half-max and Gaussian heatmap."""
import numpy as np

RATE_HZ = 50.0


def refine_boundary(energy, t_start, t_end, stream_dur_s, context_s=2.0, frac=0.4):
    """Refine event boundaries using half-max of energy within extended window."""
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
    """Refine event boundaries using weighted Gaussian heatmap (mean ± k*σ)."""
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
