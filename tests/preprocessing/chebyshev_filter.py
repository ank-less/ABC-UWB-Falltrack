"""Chebyshev bandpass filter for UWB CIR preprocessing."""
import numpy as np
from scipy.signal import cheby1, sosfilt, sosfilt_zi

RATE_HZ = 50.0


def chebyshev_bp(cir, low=0.5, high=15.0, order=4):
    """Apply Chebyshev Type-I bandpass filter to CIR tensor (R, T, B)."""
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
