"""Envelope computation for GĐ4.

Based on physics_checks: axis_layout = 'range_368'
(raw range profile, not IQ split — bin 0..367 are all range bins).
"""
from __future__ import annotations
import numpy as np


def envelope(cir: np.ndarray, layout: str = "range_368") -> np.ndarray:
    """Compute amplitude envelope of CIR.

    Args:
        cir:    (3, T, 368) float32 array
        layout: one of 'range_368' | 'iq_concat_184' | 'iq_interleaved_184'

    Returns:
        amp: (3, T, N_bins) non-negative amplitude array
             N_bins = 368 for range_368, 184 for IQ variants.
    """
    x = cir.astype(np.float64)
    if layout == "range_368":
        return np.abs(x)
    elif layout == "iq_concat_184":
        # First 184 bins = I, last 184 = Q
        I = x[:, :, :184]
        Q = x[:, :, 184:]
        return np.sqrt(I ** 2 + Q ** 2)
    elif layout == "iq_interleaved_184":
        # Even indices = I, odd = Q
        I = x[:, :, 0::2]
        Q = x[:, :, 1::2]
        bins = min(I.shape[2], Q.shape[2])
        return np.sqrt(I[:, :, :bins] ** 2 + Q[:, :, :bins] ** 2)
    else:
        raise ValueError(f"Unknown layout: {layout}")
