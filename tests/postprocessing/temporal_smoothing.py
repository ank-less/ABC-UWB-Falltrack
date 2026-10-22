"""Gaussian temporal smoothing for energy signals."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

RATE_HZ = 50.0


def smooth(energy, sigma_s=0.3):
    """Apply Gaussian smoothing with sigma in seconds."""
    return gaussian_filter1d(energy.astype(np.float64), sigma=sigma_s * RATE_HZ)
