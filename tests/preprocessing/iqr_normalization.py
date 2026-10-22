"""IQR/MAD normalization for energy signals."""
import numpy as np


def iqr_norm(energy, eps=1e-9):
    """Normalize energy by IQR (robust to outliers)."""
    q25, q75 = np.percentile(energy, [25, 75])
    iqr = q75 - q25
    return (energy - q25) / (iqr + eps)
