"""HSMM-style 3-state detector: normal → fall → lying."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abc2027.baseline import runs_to_events

RATE_HZ = 50.0


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


def hsmm_events(energy, rate_hz=RATE_HZ):
    """Detect falls via 3-state HSMM: fall (high energy burst) + lying extension."""
    fall_q = float(np.percentile(energy, 95))
    fall_mask = energy > fall_q
    lying_frames = int(2.0 * rate_hz)
    extended = forward_dilate(fall_mask, lying_frames)
    return runs_to_events(
        extended,
        energy / (fall_q + 1e-9),
        min_duration=0.6,
        merge_gap=1.0,
    )
