"""Temporal NMS / fragment merging for pipeline_22.

Suppresses predictions that are too close in time to a higher-score prediction
(within K seconds of its centre).
"""
from __future__ import annotations

from typing import List


def temporal_nms(events: List[dict], K: float) -> List[dict]:
    """Suppress low-score events within K seconds of a higher-score centre.

    Args:
        events: list of dicts with t_start, t_end, score, t_alarm
        K:      suppression radius in seconds
    Returns:
        Filtered list (subset), sorted by t_start.
    """
    if not events:
        return []
    # Sort by score descending
    sorted_ev = sorted(events, key=lambda e: -e["score"])
    kept = []
    suppressed = set()
    for i, ev in enumerate(sorted_ev):
        if i in suppressed:
            continue
        kept.append(ev)
        c_i = (ev["t_start"] + ev["t_end"]) / 2.0
        for j in range(i + 1, len(sorted_ev)):
            if j in suppressed:
                continue
            c_j = (sorted_ev[j]["t_start"] + sorted_ev[j]["t_end"]) / 2.0
            if abs(c_i - c_j) <= K:
                suppressed.add(j)
    return sorted(kept, key=lambda e: e["t_start"])
