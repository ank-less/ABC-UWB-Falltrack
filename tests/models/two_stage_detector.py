"""Two-stage detector: coarse energy candidates → LightGBM gate confirmation."""
import numpy as np

RATE_HZ = 50.0


def two_stage_gate(candidates, lgb_scores, T, rate_hz=RATE_HZ, lgb_threshold=0.2):
    """Filter energy candidates using LightGBM score gate."""
    events = []
    for ev in candidates:
        s = int(ev["t_start"] * rate_hz)
        e = min(int(ev["t_end"] * rate_hz) + 1, T)
        if s >= e:
            continue
        max_lgb = float(np.max(lgb_scores[s:e]))
        if max_lgb > lgb_threshold:
            ev = dict(ev)
            ev["score"] = max_lgb
            events.append(ev)
    return events
