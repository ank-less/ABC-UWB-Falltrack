"""Shrink event windows to a target duration around the energy peak."""
import numpy as np

RATE_HZ = 50.0


def shrink_to_peak(events, energy, stream_dur_s, target_dur=1.6):
    """Shrink each event to target_dur seconds centered on the energy peak."""
    T = len(energy)
    shrunken = []
    for ev in events:
        s = int(ev["t_start"] * RATE_HZ)
        e = min(int(ev["t_end"] * RATE_HZ) + 1, T)
        if s >= e:
            shrunken.append(ev)
            continue
        peak_idx = s + int(np.argmax(energy[s:e]))
        half = target_dur / 2.0
        new_start = max(0.0, peak_idx / RATE_HZ - half)
        new_end   = min(stream_dur_s, peak_idx / RATE_HZ + half)
        if new_end - new_start < 0.5:
            shrunken.append(ev)
            continue
        shrunken.append({
            "t_start": new_start,
            "t_end":   new_end,
            "score":   ev["score"],
            "t_alarm": new_start,
        })
    return shrunken
