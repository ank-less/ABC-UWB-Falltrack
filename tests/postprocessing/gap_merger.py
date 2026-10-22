"""Merge events with small temporal gaps."""


def gap_merge(events, max_gap=1.5):
    """Merge consecutive events separated by at most max_gap seconds."""
    if not events:
        return []
    evs = sorted(events, key=lambda e: e["t_start"])
    m = [dict(evs[0])]
    for e in evs[1:]:
        if e["t_start"] - m[-1]["t_end"] <= max_gap:
            m[-1]["t_end"] = max(m[-1]["t_end"], e["t_end"])
            m[-1]["score"] = max(m[-1]["score"], e["score"])
        else:
            m.append(dict(e))
    return m
