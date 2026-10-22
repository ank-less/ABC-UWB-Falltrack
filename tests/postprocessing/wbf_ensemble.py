"""1D late-fusion / majority-vote ensemble for multi-radar detection."""


def late_fusion_majority(events_r, min_support=2):
    """Keep events detected by at least min_support radars; deduplicate overlaps."""
    n_radars = len(events_r)
    confirmed = []
    for r in range(n_radars):
        for ev in events_r[r]:
            support = 1
            for r2 in range(n_radars):
                if r2 == r:
                    continue
                for ev2 in events_r[r2]:
                    overlap = (min(ev["t_end"], ev2["t_end"])
                               - max(ev["t_start"], ev2["t_start"]))
                    if overlap > 0:
                        support += 1
                        break
            if support >= min_support:
                confirmed.append(dict(ev))
    if not confirmed:
        return []
    confirmed.sort(key=lambda e: e["t_start"])
    merged = [confirmed[0]]
    for ev in confirmed[1:]:
        if ev["t_start"] < merged[-1]["t_end"]:
            merged[-1]["t_end"]   = max(merged[-1]["t_end"], ev["t_end"])
            merged[-1]["score"]   = max(merged[-1]["score"], ev["score"])
        else:
            merged.append(ev)
    return merged
