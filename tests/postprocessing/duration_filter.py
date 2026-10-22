"""Filter events by minimum duration."""


def dur_filter(events, min_dur=1.0):
    """Remove events shorter than min_dur seconds."""
    return [e for e in events if e["t_end"] - e["t_start"] >= min_dur]
