"""Turn window scores of one stream into fall events (``t_start, t_end, score, t_alarm``)."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import get

SUBMISSION_COLUMNS = ("stream_id", "t_start", "t_end", "score", "t_alarm")


@dataclass(frozen=True)
class PostprocessParams:
    threshold: float
    smoothing_windows: int = 1
    minimum_positive_windows: int = 1
    merge_gap_seconds: float = 0.0
    minimum_event_duration_seconds: float = 0.0
    cooldown_seconds: float = 0.0
    interval_mode: str = "center"

    def to_dict(self) -> dict:
        return asdict(self)


def postprocess_grid(cfg: dict) -> list[PostprocessParams]:
    """Every combination of the threshold and post-processing lists in the config."""

    def options(key: str, default):
        value = get(cfg, key, default)
        return value if isinstance(value, list) else [value]

    grid = itertools.product(
        options("threshold.candidate_values", 0.5),
        options("postprocessing.smoothing_windows", 1),
        options("postprocessing.minimum_positive_windows", 1),
        options("postprocessing.merge_gap_seconds", 0.0),
        options("postprocessing.minimum_event_duration_seconds", 0.0),
        options("postprocessing.cooldown_seconds", 0.0),
    )
    mode = get(cfg, "postprocessing.interval_mode", "center")
    return [
        PostprocessParams(float(t), int(s), int(p), float(g), float(d), float(c), mode) for t, s, p, g, d, c in grid
    ]


def smooth_scores(scores: np.ndarray, width: int, w_index: np.ndarray | None = None) -> np.ndarray:
    """Centred moving average over ``width`` windows (1 = unchanged); edges use fewer windows.

    With ``w_index``, each block of consecutive indices is smoothed on its own, so the
    average never reaches across a missing window.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if width <= 1 or len(scores) == 0:
        return scores.copy()
    if w_index is not None:
        breaks = np.flatnonzero(np.diff(np.asarray(w_index)) != 1) + 1
        if len(breaks):
            return np.concatenate([smooth_scores(block, width) for block in np.split(scores, breaks)])
    kernel = np.ones(int(width))
    # "full" then slice: "same" would return len(kernel) values for a block shorter than it.
    offset = (int(width) - 1) // 2
    total = np.convolve(scores, kernel, mode="full")[offset : offset + len(scores)]
    count = np.convolve(np.ones_like(scores), kernel, mode="full")[offset : offset + len(scores)]
    return total / count


def scores_to_events(
    t_start: np.ndarray,
    t_end: np.ndarray,
    w_index: np.ndarray,
    scores: np.ndarray,
    params: PostprocessParams,
    stream_duration: float,
) -> list[dict]:
    """Events for one stream. Windows must be sorted and come from a single stream.

    Steps: smooth -> threshold -> runs of consecutive windows (a gap in ``w_index``
    breaks a run) -> drop runs shorter than ``minimum_positive_windows`` -> interval
    (``center``: first centre - stride/2 to last centre + stride/2; ``window``: first
    window start to last window end) -> merge events closer than ``merge_gap_seconds``
    -> drop events shorter than ``minimum_event_duration_seconds`` -> drop events that
    start within ``cooldown_seconds`` of the previous kept event -> clip to the stream.
    ``score`` is the highest smoothed score in the event; ``t_alarm`` is the end of the
    first window above the threshold (the earliest time the alarm could be raised).
    A window whose score is NaN (not scored, e.g. an out-of-fold window that straddles
    two folds) is treated as missing: smoothing and runs both stop at it.
    """
    scored = np.isfinite(np.asarray(scores, dtype=np.float64))
    t_start = np.asarray(t_start, dtype=np.float64)[scored]
    t_end = np.asarray(t_end, dtype=np.float64)[scored]
    w_index = np.asarray(w_index)[scored]
    scores = np.asarray(scores, dtype=np.float64)[scored]
    if len(t_start) == 0:
        return []
    smoothed = smooth_scores(scores, params.smoothing_windows, w_index)
    active = smoothed >= params.threshold
    centres = 0.5 * (t_start + t_end)
    stride = float(np.median(np.diff(t_start))) if len(t_start) > 1 else float(t_end[0] - t_start[0])
    runs: list[list[float]] = []
    index = 0
    n = len(active)
    while index < n:
        if not active[index]:
            index += 1
            continue
        stop = index
        while stop + 1 < n and active[stop + 1] and w_index[stop + 1] == w_index[stop] + 1:
            stop += 1
        if stop - index + 1 >= params.minimum_positive_windows:
            if params.interval_mode == "window":
                begin, end = t_start[index], t_end[stop]
            else:
                begin, end = centres[index] - stride / 2.0, centres[stop] + stride / 2.0
            runs.append([begin, end, float(smoothed[index : stop + 1].max()), float(t_end[index])])
        index = stop + 1
    merged: list[list[float]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= params.merge_gap_seconds:
            merged[-1][1] = max(merged[-1][1], run[1])
            merged[-1][2] = max(merged[-1][2], run[2])
        else:
            merged.append(list(run))
    events: list[dict] = []
    last_end = -np.inf
    for begin, end, score, alarm in merged:
        begin, end = max(0.0, begin), min(stream_duration, end)
        if end - begin < max(params.minimum_event_duration_seconds, 1e-9):
            continue
        if begin < last_end + params.cooldown_seconds and events:
            continue
        events.append(
            {"t_start": float(begin), "t_end": float(end), "score": float(score), "t_alarm": float(min(max(alarm, begin), end))}
        )
        last_end = end
    return events


def events_for_streams(
    windows: pd.DataFrame, scores: np.ndarray, params: PostprocessParams, durations: dict[str, float]
) -> pd.DataFrame:
    """Apply ``scores_to_events`` stream by stream; events never mix streams."""
    frame = windows[["stream_id", "w_index", "t_start", "t_end"]].copy()
    frame["score"] = np.asarray(scores, dtype=np.float64)
    rows = []
    for stream_id, group in frame.groupby("stream_id", sort=True):
        group = group.sort_values("w_index")
        for event in scores_to_events(
            group["t_start"].to_numpy(),
            group["t_end"].to_numpy(),
            group["w_index"].to_numpy(),
            group["score"].to_numpy(),
            params,
            durations[stream_id],
        ):
            rows.append({"stream_id": stream_id, **event})
    return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))
