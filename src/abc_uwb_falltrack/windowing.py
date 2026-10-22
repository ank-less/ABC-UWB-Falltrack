"""Fixed-length windows inside one stream, labelled by temporal overlap with falls."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ConfigError, get

LOG = logging.getLogger(__name__)

POSITIVE, NEGATIVE, AMBIGUOUS = 1, 0, -1


@dataclass(frozen=True)
class WindowSpec:
    duration_s: float
    stride_s: float
    rate_hz: float
    minimum_valid_fraction: float = 1.0
    drop_last: bool = True

    @property
    def frames(self) -> int:
        return _whole_frames(self.duration_s, self.rate_hz, "windowing.duration_seconds")

    @property
    def step(self) -> int:
        return _whole_frames(self.stride_s, self.rate_hz, "windowing.stride_seconds")


def _whole_frames(seconds: float, rate_hz: float, name: str) -> int:
    frames = seconds * rate_hz
    if frames < 1 or abs(frames - round(frames)) > 1e-6:
        raise ConfigError(f"{name}={seconds} s is not a whole number of frames at {rate_hz} Hz")
    return int(round(frames))


def window_spec(cfg: dict) -> WindowSpec:
    return WindowSpec(
        duration_s=float(get(cfg, "windowing.duration_seconds")),
        stride_s=float(get(cfg, "windowing.stride_seconds")),
        rate_hz=float(get(cfg, "data.rate_hz")),
        minimum_valid_fraction=float(get(cfg, "windowing.minimum_valid_fraction", 1.0)),
        drop_last=bool(get(cfg, "windowing.drop_last", True)),
    )


def window_starts(n_frames: int, spec: WindowSpec) -> np.ndarray:
    """Start frames of windows that fit entirely inside ``[0, n_frames)``.

    With ``drop_last: false`` one extra window aligned to the stream end covers the tail.
    Windows never extend past the stream, so they never mix two streams.
    """
    if n_frames < spec.frames:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n_frames - spec.frames + 1, spec.step, dtype=np.int64)
    if not spec.drop_last and starts[-1] + spec.frames < n_frames:
        starts = np.append(starts, n_frames - spec.frames)
    return starts


def make_windows(stream_id: str, timestamps: np.ndarray, valid: np.ndarray, spec: WindowSpec) -> pd.DataFrame:
    """Window table for one stream. Times come from the stream's own timestamps."""
    n_frames = len(timestamps)
    if len(valid) != n_frames:
        raise ValueError("valid mask and timestamps differ in length")
    starts = window_starts(n_frames, spec)
    stops = starts + spec.frames
    step_s = 1.0 / spec.rate_hz
    good = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    valid_fraction = (good[stops] - good[starts]) / spec.frames if len(starts) else np.zeros(0)
    t_start = timestamps[starts] if len(starts) else np.zeros(0)
    t_end = timestamps[stops - 1] + step_s if len(starts) else np.zeros(0)
    table = pd.DataFrame(
        {
            "stream_id": stream_id,
            "w_index": np.arange(len(starts), dtype=np.int64),
            "f_start": starts,
            "f_stop": stops,
            "t_start": t_start,
            "t_end": t_end,
            "t_center": 0.5 * (t_start + t_end),
            "valid_fraction": valid_fraction,
        }
    )
    table["keep"] = table["valid_fraction"] >= spec.minimum_valid_fraction
    dropped = int((~table["keep"]).sum())
    if dropped:
        LOG.warning("%s: %d windows below minimum_valid_fraction are dropped", stream_id, dropped)
    return table


def label_windows(
    t_start: np.ndarray,
    t_end: np.ndarray,
    events: pd.DataFrame,
    positive_threshold: float,
    negative_max: float = 0.0,
    flagged_policy: str = "positive",
) -> tuple[np.ndarray, np.ndarray]:
    """Label windows from fall intervals by time overlap.

    ``coverage = overlap / min(window length, fall length)``, so a short fall fully
    inside a window and a window fully inside a long fall both reach 1.
    Positive: coverage >= ``positive_threshold``. Negative: coverage <= ``negative_max``
    for every fall (with the default 0, the window does not touch any fall; touching
    end points do not count). Everything else is ambiguous. With
    ``flagged_policy='ambiguous'``, windows overlapping a ``flagged`` fall are ambiguous
    unless an unflagged fall makes them positive.
    """
    t_start = np.asarray(t_start, dtype=np.float64)
    t_end = np.asarray(t_end, dtype=np.float64)
    length = t_end - t_start
    best_any = np.zeros(len(t_start))
    best_clean = np.zeros(len(t_start))
    flagged = events["flagged"].to_numpy(bool) if "flagged" in events else np.zeros(len(events), bool)
    for ev_start, ev_end, is_flagged in zip(events["t_start"].to_numpy(float), events["t_end"].to_numpy(float), flagged):
        overlap = np.minimum(t_end, ev_end) - np.maximum(t_start, ev_start)
        coverage = np.clip(overlap, 0.0, None) / np.minimum(length, ev_end - ev_start)
        best_any = np.maximum(best_any, coverage)
        if not (is_flagged and flagged_policy == "ambiguous"):
            best_clean = np.maximum(best_clean, coverage)
    labels = np.full(len(t_start), AMBIGUOUS, dtype=np.int8)
    labels[best_any <= negative_max] = NEGATIVE
    labels[best_clean >= positive_threshold] = POSITIVE
    return labels, best_any
