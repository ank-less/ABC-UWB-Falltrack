"""Window features computed from per-frame series; the raw CIR tensor is never flattened.

The time axis is axis 1 of ``cir`` (confirmed: 50 Hz timestamps), so per-frame series
are time series and slopes/differences along them are meaningful.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

STATS = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q25",
    "q75",
    "iqr",
    "rms",
    "slope",
    "max_abs_diff",
    "peak_to_median",
)
ENERGY_SERIES = {"bg_energy", "diff_energy"}
LOG_FLOOR = 1e-12


def _stats(win: np.ndarray, enabled: list[str], rate_hz: float) -> dict[str, np.ndarray]:
    """``win`` has shape ``(windows, radars, frames)``; each stat has shape ``(windows, radars)``."""
    out: dict[str, np.ndarray] = {}
    need_quantiles = {"median", "q25", "q75", "iqr", "peak_to_median"} & set(enabled)
    if need_quantiles:
        q25, median, q75 = np.quantile(win, [0.25, 0.5, 0.75], axis=2)
    for name in enabled:
        if name == "mean":
            out[name] = win.mean(axis=2)
        elif name == "std":
            out[name] = win.std(axis=2)
        elif name == "min":
            out[name] = win.min(axis=2)
        elif name == "max":
            out[name] = win.max(axis=2)
        elif name == "median":
            out[name] = median
        elif name == "q25":
            out[name] = q25
        elif name == "q75":
            out[name] = q75
        elif name == "iqr":
            out[name] = q75 - q25
        elif name == "rms":
            out[name] = np.sqrt(np.mean(win * win, axis=2))
        elif name == "slope":
            x = np.arange(win.shape[2], dtype=np.float64)
            x -= x.mean()
            centred = win - win.mean(axis=2, keepdims=True)
            out[name] = (centred @ x) / float(x @ x) * rate_hz
        elif name == "max_abs_diff":
            out[name] = np.abs(np.diff(win, axis=2)).max(axis=2) if win.shape[2] > 1 else np.zeros(win.shape[:2])
        elif name == "peak_to_median":
            out[name] = win.max(axis=2) - median
        else:
            raise ValueError(f"unknown feature {name!r}; choose from {STATS}")
    return out


def window_features(
    values: dict[str, np.ndarray],
    starts: np.ndarray,
    frames: int,
    series: list[str],
    enabled: list[str],
    rate_hz: float,
    log_transform: bool = True,
    cross_radar: bool = True,
    context_seconds: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix of shape ``(len(starts), n_features)`` and the matching names.

    ``values[name]`` has shape ``(frames_in_stream, radars)``. Each window covers
    frames ``[start, start + frames)``. The context contrast uses only frames before
    the window (causal); it is NaN at the very start of a stream and is imputed later.
    Infinite values become NaN.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    starts = np.asarray(starts, dtype=np.int64)
    for name in series:
        if name not in values:
            raise KeyError(f"series {name!r} was not computed")
        arr = np.asarray(values[name], dtype=np.float64)
        if log_transform and name in ENERGY_SERIES:
            arr = np.log10(np.clip(arr, 0.0, None) + LOG_FLOOR)
        n_frames, n_radars = arr.shape
        if len(starts) and starts.max() + frames > n_frames:
            raise ValueError("a window extends past the end of the stream")
        view = sliding_window_view(arr, frames, axis=0)  # (n_frames - frames + 1, radars, frames)
        win = view[starts] if len(starts) else np.zeros((0, n_radars, frames))
        with np.errstate(invalid="ignore"):  # NaN inputs give NaN features, imputed later
            stats = _stats(win, list(enabled), rate_hz)
        for stat_name, matrix in stats.items():
            for radar in range(n_radars):
                columns.append(matrix[:, radar])
                names.append(f"{name}_r{radar}_{stat_name}")
        if cross_radar:
            for stat_name in ("mean", "max"):
                matrix = stats[stat_name] if stat_name in stats else _stats(win, [stat_name], rate_hz)[stat_name]
                columns += [matrix.max(axis=1), matrix.mean(axis=1)]
                names += [f"{name}_radarmax_{stat_name}", f"{name}_radarmean_{stat_name}"]
        if context_seconds:
            context = int(round(context_seconds * rate_hz))
            cumulative = np.vstack([np.zeros((1, n_radars)), np.cumsum(arr, axis=0)])
            begin = np.maximum(starts - context, 0)
            count = (starts - begin).astype(np.float64)[:, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                before = (cumulative[starts] - cumulative[begin]) / count
            inside = win.mean(axis=2)
            contrast = np.where(count > 0, inside - before, np.nan)
            for radar in range(n_radars):
                columns.append(contrast[:, radar])
                names.append(f"{name}_r{radar}_contrast")
    matrix = np.column_stack(columns) if columns else np.zeros((len(starts), 0))
    matrix[~np.isfinite(matrix)] = np.nan
    if matrix.shape != (len(starts), len(names)):
        raise AssertionError(f"feature matrix shape {matrix.shape} does not match {len(names)} names")
    return matrix, names
