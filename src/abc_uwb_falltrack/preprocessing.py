"""Stateless signal preprocessing, streamed chunk by chunk.

Nothing here is fitted on data: the background filter is causal and runs on each
stream independently, so it cannot leak information between splits. Fitted
transforms (imputation, scaling) live inside the sklearn pipelines of ``modeling``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .data_io import iter_cir_chunks

LOG = logging.getLogger(__name__)

SERIES = ("bg_energy", "diff_energy", "range_centroid")
CACHE_VERSION = "frame-series-v1"


class CausalBackgroundRemover:
    """Subtract a causal exponential moving average along time, per radar and per bin.

    ``s_t = alpha * s_{t-1} + (1 - alpha) * x_t`` with ``alpha = exp(-1 / (rate * tau))``,
    started as if the first frame had been held forever. The state carries over
    between chunks, so chunked and whole-stream processing give the same result.
    """

    def __init__(self, tau_s: float, rate_hz: float):
        if tau_s <= 0 or rate_hz <= 0:
            raise ValueError("tau_s and rate_hz must be positive")
        alpha = float(np.exp(-1.0 / (rate_hz * tau_s)))
        self.b = np.array([1.0 - alpha])
        self.a = np.array([1.0, -alpha])
        self.steady = float(lfilter_zi(self.b, self.a)[0])
        self.state: np.ndarray | None = None

    def process(self, block: np.ndarray) -> np.ndarray:
        """``block`` has shape ``(radars, frames, bins)``; returns the float64 residual."""
        x = np.asarray(block, dtype=np.float64)
        if self.state is None:
            self.state = self.steady * x[:, :1, :]
        average, self.state = lfilter(self.b, self.a, x, axis=1, zi=self.state)
        return x - average


def frame_series(
    path: Path,
    rate_hz: float,
    tau_s: float,
    chunk_frames: int,
    stop: int | None = None,
    series: tuple[str, ...] = SERIES,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Per-frame, per-radar summaries of one stream, read in chunks.

    Returns ``(values, valid, stats)``: ``values[name]`` has shape ``(frames, radars)``,
    ``valid`` marks frames where every CIR value was finite. Non-finite values are
    replaced by 0 before filtering and counted in ``stats``.
    """
    unknown = set(series) - set(SERIES)
    if unknown:
        raise ValueError(f"unknown series {sorted(unknown)}; choose from {SERIES}")
    remover = CausalBackgroundRemover(tau_s, rate_hz)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in series}
    valid_parts: list[np.ndarray] = []
    previous: np.ndarray | None = None
    nonfinite_values = 0
    bins = None
    for _start, block in iter_cir_chunks(path, chunk_frames, stop):
        finite = np.isfinite(block)
        if not finite.all():
            nonfinite_values += int((~finite).sum())
            block = np.where(finite, block, 0.0).astype(np.float32)
        valid_parts.append(finite.all(axis=(0, 2)))
        residual = remover.process(block)
        power = residual * residual
        if "bg_energy" in parts:
            parts["bg_energy"].append(power.sum(axis=2).T)
        if "diff_energy" in parts:
            head = block[:, :1, :] if previous is None else previous
            delta = np.diff(np.concatenate([head, block], axis=1).astype(np.float64), axis=1)
            parts["diff_energy"].append((delta * delta).sum(axis=2).T)
        if "range_centroid" in parts:
            if bins is None:
                bins = np.arange(block.shape[2], dtype=np.float64)
            total = power.sum(axis=2)
            parts["range_centroid"].append(((power * bins).sum(axis=2) / np.maximum(total, 1e-30)).T)
        previous = block[:, -1:, :]
    if not valid_parts:
        raise ValueError(f"{path}: no frames read")
    values = {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}
    valid = np.concatenate(valid_parts)
    stats = {"frames": int(len(valid)), "invalid_frames": int((~valid).sum()), "nonfinite_values": nonfinite_values}
    return values, valid, stats


def _cache_key(path: Path, stop: int | None, rate_hz: float, tau_s: float, series: tuple[str, ...]) -> str:
    stat = Path(path).stat()
    payload = json.dumps(
        [CACHE_VERSION, Path(path).name, Path(path).parent.name, stat.st_size, stat.st_mtime_ns, stop, rate_hz, tau_s, list(series)]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def cached_frame_series(
    path: Path,
    rate_hz: float,
    tau_s: float,
    chunk_frames: int,
    stop: int | None,
    series: tuple[str, ...],
    cache_dir: Path | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """``frame_series`` with an optional on-disk cache of the (small) per-frame summaries."""
    if cache_dir is None:
        return frame_series(path, rate_hz, tau_s, chunk_frames, stop, series)
    cache_dir = Path(cache_dir)
    target = cache_dir / f"{Path(path).parent.name}_{Path(path).stem}_{_cache_key(path, stop, rate_hz, tau_s, series)}.npz"
    if target.is_file():
        with np.load(target, allow_pickle=False) as data:
            values = {name: data[name] for name in series}
            valid = data["valid"]
            stats = json.loads(str(data["stats"]))
        stats["cache"] = "hit"
        return values, valid, stats
    values, valid, stats = frame_series(path, rate_hz, tau_s, chunk_frames, stop, series)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez(temporary, valid=valid, stats=json.dumps(stats), **values)
    temporary.replace(target)
    stats["cache"] = "miss"
    return values, valid, stats
