"""Read stream metadata and CIR chunks without loading a whole stream into memory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd

STREAM_FILE_RE = re.compile(r"^stream_(\d+)\.h5$")
OFFICIAL_SPLITS = ("train", "val", "test")


class DataError(RuntimeError):
    """The data on disk does not match what the pipeline expects."""


@dataclass(frozen=True)
class StreamInfo:
    stream_id: str
    split: str
    index: int
    path: Path
    n_radars: int
    n_frames: int
    n_bins: int
    dtype: str
    t_first: float
    t_last: float
    dt_min: float
    dt_max: float
    monotonic: bool
    radar_serials: tuple[int, ...]
    file_size: int
    attrs: dict


def stream_id_for(path: Path) -> str:
    """Official identifier ``<split>_XX`` from ``streams/<split>/stream_XX.h5``."""
    match = STREAM_FILE_RE.match(path.name)
    if not match:
        raise DataError(f"{path}: file name must look like stream_XX.h5")
    split = path.parent.name
    if split not in OFFICIAL_SPLITS:
        raise DataError(f"{path}: parent folder must be one of {OFFICIAL_SPLITS}")
    return f"{split}_{int(match.group(1)):02d}"


def discover_streams(streams_root: Path, include_splits: list[str]) -> list[Path]:
    streams_root = Path(streams_root)
    if not streams_root.is_dir():
        raise DataError(
            f"streams folder not found: {streams_root}. Download the package (see data/README.md) "
            "and set data.root in the config."
        )
    files: list[Path] = []
    for split in include_splits:
        folder = streams_root / split
        if not folder.is_dir():
            raise DataError(f"split folder not found: {folder}")
        found = sorted(p for p in folder.iterdir() if STREAM_FILE_RE.match(p.name))
        if not found:
            raise DataError(f"no stream_XX.h5 files in {folder}")
        files.extend(found)
    return files


def read_stream_info(path: Path) -> StreamInfo:
    """Read shapes, timestamps and attributes. The CIR array itself is not read."""
    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise DataError(f"{path}: cannot open as HDF5 ({error})") from error
    with handle:
        for name in ("cir", "t", "radar_serials"):
            if name not in handle:
                raise DataError(f"{path}: dataset '{name}' is missing")
        cir = handle["cir"]
        if cir.ndim != 3:
            raise DataError(f"{path}: cir must have 3 dimensions, found shape {cir.shape}")
        t = np.asarray(handle["t"][:], dtype=np.float64)
        serials = tuple(int(v) for v in np.asarray(handle["radar_serials"][()]).ravel())
        attrs = {key: _plain(value) for key, value in handle.attrs.items()}
        n_radars, n_frames, n_bins = (int(v) for v in cir.shape)
        dtype = str(cir.dtype)
    if len(t) != n_frames:
        raise DataError(f"{path}: t has {len(t)} values but cir has {n_frames} frames")
    diffs = np.diff(t) if len(t) > 1 else np.array([np.nan])
    return StreamInfo(
        stream_id=stream_id_for(path),
        split=path.parent.name,
        index=int(STREAM_FILE_RE.match(path.name).group(1)),
        path=path,
        n_radars=n_radars,
        n_frames=n_frames,
        n_bins=n_bins,
        dtype=dtype,
        t_first=float(t[0]) if len(t) else float("nan"),
        t_last=float(t[-1]) if len(t) else float("nan"),
        dt_min=float(np.min(diffs)),
        dt_max=float(np.max(diffs)),
        monotonic=bool(np.all(diffs > 0)),
        radar_serials=serials,
        file_size=path.stat().st_size,
        attrs=attrs,
    )


def read_timestamps(path: Path, stop: int | None = None) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["t"][:stop], dtype=np.float64)


def iter_cir_chunks(path: Path, chunk_frames: int, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first_frame, block)`` with ``block`` of shape ``(radars, n, bins)`` as float32."""
    with h5py.File(path, "r") as handle:
        cir = handle["cir"]
        total = cir.shape[1] if stop is None else min(stop, cir.shape[1])
        for start in range(0, total, chunk_frames):
            end = min(start + chunk_frames, total)
            yield start, np.asarray(cir[:, start:end, :], dtype=np.float32)


def read_cir_slice(path: Path, start: int, stop: int) -> np.ndarray:
    """Read frames ``[start, stop)`` of every radar. Meant for plots and small checks."""
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["cir"][:, start:stop, :], dtype=np.float32)


def load_events(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"events file not found: {path}")
    events = pd.read_csv(path, dtype={"stream_id": str})
    required = {"stream_id", "t_start", "t_end"}
    missing = required - set(events.columns)
    if missing:
        raise DataError(f"{path}: missing columns {sorted(missing)}")
    for column in ("ignore", "needs_review"):
        if column in events.columns:
            events[column] = events[column].map(_as_bool)
        else:
            events[column] = False
    if "quality" not in events.columns:
        events["quality"] = ""
    events["quality"] = events["quality"].fillna("").astype(str).str.strip().str.lower()
    events["flagged"] = (events["quality"] == "low") | events["needs_review"]
    return events


def load_composition(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"composition file not found: {path}")
    composition = pd.read_csv(path, dtype={"stream_id": str, "path": str, "kind": str})
    required = {"stream_id", "kind", "path", "t_start", "t_end"}
    missing = required - set(composition.columns)
    if missing:
        raise DataError(f"{path}: missing columns {sorted(missing)}")
    return composition


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value
