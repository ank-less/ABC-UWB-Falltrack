"""Stream-level manifest and segment table. No CIR values are stored here."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

from .config import data_path, get
from .data_io import DataError, discover_streams, load_composition, load_events, read_stream_info

LOG = logging.getLogger(__name__)

MANIFEST_COLUMNS = (
    "record_id",
    "stream_id",
    "official_split",
    "stream_index",
    "file_path",
    "file_size_bytes",
    "n_radars",
    "n_frames",
    "n_bins",
    "dtype",
    "rate_hz",
    "start_time",
    "end_time",
    "duration_s",
    "dt_min",
    "dt_max",
    "timestamps_monotonic",
    "radar_serials",
    "n_events",
    "n_events_flagged",
    "fall_seconds",
    "n_segments",
    "n_subject_groups",
    "subject_groups",
)

# Source paths, as written in streams_composition.csv:
#   falls/Dataset 3/Test_P3_3/Chute03_152.json   (fall and "other" recordings)
#   activities/Faire_pates_9.json                (activities of daily living)
FALL_PATH_RE = re.compile(r"^falls/Dataset (?P<dataset>\d+)/Test_P(?P<position>\d+)_(?P<person>\d+)/")
ADL_PATH_RE = re.compile(r"^activities/(?P<activity>.+)_(?P<subject>\d+)\.json$")


def subject_group(kind: str, path: str) -> str:
    """Grouping key for leakage-safe CV.

    Hypothesis checked on the package: the trailing number of ``Test_Px_y`` and the
    suffix of activity files are disjoint between official train and val, so they
    behave like person identifiers. Fall people are grouped across the three mat
    positions (``Dataset 1-3``), which is the coarser and safer choice.
    """
    path = str(path).replace("\\", "/")
    fall = FALL_PATH_RE.match(path)
    if fall:
        return f"fall_person_{int(fall.group('person'))}"
    adl = ADL_PATH_RE.match(path)
    if adl:
        return f"adl_subject_{int(adl.group('subject'))}"
    raise DataError(f"cannot derive a subject group from {kind!r} source path {path!r}")


def segment_table(composition: pd.DataFrame) -> pd.DataFrame:
    segments = composition.copy()
    segments["subject_group"] = [subject_group(k, p) for k, p in zip(segments["kind"], segments["path"])]
    segments["corpus"] = segments["subject_group"].str.split("_").str[0]
    segments["official_split"] = segments["stream_id"].str.rsplit("_", n=1).str[0]
    return segments.sort_values(["stream_id", "t_start"]).reset_index(drop=True)


def _relative(path: Path) -> str:
    try:
        return Path(os.path.relpath(path)).as_posix()
    except ValueError:  # another drive on Windows
        return Path(path).as_posix()


def build_manifest(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(manifest, segments, events)`` for the configured splits."""
    rate = float(get(cfg, "data.rate_hz"))
    files = discover_streams(data_path(cfg, "streams_dir"), list(get(cfg, "data.include_splits")))
    events = load_events(data_path(cfg, "events_csv"))
    segments = segment_table(load_composition(data_path(cfg, "composition_csv")))
    rows = []
    for path in files:
        info = read_stream_info(path)
        stream_events = events[events["stream_id"] == info.stream_id]
        stream_segments = segments[segments["stream_id"] == info.stream_id]
        groups = sorted(stream_segments["subject_group"].unique())
        rows.append(
            {
                "record_id": info.stream_id,
                "stream_id": info.stream_id,
                "official_split": info.split,
                "stream_index": info.index,
                "file_path": _relative(info.path),
                "file_size_bytes": info.file_size,
                "n_radars": info.n_radars,
                "n_frames": info.n_frames,
                "n_bins": info.n_bins,
                "dtype": info.dtype,
                "rate_hz": rate,
                "start_time": info.t_first,
                "end_time": info.t_last,
                "duration_s": info.n_frames / rate,
                "dt_min": info.dt_min,
                "dt_max": info.dt_max,
                "timestamps_monotonic": info.monotonic,
                "radar_serials": ";".join(str(s) for s in info.radar_serials),
                "n_events": int(len(stream_events)),
                "n_events_flagged": int(stream_events["flagged"].sum()),
                "fall_seconds": float((stream_events["t_end"] - stream_events["t_start"]).sum()),
                "n_segments": int(len(stream_segments)),
                "n_subject_groups": len(groups),
                "subject_groups": ";".join(groups),
            }
        )
        LOG.info("manifest: %s %d frames, %d events", info.stream_id, info.n_frames, len(stream_events))
    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    check_manifest_schema(manifest)
    known = set(manifest["stream_id"])
    return manifest, segments[segments["stream_id"].isin(known)].reset_index(drop=True), events


def check_manifest_schema(manifest: pd.DataFrame) -> None:
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise DataError(f"manifest is missing columns: {missing}")
    if manifest["record_id"].duplicated().any():
        raise DataError("manifest has duplicated record_id values")
    if manifest.empty:
        raise DataError("manifest is empty")


def save_manifest(manifest: pd.DataFrame, segments: pd.DataFrame, manifest_path: Path, segments_path: Path) -> None:
    for path, frame in ((manifest_path, manifest), (segments_path, segments)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        LOG.info("wrote %s (%d rows)", path, len(frame))
