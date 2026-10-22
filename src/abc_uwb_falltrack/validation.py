"""Schema, timestamp, label and split checks. Errors stop the run when ``strict``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import get

LOG = logging.getLogger(__name__)


class ValidationFailed(RuntimeError):
    """Data validation found errors in strict mode."""


class LeakageError(RuntimeError):
    """The same group, stream or window was found on both sides of a split."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.errors.append(message)
        LOG.error(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        LOG.warning(message)

    def to_dict(self) -> dict:
        return {"ok": not self.errors, "errors": self.errors, "warnings": self.warnings, "details": self.details}

    def raise_if_failed(self, strict: bool) -> None:
        if self.errors and strict:
            raise ValidationFailed(
                f"{len(self.errors)} validation error(s); first: {self.errors[0]}. "
                "Fix the data or set validation.strict: false to continue with warnings."
            )


def validate_manifest(manifest: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    rate = float(get(cfg, "data.rate_hz"))
    step = 1.0 / rate
    tolerance = float(get(cfg, "validation.timestamp_tolerance_s", 1e-6))
    radars = int(get(cfg, "data.expected_radars"))
    bins = int(get(cfg, "data.expected_bins"))
    for row in manifest.itertuples(index=False):
        where = f"{row.stream_id} ({row.file_path})"
        if row.n_radars != radars or row.n_bins != bins:
            report.error(f"{where}: cir shape (radars={row.n_radars}, bins={row.n_bins}), expected ({radars}, {bins})")
        if row.dtype != "float32":
            report.error(f"{where}: cir dtype {row.dtype}, expected float32 (real-valued)")
        if row.n_frames < 2:
            report.error(f"{where}: only {row.n_frames} frames")
        if get(cfg, "validation.check_timestamps", True):
            if not row.timestamps_monotonic:
                report.error(f"{where}: timestamps are not strictly increasing")
            if abs(row.dt_min - step) > tolerance or abs(row.dt_max - step) > tolerance:
                report.error(
                    f"{where}: timestamp step in [{row.dt_min:.6f}, {row.dt_max:.6f}] s, expected {step:.6f} s "
                    f"(data.rate_hz={rate}); gaps or another rate would break windowing"
                )
            if abs(row.start_time) > tolerance:
                report.warn(f"{where}: first timestamp is {row.start_time}, expected 0")
    report.details["streams"] = int(len(manifest))
    report.details["frames"] = int(manifest["n_frames"].sum())
    report.details["hours"] = float(manifest["duration_s"].sum() / 3600.0)


def validate_events(events: pd.DataFrame, manifest: pd.DataFrame, report: ValidationReport) -> None:
    durations = dict(zip(manifest["stream_id"], manifest["duration_s"]))
    splits = set(manifest["official_split"])
    relevant = events[events["stream_id"].str.rsplit("_", n=1).str[0].isin(splits)]
    unknown = sorted(set(relevant["stream_id"]) - set(durations))
    if unknown:
        report.warn(f"events reference {len(unknown)} stream(s) that are not on disk (partial download?): {unknown[:5]}")
        relevant = relevant[relevant["stream_id"].isin(durations)]
    bad_order = relevant[~(relevant["t_start"] < relevant["t_end"])]
    for row in bad_order.itertuples(index=False):
        report.error(f"event {getattr(row, 'event_id', '?')} in {row.stream_id}: t_start >= t_end")
    for row in relevant.itertuples(index=False):
        duration = durations.get(row.stream_id)
        if duration is not None and (row.t_start < 0 or row.t_end > duration + 1e-6):
            report.error(f"event {getattr(row, 'event_id', '?')} in {row.stream_id} lies outside [0, {duration:.2f}] s")
    overlaps = 0
    for _, group in relevant.sort_values("t_start").groupby("stream_id"):
        starts = group["t_start"].to_numpy()
        ends = group["t_end"].to_numpy()
        overlaps += int(np.sum(starts[1:] < ends[:-1]))
    if overlaps:
        report.warn(f"{overlaps} pairs of fall events overlap inside a stream")
    lengths = relevant["t_end"] - relevant["t_start"]
    report.details["events"] = {
        "count": int(len(relevant)),
        "flagged": int(relevant["flagged"].sum()),
        "ignore": int(relevant["ignore"].sum()),
        "duration_s_min": float(lengths.min()) if len(lengths) else None,
        "duration_s_median": float(lengths.median()) if len(lengths) else None,
        "duration_s_max": float(lengths.max()) if len(lengths) else None,
        "by_split": relevant["stream_id"].str.rsplit("_", n=1).str[0].value_counts().to_dict(),
    }


def validate_segments(segments: pd.DataFrame, manifest: pd.DataFrame, report: ValidationReport) -> None:
    durations = dict(zip(manifest["stream_id"], manifest["duration_s"]))
    missing = sorted(set(durations) - set(segments["stream_id"]))
    if missing:
        report.error(f"streams without composition rows (needed for grouped CV): {missing[:5]}")
    outside = segments[(segments["t_start"] < -1e-6) | (segments["t_end"] > segments["stream_id"].map(durations) + 0.5)]
    if len(outside):
        report.warn(f"{len(outside)} composition segments extend past their stream")
    report.details["segments"] = {
        "count": int(len(segments)),
        "by_kind": segments["kind"].value_counts().to_dict(),
        "subject_groups": int(segments["subject_group"].nunique()),
    }


def check_group_leakage(segments: pd.DataFrame, stream_split: dict[str, str]) -> dict:
    """Raise ``LeakageError`` if one subject group feeds streams of two official splits."""
    split_of = segments["stream_id"].map(stream_split)
    if split_of.isna().any():
        unknown = sorted(segments.loc[split_of.isna(), "stream_id"].unique())
        raise LeakageError(f"segments belong to streams with no split assignment: {unknown[:5]}")
    by_group = segments.assign(split=split_of).groupby("subject_group")["split"].agg(lambda s: sorted(set(s)))
    shared = {group: splits for group, splits in by_group.items() if len(splits) > 1}
    if shared:
        raise LeakageError(f"subject groups appear in more than one split: {dict(list(shared.items())[:5])}")
    return {group: splits[0] for group, splits in by_group.items()}


def check_disjoint(named_sets: dict[str, set], what: str) -> None:
    """Raise ``LeakageError`` if any two named sets intersect."""
    names = list(named_sets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            common = named_sets[first] & named_sets[second]
            if common:
                raise LeakageError(f"{what} shared between {first} and {second}: {sorted(common)[:5]}")
