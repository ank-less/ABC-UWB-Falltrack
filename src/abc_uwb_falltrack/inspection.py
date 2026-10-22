"""Data inspection report, computed chunk by chunk (never a whole stream in memory)."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import data_path, get
from .data_io import DataError, discover_streams, iter_cir_chunks, load_composition, load_events, read_stream_info
from .manifest import segment_table

LOG = logging.getLogger(__name__)


def hdf5_tree(path: Path) -> list[dict]:
    items: list[dict] = []
    with h5py.File(path, "r") as handle:

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append(
                    {
                        "name": name,
                        "kind": "dataset",
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "chunks": list(obj.chunks) if obj.chunks else None,
                        "compression": obj.compression,
                    }
                )
            else:
                items.append({"name": name, "kind": "group"})

        handle.visititems(visit)
    return items


def scan_values(path: Path, chunk_frames: int) -> dict:
    """Non-finite counts and per-radar value statistics over the whole CIR."""
    count = 0
    nonfinite = 0
    zero_frames = 0
    total = total_sq = None
    low = high = None
    for _start, block in iter_cir_chunks(path, chunk_frames):
        finite = np.isfinite(block)
        nonfinite += int((~finite).sum())
        clean = np.where(finite, block, 0.0).astype(np.float64)
        zero_frames += int(np.sum(np.all(clean == 0.0, axis=2)))
        part_sum = clean.sum(axis=(1, 2))
        part_sq = (clean * clean).sum(axis=(1, 2))
        part_min = np.where(finite, block, np.inf).min(axis=(1, 2))
        part_max = np.where(finite, block, -np.inf).max(axis=(1, 2))
        total = part_sum if total is None else total + part_sum
        total_sq = part_sq if total_sq is None else total_sq + part_sq
        low = part_min if low is None else np.minimum(low, part_min)
        high = part_max if high is None else np.maximum(high, part_max)
        count += block.shape[1] * block.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0))
    return {
        "nonfinite_values": nonfinite,
        "all_zero_radar_frames": zero_frames,
        "per_radar": [
            {"radar": r, "mean": float(mean[r]), "std": float(std[r]), "min": float(low[r]), "max": float(high[r])}
            for r in range(len(mean))
        ],
    }


def inspect_dataset(cfg: dict, scan: bool = True) -> dict:
    root = Path(get(cfg, "data.root"))
    if not root.is_dir():
        raise DataError(f"data.root not found: {root}. See data/README.md for the download.")
    rate = float(get(cfg, "data.rate_hz"))
    chunk = int(get(cfg, "data.chunk_frames"))
    issues: list[str] = []

    files = [p for p in root.rglob("*") if p.is_file()]
    by_folder: dict[str, dict] = {}
    for path in files:
        top = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "."
        entry = by_folder.setdefault(top, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += path.stat().st_size
    listed = [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size}
        for p in sorted(files)
        if p.relative_to(root).parts[0] != "segments"
    ]
    documents = {name: (root / name).is_file() for name in ("README.md", "LICENSE", "sample_submission.csv")}
    documents["official_evaluator"] = bool(get(cfg, "evaluation.official_evaluator")) and Path(
        get(cfg, "evaluation.official_evaluator")
    ).is_file()
    if (root / "segments").is_dir():
        issues.append(
            "segments/ (aligned source recordings) is present; the package README forbids training or inference "
            "on the source LIARA recordings, so the pipeline does not read it"
        )

    streams = []
    for path in discover_streams(data_path(cfg, "streams_dir"), list(get(cfg, "data.include_splits"))):
        info = read_stream_info(path)
        entry = {
            "stream_id": info.stream_id,
            "file": path.relative_to(root).as_posix(),
            "bytes": info.file_size,
            "tree": hdf5_tree(path),
            "attrs": info.attrs,
            "radars": info.n_radars,
            "frames": info.n_frames,
            "bins": info.n_bins,
            "dtype": info.dtype,
            "radar_serials": list(info.radar_serials),
            "t_first": info.t_first,
            "t_last": info.t_last,
            "dt_min": info.dt_min,
            "dt_max": info.dt_max,
            "monotonic": info.monotonic,
            "duration_s": info.n_frames / rate,
        }
        if not info.monotonic:
            issues.append(f"{info.stream_id}: timestamps not strictly increasing")
        if abs(info.dt_min - 1 / rate) > 1e-6 or abs(info.dt_max - 1 / rate) > 1e-6:
            issues.append(f"{info.stream_id}: step outside 1/{rate} s")
        if scan:
            entry["values"] = scan_values(path, chunk)
            if entry["values"]["nonfinite_values"]:
                issues.append(f"{info.stream_id}: {entry['values']['nonfinite_values']} NaN/Inf values")
        streams.append(entry)
        LOG.info("inspected %s", info.stream_id)

    events = load_events(data_path(cfg, "events_csv"))
    events["split"] = events["stream_id"].str.rsplit("_", n=1).str[0]
    events["duration"] = events["t_end"] - events["t_start"]
    hours = pd.Series({s["stream_id"]: s["duration_s"] for s in streams}).groupby(lambda s: s.rsplit("_", 1)[0]).sum() / 3600
    event_summary = {
        "count": int(len(events)),
        "by_split": events["split"].value_counts().to_dict(),
        "per_hour_by_split": {k: float(v / hours[k]) for k, v in events["split"].value_counts().items() if k in hours},
        "flagged_by_split": events.groupby("split")["flagged"].sum().astype(int).to_dict(),
        "ignore_true": int(events["ignore"].sum()),
        "duration_s": events["duration"].describe().to_dict(),
        "fall_type": events["fall_type"].value_counts().to_dict() if "fall_type" in events else None,
        "dataset": events["dataset"].value_counts().to_dict() if "dataset" in events else None,
        "fall_seconds_fraction": float(events["duration"].sum() / (hours.sum() * 3600)),
    }

    segments = segment_table(load_composition(data_path(cfg, "composition_csv")))
    segments["seconds"] = segments["t_end"] - segments["t_start"]
    groups_by_split = segments.groupby("official_split")["subject_group"].agg(lambda s: sorted(set(s))).to_dict()
    shared = set.intersection(*(set(v) for v in groups_by_split.values())) if len(groups_by_split) > 1 else set()
    if shared:
        issues.append(f"subject groups shared between splits: {sorted(shared)}")
    reuse = segments[segments["kind"] == "adl"].groupby("path")["stream_id"].nunique()
    composition_summary = {
        "segments_by_kind": segments["kind"].value_counts().to_dict(),
        "seconds_by_kind_and_split": segments.groupby(["official_split", "kind"])["seconds"].sum().round(1).unstack().to_dict(),
        "activities": segments.loc[segments["kind"] == "adl", "label"].value_counts().to_dict() if "label" in segments else None,
        "subject_groups_by_split": groups_by_split,
        "groups_shared_between_splits": sorted(shared),
        "adl_source_file_reused_in_streams": {"min": int(reuse.min()), "median": float(reuse.median()), "max": int(reuse.max())}
        if len(reuse)
        else None,
    }
    return {
        "root": root.as_posix(),
        "folders": by_folder,
        "files": listed,
        "documents": documents,
        "streams": streams,
        "events": event_summary,
        "composition": composition_summary,
        "issues": issues,
    }


def summary_markdown(report: dict) -> str:
    lines = ["# Data inspection", "", f"Root: `{report['root']}`", "", "| stream | frames | duration (s) | dtype | t monotonic | NaN/Inf |", "|---|---|---|---|---|---|"]
    for s in report["streams"]:
        nonfinite = s.get("values", {}).get("nonfinite_values", "not scanned")
        lines.append(f"| {s['stream_id']} | {s['frames']} | {s['duration_s']:.1f} | {s['dtype']} | {s['monotonic']} | {nonfinite} |")
    lines += ["", f"Events: {report['events']['by_split']}", "", "Issues:"]
    lines += [f"- {issue}" for issue in report["issues"]] or ["- none"]
    return "\n".join(lines) + "\n"
