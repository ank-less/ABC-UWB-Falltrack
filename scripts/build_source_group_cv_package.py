#!/usr/bin/env python3
"""Repack only published train-stream clips so each source file is one CV group.

This reads train rows from the local composition/annotation files, then crops
the corresponding samples from the already published train HDF5 streams. It
never reads validation rows or the package's original segment arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "uwb_falltrack" / "abc2027_uwb_train_v1"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "quantum_source_group_cv_package_20261020"
RATE_HZ = 50.0
STEP_S = 1.0 / RATE_HZ
FADE_N = 10


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _stream_id(path: Path) -> str:
    return f"train_{int(path.stem.split('_')[-1]):02d}"


def _read_train_metadata(data_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    stream_paths = sorted((data_root / "streams" / "train").glob("stream_*.h5"))
    train_ids = {_stream_id(path) for path in stream_paths}
    if not train_ids:
        raise FileNotFoundError(f"no train streams under {data_root / 'streams' / 'train'}")

    composition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (data_root / "streams_composition.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "stream_id", "segment_id", "kind", "path", "t_start", "t_end",
            "source_t0", "source_t1",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"composition file is missing columns: {sorted(missing)}")
        for row in reader:
            stream_id = (row.get("stream_id") or "").strip()
            if stream_id not in train_ids:
                continue
            composition[stream_id].append(row)

    events: dict[tuple[str, str], dict[str, Any]] = {}
    with (data_root / "streams_events.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stream_id", "event_id", "t_start", "t_end", "ignore", "quality", "needs_review"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"event file is missing columns: {sorted(missing)}")
        for row in reader:
            stream_id = (row.get("stream_id") or "").strip()
            if stream_id not in train_ids:
                continue
            events[(stream_id, row["event_id"])] = row

    for stream_id, rows in composition.items():
        rows.sort(key=lambda row: (float(row["t_start"]), row["segment_id"]))
        for index, row in enumerate(rows):
            if row["kind"] != "fall":
                continue
            if (stream_id, row["segment_id"]) not in events:
                raise ValueError(f"train fall segment is missing its event row: {stream_id}")
    fall_segments = [row for rows in composition.values() for row in rows if row["kind"] == "fall"]
    if len(fall_segments) != len(events):
        raise ValueError("train fall segment/event counts do not match")
    return dict(composition), events


def _pure_clip_bounds(rows: list[dict[str, Any]], index: int) -> tuple[float, float]:
    row = rows[index]
    start = float(row["t_start"])
    end = float(row["t_end"])
    if index and start < float(rows[index - 1]["t_end"]):
        # The published stream stores the 0.2 s crossfade in both rows.
        start = max(start, float(rows[index - 1]["t_end"]))
    if index + 1 < len(rows) and float(rows[index + 1]["t_start"]) < end:
        end = min(end, float(rows[index + 1]["t_start"]))
    if start >= end:
        raise ValueError(f"crossfade trimming removed train segment {row['segment_id']}")
    return start, end


def _collect_clips(
    data_root: Path,
    composition: dict[str, list[dict[str, Any]]],
    events: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    stream_paths = {_stream_id(path): path for path in (data_root / "streams" / "train").glob("stream_*.h5")}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stream_id in sorted(composition):
        rows = composition[stream_id]
        with h5py.File(stream_paths[stream_id], "r") as handle:
            cir = handle["cir"]
            for index, row in enumerate(rows):
                start_s, end_s = _pure_clip_bounds(rows, index)
                first = int(round(start_s * RATE_HZ))
                stop = int(round(end_s * RATE_HZ))
                if first < 0 or stop > cir.shape[1] or first >= stop:
                    raise ValueError(f"invalid train crop for {stream_id}/{row['segment_id']}")
                block = np.asarray(cir[:, first:stop, :], dtype=np.float32)
                if block.shape[1] < 5:
                    raise ValueError(f"train crop is too short: {stream_id}/{row['segment_id']}")

                source_start = float(row["source_t0"]) + start_s - float(row["t_start"])
                source_end = source_start + block.shape[1] * STEP_S
                clip: dict[str, Any] = {
                    "block": block,
                    "stream_id": stream_id,
                    "segment_id": row["segment_id"],
                    "source_path": row["path"],
                    "kind": row["kind"],
                    "label": row.get("label", ""),
                    "dataset": row.get("dataset", ""),
                    "source_t0": source_start,
                    "source_t1": source_end,
                    "event": None,
                }
                if row["kind"] == "fall":
                    event = events[(stream_id, row["segment_id"])]
                    event_start = float(event["t_start"])
                    event_end = float(event["t_end"])
                    if not (start_s <= event_start < event_end <= end_s):
                        raise ValueError(f"crossfade trim intersects fall event in {stream_id}")
                    clip["event"] = {
                        "local_start": event_start - start_s,
                        "local_end": event_end - start_s,
                        "ignore": _truthy(event.get("ignore")),
                        "quality": event.get("quality", ""),
                        "needs_review": _truthy(event.get("needs_review")),
                        "fall_type": event.get("fall_type", ""),
                    }
                groups[row["path"]].append(clip)
    for path_clips in groups.values():
        path_clips.sort(key=lambda clip: (clip["stream_id"], clip["segment_id"]))
    return dict(groups)


def _pack_group(
    clips: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[np.ndarray] = []
    composition_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    cursor = 0
    last_event_end = -1
    for order, clip in enumerate(clips):
        block = clip["block"]
        event = clip["event"]
        blend = 0
        if chunks:
            protect = block.shape[1]
            if event is not None:
                protect = max(0, int(round(event["local_start"] * RATE_HZ)))
            tail_room = cursor - last_event_end
            blend = min(FADE_N, block.shape[1], chunks[-1].shape[1], protect, max(tail_room, 0))
            if blend > 0:
                weight = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
                chunks[-1][:, -blend:] = chunks[-1][:, -blend:] * (1.0 - weight) + block[:, :blend] * weight
        origin = cursor - blend
        if block.shape[1] > blend:
            chunks.append(block[:, blend:])
        cursor = origin + block.shape[1]

        segment_id = f"segment_{order:04d}"
        composition_rows.append(
            {
                "segment_id": segment_id,
                "kind": clip["kind"],
                "path": clip["group_hash"],
                "label": clip["label"],
                "dataset": clip["dataset"],
                "t_start": origin * STEP_S,
                "t_end": cursor * STEP_S,
                "source_t0": clip["source_t0"],
                "source_t1": clip["source_t1"],
            }
        )
        if event is not None:
            event_a = origin + int(round(event["local_start"] * RATE_HZ))
            event_b = origin + int(round(event["local_end"] * RATE_HZ))
            last_event_end = max(last_event_end, event_b)
            events.append(
                {
                    "event_id": segment_id,
                    "t_start": event_a * STEP_S,
                    "t_end": event_b * STEP_S,
                    "fall_type": event["fall_type"],
                    "ignore": str(event["ignore"]).lower(),
                    "quality": event["quality"],
                    "needs_review": str(event["needs_review"]).lower(),
                    "dataset": clip["dataset"],
                }
            )
    if not chunks:
        raise ValueError("cannot write an empty source group")
    return np.concatenate(chunks, axis=1), composition_rows, events


def build(data_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    composition, events = _read_train_metadata(data_root)
    clips_by_path = _collect_clips(data_root, composition, events)
    output_train = output_root / "streams" / "train"
    output_train.mkdir(parents=True, exist_ok=False)
    out_events: list[dict[str, Any]] = []
    out_composition: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for index, source_path in enumerate(sorted(clips_by_path, key=lambda value: hashlib.sha256(value.encode()).hexdigest())):
        stream_id = f"train_{index:02d}"
        stream_file = output_train / f"stream_{index:03d}.h5"
        group_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:20]
        clips = clips_by_path[source_path]
        for clip in clips:
            clip["group_hash"] = group_hash
        cir, group_composition, group_events = _pack_group(clips)
        with h5py.File(stream_file, "w") as handle:
            handle.create_dataset("cir", data=cir, chunks=(3, min(cir.shape[1], 5000), 368))
            handle.create_dataset("t", data=np.arange(cir.shape[1], dtype=np.float64) * STEP_S)
            handle.attrs["split"] = "train"
            handle.attrs["stream_index"] = index
            handle.attrs["source_group_hash"] = group_hash
        for row in group_composition:
            row["stream_id"] = stream_id
            out_composition.append(row)
        for row in group_events:
            row["stream_id"] = stream_id
            out_events.append(row)
        manifest.append(
            {
                "stream_id": stream_id,
                "source_group_hash": group_hash,
                "clip_count": len(clips),
                "duration_s": round(cir.shape[1] * STEP_S, 3),
                "evaluable_fall_count": sum(
                    not _truthy(event["ignore"])
                    and event["quality"].lower() != "low"
                    and not _truthy(event["needs_review"])
                    for event in group_events
                ),
                "ignored_fall_count": sum(
                    _truthy(event["ignore"])
                    or event["quality"].lower() == "low"
                    or _truthy(event["needs_review"])
                    for event in group_events
                ),
            }
        )

    with (output_root / "streams_events.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["stream_id", "event_id", "t_start", "t_end", "fall_type", "ignore", "quality", "needs_review", "dataset"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in out_events:
            writer.writerow(row)
    with (output_root / "streams_composition.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["stream_id", "segment_id", "kind", "path", "label", "dataset", "t_start", "t_end", "source_t0", "source_t1"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in out_composition:
            writer.writerow(row)
    (output_root / "source_group_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Use the same local evaluator implementation as the published package.
    evaluator_source = data_root / "src" / "abc2027" / "evaluate.py"
    evaluator_dest = output_root / "src" / "abc2027" / "evaluate.py"
    evaluator_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(evaluator_source, evaluator_dest)

    summary = {
        "data_scope": "published train HDF5 samples only",
        "validation_rows_read": False,
        "original_segment_arrays_read": False,
        "source_group_count": len(manifest),
        "train_event_count": len(out_events),
        "train_duration_hours": round(sum(row["duration_s"] for row in manifest) / 3600.0, 4),
        "source_group_manifest": "source_group_manifest.json",
        "grouping_unit": "source file path; each hashed source appears in exactly one pseudo-stream",
        "crossfade_policy": "trim original crossfade mixtures, then reapply 0.2 s fades within the same source group",
    }
    (output_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.data_root.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
