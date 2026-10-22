#!/usr/bin/env python3
"""Ghép train-only source groups thành các luồng dài có background liên tục.

Mỗi source-path hash chỉ thuộc một bundle. Label chỉ dùng để dựng tập train
tổng hợp, bảo vệ event khỏi crossfade và cân bằng nhóm; detector không đọc
composition hay quy tắc sắp xếp này. Không đọc validation hoặc segment arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import h5py
import numpy as np

from build_source_group_cv_package import _pure_clip_bounds, _read_train_metadata, _truthy

ROOT = Path(__file__).resolve().parents[1]
RATE_HZ = 50.0
FADE_FRAMES = 10
DEFAULT_SOURCE = ROOT / "data/processed/quantum_source_group_cv_package_20261020"
DEFAULT_OUTPUT = ROOT / "data/processed/manybody_context_cv_package_r32"
DEFAULT_ARTIFACTS = ROOT / "ABC testing/r32/build"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _ignored(event: dict) -> bool:
    return _truthy(event.get("ignore")) or event.get("quality", "").lower() == "low" or _truthy(event.get("needs_review"))


@dataclass(frozen=True)
class Clip:
    source_stream: str
    source_segment: str
    source_hash: str
    first: int
    stop: int
    source_t0: float
    source_t1: float
    kind: str
    label: str
    dataset: str
    event: dict | None

    @property
    def frames(self) -> int:
        return self.stop - self.first


def _metadata(source_root: Path) -> tuple[dict, dict, dict, dict, list]:
    manifest = json.loads((source_root / "source_group_manifest.json").read_text())
    source_hashes = {row["stream_id"]: row["source_group_hash"] for row in manifest}
    if len(source_hashes) != 467 or len(set(source_hashes.values())) != len(source_hashes):
        raise ValueError("expected exactly 467 unique input source groups")
    composition, events = _read_train_metadata(source_root)
    paths = {f"train_{int(p.stem.split('_')[-1]):02d}": p for p in (source_root / "streams/train").glob("stream_*.h5")}
    if set(paths) != set(source_hashes) or set(composition) != set(source_hashes):
        raise ValueError("input manifest, HDF5 and train composition scopes differ")
    groups = defaultdict(list)
    omitted_background = []
    for stream in sorted(paths):
        with h5py.File(paths[stream], "r") as handle:
            shape = handle["cir"].shape
            if shape[0] != 3 or shape[2] != 368 or str(handle.attrs["source_group_hash"]) != source_hashes[stream]:
                raise ValueError("input source hash or CIR schema mismatch")
        # A wholly blended short clip can have a later t_start than the next
        # appended clip. The repack's segment IDs preserve true append order.
        rows = sorted(composition[stream], key=lambda row: row["segment_id"])
        for index, row in enumerate(rows):
            if row["path"] != source_hashes[stream]:
                raise ValueError("input clip contains another source group's path")
            try:
                start, end = _pure_clip_bounds(rows, index)
            except ValueError:
                if row["kind"] == "fall":
                    raise
                omitted_background.append({"source_stream_id": stream, "source_segment_id": row["segment_id"],
                    "source_hash": source_hashes[stream], "original_duration_s": float(row["t_end"]) - float(row["t_start"]),
                    "reason": "no pure interior after removing existing crossfade mixtures"})
                continue
            first, stop = int(round(start * RATE_HZ)), int(round(end * RATE_HZ))
            if row["kind"] != "fall" and 0 <= first < stop <= shape[1] and stop - first < 5:
                omitted_background.append({"source_stream_id": stream, "source_segment_id": row["segment_id"],
                    "source_hash": source_hashes[stream], "original_duration_s": float(row["t_end"]) - float(row["t_start"]),
                    "reason": "pure nonfall interior shorter than five frames"})
                continue
            if not 0 <= first < stop <= shape[1] or stop - first < 5:
                raise ValueError("invalid pure source clip crop")
            event = None
            if row["kind"] == "fall":
                raw = events[(stream, row["segment_id"])]
                event_first = int(round(float(raw["t_start"]) * RATE_HZ))
                event_stop = int(round(float(raw["t_end"]) * RATE_HZ))
                if not first <= event_first < event_stop <= stop:
                    raise ValueError("pure crop would intersect an event; refusing data loss")
                event = {**raw, "local_first": event_first - first, "local_stop": event_stop - first}
            source_t0 = float(row["source_t0"]) + first / RATE_HZ - float(row["t_start"])
            groups[source_hashes[stream]].append(Clip(
                source_stream=stream, source_segment=row["segment_id"], source_hash=source_hashes[stream],
                first=first, stop=stop, source_t0=source_t0, source_t1=source_t0 + (stop - first) / RATE_HZ,
                kind=row["kind"], label=row.get("label", ""), dataset=row.get("dataset", ""), event=event,
            ))
    valid = sum(not _ignored(event) for event in events.values())
    if (len(events), valid, len(events) - valid) != (359, 186, 173):
        raise ValueError("input event counts differ from declared 359/186/173; no silent protocol change")
    if set(groups) != set(source_hashes.values()) or sum(len(c) for c in groups.values()) + len(omitted_background) != sum(map(len, composition.values())):
        raise ValueError("input source/clip coverage is incomplete")
    return dict(groups), paths, events, source_hashes, omitted_background


def _group_stats(clips: list[Clip]) -> dict:
    return {
        "valid": sum(c.event is not None and not _ignored(c.event) for c in clips),
        "ignored": sum(c.event is not None and _ignored(c.event) for c in clips),
        "background_frames": sum(c.frames for c in clips if c.event is None),
        "total_frames": sum(c.frames for c in clips),
    }


def _assign_groups(groups: dict[str, list[Clip]], bundles: int, rng: np.random.Generator) -> list[list[str]]:
    """Giữ group nguyên vẹn, cân bằng fall strata rồi background duration."""
    stats = {key: _group_stats(clips) for key, clips in groups.items()}
    assignments = [[] for _ in range(bundles)]
    totals = [{"valid": 0, "ignored": 0, "background_frames": 0, "total_frames": 0} for _ in range(bundles)]
    # Random tie breakers are fixed for a seed; decreasing group size handles
    # future multi-event groups without breaking source disjointness.
    tie = {key: float(rng.random()) for key in sorted(groups)}
    fall = sorted((key for key, s in stats.items() if s["valid"] + s["ignored"]),
                  key=lambda key: (-stats[key]["valid"], -stats[key]["ignored"], -stats[key]["total_frames"], tie[key]))
    background = sorted((key for key, s in stats.items() if not (s["valid"] + s["ignored"])),
                        key=lambda key: (-stats[key]["background_frames"], tie[key]))
    for key in fall + background:
        stat = stats[key]
        permutation = rng.permutation(bundles).tolist()
        if stat["valid"]:
            order = lambda i: (totals[i]["valid"], totals[i]["valid"] + totals[i]["ignored"], totals[i]["total_frames"])
        elif stat["ignored"]:
            order = lambda i: (totals[i]["ignored"], totals[i]["valid"] + totals[i]["ignored"], totals[i]["total_frames"])
        else:
            order = lambda i: (totals[i]["total_frames"], totals[i]["background_frames"])
        chosen = min(permutation, key=order)
        assignments[chosen].append(key)
        for name in totals[chosen]:
            totals[chosen][name] += stat[name]
    if sorted(key for bundle in assignments for key in bundle) != sorted(groups):
        raise ValueError("source assignment lost or duplicated a group")
    if any(not any(c.event is None for key in bundle for c in groups[key]) for bundle in assignments):
        raise ValueError("each continuous bundle needs nonfall context")
    return assignments


def _interleave(group_ids: list[str], groups: dict[str, list[Clip]], rng: np.random.Generator) -> list[Clip]:
    clips = [c for key in sorted(group_ids) for c in groups[key]]
    falls = [c for c in clips if c.event is not None]
    background = [c for c in clips if c.event is None]
    rng.shuffle(falls)
    rng.shuffle(background)
    if not falls:
        return background
    total_background = sum(c.frames for c in background)
    if not total_background:
        raise ValueError("cannot fix reset confound without background context")
    # One jittered target per time stratum, expressed in accumulated background
    # duration. This spreads events through the stream and always starts with
    # nonfall context. This is labeled TRAIN construction, never inference.
    targets = (np.arange(len(falls)) + rng.uniform(0.25, 0.75, len(falls))) / len(falls) * total_background
    ordered, cursor, consumed = [], 0, 0
    for fall, target in zip(falls, targets):
        while cursor < len(background) and consumed < target:
            clip = background[cursor]
            ordered.append(clip)
            consumed += clip.frames
            cursor += 1
        ordered.append(fall)
    ordered.extend(background[cursor:])
    if ordered[0].event is not None or len(ordered) != len(clips):
        raise ValueError("interleaving failed context or coverage invariant")
    return ordered


def _plan(ordered: list[Clip]) -> tuple[list[dict], int]:
    result, cursor, last_event_end, previous_tail = [], 0, -1, 0
    for clip in ordered:
        protect = clip.frames if clip.event is None else clip.event["local_first"]
        blend = min(FADE_FRAMES, clip.frames, previous_tail, protect, max(cursor - last_event_end, 0)) if result else 0
        origin = cursor - blend
        event_first = origin + clip.event["local_first"] if clip.event is not None else None
        event_stop = origin + clip.event["local_stop"] if clip.event is not None else None
        result.append({"clip": clip, "origin": origin, "blend": blend,
                       "event_first": event_first, "event_stop": event_stop})
        cursor = origin + clip.frames
        previous_tail = clip.frames - blend
        if event_stop is not None:
            last_event_end = max(last_event_end, event_stop)
    return result, cursor


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(source_root: Path, output_root: Path, artifact_root: Path, seed: int = 2028, bundles: int = 13) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {_relative(output_root)}")
    if bundles != 13:
        raise ValueError("this predeclared context package contains exactly 13 bundles")
    for path in (source_root, output_root, artifact_root):
        _relative(path)  # Metadata may not contain a machine-specific path.
    started = time.time()
    groups, paths, input_events, input_hashes, omitted_background = _metadata(source_root)
    rng = np.random.default_rng(seed)
    assignments = _assign_groups(groups, bundles, rng)
    plans = [_plan(_interleave(group_ids, groups, rng)) for group_ids in assignments]
    before_first30 = sum(not _ignored(e) and float(e["t_start"]) < 30 for e in input_events.values())
    artifact_root.mkdir(parents=True, exist_ok=True)
    _dump(artifact_root / "build_plan.json", {
        "source_root": _relative(source_root), "output_root": _relative(output_root), "seed": seed,
        "source_manifest_sha256": _digest(source_root / "source_group_manifest.json"),
        "builder_sha256": _digest(Path(__file__)), "rate_hz": RATE_HZ, "crossfade_frames": FADE_FRAMES,
        "grouping": "source-path-disjoint bundles; not proven subject-disjoint",
        "interleaving": "seeded shuffled clips; jittered uniform targets in accumulated nonfall duration; first nonfall",
        "bundle_source_hashes": [sorted(keys) for keys in assignments],
        "omitted_background_clips_without_usable_pure_interior": omitted_background,
        "validation_rows_read": False, "test_rows_read": False, "original_segment_arrays_read": False,
        "composition_use": "sourcepack TRAIN rows only during labeled dataset preparation; forbidden at inference",
    })
    train_dir = output_root / "streams/train"
    train_dir.mkdir(parents=True, exist_ok=False)
    out_events, out_composition, manifest, event_provenance = [], [], [], []
    checks = {"event_sample_slices_checked": 0, "event_sample_slices_bit_exact": True,
              "timestamp_grid_checked": True, "finite_cir_checked": True, "event_flags_preserved": True}
    for bundle_index, (group_ids, (plan, total_frames)) in enumerate(zip(assignments, plans)):
        stream = f"train_{bundle_index:02d}"
        filename = train_dir / f"stream_{bundle_index:02d}.h5"
        group_hash = hashlib.sha256("\n".join(sorted(group_ids)).encode()).hexdigest()
        bundle_events = []
        with h5py.File(filename, "w", rdcc_nbytes=32 * 1024 * 1024) as output:
            cir = output.create_dataset("cir", shape=(3, total_frames, 368), dtype="float32", chunks=(1, min(1024, total_frames), 368))
            output.create_dataset("t", data=np.arange(total_frames, dtype=np.float64) / RATE_HZ)
            output.attrs["split"] = "train"
            output.attrs["stream_index"] = bundle_index
            output.attrs["source_group_hash"] = group_hash
            output.attrs["source_hashes"] = json.dumps(sorted(group_ids))
            output.attrs["sample_rate_hz"] = RATE_HZ
            for order, item in enumerate(plan):
                clip, origin, blend = item["clip"], item["origin"], item["blend"]
                with h5py.File(paths[clip.source_stream], "r") as source:
                    block = np.asarray(source["cir"][:, clip.first:clip.stop, :], dtype=np.float32)
                if block.shape != (3, clip.frames, 368) or not np.isfinite(block).all():
                    raise ValueError("invalid source clip samples")
                if blend:
                    weights = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
                    tail = np.asarray(cir[:, origin:origin + blend, :])
                    cir[:, origin:origin + blend, :] = tail * (1.0 - weights) + block[:, :blend, :] * weights
                cir[:, origin + blend:origin + clip.frames, :] = block[:, blend:, :]
                segment = f"segment_{order:04d}"
                out_composition.append({
                    "stream_id": stream, "segment_id": segment, "kind": clip.kind, "path": clip.source_hash,
                    "label": clip.label, "dataset": clip.dataset, "t_start": origin / RATE_HZ,
                    "t_end": (origin + clip.frames) / RATE_HZ, "source_t0": clip.source_t0, "source_t1": clip.source_t1,
                    "source_stream_id": clip.source_stream, "source_segment_id": clip.source_segment,
                    "source_crop_first_frame": clip.first, "source_crop_stop_frame": clip.stop,
                    "new_crossfade_frames": blend,
                })
                if clip.event is not None:
                    raw = clip.event
                    row = {"stream_id": stream, "event_id": segment,
                           "t_start": item["event_first"] / RATE_HZ, "t_end": item["event_stop"] / RATE_HZ,
                           "fall_type": raw.get("fall_type", ""), "ignore": raw.get("ignore", "false"),
                           "quality": raw.get("quality", ""), "needs_review": raw.get("needs_review", "false"),
                           "dataset": raw.get("dataset", clip.dataset)}
                    out_events.append(row)
                    bundle_events.append(row)
                    event_provenance.append({
                        "output_stream_id": stream, "output_event_id": segment, "source_hash": clip.source_hash,
                        "input_stream_id": clip.source_stream, "input_event_id": clip.source_segment,
                        "input_first_frame": clip.first + raw["local_first"], "input_stop_frame": clip.first + raw["local_stop"],
                        "output_first_frame": item["event_first"], "output_stop_frame": item["event_stop"],
                        "ignored": _ignored(raw),
                    })
            # Verify after ALL later crossfades are written, so a subsequent
            # join cannot silently alter a previously checked event tail.
            for event in (e for e in event_provenance if e["output_stream_id"] == stream):
                with h5py.File(paths[event["input_stream_id"]], "r") as source:
                    original = np.asarray(source["cir"][:, event["input_first_frame"]:event["input_stop_frame"], :])
                rebuilt = np.asarray(cir[:, event["output_first_frame"]:event["output_stop_frame"], :])
                if not np.array_equal(original, rebuilt):
                    raise ValueError("event samples changed during context packing")
                checks["event_sample_slices_checked"] += 1
            t = np.asarray(output["t"])
            if t[0] != 0.0 or not np.allclose(np.diff(t), 1.0 / RATE_HZ, atol=1e-10, rtol=0):
                raise ValueError("output timestamp grid is not relative 50 Hz")
        manifest.append({
            "stream_id": stream, "source_group_hash": group_hash, "source_hashes": sorted(group_ids),
            "clip_count": len(plan), "duration_s": total_frames / RATE_HZ,
            "evaluable_fall_count": sum(not _ignored(e) for e in bundle_events),
            "ignored_fall_count": sum(_ignored(e) for e in bundle_events),
        })
        progress = {"bundles_written": bundle_index + 1, "total_bundles": bundles,
                    "events_preserved": len(out_events), "elapsed_s": round(time.time() - started, 2)}
        _dump(artifact_root / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    event_columns = ["stream_id", "event_id", "t_start", "t_end", "fall_type", "ignore", "quality", "needs_review", "dataset"]
    composition_columns = ["stream_id", "segment_id", "kind", "path", "label", "dataset", "t_start", "t_end", "source_t0", "source_t1",
                           "source_stream_id", "source_segment_id", "source_crop_first_frame", "source_crop_stop_frame", "new_crossfade_frames"]
    _write_csv(output_root / "streams_events.csv", out_events, event_columns)
    _write_csv(output_root / "streams_composition.csv", out_composition, composition_columns)
    _dump(output_root / "source_group_manifest.json", manifest)
    copied_sources = [key for row in manifest for key in row["source_hashes"]]
    if len(copied_sources) != 467 or len(set(copied_sources)) != 467 or set(copied_sources) != set(input_hashes.values()):
        raise ValueError("underlying source coverage or disjointness failed")
    if len(out_composition) != sum(len(c) for c in groups.values()) or len(out_events) != len(input_events):
        raise ValueError("clip/event coverage failed")
    provenance_keys = {(r["input_stream_id"], r["input_event_id"]) for r in event_provenance}
    if provenance_keys != set(input_events) or len(provenance_keys) != len(event_provenance):
        raise ValueError("event mapping is not a bijection")
    event_lookup = {(r["stream_id"], r["event_id"]): r for r in out_events}
    for mapping in event_provenance:
        old = input_events[(mapping["input_stream_id"], mapping["input_event_id"])]
        new = event_lookup[(mapping["output_stream_id"], mapping["output_event_id"])]
        for field in ("ignore", "quality", "needs_review", "fall_type", "dataset"):
            if old.get(field, "") != new.get(field, ""):
                raise ValueError("event metadata changed")
        if not np.isclose(float(old["t_end"]) - float(old["t_start"]), float(new["t_end"]) - float(new["t_start"]), atol=1e-10):
            raise ValueError("event duration changed")
    if sum(not _ignored(e) for e in out_events) != 186 or sum(_ignored(e) for e in out_events) != 173:
        raise ValueError("valid/flagged event counts changed")
    evaluator = output_root / "src/abc2027/evaluate.py"
    evaluator.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "src/abc2027/evaluate.py", evaluator)
    durations = {row["stream_id"]: row["duration_s"] for row in manifest}
    valid_events = [e for e in out_events if not _ignored(e)]
    if any(not (0 <= e["t_start"] < e["t_end"] <= durations[e["stream_id"]]) for e in out_events):
        raise ValueError("event coordinates lie outside the continuous bundle")
    early_count = sum(e["t_start"] < 30 for e in valid_events)
    relative_positions = [e["t_start"] / durations[e["stream_id"]] for e in valid_events]
    checks.update({
        "underlying_source_hashes_disjoint": True, "underlying_source_hashes_covered_exactly_once": 467,
        "event_mapping_bijection": True, "event_durations_preserved": True,
        "input_event_count": 359, "output_event_count": len(out_events), "valid_event_count": 186, "flagged_event_count": 173,
        "output_clip_count": len(out_composition), "evaluator_copy_sha256_matches": _digest(evaluator) == _digest(source_root / "src/abc2027/evaluate.py"),
        "valid_start_fraction_by_stream_quartile": np.histogram(relative_positions, bins=[0,.25,.5,.75,1])[0].tolist(),
    })
    summary = {
        "data_scope": "existing TRAIN-only source-repack HDF5 clips, pure interiors only",
        "source_root": _relative(source_root), "output_root": _relative(output_root), "seed": seed,
        "source_group_count": 467, "bundle_count": bundles, "train_event_count": 359,
        "evaluable_event_count": 186, "ignored_event_count": 173, "clip_count": len(out_composition),
        "input_clip_count": len(out_composition) + len(omitted_background),
        "omitted_nonfall_clip_count": len(omitted_background),
        "omitted_nonfall_total_original_duration_s": sum(row["original_duration_s"] for row in omitted_background),
        "train_duration_hours": sum(durations.values()) / 3600.0,
        "duration_s_by_bundle": durations,
        "valid_fall_count_by_bundle": {r["stream_id"]: r["evaluable_fall_count"] for r in manifest},
        "ignored_fall_count_by_bundle": {r["stream_id"]: r["ignored_fall_count"] for r in manifest},
        "input_valid_falls_first30s": before_first30, "input_valid_first30_fraction": before_first30 / 186.0,
        "output_valid_falls_first30s": early_count, "output_valid_first30_fraction": early_count / 186.0,
        "valid_event_start_quantiles_s": dict(zip(["min","q25","median","q75","max"], np.quantile([e["t_start"] for e in valid_events], [0,.25,.5,.75,1]).tolist())),
        "grouping_unit": "whole source-path groups assigned once to source-disjoint bundles; not subjects",
        "bundle_hash_definition": "SHA256 of newline-joined sorted underlying source-path hashes",
        "crossfade_policy": "trim previous within-group crossfade mixtures; reapply <=0.2 s outside all event sample intervals",
        "composition_used_for_dataset_preparation_only": True,
        "validation_rows_read": False, "test_rows_read": False, "original_segment_arrays_read": False,
        "label_dependent_inference": False,
        "limitations": [
            "Synthetic reordered context, not the original published continuous stream distribution",
            "Re-cropping previous repack crossfades removes some non-event context samples",
            "Underlying source-path grouping does not establish subject disjointness",
            "Labels guide synthetic training construction and group stratification; inference must read only CIR/t",
        ],
        "builder_sha256": _digest(Path(__file__)), "elapsed_s": round(time.time() - started, 2),
        "checks_passed": True, "event_sample_slices_bit_exact_checked": checks["event_sample_slices_checked"],
    }
    _dump(output_root / "build_summary.json", summary)
    _dump(artifact_root / "build_summary.json", summary)
    _dump(artifact_root / "numerical_provenance_checks.json", checks)
    _dump(artifact_root / "event_coordinate_map.json", event_provenance)
    _dump(artifact_root / "omitted_nonfall_clips.json", omitted_background)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--bundles", type=int, default=13)
    args = parser.parse_args()
    build(args.source_root.resolve(), args.output_dir.resolve(), args.artifact_dir.resolve(), args.seed, args.bundles)


if __name__ == "__main__":
    main()
