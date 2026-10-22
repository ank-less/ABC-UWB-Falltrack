#!/usr/bin/env python3
"""Audit raw-source interval overlap across train-only nested OOF folds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_seconds(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _load_train_segments(
    path: Path, train_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stream_id", "kind", "path", "source_t0", "source_t1"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"composition file is missing columns: {sorted(missing)}")
        for row in reader:
            stream_id = (row.get("stream_id") or "").strip()
            # Do not inspect composition fields for validation or any other split.
            if stream_id not in train_ids:
                continue
            start = float(row["source_t0"])
            end = float(row["source_t1"])
            if start >= end:
                raise ValueError(f"invalid source interval in train stream {stream_id}")
            by_stream[stream_id].append(
                {
                    "path": row["path"],
                    "kind": row["kind"],
                    "start": start,
                    "end": end,
                }
            )
    return by_stream


def _group_segments(
    stream_ids: set[str], by_stream: dict[str, list[dict[str, Any]]]
) -> tuple[dict[tuple[str, str], list[tuple[float, float]]], set[tuple[str, str, float, float]]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    exact: set[tuple[str, str, float, float]] = set()
    for stream_id in stream_ids:
        for segment in by_stream.get(stream_id, []):
            key = (segment["path"], segment["kind"])
            start = float(segment["start"])
            end = float(segment["end"])
            grouped[key].append((start, end))
            exact.add((segment["path"], segment["kind"], round(start, 6), round(end, 6)))
    return {key: _merge(values) for key, values in grouped.items()}, exact


def _audit_scope(
    name: str,
    fit_ids: set[str],
    scored_ids: set[str],
    by_stream: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    fit, fit_exact = _group_segments(fit_ids, by_stream)
    scored, scored_exact = _group_segments(scored_ids, by_stream)
    shared = set(fit).intersection(scored)
    heldout_seconds: dict[str, float] = defaultdict(float)
    overlap_seconds: dict[str, float] = defaultdict(float)
    shared_sources: dict[str, int] = defaultdict(int)
    for (path, kind), intervals in scored.items():
        heldout_seconds[kind] += _duration(intervals)
        if (path, kind) in fit:
            shared_sources[kind] += 1
            overlap_seconds[kind] += _intersection_seconds(fit[(path, kind)], intervals)
    kinds = sorted(set(heldout_seconds) | set(overlap_seconds))
    by_kind = {
        kind: {
            "shared_source_files": int(shared_sources[kind]),
            "heldout_source_seconds": round(heldout_seconds[kind], 3),
            "overlapping_source_seconds": round(overlap_seconds[kind], 3),
            "heldout_overlap_fraction": (
                round(overlap_seconds[kind] / heldout_seconds[kind], 6)
                if heldout_seconds[kind]
                else None
            ),
        }
        for kind in kinds
    }
    exact_duplicates = len(fit_exact.intersection(scored_exact))
    total_overlap = sum(overlap_seconds.values())
    return {
        "scope": name,
        "fit_stream_count": len(fit_ids),
        "scored_stream_count": len(scored_ids),
        "shared_source_file_count": len(shared),
        "exact_duplicate_source_intervals": exact_duplicates,
        "overlap_by_kind": by_kind,
        "has_source_interval_overlap": total_overlap > 1e-6,
    }


def _overlap_components(
    train_ids: set[str], by_stream: dict[str, list[dict[str, Any]]]
) -> list[list[str]]:
    grouped: dict[str, dict[tuple[str, str], list[tuple[float, float]]]] = {}
    for stream_id in train_ids:
        per_source: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for row in by_stream.get(stream_id, []):
            per_source[(row["path"], row["kind"])].append((row["start"], row["end"]))
        grouped[stream_id] = {key: _merge(values) for key, values in per_source.items()}
    adjacency = {stream_id: set() for stream_id in train_ids}
    ordered = sorted(train_ids)
    for i, left_id in enumerate(ordered):
        left = grouped[left_id]
        for right_id in ordered[i + 1 :]:
            right = grouped[right_id]
            for key in set(left).intersection(right):
                if _intersection_seconds(left[key], right[key]) > 1e-6:
                    adjacency[left_id].add(right_id)
                    adjacency[right_id].add(left_id)
                    break
    unseen = set(train_ids)
    components: list[list[str]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current].intersection(unseen):
                unseen.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "uwb_falltrack" / "abc2027_uwb_train_v1",
    )
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    report = json.loads((artifact_dir / "nested_oof_report.json").read_text(encoding="utf-8"))
    train_ids = set(report["train_streams"])
    by_stream = _load_train_segments(data_root / "streams_composition.csv", train_ids)

    scope_audits = []
    for outer in report["outer_folds"]:
        outer_train = set(outer["train_streams"])
        outer_heldout = set(outer["held_out_streams"])
        scope_audits.append(
            _audit_scope(f"outer_{outer['fold']}", outer_train, outer_heldout, by_stream)
        )
        for inner in outer["inner_folds"]:
            scope_audits.append(
                _audit_scope(
                    f"outer_{outer['fold']}_inner_{inner['fold']}",
                    set(inner["train_streams"]),
                    set(inner["held_out_streams"]),
                    by_stream,
                )
            )

    components = _overlap_components(train_ids, by_stream)
    output = {
        "data_scope": "train composition rows only",
        "validation_rows_used": False,
        "source_time_semantics": "source_t0/source_t1 are passed directly to the package CIR slice function",
        "train_stream_overlap_components": components,
        "scope_audits": scope_audits,
        "stream_level_oof_source_disjoint": not any(
            result["has_source_interval_overlap"] for result in scope_audits
        ),
    }
    target = artifact_dir / "source_overlap_audit_train_only.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for result in scope_audits:
        if not result["scope"].startswith("outer_") or "_inner_" in result["scope"]:
            continue
        print(
            result["scope"],
            {
                kind: values["heldout_overlap_fraction"]
                for kind, values in result["overlap_by_kind"].items()
            },
        )
    print("stream_overlap_component_sizes", [len(component) for component in components])
    print("stream_level_oof_source_disjoint", output["stream_level_oof_source_disjoint"])
    print(f"artifact={target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
