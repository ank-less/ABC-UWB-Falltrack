#!/usr/bin/env python3
"""Audit nested OOF errors using train annotations and train composition only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc_uwb_falltrack.oof import (  # noqa: E402
    assert_fold_partition,
    assert_train_heldout_partition,
    assert_train_stream_ids,
    read_train_events,
)


def _load_train_composition(path: Path, train_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stream_id", "kind", "label", "t_start", "t_end"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"composition file is missing columns: {sorted(missing)}")
        for row in reader:
            stream_id = (row.get("stream_id") or "").strip()
            if stream_id not in train_ids:
                continue
            by_stream[stream_id].append(
                {
                    "kind": row["kind"],
                    "label": row["label"],
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                }
            )
    return by_stream


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p25": None, "median": None, "p75": None}
    quantiles = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    return {
        "n": len(values),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "quantum_diff_histgb_nested_readout_sweep_20261020",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "uwb_falltrack" / "abc2027_uwb_train_v1",
    )
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    report_path = artifact_dir / "nested_oof_report.json"
    predictions_path = artifact_dir / "nested_oof_predictions_train_only.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train_id_list = report["train_streams"]
    assert_train_stream_ids(train_id_list, scope="nested OOF report train universe")
    train_ids = set(train_id_list)
    if report.get("validation_used") is not False:
        raise ValueError("nested OOF report does not confirm validation isolation")
    if report.get("composition_file_used") is not False:
        raise ValueError("nested OOF report does not confirm composition isolation")

    outer_records = report["outer_folds"]
    boundary_enabled = bool(report.get("boundary_readout", {}).get("enabled", False))
    boundary_scope_audit = not boundary_enabled
    if boundary_enabled:
        checks = report.get("zero_leakage_checks", {})
        required_checks = (
            "inner_boundary_heads_fit_on_inner_train_only",
            "inner_boundary_scores_limited_to_outer_train_oof",
            "outer_boundary_heads_fit_on_outer_train_only",
            "outer_boundary_scores_limited_to_outer_held_out",
        )
        if not all(checks.get(name) is True for name in required_checks):
            raise ValueError("report boundary-head leakage checks are incomplete or false")
        boundary_radii = {
            float(value)
            for value in report.get("boundary_refinement", {}).get("radii_s", [])
        }
        for outer_index, record in enumerate(outer_records):
            inner_records = record["inner_folds"]
            fit_scopes = record.get("inner_boundary_fit_scopes", [])
            if len(fit_scopes) != len(inner_records):
                raise ValueError("selected candidate is missing an inner boundary fit scope")
            for scope, inner in zip(fit_scopes, inner_records):
                if set(scope.get("fit_stream_ids", [])) != set(inner["train_streams"]):
                    raise ValueError("inner boundary fit does not match the inner-train scope")
                if set(scope.get("scored_stream_ids", [])) != set(inner["held_out_streams"]):
                    raise ValueError("inner boundary scores do not match the inner-held-out scope")
                if set(scope.get("fit_stream_ids", [])).intersection(
                    scope.get("scored_stream_ids", [])
                ):
                    raise ValueError("inner boundary fit and score scopes overlap")
            outer_train = set(record["train_streams"])
            held_out = set(record["held_out_streams"])
            if set(record.get("outer_boundary_fit_stream_ids", [])) != outer_train:
                raise ValueError("outer boundary fit does not match the outer-train scope")
            if set(record.get("outer_boundary_scored_stream_ids", [])) != held_out:
                raise ValueError("outer boundary scores do not match the outer-held-out scope")
            if outer_train.intersection(record.get("outer_boundary_scored_stream_ids", [])):
                raise ValueError("outer boundary model scored a training stream")
            chosen_radius = float(record["selected_decoder"].get("boundary_radius_s", -1.0))
            if chosen_radius not in boundary_radii:
                raise ValueError("selected boundary radius is outside the configured inner search")
        boundary_scope_audit = True
    duration_calibration_enabled = bool(report.get("duration_calibration", {}).get("enabled", False))
    duration_calibration_scope_audit = True
    if duration_calibration_enabled:
        configured_scales = {
            float(value) for value in report.get("duration_calibration", {}).get("scales", [])
        }
        if not configured_scales:
            raise ValueError("duration calibration report is missing its inner-search scale values")
        for record in outer_records:
            chosen_scale = float(record["selected_decoder"].get("duration_scale", -1.0))
            if chosen_scale not in configured_scales:
                raise ValueError("selected duration scale is outside the inner OOF search")
    outer_held_out = [record["held_out_streams"] for record in outer_records]
    assert_fold_partition(train_id_list, outer_held_out, scope="audited outer folds")
    for outer_index, record in enumerate(outer_records):
        outer_train = record["train_streams"]
        held_out = record["held_out_streams"]
        assert_train_heldout_partition(
            train_id_list, outer_train, held_out, scope=f"audited outer fold {outer_index}"
        )
        inner_records = record["inner_folds"]
        inner_held_out = [inner["held_out_streams"] for inner in inner_records]
        assert_fold_partition(
            outer_train, inner_held_out, scope=f"audited outer fold {outer_index} inner folds"
        )
        for inner_index, inner in enumerate(inner_records):
            assert_train_heldout_partition(
                outer_train,
                inner["train_streams"],
                inner["held_out_streams"],
                scope=f"audited outer fold {outer_index} inner fold {inner_index}",
            )
            if set(held_out).intersection(inner["train_streams"] + inner["held_out_streams"]):
                raise ValueError("outer-held streams entered an inner fit or held-out scope")

    logits_path = artifact_dir / report["nested_oof_logits_file"]
    with np.load(logits_path) as logits:
        if set(logits.files) != train_ids:
            raise ValueError("nested OOF logits do not cover exactly the train stream universe")
    events = read_train_events(data_root / "streams_events.csv", train_ids)
    events_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_stream[event.stream_id].append(
            {
                "stream_id": event.stream_id,
                "t_start": event.t_start,
                "t_end": event.t_end,
                "ignore": event.ignore,
            }
        )
    predictions_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] not in train_ids:
                raise ValueError(f"unexpected non-train prediction: {row['stream_id']}")
            predictions_by_stream[row["stream_id"]].append(
                {
                    "stream_id": row["stream_id"],
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "score": float(row["score"]),
                    "t_alarm": float(row["t_alarm"]) if row.get("t_alarm") else None,
                }
            )

    composition = _load_train_composition(data_root / "streams_composition.csv", train_ids)
    sys.path.insert(0, str(data_root / "src"))
    from abc2027.evaluate import _match_stream, temporal_iou  # noqa: E402

    totals: Counter[str] = Counter()
    fp_by_kind: Counter[str] = Counter()
    fp_by_label: Counter[str] = Counter()
    fp_phase: Counter[str] = Counter()
    start_errors: list[float] = []
    end_errors: list[float] = []
    all_gt_durations: list[float] = []
    all_pred_durations: list[float] = []
    boundary_values = {
        f"{theta:.1f}": {
            "tp": 0,
            "start_signed": [],
            "start_absolute": [],
            "end_signed": [],
            "end_absolute": [],
            "duration_signed": [],
            "duration_absolute": [],
            "duration_ratio": [],
        }
        for theta in (0.1, 0.3, 0.5)
    }
    unmatched_events = []
    fp_details = []
    per_stream = {}

    for stream_id in sorted(train_ids):
        ground = events_by_stream.get(stream_id, [])
        predictions = predictions_by_stream.get(stream_id, [])
        matched = _match_stream(ground, predictions, 0.3)
        matched_pred = {pair[0] for pair in matched["pairs"]}
        matched_gt = {pair[1] for pair in matched["pairs"]}
        totals["tp"] += len(matched["pairs"])
        totals["fp"] += matched["false_positive"]
        totals["fn"] += matched["false_negative"]
        scored_ground = matched["scored"]
        all_gt_durations.extend(event["t_end"] - event["t_start"] for event in scored_ground)
        all_pred_durations.extend(
            prediction["t_end"] - prediction["t_start"] for prediction in predictions
        )

        for theta in (0.1, 0.3, 0.5):
            theta_result = matched if theta == 0.3 else _match_stream(ground, predictions, theta)
            bucket = boundary_values[f"{theta:.1f}"]
            bucket["tp"] += len(theta_result["pairs"])
            for pred_index, gt_index, _iou in theta_result["pairs"]:
                prediction = predictions[pred_index]
                event = theta_result["scored"][gt_index]
                pred_duration = prediction["t_end"] - prediction["t_start"]
                gt_duration = event["t_end"] - event["t_start"]
                start_signed = prediction["t_start"] - event["t_start"]
                end_signed = prediction["t_end"] - event["t_end"]
                duration_signed = pred_duration - gt_duration
                bucket["start_signed"].append(start_signed)
                bucket["start_absolute"].append(abs(start_signed))
                bucket["end_signed"].append(end_signed)
                bucket["end_absolute"].append(abs(end_signed))
                bucket["duration_signed"].append(duration_signed)
                bucket["duration_absolute"].append(abs(duration_signed))
                bucket["duration_ratio"].append(pred_duration / max(gt_duration, 1e-12))
        for pred_index, gt_index, _iou in matched["pairs"]:
            prediction = predictions[pred_index]
            event = matched["scored"][gt_index]
            start_errors.append(abs(prediction["t_start"] - event["t_start"]))
            end_errors.append(abs(prediction["t_end"] - event["t_end"]))
        for gt_index, event in enumerate(matched["scored"]):
            if gt_index in matched_gt:
                continue
            best_iou = max(
                (
                    temporal_iou(
                        prediction["t_start"],
                        prediction["t_end"],
                        event["t_start"],
                        event["t_end"],
                    )
                    for prediction in predictions
                ),
                default=0.0,
            )
            unmatched_events.append(
                {
                    "stream_id": stream_id,
                    "t_start": event["t_start"],
                    "t_end": event["t_end"],
                    "best_prediction_tiou": best_iou,
                }
            )

        stream_fp = 0
        for pred_index, prediction in enumerate(predictions):
            if pred_index in matched_pred:
                continue
            in_ignore = any(
                _overlap(
                    prediction["t_start"],
                    prediction["t_end"],
                    event["t_start"] - 2.0,
                    event["t_end"] + 2.0,
                ) > 0.0
                for event in ground
                if event["ignore"]
            )
            if in_ignore:
                continue
            stream_fp += 1
            segments = composition.get(stream_id, [])
            dominant = max(
                segments,
                key=lambda segment: _overlap(
                    prediction["t_start"],
                    prediction["t_end"],
                    segment["t_start"],
                    segment["t_end"],
                ),
                default={"kind": "unknown", "label": "unknown"},
            )
            kind = str(dominant["kind"])
            label = str(dominant["label"])
            fp_by_kind[kind] += 1
            fp_by_label[label] += 1
            if kind == "fall":
                closest = min(
                    ground,
                    key=lambda event: min(
                        abs(prediction["t_end"] - event["t_start"]),
                        abs(prediction["t_start"] - event["t_end"]),
                    ),
                    default=None,
                )
                if closest is None:
                    phase = "no_annotation"
                elif _overlap(
                    prediction["t_start"],
                    prediction["t_end"],
                    closest["t_start"],
                    closest["t_end"],
                ) > 0.0:
                    phase = "overlaps_event_but_below_iou"
                elif prediction["t_end"] <= closest["t_start"]:
                    phase = "before_event"
                else:
                    phase = "after_event"
                fp_phase[phase] += 1
            fp_details.append(
                {
                    "stream_id": stream_id,
                    "t_start": prediction["t_start"],
                    "t_end": prediction["t_end"],
                    "score": prediction["score"],
                    "dominant_kind": kind,
                    "dominant_label": label,
                }
            )
        per_stream[stream_id] = {
            "tp": len(matched["pairs"]),
            "fp": stream_fp,
            "fn": matched["false_negative"],
            "n_predictions": len(predictions),
        }

    output = {
        "data_scope": "nested OOF train streams only",
        "validation_used": False,
        "composition_used": "train error audit only",
        "zero_leakage_checks": {
            "train_ids_match_train_namespace": True,
            "outer_folds_partition_train_universe": True,
            "inner_folds_partition_each_outer_train_scope": True,
            "outer_train_heldout_disjoint": True,
            "outer_heldout_absent_from_inner_folds": True,
            "outer_logits_cover_each_train_stream_once": True,
            "predictions_reference_train_streams_only": True,
            "validation_used": False,
            "composition_used_for_training_or_selection": False,
            "boundary_scopes_match_nested_folds": boundary_scope_audit,
            "duration_scale_in_inner_oof_search": duration_calibration_scope_audit,
        },
        "theta_0_3": {
            "tp": totals["tp"],
            "fp": totals["fp"],
            "fn": totals["fn"],
            "matched_start_error_median_abs": float(np.median(start_errors)) if start_errors else None,
            "matched_end_error_median_abs": float(np.median(end_errors)) if end_errors else None,
        },
        "event_duration_seconds": {
            "evaluable_ground_truth": _distribution(all_gt_durations),
            "all_predictions": _distribution(all_pred_durations),
        },
        "matched_boundary_geometry_by_theta": {
            theta: {
                "tp": values["tp"],
                "start_error_signed_pred_minus_gt_seconds": _distribution(values["start_signed"]),
                "start_error_absolute_seconds": _distribution(values["start_absolute"]),
                "end_error_signed_pred_minus_gt_seconds": _distribution(values["end_signed"]),
                "end_error_absolute_seconds": _distribution(values["end_absolute"]),
                "duration_error_signed_pred_minus_gt_seconds": _distribution(values["duration_signed"]),
                "duration_error_absolute_seconds": _distribution(values["duration_absolute"]),
                "predicted_to_ground_truth_duration_ratio": _distribution(values["duration_ratio"]),
            }
            for theta, values in boundary_values.items()
        },
        "false_alarm_by_kind": dict(fp_by_kind.most_common()),
        "false_alarm_by_label": dict(fp_by_label.most_common()),
        "fall_source_false_alarm_phase": dict(fp_phase.most_common()),
        "unmatched_evaluable_events": unmatched_events,
        "false_alarm_details": fp_details,
        "per_stream": per_stream,
    }
    target = artifact_dir / "nested_error_audit_train_only.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("theta_0_3", output["theta_0_3"])
    print("false_alarm_by_kind", dict(fp_by_kind.most_common()))
    print("false_alarm_by_label", dict(fp_by_label.most_common(12)))
    print("fall_source_false_alarm_phase", dict(fp_phase.most_common()))
    print("unmatched_evaluable_events", len(unmatched_events))
    print("event_duration_seconds", output["event_duration_seconds"])
    print("matched_boundary_geometry_by_theta", output["matched_boundary_geometry_by_theta"])
    print(f"artifact={target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
