#!/usr/bin/env python3
"""Audit OOF train errors and map false alarms to train composition segments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc_uwb_falltrack.oof import (  # noqa: E402
    _events_from_state,
    _online_ising,
    read_train_events,
)


def _load_composition_train(path: Path) -> dict[str, list[dict[str, object]]]:
    by_stream: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stream_id", "kind", "label", "t_start", "t_end"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"composition file is missing columns: {sorted(missing)}")
        for row in reader:
            stream_id = (row.get("stream_id") or "").strip()
            if not stream_id.startswith("train_"):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "quantum_diff_oof_20261020_r1",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "uwb_falltrack" / "abc2027_uwb_train_v1",
    )
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    report = json.loads((artifact_dir / "oof_report.json").read_text(encoding="utf-8"))
    config = json.loads((artifact_dir / "config_used.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "oof_logits_train_only.npz", allow_pickle=False) as archive:
        logits_by_stream = {key: archive[key] for key in archive.files}
    train_ids = sorted(logits_by_stream)
    events = read_train_events(data_root / "streams_events.csv", set(train_ids))
    events_by_stream: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        events_by_stream[event.stream_id].append(
            {
                "stream_id": event.stream_id,
                "t_start": event.t_start,
                "t_end": event.t_end,
                "ignore": event.ignore,
            }
        )
    composition = _load_composition_train(data_root / "streams_composition.csv")

    best = report["best_oof"]
    step_s = 5.0 / 50.0
    predictions_by_stream: dict[str, list[dict[str, object]]] = defaultdict(list)
    for stream_id in train_ids:
        cache_root = (
            (ROOT / config["feature_cache_dir"]).resolve()
            if config.get("feature_cache_dir")
            else artifact_dir / "feature_cache"
        )
        cache_path = cache_root / f"{stream_id}.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            duration_s = float(cache["duration_s"].item())
        state = _online_ising(
            logits_by_stream[stream_id],
            best["bias"],
            best["transition_cost"],
            entry_cost=float(best.get("entry_cost", best["transition_cost"])),
            exit_cost=float(best.get("exit_cost", best["transition_cost"])),
        )
        predictions_by_stream[stream_id] = _events_from_state(
            stream_id,
            state,
            logits_by_stream[stream_id],
            duration_s,
            step_s,
            best["min_duration_s"],
            start_offset_s=float(best.get("start_offset_s", 0.0)),
            end_offset_s=float(best.get("end_offset_s", 0.0)),
            duration_profile=best.get("duration_profile"),
        )

    sys.path.insert(0, str(data_root / "src"))
    from abc2027.evaluate import _match_stream, temporal_iou  # noqa: E402

    results: dict[str, object] = {
        "data_scope": "OOF train streams only",
        "composition_used": "train error audit only",
        "validation_used": False,
        "theta_0_3": {},
        "unmatched_evaluable_gt": [],
    }
    fp_by_kind: Counter[str] = Counter()
    fp_by_label: Counter[str] = Counter()
    fall_margin_phase: Counter[str] = Counter()
    matched_start_errors: list[float] = []
    matched_end_errors: list[float] = []
    matched_ious: list[float] = []
    fp_details: list[dict[str, object]] = []
    totals = Counter()
    for stream_id in train_ids:
        ground = events_by_stream.get(stream_id, [])
        predictions = predictions_by_stream.get(stream_id, [])
        matched_result = _match_stream(ground, predictions, 0.3)
        pairs = matched_result["pairs"]
        matched_pred = {pair[0] for pair in pairs}
        matched_gt = {pair[1] for pair in pairs}
        totals["tp"] += len(pairs)
        totals["fn"] += matched_result["false_negative"]
        totals["fp"] += matched_result["false_positive"]
        for pred_index, gt_index, iou in pairs:
            prediction = predictions[pred_index]
            event = matched_result["scored"][gt_index]
            matched_start_errors.append(abs(float(prediction["t_start"]) - float(event["t_start"])))
            matched_end_errors.append(abs(float(prediction["t_end"]) - float(event["t_end"])))
            matched_ious.append(float(iou))
        for gt_index, event in enumerate(matched_result["scored"]):
            if gt_index in matched_gt:
                continue
            best_iou = max(
                (
                    temporal_iou(
                        float(prediction["t_start"]),
                        float(prediction["t_end"]),
                        float(event["t_start"]),
                        float(event["t_end"]),
                    )
                    for prediction in predictions
                ),
                default=0.0,
            )
            results["unmatched_evaluable_gt"].append(
                {
                    "stream_id": stream_id,
                    "t_start": event["t_start"],
                    "t_end": event["t_end"],
                    "best_prediction_tiou": best_iou,
                }
            )
        for pred_index, prediction in enumerate(predictions):
            if pred_index in matched_pred:
                continue
            if any(
                _overlap(
                    float(prediction["t_start"]),
                    float(prediction["t_end"]),
                    float(event["t_start"]) - 2.0,
                    float(event["t_end"]) + 2.0,
                ) > 0.0
                for event in ground
                if event["ignore"]
            ):
                continue
            segments = composition.get(stream_id, [])
            dominant = max(
                segments,
                key=lambda segment: _overlap(
                    float(prediction["t_start"]),
                    float(prediction["t_end"]),
                    float(segment["t_start"]),
                    float(segment["t_end"]),
                ),
                default={"kind": "unknown", "label": "unknown"},
            )
            kind = str(dominant["kind"])
            label = str(dominant["label"])
            fp_by_kind[kind] += 1
            fp_by_label[label] += 1
            if kind == "fall":
                related = events_by_stream.get(stream_id, [])
                closest = min(
                    related,
                    key=lambda event: min(
                        abs(float(prediction["t_end"]) - float(event["t_start"])),
                        abs(float(prediction["t_start"]) - float(event["t_end"])),
                    ),
                    default=None,
                )
                if closest is None:
                    phase = "no_annotation"
                elif _overlap(
                    float(prediction["t_start"]),
                    float(prediction["t_end"]),
                    float(closest["t_start"]),
                    float(closest["t_end"]),
                ) > 0.0:
                    phase = "overlaps_event_but_below_iou"
                elif float(prediction["t_end"]) <= float(closest["t_start"]):
                    phase = "before_event"
                else:
                    phase = "after_event"
                fall_margin_phase[phase] += 1
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

    results["theta_0_3"] = {
        "tp": int(totals["tp"]),
        "fp": int(totals["fp"]),
        "fn": int(totals["fn"]),
        "matched_start_error_median_abs": float(np.median(matched_start_errors)) if matched_start_errors else None,
        "matched_end_error_median_abs": float(np.median(matched_end_errors)) if matched_end_errors else None,
        "matched_tiou_mean": float(np.mean(matched_ious)) if matched_ious else None,
    }
    results["false_alarm_by_kind"] = dict(fp_by_kind.most_common())
    results["false_alarm_by_label"] = dict(fp_by_label.most_common())
    results["fall_source_false_alarm_phase"] = dict(fall_margin_phase.most_common())
    results["false_alarm_details"] = fp_details
    results["n_predictions_by_stream"] = {
        stream_id: len(predictions_by_stream[stream_id]) for stream_id in train_ids
    }
    output = artifact_dir / "error_audit_train.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("false_alarm_by_kind", dict(fp_by_kind.most_common()))
    print("false_alarm_by_label", dict(fp_by_label.most_common(12)))
    print("fall_source_false_alarm_phase", dict(fall_margin_phase.most_common()))
    print("unmatched_evaluable_events", len(results["unmatched_evaluable_gt"]))
    print("matched_start_error_median_abs", results["theta_0_3"]["matched_start_error_median_abs"])
    print("matched_end_error_median_abs", results["theta_0_3"]["matched_end_error_median_abs"])
    print(f"artifact={output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
