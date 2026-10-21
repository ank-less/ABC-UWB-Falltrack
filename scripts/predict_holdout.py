#!/usr/bin/env python3
"""Fit the frozen train-only model and predict one configured stream folder."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc_uwb_falltrack.oof import (  # noqa: E402
    _decision_scores,
    _events_from_state,
    _fit_readout,
    _online_ising,
    extract_train_streams,
    load_config,
    make_targets,
    read_train_events,
)
from abc_uwb_falltrack.quantum_features import extract_stream_features, feature_names  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "quantum_histgb_final_candidate_r17.yaml",
        help="Train-only selected candidate configuration (default: R17)",
    )
    parser.add_argument(
        "--streams-dir",
        type=Path,
        required=True,
        help="Fresh holdout/test stream folder; the consumed local validation split is blocked.",
    )
    parser.add_argument(
        "--split-name",
        help="Prefix for prediction stream IDs and output files; defaults to the stream folder name.",
    )
    parser.add_argument(
        "--validation-diagnostic",
        action="store_true",
        help=(
            "Explicitly allow one descriptive run on the previously consumed local validation split; "
            "do not use its score to tune or select the model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact folder; defaults to output_dir in the selected config.",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    project_root = Path(config["project_root"])
    data_root = (project_root / config["data_root"]).resolve()
    output_dir_arg = args.output_dir or Path(config["output_dir"])
    output_dir = (
        output_dir_arg.resolve()
        if output_dir_arg.is_absolute()
        else (project_root / output_dir_arg).resolve()
    )
    cache_dir = (project_root / config["feature_cache_dir"]).resolve()
    streams_dir = args.streams_dir
    streams_dir = (
        streams_dir.resolve()
        if streams_dir.is_absolute()
        else (project_root / streams_dir).resolve()
    )
    split_name = args.split_name or streams_dir.name
    validation_dir = (data_root / "streams" / "val").resolve()
    validation_alias = any(
        re.fullmatch(r"val(?:idation)?(?:[_-].*)?", part, re.IGNORECASE)
        for part in streams_dir.parts
    )
    validation_name = bool(
        re.fullmatch(r"val(?:idation)?(?:[_-].*)?", split_name, re.IGNORECASE)
    )
    is_validation = streams_dir == validation_dir or validation_alias or validation_name
    if is_validation:
        if not args.validation_diagnostic:
            raise ValueError(
                "the local validation split was already consumed and is blocked; "
                "use --validation-diagnostic only for a fixed, descriptive one-off run"
            )
        if streams_dir != validation_dir or split_name != "val":
            raise ValueError(
                "--validation-diagnostic requires the configured local validation directory "
                "and split name 'val' so stream IDs match the ground truth"
            )
    elif args.validation_diagnostic:
        raise ValueError("--validation-diagnostic may only be used with the configured local validation split")
    if split_name.casefold() == "train":
        raise ValueError("the prediction command accepts only a fresh holdout/test split")
    prediction_name = f"{split_name}_predictions.csv"
    metadata_name = f"{split_name}_prediction_metadata.json"
    prediction_path = output_dir / prediction_name
    model_path = output_dir / "final_model.pkl"
    metadata_path = output_dir / metadata_name
    if any(path.exists() for path in (prediction_path, model_path, metadata_path)):
        try:
            output_label = str(output_dir.relative_to(project_root))
        except ValueError:
            output_label = str(output_dir)
        raise FileExistsError(
            f"prediction artifacts already exist under {output_label}; "
            "refusing to overwrite a final holdout run"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_features, names = extract_train_streams(data_root, cache_dir, config)
    train_ids = sorted(train_features)
    train_events = read_train_events(data_root / "streams_events.csv", set(train_ids))
    events_by_id: dict[str, list] = defaultdict(list)
    for event in train_events:
        events_by_id[event.stream_id].append(event)
    labels_by_id = {}
    usable_by_id = {}
    for stream_id in train_ids:
        labels, usable = make_targets(train_features[stream_id]["centers_s"], events_by_id[stream_id])
        labels_by_id[stream_id] = labels
        usable_by_id[stream_id] = usable

    model_config = dict(config["readout"])
    model_config["seed"] = int(config["training"]["seed"])
    model = _fit_readout(
        train_features,
        labels_by_id,
        usable_by_id,
        train_ids,
        model_config,
    )

    stream_paths = sorted(streams_dir.glob("stream_*.h5"))
    if not stream_paths:
        raise FileNotFoundError(f"no prediction streams found under {streams_dir}")
    decoder = config["decoder"]
    step_s = int(config["pool_frames"]) / float(config["rate_hz"])
    predictions = []
    durations_by_id: dict[str, float] = {}
    for path in stream_paths:
        index = int(path.stem.split("_")[-1])
        stream_id = f"{split_name}_{index:02d}"
        features, _centers_s, duration_s = extract_stream_features(
            path,
            rate_hz=float(config["rate_hz"]),
            pool_frames=int(config["pool_frames"]),
            frames_per_chunk=int(config["chunk_frames"]),
            static_background_tau_s=(
                float(config["static_background_tau_s"])
                if config.get("static_background_tau_s") is not None
                else None
            ),
            static_motion_gamma=float(config.get("static_motion_gamma", 1.0)),
            include_fidelity_contrast=bool(config.get("include_fidelity_contrast", False)),
            include_normalized_rho_similarity=bool(
                config.get("include_normalized_rho_similarity", False)
            ),
            include_quantum_reservoir=bool(config.get("include_quantum_reservoir", False)),
            quantum_reservoir_memory_tau_s=float(
                config.get("quantum_reservoir_memory_tau_s", 0.8)
            ),
            quantum_reservoir_phase_scale=float(
                config.get("quantum_reservoir_phase_scale", np.pi / 2.0)
            ),
            quantum_reservoir_entangler_angle=float(
                config.get("quantum_reservoir_entangler_angle", np.pi / 8.0)
            ),
        )
        durations_by_id[stream_id] = duration_s
        logits = _decision_scores(model, features)
        state = _online_ising(
            logits,
            float(decoder["bias"]),
            float(decoder["transition_cost"]),
            entry_cost=float(decoder.get("entry_cost", decoder["transition_cost"])),
            exit_cost=float(decoder.get("exit_cost", decoder["transition_cost"])),
        )
        predictions.extend(
            _events_from_state(
                stream_id,
                state,
                logits,
                duration_s,
                step_s,
                float(decoder["min_duration_s"]),
                start_offset_s=float(decoder.get("start_offset_s", 0.0)),
                end_offset_s=float(decoder.get("end_offset_s", 0.0)),
                duration_profile=decoder.get("duration_profile"),
            )
        )
        print(f"predicted {stream_id}: {sum(p['stream_id'] == stream_id for p in predictions)} events")

    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        writer.writeheader()
        writer.writerows(predictions)

    # Validate only the generated file structure and time bounds. This evaluator
    # call receives durations from the input streams and does not load labels.
    sys.path.insert(0, str(data_root / "src"))
    from abc2027.evaluate import validate_predictions  # noqa: E402

    try:
        validated_predictions = validate_predictions(prediction_path, durations_by_id)
    except Exception:
        prediction_path.unlink(missing_ok=True)
        raise

    with model_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "experiment": config["experiment_name"],
        "prediction_split": split_name,
        "selection_source": decoder["selection_source"],
        "train_oof_source_overlap_note": (
            "A posthoc train-only composition audit found 74.9%-78.3% overlap between "
            "held-out and fit ADL source-time intervals across stream_id GroupKFold splits; "
            "OOF-based candidate selection may be optimistic."
        ),
        "train_streams": train_ids,
        "train_events": len(train_events),
        "train_ignored_events": sum(event.ignore for event in train_events),
        "feature_names": feature_names(
            include_static_state=config.get("static_background_tau_s") is not None,
            include_fidelity_contrast=bool(config.get("include_fidelity_contrast", False)),
            include_normalized_rho_similarity=bool(
                config.get("include_normalized_rho_similarity", False)
            ),
            include_quantum_reservoir=bool(config.get("include_quantum_reservoir", False)),
        ),
        "readout": config["readout"],
        "decoder": decoder,
        "prediction_labels_read": False,
        "composition_file_used_for_prediction": False,
        "validation_diagnostic_only": bool(args.validation_diagnostic),
        "validation_previous_use_note": (
            "Validation has already been used for R2, R17, R23, R26, and R28 diagnostics; "
            "this diagnostic score is not independent."
            if args.validation_diagnostic
            else None
        ),
        "prediction_format_validation": {
            "status": "passed",
            "validator": "local abc2027 package evaluator",
            "validated_rows": len(validated_predictions),
            "labels_read": False,
        },
        "package_evaluator_status": "local package protocol, not confirmed as official ranking formula",
        "python_sklearn_version": sklearn.__version__,
        "prediction_count": len(predictions),
        "prediction_file": str(prediction_path.relative_to(project_root)),
        "model_file": str(model_path.relative_to(project_root)),
    }
    public_config = {key: value for key, value in config.items() if key not in {"project_root", "config_path"}}
    metadata["config"] = public_config
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"holdout_predictions={len(predictions)}")
    print(f"prediction_file={prediction_path.relative_to(project_root)}")
    print("prediction labels have not been read; no composition metadata was used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
