#!/usr/bin/env python3
"""Estimate the train-only pipeline with nested stream-grouped OOF selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc_uwb_falltrack.oof import (  # noqa: E402
    _decision_scores,
    _events_from_state,
    _fit_readout,
    _fit_boundary_readouts,
    _causal_semimarkov_filter,
    _events_from_causal_state,
    contract_event_intervals,
    _online_ising,
    _package_evaluator,
    assert_fold_partition,
    assert_train_heldout_partition,
    assert_train_stream_ids,
    extract_train_streams,
    fit_event_duration_prior,
    load_config,
    make_group_folds,
    make_boundary_targets,
    make_targets,
    read_train_events,
    refine_event_boundaries,
)


def _records_for(stream_ids: list[str], events_by_id: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stream_id": stream_id,
            "t_start": event.t_start,
            "t_end": event.t_end,
            "ignore": event.ignore,
        }
        for stream_id in stream_ids
        for event in events_by_id[stream_id]
    ]


def _ranking_key(
    row: dict[str, Any],
    objective: str = "package_mean_f1",
    minimum_f1_target: float = 0.75,
    maximum_tiou_spread: float = 0.10,
) -> tuple[float, ...]:
    ranking = row["ranking_score"]
    start_error = ranking["start_error_median_abs"]
    false_alarms = ranking["false_alarms_per_hour"]
    if objective == "minimum_f1_then_spread":
        by_theta = row.get("by_theta", {})
        values = [
            float(by_theta[str(theta)]["f1"])
            for theta in (0.1, 0.3, 0.5)
            if str(theta) in by_theta
        ]
        if len(values) != 3:
            raise ValueError(
                "minimum_f1_then_spread selection requires F1 at tIoU 0.1, 0.3 and 0.5"
            )
        spread = max(values) - min(values)
        return (
            min(values),
            -spread,
            float(ranking["f1_mean"]),
            -float(start_error) if start_error is not None else -float("inf"),
            -float(false_alarms) if false_alarms is not None else -float("inf"),
        )
    if objective == "target_floor_then_spread":
        by_theta = row.get("by_theta", {})
        values = [
            float(by_theta[str(theta)]["f1"])
            for theta in (0.1, 0.3, 0.5)
            if str(theta) in by_theta
        ]
        if len(values) != 3:
            raise ValueError(
                "target_floor_then_spread requires F1 at tIoU 0.1, 0.3 and 0.5"
            )
        floor = min(values)
        spread = max(values) - floor
        mean = float(ranking["f1_mean"])
        start_key = -float(start_error) if start_error is not None else -float("inf")
        false_alarm_key = (
            -float(false_alarms) if false_alarms is not None else -float("inf")
        )
        if floor >= minimum_f1_target and spread <= maximum_tiou_spread:
            return (1.0, -spread, mean, start_key, false_alarm_key)
        return (0.0, floor, -spread, mean, start_key, false_alarm_key)
    return (
        float(ranking["f1_mean"]),
        -float(start_error) if start_error is not None else -float("inf"),
        -float(false_alarms) if false_alarms is not None else -float("inf"),
    )


def _select_semimarkov_decoder(
    logits_by_id: dict[str, np.ndarray],
    stream_ids: list[str],
    events_by_id: dict[str, list[Any]],
    durations: dict[str, float],
    inner_folds: list[list[str]],
    config: dict[str, Any],
    step_s: float,
    evaluate: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose causal semi-Markov settings from fold-scoped inner OOF predictions."""
    if set(logits_by_id) != set(stream_ids):
        raise ValueError("semi-Markov logits must cover exactly the active outer-train scope")
    if set(events_by_id) != set(stream_ids):
        raise ValueError("semi-Markov events must cover exactly the active outer-train scope")
    if set(durations) != set(stream_ids):
        raise ValueError("semi-Markov durations must cover exactly the active outer-train scope")
    assert_fold_partition(stream_ids, inner_folds, scope="semi-Markov inner folds")

    search = config["decoder_search"]
    logits_all = np.concatenate([logits_by_id[key] for key in stream_ids])
    quantiles = [float(value) for value in search["logit_quantiles"]]
    biases = sorted(set(float(value) for value in np.quantile(logits_all, quantiles)))
    entry_costs = [float(value) for value in search["entry_costs"]]
    exit_costs = [float(value) for value in search["exit_costs"]]
    duration_penalties = [float(value) for value in search["duration_penalties"]]
    thresholds_on = [float(value) for value in search["thresholds_on"]]
    min_durations = [float(value) for value in search["min_duration_s"]]
    max_duration_s = float(search["max_duration_s"])
    threshold_off_ratio = float(search.get("threshold_off_ratio", 0.6))
    start_offset_s = float(search.get("start_offset_s", 0.0))
    end_offset_s = float(search.get("end_offset_s", 0.0))
    scoped_durations = {stream_id: durations[stream_id] for stream_id in stream_ids}
    ground_truth = _records_for(stream_ids, events_by_id)

    fold_scopes = []
    fold_priors = []
    for fold_index, inner_held_out in enumerate(inner_folds):
        held_out_set = set(inner_held_out)
        inner_train = [stream_id for stream_id in stream_ids if stream_id not in held_out_set]
        assert_train_heldout_partition(
            stream_ids,
            inner_train,
            inner_held_out,
            scope=f"semi-Markov inner fold {fold_index}",
        )
        prior = fit_event_duration_prior(events_by_id, inner_train)
        if set(prior["fit_stream_ids"]).intersection(inner_held_out):
            raise ValueError("inner held-out streams entered the duration-prior fit")
        fold_priors.append(prior)
        fold_scopes.append(
            {
                "fold": fold_index,
                "fit_stream_ids": inner_train,
                "held_out_stream_ids": inner_held_out,
                "valid_event_count": prior["n_valid_events"],
                "prior_enabled": prior["enabled"],
                "mu_log_s": prior["mu_log_s"],
                "sigma_log_s": prior["sigma_log_s"],
            }
        )

    candidates = []
    for bias in biases:
        for entry_cost in entry_costs:
            for exit_cost in exit_costs:
                for duration_penalty in duration_penalties:
                    for threshold_on in thresholds_on:
                        for min_duration_s in min_durations:
                            predictions = []
                            for fold_index, inner_held_out in enumerate(inner_folds):
                                prior = fold_priors[fold_index]
                                for stream_id in inner_held_out:
                                    state, posterior = _causal_semimarkov_filter(
                                        logits_by_id[stream_id],
                                        bias,
                                        entry_cost,
                                        exit_cost,
                                        prior,
                                        step_s,
                                        duration_penalty=duration_penalty,
                                        max_duration_s=max_duration_s,
                                        threshold_on=threshold_on,
                                        threshold_off_ratio=threshold_off_ratio,
                                    )
                                    predictions.extend(
                                        _events_from_causal_state(
                                            stream_id,
                                            state,
                                            posterior,
                                            logits_by_id[stream_id],
                                            durations[stream_id],
                                            step_s,
                                            min_duration_s,
                                            start_offset_s=start_offset_s,
                                            end_offset_s=end_offset_s,
                                        )
                                    )
                            if {
                                prediction["stream_id"] for prediction in predictions
                            }.difference(stream_ids):
                                raise ValueError("semi-Markov candidate emitted a stream outside its train scope")
                            report = evaluate(ground_truth, predictions, scoped_durations, theta=0.3)
                            candidates.append(
                                {
                                    "decoder_method": "causal_semimarkov",
                                    "bias": bias,
                                    "transition_cost": entry_cost,
                                    "entry_cost": entry_cost,
                                    "exit_cost": exit_cost,
                                    "duration_penalty": duration_penalty,
                                    "threshold_on": threshold_on,
                                    "threshold_off_ratio": threshold_off_ratio,
                                    "min_duration_s": min_duration_s,
                                    "max_duration_s": max_duration_s,
                                    "start_offset_s": start_offset_s,
                                    "end_offset_s": end_offset_s,
                                    "n_predictions": len(predictions),
                                    "ranking_score": report["ranking_score"],
                                    "primary": report["primary"],
                                    "by_theta": report["by_theta"],
                                }
                            )
        candidates.sort(
            key=lambda row: _ranking_key(
                row,
                str(config.get("inner_selection_objective", "package_mean_f1")),
                float(config.get("requested_internal_min_f1", 0.75)),
                float(config.get("maximum_target_tiou_spread", 0.10)),
            ),
            reverse=True,
        )
    best = candidates[0]
    best["inner_duration_prior_fits"] = fold_scopes
    return best, candidates


def _select_decoder(
    logits_by_id: dict[str, np.ndarray],
    stream_ids: list[str],
    events_by_id: dict[str, list[Any]],
    durations: dict[str, float],
    inner_folds: list[list[str]],
    config: dict[str, Any],
    step_s: float,
    evaluate: Any,
    *,
    onset_scores_by_id: dict[str, np.ndarray] | None = None,
    offset_scores_by_id: dict[str, np.ndarray] | None = None,
    centers_by_id: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(logits_by_id) != set(stream_ids):
        raise ValueError("decoder selection logits must cover exactly the active train scope")
    if set(events_by_id) != set(stream_ids):
        raise ValueError("decoder selection events must cover exactly the active train scope")
    if set(durations) != set(stream_ids):
        raise ValueError("decoder selection durations must cover exactly the active train scope")
    search = config["decoder_search"]
    boundary_config = config.get("boundary_refinement", {})
    boundary_enabled = bool(boundary_config.get("enabled", False))
    duration_calibration_config = config.get("duration_calibration", {})
    duration_calibration_enabled = bool(duration_calibration_config.get("enabled", False))
    if duration_calibration_enabled:
        duration_scales = [float(value) for value in duration_calibration_config["scales"]]
        if not duration_scales or any(
            not np.isfinite(value) or not 0.0 < value <= 1.0 for value in duration_scales
        ):
            raise ValueError("duration calibration scales must be a non-empty list in (0, 1]")
    else:
        duration_scales = [1.0]
    if boundary_enabled:
        if search.get("method", "online_ising") == "causal_semimarkov":
            raise ValueError("boundary refinement is currently supported with online Ising only")
        if onset_scores_by_id is None or offset_scores_by_id is None or centers_by_id is None:
            raise ValueError("boundary refinement requires inner OOF onset/offset scores and centers")
        for name, values in (
            ("onset scores", onset_scores_by_id),
            ("offset scores", offset_scores_by_id),
            ("centers", centers_by_id),
        ):
            if set(values) != set(stream_ids):
                raise ValueError(f"inner OOF {name} must cover exactly the active train scope")
        boundary_radii = [float(value) for value in boundary_config["radii_s"]]
        if not boundary_radii or any(not np.isfinite(value) or value < 0 for value in boundary_radii):
            raise ValueError("boundary refinement radii must be a non-empty list of finite non-negative values")
    else:
        boundary_radii = [0.0]
    if search.get("method", "online_ising") == "causal_semimarkov":
        return _select_semimarkov_decoder(
            logits_by_id,
            stream_ids,
            events_by_id,
            durations,
            inner_folds,
            config,
            step_s,
            evaluate,
        )
    logits_all = np.concatenate([logits_by_id[key] for key in stream_ids])
    quantiles = [float(value) for value in search["logit_quantiles"]]
    biases = sorted(set(float(value) for value in np.quantile(logits_all, quantiles)))
    transition_costs = [float(value) for value in search["transition_costs"]]
    exit_costs = (
        [float(value) for value in search["exit_costs"]]
        if "exit_costs" in search
        else None
    )
    min_durations = [float(value) for value in search["min_duration_s"]]
    start_offsets = [float(value) for value in search.get("start_offsets_s", [0.0])]
    end_offsets = [float(value) for value in search.get("end_offsets_s", [0.0])]
    duration_profiles = search.get("duration_profiles") or [{"name": "uniform"}]
    ground_truth = _records_for(stream_ids, events_by_id)
    scoped_durations = {stream_id: durations[stream_id] for stream_id in stream_ids}

    candidates = []
    for bias in biases:
        for transition_cost in transition_costs:
            candidate_exit_costs = exit_costs or [transition_cost]
            for exit_cost in candidate_exit_costs:
                # State boundaries do not depend on min duration; retain the
                # Ising path while regenerating only the event list per duration.
                states = {
                    stream_id: _online_ising(
                        logits_by_id[stream_id],
                        bias,
                        transition_cost,
                        entry_cost=transition_cost,
                        exit_cost=exit_cost,
                    )
                    for stream_id in stream_ids
                }
                for min_duration_s in min_durations:
                    for start_offset_s in start_offsets:
                        for end_offset_s in end_offsets:
                            base_predictions = []
                            for stream_id in stream_ids:
                                base_predictions.extend(
                                    _events_from_state(
                                        stream_id,
                                        states[stream_id],
                                        logits_by_id[stream_id],
                                        durations[stream_id],
                                        step_s,
                                        min_duration_s,
                                    )
                                )
                            base_predicted_durations = np.asarray(
                                [event["t_end"] - event["t_start"] for event in base_predictions],
                                dtype=np.float64,
                            )
                            for duration_profile_spec in duration_profiles:
                                duration_profile = dict(duration_profile_spec)
                                quantiles_for_threshold = duration_profile.get("threshold_quantiles")
                                if quantiles_for_threshold:
                                    duration_profile["thresholds_s"] = (
                                        np.quantile(
                                            base_predicted_durations,
                                            [float(value) for value in quantiles_for_threshold],
                                        ).astype(float).tolist()
                                        if base_predicted_durations.size
                                        else [float("inf")] * len(quantiles_for_threshold)
                                    )
                                base_predictions = []
                                for stream_id in stream_ids:
                                    base_predictions.extend(
                                        _events_from_state(
                                            stream_id,
                                            states[stream_id],
                                            logits_by_id[stream_id],
                                            durations[stream_id],
                                            step_s,
                                            min_duration_s,
                                            start_offset_s=start_offset_s,
                                            end_offset_s=end_offset_s,
                                            duration_profile=duration_profile,
                                        )
                                    )
                                for boundary_radius_s in boundary_radii:
                                    predictions = base_predictions
                                    if boundary_enabled and boundary_radius_s > 0.0:
                                        predictions = refine_event_boundaries(
                                            base_predictions,
                                            onset_scores_by_id,
                                            offset_scores_by_id,
                                            centers_by_id,
                                            radius_s=boundary_radius_s,
                                            band_s=float(boundary_config["band_s"]),
                                            step_s=step_s,
                                            min_duration_s=min_duration_s,
                                        )
                                    for duration_scale in duration_scales:
                                        calibrated_predictions = predictions
                                        if duration_calibration_enabled:
                                            calibrated_predictions = contract_event_intervals(
                                                predictions,
                                                scale=duration_scale,
                                                min_duration_s=min_duration_s,
                                            )
                                        report = evaluate(
                                            ground_truth,
                                            calibrated_predictions,
                                            scoped_durations,
                                            theta=0.3,
                                        )
                                        candidates.append(
                                            {
                                                "bias": bias,
                                                "transition_cost": transition_cost,
                                                "entry_cost": transition_cost,
                                                "exit_cost": exit_cost,
                                                "min_duration_s": min_duration_s,
                                                "start_offset_s": start_offset_s,
                                                "end_offset_s": end_offset_s,
                                                "duration_profile": duration_profile,
                                                "boundary_radius_s": boundary_radius_s,
                                                "boundary_band_s": float(boundary_config["band_s"]) if boundary_enabled else None,
                                                "duration_scale": duration_scale,
                                                "n_predictions": len(calibrated_predictions),
                                                "ranking_score": report["ranking_score"],
                                                "primary": report["primary"],
                                                "by_theta": report["by_theta"],
                                            }
                                        )
    objective = str(config.get("inner_selection_objective", "package_mean_f1"))
    candidates.sort(
        key=lambda row: _ranking_key(
            row,
            objective,
            float(config.get("requested_internal_min_f1", 0.75)),
            float(config.get("maximum_target_tiou_spread", 0.10)),
        ),
        reverse=True,
    )
    return candidates[0], candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "quantum_histgb_nested_oof.yaml",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    project_root = Path(config["project_root"])
    output_dir = (project_root / config["output_dir"]).resolve()
    report_path = output_dir / "nested_oof_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path.relative_to(project_root)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = (project_root / config["data_root"]).resolve()
    cache_dir = (project_root / config["feature_cache_dir"]).resolve()
    feature_candidates = config.get("feature_candidates") or [{"name": "base", "overrides": {}}]
    features_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    names_by_variant: dict[str, list[str]] = {}
    feature_configs: dict[str, dict[str, Any]] = {}
    for feature_index, feature_candidate in enumerate(feature_candidates):
        feature_name = str(feature_candidate.get("name", f"feature_{feature_index}"))
        feature_config = dict(config)
        feature_config.update(feature_candidate.get("overrides", {}))
        variant_cache_dir = cache_dir / feature_name if config.get("feature_candidates") else cache_dir
        variant_features, variant_names = extract_train_streams(
            data_root, variant_cache_dir, feature_config
        )
        features_by_variant[feature_name] = variant_features
        names_by_variant[feature_name] = variant_names
        feature_configs[feature_name] = {
            key: feature_config.get(key)
            for key in (
                "static_background_tau_s",
                "static_motion_gamma",
                "include_fidelity_contrast",
                "include_normalized_rho_similarity",
                "include_quantum_reservoir",
                "quantum_reservoir_memory_tau_s",
                "quantum_reservoir_phase_scale",
                "quantum_reservoir_entangler_angle",
            )
            if key in feature_config
        }
    base_feature_name = str(feature_candidates[0].get("name", "base"))
    features_by_id = features_by_variant[base_feature_name]
    names = names_by_variant[base_feature_name]
    stream_ids = sorted(features_by_id)
    assert_train_stream_ids(stream_ids, scope="nested OOF train universe")
    train_events = read_train_events(data_root / "streams_events.csv", set(stream_ids))
    events_by_id: dict[str, list[Any]] = {stream_id: [] for stream_id in stream_ids}
    for event in train_events:
        events_by_id[event.stream_id].append(event)

    labels_by_id = {}
    usable_by_id = {}
    for stream_id in stream_ids:
        labels_by_id[stream_id], usable_by_id[stream_id] = make_targets(
            features_by_id[stream_id]["centers_s"], events_by_id[stream_id]
        )

    boundary_readout_config = config.get("boundary_readout", {})
    boundary_refinement_config = config.get("boundary_refinement", {})
    duration_calibration_config = config.get("duration_calibration", {})
    duration_calibration_enabled = bool(duration_calibration_config.get("enabled", False))
    boundary_enabled = bool(boundary_readout_config.get("enabled", False))
    if bool(boundary_refinement_config.get("enabled", False)) != boundary_enabled:
        raise ValueError("boundary readouts and boundary refinement must be enabled together")
    if boundary_enabled:
        boundary_band_s = float(boundary_readout_config["band_s"])
        if not np.isfinite(boundary_band_s) or boundary_band_s <= 0:
            raise ValueError("boundary readout band_s must be finite and positive")
        if not np.isclose(float(boundary_refinement_config["band_s"]), boundary_band_s):
            raise ValueError("boundary target and refinement band_s must match")
        onset_targets_by_id = {}
        offset_targets_by_id = {}
        boundary_usable_by_id = {}
        for stream_id in stream_ids:
            (
                onset_targets_by_id[stream_id],
                offset_targets_by_id[stream_id],
                boundary_usable_by_id[stream_id],
            ) = make_boundary_targets(
                features_by_id[stream_id]["centers_s"],
                events_by_id[stream_id],
                band_s=boundary_band_s,
            )
    else:
        onset_targets_by_id = {}
        offset_targets_by_id = {}
        boundary_usable_by_id = {}

    cv = config["cross_validation"]
    outer_seed = int(cv["seed"])
    outer_folds = make_group_folds(stream_ids, int(cv["outer_splits"]), outer_seed)
    assert_fold_partition(stream_ids, outer_folds, scope="outer folds")
    inner_splits = int(cv["inner_splits"])
    readout_candidates = config.get("readout_candidates") or [{"name": "base", "overrides": {}}]
    step_s = int(config["pool_frames"]) / float(config["rate_hz"])
    durations = {key: float(features_by_id[key]["duration_s"]) for key in stream_ids}
    evaluate = _package_evaluator(data_root)
    nested_logits: dict[str, np.ndarray] = {}
    nested_predictions: list[dict[str, Any]] = []
    fold_results = []
    inner_boundary_fit_scope_checks = []
    inner_boundary_score_scope_checks = []
    outer_boundary_fit_scope_checks = []
    outer_boundary_score_scope_checks = []

    for outer_index, held_out in enumerate(outer_folds):
        held_out_set = set(held_out)
        outer_train = [stream_id for stream_id in stream_ids if stream_id not in held_out_set]
        assert_train_heldout_partition(
            stream_ids, outer_train, held_out, scope=f"outer fold {outer_index}"
        )
        inner_seed = outer_seed + 1000 + outer_index
        inner_folds = make_group_folds(outer_train, inner_splits, inner_seed)
        assert_fold_partition(outer_train, inner_folds, scope=f"outer fold {outer_index} inner folds")
        inner_fold_records = []
        for inner_index, inner_held_out in enumerate(inner_folds):
            inner_held_out_set = set(inner_held_out)
            inner_train = [key for key in outer_train if key not in inner_held_out_set]
            assert_train_heldout_partition(
                outer_train,
                inner_train,
                inner_held_out,
                scope=f"outer fold {outer_index} inner fold {inner_index}",
            )
            if held_out_set.intersection(inner_train) or held_out_set.intersection(inner_held_out):
                raise ValueError("outer-held streams entered an inner fit or held-out scope")
            inner_fold_records.append(
                {
                    "fold": inner_index,
                    "train_streams": inner_train,
                    "held_out_streams": inner_held_out,
                }
            )

        candidate_results = []
        for feature_candidate in feature_candidates:
            feature_name = str(feature_candidate.get("name", "base"))
            variant_features = features_by_variant[feature_name]
            for candidate_index, candidate in enumerate(readout_candidates):
                readout_name = str(candidate.get("name", f"variant_{candidate_index}"))
                variant_name = f"{feature_name}/{readout_name}"
                variant_config = dict(config["readout"])
                variant_config.update(candidate.get("overrides", {}))
                inner_logits: dict[str, np.ndarray] = {}
                inner_onset_scores: dict[str, np.ndarray] = {}
                inner_offset_scores: dict[str, np.ndarray] = {}
                inner_boundary_fit_scopes = []
                for inner_index, inner_held_out in enumerate(inner_folds):
                    inner_train = [key for key in outer_train if key not in set(inner_held_out)]
                    assert_train_heldout_partition(
                        outer_train,
                        inner_train,
                        inner_held_out,
                        scope=f"outer fold {outer_index} inner fold {inner_index}",
                    )
                    model_config = dict(variant_config)
                    model_config["seed"] = inner_seed
                    scoped_features = {key: variant_features[key] for key in inner_train}
                    scoped_labels = {key: labels_by_id[key] for key in inner_train}
                    scoped_usable = {key: usable_by_id[key] for key in inner_train}
                    model = _fit_readout(
                        scoped_features,
                        scoped_labels,
                        scoped_usable,
                        inner_train,
                        model_config,
                    )
                    if boundary_enabled:
                        boundary_model_config = dict(model_config)
                        boundary_model_config.update(
                            boundary_readout_config.get("model_overrides", {})
                        )
                        boundary_model_config["positive_weight"] = float(
                            boundary_readout_config.get(
                                "positive_weight", model_config["positive_weight"]
                            )
                        )
                        scoped_onset_targets = {
                            key: onset_targets_by_id[key] for key in inner_train
                        }
                        scoped_offset_targets = {
                            key: offset_targets_by_id[key] for key in inner_train
                        }
                        scoped_boundary_usable = {
                            key: boundary_usable_by_id[key] for key in inner_train
                        }
                        onset_model, offset_model = _fit_boundary_readouts(
                            scoped_features,
                            scoped_onset_targets,
                            scoped_offset_targets,
                            scoped_boundary_usable,
                            inner_train,
                            boundary_model_config,
                        )
                        fit_scope_ok = (
                            set(scoped_features)
                            == set(scoped_onset_targets)
                            == set(scoped_offset_targets)
                            == set(scoped_boundary_usable)
                            == set(inner_train)
                            and not set(inner_held_out).intersection(scoped_features)
                            and not held_out_set.intersection(scoped_features)
                        )
                        inner_boundary_fit_scope_checks.append(fit_scope_ok)
                        if not fit_scope_ok:
                            raise ValueError("inner boundary heads were not fit on the exact inner-train scope")
                        inner_boundary_fit_scopes.append(
                            {
                                "fold": inner_index,
                                "fit_stream_ids": inner_train,
                                "scored_stream_ids": inner_held_out,
                            }
                        )
                    for stream_id in inner_held_out:
                        inner_logits[stream_id] = _decision_scores(
                            model, variant_features[stream_id]["features"]
                        )
                        if boundary_enabled:
                            if stream_id in inner_onset_scores or stream_id in inner_offset_scores:
                                raise ValueError("inner boundary scores were generated more than once")
                            inner_onset_scores[stream_id] = _decision_scores(
                                onset_model, variant_features[stream_id]["features"]
                            )
                            inner_offset_scores[stream_id] = _decision_scores(
                                offset_model, variant_features[stream_id]["features"]
                            )
                    print(
                        f"outer={outer_index + 1}/{len(outer_folds)} "
                        f"variant={variant_name} inner={inner_index + 1}/{len(inner_folds)} "
                        f"train_streams={len(inner_train)} oof_streams={len(inner_held_out)}"
                    )

                if set(inner_logits) != set(outer_train):
                    raise ValueError("inner OOF logits must cover each outer-train stream exactly once")
                if boundary_enabled:
                    if set(inner_onset_scores) != set(outer_train) or set(inner_offset_scores) != set(outer_train):
                        raise ValueError("inner OOF boundary scores must cover each outer-train stream exactly once")
                    score_scope_ok = (
                        set(inner_onset_scores)
                        == set(inner_offset_scores)
                        == set(outer_train)
                        and not held_out_set.intersection(inner_onset_scores)
                        and not held_out_set.intersection(inner_offset_scores)
                    )
                    inner_boundary_score_scope_checks.append(score_scope_ok)
                    if not score_scope_ok:
                        raise ValueError("inner boundary scores escaped their outer-train scope")
                scoped_events = {key: events_by_id[key] for key in outer_train}
                scoped_durations = {key: durations[key] for key in outer_train}
                best_decoder, decoder_candidates = _select_decoder(
                    inner_logits,
                    outer_train,
                    scoped_events,
                    scoped_durations,
                    inner_folds,
                    config,
                    step_s,
                    evaluate,
                    onset_scores_by_id=inner_onset_scores if boundary_enabled else None,
                    offset_scores_by_id=inner_offset_scores if boundary_enabled else None,
                    centers_by_id=(
                        {key: variant_features[key]["centers_s"] for key in outer_train}
                        if boundary_enabled
                        else None
                    ),
                )
                candidate_results.append(
                    {
                        "name": variant_name,
                        "feature_variant": feature_name,
                        "feature_config": feature_configs[feature_name],
                        "readout_variant": readout_name,
                        "readout": variant_config,
                        "selected_decoder": best_decoder,
                        "inner_boundary_fit_scopes": inner_boundary_fit_scopes,
                        "inner_top_decoders": decoder_candidates[:5],
                    }
                )

        selection_objective = str(config.get("inner_selection_objective", "package_mean_f1"))
        candidate_results.sort(
            key=lambda result: _ranking_key(
                result["selected_decoder"],
                selection_objective,
                float(config.get("requested_internal_min_f1", 0.75)),
                float(config.get("maximum_target_tiou_spread", 0.10)),
            ),
            reverse=True,
        )
        selected_variant = candidate_results[0]
        best_decoder = selected_variant["selected_decoder"]
        selected_features = features_by_variant[selected_variant["feature_variant"]]
        final_model_config = dict(selected_variant["readout"])
        final_model_config["seed"] = outer_seed + outer_index
        outer_train_features = {key: selected_features[key] for key in outer_train}
        outer_train_labels = {key: labels_by_id[key] for key in outer_train}
        outer_train_usable = {key: usable_by_id[key] for key in outer_train}
        final_model = _fit_readout(
            outer_train_features,
            outer_train_labels,
            outer_train_usable,
            outer_train,
            final_model_config,
        )
        final_onset_model = None
        final_offset_model = None
        if boundary_enabled:
            boundary_model_config = dict(final_model_config)
            boundary_model_config.update(
                boundary_readout_config.get("model_overrides", {})
            )
            boundary_model_config["positive_weight"] = float(
                boundary_readout_config.get(
                    "positive_weight", final_model_config["positive_weight"]
                )
            )
            outer_onset_targets = {key: onset_targets_by_id[key] for key in outer_train}
            outer_offset_targets = {key: offset_targets_by_id[key] for key in outer_train}
            outer_boundary_usable = {key: boundary_usable_by_id[key] for key in outer_train}
            final_onset_model, final_offset_model = _fit_boundary_readouts(
                outer_train_features,
                outer_onset_targets,
                outer_offset_targets,
                outer_boundary_usable,
                outer_train,
                boundary_model_config,
            )
            outer_fit_scope_ok = (
                set(outer_train_features)
                == set(outer_onset_targets)
                == set(outer_offset_targets)
                == set(outer_boundary_usable)
                == set(outer_train)
                and not held_out_set.intersection(outer_train_features)
            )
            outer_boundary_fit_scope_checks.append(outer_fit_scope_ok)
            if not outer_fit_scope_ok:
                raise ValueError("outer boundary heads were not fit on the exact outer-train scope")
        decoder_method = best_decoder.get("decoder_method", "online_ising")
        outer_duration_prior = None
        if decoder_method == "causal_semimarkov":
            outer_duration_prior = fit_event_duration_prior(events_by_id, outer_train)
            if set(outer_duration_prior["fit_stream_ids"]).intersection(held_out):
                raise ValueError("outer held-out streams entered the duration-prior fit")
            if set(outer_duration_prior["fit_stream_ids"]) != set(outer_train):
                raise ValueError("outer duration prior did not use exactly the outer-train scope")
        if set(nested_logits).intersection(held_out_set):
            raise ValueError("a stream is receiving more than one outer OOF prediction")
        fold_predictions = []
        outer_boundary_scored_stream_ids = []
        for stream_id in held_out:
            logits = _decision_scores(final_model, selected_features[stream_id]["features"])
            nested_logits[stream_id] = logits
            onset_scores = None
            offset_scores = None
            if boundary_enabled:
                onset_scores = _decision_scores(
                    final_onset_model, selected_features[stream_id]["features"]
                )
                offset_scores = _decision_scores(
                    final_offset_model, selected_features[stream_id]["features"]
                )
                outer_boundary_scored_stream_ids.append(stream_id)
            if decoder_method == "causal_semimarkov":
                state, posterior = _causal_semimarkov_filter(
                    logits,
                    best_decoder["bias"],
                    best_decoder["entry_cost"],
                    best_decoder["exit_cost"],
                    outer_duration_prior,
                    step_s,
                    duration_penalty=best_decoder["duration_penalty"],
                    max_duration_s=best_decoder["max_duration_s"],
                    threshold_on=best_decoder["threshold_on"],
                    threshold_off_ratio=best_decoder["threshold_off_ratio"],
                )
                predictions = _events_from_causal_state(
                    stream_id,
                    state,
                    posterior,
                    logits,
                    durations[stream_id],
                    step_s,
                    best_decoder["min_duration_s"],
                    start_offset_s=best_decoder["start_offset_s"],
                    end_offset_s=best_decoder["end_offset_s"],
                )
            else:
                state = _online_ising(
                    logits,
                    best_decoder["bias"],
                    best_decoder["transition_cost"],
                    entry_cost=best_decoder.get("entry_cost", best_decoder["transition_cost"]),
                    exit_cost=best_decoder.get("exit_cost", best_decoder["transition_cost"]),
                )
                predictions = _events_from_state(
                    stream_id,
                    state,
                    logits,
                    durations[stream_id],
                    step_s,
                    best_decoder["min_duration_s"],
                    start_offset_s=best_decoder.get("start_offset_s", 0.0),
                    end_offset_s=best_decoder.get("end_offset_s", 0.0),
                    duration_profile=best_decoder.get("duration_profile"),
                )
            if boundary_enabled:
                predictions = refine_event_boundaries(
                    predictions,
                    {stream_id: onset_scores},
                    {stream_id: offset_scores},
                    {stream_id: selected_features[stream_id]["centers_s"]},
                    radius_s=float(best_decoder["boundary_radius_s"]),
                    band_s=float(boundary_refinement_config["band_s"]),
                    step_s=step_s,
                    min_duration_s=float(best_decoder["min_duration_s"]),
                )
            if duration_calibration_enabled:
                predictions = contract_event_intervals(
                    predictions,
                    scale=float(best_decoder["duration_scale"]),
                    min_duration_s=float(best_decoder["min_duration_s"]),
                )
            nested_predictions.extend(predictions)
            fold_predictions.extend(predictions)

        if boundary_enabled:
            outer_score_scope_ok = (
                set(outer_boundary_scored_stream_ids) == held_out_set
                and not set(outer_train).intersection(outer_boundary_scored_stream_ids)
            )
            outer_boundary_score_scope_checks.append(outer_score_scope_ok)
            if not outer_score_scope_ok:
                raise ValueError("outer boundary scores were not restricted to held-out streams")

        if set(nested_logits).intersection(held_out_set) != held_out_set:
            raise ValueError("outer OOF logits are missing one or more held-out streams")
        if any(prediction["stream_id"] not in held_out_set for prediction in fold_predictions):
            raise ValueError("outer fold emitted a prediction outside its held-out streams")

        fold_report = evaluate(
            _records_for(held_out, events_by_id),
            fold_predictions,
            {key: durations[key] for key in held_out},
            theta=0.3,
        )
        fold_results.append(
            {
                "fold": outer_index,
                "train_streams": outer_train,
                "held_out_streams": held_out,
                "inner_seed": inner_seed,
                "inner_folds": inner_fold_records,
                "selected_feature_variant": selected_variant["feature_variant"],
                "selected_feature_config": selected_variant["feature_config"],
                "selected_readout_variant": selected_variant["readout_variant"],
                "selected_readout": selected_variant["readout"],
                "inner_boundary_fit_scopes": selected_variant["inner_boundary_fit_scopes"],
                "selected_decoder": best_decoder,
                "outer_boundary_fit_stream_ids": outer_train if boundary_enabled else [],
                "outer_boundary_scored_stream_ids": outer_boundary_scored_stream_ids,
                "outer_duration_prior_fit": outer_duration_prior,
                "inner_candidate_ranking": [
                    {
                        "name": candidate["name"],
                        "feature_variant": candidate["feature_variant"],
                        "feature_config": candidate["feature_config"],
                        "readout_variant": candidate["readout_variant"],
                        "readout": candidate["readout"],
                        "inner_boundary_fit_scopes": candidate["inner_boundary_fit_scopes"],
                        "selected_decoder": candidate["selected_decoder"],
                    }
                    for candidate in candidate_results
                ],
                "outer_metrics": fold_report,
            }
        )
        print(
            f"outer={outer_index + 1}/{len(outer_folds)} "
            f"held_out={','.join(held_out)} "
            f"readout={selected_variant['name']} "
            f"decoder={decoder_method} bias={best_decoder['bias']:.5f} "
            f"entry/exit={best_decoder.get('entry_cost', best_decoder['transition_cost']):.2f}/"
            f"{best_decoder.get('exit_cost', best_decoder['transition_cost']):.2f} "
            f"duration={best_decoder['min_duration_s']:.2f}s "
            f"scale={best_decoder.get('duration_scale', 1.0):.2f} "
            f"outer_f1_mean={fold_report['ranking_score']['f1_mean']:.4f}"
        )

    if set(nested_logits) != set(stream_ids):
        raise ValueError("nested OOF logits must cover every train stream exactly once")
    if any(prediction["stream_id"] not in set(stream_ids) for prediction in nested_predictions):
        raise ValueError("nested OOF predictions contain a non-train stream")
    overall_report = evaluate(
        _records_for(stream_ids, events_by_id),
        nested_predictions,
        durations,
        theta=0.3,
    )
    outer_fold_metric_spread = {}
    for theta in ("0.1", "0.3", "0.5"):
        values = np.asarray(
            [fold["outer_metrics"]["by_theta"][theta]["f1"] for fold in fold_results],
            dtype=np.float64,
        )
        outer_fold_metric_spread[theta] = {
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "fold_values": values.astype(float).tolist(),
        }
    inner_duration_priors_fit_on_scope = True
    outer_duration_priors_fit_on_scope = True
    if config["decoder_search"].get("method", "online_ising") == "causal_semimarkov":
        for fold in fold_results:
            outer_train = set(fold["train_streams"])
            inner_prior_fits = fold["selected_decoder"].get("inner_duration_prior_fits", [])
            if len(inner_prior_fits) != len(fold["inner_folds"]):
                inner_duration_priors_fit_on_scope = False
                continue
            for prior_fit, inner_fold in zip(inner_prior_fits, fold["inner_folds"]):
                expected_inner_train = set(inner_fold["train_streams"])
                if set(prior_fit["fit_stream_ids"]) != expected_inner_train:
                    inner_duration_priors_fit_on_scope = False
                if set(prior_fit["fit_stream_ids"]).intersection(inner_fold["held_out_streams"]):
                    inner_duration_priors_fit_on_scope = False
            prior = fold["outer_duration_prior_fit"]
            if prior is None or set(prior["fit_stream_ids"]) != outer_train:
                outer_duration_priors_fit_on_scope = False
            if prior is None or set(prior["fit_stream_ids"]).intersection(fold["held_out_streams"]):
                outer_duration_priors_fit_on_scope = False
    report = {
        "experiment": config["experiment_name"],
        "data_status": "local package only; nested grouped OOF on train streams",
        "selection_protocol": "outer stream-grouped OOF; feature, readout, decoder, and enabled interval calibration selected on inner stream-grouped OOF",
        "inner_selection_objective": config.get("inner_selection_objective", "package_mean_f1"),
        "decoder_method": config["decoder_search"].get("method", "online_ising"),
        "decoder_search": config["decoder_search"],
        "step_s": step_s,
        "validation_used": False,
        "composition_file_used": False,
        "ignore_flagged": True,
        "boundary_readout": boundary_readout_config if boundary_enabled else {"enabled": False},
        "boundary_refinement": boundary_refinement_config if boundary_enabled else {"enabled": False},
        "duration_calibration": duration_calibration_config if duration_calibration_enabled else {"enabled": False},
        "feature_names": names,
        "feature_names_by_variant": names_by_variant,
        "feature_variants": feature_configs,
        "train_streams": stream_ids,
        "train_events": len(train_events),
        "train_ignored_events": sum(event.ignore for event in train_events),
        "outer_folds": fold_results,
        "outer_fold_metric_spread": outer_fold_metric_spread,
        "nested_oof_metrics": overall_report,
        "nested_oof_logits_file": "nested_oof_logits_train_only.npz",
        "nested_oof_predictions_file": "nested_oof_predictions_train_only.csv",
        "zero_leakage_checks": {
            "train_ids_match_train_namespace": True,
            "outer_folds_partition_train_universe": True,
            "inner_folds_partition_each_outer_train_scope": True,
            "outer_train_heldout_disjoint": True,
            "outer_heldout_absent_from_inner_folds": True,
            "inner_oof_covers_each_outer_train_stream_once": True,
            "outer_oof_covers_each_train_stream_once": True,
            "predictions_reference_train_streams_only": True,
            "inner_duration_priors_fit_on_inner_train_only": inner_duration_priors_fit_on_scope,
            "outer_duration_priors_fit_on_outer_train_only": outer_duration_priors_fit_on_scope,
            "validation_used": False,
            "composition_file_used": False,
            "inner_boundary_heads_fit_on_inner_train_only": (
                all(inner_boundary_fit_scope_checks) if boundary_enabled else True
            ),
            "inner_boundary_scores_limited_to_outer_train_oof": (
                all(inner_boundary_score_scope_checks) if boundary_enabled else True
            ),
            "outer_boundary_heads_fit_on_outer_train_only": (
                all(outer_boundary_fit_scope_checks) if boundary_enabled else True
            ),
            "outer_boundary_scores_limited_to_outer_held_out": (
                all(outer_boundary_score_scope_checks) if boundary_enabled else True
            ),
        },
    }
    if not inner_duration_priors_fit_on_scope or not outer_duration_priors_fit_on_scope:
        raise ValueError("duration-prior fit scopes failed the nested leakage audit")
    np.savez_compressed(
        output_dir / "nested_oof_logits_train_only.npz",
        **{key: value for key, value in nested_logits.items()},
    )
    with (output_dir / "nested_oof_predictions_train_only.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"],
        )
        writer.writeheader()
        writer.writerows(nested_predictions)
    public_config = {
        key: value for key, value in config.items() if key not in {"project_root", "config_path"}
    }
    (output_dir / "config_used.json").write_text(json.dumps(public_config, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    score = overall_report["ranking_score"]
    print(
        "nested_oof "
        f"mean_event_f1={score['f1_mean']:.4f} "
        f"f1@0.1={overall_report['by_theta']['0.1']['f1']:.4f} "
        f"f1@0.3={overall_report['by_theta']['0.3']['f1']:.4f} "
        f"f1@0.5={overall_report['by_theta']['0.5']['f1']:.4f} "
        f"events={len(nested_predictions)}"
    )
    for theta, spread in outer_fold_metric_spread.items():
        print(
            f"outer_fold_f1@{theta} "
            f"mean={spread['mean']:.4f} std={spread['std_population']:.4f} "
            f"min={spread['min']:.4f} max={spread['max']:.4f}"
        )
    print(f"artifact_dir={output_dir.relative_to(project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
