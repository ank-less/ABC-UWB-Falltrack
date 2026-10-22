#!/usr/bin/env python3
"""Run a fixed 4-qubit VQC baseline on source-disjoint train-only OOF folds.

The statevector simulator uses NumPy on the local CPU. It is a variational
quantum circuit model, but this experiment does not use physical QPU hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc_uwb_falltrack.oof import (  # noqa: E402
    _events_from_state,
    assert_fold_partition,
    assert_train_heldout_partition,
    extract_train_streams,
    load_config,
    make_group_folds,
    make_targets,
    read_train_events,
)


N_QUBITS = 4
N_STATES = 1 << N_QUBITS
_BASIS = np.arange(N_STATES, dtype=np.int64)
_Z_TABLE = np.stack(
    [1.0 - 2.0 * ((_BASIS >> wire) & 1) for wire in range(N_QUBITS)], axis=0
)


def _apply_ry(state: np.ndarray, wire: int, angle: float | np.ndarray) -> None:
    mask = 1 << wire
    lower = _BASIS[(_BASIS & mask) == 0]
    upper = lower | mask
    theta = np.asarray(angle, dtype=np.float64)
    if theta.ndim == 0:
        cosine = np.full(state.shape[0], np.cos(float(theta) / 2.0))
        sine = np.full(state.shape[0], np.sin(float(theta) / 2.0))
    else:
        cosine = np.cos(theta / 2.0)
        sine = np.sin(theta / 2.0)
    left = state[:, lower].copy()
    right = state[:, upper].copy()
    state[:, lower] = cosine[:, None] * left - sine[:, None] * right
    state[:, upper] = sine[:, None] * left + cosine[:, None] * right


def _apply_rz(state: np.ndarray, wire: int, angle: float) -> None:
    mask = 1 << wire
    lower = _BASIS[(_BASIS & mask) == 0]
    upper = lower | mask
    state[:, lower] *= np.exp(-0.5j * angle)
    state[:, upper] *= np.exp(0.5j * angle)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> None:
    control_mask = 1 << control
    target_mask = 1 << target
    lower = _BASIS[
        ((_BASIS & control_mask) != 0) & ((_BASIS & target_mask) == 0)
    ]
    upper = lower | target_mask
    left = state[:, lower].copy()
    state[:, lower] = state[:, upper]
    state[:, upper] = left


def _mean_z_expectation(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Exact analytic expectation for a batched 4-qubit VQC statevector."""
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_QUBITS:
        raise ValueError(f"VQC inputs must have shape (n, {N_QUBITS})")
    if not np.isfinite(values).all() or not np.isfinite(parameters).all():
        raise ValueError("VQC inputs and parameters must be finite")
    if values.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    state = np.zeros((values.shape[0], N_STATES), dtype=np.complex128)
    state[:, 0] = 1.0
    for wire in range(N_QUBITS):
        _apply_ry(state, wire, values[:, wire])

    for layer in range(parameters.shape[0]):
        for wire in range(N_QUBITS):
            _apply_ry(state, wire, float(parameters[layer, wire, 0]))
            _apply_rz(state, wire, float(parameters[layer, wire, 1]))
        for wire in range(N_QUBITS):
            _apply_cnot(state, wire, (wire + 1) % N_QUBITS)

    probabilities = np.abs(state) ** 2
    expectations = probabilities @ _Z_TABLE.T
    return expectations.mean(axis=1)


def _predict_probabilities(inputs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    mean_z = _mean_z_expectation(inputs, parameters)
    return np.clip((1.0 - mean_z) / 2.0, 1e-6, 1.0 - 1e-6)


def _balanced_log_loss(
    inputs: np.ndarray, labels: np.ndarray, parameters: np.ndarray
) -> float:
    probabilities = _predict_probabilities(inputs, parameters)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        raise ValueError("VQC training sample must contain both classes")
    positive_loss = -np.log(probabilities[positive]).mean()
    negative_loss = -np.log1p(-probabilities[negative]).mean()
    return float(0.5 * (positive_loss + negative_loss))


def _fit_vqc(
    inputs: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit with fixed-step SPSA using analytic statevector expectations."""
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("VQC training sample must contain both classes")
    rng = np.random.default_rng(seed)
    layers = int(config["layers"])
    parameters = rng.normal(0.0, float(config["init_scale"]), size=(layers, N_QUBITS, 2))
    initial_loss = _balanced_log_loss(inputs, labels, parameters)

    a = float(config["spsa_a"])
    c = float(config["spsa_c"])
    alpha = float(config["spsa_alpha"])
    gamma = float(config["spsa_gamma"])
    stability = float(config["spsa_A"])
    for step in range(1, int(config["spsa_steps"]) + 1):
        perturbation = rng.choice(np.asarray([-1.0, 1.0]), size=parameters.shape)
        c_step = c / (step**gamma)
        loss_plus = _balanced_log_loss(inputs, labels, parameters + c_step * perturbation)
        loss_minus = _balanced_log_loss(inputs, labels, parameters - c_step * perturbation)
        gradient = ((loss_plus - loss_minus) / (2.0 * c_step)) * perturbation
        a_step = a / ((stability + step) ** alpha)
        parameters -= a_step * gradient
        # Rotation angles are periodic; keeping them bounded avoids needless
        # numerical growth without selecting on held-out performance.
        parameters = (parameters + np.pi) % (2.0 * np.pi) - np.pi

    final_loss = _balanced_log_loss(inputs, labels, parameters)
    return parameters, {"initial_balanced_log_loss": initial_loss, "final_balanced_log_loss": final_loss}


def _sample_training_examples(
    fit_ids: list[str],
    features_by_id: dict[str, dict[str, Any]],
    events_by_id: dict[str, list[Any]],
    max_per_class_per_group: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    class_counts = {"negative": 0, "positive": 0}
    for stream_id in fit_ids:
        pack = features_by_id[stream_id]
        labels, usable = make_targets(pack["centers_s"], events_by_id[stream_id])
        raw = pack["features"][:, :48]
        for label, name in ((0, "negative"), (1, "positive")):
            indices = np.flatnonzero(usable & (labels == label))
            if indices.size > max_per_class_per_group:
                indices = rng.choice(indices, size=max_per_class_per_group, replace=False)
            if indices.size:
                indices.sort()
                x_parts.append(raw[indices].astype(np.float64, copy=False))
                y_parts.append(np.full(indices.size, label, dtype=np.uint8))
                class_counts[name] += int(indices.size)
    if not x_parts or class_counts["positive"] == 0 or class_counts["negative"] == 0:
        raise ValueError("fold training scope did not produce both VQC classes")
    permutation = rng.permutation(sum(len(part) for part in y_parts))
    x_train = np.concatenate(x_parts, axis=0)[permutation]
    y_train = np.concatenate(y_parts, axis=0)[permutation]
    return x_train, y_train, class_counts


def _fit_input_map(
    training_features: np.ndarray,
) -> tuple[StandardScaler, PCA, StandardScaler, np.ndarray]:
    raw_scaler = StandardScaler().fit(training_features)
    standardized = raw_scaler.transform(training_features)
    pca = PCA(n_components=N_QUBITS, svd_solver="full", random_state=2028).fit(standardized)
    components = pca.transform(standardized)
    component_scaler = StandardScaler().fit(components)
    angles = _to_angles(training_features, raw_scaler, pca, component_scaler)
    return raw_scaler, pca, component_scaler, angles


def _to_angles(
    raw_features: np.ndarray,
    raw_scaler: StandardScaler,
    pca: PCA,
    component_scaler: StandardScaler,
) -> np.ndarray:
    standardized = raw_scaler.transform(raw_features)
    components = pca.transform(standardized)
    scaled_components = component_scaler.transform(components)
    return np.clip(scaled_components, -2.0, 2.0) * (np.pi / 2.0)


def _predict_stream(
    raw_features: np.ndarray,
    raw_scaler: StandardScaler,
    pca: PCA,
    component_scaler: StandardScaler,
    parameters: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    probabilities = np.empty(raw_features.shape[0], dtype=np.float64)
    for start in range(0, raw_features.shape[0], batch_size):
        stop = min(start + batch_size, raw_features.shape[0])
        angles = _to_angles(raw_features[start:stop], raw_scaler, pca, component_scaler)
        probabilities[start:stop] = _predict_probabilities(angles, parameters)
    return probabilities


def _close_short_gaps(state: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0 or not state.any():
        return state
    output = state.copy()
    inactive = np.flatnonzero(~output)
    if inactive.size == 0:
        return output
    cuts = np.flatnonzero(np.diff(inactive) > 1)
    starts = np.concatenate([[inactive[0]], inactive[cuts + 1]])
    stops = np.concatenate([inactive[cuts], [inactive[-1]]]) + 1
    for start, stop in zip(starts, stops):
        if start > 0 and stop < output.size and stop - start <= max_gap_frames:
            output[start:stop] = True
    return output


def _group_train_events(
    events: list[Any], stream_ids: list[str]
) -> dict[str, list[Any]]:
    grouped = {stream_id: [] for stream_id in stream_ids}
    for event in events:
        if event.stream_id in grouped:
            grouped[event.stream_id].append(event)
    return grouped


def run(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"]).resolve()
    data_root = (project_root / config["data_root"]).resolve()
    output_dir = (project_root / config["output_dir"]).resolve()
    cache_dir = (project_root / config["feature_cache_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_root / "source_group_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stream_to_source = {row["stream_id"]: row["source_group_hash"] for row in manifest}
    if len(stream_to_source) != len(manifest) or len(set(stream_to_source.values())) != len(manifest):
        raise ValueError("source-group manifest must map one unique source hash to each stream")

    print("extracting train-only features", flush=True)
    features_by_id, feature_names = extract_train_streams(data_root, cache_dir, config)
    stream_ids = sorted(features_by_id)
    expected_motion_names = [
        f"motion_log_bin_r{radar}_{bin_index}"
        for radar in range(3)
        for bin_index in range(16)
    ]
    if feature_names[:48] != expected_motion_names:
        raise ValueError("first 48 features do not match the configured raw motion-bin inputs")
    if set(stream_ids) != set(stream_to_source):
        raise ValueError("feature streams do not exactly match the source-group manifest")

    cv = config["cross_validation"]
    folds = make_group_folds(stream_ids, int(cv["n_splits"]), int(cv["seed"]))
    assert_fold_partition(stream_ids, folds, scope="source-group outer folds")
    durations = {stream_id: float(features_by_id[stream_id]["duration_s"]) for stream_id in stream_ids}
    package_src = data_root / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    from abc2027.evaluate import evaluate as package_evaluate  # noqa: E402
    from abc2027.evaluate import load_ground_truth  # noqa: E402

    vqc_config = config["vqc"]
    decoder_config = config["decoder"]
    if int(vqc_config["n_qubits"]) != N_QUBITS:
        raise ValueError(f"this baseline implementation requires n_qubits={N_QUBITS}")
    step_s = int(config["pool_frames"]) / float(config["rate_hz"])
    max_gap_frames = int(round(float(decoder_config["merge_gap_s"]) / step_s))
    threshold = float(decoder_config["probability_threshold"])
    oof_probabilities: dict[str, np.ndarray] = {}
    predictions: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    fold_parameters: list[np.ndarray] = []
    fold_raw_means: list[np.ndarray] = []
    fold_raw_scales: list[np.ndarray] = []
    fold_pca_components: list[np.ndarray] = []
    fold_component_means: list[np.ndarray] = []
    fold_component_scales: list[np.ndarray] = []
    fold_explained_variance: list[np.ndarray] = []

    for fold_index, held_out in enumerate(folds):
        held_out_set = set(held_out)
        fit_ids = [stream_id for stream_id in stream_ids if stream_id not in held_out_set]
        assert_train_heldout_partition(
            stream_ids, fit_ids, held_out, scope=f"source-group fold {fold_index}"
        )
        fit_sources = {stream_to_source[stream_id] for stream_id in fit_ids}
        held_sources = {stream_to_source[stream_id] for stream_id in held_out}
        if fit_sources.intersection(held_sources):
            raise ValueError(f"source hashes overlap in fold {fold_index}")

        # Read only the current fold's fit labels. Held-out annotations are
        # first passed to the official package evaluator after predictions lock.
        train_events = read_train_events(data_root / "streams_events.csv", set(fit_ids))
        events_by_id = _group_train_events(train_events, fit_ids)
        sample_x, sample_y, sampled_counts = _sample_training_examples(
            fit_ids,
            features_by_id,
            events_by_id,
            int(vqc_config["max_examples_per_class_per_group"]),
            int(cv["seed"]) + fold_index,
        )
        raw_scaler, pca, component_scaler, train_angles = _fit_input_map(sample_x)
        parameters, losses = _fit_vqc(
            train_angles,
            sample_y,
            vqc_config,
            int(cv["seed"]) + fold_index,
        )
        fold_parameters.append(parameters)
        fold_raw_means.append(raw_scaler.mean_)
        fold_raw_scales.append(raw_scaler.scale_)
        fold_pca_components.append(pca.components_)
        fold_component_means.append(component_scaler.mean_)
        fold_component_scales.append(component_scaler.scale_)
        fold_explained_variance.append(pca.explained_variance_ratio_)

        for stream_id in held_out:
            raw = features_by_id[stream_id]["features"][:, :48]
            probabilities = _predict_stream(
                raw,
                raw_scaler,
                pca,
                component_scaler,
                parameters,
                int(vqc_config["prediction_batch_size"]),
            )
            oof_probabilities[stream_id] = probabilities
            logits = np.log(probabilities / (1.0 - probabilities))
            state = _close_short_gaps(probabilities >= threshold, max_gap_frames)
            stream_predictions = _events_from_state(
                stream_id,
                state,
                logits,
                durations[stream_id],
                step_s,
                float(decoder_config["min_duration_s"]),
            )
            for prediction in stream_predictions:
                prediction["t_alarm"] = None
            predictions.extend(stream_predictions)

        fold_record = {
            "fold": fold_index,
            "fit_source_groups": len(fit_ids),
            "held_out_source_groups": len(held_out),
            "source_hash_overlap": 0,
            "sampled_training_examples": int(sample_y.size),
            "sampled_positive_frames": sampled_counts["positive"],
            "sampled_negative_frames": sampled_counts["negative"],
            **losses,
        }
        fold_records.append(fold_record)
        print(
            f"fold={fold_index + 1}/{len(folds)} groups={len(held_out)} "
            f"samples={sample_y.size} loss={losses['initial_balanced_log_loss']:.4f}"
            f"->{losses['final_balanced_log_loss']:.4f} predictions={len(predictions)}",
            flush=True,
        )

    if set(oof_probabilities) != set(stream_ids):
        raise ValueError("OOF VQC probabilities do not cover exactly all source groups")

    # Only now, after every prediction is locked, load held-out train labels
    # for the package-local event metric.
    ground_truth = load_ground_truth(data_root / "streams_events.csv", ignore_flagged=True)
    report = package_evaluate(ground_truth, predictions, durations, theta=0.3)
    fold_reports = []
    for fold_index, held_out in enumerate(folds):
        held_set = set(held_out)
        fold_gt = [event for event in ground_truth if event["stream_id"] in held_set]
        fold_pred = [event for event in predictions if event["stream_id"] in held_set]
        fold_durations = {stream_id: durations[stream_id] for stream_id in held_out}
        fold_report = package_evaluate(fold_gt, fold_pred, fold_durations, theta=0.3)
        fold_reports.append(
            {
                "fold": fold_index,
                "held_out_source_groups": len(held_out),
                "ranking_score": fold_report["ranking_score"],
                "by_theta": fold_report["by_theta"],
            }
        )

    result = {
        "experiment": config["experiment_name"],
        "model": {
            "kind": "4-qubit variational quantum classifier",
            "simulator": "local NumPy exact statevector; analytic expectation values",
            "hardware_qpu_used": False,
            "feature_map": "train-fold StandardScaler + PCA(4), clipped standardized PCs mapped to RY angles",
            "ansatz": "2 layers of per-qubit RY/RZ rotations and a 4-qubit directed CNOT ring",
            "readout": "mean Pauli-Z expectation across the four qubits, mapped to P(fall)",
            "optimizer": "fixed-step SPSA using balanced binary cross-entropy",
            "classical_input_features": feature_names[:48],
        },
        "data_scope": "467 pseudo-streams formed from unique hashed source-file groups in published train HDF5 only",
        "validation_used": False,
        "test_used": False,
        "held_out_annotations_read_during_prediction": False,
        "source_hash_overlap_across_each_fold": 0,
        "source_group_count": len(stream_ids),
        "train_event_count": len(ground_truth),
        "ignore_flagged": True,
        "metric_status": "package-local metric; official challenge ranking metric remains unconfirmed",
        "threshold_selection": "fixed P(fall) >= 0.5; no threshold or decoder tuning on OOF labels",
        "decoder": {
            "probability_threshold": threshold,
            "merge_gap_s": float(decoder_config["merge_gap_s"]),
            "min_duration_s": float(decoder_config["min_duration_s"]),
        },
        "cross_validation": {
            "method": "5-fold shuffled group partition; one unique source-file hash per pseudo-stream",
            "seed": int(cv["seed"]),
            "folds": fold_records,
        },
        "overall": report,
        "fold_metrics": fold_reports,
        "internal_target": {
            "min_f1_each_tiou": 0.75,
            "max_tiou_spread": 0.10,
            "reached": (
                min(float(report["by_theta"][key]["f1"]) for key in ("0.1", "0.3", "0.5"))
                >= 0.75
                and max(float(report["by_theta"][key]["f1"]) for key in ("0.1", "0.3", "0.5"))
                - min(float(report["by_theta"][key]["f1"]) for key in ("0.1", "0.3", "0.5"))
                <= 0.10
            ),
        },
    }

    prediction_path = output_dir / "vqc_oof_predictions_train_only.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["stream_id", "t_start", "t_end", "score"]
        )
        writer.writeheader()
        writer.writerows(
            {key: prediction[key] for key in ("stream_id", "t_start", "t_end", "score")}
            for prediction in predictions
        )
    np.savez_compressed(
        output_dir / "vqc_oof_probabilities_train_only.npz",
        **oof_probabilities,
    )
    np.savez_compressed(
        output_dir / "vqc_fold_models.npz",
        parameters=np.stack(fold_parameters),
        raw_mean=np.stack(fold_raw_means),
        raw_scale=np.stack(fold_raw_scales),
        pca_components=np.stack(fold_pca_components),
        component_mean=np.stack(fold_component_means),
        component_scale=np.stack(fold_component_scales),
        explained_variance_ratio=np.stack(fold_explained_variance),
    )
    (output_dir / "vqc_oof_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    public_config = {
        key: value for key, value in config.items() if key not in {"project_root", "config_path"}
    }
    (output_dir / "config_used.json").write_text(
        json.dumps(public_config, indent=2), encoding="utf-8"
    )
    print(
        "oof_result "
        f"mean_f1={report['ranking_score']['f1_mean']:.4f} "
        f"f1@0.1={report['by_theta']['0.1']['f1']:.4f} "
        f"f1@0.3={report['by_theta']['0.3']['f1']:.4f} "
        f"f1@0.5={report['by_theta']['0.5']['f1']:.4f} "
        f"output={prediction_path.relative_to(project_root)}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vqc_source_group_baseline_r30.yaml",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(load_config(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
