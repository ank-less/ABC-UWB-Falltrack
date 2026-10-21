"""Train-stream out-of-fold experiments for the local UWB package."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .quantum_features import extract_stream_features, feature_names


@dataclass(frozen=True)
class TrainEvent:
    stream_id: str
    t_start: float
    t_end: float
    ignore: bool


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_train_events(path: Path, train_ids: set[str]) -> list[TrainEvent]:
    """Read only train annotation fields from the package's mixed split CSV."""
    events: list[TrainEvent] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stream_id", "t_start", "t_end"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"annotation file is missing columns: {sorted(missing)}")
        for row in reader:
            stream_id = (row.get("stream_id") or "").strip()
            if stream_id not in train_ids:
                # Do not inspect validation annotation fields.
                continue
            flagged = str(row.get("quality", "")).strip().lower() == "low"
            flagged = flagged or _truthy(row.get("needs_review"))
            ignore = _truthy(row.get("ignore")) or flagged
            start = float(row["t_start"])
            end = float(row["t_end"])
            if not (math.isfinite(start) and math.isfinite(end) and start < end):
                raise ValueError(f"invalid train event interval in {stream_id}")
            events.append(TrainEvent(stream_id, start, end, ignore))
    if not events:
        raise ValueError("no train events were read from the annotation file")
    return events


def make_targets(
    centers_s: np.ndarray,
    events: list[TrainEvent],
    *,
    ignore_margin_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build frame labels while masking flagged/ignored zones for training."""
    labels = np.zeros(centers_s.shape[0], dtype=np.uint8)
    usable = np.ones(centers_s.shape[0], dtype=bool)
    for event in events:
        if event.ignore:
            in_zone = (centers_s >= event.t_start - ignore_margin_s) & (
                centers_s <= event.t_end + ignore_margin_s
            )
            usable[in_zone] = False
    for event in events:
        if event.ignore:
            continue
        positive = (centers_s >= event.t_start) & (centers_s <= event.t_end)
        labels[positive] = 1
        # A valid event remains a positive even if it touches a neighbouring
        # ignore zone; this mirrors evaluator matching order.
        usable[positive] = True
    return labels, usable


def make_boundary_targets(
    centers_s: np.ndarray,
    events: list[TrainEvent],
    *,
    band_s: float = 0.1,
    ignore_margin_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal onset/offset labels from valid train events only.

    Onset positives occupy the first ``band_s`` after an event starts; offset
    positives occupy the final ``band_s`` before it ends. Ignored-event zones
    remain masked, except valid boundary positives which take precedence just
    as valid event positives do in :func:`make_targets`.
    """
    if not math.isfinite(band_s) or band_s <= 0.0:
        raise ValueError("boundary target band_s must be finite and positive")
    onset = np.zeros(centers_s.shape[0], dtype=np.uint8)
    offset = np.zeros(centers_s.shape[0], dtype=np.uint8)
    usable = np.ones(centers_s.shape[0], dtype=bool)
    for event in events:
        if event.ignore:
            in_zone = (centers_s >= event.t_start - ignore_margin_s) & (
                centers_s <= event.t_end + ignore_margin_s
            )
            usable[in_zone] = False
    for event in events:
        if event.ignore:
            continue
        onset_positive = (centers_s >= event.t_start) & (
            centers_s < event.t_start + band_s
        )
        offset_positive = (centers_s > event.t_end - band_s) & (
            centers_s <= event.t_end
        )
        onset[onset_positive] = 1
        offset[offset_positive] = 1
        usable[onset_positive | offset_positive] = True
    return onset, offset, usable


def _cache_signature(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    static_tau_s = config.get("static_background_tau_s")
    include_fidelity_contrast = bool(config.get("include_fidelity_contrast", False))
    include_normalized_rho_similarity = bool(
        config.get("include_normalized_rho_similarity", False)
    )
    include_quantum_reservoir = bool(config.get("include_quantum_reservoir", False))
    feature_version = (
        "q5-causal-static-hybrid-v1" if static_tau_s is not None else "q4-causal-diff-v2"
    )
    if include_fidelity_contrast:
        feature_version += "+fidelity-contrast-v1"
    if include_normalized_rho_similarity:
        feature_version += "+normalized-rho-temporal-v1"
    if include_quantum_reservoir:
        feature_version += "+quantum-reservoir-channel-v1"
    signature = {
        "feature_version": feature_version,
        "include_fidelity_contrast": include_fidelity_contrast,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "rate_hz": float(config["rate_hz"]),
        "pool_frames": int(config["pool_frames"]),
        "chunk_frames": int(config["chunk_frames"]),
        "static_background_tau_s": float(static_tau_s) if static_tau_s is not None else None,
        "static_motion_gamma": float(config.get("static_motion_gamma", 1.0)),
    }
    if include_normalized_rho_similarity:
        signature["include_normalized_rho_similarity"] = True
    if include_quantum_reservoir:
        signature.update(
            {
                "include_quantum_reservoir": True,
                "quantum_reservoir_memory_tau_s": float(
                    config.get("quantum_reservoir_memory_tau_s", 0.8)
                ),
                "quantum_reservoir_phase_scale": float(
                    config.get("quantum_reservoir_phase_scale", np.pi / 2.0)
                ),
                "quantum_reservoir_entangler_angle": float(
                    config.get("quantum_reservoir_entangler_angle", np.pi / 8.0)
                ),
            }
        )
    return signature


def extract_train_streams(
    data_root: Path,
    cache_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Extract/cache features from train HDF5 files only."""
    train_dir = data_root / "streams" / "train"
    paths = sorted(train_dir.glob("stream_*.h5"))
    if not paths:
        raise FileNotFoundError(f"no train streams found under {train_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Any]] = {}
    include_static_state = config.get("static_background_tau_s") is not None
    include_fidelity_contrast = bool(config.get("include_fidelity_contrast", False))
    include_normalized_rho_similarity = bool(
        config.get("include_normalized_rho_similarity", False)
    )
    include_quantum_reservoir = bool(config.get("include_quantum_reservoir", False))
    names = feature_names(
        include_static_state=include_static_state,
        include_fidelity_contrast=include_fidelity_contrast,
        include_normalized_rho_similarity=include_normalized_rho_similarity,
        include_quantum_reservoir=include_quantum_reservoir,
    )
    for path in paths:
        index = path.stem.split("_")[-1]
        stream_id = f"train_{int(index):02d}"
        cache_path = cache_dir / f"{stream_id}.npz"
        signature = _cache_signature(path, config)
        loaded = False
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                metadata = json.loads(str(cached["metadata"].item()))
                if metadata == signature:
                    features = cached["features"].astype(np.float32, copy=False)
                    centers = cached["centers_s"].astype(np.float64, copy=False)
                    duration_s = float(cached["duration_s"].item())
                    loaded = True
        if not loaded:
            features, centers, duration_s = extract_stream_features(
                path,
                rate_hz=float(config["rate_hz"]),
                pool_frames=int(config["pool_frames"]),
                frames_per_chunk=int(config["chunk_frames"]),
                static_background_tau_s=(
                    float(config["static_background_tau_s"])
                    if include_static_state
                    else None
                ),
                static_motion_gamma=float(config.get("static_motion_gamma", 1.0)),
                include_fidelity_contrast=include_fidelity_contrast,
                include_normalized_rho_similarity=include_normalized_rho_similarity,
                include_quantum_reservoir=include_quantum_reservoir,
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
            np.savez_compressed(
                cache_path,
                features=features,
                centers_s=centers,
                duration_s=np.asarray(duration_s, dtype=np.float64),
                metadata=np.asarray(json.dumps(signature, sort_keys=True)),
            )
        if features.shape[1] != len(names):
            raise ValueError(f"feature-name count does not match feature shape for {stream_id}")
        result[stream_id] = {
            "features": features,
            "centers_s": centers,
            "duration_s": duration_s,
            "source": str(path),
        }
    return result, names


def assert_train_stream_ids(stream_ids: list[str], *, scope: str) -> None:
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError(f"{scope} contains duplicate stream IDs")
    invalid = [stream_id for stream_id in stream_ids if not re.fullmatch(r"train_\d+", stream_id)]
    if invalid:
        raise ValueError(f"{scope} contains non-train stream IDs: {invalid[:5]}")


def assert_fold_partition(
    parent_ids: list[str], folds: list[list[str]], *, scope: str
) -> None:
    """Require nonempty, disjoint folds that cover exactly the train parent scope."""
    assert_train_stream_ids(parent_ids, scope=f"{scope} parent")
    parent = set(parent_ids)
    seen: set[str] = set()
    for fold_index, fold in enumerate(folds):
        assert_train_stream_ids(fold, scope=f"{scope} fold {fold_index}")
        if not fold:
            raise ValueError(f"{scope} fold {fold_index} is empty")
        unexpected = set(fold).difference(parent)
        if unexpected:
            raise ValueError(f"{scope} fold {fold_index} contains IDs outside its parent: {sorted(unexpected)}")
        overlap = seen.intersection(fold)
        if overlap:
            raise ValueError(f"{scope} folds overlap: {sorted(overlap)}")
        seen.update(fold)
    if seen != parent:
        missing = sorted(parent.difference(seen))
        extra = sorted(seen.difference(parent))
        raise ValueError(f"{scope} do not cover their parent exactly; missing={missing}, extra={extra}")


def assert_train_heldout_partition(
    parent_ids: list[str],
    train_ids: list[str],
    held_out_ids: list[str],
    *,
    scope: str,
) -> None:
    """Require a train/held-out pair to partition exactly the parent stream scope."""
    assert_train_stream_ids(parent_ids, scope=f"{scope} parent")
    assert_train_stream_ids(train_ids, scope=f"{scope} train")
    assert_train_stream_ids(held_out_ids, scope=f"{scope} held-out")
    train = set(train_ids)
    held_out = set(held_out_ids)
    parent = set(parent_ids)
    if train.intersection(held_out):
        raise ValueError(f"{scope} train and held-out IDs overlap")
    if train.union(held_out) != parent:
        raise ValueError(f"{scope} train and held-out IDs do not partition the parent exactly")


def make_group_folds(stream_ids: list[str], n_splits: int, seed: int) -> list[list[str]]:
    assert_train_stream_ids(stream_ids, scope="group-fold input")
    if n_splits < 2 or n_splits > len(stream_ids):
        raise ValueError("n_splits must be between 2 and the number of train streams")
    shuffled = list(stream_ids)
    random.Random(seed).shuffle(shuffled)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for index, stream_id in enumerate(shuffled):
        folds[index % n_splits].append(stream_id)
    assert_fold_partition(stream_ids, folds, scope="group folds")
    return folds


def _fit_readout(
    features_by_id: dict[str, dict[str, Any]],
    labels_by_id: dict[str, np.ndarray],
    usable_by_id: dict[str, np.ndarray],
    train_ids: list[str],
    model_config: dict[str, Any],
) -> Any:
    x_parts = [features_by_id[key]["features"][usable_by_id[key]] for key in train_ids]
    y_parts = [labels_by_id[key][usable_by_id[key]] for key in train_ids]
    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    if not np.isfinite(x_train).all():
        raise ValueError("non-finite feature found in train fold")
    if np.unique(y_train).size != 2:
        raise ValueError("a train fold must contain both positive and negative frames")
    model_kind = str(model_config.get("kind", "logistic"))
    if model_kind == "logistic":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "readout",
                    LogisticRegression(
                        C=float(model_config["C"]),
                        class_weight="balanced",
                        max_iter=int(model_config["max_iter"]),
                        solver="lbfgs",
                        random_state=int(model_config["seed"]),
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        return model
    if model_kind == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=float(model_config["learning_rate"]),
            max_iter=int(model_config["max_iter"]),
            max_leaf_nodes=int(model_config["max_leaf_nodes"]),
            min_samples_leaf=int(model_config["min_samples_leaf"]),
            l2_regularization=float(model_config["l2_regularization"]),
            max_bins=int(model_config["max_bins"]),
            early_stopping=False,
            random_state=int(model_config["seed"]),
        )
        positive_weight = float(model_config["positive_weight"])
        stream_weight_exponent = float(model_config.get("stream_weight_exponent", 0.0))
        if not 0.0 <= stream_weight_exponent <= 1.0:
            raise ValueError("stream_weight_exponent must be between 0 and 1")
        if stream_weight_exponent == 0.0:
            sample_weight = np.where(y_train == 1, positive_weight, 1.0)
        else:
            sample_weight_parts = []
            total_samples = len(y_train)
            for stream_id in train_ids:
                usable = usable_by_id[stream_id]
                stream_labels = labels_by_id[stream_id][usable]
                stream_samples = len(stream_labels)
                if stream_samples == 0:
                    raise ValueError(f"train stream {stream_id} has no usable frames")
                stream_factor = (
                    total_samples / (len(train_ids) * stream_samples)
                ) ** stream_weight_exponent
                sample_weight_parts.append(
                    stream_factor * np.where(stream_labels == 1, positive_weight, 1.0)
                )
            sample_weight = np.concatenate(sample_weight_parts).astype(np.float64, copy=False)
            sample_weight /= float(np.mean(sample_weight))
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return model
    raise ValueError(f"unsupported readout kind: {model_kind}")


def _fit_boundary_readouts(
    features_by_id: dict[str, dict[str, Any]],
    onset_targets_by_id: dict[str, np.ndarray],
    offset_targets_by_id: dict[str, np.ndarray],
    usable_by_id: dict[str, np.ndarray],
    fit_stream_ids: list[str],
    model_config: dict[str, Any],
) -> tuple[Any, Any]:
    """Fit onset and offset heads on exactly the supplied train-stream scope."""
    assert_train_stream_ids(fit_stream_ids, scope="boundary-head fit")
    expected = set(fit_stream_ids)
    for name, values in (
        ("features", features_by_id),
        ("onset targets", onset_targets_by_id),
        ("offset targets", offset_targets_by_id),
        ("usable masks", usable_by_id),
    ):
        if set(values) != expected:
            raise ValueError(f"boundary-head {name} must cover exactly the fit scope")
    onset_config = dict(model_config)
    offset_config = dict(model_config)
    offset_config["seed"] = int(offset_config.get("seed", 0)) + 1
    onset_model = _fit_readout(
        features_by_id,
        onset_targets_by_id,
        usable_by_id,
        fit_stream_ids,
        onset_config,
    )
    offset_model = _fit_readout(
        features_by_id,
        offset_targets_by_id,
        usable_by_id,
        fit_stream_ids,
        offset_config,
    )
    return onset_model, offset_model


def _decision_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=np.float64)
    probability = np.clip(model.predict_proba(features)[:, 1], 1e-8, 1.0 - 1e-8)
    return np.log(probability / (1.0 - probability))


def _online_ising(
    logits: np.ndarray,
    bias: float,
    transition_cost: float,
    *,
    entry_cost: float | None = None,
    exit_cost: float | None = None,
) -> np.ndarray:
    """Causal forward update with optional asymmetric entry/exit penalties."""
    if logits.size == 0:
        return np.zeros(0, dtype=bool)
    entry_penalty = float(transition_cost if entry_cost is None else entry_cost)
    exit_penalty = float(transition_cost if exit_cost is None else exit_cost)
    field = np.clip(logits.astype(np.float64, copy=False) - bias, -50.0, 50.0)
    state = np.zeros(field.size, dtype=bool)
    score_bg = 0.0
    score_event = float(field[0])
    state[0] = score_event > score_bg
    for index in range(1, field.size):
        next_bg = max(score_bg, score_event - exit_penalty)
        next_event = field[index] + max(score_event, score_bg - entry_penalty)
        score_bg, score_event = next_bg, next_event
        state[index] = score_event > score_bg
    return state


def _events_from_state(
    stream_id: str,
    state: np.ndarray,
    logits: np.ndarray,
    duration_s: float,
    step_s: float,
    min_duration_s: float,
    *,
    start_offset_s: float = 0.0,
    end_offset_s: float = 0.0,
    duration_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    active = np.flatnonzero(state)
    if active.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(active) > 1)
    starts = np.concatenate([[active[0]], active[cuts + 1]])
    stops = np.concatenate([active[cuts], [active[-1]]]) + 1
    minimum = max(1, int(math.ceil(min_duration_s / step_s)))
    events: list[dict[str, Any]] = []
    for start, stop in zip(starts, stops):
        if stop - start < minimum:
            continue
        raw_start = float(start * step_s)
        raw_end = min(float(stop * step_s), duration_s)
        event_start_offset = float(
            duration_profile.get("global_start_offset_s", start_offset_s)
            if duration_profile
            else start_offset_s
        )
        event_end_offset = float(
            duration_profile.get("global_end_offset_s", end_offset_s)
            if duration_profile
            else end_offset_s
        )
        duration_region: str | None = None
        predicted_duration = raw_end - raw_start
        if duration_profile and duration_profile.get("thresholds_s") is not None:
            thresholds = [float(value) for value in duration_profile["thresholds_s"]]
            if len(thresholds) >= 2:
                if predicted_duration <= thresholds[0]:
                    duration_region = "short"
                elif predicted_duration <= thresholds[1]:
                    duration_region = "middle"
                else:
                    duration_region = "long"
            elif thresholds:
                duration_region = "short" if predicted_duration <= thresholds[0] else "long"
        elif duration_profile and duration_profile.get("threshold_s") is not None:
            duration_region = (
                "short"
                if predicted_duration <= float(duration_profile["threshold_s"])
                else "long"
            )
        if duration_profile and duration_region is not None:
            event_start_offset = float(
                duration_profile.get(f"{duration_region}_start_offset_s", event_start_offset)
            )
            event_end_offset = float(
                duration_profile.get(f"{duration_region}_end_offset_s", event_end_offset)
            )
        t_start = min(max(raw_start + event_start_offset, 0.0), duration_s)
        t_end = min(max(raw_end + event_end_offset, 0.0), duration_s)
        if t_end <= t_start:
            continue
        score = float(np.max(logits[int(start) : int(stop)]))
        alarm = min(raw_start + min_duration_s, raw_end)
        events.append(
            {
                "stream_id": stream_id,
                "t_start": t_start,
                "t_end": t_end,
                "score": score,
                "t_alarm": alarm,
            }
        )
    return events


def refine_event_boundaries(
    predictions: list[dict[str, Any]],
    onset_scores_by_id: dict[str, np.ndarray],
    offset_scores_by_id: dict[str, np.ndarray],
    centers_by_id: dict[str, np.ndarray],
    *,
    radius_s: float,
    band_s: float,
    step_s: float,
    min_duration_s: float,
) -> list[dict[str, Any]]:
    """Causally move starts later and ends earlier using local head scores.

    The onset search is bounded by ``t_alarm`` so it cannot inspect samples
    after the alarm time. The offset search uses only samples at or before the
    current proposed end. ``t_alarm`` itself is never altered.
    """
    if radius_s < 0.0 or band_s <= 0.0 or step_s <= 0.0:
        raise ValueError("boundary refinement time values must be valid")
    stream_ids = {str(row["stream_id"]) for row in predictions}
    for name, values in (
        ("onset scores", onset_scores_by_id),
        ("offset scores", offset_scores_by_id),
        ("centers", centers_by_id),
    ):
        if not stream_ids.issubset(values):
            raise ValueError(f"boundary refinement is missing {name} for a predicted stream")
    if radius_s == 0.0 or not predictions:
        return [dict(row) for row in predictions]

    refined: list[dict[str, Any]] = []
    half_band = band_s / 2.0
    half_step = step_s / 2.0

    def best_time(candidates: np.ndarray, scores: np.ndarray, reference: float) -> float | None:
        if candidates.size == 0:
            return None
        best_score = float(np.max(scores))
        best_positions = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-10, atol=1e-12))
        best_position = min(
            best_positions.tolist(),
            key=lambda position: (abs(float(candidates[position]) - reference), position),
        )
        return float(candidates[best_position])

    for row in predictions:
        updated = dict(row)
        stream_id = str(row["stream_id"])
        centers = np.asarray(centers_by_id[stream_id], dtype=np.float64)
        onset_scores = np.asarray(onset_scores_by_id[stream_id], dtype=np.float64)
        offset_scores = np.asarray(offset_scores_by_id[stream_id], dtype=np.float64)
        if not (centers.size == onset_scores.size == offset_scores.size):
            raise ValueError(f"boundary score length mismatch for {stream_id}")

        base_start = float(row["t_start"])
        base_end = float(row["t_end"])
        alarm = float(row["t_alarm"])
        onset_times = centers - half_band
        onset_mask = (
            (onset_times >= base_start)
            & (onset_times <= min(base_start + radius_s, alarm))
            & (centers <= alarm + 1e-9)
        )
        refined_start = best_time(onset_times[onset_mask], onset_scores[onset_mask], base_start)

        offset_times = centers + half_band
        offset_mask = (
            (offset_times >= max(base_end - radius_s, base_start + min_duration_s))
            & (offset_times <= base_end)
            & (centers <= base_end + 1e-9)
        )
        refined_end = best_time(offset_times[offset_mask], offset_scores[offset_mask], base_end)
        if refined_start is not None:
            updated["t_start"] = refined_start
        if refined_end is not None:
            updated["t_end"] = refined_end
        if (
            updated["t_end"] <= updated["t_start"]
            or updated["t_end"] - updated["t_start"] < min_duration_s
        ):
            updated["t_start"] = base_start
            updated["t_end"] = base_end
        refined.append(updated)
    return refined


def contract_event_intervals(
    predictions: list[dict[str, Any]],
    *,
    scale: float,
    min_duration_s: float,
) -> list[dict[str, Any]]:
    """Contract event intervals symmetrically without changing alarm times.

    ``scale`` is restricted to (0, 1], so boundaries remain within their
    already-predicted interval and do not require future extrapolation.
    Intervals shorter than the selected minimum duration are left unchanged.
    """
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("interval contraction scale must be finite and in (0, 1]")
    if not math.isfinite(min_duration_s) or min_duration_s <= 0.0:
        raise ValueError("minimum event duration must be finite and positive")
    contracted: list[dict[str, Any]] = []
    for row in predictions:
        updated = dict(row)
        start = float(row["t_start"])
        end = float(row["t_end"])
        duration = end - start
        if duration <= 0.0:
            raise ValueError("cannot contract a non-positive event interval")
        if duration > min_duration_s and scale < 1.0:
            new_duration = max(min_duration_s, duration * scale)
            center = (start + end) / 2.0
            updated["t_start"] = center - new_duration / 2.0
            updated["t_end"] = center + new_duration / 2.0
        contracted.append(updated)
    return contracted


def fit_event_duration_prior(
    events_by_id: dict[str, list[TrainEvent]],
    fit_stream_ids: list[str],
    *,
    minimum_events: int = 8,
) -> dict[str, Any]:
    """Fit a robust log-duration prior from the explicitly supplied train scope."""
    assert_train_stream_ids(fit_stream_ids, scope="duration-prior fit")
    if not set(fit_stream_ids).issubset(events_by_id):
        raise ValueError("duration-prior fit IDs are outside the provided event scope")
    durations = np.asarray(
        [
            event.t_end - event.t_start
            for stream_id in fit_stream_ids
            for event in events_by_id[stream_id]
            if not event.ignore
        ],
        dtype=np.float64,
    )
    if durations.size < minimum_events:
        return {
            "enabled": False,
            "reason": "too_few_valid_fit_events",
            "n_valid_events": int(durations.size),
            "minimum_events": int(minimum_events),
            "fit_stream_ids": sorted(fit_stream_ids),
            "mu_log_s": None,
            "sigma_log_s": None,
        }

    log_durations = np.log(np.maximum(durations, 1e-3))
    q25, median, q75 = np.quantile(log_durations, [0.25, 0.5, 0.75])
    sigma = max(float((q75 - q25) / (2.0 * 0.6744897501960817)), 0.25)
    return {
        "enabled": True,
        "reason": "robust_log_duration_fit",
        "n_valid_events": int(durations.size),
        "minimum_events": int(minimum_events),
        "fit_stream_ids": sorted(fit_stream_ids),
        "mu_log_s": float(median),
        "sigma_log_s": sigma,
    }


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _causal_semimarkov_filter(
    logits: np.ndarray,
    bias: float,
    entry_cost: float,
    exit_cost: float,
    duration_prior: dict[str, Any],
    step_s: float,
    *,
    duration_penalty: float,
    max_duration_s: float,
    threshold_on: float,
    threshold_off_ratio: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-filter event age states using only logits through the current frame.

    The log-space transition potentials are equivalent to a semi-Markov chain:
    event entry pays ``entry_cost``, event frames receive the classifier logit
    margin, and an age-dependent hazard comes from a duration distribution fit
    on the current training scope. ``duration_penalty`` tempers that empirical
    distribution from uniform (0) to the fitted prior (1). The returned mask
    uses causal posterior hysteresis.
    """
    if not 0.0 < threshold_on < 1.0:
        raise ValueError("threshold_on must be strictly between zero and one")
    if not 0.0 < threshold_off_ratio <= 1.0:
        raise ValueError("threshold_off_ratio must be in (0, 1]")
    if logits.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    if step_s <= 0.0 or max_duration_s < step_s:
        raise ValueError("semi-Markov time step and duration cap must be positive")

    age_count = max(1, int(math.floor(max_duration_s / step_s)))
    ages_s = np.arange(1, age_count + 1, dtype=np.float64) * step_s
    if duration_prior.get("enabled"):
        mu = float(duration_prior["mu_log_s"])
        sigma = max(float(duration_prior["sigma_log_s"]), 0.25)
        log_duration_energy = (
            0.5 * ((np.log(ages_s) - mu) / sigma) ** 2 + np.log(ages_s)
        )
        log_duration_energy -= float(np.min(log_duration_energy))
        log_duration_mass = -float(duration_penalty) * log_duration_energy
        log_duration_mass -= _logsumexp(log_duration_mass)
        duration_mass = np.exp(log_duration_mass)
    else:
        duration_mass = np.full(age_count, 1.0 / age_count, dtype=np.float64)
    survival_mass = np.cumsum(duration_mass[::-1])[::-1]
    hazard = np.divide(
        duration_mass,
        survival_mass,
        out=np.ones_like(duration_mass),
        where=survival_mass > 0.0,
    )
    hazard = np.clip(hazard, 1e-12, 1.0)
    log_hazard = np.log(hazard)
    log_survival = np.log1p(-np.minimum(hazard, 1.0 - 1e-12))
    log_survival[-1] = -np.inf

    log_background = 0.0
    log_event_age = np.full(age_count, -np.inf, dtype=np.float64)
    field = np.clip(logits.astype(np.float64, copy=False) - float(bias), -50.0, 50.0)
    posterior = np.zeros(field.size, dtype=np.float64)
    for index, frame_score in enumerate(field):
        next_background = _logsumexp(
            np.concatenate(
                ([log_background], log_event_age + log_hazard - float(exit_cost))
            )
        )
        next_event_age = np.full(age_count, -np.inf, dtype=np.float64)
        next_event_age[0] = log_background - float(entry_cost) + float(frame_score)
        if age_count > 1:
            next_event_age[1:] = (
                log_event_age[:-1] + log_survival[:-1] + float(frame_score)
            )
        normalizer = _logsumexp(np.concatenate(([next_background], next_event_age)))
        log_background = next_background - normalizer
        log_event_age = next_event_age - normalizer
        posterior[index] = math.exp(_logsumexp(log_event_age))

    state = np.zeros(field.size, dtype=bool)
    active = False
    threshold_off = threshold_on * threshold_off_ratio
    for index, probability in enumerate(posterior):
        if active:
            if probability <= threshold_off:
                active = False
        elif probability >= threshold_on:
            active = True
        state[index] = active
    return state, posterior


def _events_from_causal_state(
    stream_id: str,
    state: np.ndarray,
    posterior: np.ndarray,
    logits: np.ndarray,
    duration_s: float,
    step_s: float,
    min_duration_s: float,
    *,
    start_offset_s: float = 0.0,
    end_offset_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Turn the causal filtered state into intervals; alarm stays at threshold crossing."""
    if state.size != logits.size or posterior.size != logits.size:
        raise ValueError("causal state, posterior, and logits must have matching lengths")
    active = np.flatnonzero(state)
    if active.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(active) > 1)
    starts = np.concatenate([[active[0]], active[cuts + 1]])
    stops = np.concatenate([active[cuts], [active[-1]]]) + 1
    minimum = max(1, int(math.ceil(min_duration_s / step_s)))
    events: list[dict[str, Any]] = []
    for start, stop in zip(starts, stops):
        if stop - start < minimum:
            continue
        raw_start = float(start * step_s)
        raw_end = min(float(stop * step_s), duration_s)
        t_start = min(max(raw_start + start_offset_s, 0.0), duration_s)
        t_end = min(max(raw_end + end_offset_s, 0.0), duration_s)
        if t_end <= t_start:
            continue
        events.append(
            {
                "stream_id": stream_id,
                "t_start": t_start,
                "t_end": t_end,
                "score": float(np.max(logits[int(start) : int(stop)])),
                # The alarm is the first causal posterior threshold crossing;
                # the reported start may be corrected slightly backward.
                "t_alarm": raw_start,
            }
        )
    return events


def _package_evaluator(data_root: Path):
    import sys

    source_dir = data_root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from abc2027.evaluate import evaluate

    return evaluate


def run_oof(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    project_root = Path(config["project_root"]).resolve()
    data_root = (project_root / config["data_root"]).resolve()
    output_dir = (project_root / config["output_dir"]).resolve()
    cache_dir = (project_root / config["feature_cache_dir"]).resolve() if config.get("feature_cache_dir") else output_dir / "feature_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    features_by_id, names = extract_train_streams(data_root, cache_dir, config)
    stream_ids = sorted(features_by_id)
    train_events = read_train_events(data_root / "streams_events.csv", set(stream_ids))
    events_by_id: dict[str, list[TrainEvent]] = {stream_id: [] for stream_id in stream_ids}
    for event in train_events:
        events_by_id[event.stream_id].append(event)

    labels_by_id: dict[str, np.ndarray] = {}
    usable_by_id: dict[str, np.ndarray] = {}
    gt_records = []
    for stream_id in stream_ids:
        feature_pack = features_by_id[stream_id]
        labels, usable = make_targets(feature_pack["centers_s"], events_by_id[stream_id])
        labels_by_id[stream_id] = labels
        usable_by_id[stream_id] = usable
        for event in events_by_id[stream_id]:
            gt_records.append(
                {
                    "stream_id": event.stream_id,
                    "t_start": event.t_start,
                    "t_end": event.t_end,
                    "ignore": event.ignore,
                }
            )

    cv_config = config["cross_validation"]
    folds = make_group_folds(stream_ids, int(cv_config["n_splits"]), int(cv_config["seed"]))
    oof_logits: dict[str, np.ndarray] = {}
    fold_records = []
    model_config = dict(config["readout"])
    model_config["seed"] = int(cv_config["seed"])
    for fold_index, held_out in enumerate(folds):
        fit_ids = [stream_id for stream_id in stream_ids if stream_id not in set(held_out)]
        model = _fit_readout(features_by_id, labels_by_id, usable_by_id, fit_ids, model_config)
        for stream_id in held_out:
            oof_logits[stream_id] = _decision_scores(model, features_by_id[stream_id]["features"])
        fold_records.append({"fold": fold_index, "train_streams": fit_ids, "held_out_streams": held_out})
        estimator = model[-1] if isinstance(model, Pipeline) else model
        iteration_count = getattr(estimator, "n_iter_", None)
        if isinstance(iteration_count, np.ndarray):
            iteration_count = int(iteration_count.max())
        print(
            f"fold={fold_index + 1}/{len(folds)} train_streams={len(fit_ids)} "
            f"oof_streams={len(held_out)} iterations={iteration_count}"
        )

    step_s = int(config["pool_frames"]) / float(config["rate_hz"])
    logits_all = np.concatenate([oof_logits[stream_id] for stream_id in stream_ids])
    quantiles = [float(value) for value in config["decoder_search"]["logit_quantiles"]]
    biases = sorted(set(float(value) for value in np.quantile(logits_all, quantiles)))
    transition_costs = [float(value) for value in config["decoder_search"]["transition_costs"]]
    exit_costs = (
        [float(value) for value in config["decoder_search"]["exit_costs"]]
        if "exit_costs" in config["decoder_search"]
        else None
    )
    min_durations = [float(value) for value in config["decoder_search"]["min_duration_s"]]
    start_offsets = [float(value) for value in config["decoder_search"].get("start_offsets_s", [0.0])]
    end_offsets = [float(value) for value in config["decoder_search"].get("end_offsets_s", [0.0])]
    duration_profiles = config["decoder_search"].get("duration_profiles") or [{"name": "uniform"}]

    package_evaluate = _package_evaluator(data_root)
    durations = {key: float(features_by_id[key]["duration_s"]) for key in stream_ids}
    candidates: list[dict[str, Any]] = []
    for bias in biases:
        for transition_cost in transition_costs:
            candidate_exit_costs = exit_costs or [transition_cost]
            for exit_cost in candidate_exit_costs:
                states = {
                    stream_id: _online_ising(
                        oof_logits[stream_id],
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
                                        oof_logits[stream_id],
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
                                predictions = []
                                for stream_id in stream_ids:
                                    predictions.extend(
                                        _events_from_state(
                                            stream_id,
                                            states[stream_id],
                                            oof_logits[stream_id],
                                            durations[stream_id],
                                            step_s,
                                            min_duration_s,
                                            start_offset_s=start_offset_s,
                                            end_offset_s=end_offset_s,
                                            duration_profile=duration_profile,
                                        )
                                    )
                                report = package_evaluate(gt_records, predictions, durations, theta=0.3)
                                score = report["ranking_score"]
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
                                        "n_predictions": len(predictions),
                                        "ranking_score": score,
                                        "primary": report["primary"],
                                        "by_theta": report["by_theta"],
                                    }
                                )

    def ranking_key(row: dict[str, Any]) -> tuple[float, float, float]:
        ranking = row["ranking_score"]
        start_error = ranking["start_error_median_abs"]
        false_alarms = ranking["false_alarms_per_hour"]
        return (
            float(ranking["f1_mean"]),
            -float(start_error) if start_error is not None else -float("inf"),
            -float(false_alarms) if false_alarms is not None else -float("inf"),
        )

    candidates.sort(key=ranking_key, reverse=True)
    best = candidates[0]
    result = {
        "experiment": config["experiment_name"],
        "data_status": "local_package_only; package evaluator is not yet confirmed as official ranking protocol",
        "requested_internal_target_mean_event_f1": float(config["requested_internal_target_mean_event_f1"]),
        "validation_used": False,
        "composition_file_used": False,
        "ignore_flagged": True,
        "feature_names": names,
        "train_streams": stream_ids,
        "folds": fold_records,
        "train_events": len(train_events),
        "train_ignored_events": sum(event.ignore for event in train_events),
        "readout": config["readout"],
        "decoder_search": {
            "logit_quantiles": quantiles,
            "bias_values": biases,
            "transition_costs": transition_costs,
            "exit_costs": exit_costs,
            "min_duration_s": min_durations,
            "start_offsets_s": start_offsets,
            "end_offsets_s": end_offsets,
            "duration_profiles": duration_profiles,
        },
        "best_oof": best,
        "top_candidates": candidates[:20],
    }

    np.savez_compressed(
        output_dir / "oof_logits_train_only.npz",
        **{key: value for key, value in oof_logits.items()},
    )
    (output_dir / "oof_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    public_config = {key: value for key, value in config.items() if key not in {"project_root", "config_path"}}
    (output_dir / "config_used.json").write_text(json.dumps(public_config, indent=2), encoding="utf-8")
    (output_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(
        "best_oof "
        f"mean_event_f1={best['ranking_score']['f1_mean']:.4f} "
        f"f1@0.1={best['by_theta']['0.1']['f1']:.4f} "
        f"f1@0.3={best['by_theta']['0.3']['f1']:.4f} "
        f"f1@0.5={best['by_theta']['0.5']['f1']:.4f} "
        f"events={best['n_predictions']}"
    )
    print(f"artifact_dir={output_dir.relative_to(project_root)}")
    return result


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["project_root"] = str(path.resolve().parents[1])
    config["config_path"] = str(path.resolve())
    return config
