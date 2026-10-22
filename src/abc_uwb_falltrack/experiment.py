"""End-to-end baseline run: data checks, grouped CV, tuning on train, holdout report.

Stage order inside ``run_train``: manifest -> validation -> official split + leakage
checks -> per-stream preprocessing/windowing/features -> grouped out-of-fold scores
-> threshold/post-processing tuned on those scores -> final fit on all of train.
``run_evaluate`` then scores the holdout split once, with everything frozen.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .config import get, resolved_copy
from .data_io import read_timestamps
from .features import window_features
from .manifest import build_manifest, save_manifest
from .metrics import (
    event_report,
    ground_truth_records,
    load_official_evaluator,
    pr_curve_table,
    prediction_records,
    ranking_f1,
    recall_by_group,
    window_metrics,
)
from .modeling import fit_model, param_grid, predict_scores
from .postprocessing import PostprocessParams, events_for_streams, postprocess_grid
from .preprocessing import cached_frame_series
from .splitting import (
    assign_group_folds,
    check_fold_assignment,
    official_assignment,
    purged_training_mask,
    split_assignment_table,
    window_fold_membership,
)
from .validation import (
    LeakageError,
    ValidationReport,
    check_disjoint,
    check_group_leakage,
    validate_events,
    validate_manifest,
    validate_segments,
)
from .windowing import NEGATIVE, POSITIVE, label_windows, make_windows, window_spec

LOG = logging.getLogger("abc_uwb_falltrack")
PACKAGES = ("numpy", "scipy", "scikit-learn", "h5py", "pandas", "PyYAML", "joblib")


# ----------------------------------------------------------------------------- run setup


class WarningCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.levelname} {record.name}: {record.getMessage()}")


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> WarningCollector:
    root = logging.getLogger("abc_uwb_falltrack")
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    collector = WarningCollector()
    root.addHandler(collector)
    return collector


def create_run_dir(output_root: str | Path, name: str) -> Path:
    """A new, empty run folder. Existing runs are never overwritten."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{stamp}_{name}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{stamp}_{name}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def environment_info() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git": git_state(),
    }


def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=20, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def write_json(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialise {type(value)}")


# ----------------------------------------------------------------------------- data


@dataclass
class Dataset:
    manifest: pd.DataFrame
    segments: pd.DataFrame
    events: pd.DataFrame
    durations: dict[str, float]
    frame_limits: dict[str, int]
    report: ValidationReport


def prepare_dataset(cfg: dict) -> Dataset:
    """Manifest + validation, then the optional smoke-run limits."""
    manifest, segments, events = build_manifest(cfg)
    report = ValidationReport()
    validate_manifest(manifest, cfg, report)
    validate_events(events, manifest, report)
    validate_segments(segments, manifest, report)
    report.raise_if_failed(bool(get(cfg, "validation.strict", True)))

    max_streams = get(cfg, "experiment.max_streams_per_split")
    if max_streams:
        manifest = (
            manifest.sort_values(["official_split", "stream_index"])
            .groupby("official_split", group_keys=False)
            .head(int(max_streams))
            .reset_index(drop=True)
        )
        LOG.warning("limited to %d stream(s) per split (smoke run)", int(max_streams))
    rate = float(get(cfg, "data.rate_hz"))
    max_seconds = get(cfg, "experiment.max_seconds_per_stream")
    frame_limits = {}
    for row in manifest.itertuples(index=False):
        limit = row.n_frames if not max_seconds else min(row.n_frames, int(round(float(max_seconds) * rate)))
        frame_limits[row.stream_id] = int(limit)
    if max_seconds:
        LOG.warning("streams truncated to %.0f s (smoke run)", float(max_seconds))
    durations = {stream: frames / rate for stream, frames in frame_limits.items()}
    segments = segments[segments["stream_id"].isin(durations)].copy()
    segments = segments[segments["t_start"] < segments["stream_id"].map(durations)]
    segments["t_end"] = np.minimum(segments["t_end"], segments["stream_id"].map(durations))
    report.details["run_streams"] = sorted(durations)
    report.details["run_hours"] = sum(durations.values()) / 3600.0
    return Dataset(manifest, segments.reset_index(drop=True), events, durations, frame_limits, report)


@dataclass
class StreamFeatures:
    stream_id: str
    windows: pd.DataFrame
    features: np.ndarray
    stats: dict


def extract_stream(cfg: dict, row, events: pd.DataFrame, stop: int, duration: float) -> tuple[StreamFeatures, list[str]]:
    rate = float(get(cfg, "data.rate_hz"))
    spec = window_spec(cfg)
    series = tuple(get(cfg, "features.series"))
    cache_dir = get(cfg, "experiment.cache_dir")
    values, valid, stats = cached_frame_series(
        Path(row.file_path),
        rate,
        float(get(cfg, "preprocessing.background_tau_seconds")),
        int(get(cfg, "data.chunk_frames")),
        stop,
        series,
        Path(cache_dir) / "frame_series" if cache_dir else None,
    )
    if get(cfg, "validation.check_missing", True) and stats["nonfinite_values"]:
        LOG.warning("%s: %d non-finite CIR values replaced by 0", row.stream_id, stats["nonfinite_values"])
    timestamps = read_timestamps(Path(row.file_path), stop)
    windows = make_windows(row.stream_id, timestamps, valid, spec)
    windows = windows[windows["keep"]].drop(columns="keep").reset_index(drop=True)
    matrix, names = window_features(
        values,
        windows["f_start"].to_numpy(),
        spec.frames,
        list(series),
        list(get(cfg, "features.enabled_features")),
        rate,
        log_transform=bool(get(cfg, "features.log_transform", True)),
        cross_radar=bool(get(cfg, "features.cross_radar", True)),
        context_seconds=get(cfg, "features.context_seconds"),
    )
    stream_events = events[(events["stream_id"] == row.stream_id) & (events["t_end"] <= duration + 1e-9)]
    labels, coverage = label_windows(
        windows["t_start"].to_numpy(),
        windows["t_end"].to_numpy(),
        stream_events,
        float(get(cfg, "windowing.positive_overlap_threshold")),
        float(get(cfg, "windowing.negative_overlap_max", 0.0)),
        str(get(cfg, "windowing.flagged_event_policy", "positive")),
    )
    windows["label"] = labels
    windows["coverage"] = coverage
    windows["split"] = row.official_split
    stats.update({"windows": int(len(windows)), "positive": int((labels == POSITIVE).sum()), "ambiguous": int((labels < 0).sum())})
    LOG.info(
        "%s: %d windows (%d positive, %d ambiguous), cache %s",
        row.stream_id,
        len(windows),
        stats["positive"],
        stats["ambiguous"],
        stats.get("cache", "off"),
    )
    return StreamFeatures(row.stream_id, windows, matrix, stats), names


def extract_split(cfg: dict, data: Dataset, split: str) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    rows = data.manifest[data.manifest["official_split"] == split]
    if rows.empty:
        raise LeakageError(f"no streams in split {split!r}")
    parts, names, stats = [], None, {}
    for row in rows.itertuples(index=False):
        part, part_names = extract_stream(cfg, row, data.events, data.frame_limits[row.stream_id], data.durations[row.stream_id])
        if names is not None and part_names != names:
            raise AssertionError("feature names differ between streams")
        names = part_names
        parts.append(part)
        stats[row.stream_id] = part.stats
    windows = pd.concat([p.windows for p in parts], ignore_index=True)
    features = np.vstack([p.features for p in parts])
    return windows, features, names, stats


# ----------------------------------------------------------------------------- folds


def attach_folds(windows: pd.DataFrame, segments: pd.DataFrame, group_folds: dict[str, int], n_folds: int):
    """Majority group/fold per window and which folds each window touches."""
    groups = sorted(group_folds)
    group_index = {g: i for i, g in enumerate(groups)}
    to_fold = np.zeros((len(groups), n_folds), dtype=bool)
    for g, i in group_index.items():
        to_fold[i, group_folds[g]] = True
    major_group = np.full(len(windows), -1, dtype=np.int64)
    touches_group = np.zeros((len(windows), len(groups)), dtype=bool)
    for stream_id, positions in windows.groupby("stream_id").indices.items():
        seg = segments[segments["stream_id"] == stream_id]
        part = windows.iloc[positions]
        major, touches = window_fold_membership(
            part["t_start"].to_numpy(),
            part["t_end"].to_numpy(),
            seg["t_start"].to_numpy(float),
            seg["t_end"].to_numpy(float),
            seg["subject_group"].map(group_index).to_numpy(),
            len(groups),
        )
        major_group[positions] = major
        touches_group[positions] = touches
    touches_fold = (touches_group.astype(np.int64) @ to_fold.astype(np.int64)) > 0
    fold = np.where(major_group >= 0, np.array([group_folds[groups[i]] for i in np.clip(major_group, 0, None)]), -1)
    return fold.astype(np.int64), touches_fold, major_group, touches_group, groups


def cross_validate(
    name: str,
    params: dict,
    features: np.ndarray,
    labels: np.ndarray,
    fold: np.ndarray,
    touches: np.ndarray,
    n_folds: int,
    seed: int,
    class_weight,
) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold scores. Fold k's model never trains on a window touching fold k."""
    labelled = labels >= 0
    oof = np.full(len(labels), np.nan)
    rows = []
    models = {}
    for k in range(n_folds):
        fit_rows = labelled & (fold >= 0) & purged_training_mask(touches, k)
        if np.any(touches[fit_rows, k]):
            raise LeakageError(f"fold {k}: a training window touches held-out subjects")
        test_rows = fold == k
        if not test_rows.any() or not fit_rows.any():
            LOG.warning("%s fold %d: %d training and %d held-out windows; fold skipped", name, k, fit_rows.sum(), test_rows.sum())
            continue
        model = fit_model(name, params, features[fit_rows], labels[fit_rows], seed, class_weight)
        models[k] = model
        oof[test_rows] = predict_scores(model, features[test_rows])
        held = test_rows & labelled
        fit_scores = predict_scores(model, features[fit_rows])
        held_metrics = window_metrics(labels[held], oof[held], 0.5)
        fit_metrics = window_metrics(labels[fit_rows], fit_scores, 0.5)
        rows.append(
            {
                "model": name,
                "params": json.dumps(params, sort_keys=True),
                "fold": k,
                "train_windows": int(fit_rows.sum()),
                "heldout_windows": int(held.sum()),
                "heldout_positives": held_metrics["positives"],
                "train_pr_auc": fit_metrics["pr_auc"],
                "heldout_pr_auc": held_metrics["pr_auc"],
                "heldout_roc_auc": held_metrics["roc_auc"],
                "heldout_f1_at_0.5": held_metrics["f1"],
                "warnings": "; ".join(held_metrics["warnings"]),
            }
        )
    leftover = np.isnan(oof)
    if leftover.any():
        if not models:
            raise RuntimeError(f"{name}: no fold could be trained")
        first = min(models)
        oof[leftover] = predict_scores(models[first], features[leftover])
        LOG.warning("%s: %d windows outside every fold were scored by the fold-%d model", name, leftover.sum(), first)
    return oof, rows


def tune_postprocessing(
    windows: pd.DataFrame, scores: np.ndarray, grid: list[PostprocessParams], durations: dict, ground: list[dict], evaluator
) -> pd.DataFrame:
    rows = []
    for params in grid:
        events = events_for_streams(windows, scores, params, durations)
        score = ranking_f1(ground, prediction_records(events), durations, evaluator)
        rows.append({**params.to_dict(), "f1_mean": score, "n_events": int(len(events))})
    table = pd.DataFrame(rows)
    return table.sort_values(["f1_mean", "n_events"], ascending=[False, True]).reset_index(drop=True)


def params_from_row(row: pd.Series) -> PostprocessParams:
    return PostprocessParams(
        threshold=float(row["threshold"]),
        smoothing_windows=int(row["smoothing_windows"]),
        minimum_positive_windows=int(row["minimum_positive_windows"]),
        merge_gap_seconds=float(row["merge_gap_seconds"]),
        minimum_event_duration_seconds=float(row["minimum_event_duration_seconds"]),
        cooldown_seconds=float(row["cooldown_seconds"]),
        interval_mode=str(row["interval_mode"]),
    )


def learning_curve(
    name, params, features, labels, fold, touches_group, group_fold_array, n_folds, fractions, seed, class_weight
) -> list[dict]:
    """Held-out window PR-AUC when training on a fraction of the training subjects."""
    rng = np.random.default_rng(seed)
    labelled = labels >= 0
    rows = []
    for k in range(n_folds):
        train_groups = np.flatnonzero(group_fold_array != k)
        held = (fold == k) & labelled
        if not held.any() or len(np.unique(labels[held])) < 2:
            continue
        for fraction in fractions:
            count = max(1, int(round(fraction * len(train_groups))))
            chosen = np.zeros(len(group_fold_array), dtype=bool)
            chosen[rng.choice(train_groups, size=count, replace=False)] = True
            allowed = touches_group.any(axis=1) & ~np.any(touches_group & ~chosen, axis=1)
            fit_rows = labelled & allowed
            if fit_rows.sum() == 0:
                continue
            model = fit_model(name, params, features[fit_rows], labels[fit_rows], seed, class_weight)
            held_m = window_metrics(labels[held], predict_scores(model, features[held]), 0.5)
            fit_m = window_metrics(labels[fit_rows], predict_scores(model, features[fit_rows]), 0.5)
            rows.append(
                {
                    "model": name,
                    "fold": k,
                    "group_fraction": fraction,
                    "train_groups": count,
                    "train_windows": int(fit_rows.sum()),
                    "train_pr_auc": fit_m["pr_auc"],
                    "heldout_pr_auc": held_m["pr_auc"],
                }
            )
    return rows


def junction_diagnostic(
    predictions: pd.DataFrame, segments: pd.DataFrame, events: pd.DataFrame, durations: dict, margin: float, after: float = 0.0
) -> dict:
    """Do predicted events away from falls sit on segment joins more often than chance?

    Streams are built by concatenation; a detector that fires on the joins would be
    learning an artefact of the generator instead of falls. A join "zone" spans
    ``[join - margin, join + after]``: the background filter keeps a decaying residual
    for a few seconds after a join, so ``after`` is typically about tau.
    """
    joins = {s: g["t_start"].to_numpy(float)[1:] for s, g in segments.sort_values("t_start").groupby("stream_id")}
    falls = {s: g[["t_start", "t_end"]].to_numpy(float) for s, g in events.groupby("stream_id")}
    away = near = 0
    for row in predictions.itertuples(index=False):
        fall = falls.get(row.stream_id, np.zeros((0, 2)))
        if np.any((fall[:, 0] - margin < row.t_end) & (fall[:, 1] + margin > row.t_start)):
            continue
        away += 1
        stream_joins = joins.get(row.stream_id, np.zeros(0))
        if np.any((stream_joins - margin < row.t_end) & (stream_joins + after > row.t_start)):
            near += 1
    covered = 0.0
    for stream, duration in durations.items():
        mask = np.zeros(int(round(duration * 10)) + 1, dtype=bool)  # 0.1 s grid
        for join in joins.get(stream, []):
            mask[max(0, int((join - margin) * 10)) : int((join + after) * 10) + 1] = True
        covered += mask.mean() * duration
    base_rate = covered / sum(durations.values()) if durations else None
    return {
        "events_away_from_falls": away,
        "of_which_near_a_join": near,
        "fraction_near_join": near / away if away else None,
        "time_fraction_near_joins": base_rate,
        "zone_s": [-margin, after],
        "note": "fraction_near_join much larger than time_fraction_near_joins suggests the model reacts to segment joins",
    }


# ----------------------------------------------------------------------------- stages


def _model_candidates(cfg: dict) -> list[tuple[str, list[dict]]]:
    return [(c["name"], param_grid(c.get("parameters"))) for c in get(cfg, "model.candidates")]


def _save_predictions(path: Path, windows: pd.DataFrame, scores: dict[str, np.ndarray], extra: list[str]) -> None:
    columns = ["stream_id", "w_index", "t_start", "t_end", "label", "coverage", *extra]
    table = windows[columns].copy()
    for name, values in scores.items():
        table[f"score_{name}"] = values
    table.to_csv(path, index=False, float_format="%.6g", compression="gzip")


def run_train(cfg: dict, run_dir: Path) -> dict:
    started = time.time()
    seed = int(get(cfg, "experiment.seed"))
    n_folds = int(get(cfg, "split.n_folds"))
    class_weight = get(cfg, "model.class_weight", "balanced")
    holdout = str(get(cfg, "split.holdout_split", "val"))

    data = prepare_dataset(cfg)
    write_json(run_dir / "data_validation.json", data.report.to_dict())
    data.manifest.to_csv(run_dir / "manifest_snapshot.csv", index=False)
    data.segments.to_csv(run_dir / "segments_snapshot.csv", index=False)

    # Official split first; folds are built before any window exists.
    stream_split = official_assignment(data.manifest)
    if "train" not in stream_split.values():
        raise LeakageError("no official train streams")
    check_disjoint(
        {split: {s for s, v in stream_split.items() if v == split} for split in set(stream_split.values())}, "streams"
    )
    group_split = check_group_leakage(data.segments, stream_split)
    train_groups = sorted(g for g, s in group_split.items() if s == "train")
    group_folds = assign_group_folds(train_groups, n_folds, int(get(cfg, "split.seed")))
    check_fold_assignment(group_folds, n_folds)
    assignments = split_assignment_table(stream_split, group_split, group_folds, int(get(cfg, "split.seed")), str(get(cfg, "split.group_key")))
    assignments.to_csv(run_dir / "split_assignments.csv", index=False)
    LOG.info("split: %s; %d train subject groups in %d folds", pd.Series(stream_split).value_counts().to_dict(), len(train_groups), n_folds)

    windows, features, names, stream_stats = extract_split(cfg, data, "train")
    if set(windows["split"]) != {"train"}:
        raise LeakageError("non-train windows entered the training set")
    train_segments = data.segments[data.segments["official_split"] == "train"]
    fold, touches, major_group, touches_group, group_order = attach_folds(windows, train_segments, group_folds, n_folds)
    windows["fold"] = fold
    windows["subject_group"] = [group_order[i] if i >= 0 else "" for i in major_group]
    labels = windows["label"].to_numpy()
    write_json(
        run_dir / "feature_schema.json",
        {
            "n_features": len(names),
            "names": names,
            "series": get(cfg, "features.series"),
            "stats": get(cfg, "features.enabled_features"),
            "window_frames": window_spec(cfg).frames,
            "stride_frames": window_spec(cfg).step,
            "rate_hz": float(get(cfg, "data.rate_hz")),
            "per_stream": stream_stats,
        },
    )
    class_balance = {
        "positive": int((labels == POSITIVE).sum()),
        "negative": int((labels == NEGATIVE).sum()),
        "ambiguous": int((labels < 0).sum()),
        "by_fold": {int(k): {"windows": int((fold == k).sum()), "positive": int(((fold == k) & (labels == POSITIVE)).sum())} for k in range(-1, n_folds)},
    }
    LOG.info("train windows: %s", class_balance)

    durations = {s: d for s, d in data.durations.items() if stream_split[s] == "train"}
    ignore_flagged = bool(get(cfg, "evaluation.ignore_flagged", True))
    evaluator = load_official_evaluator(get(cfg, "evaluation.official_evaluator"))
    ground = ground_truth_records(data.events, durations, ignore_flagged)
    grid = postprocess_grid(cfg)

    fold_rows, tuning_tables, oof_scores, selection = [], [], {}, {}
    for name, combos in _model_candidates(cfg):
        best = None
        for params in combos:
            oof, rows = cross_validate(name, params, features, labels, fold, touches, n_folds, seed, class_weight)
            fold_rows += rows
            table = tune_postprocessing(windows, oof, grid, durations, ground, evaluator)
            table.insert(0, "model", name)
            table.insert(1, "model_params", json.dumps(params, sort_keys=True))
            tuning_tables.append(table)
            top = table.iloc[0]
            LOG.info("%s %s: out-of-fold f1_mean %.4f", name, params, top["f1_mean"])
            if best is None or top["f1_mean"] > best["f1_mean"]:
                best = {"params": params, "f1_mean": float(top["f1_mean"]), "post": params_from_row(top), "oof": oof}
        oof_scores[name] = best["oof"]
        selection[name] = {
            "model_params": best["params"],
            "postprocessing": best["post"].to_dict(),
            "oof_f1_mean": best["f1_mean"],
        }
    ranked = sorted(selection, key=lambda n: selection[n]["oof_f1_mean"], reverse=True)
    selected = ranked[0]
    LOG.info("selected model (out-of-fold f1_mean on train): %s", selected)

    tuning = pd.concat(tuning_tables, ignore_index=True)
    tuning.to_csv(run_dir / "postprocess_grid_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)

    # Sensitivity around the chosen operating point: vary the threshold only.
    sensitivity = []
    for name, choice in selection.items():
        post = choice["postprocessing"]
        mask = (tuning["model"] == name) & (tuning["model_params"] == json.dumps(choice["model_params"], sort_keys=True))
        for key in ("smoothing_windows", "minimum_positive_windows", "merge_gap_seconds", "minimum_event_duration_seconds", "cooldown_seconds"):
            mask &= tuning[key] == post[key]
        sensitivity.append(tuning[mask].sort_values("threshold"))
    pd.concat(sensitivity, ignore_index=True).to_csv(run_dir / "threshold_sensitivity.csv", index=False)

    labelled = labels >= 0
    oof_summary, final_models, train_scores = {}, {}, {}
    model_dir = run_dir / "models"
    model_dir.mkdir()
    for name, choice in selection.items():
        post = PostprocessParams(**choice["postprocessing"])
        oof = oof_scores[name]
        pr_curve_table(labels[labelled], oof[labelled]).to_csv(run_dir / f"pr_curve_oof_{name}.csv", index=False)
        oof_events = events_for_streams(windows, oof, post, durations)
        oof_events.to_csv(run_dir / f"event_predictions_train_oof_{name}.csv", index=False, float_format="%.3f")
        oof_report = event_report(ground, prediction_records(oof_events), durations, evaluator)
        # Final model: all labelled train windows, nothing else.
        if set(windows.loc[labelled, "split"]) != {"train"}:
            raise LeakageError("final fit would use non-train windows")
        model = fit_model(name, choice["model_params"], features[labelled], labels[labelled], seed, class_weight)
        joblib.dump(model, model_dir / f"{name}.joblib")
        final_models[name] = model
        train_scores[name] = predict_scores(model, features)
        insample_events = events_for_streams(windows, train_scores[name], post, durations)
        insample_report = event_report(ground, prediction_records(insample_events), durations, evaluator)
        model_folds = [r for r in fold_rows if r["model"] == name and r["params"] == json.dumps(choice["model_params"], sort_keys=True)]
        heldout_pr = [r["heldout_pr_auc"] for r in model_folds if r["heldout_pr_auc"] is not None]
        oof_summary[name] = {
            "window_oof": window_metrics(labels[labelled], oof[labelled], post.threshold),
            "window_train_insample": window_metrics(labels[labelled], train_scores[name][labelled], post.threshold),
            "fold_pr_auc_mean": float(np.mean(heldout_pr)) if heldout_pr else None,
            "fold_pr_auc_std": float(np.std(heldout_pr)) if heldout_pr else None,
            "event_oof": oof_report,
            "event_train_insample": insample_report,
        }

    _save_predictions(
        run_dir / "window_predictions_train.csv.gz",
        windows,
        {**{f"{n}_oof": s for n, s in oof_scores.items()}, **{f"{n}_insample": s for n, s in train_scores.items()}},
        ["fold", "subject_group"],
    )

    curve_fractions = get(cfg, "diagnostics.learning_curve_fractions") or []
    curve_rows = []
    if curve_fractions:
        group_fold_array = np.array([group_folds[g] for g in group_order])
        curve_rows = learning_curve(
            selected,
            selection[selected]["model_params"],
            features,
            labels,
            fold,
            touches_group,
            group_fold_array,
            n_folds,
            [float(f) for f in curve_fractions],
            seed,
            class_weight,
        )
        pd.DataFrame(curve_rows).to_csv(run_dir / "learning_curve.csv", index=False)

    junction = junction_diagnostic(
        pd.read_csv(run_dir / f"event_predictions_train_oof_{selected}.csv"),
        train_segments,
        data.events[data.events["stream_id"].isin(durations)],
        durations,
        float(get(cfg, "diagnostics.junction_margin_seconds", 0.5)),
        float(get(cfg, "diagnostics.junction_after_seconds", 0.0)),
    )

    selection_payload = {
        "selected_model": selected,
        "selection_metric": get(cfg, "threshold.selection_metric"),
        "selection_data": "out-of-fold window scores on the official train streams; folds grouped by subject_group",
        "ignore_flagged": ignore_flagged,
        "event_metric_source": "official" if evaluator is not None else "provisional",
        "models": selection,
    }
    write_json(run_dir / "selection.json", selection_payload)
    train_summary = {
        "class_balance": class_balance,
        "folds": {"n_folds": n_folds, "group_folds": group_folds},
        "models": oof_summary,
        "learning_curve": curve_rows,
        "junction_diagnostic": junction,
        "runtime_s": time.time() - started,
    }
    write_json(run_dir / "train_summary.json", train_summary)
    return {"selection": selection_payload, "summary": train_summary, "data": data, "stream_split": stream_split, "holdout": holdout}


def run_evaluate(cfg: dict, run_dir: Path, data: Dataset | None = None) -> dict:
    """Score the holdout split with frozen models and frozen post-processing."""
    started = time.time()
    run_dir = Path(run_dir)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    holdout = str(get(cfg, "split.holdout_split", "val"))
    data = data or prepare_dataset(cfg)
    stream_split = official_assignment(data.manifest)
    if holdout not in stream_split.values():
        raise LeakageError(f"no streams in holdout split {holdout!r}")
    train_streams = set(pd.read_csv(run_dir / "split_assignments.csv").query("unit == 'stream' and split == 'train'")["id"])
    holdout_streams = {s for s, v in stream_split.items() if v == holdout}
    check_disjoint({"train": train_streams, holdout: holdout_streams}, "streams")

    windows, features, names, _stats = extract_split(cfg, data, holdout)
    schema = json.loads((run_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if names != schema["names"]:
        raise AssertionError("holdout features do not match the trained feature schema")
    durations = {s: data.durations[s] for s in holdout_streams}
    ignore_flagged = bool(selection["ignore_flagged"])
    evaluator = load_official_evaluator(get(cfg, "evaluation.official_evaluator"))
    ground = ground_truth_records(data.events, durations, ignore_flagged)
    labels = windows["label"].to_numpy()
    labelled = labels >= 0
    results, scores = {}, {}
    for name, choice in selection["models"].items():
        model = joblib.load(run_dir / "models" / f"{name}.joblib")
        post = PostprocessParams(**choice["postprocessing"])
        scores[name] = predict_scores(model, features)
        events = events_for_streams(windows, scores[name], post, durations)
        events.to_csv(run_dir / f"event_predictions_{holdout}_{name}.csv", index=False, float_format="%.3f")
        records = prediction_records(events)
        per_stream = {
            s: event_report([g for g in ground if g["stream_id"] == s], [p for p in records if p["stream_id"] == s], {s: d}, evaluator)["ranking_score"]
            for s, d in sorted(durations.items())
        }
        result = {
            "window": window_metrics(labels[labelled], scores[name][labelled], post.threshold),
            "event": event_report(ground, records, durations, evaluator),
            "event_by_stream": per_stream,
        }
        if get(cfg, "evaluation.report_by_group", True):
            for column in ("dataset", "fall_type"):
                if column in data.events.columns:
                    table = recall_by_group(data.events, records, durations, ignore_flagged, column)
                    result[f"recall_by_{column}"] = table.to_dict(orient="records")
        results[name] = result
    selected = selection["selected_model"]
    shutil.copyfile(run_dir / f"event_predictions_{holdout}_{selected}.csv", run_dir / f"submission_{holdout}.csv")
    _save_predictions(run_dir / f"window_predictions_{holdout}.csv.gz", windows, scores, [])
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    final = {
        "holdout_split": holdout,
        "selected_model": selected,
        "event_metric_source": "official" if evaluator is not None else "provisional",
        "ignore_flagged": ignore_flagged,
        "note": "holdout was not used for fitting, threshold or post-processing choices",
        "models": {},
    }
    for name in selection["models"]:
        train_part = summary["models"][name]
        hold = results[name]
        final["models"][name] = {
            "train_insample": {
                "window_pr_auc": train_part["window_train_insample"]["pr_auc"],
                "event_f1_mean": train_part["event_train_insample"]["ranking_score"]["f1_mean"],
            },
            "train_oof": {
                "window_pr_auc": train_part["window_oof"]["pr_auc"],
                "fold_pr_auc_mean": train_part["fold_pr_auc_mean"],
                "fold_pr_auc_std": train_part["fold_pr_auc_std"],
                "event_f1_mean": train_part["event_oof"]["ranking_score"]["f1_mean"],
            },
            "holdout": hold,
        }
        final["models"][name]["diagnostics"] = diagnose(final["models"][name], float(get(cfg, "diagnostics.suspicious_score", 0.95)))
    final["runtime_s"] = time.time() - started
    write_json(run_dir / "final_metrics.json", final)
    return final


def diagnose(entry: dict, suspicious: float) -> list[str]:
    """Evidence-based notes about under/overfitting; they are hints, not verdicts."""
    notes = []
    insample = entry["train_insample"]["event_f1_mean"]
    oof = entry["train_oof"]["event_f1_mean"]
    held = entry["holdout"]["event"]["ranking_score"]["f1_mean"]
    pr_in = entry["train_insample"]["window_pr_auc"]
    pr_oof = entry["train_oof"]["window_pr_auc"]
    std = entry["train_oof"]["fold_pr_auc_std"]
    notes.append(f"event f1_mean: train in-sample {insample:.3f}, train out-of-fold {oof:.3f}, holdout {held:.3f}")
    if pr_in is not None and pr_oof is not None:
        notes.append(f"window PR-AUC: train in-sample {pr_in:.3f}, out-of-fold {pr_oof:.3f} (gap {pr_in - pr_oof:+.3f})")
        if pr_in < 0.5 and pr_oof < 0.5:
            notes.append("in-sample and out-of-fold PR-AUC are both low: underfitting or weak features/labels")
        if pr_in - pr_oof > 0.15:
            notes.append("large in-sample vs out-of-fold gap: the model memorises training subjects")
    if std is not None and std > 0.1:
        notes.append(f"fold PR-AUC std {std:.3f}: performance varies a lot between subject groups")
    if held - oof > 0.1:
        notes.append("holdout is clearly better than out-of-fold: check the split and the holdout difficulty")
    if oof - held > 0.15:
        notes.append("holdout is clearly worse than out-of-fold: possible domain shift between subjects")
    if max(oof, held) > suspicious:
        notes.append("score above the suspicious level: run a leakage audit before trusting it")
    return notes


def run_all(cfg: dict, config_path: str | Path | None = None, evaluate: bool = True) -> Path:
    """Train (and by default evaluate) in a new run folder. Returns the folder."""
    run_dir = create_run_dir(get(cfg, "experiment.output_root"), str(get(cfg, "experiment.name", "run")))
    collector = setup_logging(run_dir / "run.log")
    LOG.info("run folder: %s", run_dir)
    started = time.time()
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_copy(cfg), sort_keys=False), encoding="utf-8")
    environment = environment_info()
    environment["config_path"] = str(config_path) if config_path else None
    environment["seed"] = int(get(cfg, "experiment.seed"))
    write_json(run_dir / "environment.json", environment)
    status = "failed"
    try:
        trained = run_train(cfg, run_dir)
        status = "trained"
        if evaluate:
            final = run_evaluate(cfg, run_dir, trained["data"])
            status = "ok"
            LOG.info(
                "done: selected %s, holdout f1_mean %.4f",
                final["selected_model"],
                final["models"][final["selected_model"]]["holdout"]["event"]["ranking_score"]["f1_mean"],
            )
    finally:
        write_json(run_dir / "runtime.json", {"status": status, "seconds": time.time() - started})
        write_json(run_dir / "warnings.json", collector.messages)
    return run_dir


def evaluate_existing(run_dir: Path) -> dict:
    """Evaluate a trained run folder with its own resolved config. Never overwrites results."""
    run_dir = Path(run_dir)
    if not (run_dir / "selection.json").is_file():
        raise FileNotFoundError(f"{run_dir} has no selection.json; run the train stage first")
    if (run_dir / "final_metrics.json").exists():
        raise FileExistsError(f"{run_dir} is already evaluated; start a new run instead of overwriting it")
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    collector = setup_logging(run_dir / "evaluate.log")
    started = time.time()
    status = "failed"
    try:
        final = run_evaluate(cfg, run_dir)
        status = "ok"
    finally:
        write_json(run_dir / "evaluate_runtime.json", {"status": status, "seconds": time.time() - started})
        write_json(run_dir / "evaluate_warnings.json", collector.messages)
    return final


def run_manifest(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, segments, _events = build_manifest(cfg)
    save_manifest(manifest, segments, Path(get(cfg, "data.manifest_path")), Path(get(cfg, "data.segments_path")))
    return manifest, segments
