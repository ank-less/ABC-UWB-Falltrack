"""Window-level metrics and event-level scoring.

Event scores come from the official evaluator shipped with the dataset
(``src/abc2027/evaluate.py``) whenever it is configured. Without it (synthetic tests,
missing package) a local re-implementation of the rule written in the package README
is used, and every result is tagged ``"source": "provisional"``.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

LOG = logging.getLogger(__name__)

REPORT_THETAS = (0.1, 0.3, 0.5, 0.7)
RANKING_THETAS = (0.1, 0.3, 0.5)
PRIMARY_THETA = 0.3
IGNORE_MARGIN_S = 2.0


# ----------------------------------------------------------------------------- windows


def window_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Metrics on labelled windows (1 = fall, 0 = no fall; ambiguous rows must be removed).

    A set with a single class yields ``None`` for the ranking metrics plus a warning,
    instead of an exception or a misleading number.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("window labels must be 0 or 1")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    result: dict = {"n": int(len(labels)), "positives": positives, "negatives": negatives, "warnings": []}
    result["prevalence"] = positives / len(labels) if len(labels) else None
    if positives and negatives:
        result["pr_auc"] = float(average_precision_score(labels, scores))
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
        result["warnings"].append(f"only one class present (positives={positives}, negatives={negatives})")
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result.update(
        {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "balanced_accuracy": 0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0)),
        }
    )
    for message in result["warnings"]:
        LOG.warning("window metrics: %s", message)
    return result


def pr_curve_table(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels).astype(int)
    if labels.min(initial=0) == labels.max(initial=0):
        return pd.DataFrame(columns=["threshold", "precision", "recall"])
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})


# ----------------------------------------------------------------------------- events


def load_official_evaluator(path: str | Path | None) -> ModuleType | None:
    """Import the dataset's ``evaluate.py`` from its path, or return ``None``."""
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        LOG.warning("official evaluator not found at %s; event metrics will be provisional", path)
        return None
    spec = importlib.util.spec_from_file_location("abc2027_official_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("evaluate", "score_events", "temporal_iou"):
        if not hasattr(module, name):
            raise ImportError(f"{path} has no function {name}()")
    return module


def ground_truth_records(events: pd.DataFrame, durations: dict[str, float], ignore_flagged: bool) -> list[dict]:
    """Events in the evaluator's format, restricted to the scored streams.

    ``ignore = ignore or (ignore_flagged and (quality == low or needs_review))``, the rule
    of the official ``load_ground_truth``. Events cut by a shortened smoke-run stream
    (``t_end`` past the duration) are dropped.
    """
    rows = events[events["stream_id"].isin(durations)]
    records = []
    for row in rows.itertuples(index=False):
        if row.t_end > durations[row.stream_id] + 1e-9:
            continue
        records.append(
            {
                "stream_id": row.stream_id,
                "t_start": float(row.t_start),
                "t_end": float(row.t_end),
                "ignore": bool(row.ignore) or (ignore_flagged and bool(row.flagged)),
            }
        )
    return records


def prediction_records(predictions: pd.DataFrame) -> list[dict]:
    records = []
    for row in predictions.itertuples(index=False):
        alarm = getattr(row, "t_alarm", None)
        records.append(
            {
                "stream_id": row.stream_id,
                "t_start": float(row.t_start),
                "t_end": float(row.t_end),
                "score": float(row.score),
                "t_alarm": None if alarm is None or (isinstance(alarm, float) and math.isnan(alarm)) else float(alarm),
            }
        )
    return records


def event_report(ground: list[dict], predictions: list[dict], durations: dict[str, float], evaluator=None) -> dict:
    """Full report at every theta (official format when the evaluator is available)."""
    if evaluator is not None:
        report = evaluator.evaluate(ground, predictions, durations)
        report["source"] = "official"
        return report
    by_theta = {f"{theta:.1f}": provisional_score(ground, predictions, durations, theta) for theta in REPORT_THETAS}
    return {
        "source": "provisional",
        "ranking_score": {"f1_mean": float(np.mean([by_theta[f"{t:.1f}"]["f1"] for t in RANKING_THETAS]))},
        "primary": by_theta[f"{PRIMARY_THETA:.1f}"],
        "by_theta": by_theta,
    }


def ranking_f1(ground: list[dict], predictions: list[dict], durations: dict[str, float], evaluator=None) -> float:
    """Mean event F1 at theta 0.1, 0.3, 0.5 (the official ranking score), computed quickly."""
    if evaluator is not None:
        return float(np.mean([evaluator.score_events(ground, predictions, durations, t)["f1"] for t in RANKING_THETAS]))
    return float(np.mean([provisional_score(ground, predictions, durations, t)["f1"] for t in RANKING_THETAS]))


def _tiou(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = min(a1, b1) - max(a0, b0)
    if overlap <= 0.0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - overlap
    return overlap / union if union > 0 else 0.0


def greedy_match(ground: list[dict], predictions: list[dict], theta: float) -> tuple[list[tuple[int, int, float]], int, int]:
    """One stream: greedy one-to-one matching by descending score, tIoU >= theta.

    Returns ``(pairs, false_positive, ignored_predictions)`` where pairs index into the
    non-ignored ground truth, in the order of ``[g for g in ground if not g["ignore"]]``.
    """
    scored = [g for g in ground if not g["ignore"]]
    ignored = [g for g in ground if g["ignore"]]
    order = sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True)
    used: set[int] = set()
    pairs = []
    for index in order:
        p = predictions[index]
        best, best_iou = None, -1.0
        for g_index, g in enumerate(scored):
            if g_index in used:
                continue
            iou = _tiou(p["t_start"], p["t_end"], g["t_start"], g["t_end"])
            if iou >= theta and iou > best_iou:
                best, best_iou = g_index, iou
        if best is not None:
            used.add(best)
            pairs.append((index, best, best_iou))
    matched = {i for i, _g, _iou in pairs}
    false_positive = ignored_predictions = 0
    for index, p in enumerate(predictions):
        if index in matched:
            continue
        if any(
            min(p["t_end"], g["t_end"] + IGNORE_MARGIN_S) > max(p["t_start"], g["t_start"] - IGNORE_MARGIN_S)
            for g in ignored
        ):
            ignored_predictions += 1
        else:
            false_positive += 1
    return pairs, false_positive, ignored_predictions


def provisional_score(ground: list[dict], predictions: list[dict], durations: dict[str, float], theta: float) -> dict:
    tp = fp = fn = 0
    start_abs, ious = [], []
    for stream_id in sorted(durations):
        g = [e for e in ground if e["stream_id"] == stream_id]
        p = [e for e in predictions if e["stream_id"] == stream_id]
        pairs, false_positive, _ignored = greedy_match(g, p, theta)
        scored = [e for e in g if not e["ignore"]]
        tp += len(pairs)
        fp += false_positive
        fn += len(scored) - len(pairs)
        for p_index, g_index, iou in pairs:
            start_abs.append(abs(p[p_index]["t_start"] - scored[g_index]["t_start"]))
            ious.append(iou)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    hours = sum(durations.values()) / 3600.0
    return {
        "theta": theta,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarms_per_hour": fp / hours if hours else None,
        "start_error_median_abs": float(np.median(start_abs)) if start_abs else None,
        "mean_tiou": float(np.mean(ious)) if ious else None,
    }


def recall_by_group(
    events: pd.DataFrame, predictions: list[dict], durations: dict[str, float], ignore_flagged: bool, column: str, theta: float = PRIMARY_THETA
) -> pd.DataFrame:
    """Provisional per-group recall (e.g. by ``dataset`` or ``fall_type``) at one theta."""
    rows = []
    usable = events[events["stream_id"].isin(durations)]
    for stream_id, stream_events in usable.groupby("stream_id"):
        ground = ground_truth_records(stream_events, durations, ignore_flagged)
        kept = stream_events[stream_events["t_end"] <= durations[stream_id] + 1e-9].reset_index(drop=True)
        scored_rows = kept[[not g["ignore"] for g in ground]].reset_index(drop=True)
        pairs, _fp, _ig = greedy_match(ground, [p for p in predictions if p["stream_id"] == stream_id], theta)
        hit = {g_index for _p, g_index, _iou in pairs}
        for index, value in enumerate(scored_rows[column]):
            rows.append({column: value, "detected": index in hit})
    if not rows:
        return pd.DataFrame(columns=[column, "events", "detected", "recall"])
    table = pd.DataFrame(rows).groupby(column)["detected"].agg(["count", "sum"]).reset_index()
    table.columns = [column, "events", "detected"]
    table["recall"] = table["detected"] / table["events"]
    table["theta"] = theta
    table["source"] = "provisional"
    return table
