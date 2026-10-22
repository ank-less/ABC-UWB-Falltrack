from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abc_uwb_falltrack.data_io import load_events
from abc_uwb_falltrack.metrics import event_report, ground_truth_records, provisional_score, window_metrics

from conftest import REAL_PACKAGE


def test_imbalanced_windows_are_not_judged_by_accuracy():
    labels = np.zeros(1000, dtype=int)
    labels[:10] = 1
    all_negative = window_metrics(labels, np.zeros(1000), threshold=0.5)
    assert all_negative["f1"] == 0.0 and all_negative["recall"] == 0.0
    assert all_negative["balanced_accuracy"] == pytest.approx(0.5)
    assert all_negative["prevalence"] == pytest.approx(0.01)
    perfect = window_metrics(labels, labels.astype(float), threshold=0.5)
    assert perfect["pr_auc"] == pytest.approx(1.0) and perfect["f1"] == pytest.approx(1.0)
    noise = window_metrics(labels, np.random.default_rng(0).random(1000), threshold=0.5)
    assert noise["pr_auc"] < 0.2


def test_single_class_fold_warns_instead_of_crashing():
    result = window_metrics(np.zeros(50, dtype=int), np.random.default_rng(1).random(50), 0.5)
    assert result["pr_auc"] is None and result["roc_auc"] is None
    assert result["warnings"]
    empty = window_metrics(np.zeros(0, dtype=int), np.zeros(0), 0.5)
    assert empty["n"] == 0 and empty["pr_auc"] is None


def gt(stream, start, end, ignore=False):
    return {"stream_id": stream, "t_start": start, "t_end": end, "ignore": ignore}


def pred(stream, start, end, score=1.0):
    return {"stream_id": stream, "t_start": start, "t_end": end, "score": score, "t_alarm": None}


def test_provisional_event_rules():
    durations = {"val_00": 100.0}
    perfect = provisional_score([gt("val_00", 10, 12)], [pred("val_00", 10, 12)], durations, 0.5)
    assert perfect["f1"] == 1.0
    # an unmatched prediction next to an ignored fall is neither TP nor FP
    ignored = provisional_score([gt("val_00", 10, 12), gt("val_00", 50, 52, True)], [pred("val_00", 10, 12), pred("val_00", 53, 54)], durations, 0.5)
    assert ignored["fp"] == 0 and ignored["tp"] == 1
    # one ground truth, two overlapping predictions: the higher score wins, the other is a false alarm
    double = provisional_score([gt("val_00", 10, 12)], [pred("val_00", 10, 12, 0.9), pred("val_00", 10.5, 12, 0.95)], durations, 0.3)
    assert double["tp"] == 1 and double["fp"] == 1


def test_provisional_matches_official(official_evaluator):
    rng = np.random.default_rng(4)
    durations = {"val_00": 300.0, "val_01": 200.0}
    ground, predictions = [], []
    for stream, length in durations.items():
        for start in np.sort(rng.uniform(0, length - 5, 15)):
            ground.append(gt(stream, float(start), float(start + rng.uniform(0.5, 3)), bool(rng.random() < 0.2)))
        for start in rng.uniform(0, length - 5, 25):
            predictions.append(pred(stream, float(start), float(start + rng.uniform(0.3, 3)), float(rng.random())))
    for theta in (0.1, 0.3, 0.5, 0.7):
        ours = provisional_score(ground, predictions, durations, theta)
        official = official_evaluator.score_events(ground, predictions, durations, theta)
        for key in ("tp", "fp", "fn", "f1"):
            assert ours[key] == pytest.approx(official[key])
    assert event_report(ground, predictions, durations, official_evaluator)["source"] == "official"


def test_ground_truth_flag_rule_matches_official_loader(official_evaluator):
    path = REAL_PACKAGE / "streams_events.csv"
    events = load_events(path)
    durations = {s: 1e9 for s in events["stream_id"].unique()}
    ours = ground_truth_records(events, durations, ignore_flagged=True)
    official = official_evaluator.load_ground_truth(path, ignore_flagged=True)
    rows = lambda records: sorted((e["stream_id"], e["t_start"], e["t_end"], e["ignore"]) for e in records)  # noqa: E731
    assert rows(ours) == rows(official)
    assert any(e["ignore"] for e in ours)


def test_truncated_streams_drop_cut_events():
    events = pd.DataFrame(
        {"stream_id": ["val_00", "val_00"], "t_start": [10.0, 95.0], "t_end": [12.0, 105.0], "ignore": [False, False], "flagged": [False, False]}
    )
    records = ground_truth_records(events, {"val_00": 100.0}, ignore_flagged=True)
    assert len(records) == 1
