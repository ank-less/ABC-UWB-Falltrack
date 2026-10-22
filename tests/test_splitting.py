from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abc_uwb_falltrack.experiment import attach_folds, extract_split, prepare_dataset
from abc_uwb_falltrack.manifest import subject_group
from abc_uwb_falltrack.splitting import (
    assign_group_folds,
    check_fold_assignment,
    official_assignment,
    purged_training_mask,
    window_fold_membership,
)
from abc_uwb_falltrack.validation import LeakageError, check_disjoint, check_group_leakage

GROUPS = [f"fall_person_{i}" for i in (1, 2, 3, 5, 6, 7)] + [f"adl_subject_{i}" for i in (4, 5, 6, 9, 10, 12)]


def test_subject_group_parsing():
    assert subject_group("fall", "falls/Dataset 3/Test_P3_3/Chute03_152.json") == "fall_person_3"
    assert subject_group("other", "falls/Dataset 1/Test_P1_7/Autre.json") == "fall_person_7"
    assert subject_group("adl", "activities/Faire_pates_9.json") == "adl_subject_9"
    # one person, three mat positions -> one group
    assert subject_group("fall", "falls/Dataset 1/Test_P1_2/x.json") == subject_group("fall", "falls/Dataset 2/Test_P2_2/y.json")


def test_each_group_in_exactly_one_fold():
    folds = assign_group_folds(GROUPS, 3, seed=2027)
    assert set(folds) == set(GROUPS)
    check_fold_assignment(folds, 3)
    for k in range(3):
        members = [g for g, f in folds.items() if f == k]
        assert any(g.startswith("fall_") for g in members) and any(g.startswith("adl_") for g in members)


def test_same_seed_same_split_and_other_seed_still_valid():
    first = assign_group_folds(GROUPS, 3, seed=11)
    assert first == assign_group_folds(list(reversed(GROUPS)), 3, seed=11)
    other = assign_group_folds(GROUPS, 3, seed=12)
    check_fold_assignment(other, 3)
    assert set(other) == set(GROUPS)
    assert sorted(np.bincount(list(other.values()))) == sorted(np.bincount(list(first.values())))


def test_group_shared_between_official_splits_is_rejected():
    segments = pd.DataFrame(
        {"stream_id": ["train_00", "val_00"], "subject_group": ["adl_subject_4", "adl_subject_4"]}
    )
    with pytest.raises(LeakageError):
        check_group_leakage(segments, {"train_00": "train", "val_00": "val"})


def test_disjoint_check_rejects_shared_streams():
    with pytest.raises(LeakageError):
        check_disjoint({"train": {"a", "b"}, "val": {"b"}}, "streams")


def test_window_membership_and_purge():
    # three segments: [0,10) fold 0, [9.8,20) fold 1 (crossfade overlap), [20,30) fold 2
    seg_start = np.array([0.0, 9.8, 20.0])
    seg_end = np.array([10.0, 20.0, 30.0])
    seg_fold = np.array([0, 1, 2])
    win_start = np.arange(0.0, 29.0, 0.2)
    win_end = win_start + 1.0
    fold, touches = window_fold_membership(win_start, win_end, seg_start, seg_end, seg_fold, 3)
    straddle = (win_start < 10.0) & (win_end > 9.8)
    assert touches[straddle][:, [0, 1]].all()
    for k in range(3):
        allowed = purged_training_mask(touches, k)
        assert not touches[allowed, k].any()
        assert not np.any(allowed & (fold == k))
    assert fold[0] == 0 and fold[-1] == 2


def test_windows_follow_their_stream_split_and_never_leak(synthetic_cfg):
    data = prepare_dataset(synthetic_cfg)
    split = official_assignment(data.manifest)
    group_split = check_group_leakage(data.segments, split)
    train_groups = sorted(g for g, s in group_split.items() if s == "train")
    folds = assign_group_folds(train_groups, 3, seed=7)
    windows, _x, _names, _stats = extract_split(synthetic_cfg, data, "train")
    assert set(windows["split"]) == {"train"}
    assert all(split[s] == "train" for s in windows["stream_id"].unique())
    segs = data.segments[data.segments["official_split"] == "train"]
    fold, touches, _major, _tg, _groups = attach_folds(windows, segs, folds, 3)
    for k in range(3):
        allowed = purged_training_mask(touches, k) & (fold >= 0)
        assert not touches[allowed, k].any()
    val_windows, _x, _n, _s = extract_split(synthetic_cfg, data, "val")
    assert set(val_windows["stream_id"]).isdisjoint(set(windows["stream_id"]))
