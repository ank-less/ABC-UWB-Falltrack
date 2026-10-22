from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abc_uwb_falltrack.config import ConfigError
from abc_uwb_falltrack.windowing import AMBIGUOUS, NEGATIVE, POSITIVE, WindowSpec, label_windows, make_windows, window_starts

SPEC = WindowSpec(duration_s=1.0, stride_s=0.2, rate_hz=50.0)


def events(*pairs, flagged=None):
    frame = pd.DataFrame(pairs, columns=["t_start", "t_end"])
    frame["flagged"] = flagged if flagged is not None else False
    return frame


def test_windows_stay_inside_the_stream():
    for n in (49, 50, 51, 123, 1000):
        starts = window_starts(n, SPEC)
        assert np.all(starts >= 0)
        assert np.all(starts + SPEC.frames <= n)
    assert len(window_starts(49, SPEC)) == 0


def test_drop_last_false_adds_one_end_aligned_window():
    spec = WindowSpec(1.0, 0.2, 50.0, drop_last=False)
    starts = window_starts(123, spec)
    assert starts[-1] + spec.frames == 123
    assert np.all(starts + spec.frames <= 123)


def test_non_integer_frame_counts_are_rejected():
    with pytest.raises(ConfigError):
        _ = WindowSpec(1.01, 0.2, 50.0).frames


def test_window_times_come_from_timestamps_and_invalid_frames_drop_windows():
    t = np.arange(200) / 50.0
    valid = np.ones(200, dtype=bool)
    valid[120] = False
    table = make_windows("train_00", t, valid, SPEC)
    assert np.allclose(table["t_end"] - table["t_start"], 1.0)
    assert table["t_end"].max() <= 200 / 50.0 + 1e-9
    hit = (table["f_start"] <= 120) & (table["f_stop"] > 120)
    assert not table.loc[hit, "keep"].any()
    assert table.loc[~hit, "keep"].all()


def test_overlap_labels_at_the_boundaries():
    fall = events((1.0, 2.0))
    starts = np.array([0.0, 0.5, 0.6, 2.0, 2.4, 0.1])
    ends = starts + 1.0
    labels, coverage = label_windows(starts, ends, fall, positive_threshold=0.5)
    # [0,1): touches the start only -> no overlap -> negative
    assert labels[0] == NEGATIVE and coverage[0] == 0.0
    # [0.5,1.5): half the window -> positive at threshold 0.5
    assert labels[1] == POSITIVE
    # [2.0,3.0): touches the end only -> negative
    assert labels[3] == NEGATIVE
    # [0.1,1.1): 0.1 s overlap -> ambiguous
    assert labels[5] == AMBIGUOUS


def test_short_fall_inside_a_window_is_positive():
    labels, coverage = label_windows(np.array([10.0]), np.array([11.0]), events((10.2, 10.6)), 0.5)
    assert labels[0] == POSITIVE and coverage[0] == pytest.approx(1.0)


def test_flagged_policy_ambiguous():
    flagged = events((1.0, 2.0), flagged=[True])
    starts = np.array([1.0, 5.0])
    labels_pos, _ = label_windows(starts, starts + 1.0, flagged, 0.5, flagged_policy="positive")
    labels_amb, _ = label_windows(starts, starts + 1.0, flagged, 0.5, flagged_policy="ambiguous")
    assert labels_pos.tolist() == [POSITIVE, NEGATIVE]
    assert labels_amb.tolist() == [AMBIGUOUS, NEGATIVE]


def test_negative_margin_widens_the_ambiguous_band():
    fall = events((1.0, 2.0))
    starts = np.array([0.1])
    labels, _ = label_windows(starts, starts + 1.0, fall, 0.5, negative_max=0.2)
    assert labels[0] == NEGATIVE
