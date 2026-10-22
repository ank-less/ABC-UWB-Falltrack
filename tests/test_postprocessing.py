from __future__ import annotations

import numpy as np
import pandas as pd

from abc_uwb_falltrack.postprocessing import PostprocessParams, events_for_streams, scores_to_events, smooth_scores

STRIDE = 0.2


def grid(n: int):
    start = np.arange(n) * STRIDE
    return start, start + 1.0, np.arange(n)


def run(scores, **kwargs):
    start, end, index = grid(len(scores))
    params = PostprocessParams(threshold=0.5, **kwargs)
    return scores_to_events(start, end, index, np.asarray(scores, dtype=float), params, stream_duration=1000.0)


def test_runs_closer_than_merge_gap_are_merged():
    scores = [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
    assert len(run(scores, merge_gap_seconds=0.25)) == 2
    assert len(run(scores, merge_gap_seconds=0.0)) == 3
    assert len(run(scores, merge_gap_seconds=2.0)) == 1


def test_minimum_duration_and_minimum_windows():
    scores = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    assert len(run(scores, minimum_positive_windows=2)) == 1
    events = run(scores, minimum_event_duration_seconds=0.5)
    assert len(events) == 1 and events[0]["t_end"] - events[0]["t_start"] >= 0.5


def test_cooldown_drops_close_followers():
    scores = [1, 1, 0, 0, 1, 1, 0]
    assert len(run(scores)) == 2
    assert len(run(scores, cooldown_seconds=1.0)) == 1


def test_interval_modes_and_alarm_time():
    scores = [0, 0, 1, 1, 1, 0]
    centre = run(scores)[0]
    window = run(scores, interval_mode="window")[0]
    assert centre["t_start"] > window["t_start"] and centre["t_end"] < window["t_end"]
    assert centre["t_start"] <= centre["t_alarm"] <= centre["t_end"]
    assert centre["t_start"] < centre["t_end"]


def test_gap_in_window_index_breaks_a_run():
    index = np.array([0, 1, 2, 5, 6, 7])  # windows 3 and 4 were dropped (invalid frames)
    start = index * STRIDE
    params = PostprocessParams(threshold=0.5)
    events = scores_to_events(start, start + 1.0, index, np.ones(6), params, 1000.0)
    assert len(events) == 2


def test_events_never_cross_streams_and_stay_inside():
    start, end, index = grid(10)
    windows = pd.DataFrame(
        {
            "stream_id": ["a"] * 10 + ["b"] * 10,
            "w_index": np.concatenate([index, index]),
            "t_start": np.concatenate([start, start]),
            "t_end": np.concatenate([end, end]),
        }
    )
    scores = np.ones(20)
    out = events_for_streams(windows, scores, PostprocessParams(threshold=0.5, merge_gap_seconds=100.0), {"a": 2.0, "b": 2.5})
    assert sorted(out["stream_id"]) == ["a", "b"]
    assert out.loc[out.stream_id == "a", "t_end"].max() <= 2.0
    assert (out["t_start"] >= 0).all() and (out["t_start"] < out["t_end"]).all()


def test_smoothing_keeps_length_and_constant_signal():
    values = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    assert len(smooth_scores(values, 3)) == 5
    assert np.allclose(smooth_scores(np.ones(7), 3), 1.0)
    assert np.allclose(smooth_scores(values, 1), values)
