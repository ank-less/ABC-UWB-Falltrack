from __future__ import annotations

import sys

import numpy as np
import pytest

from abc_uwb_falltrack.features import window_features
from abc_uwb_falltrack.modeling import fit_model
from abc_uwb_falltrack.preprocessing import CausalBackgroundRemover, frame_series

from conftest import REAL_PACKAGE


def reference_ema_residual(x: np.ndarray, tau: float, rate: float) -> np.ndarray:
    alpha = np.exp(-1.0 / (rate * tau))
    s = x[:, 0, :].astype(np.float64).copy()
    out = np.empty_like(x, dtype=np.float64)
    for t in range(x.shape[1]):
        s = alpha * s + (1 - alpha) * x[:, t, :]
        out[:, t, :] = x[:, t, :] - s
    return out


def test_chunked_background_removal_matches_whole_stream():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 437, 8)).astype(np.float32)
    whole = CausalBackgroundRemover(3.0, 50.0).process(x)
    remover = CausalBackgroundRemover(3.0, 50.0)
    chunked = np.concatenate([remover.process(x[:, a : a + 100]) for a in range(0, 437, 100)], axis=1)
    assert np.allclose(whole, chunked)
    assert np.allclose(whole, reference_ema_residual(x, 3.0, 50.0))


def test_matches_official_background_filter():
    if not (REAL_PACKAGE / "src" / "abc2027" / "background.py").is_file():
        pytest.skip("dataset package not downloaded")
    sys.path.insert(0, str(REAL_PACKAGE / "src"))
    try:
        from abc2027.background import subtract_background
    finally:
        sys.path.pop(0)
    x = np.random.default_rng(1).normal(size=(3, 300, 6)).astype(np.float32)
    ours = CausalBackgroundRemover(3.0, 50.0).process(x)
    assert np.allclose(ours, subtract_background(x, 3.0), atol=1e-6)


def test_frame_series_does_not_depend_on_chunk_size(synthetic_root):
    path = sorted((synthetic_root / "streams" / "train").glob("stream_*.h5"))[0]
    a, valid_a, _ = frame_series(path, 50.0, 3.0, chunk_frames=333)
    b, valid_b, _ = frame_series(path, 50.0, 3.0, chunk_frames=5000)
    for name in a:
        assert np.allclose(a[name], b[name])
    assert np.array_equal(valid_a, valid_b)


def test_features_are_deterministic_finite_or_nan_and_well_shaped():
    rng = np.random.default_rng(2)
    values = {"bg_energy": rng.random((500, 3)), "diff_energy": rng.random((500, 3))}
    values["bg_energy"][10, 1] = np.inf
    starts = np.arange(0, 450, 10)
    first, names = window_features(values, starts, 50, ["bg_energy", "diff_energy"], ["mean", "std", "slope"], 50.0, context_seconds=1.0)
    second, _ = window_features(values, starts, 50, ["bg_energy", "diff_energy"], ["mean", "std", "slope"], 50.0, context_seconds=1.0)
    assert first.shape == (len(starts), len(names))
    assert len(set(names)) == len(names)
    assert np.array_equal(np.nan_to_num(first, nan=-1), np.nan_to_num(second, nan=-1))
    assert not np.isinf(first).any()
    with pytest.raises(ValueError):
        window_features(values, np.array([460]), 50, ["bg_energy"], ["mean"], 50.0)


def test_scaler_and_imputer_are_fitted_on_training_rows_only():
    rng = np.random.default_rng(3)
    x_train = rng.normal(0.0, 1.0, size=(200, 4))
    x_train[0, 2] = np.nan
    y_train = (rng.random(200) < 0.2).astype(int)
    model = fit_model("logistic", {"C": 1.0}, x_train, y_train, seed=0, class_weight="balanced")
    scaler = model.named_steps["scale"]
    imputer = model.named_steps["impute"]
    filled = imputer.transform(x_train)
    assert scaler.n_samples_seen_ == 200
    assert np.allclose(scaler.mean_, filled.mean(axis=0))
    before = scaler.mean_.copy()
    model.predict_proba(rng.normal(50.0, 1.0, size=(100, 4)))  # a shifted "validation" set
    assert np.array_equal(before, model.named_steps["scale"].mean_)


def test_single_class_training_gives_a_constant_scorer():
    model = fit_model("logistic", {}, np.zeros((10, 2)), np.zeros(10, dtype=int), seed=0, class_weight="balanced")
    assert np.allclose(model.predict_proba(np.zeros((3, 2)))[:, 1], 0.0)
