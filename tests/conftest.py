from __future__ import annotations

from pathlib import Path

import pytest

from abc_uwb_falltrack.synthetic import make_synthetic_package, synthetic_config

REPO = Path(__file__).resolve().parents[1]
REAL_PACKAGE = REPO / "data" / "uwb_falltrack" / "abc2027_uwb_train_v1"
OFFICIAL_EVALUATOR = REAL_PACKAGE / "src" / "abc2027" / "evaluate.py"


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    return make_synthetic_package(tmp_path_factory.mktemp("package"), n_train=3, n_val=2, seconds=120.0)


@pytest.fixture()
def synthetic_cfg(synthetic_root, tmp_path) -> dict:
    return synthetic_config(synthetic_root, tmp_path / "outputs")


@pytest.fixture(scope="session")
def official_evaluator():
    if not OFFICIAL_EVALUATOR.is_file():
        pytest.skip("official evaluator not available (dataset package not downloaded)")
    from abc_uwb_falltrack.metrics import load_official_evaluator

    return load_official_evaluator(OFFICIAL_EVALUATOR)
