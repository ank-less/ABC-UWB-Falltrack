"""Baseline classifiers. Every fitted transform sits inside an sklearn Pipeline, so it
only ever sees the rows passed to ``fit`` (a training fold, or the official train split)."""

from __future__ import annotations

import itertools
import logging

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)

MODEL_NAMES = ("dummy", "logistic", "hist_gradient_boosting")


class ConstantScorer:
    """Stand-in when a training set holds a single class: predicts that class's rate."""

    def __init__(self, positive_rate: float):
        self.positive_rate = float(positive_rate)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        rate = np.full(len(features), self.positive_rate)
        return np.column_stack([1.0 - rate, rate])


def param_grid(parameters: dict | None) -> list[dict]:
    """Expand ``{"C": [0.1, 1.0]}`` into ``[{"C": 0.1}, {"C": 1.0}]``; scalars stay fixed."""
    if not parameters:
        return [{}]
    keys = sorted(parameters)
    options = [parameters[k] if isinstance(parameters[k], list) else [parameters[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*options)]


def make_model(name: str, params: dict, seed: int, class_weight: str | None = "balanced") -> Pipeline:
    if name == "dummy":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("clf", DummyClassifier(strategy="prior")),
            ]
        )
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(params.get("C", 1.0)),
                        class_weight=class_weight,
                        max_iter=int(params.get("max_iter", 5000)),
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        class_weight=class_weight,
                        max_iter=int(params.get("max_iter", 200)),
                        learning_rate=float(params.get("learning_rate", 0.1)),
                        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
                        early_stopping=False,
                        random_state=seed,
                    ),
                )
            ]
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def fit_model(name: str, params: dict, features: np.ndarray, labels: np.ndarray, seed: int, class_weight: str | None):
    """Fit on the given rows only. A single-class training set yields a ``ConstantScorer``."""
    labels = np.asarray(labels).astype(int)
    if len(labels) == 0:
        raise ValueError("no training rows")
    classes = np.unique(labels)
    if len(classes) < 2:
        LOG.warning("%s: training rows hold only class %s; using a constant scorer", name, classes.tolist())
        return ConstantScorer(float(classes[0] == 1))
    model = make_model(name, params, seed, class_weight)
    model.fit(features, labels)
    return model


def predict_scores(model, features: np.ndarray) -> np.ndarray:
    """Probability of the fall class."""
    if len(features) == 0:
        return np.zeros(0)
    proba = model.predict_proba(features)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return np.zeros(len(features))
    return proba[:, classes.index(1)]
