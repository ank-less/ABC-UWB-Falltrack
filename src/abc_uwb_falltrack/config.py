"""Load, override and check the YAML experiment config."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = (
    "data",
    "validation",
    "split",
    "preprocessing",
    "windowing",
    "features",
    "model",
    "threshold",
    "postprocessing",
    "evaluation",
    "experiment",
)

# Values that must be set explicitly: the pipeline never guesses them.
REQUIRED_VALUES = (
    "data.root",
    "data.streams_dir",
    "data.events_csv",
    "data.composition_csv",
    "data.rate_hz",
    "data.expected_radars",
    "data.expected_bins",
    "data.chunk_frames",
    "split.n_folds",
    "split.seed",
    "preprocessing.background_tau_seconds",
    "windowing.duration_seconds",
    "windowing.stride_seconds",
    "windowing.positive_overlap_threshold",
    "threshold.candidate_values",
    "experiment.seed",
    "experiment.output_root",
)


class ConfigError(ValueError):
    """The config is missing a value or holds an invalid one."""


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def apply_override(cfg: dict, item: str) -> None:
    """Apply ``section.key=value``; the value is parsed as YAML (``null``, numbers, lists)."""
    if "=" not in item:
        raise ConfigError(f"override {item!r} must look like section.key=value")
    dotted, raw = item.split("=", 1)
    keys = dotted.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {item!r}: {key} is not a section")
    node[keys[-1]] = yaml.safe_load(raw)


def check_config(cfg: dict) -> None:
    missing = [name for name in REQUIRED_SECTIONS if name not in cfg]
    if missing:
        raise ConfigError(f"config is missing sections: {', '.join(missing)}")
    unset = [name for name in REQUIRED_VALUES if get(cfg, name) is None]
    if unset:
        raise ConfigError(
            "config values must be set explicitly (they are not guessed): "
            + ", ".join(unset)
            + ". See docs/baseline_pipeline.md for the confirmed values."
        )
    positive = (
        "data.rate_hz",
        "data.chunk_frames",
        "preprocessing.background_tau_seconds",
        "windowing.duration_seconds",
        "windowing.stride_seconds",
    )
    for name in positive:
        if float(get(cfg, name)) <= 0:
            raise ConfigError(f"{name} must be positive")
    if int(get(cfg, "split.n_folds")) < 2:
        raise ConfigError("split.n_folds must be at least 2")
    pos = float(get(cfg, "windowing.positive_overlap_threshold"))
    neg = float(get(cfg, "windowing.negative_overlap_max", 0.0))
    if not 0.0 < pos <= 1.0 or not 0.0 <= neg < pos:
        raise ConfigError("need 0 <= windowing.negative_overlap_max < positive_overlap_threshold <= 1")
    policy = get(cfg, "windowing.flagged_event_policy", "positive")
    if policy not in {"positive", "ambiguous"}:
        raise ConfigError("windowing.flagged_event_policy must be 'positive' or 'ambiguous'")
    if get(cfg, "postprocessing.interval_mode", "center") not in {"center", "window"}:
        raise ConfigError("postprocessing.interval_mode must be 'center' or 'window'")
    if not get(cfg, "model.candidates"):
        raise ConfigError("model.candidates must list at least one model")


def load_config(path: str | Path, overrides: list[str] | None = None, smoke: bool = False) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if smoke:
        smoke_values = get(cfg, "experiment.smoke") or {}
        for key, value in smoke_values.items():
            cfg["experiment"][key] = value
        cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'run')}_smoke"
    for item in overrides or []:
        apply_override(cfg, item)
    check_config(cfg)
    return cfg


def resolved_copy(cfg: dict) -> dict:
    """A deep copy suitable for ``resolved_config.yaml``."""
    return copy.deepcopy(cfg)


def data_path(cfg: dict, key: str) -> Path:
    """Resolve ``data.<key>`` against ``data.root`` unless it is absolute."""
    value = Path(get(cfg, f"data.{key}"))
    return value if value.is_absolute() else Path(get(cfg, "data.root")) / value
