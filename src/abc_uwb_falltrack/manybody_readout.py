"""Readout đa nhiệm gọn nhẹ cho các quantum observables bất biến.

Module không mở tệp dữ liệu hoặc nhãn. Caller phải truyền đúng ``fit_ids``;
``events`` chứa ID ngoài phạm vi fit sẽ bị từ chối. Các biến đổi theo thời gian
chỉ dùng hiện tại/quá khứ của từng stream, không trộn trạng thái giữa stream.
Các xác suất là soft-target readouts, chưa được calibration trên held-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


HEAD_NAMES = ("core", "onset", "offset")


def _field(event: Any, name: str, default: Any = None) -> Any:
    return event.get(name, default) if isinstance(event, Mapping) else getattr(event, name, default)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def event_is_ignored(event: Any) -> bool:
    """Dùng cùng semantics cờ ignore/quality/needs_review của package."""
    return (
        _truthy(_field(event, "ignore", False))
        or str(_field(event, "quality", "")).strip().lower() == "low"
        or _truthy(_field(event, "needs_review", False))
    )


def _event_bounds(event: Any) -> tuple[float, float]:
    start, end = float(_field(event, "t_start")), float(_field(event, "t_end"))
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("event bounds must be finite with t_start < t_end")
    return start, end


def make_multitask_targets(
    centers_s: np.ndarray,
    events: Iterable[Any],
    *,
    onset_sigma_s: float = 0.18,
    offset_sigma_s: float = 0.18,
    core_ramp_s: float = 0.10,
    ignore_margin_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo [core, onset, offset] mềm chỉ từ nhãn trong phạm vi fit.

    Core có ramp ngắn ở bên trong hai biên. Boundary heads là Gaussian quanh
    timestamp biên. Vùng sự kiện bị cờ được mask ±2 s; phần nằm trong một
    sự kiện hợp lệ được ưu tiên phục hồi giống ``oof.make_targets``.
    """
    centers = np.asarray(centers_s, dtype=np.float64)
    if centers.ndim != 1 or not np.isfinite(centers).all():
        raise ValueError("centers_s must be a finite one-dimensional array")
    if centers.size > 1 and np.any(np.diff(centers) <= 0):
        raise ValueError("centers_s must be strictly increasing")
    if not np.isfinite([onset_sigma_s, offset_sigma_s, core_ramp_s, ignore_margin_s]).all():
        raise ValueError("target widths must be finite")
    if min(onset_sigma_s, offset_sigma_s, core_ramp_s) <= 0 or ignore_margin_s < 0:
        raise ValueError("target widths must be positive and ignore margin nonnegative")
    rows = list(events)
    targets = np.zeros((centers.size, 3), dtype=np.float64)
    usable = np.ones(centers.size, dtype=bool)
    for event in rows:
        start, end = _event_bounds(event)
        if event_is_ignored(event):
            usable[(centers >= start - ignore_margin_s) & (centers <= end + ignore_margin_s)] = False
    for event in rows:
        if event_is_ignored(event):
            continue
        start, end = _event_bounds(event)
        inside = (centers >= start) & (centers <= end)
        ramp = min(core_ramp_s, (end - start) / 4.0)
        core = np.clip(np.minimum((centers - start) / ramp, (end - centers) / ramp), 0.0, 1.0)
        targets[:, 0] = np.maximum(targets[:, 0], core)
        for column, boundary, sigma in ((1, start, onset_sigma_s), (2, end, offset_sigma_s)):
            distance = (centers - boundary) / sigma
            gaussian = np.exp(-0.5 * np.square(distance))
            gaussian[np.abs(distance) > 4.0] = 0.0
            targets[:, column] = np.maximum(targets[:, column], gaussian)
        usable[inside] = True
    return targets.astype(np.float32), usable


def temporal_features(
    features: np.ndarray,
    step_s: float,
    windows_s: Iterable[float] = (0.2, 0.6, 1.2),
    *,
    include_std: bool = False,
    clip: float | None = None,
) -> np.ndarray:
    """Ghép current/causal mean/current-minus-mean ở nhiều thang thời gian.

    Không padding bằng dữ liệu tương lai. Những frame đầu chỉ dùng prefix
    đang có. Số cột là F*(1+2*len(windows)); ``include_std`` thêm F mỗi cửa sổ.
    Clip nếu có là ngưỡng cố định trong config, không học từ held-out.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite N by F matrix")
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("step_s must be finite and positive")
    if clip is not None:
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")
        x = np.clip(x, -clip, clip)
    windows = tuple(float(value) for value in windows_s)
    if any(not np.isfinite(value) or value <= 0 for value in windows):
        raise ValueError("temporal windows must be finite and positive")
    if not windows or x.shape[0] == 0:
        return x.copy()
    prefix = np.vstack((np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)))
    prefix_sq = None
    if include_std:
        prefix_sq = np.vstack((np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(np.square(x, dtype=np.float64), axis=0)))
    ends = np.arange(1, x.shape[0] + 1)
    parts = [x]
    for window in windows:
        width = max(1, int(round(window / step_s)))
        starts = np.maximum(0, ends - width)
        count = (ends - starts)[:, None]
        means = ((prefix[ends] - prefix[starts]) / count).astype(np.float32)
        parts.extend((means, x - means))
        if prefix_sq is not None:
            variance = (prefix_sq[ends] - prefix_sq[starts]) / count - np.square(means, dtype=np.float64)
            parts.append(np.sqrt(np.maximum(variance, 0.0)).astype(np.float32))
    return np.concatenate(parts, axis=1)


@dataclass
class ManyBodyReadout:
    """Pickle/joblib-compatible artifact; scaler fitted on selected fit rows."""

    kind: str
    models: tuple[Any, Any, Any]
    scaler: Any | None
    config: dict[str, Any]
    step_s: float
    input_features: int
    training_metadata: dict[str, Any]


def _row_quotas(counts: np.ndarray, budget: int) -> np.ndarray:
    """Water filling: phân bổ số hàng gần bằng nhau giữa các fit groups."""
    quotas = np.zeros_like(counts)
    remaining = min(int(counts.sum()), budget)
    while remaining:
        active = np.flatnonzero(quotas < counts)
        share = max(1, remaining // active.size)
        for index in active:
            take = min(share, int(counts[index] - quotas[index]), remaining)
            quotas[index] += take
            remaining -= take
            if remaining == 0:
                break
    return quotas


def _sample_rows(
    targets: np.ndarray, usable: np.ndarray, quota: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Giữ vùng nhãn hữu ích, bù xác suất lấy mẫu cho từng stratum."""
    positive = np.flatnonzero(usable & (np.max(targets, axis=1) >= 0.10))
    negative = np.flatnonzero(usable & (np.max(targets, axis=1) < 0.10))
    n_positive = min(positive.size, max(1, quota // 2)) if positive.size else 0
    n_negative = min(negative.size, quota - n_positive)
    n_positive = min(positive.size, quota - n_negative)
    selected, weights = [], []
    for pool, count in ((positive, n_positive), (negative, n_negative)):
        if count:
            rows = pool if count == pool.size else rng.choice(pool, size=count, replace=False)
            selected.append(rows)
            weights.append(np.full(count, pool.size / count, dtype=np.float64))
    index = np.concatenate(selected)
    correction = np.concatenate(weights)
    order = np.argsort(index)
    return index[order], correction[order], {
        "available_positive_rows": int(positive.size),
        "available_negative_rows": int(negative.size),
        "sampled_positive_rows": int(n_positive),
        "sampled_negative_rows": int(n_negative),
    }


def fit_readout(
    packs: dict[str, dict[str, Any]],
    events: dict[str, Iterable[Any]],
    fit_ids: list[str],
    config: dict[str, Any],
    seed: int,
) -> ManyBodyReadout:
    """Fit 3 regularized heads trên đúng IDs được truyền, tối đa 60k hàng.

    ``kind='histgb'`` mặc định: 7 lá, 100 iterations, L2=10, không early
    stopping trên held-out. ``kind='ridge'``: train-only StandardScaler và
    Ridge(alpha=100). Trọng số mỗi group có tổng bằng nhau trước khi cân bằng
    soft class. Subsampling được bù bằng inverse inclusion probability.
    """
    if not fit_ids or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("fit_ids must be nonempty and unique")
    scope = set(fit_ids)
    if set(events).difference(scope):
        raise ValueError("events contains IDs outside the fit scope")
    if not scope.issubset(packs) or not scope.issubset(events):
        raise ValueError("each fit ID must have a feature pack and event list")
    cfg = dict(config)
    kind = str(cfg.get("kind", "histgb"))
    if kind == "hist_gradient_boosting":
        kind = "histgb"
    if kind not in {"ridge", "histgb"}:
        raise ValueError(f"unsupported manybody readout kind: {kind}")
    budget = int(cfg.get("max_train_rows", cfg.get("max_training_rows", 60000)))
    if budget < len(fit_ids) or budget > 60000:
        raise ValueError("max_train_rows must cover every fit group and be <= 60000")
    step_values = []
    skipped_ids = []
    raw_features, target_parts, mask_parts = {}, {}, {}
    feature_count = None
    for key in fit_ids:
        pack = packs[key]
        centers = np.asarray(pack["centers_s"], dtype=np.float64)
        x = np.asarray(pack["features"], dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != centers.size or not np.isfinite(x).all():
            raise ValueError(f"invalid feature matrix for fit group {key}")
        if feature_count is not None and feature_count != x.shape[1]:
            raise ValueError("fit groups must share the same feature count")
        feature_count = x.shape[1]
        if centers.size > 1:
            step_values.append(float(np.median(np.diff(centers))))
        group_events = list(events[key])
        if any(_field(event, "stream_id", key) != key for event in group_events):
            raise ValueError(f"event stream_id does not match fit group {key}")
        targets, usable = make_multitask_targets(
            centers, group_events,
            onset_sigma_s=float(cfg.get("onset_sigma_s", 0.18)),
            offset_sigma_s=float(cfg.get("offset_sigma_s", 0.18)),
            core_ramp_s=float(cfg.get("core_ramp_s", 0.10)),
            ignore_margin_s=float(cfg.get("ignore_margin_s", 2.0)),
        )
        if not usable.any():
            # Một source clip ngắn có thể bị mask hoàn toàn bởi ±2 s quanh
            # sự kiện flagged. Bỏ group đó, không khôi phục dữ liệu bị ignore.
            skipped_ids.append(key)
            continue
        raw_features[key], target_parts[key], mask_parts[key] = x, targets, usable
    effective_fit_ids = [key for key in fit_ids if key not in skipped_ids]
    if not effective_fit_ids:
        raise ValueError("fit scope has no usable rows after ignored-event masking")
    step_s = float(cfg.get("step_s", np.median(step_values) if step_values else 0.1))
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("readout step_s must be finite and positive")
    if step_values and not np.allclose(step_values, step_s, rtol=0.01, atol=1e-6):
        raise ValueError("fit groups have inconsistent temporal grids")
    windows = tuple(float(value) for value in cfg.get("temporal_windows_s", (0.2, 0.6, 1.2)))
    cfg["temporal_windows_s"] = windows
    cfg["temporal_include_std"] = bool(cfg.get("temporal_include_std", False))
    available_counts = np.asarray([mask_parts[key].sum() for key in effective_fit_ids], dtype=int)
    per_group_limit = int(cfg.get("max_rows_per_group", budget))
    if per_group_limit < 1:
        raise ValueError("max_rows_per_group must be positive")
    quotas = _row_quotas(np.minimum(available_counts, per_group_limit), budget)
    rng = np.random.default_rng(seed)
    x_parts, y_parts, w_parts, sampling = [], [], [], {}
    for key, quota in zip(effective_fit_ids, quotas):
        indices, correction, summary = _sample_rows(target_parts[key], mask_parts[key], int(quota), rng)
        transformed = temporal_features(
            raw_features[key], step_s, windows,
            include_std=cfg["temporal_include_std"], clip=cfg.get("feature_clip"),
        )
        x_parts.append(transformed[indices])
        y_parts.append(target_parts[key][indices])
        w_parts.append(correction / float(mask_parts[key].sum()))
        sampling[key] = summary
    x_train = np.concatenate(x_parts)
    y_train = np.concatenate(y_parts)
    group_weight = np.concatenate(w_parts)
    group_weight /= group_weight.mean()
    scaler = None
    if kind == "ridge":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train, sample_weight=group_weight)
    positive_weights = np.asarray(cfg.get("positive_weights", (
        float(cfg.get("positive_weight", 4.0)),
        float(cfg.get("boundary_positive_weight", 8.0)),
        float(cfg.get("boundary_positive_weight", 8.0)),
    )), dtype=float)
    if positive_weights.shape != (3,) or not np.isfinite(positive_weights).all() or np.any(positive_weights < 1):
        raise ValueError("positive_weights must contain three finite values >= 1")
    models = []
    for column in range(3):
        weights = group_weight * (1.0 + (positive_weights[column] - 1.0) * y_train[:, column])
        weights /= weights.mean()
        if kind == "ridge":
            model = Ridge(alpha=float(cfg.get("alpha", 100.0)), solver="lsqr", tol=1e-4)
        else:
            model = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=float(cfg.get("learning_rate", 0.06)),
                max_iter=int(cfg.get("max_iter", 100)), max_leaf_nodes=int(cfg.get("max_leaf_nodes", 7)),
                min_samples_leaf=int(cfg.get("min_samples_leaf", 60)),
                l2_regularization=float(cfg.get("l2_regularization", 10.0)),
                max_bins=int(cfg.get("max_bins", 128)), early_stopping=False,
                random_state=int(seed + column),
            )
        model.fit(x_train, y_train[:, column], sample_weight=weights)
        models.append(model)
    return ManyBodyReadout(
        kind=kind, models=tuple(models), scaler=scaler, config=cfg,
        step_s=step_s, input_features=int(feature_count),
        training_metadata={
            "fit_ids": effective_fit_ids, "requested_fit_ids": list(fit_ids),
            "skipped_no_usable_rows": skipped_ids,
            "seed": int(seed), "heads": list(HEAD_NAMES),
            "sampled_rows": int(x_train.shape[0]), "transformed_features": int(x_train.shape[1]),
            "sampling": sampling, "group_weighting": "equal mass with stratum inclusion correction",
            "positive_weights": positive_weights.tolist(), "ignore_margin_s": float(cfg.get("ignore_margin_s", 2.0)),
            "calibrated": False,
        },
    )


def predict_readout(model: ManyBodyReadout, features: np.ndarray) -> np.ndarray:
    """Dự đoán một stream mới, cùng grid step_s; không nhận nhãn hay metadata."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != model.input_features:
        raise ValueError("inference features do not match readout input shape")
    transformed = temporal_features(
        x, model.step_s, model.config["temporal_windows_s"],
        include_std=model.config.get("temporal_include_std", False),
        clip=model.config.get("feature_clip"),
    )
    if x.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if model.scaler is not None:
        transformed = model.scaler.transform(transformed)
    heads = np.column_stack([head.predict(transformed) for head in model.models])
    return np.clip(heads, 0.0, 1.0).astype(np.float32)
