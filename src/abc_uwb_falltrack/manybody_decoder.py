"""Exact semi-Markov segment energy decoder cho many-body observables.

Đây là dynamic programming cổ điển giải đúng ground state của năng lượng
đoạn đã định nghĩa. Không phải QPU, quantum annealing hoặc bằng chứng quantum
advantage. Readout/QFI là quan sát đầu vào; decoder không nhận nhãn inference.
Duration prior mặc định là giả thuyết nghiên cứu, không phải luật challenge.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .manybody_readout import _event_bounds, event_is_ignored


def learn_duration_prior(
    events: dict[str, Iterable[Any]], fit_ids: list[str]
) -> dict[str, float]:
    """Ước lượng log-duration từ sự kiện hợp lệ của đúng phạm vi fit.

    Caller lưu output trong artifact/config trước khi mở nhãn held-out.
    Không đọc labels của ID khác, kể cả khi caller vô tình truyền thêm.
    """
    if not fit_ids or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("fit_ids must be nonempty and unique")
    if set(events) != set(fit_ids):
        raise ValueError("duration prior events must match exactly the fit scope")
    lengths = []
    for key in fit_ids:
        for event in events[key]:
            if not event_is_ignored(event):
                start, end = _event_bounds(event)
                lengths.append(end - start)
    if not lengths:
        raise ValueError("duration prior requires at least one valid fit event")
    logarithms = np.log(np.asarray(lengths))
    return {
        "duration_median_s": float(np.exp(np.median(logarithms))),
        "duration_log_sigma": float(max(0.25, np.std(logarithms))),
        "duration_fit_events": int(len(lengths)),
    }


def _causal_boundary_scale(signal: np.ndarray) -> np.ndarray:
    """Bound positive QFI using prefix RMS, without a held-out fitted scale."""
    positive = np.maximum(np.asarray(signal, dtype=np.float64), 0.0)
    # Scaling before squaring prevents overflow and cancels in the RMS ratio.
    positive = np.minimum(positive, 1e100)
    rms = np.sqrt(np.cumsum(np.square(positive)) / np.arange(1, positive.size + 1))
    return np.tanh(positive / np.maximum(rms, 1e-12))


def decode_semimarkov(
    heads: np.ndarray,
    centers_s: np.ndarray,
    duration_s: float,
    boundary_signal: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Giải chính xác tổng năng lượng các đoạn không chồng lấn.

    E(a,b) = integral(core-theta) + boundary_weight*(onset[a]+offset[b])
    + qfi_weight*(QFI[a]+QFI[b]) + duration_weight*log_shape(duration)
    - segment_penalty. Tối ưu tổng E với khoảng nghỉ ít nhất gap_s.

    Complexity O(N*D), bộ nhớ O(N+B*D), D là số frame trong max_duration_s.
    Các endpoint trong một block có cùng prefix dependencies đã biết được
    vector hóa; không pruning ứng viên. ``max_dp_block`` giới hạn bộ nhớ.
    Confidence dùng predicted heads, QFI và duration; không thể bảo đảm thứ tự
    TP/tIoU thực tế khi chưa có ground truth.
    """
    cfg = dict(config or {})
    values = np.asarray(heads, dtype=np.float64)
    centers = np.asarray(centers_s, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or centers.ndim != 1 or values.shape[0] != centers.size:
        raise ValueError("heads must have shape N by 3 and centers_s shape N")
    if not np.isfinite(values).all() or not np.isfinite(centers).all():
        raise ValueError("decoder inputs must be finite")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    n = centers.size
    if n == 0:
        return []
    if np.any(np.diff(centers) <= 0) or centers[0] < 0 or centers[-1] > duration_s:
        raise ValueError("centers_s must be increasing and inside the stream")
    values = np.clip(values, 0.0, 1.0)
    minimum = float(cfg.get("min_duration_s", 0.4))
    maximum = float(cfg.get("max_duration_s", 4.0))
    gap = float(cfg.get("gap_s", 0.2))
    theta = float(cfg.get("theta", 0.35))
    boundary_weight = float(cfg.get("boundary_weight", 0.12))
    qfi_weight = float(cfg.get("qfi_weight", 0.04))
    duration_weight = float(cfg.get("duration_weight", 0.08))
    penalty = float(cfg.get("segment_penalty", 0.12))
    median = float(cfg.get("duration_median_s", 1.74))
    sigma = float(cfg.get("duration_log_sigma", 0.5))
    parameters = [minimum, maximum, gap, theta, boundary_weight, qfi_weight, duration_weight, penalty, median, sigma]
    if not np.isfinite(parameters).all() or minimum <= 0 or maximum < minimum or gap < 0 or median <= 0 or sigma <= 0:
        raise ValueError("invalid semi-Markov duration/energy parameters")
    if not 0 <= theta <= 1 or min(boundary_weight, qfi_weight, duration_weight, penalty) < 0:
        raise ValueError("theta must be in [0,1] and energy weights nonnegative")
    max_block = int(cfg.get("max_dp_block", 512))
    if max_block <= 0:
        raise ValueError("max_dp_block must be positive")
    if boundary_signal is None:
        qfi = np.zeros(n)
    else:
        signal = np.asarray(boundary_signal, dtype=np.float64)
        if signal.shape != (n,) or not np.isfinite(signal).all():
            raise ValueError("boundary_signal must be a finite array of length N")
        qfi = _causal_boundary_scale(signal)
    step = float(np.median(np.diff(centers))) if n > 1 else duration_s
    edges = np.empty(n + 1)
    alignment = str(cfg.get("timestamp_alignment", "center"))
    if alignment == "right_edge":
        edges[1:] = centers
        edges[0] = max(0.0, centers[0] - step)
    elif alignment == "center":
        edges[1:n] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = max(0.0, centers[0] - step / 2.0)
        edges[n] = min(duration_s, centers[-1] + step / 2.0)
    else:
        raise ValueError("timestamp_alignment must be center or right_edge")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("inferred frame cells must have positive duration")
    prefix = np.r_[0.0, np.cumsum((values[:, 0] - theta) * np.diff(edges))]
    lower = np.searchsorted(edges, edges - maximum - 1e-10, side="left")
    upper = np.searchsorted(edges, edges - minimum + 1e-10, side="right") - 1
    # Prefix k contains cells [0,k); a previous event may end at edge[k].
    predecessor = np.maximum(0, np.searchsorted(edges, edges[:-1] - gap + 1e-10, side="right") - 1)
    max_predecessor = np.full(n + 1, -1, dtype=np.int64)
    valid_upper = upper >= 0
    max_predecessor[valid_upper] = predecessor[np.minimum(upper[valid_upper], n - 1)]
    max_frames = int(np.max(np.arange(1, n + 1) - lower[1:]))
    if max_frames <= 0 or not np.any(upper[1:] >= lower[1:]):
        return []
    offsets = np.arange(1, max_frames + 1)
    optimum = np.zeros(n + 1)
    chosen_start = np.full(n + 1, -1, dtype=np.int64)
    parent = np.arange(n + 1, dtype=np.int64) - 1
    cursor = 1
    while cursor <= n:
        # Every candidate predecessor in this block is strictly before cursor.
        # Therefore segment values can be computed together, and the skip
        # transition is exactly a cumulative maximum of these values.
        stop = min(n + 1, cursor + max_block, int(np.searchsorted(max_predecessor, cursor, side="left")))
        stop = max(cursor + 1, stop)
        ends = np.arange(cursor, stop)
        starts = ends[:, None] - offsets[None, :]
        valid = (starts >= lower[ends, None]) & (starts <= upper[ends, None]) & (starts >= 0)
        safe_starts = np.maximum(starts, 0)
        lengths = edges[ends, None] - edges[safe_starts]
        log_shape = -0.5 * np.square(np.log(np.maximum(lengths, 1e-12) / median) / sigma)
        energies = (
            prefix[ends, None] - prefix[safe_starts]
            + boundary_weight * (values[safe_starts, 1] + values[ends - 1, 2, None])
            + qfi_weight * (qfi[safe_starts] + qfi[ends - 1, None])
            + duration_weight * log_shape - penalty
        )
        candidate_values = optimum[predecessor[safe_starts]] + energies
        candidate_values[~valid] = -np.inf
        best_column = np.argmax(candidate_values, axis=1)
        best_value = candidate_values[np.arange(ends.size), best_column]
        optimum[cursor:stop] = np.maximum.accumulate(np.maximum(optimum[cursor - 1], best_value))
        selected = best_value > optimum[ends - 1]
        selected_ends = ends[selected]
        selected_starts = safe_starts[np.arange(ends.size), best_column][selected]
        chosen_start[selected_ends] = selected_starts
        parent[selected_ends] = predecessor[selected_starts]
        cursor = stop
    result = []
    end = n
    while end > 0:
        start = int(chosen_start[end])
        if start >= 0:
            length = edges[end] - edges[start]
            shape_confidence = float(np.exp(-0.5 * (np.log(length / median) / sigma) ** 2))
            boundary_confidence = float(np.sqrt(values[start, 1] * values[end - 1, 2]))
            core = values[start:end, 0]
            score = (
                0.45 * float(np.mean(core)) + 0.25 * float(np.max(core))
                + 0.20 * boundary_confidence + 0.05 * shape_confidence
                + 0.025 * float(qfi[start] + qfi[end - 1])
            )
            result.append({"t_start": float(edges[start]), "t_end": float(edges[end]), "score": float(np.clip(score, 0.0, 1.0))})
        end = int(parent[end])
    result.reverse()
    return result
