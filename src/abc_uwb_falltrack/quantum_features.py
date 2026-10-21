"""Causal UWB motion features and exact density-matrix observables.

Each 0.1 s feature step is derived from adjacent CIR frames, so static range
profiles cancel before the quantum state is prepared. The four-qubit state is
an amplitude encoding of 16 pooled motion-energy bins for each radar. The
three radar states form an ensemble whose density-matrix observables feed a
classical temporal readout. An optional five-qubit branch combines that motion
state with a causal per-stream static-profile contrast.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import lfilter


def _pauli_z_observables() -> tuple[np.ndarray, list[str]]:
    basis = np.arange(16, dtype=np.int64)
    z = np.stack([1.0 - 2.0 * ((basis >> bit) & 1) for bit in range(4)], axis=1)
    observables = [z[:, bit] for bit in range(4)]
    names = [f"q_z_{bit}" for bit in range(4)]
    for first in range(4):
        for second in range(first + 1, 4):
            observables.append(z[:, first] * z[:, second])
            names.append(f"q_zz_{first}_{second}")
    return np.stack(observables, axis=1), names


_Z_OBSERVABLES, _Z_NAMES = _pauli_z_observables()


def _reservoir_entangler_phases(entangler_angle: float) -> np.ndarray:
    """Diagonal phases for a four-qubit ring of ZZ interactions."""
    basis = np.arange(16, dtype=np.int64)
    bits = ((basis[:, None] >> np.arange(4, dtype=np.int64)) & 1).astype(np.float64)
    z = 1.0 - 2.0 * bits
    ring_interaction = np.sum(z * np.roll(z, -1, axis=1), axis=1)
    return np.exp(-1j * entangler_angle * ring_interaction)


def _hadamard_density_transform(density: np.ndarray) -> np.ndarray:
    """Apply H^tensor4 to both sides of a batch of 16x16 matrices."""
    transformed = density.copy()
    scale = 1.0 / np.sqrt(2.0)
    basis = np.arange(16, dtype=np.int64)
    for axis in (1, 2):
        for bit in range(4):
            stride = 1 << bit
            lower = basis[(basis & stride) == 0]
            upper = lower | stride
            updated = transformed.copy()
            if axis == 1:
                left = transformed[:, lower, :]
                right = transformed[:, upper, :]
                updated[:, lower, :] = (left + right) * scale
                updated[:, upper, :] = (left - right) * scale
            else:
                left = transformed[:, :, lower]
                right = transformed[:, :, upper]
                updated[:, :, lower] = (left + right) * scale
                updated[:, :, upper] = (left - right) * scale
            transformed = updated
    return transformed


def _reservoir_readout_operators() -> tuple[np.ndarray, list[str]]:
    """Return diagonal Z/ZZ tables and bit-flip masks for X/Y readouts."""
    basis = np.arange(16, dtype=np.int64)
    bits = ((basis[:, None] >> np.arange(4, dtype=np.int64)) & 1).astype(np.float64)
    z = 1.0 - 2.0 * bits
    diagonal_operators = [z[:, bit] for bit in range(4)]
    names = [f"z_q{bit}" for bit in range(4)]
    for bit in range(4):
        next_bit = (bit + 1) % 4
        diagonal_operators.append(z[:, bit] * z[:, next_bit])
        names.append(f"zz_q{bit}_q{next_bit}")
    return np.stack(diagonal_operators, axis=0), names


_RESERVOIR_DIAGONAL_READOUTS, _RESERVOIR_DIAGONAL_NAMES = _reservoir_readout_operators()
_RESERVOIR_BASIS = np.arange(16, dtype=np.int64)
_RESERVOIR_Z = 1.0 - 2.0 * ((_RESERVOIR_BASIS[:, None] >> np.arange(4)) & 1)


def feature_names(
    *,
    include_static_state: bool = False,
    include_fidelity_contrast: bool = False,
    include_normalized_rho_similarity: bool = False,
    include_quantum_reservoir: bool = False,
) -> list[str]:
    names = [f"motion_log_bin_r{radar}_{bin_index}" for radar in range(3) for bin_index in range(16)]
    names.extend(f"motion_log_energy_r{radar}" for radar in range(3))
    names.extend(f"motion_share_r{radar}" for radar in range(3))
    names.append("motion_log_energy_all")
    names.extend(f"rho_eigenvalue_{index}" for index in range(3))
    names.extend(["rho_von_neumann_entropy", "rho_purity", "rho_temporal_overlap"])
    names.extend(["state_entanglement_mean", "state_entanglement_std"])
    names.extend(f"{name}_mean_radars" for name in _Z_NAMES)
    names.extend(f"{name}_std_radars" for name in _Z_NAMES)
    for lag in (1, 3):
        names.extend(f"state_fidelity_lag{lag}_r{radar}" for radar in range(3))
    if include_normalized_rho_similarity:
        names.append("rho_temporal_similarity_normalized")
    if include_fidelity_contrast:
        names.extend(f"state_fidelity_lag1_minus_lag3_r{radar}" for radar in range(3))
    for window_s in (0.4, 0.8):
        names.extend(
            [
                f"rho_window_entropy_{window_s:g}s_mean_radars",
                f"rho_window_entropy_{window_s:g}s_std_radars",
                f"rho_window_purity_{window_s:g}s_mean_radars",
                f"rho_window_purity_{window_s:g}s_std_radars",
            ]
        )
    for tau in (0.4, 1.2):
        names.extend(f"motion_log_energy_ema_{tau:g}s_r{radar}" for radar in range(3))
    if include_static_state:
        names.extend(f"static_log_contrast_energy_r{radar}" for radar in range(3))
        names.extend(f"hybrid_static_fraction_r{radar}" for radar in range(3))
        names.extend(["hybrid_rho_von_neumann_entropy", "hybrid_rho_purity"])
        names.extend(["hybrid_flag_z_mean_radars", "hybrid_flag_z_std_radars"])
        names.extend(["hybrid_flag_entanglement_mean", "hybrid_flag_entanglement_std"])
        for lag in (1, 3):
            names.extend(f"hybrid_state_fidelity_lag{lag}_r{radar}" for radar in range(3))
    if include_quantum_reservoir:
        for radar in range(3):
            names.extend(f"qreservoir_x_r{radar}_q{qubit}" for qubit in range(4))
            names.extend(f"qreservoir_y_r{radar}_q{qubit}" for qubit in range(4))
            names.extend(f"qreservoir_z_r{radar}_q{qubit}" for qubit in range(4))
            names.extend(
                f"qreservoir_zz_r{radar}_q{qubit}_q{(qubit + 1) % 4}"
                for qubit in range(4)
            )
    return names


def _quantum_reservoir_features(
    amplitudes: np.ndarray,
    initial_density: np.ndarray | None,
    step_s: float,
    *,
    memory_tau_s: float,
    phase_scale: float,
    entangler_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve a causal, reset-channel quantum reservoir for each radar."""
    count = amplitudes.shape[0]
    if count == 0:
        raise ValueError("quantum reservoir requires at least one feature step")
    if memory_tau_s <= 0.0:
        raise ValueError("quantum reservoir memory tau must be positive")
    if not np.isfinite(phase_scale) or not np.isfinite(entangler_angle):
        raise ValueError("quantum reservoir angles must be finite")

    plus = np.full(16, 0.25, dtype=np.complex128)
    reset_density = np.outer(plus, plus.conj())
    density = (
        np.repeat(reset_density[None, :, :], 3, axis=0)
        if initial_density is None
        else initial_density.astype(np.complex128, copy=True)
    )
    if density.shape != (3, 16, 16):
        raise ValueError("quantum reservoir state must have shape (3, 16, 16)")

    memory = float(np.exp(-step_s / memory_tau_s))
    entangler = _reservoir_entangler_phases(entangler_angle)
    readouts = np.empty((count, 3, 16), dtype=np.float64)
    diagonal_operators = _RESERVOIR_DIAGONAL_READOUTS

    for index in range(count):
        probabilities = amplitudes[index] * amplitudes[index]
        phase = np.exp(1j * phase_scale * (16.0 * probabilities - 1.0) / 15.0)
        phase *= entangler[None, :]
        phase_conjugate = phase.conj()
        encoded = phase[:, :, None] * density * phase_conjugate[:, None, :]
        evolved = _hadamard_density_transform(encoded)
        density = memory * evolved + (1.0 - memory) * reset_density[None, :, :]
        density = 0.5 * (density + density.conj().transpose(0, 2, 1))

        diagonal = np.real(np.diagonal(density, axis1=1, axis2=2))
        diagonal_values = diagonal @ diagonal_operators.T
        x_values = np.empty((3, 4), dtype=np.float64)
        y_values = np.empty((3, 4), dtype=np.float64)
        for qubit in range(4):
            mask = 1 << qubit
            flipped = density[:, _RESERVOIR_BASIS ^ mask, _RESERVOIR_BASIS]
            x_values[:, qubit] = np.real(np.sum(flipped, axis=1))
            y_values[:, qubit] = np.imag(np.sum(flipped * _RESERVOIR_Z[:, qubit], axis=1))
        readouts[index] = np.concatenate([x_values, y_values, diagonal_values], axis=1)

    return readouts.reshape(count, 48), density


def _causal_ema(values: np.ndarray, tau_s: float, step_s: float, initial: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.exp(-step_s / tau_s))
    smoothed = np.empty_like(values, dtype=np.float64)
    state = values[0].astype(np.float64, copy=True) if initial is None else initial.astype(np.float64, copy=True)
    for index, value in enumerate(values):
        state = alpha * state + (1.0 - alpha) * value
        smoothed[index] = state
    return smoothed, state


def _state_features(
    states: np.ndarray,
    history_states: np.ndarray,
    *,
    include_fidelity_contrast: bool = False,
    include_normalized_rho_similarity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return density-matrix, temporal-fidelity and rolling-state features."""
    count = states.shape[0]
    gram = np.einsum("brd,bsd->brs", states, states, optimize=True) / 3.0
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    entropy = -np.sum(eigenvalues * np.log(np.maximum(eigenvalues, 1e-15)), axis=1) / np.log(3.0)
    purity = np.sum(eigenvalues * eigenvalues, axis=1)

    matrices = states.reshape(count, 3, 4, 4)
    reduced = np.einsum("brik,brjk->brij", matrices, matrices, optimize=True)
    reduced_eigenvalues = np.maximum(np.linalg.eigvalsh(reduced), 0.0)
    entropies = -np.sum(
        reduced_eigenvalues * np.log(np.maximum(reduced_eigenvalues, 1e-15)), axis=2
    ) / np.log(4.0)
    entanglement = np.stack([entropies.mean(axis=1), entropies.std(axis=1)], axis=1)

    probabilities = states * states
    measured = np.einsum("brd,dq->brq", probabilities, _Z_OBSERVABLES, optimize=True)
    observables = np.concatenate([measured.mean(axis=1), measured.std(axis=1)], axis=1)

    overlap_values = np.ones(count, dtype=np.float64)
    if history_states.shape[0]:
        first = np.einsum("rd,sd->rs", states[0], history_states[-1], optimize=True)
        overlap_values[0] = np.sum(first * first) / 9.0
    if count > 1:
        adjacent = np.einsum("brd,bsd->brs", states[1:], states[:-1], optimize=True)
        overlap_values[1:] = np.sum(adjacent * adjacent, axis=(1, 2), keepdims=False) / 9.0
    overlap = overlap_values[:, None]

    normalized_similarity = None
    if include_normalized_rho_similarity:
        previous_states = np.empty_like(states)
        previous_states[0] = history_states[-1] if history_states.shape[0] else states[0]
        if count > 1:
            previous_states[1:] = states[:-1]
        previous_gram = np.einsum(
            "brd,bsd->brs", previous_states, previous_states, optimize=True
        ) / 3.0
        previous_purity = np.sum(previous_gram * previous_gram, axis=(1, 2))
        cross_overlap = np.einsum(
            "brd,bsd->brs", states, previous_states, optimize=True
        )
        numerator = np.sum(cross_overlap * cross_overlap, axis=(1, 2)) / 9.0
        normalized_similarity = np.clip(
            numerator / (np.sqrt(purity * previous_purity) + 1e-12),
            0.0,
            1.0,
        )[:, None]

    # Same-radar projective fidelity separates each sensor's temporal change
    # from the cross-radar mixture used by rho_temporal_overlap.
    context = np.concatenate([history_states, states], axis=0)
    history_count = history_states.shape[0]
    if history_count < 7:
        padding = np.repeat(states[:1], 7 - history_count, axis=0)
        context = np.concatenate([padding, context], axis=0)
        history_count = 7
    temporal_fidelities = []
    for lag in (1, 3):
        previous = context[history_count + np.arange(count) - lag]
        dot = np.sum(states * previous, axis=2)
        temporal_fidelities.append(dot * dot)

    # A short-window density matrix measures how much each radar's motion
    # profile explores range-state space over the recent causal history.
    rolling_summaries = []
    end_indices = history_count + np.arange(count)
    for window_length in (4, 8):
        indices = end_indices[:, None] - window_length + 1 + np.arange(window_length)[None, :]
        state_windows = context[indices]
        gram = np.einsum("bkrl,bjrl->brkj", state_windows, state_windows, optimize=True)
        gram /= float(window_length)
        window_eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
        window_entropy = -np.sum(
            window_eigenvalues * np.log(np.maximum(window_eigenvalues, 1e-15)), axis=2
        ) / np.log(float(window_length))
        window_purity = np.sum(window_eigenvalues * window_eigenvalues, axis=2)
        rolling_summaries.append(
            np.stack(
                [
                    window_entropy.mean(axis=1),
                    window_entropy.std(axis=1),
                    window_purity.mean(axis=1),
                    window_purity.std(axis=1),
                ],
                axis=1,
            )
        )

    temporal_features = list(temporal_fidelities)
    if normalized_similarity is not None:
        temporal_features.append(normalized_similarity)
    if include_fidelity_contrast:
        temporal_features.append(temporal_fidelities[0] - temporal_fidelities[1])
    result = np.concatenate(
        [
            eigenvalues,
            entropy[:, None],
            purity[:, None],
            overlap,
            entanglement,
            observables,
            *temporal_features,
            *rolling_summaries,
        ],
        axis=1,
    )
    return result, context[-7:].copy()


def _hybrid_state_features(
    states: np.ndarray,
    history_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure a 5-qubit motion/static state and its causal evolution."""
    count = states.shape[0]
    gram = np.einsum("brd,bsd->brs", states, states, optimize=True) / 3.0
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    entropy = -np.sum(eigenvalues * np.log(np.maximum(eigenvalues, 1e-15)), axis=1) / np.log(3.0)
    purity = np.sum(eigenvalues * eigenvalues, axis=1)

    probabilities = states * states
    flag_z = probabilities[:, :, :16].sum(axis=2) - probabilities[:, :, 16:].sum(axis=2)
    matrices = states.reshape(count, 3, 2, 16)
    reduced_flag = np.einsum("brik,brjk->brij", matrices, matrices, optimize=True)
    flag_eigenvalues = np.maximum(np.linalg.eigvalsh(reduced_flag), 0.0)
    flag_entropy = -np.sum(
        flag_eigenvalues * np.log(np.maximum(flag_eigenvalues, 1e-15)), axis=2
    ) / np.log(2.0)

    context = np.concatenate([history_states, states], axis=0)
    history_count = history_states.shape[0]
    if history_count < 3:
        padding = np.repeat(states[:1], 3 - history_count, axis=0)
        context = np.concatenate([padding, context], axis=0)
        history_count = 3
    temporal_fidelities = []
    for lag in (1, 3):
        previous = context[history_count + np.arange(count) - lag]
        dot = np.sum(states * previous, axis=2)
        temporal_fidelities.append(dot * dot)

    result = np.concatenate(
        [
            entropy[:, None],
            purity[:, None],
            flag_z.mean(axis=1, keepdims=True),
            flag_z.std(axis=1, keepdims=True),
            flag_entropy.mean(axis=1, keepdims=True),
            flag_entropy.std(axis=1, keepdims=True),
            *temporal_fidelities,
        ],
        axis=1,
    )
    return result, context[-3:].copy()


def _block_features(
    differences: np.ndarray,
    history_states: np.ndarray,
    ema_states: dict[float, np.ndarray],
    step_s: float,
    *,
    raw_blocks: np.ndarray | None = None,
    background_state: np.ndarray | None = None,
    static_tau_s: float | None = None,
    static_motion_gamma: float = 1.0,
    include_fidelity_contrast: bool = False,
    include_normalized_rho_similarity: bool = False,
    include_quantum_reservoir: bool = False,
    quantum_reservoir_state: np.ndarray | None = None,
    quantum_reservoir_memory_tau_s: float = 0.8,
    quantum_reservoir_phase_scale: float = np.pi / 2.0,
    quantum_reservoir_entangler_angle: float = np.pi / 8.0,
    hybrid_history: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[float, np.ndarray],
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
]:
    blocks = differences.shape[0] // 5
    shaped = differences[: blocks * 5].reshape(blocks, 5, 3, 368).astype(np.float64, copy=False)
    motion = np.sqrt(np.mean(shaped * shaped, axis=1))
    pooled = motion.reshape(blocks, 3, 16, 23).mean(axis=-1)
    pooled = np.maximum(pooled, 0.0)

    energy = pooled.sum(axis=2)
    energy_total = np.maximum(energy.sum(axis=1, keepdims=True), 1e-12)
    amplitudes = np.sqrt(pooled / np.maximum(energy[:, :, None], 1e-12))
    silent = energy <= 1e-12
    if np.any(silent):
        amplitudes[silent] = 0.25

    density, next_history_states = _state_features(
        amplitudes,
        history_states,
        include_fidelity_contrast=include_fidelity_contrast,
        include_normalized_rho_similarity=include_normalized_rho_similarity,
    )
    radar_share = energy / energy_total
    log_energy = np.log(np.maximum(energy, 1e-12))
    classical = np.concatenate(
        [
            np.log(np.maximum(pooled, 1e-12)).reshape(blocks, 48),
            log_energy,
            radar_share,
            np.log(energy_total),
        ],
        axis=1,
    )

    ema_features = []
    next_ema: dict[float, np.ndarray] = {}
    for tau in (0.4, 1.2):
        smoothed, next_ema[tau] = _causal_ema(log_energy, tau, step_s, ema_states.get(tau))
        ema_features.append(smoothed)

    static_features: list[np.ndarray] = []
    reservoir_features: list[np.ndarray] = []
    next_reservoir_state = quantum_reservoir_state
    next_background = background_state
    next_hybrid_history = (
        np.empty((0, 3, 32), dtype=np.float64) if hybrid_history is None else hybrid_history
    )
    if static_tau_s is not None:
        if raw_blocks is None or raw_blocks.shape != (blocks, 5, 3, 368):
            raise ValueError("static-state features require raw blocks shaped (blocks, 5, 3, 368)")
        if static_tau_s <= 0.0 or static_motion_gamma <= 0.0:
            raise ValueError("static background tau and motion gamma must be positive")
        background = (
            raw_blocks[0, 0].astype(np.float64, copy=True)
            if background_state is None
            else background_state.astype(np.float64, copy=True)
        )
        alpha = float(np.exp(-step_s / static_tau_s))
        block_means = raw_blocks.astype(np.float64, copy=False).mean(axis=1)
        updated_backgrounds, _ = lfilter(
            [1.0 - alpha],
            [1.0, -alpha],
            block_means,
            axis=0,
            zi=alpha * background[None, :, :],
        )
        previous_backgrounds = np.concatenate(
            [background[None, :, :], updated_backgrounds[:-1]], axis=0
        )
        static_contrast = np.mean(
            np.abs(raw_blocks.astype(np.float64, copy=False) - previous_backgrounds[:, None, :, :]),
            axis=1,
        )
        next_background = updated_backgrounds[-1]
        static_pooled = static_contrast.reshape(blocks, 3, 16, 23).mean(axis=-1)
        static_pooled = np.maximum(static_pooled, 0.0)
        static_energy = static_pooled.sum(axis=2)
        static_amplitudes = np.sqrt(static_pooled / np.maximum(static_energy[:, :, None], 1e-12))
        static_silent = static_energy <= 1e-12
        if np.any(static_silent):
            static_amplitudes[static_silent] = 0.25
        eta = static_energy / np.maximum(
            static_energy + static_motion_gamma * energy,
            1e-12,
        )
        eta = np.clip(eta, 0.0, 1.0)
        hybrid_states = np.concatenate(
            [
                amplitudes * np.sqrt(1.0 - eta[:, :, None]),
                static_amplitudes * np.sqrt(eta[:, :, None]),
            ],
            axis=2,
        )
        hybrid_features, next_hybrid_history = _hybrid_state_features(
            hybrid_states, next_hybrid_history
        )
        static_features = [
            np.log(np.maximum(static_energy, 1e-12)),
            eta,
            hybrid_features,
        ]

    if include_quantum_reservoir:
        reservoir_batch, next_reservoir_state = _quantum_reservoir_features(
            amplitudes,
            quantum_reservoir_state,
            step_s,
            memory_tau_s=quantum_reservoir_memory_tau_s,
            phase_scale=quantum_reservoir_phase_scale,
            entangler_angle=quantum_reservoir_entangler_angle,
        )
        reservoir_features = [reservoir_batch]

    features = np.concatenate(
        [classical, density, *ema_features, *static_features, *reservoir_features], axis=1
    )
    return (
        features.astype(np.float32, copy=False),
        next_history_states,
        next_ema,
        next_background,
        next_hybrid_history,
        next_reservoir_state,
    )


def _chunks(frame_count: int, frames_per_chunk: int) -> Iterator[tuple[int, int]]:
    for start in range(0, frame_count, frames_per_chunk):
        yield start, min(start + frames_per_chunk, frame_count)


def extract_stream_features(
    path: Path,
    *,
    rate_hz: float = 50.0,
    pool_frames: int = 5,
    frames_per_chunk: int = 5_000,
    static_background_tau_s: float | None = None,
    static_motion_gamma: float = 1.0,
    include_fidelity_contrast: bool = False,
    include_normalized_rho_similarity: bool = False,
    include_quantum_reservoir: bool = False,
    quantum_reservoir_memory_tau_s: float = 0.8,
    quantum_reservoir_phase_scale: float = np.pi / 2.0,
    quantum_reservoir_entangler_angle: float = np.pi / 8.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract motion and quantum observables from one continuous stream.

    The function reads only the supplied HDF5 path, maintains state across
    chunks, never uses future frames, and does not consult stream composition.
    """
    if pool_frames != 5:
        raise ValueError("the quantum feature map currently pools exactly five frames")
    if frames_per_chunk < pool_frames:
        raise ValueError("frames_per_chunk must be at least pool_frames")
    frames_per_chunk -= frames_per_chunk % pool_frames

    with h5py.File(path, "r") as handle:
        if "cir" not in handle:
            raise ValueError(f"{path} has no 'cir' dataset")
        dataset = handle["cir"]
        if dataset.ndim != 3 or dataset.shape[0] != 3 or dataset.shape[2] != 368:
            raise ValueError(f"{path} has unexpected CIR shape {dataset.shape}; expected (3, T, 368)")
        frame_count = int(dataset.shape[1])
        duration_s = frame_count / rate_hz
        previous_raw: np.ndarray | None = None
        history_states = np.empty((0, 3, 16), dtype=np.float64)
        hybrid_history = np.empty((0, 3, 32), dtype=np.float64)
        quantum_reservoir_state: np.ndarray | None = None
        background_state: np.ndarray | None = None
        ema_states: dict[float, np.ndarray] = {}
        pending = np.empty((0, 3, 368), dtype=np.float32)
        pending_raw = np.empty((0, 3, 368), dtype=np.float32)
        batches: list[np.ndarray] = []

        for start, stop in _chunks(frame_count, frames_per_chunk):
            raw = np.moveaxis(dataset[:, start:stop, :], 1, 0).astype(np.float32, copy=False)
            if previous_raw is None:
                previous_raw = raw[0].copy()
            differences = np.diff(np.concatenate([previous_raw[None, ...], raw], axis=0), axis=0)
            previous_raw = raw[-1].copy()
            if pending.shape[0]:
                differences = np.concatenate([pending, differences], axis=0)
            complete = differences.shape[0] - differences.shape[0] % pool_frames
            if complete:
                raw_sequence = (
                    np.concatenate([pending_raw, raw], axis=0)
                    if pending_raw.shape[0]
                    else raw
                )
                raw_complete = raw_sequence.shape[0] - raw_sequence.shape[0] % pool_frames
                if raw_complete != complete:
                    raise ValueError("raw frames and frame differences lost block alignment")
                raw_blocks = raw_sequence[:raw_complete].reshape(-1, pool_frames, 3, 368)
                (
                    feature_batch,
                    history_states,
                    ema_states,
                    background_state,
                    hybrid_history,
                    quantum_reservoir_state,
                ) = _block_features(
                    differences[:complete],
                    history_states,
                    ema_states,
                    pool_frames / rate_hz,
                    raw_blocks=raw_blocks if static_background_tau_s is not None else None,
                    background_state=background_state,
                    static_tau_s=static_background_tau_s,
                    static_motion_gamma=static_motion_gamma,
                    include_fidelity_contrast=include_fidelity_contrast,
                    include_normalized_rho_similarity=include_normalized_rho_similarity,
                    include_quantum_reservoir=include_quantum_reservoir,
                    quantum_reservoir_state=quantum_reservoir_state,
                    quantum_reservoir_memory_tau_s=quantum_reservoir_memory_tau_s,
                    quantum_reservoir_phase_scale=quantum_reservoir_phase_scale,
                    quantum_reservoir_entangler_angle=quantum_reservoir_entangler_angle,
                    hybrid_history=hybrid_history,
                )
                batches.append(feature_batch)
                pending_raw = raw_sequence[raw_complete:]
            pending = differences[complete:]

    if not batches:
        raise ValueError(f"{path} has too few frames to form one feature step")
    features = np.concatenate(batches, axis=0)
    step_s = pool_frames / rate_hz
    centers_s = (np.arange(features.shape[0], dtype=np.float64) + 0.5) * step_s
    return features, centers_s, duration_s
