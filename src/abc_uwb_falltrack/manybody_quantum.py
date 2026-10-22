"""Đặc trưng many-body quantum được mô phỏng chính xác bằng đại số phức.

CIR là dữ liệu cổ điển được mã hóa thành ba spin 1/2, không phải bằng chứng
radar tạo trạng thái lượng tử hay lợi thế tính toán lượng tử. Module chỉ đọc
``cir`` và ``t`` của HDF5 được truyền vào; không đọc nhãn hoặc composition.

Tín hiệu giải tích dùng đầu ra cuối của FFT/Hilbert trên cửa sổ *quá khứ* hữu
hạn. Đây là xấp xỉ causal của tín hiệu giải tích, không phải Hilbert lý tưởng
hai phía. Trạng thái joint là tensor product phức rồi tiến hóa dưới XXZ đối
xứng. Memory là kênh CPTP: hỗn hợp của trạng thái mới và unitary evolution của
ma trận mật độ trước, thay cho phép cộng wavefunction không tuyến tính.

Entropy của reduced mixed state được gọi là mixedness. Chỉ Schmidt/Meyer-
Wallach của pure injected state và negativity của reduced pair được gọi là
entanglement. Coherence L1 phụ thuộc computational basis và được đặt tên rõ.
QFI/QGT theo thời gian là ước lượng finite-difference, không phải đạo hàm exact.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter


_EPS = 1e-12
_SUMMARY = ("mean", "std", "min", "max")
_PAIRS = ((0, 1), (0, 2), (1, 2))


def _geometry_lags(config: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in config.get("geometry_lags_s", config.get("lags_s", (0.06, 0.16, 0.4))))


def _symmetric(values: np.ndarray) -> np.ndarray:
    """Mean/std/min/max over radar or unordered-pair axis 1."""
    return np.stack(
        (values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1)),
        axis=-1,
    ).reshape(values.shape[0], -1)


def tensor_product_states(states: np.ndarray) -> np.ndarray:
    """Batched, genuine Kronecker product of states shaped (..., 3, 2)."""
    states = np.asarray(states, dtype=np.complex128)
    if states.shape[-2:] != (3, 2):
        raise ValueError("local spin states must have shape (..., 3, 2)")
    return np.einsum(
        "...i,...j,...k->...ijk", states[..., 0, :], states[..., 1, :], states[..., 2, :]
    ).reshape(*states.shape[:-2], 8)


def partial_trace(density: np.ndarray, keep: tuple[int, ...]) -> np.ndarray:
    """Trace all spins except ``keep``; dimensions are exactly (2, 2, 2).

    Supports arbitrary leading batch axes. Axis order in ``keep`` is respected.
    """
    density = np.asarray(density, dtype=np.complex128)
    if density.shape[-2:] != (8, 8):
        raise ValueError("joint density must have shape (..., 8, 8)")
    if len(set(keep)) != len(keep) or any(index not in (0, 1, 2) for index in keep):
        raise ValueError("keep must contain distinct spin indices in {0,1,2}")
    batch_shape = density.shape[:-2]
    traced = tuple(index for index in range(3) if index not in keep)
    batch_count = len(batch_shape)
    axes = (
        list(range(batch_count))
        + [batch_count + index for index in keep]
        + [batch_count + index for index in traced]
        + [batch_count + 3 + index for index in keep]
        + [batch_count + 3 + index for index in traced]
    )
    dim_keep, dim_traced = 2 ** len(keep), 2 ** len(traced)
    reshaped = density.reshape(*batch_shape, 2, 2, 2, 2, 2, 2).transpose(axes)
    reshaped = reshaped.reshape(*batch_shape, dim_keep, dim_traced, dim_keep, dim_traced)
    return np.trace(reshaped, axis1=-3, axis2=-1)


def pair_negativity(density: np.ndarray) -> np.ndarray:
    """(norm1(partial-transpose)-1)/2 for two-spin density matrices.

    A positive value certifies bipartite entanglement; for 2x2 this PPT test is
    necessary and sufficient. Zero pair negativity does not exclude GHZ-like
    genuine tripartite entanglement in the joint state.
    """
    density = np.asarray(density, dtype=np.complex128)
    if density.shape[-2:] != (4, 4):
        raise ValueError("pair density must have shape (..., 4, 4)")
    pt = density.reshape(*density.shape[:-2], 2, 2, 2, 2).swapaxes(-3, -1)
    eigenvalues = np.linalg.eigvalsh(pt.reshape(density.shape))
    return np.maximum(-np.minimum(eigenvalues, 0.0).sum(axis=-1), 0.0)


def bures_distance_squared(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Exact mixed-state Bures squared distance 2(1-Tr sqrt(sqrt(rho)sigma sqrt(rho)))."""
    eigenvalues, vectors = np.linalg.eigh(first)
    root = (vectors * np.sqrt(np.maximum(eigenvalues, 0.0))[..., None, :]) @ vectors.conj().swapaxes(-1, -2)
    middle = root @ second @ root
    middle = 0.5 * (middle + middle.conj().swapaxes(-1, -2))
    root_fidelity = np.sqrt(np.maximum(np.linalg.eigvalsh(middle), 0.0)).sum(axis=-1)
    return 2.0 * (1.0 - np.clip(root_fidelity, 0.0, 1.0))


def symmetric_hamiltonian(
    *, j_xy: float = 2.5, j_z: float = 1.3, field_x: float = 1.1, field_z: float = 0.5
) -> np.ndarray:
    """All-to-all XXZ Hamiltonian with identical fields, in angular-frequency units."""
    identity = np.eye(2, dtype=np.complex128)
    paulis = (
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    )
    embedded = np.empty((3, 3, 8, 8), dtype=np.complex128)
    for spin in range(3):
        for axis, operator in enumerate(paulis):
            factors = [operator if index == spin else identity for index in range(3)]
            embedded[spin, axis] = np.kron(np.kron(factors[0], factors[1]), factors[2])
    hamiltonian = field_x * embedded[:, 0].sum(axis=0) + field_z * embedded[:, 2].sum(axis=0)
    for first, second in _PAIRS:
        hamiltonian += j_xy * (
            embedded[first, 0] @ embedded[second, 0] + embedded[first, 1] @ embedded[second, 1]
        ) + j_z * embedded[first, 2] @ embedded[second, 2]
    return hamiltonian


def _eigen_summary(density: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues = np.maximum(np.linalg.eigvalsh(density), 0.0)
    eigenvalues /= np.maximum(eigenvalues.sum(axis=-1, keepdims=True), _EPS)
    purity = np.square(eigenvalues).sum(axis=-1)
    entropy = -(eigenvalues * np.log(np.maximum(eigenvalues, _EPS))).sum(axis=-1)
    abs_density = np.abs(density)
    coherence = abs_density.sum(axis=(-2, -1)) - np.abs(np.diagonal(density, axis1=-2, axis2=-1)).sum(axis=-1)
    return eigenvalues, purity, entropy, np.maximum(coherence, 0.0)


def _density_observables(density: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    eig, purity, entropy, coherence = _eigen_summary(density)
    chunks = [eig, purity[:, None], entropy[:, None], coherence[:, None]]
    names = [f"{prefix}_global_eigenvalue_{index}" for index in range(8)]
    names += [f"{prefix}_global_purity", f"{prefix}_global_mixedness_entropy", f"{prefix}_global_basis_l1_coherence"]
    singles = np.stack([partial_trace(density, (spin,)) for spin in range(3)], axis=1)
    _, local_purity, local_entropy, local_coherence = _eigen_summary(singles)
    for value, name in (
        (local_purity, "local_purity"), (local_entropy, "local_mixedness_entropy"),
        (local_coherence, "local_basis_l1_coherence"),
    ):
        chunks.append(_symmetric(value))
        names.extend(f"{prefix}_{name}_{summary}" for summary in _SUMMARY)
    pairs = np.stack([partial_trace(density, pair) for pair in _PAIRS], axis=1)
    pair_eig, pair_purity, pair_entropy, pair_coherence = _eigen_summary(pairs)
    for value, name in (
        (pair_purity, "pair_purity"), (pair_entropy, "pair_mixedness_entropy"),
        (pair_coherence, "pair_basis_l1_coherence"), (pair_negativity(pairs), "pair_entanglement_negativity"),
    ):
        chunks.append(_symmetric(value))
        names.extend(f"{prefix}_{name}_{summary}" for summary in _SUMMARY)
    chunks.extend([pair_eig.mean(axis=1), pair_eig.std(axis=1)])
    for summary in ("mean", "std"):
        names.extend(f"{prefix}_pair_eigenvalue_{index}_{summary}" for index in range(4))
    return np.concatenate(chunks, axis=1), names


def _causal_hilbert_kernel(window: int) -> np.ndarray:
    """FIR implementing the current endpoint of an analytic sliding-window FFT."""
    multiplier = np.zeros(window, dtype=np.complex128)
    positive_end = (window + 1) // 2
    multiplier[1:positive_end] = -1j
    multiplier[window // 2 + 1:] = 1j
    return np.fft.ifft(multiplier).real


class _CausalEncoding:
    def __init__(self, config: dict[str, Any]):
        self.rate_hz = float(config.get("rate_hz", 50.0))
        self.step_s = 1.0 / self.rate_hz
        self.alpha = float(np.exp(-self.step_s / float(config.get("background_tau_s", 2.0))))
        self.energy_alpha = float(np.exp(-self.step_s / float(config.get("energy_scale_tau_s", 2.0))))
        self.kernel = _causal_hilbert_kernel(int(config.get("analytic_window_frames", 32)))
        self.background_zi: np.ndarray | None = None
        self.hilbert_zi = np.zeros((len(self.kernel) - 1, 3, 368), dtype=np.float64)
        self.energy_zi: np.ndarray | None = None
        self.previous_z: np.ndarray | None = None
        self.previous_center: np.ndarray | None = None
        self.previous_active: np.ndarray | None = None
        self.coordinate = (np.arange(368, dtype=np.float64) + 0.5) / 368.0 - 0.5
        self.powers = np.stack([self.coordinate ** order for order in range(1, 5)], axis=1)
        self.wavenumbers = np.array([1.0, 2.0, 4.0])
        self.fourier = np.exp(-2j * np.pi * self.coordinate[:, None] * self.wavenumbers[None, :])

    def transform(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if self.background_zi is None:
            self.background_zi = self.alpha * raw[0:1].astype(np.float64)
        background, self.background_zi = lfilter(
            [1.0 - self.alpha], [1.0, -self.alpha], raw, axis=0, zi=self.background_zi
        )
        residual = raw - background
        quadrature, self.hilbert_zi = lfilter(self.kernel, [1.0], residual, axis=0, zi=self.hilbert_zi)
        z = residual.astype(np.complex128) + 1j * quadrature
        power = np.square(z.real) + np.square(z.imag)
        energy = power.sum(axis=-1)
        safe_energy = np.maximum(energy, _EPS)
        moments = np.einsum("trb,bk->trk", power, self.powers, optimize=True) / safe_energy[..., None]
        center = moments[..., 0]
        variance = np.maximum(moments[..., 1] - center ** 2, 0.0)
        dispersion = np.sqrt(variance)
        central3 = moments[..., 2] - 3.0 * center * moments[..., 1] + 2.0 * center ** 3
        central4 = moments[..., 3] - 4.0 * center * moments[..., 2] + 6.0 * center ** 2 * moments[..., 1] - 3.0 * center ** 4
        skew = np.clip(central3 / np.maximum(dispersion ** 3, 1e-9), -10.0, 10.0)
        kurtosis = np.clip(central4 / np.maximum(variance ** 2, 1e-9), 0.0, 30.0)
        previous_center = np.concatenate([
            center[0:1] if self.previous_center is None else self.previous_center[None, :], center[:-1]
        ], axis=0)
        center_velocity = (center - previous_center) / self.step_s
        active = energy > _EPS
        previous_active = np.concatenate([
            active[0:1] if self.previous_active is None else self.previous_active[None, :], active[:-1]
        ], axis=0)
        # A silent state has no defined center. Do not turn appearance of energy
        # into a velocity measured from the arbitrary coordinate origin.
        center_velocity[~(active & previous_active)] = 0.0
        self.previous_center = center[-1].copy()
        self.previous_active = active[-1].copy()
        previous_z = np.concatenate([
            z[0:1] if self.previous_z is None else self.previous_z[None, ...], z[:-1]
        ], axis=0)
        phase_cross = (z * previous_z.conj()).sum(axis=-1)
        phase_denominator = np.sqrt(safe_energy * np.maximum(np.square(np.abs(previous_z)).sum(axis=-1), _EPS))
        phase_alignment = np.clip(np.abs(phase_cross) / phase_denominator, 0.0, 1.0)
        phase_velocity = np.angle(phase_cross) / self.step_s
        self.previous_z = z[-1].copy()
        # The center phase removes range translation; the center itself is never returned.
        spectral = np.einsum("trb,bk->trk", z, self.fourier, optimize=True)
        spectral *= np.exp(2j * np.pi * center[..., None] * self.wavenumbers)
        spectral /= np.sqrt(368.0 * safe_energy[..., None])
        if self.energy_zi is None:
            self.energy_zi = self.energy_alpha * energy[0:1]
        energy_scale, self.energy_zi = lfilter(
            [1.0 - self.energy_alpha], [1.0, -self.energy_alpha], energy, axis=0, zi=self.energy_zi
        )
        energy_contrast = np.clip(np.log((energy + _EPS) / (energy_scale + _EPS)), -10.0, 10.0)
        relative_phase = np.angle(spectral[..., 0])
        phase_unit = np.exp(1j * relative_phase)
        # Same encoding on all sensors. The population uses only translation invariants.
        excitation = 1.0 / (1.0 + np.exp(-np.clip(0.8 * energy_contrast + 2.0 * dispersion - 0.4, -20.0, 20.0)))
        local_states = np.stack([np.sqrt(1.0 - excitation), np.sqrt(excitation) * phase_unit], axis=-1)
        local_states[energy <= _EPS] = np.array([1.0, 0.0], dtype=np.complex128)
        descriptors = np.stack([
            np.log1p(energy / 368.0), energy_contrast, dispersion, skew, kurtosis,
            center_velocity, phase_velocity, phase_alignment,
            np.abs(spectral[..., 0]), np.abs(spectral[..., 1]), np.abs(spectral[..., 2]),
            np.cos(relative_phase), np.sin(relative_phase),
        ], axis=-1)
        silent = energy <= _EPS
        descriptors[silent] = 0.0
        descriptor_names = [
            "analytic_log_energy", "causal_energy_contrast", "central_range_dispersion", "central_range_skewness",
            "central_range_kurtosis", "range_center_velocity", "analytic_phase_velocity", "analytic_phase_alignment",
            "centered_complex_spectral_abs_k1", "centered_complex_spectral_abs_k2", "centered_complex_spectral_abs_k4",
            "centered_complex_phase_cos", "centered_complex_phase_sin",
        ]
        return local_states, descriptors, descriptor_names


class _QuantumReservoir:
    def __init__(self, config: dict[str, Any], step_s: float):
        coupling = float(config.get("xxz_coupling", 2.5))
        hamiltonian = symmetric_hamiltonian(
            j_xy=float(config.get("hamiltonian_j_xy", coupling)),
            j_z=float(config.get("hamiltonian_j_z", coupling * float(config.get("zz_anisotropy", 1.3 / 2.5)))),
            field_x=float(config.get("hamiltonian_field_x", config.get("transverse_field", 1.1))),
            field_z=float(config.get("hamiltonian_field_z", 0.5)),
        )
        self.unitary = expm(-1j * hamiltonian * float(config.get("reservoir_evolution_s", config.get("evolution_dt", 0.35))))
        self.memory_unitary = expm(-1j * hamiltonian * step_s)
        self.memory = float(np.exp(-step_s / float(config.get("memory_tau_s", 0.25))))
        self.state = np.eye(8, dtype=np.complex128) / 8.0
        self.taus = tuple(float(value) for value in config.get("density_smoothing_tau_s", config.get("smoothing_taus_s", (0.1, 0.25, 0.5))))
        self.smoothing_alpha = np.exp(-step_s / np.array(self.taus))
        self.smoothed = [self.state.copy() for _ in self.taus]
        self.step_s = step_s
        self.requested_lags = _geometry_lags(config)
        self.mixed_lags = tuple(max(1, int(np.floor(value / step_s + 0.5))) for value in self.requested_lags)
        self.history: deque[np.ndarray] = deque(maxlen=max(self.mixed_lags))
        self.trace_error = 0.0
        self.hermiticity_error = 0.0
        self.min_eigenvalue = 1.0
        self.unitary_error = float(np.max(np.abs(self.unitary.conj().T @ self.unitary - np.eye(8))))

    def transform(self, injections: np.ndarray) -> tuple[np.ndarray, list[str]]:
        count = len(injections)
        densities = np.empty((count, 8, 8), dtype=np.complex128)
        smoothed = np.empty((len(self.taus), count, 8, 8), dtype=np.complex128)
        previous_by_lag = [np.empty_like(densities) for _ in self.mixed_lags]
        for index, injection in enumerate(injections):
            previous_state = self.state
            self.state = (
                self.memory * (self.memory_unitary @ previous_state @ self.memory_unitary.conj().T)
                + (1.0 - self.memory) * injection
            )
            # Floating-point symmetrization; no eigenvalue projection or data-dependent renormalization.
            self.state = 0.5 * (self.state + self.state.conj().T)
            densities[index] = self.state
            for lag_index, lag in enumerate(self.mixed_lags):
                previous_by_lag[lag_index][index] = self.history[-lag] if len(self.history) >= lag else self.state
            self.history.append(self.state.copy())
            for scale, alpha in enumerate(self.smoothing_alpha):
                self.smoothed[scale] = alpha * self.smoothed[scale] + (1.0 - alpha) * self.state
                smoothed[scale, index] = self.smoothed[scale]
        self.trace_error = max(self.trace_error, float(np.max(np.abs(np.trace(densities, axis1=-2, axis2=-1) - 1.0))))
        self.hermiticity_error = max(self.hermiticity_error, float(np.max(np.abs(densities - densities.conj().swapaxes(-1, -2)))))
        self.min_eigenvalue = min(self.min_eigenvalue, float(np.linalg.eigvalsh(densities).min()))
        chunks: list[np.ndarray] = []
        names: list[str] = []
        for scale, tau in enumerate(self.taus):
            values, labels = _density_observables(smoothed[scale], f"rho_tau{tau:g}s")
            chunks.append(values)
            names.extend(labels)
        for lag, previous in zip(self.mixed_lags, previous_by_lag):
            lag_s = lag * self.step_s
            distance = bures_distance_squared(densities, previous)
            chunks.append(np.stack([distance, 4.0 * distance / lag_s ** 2], axis=1))
            names.extend([f"mixed_bures_squared_lag{lag_s:g}s", f"mixed_qfi_fd_bures_lag{lag_s:g}s"])
        return np.concatenate(chunks, axis=1), names


def _pure_observables(
    states: np.ndarray, history: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, list[str], np.ndarray]:
    raw_step_s = 1.0 / float(config.get("rate_hz", 50.0))
    lags = [max(1, int(np.floor(value / raw_step_s + 0.5))) for value in _geometry_lags(config)]
    context = np.concatenate([history, states], axis=0)
    history_size = len(history)
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for lag in lags:
        previous_index = history_size + np.arange(len(states)) - lag
        previous = context[np.maximum(previous_index, 0)]
        overlap_abs = np.clip(np.abs(np.sum(states.conj() * previous, axis=1)), 0.0, 1.0)
        lag_s = lag * raw_step_s
        bures = 2.0 * (1.0 - overlap_abs)
        qgt = np.maximum(1.0 - overlap_abs ** 2, 0.0) / lag_s ** 2
        chunks.append(np.stack([overlap_abs ** 2, bures, qgt, 4.0 * qgt], axis=1))
        names.extend([
            f"pure_fidelity_lag{lag_s:g}s", f"pure_bures_squared_lag{lag_s:g}s",
            f"pure_qgt_projected_fd_lag{lag_s:g}s", f"pure_qfi_fd_lag{lag_s:g}s",
        ])
    density = states[:, :, None] * states.conj()[:, None, :]
    singles = np.stack([partial_trace(density, (spin,)) for spin in range(3)], axis=1)
    eigenvalues, purity, entropy, _ = _eigen_summary(singles)
    chunks.extend([
        (2.0 * (1.0 - purity.mean(axis=1)))[:, None], _symmetric(entropy), _symmetric(eigenvalues[..., -1])
    ])
    names.append("pure_meyer_wallach_global_entanglement")
    names.extend(f"pure_schmidt_entropy_{summary}" for summary in _SUMMARY)
    names.extend(f"pure_schmidt_largest_eigenvalue_{summary}" for summary in _SUMMARY)
    return np.concatenate(chunks, axis=1), names, context[-max(lags):].copy()


def extract_manybody_features(
    path: Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float, list[str], dict[str, Any]]:
    """Extract one stream without reading labels, split metadata or composition.

    Returns ``features float32 (N,F), centers float64 (N,), duration_s, names,
    diagnostics``. ``centers`` are *right-edge availability times*, relative to
    the stream's first timestamp. A pooled row uses only samples at or before
    that returned time; no midpoint look-ahead is hidden. Incomplete final pools
    are discarded. All preprocessing is reset per stream and state persists
    across I/O chunks. No fitted statistics are shared across streams.
    """
    config = dict(config)
    rate_hz = float(config.get("rate_hz", 50.0))
    pool_frames = int(config.get("pool_frames", 5))
    window = int(config.get("analytic_window_frames", 32))
    chunk_frames = int(config.get("frames_per_chunk", 2500))
    if rate_hz <= 0 or pool_frames < 1 or window < 4 or chunk_frames < pool_frames:
        raise ValueError("rate_hz/pool/chunk must be positive; analytic_window_frames >=4")
    if any(float(config.get(key, default)) <= 0.0 for key, default in (
        ("background_tau_s", 2.0), ("energy_scale_tau_s", 2.0), ("memory_tau_s", 0.25)
    )):
        raise ValueError("all memory/background time constants must be positive")
    if any(float(value) <= 0 for value in config.get("density_smoothing_tau_s", config.get("smoothing_taus_s", (0.1, 0.25, 0.5)))):
        raise ValueError("density smoothing time constants must be positive")
    if not _geometry_lags(config) or any(value <= 0.0 for value in _geometry_lags(config)):
        raise ValueError("geometry lags must be nonempty and positive")
    chunk_frames -= chunk_frames % pool_frames
    step_s = pool_frames / rate_hz
    encoder = _CausalEncoding(config)
    reservoir = _QuantumReservoir(config, step_s)
    pure_history = np.empty((0, 8), dtype=np.complex128)
    batches: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    names: list[str] = []
    state_norm_error = 0.0
    injection_trace_error = 0.0
    maximum_timestamp_grid_error = 0.0
    with h5py.File(Path(path), "r") as handle:
        if "cir" not in handle:
            raise ValueError("HDF5 must contain cir")
        cir = handle["cir"]
        if cir.ndim != 3 or cir.shape[0] != 3 or cir.shape[2] != 368:
            raise ValueError(f"expected CIR shape (3,T,368), got {cir.shape}")
        if np.issubdtype(cir.dtype, np.complexfloating):
            raise ValueError("expected real CIR; complex IQ requires a separate documented encoder")
        frame_count = int(cir.shape[1])
        complete_count = frame_count - frame_count % pool_frames
        if not complete_count:
            raise ValueError("stream is shorter than one feature pool")
        t_dataset = handle.get("t")
        if t_dataset is not None and t_dataset.shape != (frame_count,):
            raise ValueError("t must have shape (T,)")
        t0 = float(t_dataset[0]) if t_dataset is not None else 0.0
        duration_s = (float(t_dataset[-1]) - t0 + 1.0 / rate_hz) if t_dataset is not None else frame_count / rate_hz
        for start in range(0, complete_count, chunk_frames):
            stop = min(start + chunk_frames, complete_count)
            raw = np.moveaxis(cir[:, start:stop, :], 1, 0).astype(np.float64, copy=False)
            if not np.isfinite(raw).all():
                raise ValueError("non-finite CIR values; refuse silent input imputation")
            if t_dataset is not None:
                raw_t = np.asarray(t_dataset[start:stop], dtype=np.float64) - t0
                expected_t = np.arange(start, stop, dtype=np.float64) / rate_hz
                if not np.isfinite(raw_t).all() or np.any(np.diff(raw_t) <= 0):
                    raise ValueError("timestamps must be finite and strictly increasing")
                maximum_timestamp_grid_error = max(maximum_timestamp_grid_error, float(np.max(np.abs(raw_t - expected_t))))
                if maximum_timestamp_grid_error > max(1e-5, 0.1 / rate_hz):
                    raise ValueError("timestamps deviate from declared uniform rate; resampling must be explicit")
            else:
                raw_t = np.arange(start, stop, dtype=np.float64) / rate_hz
            local_states, descriptors, descriptor_names = encoder.transform(raw)
            joint = tensor_product_states(local_states)
            pure_states = joint @ reservoir.unitary.T
            state_norm_error = max(state_norm_error, float(np.max(np.abs(np.square(np.abs(pure_states)).sum(axis=1) - 1.0))))
            pure, pure_names, pure_history = _pure_observables(pure_states, pure_history, config)
            pool_count = len(raw) // pool_frames
            # Average of pure projectors is a valid input density for a complete causal pool.
            reshaped = pure_states.reshape(pool_count, pool_frames, 8)
            injections = np.einsum("bki,bkj->bij", reshaped, reshaped.conj(), optimize=True) / pool_frames
            injection_trace_error = max(injection_trace_error, float(np.max(np.abs(np.trace(injections, axis1=-2, axis2=-1) - 1.0))))
            quantum, quantum_names = reservoir.transform(injections)
            pooled_descriptors = descriptors.reshape(pool_count, pool_frames, 3, -1).mean(axis=1)
            invariant_descriptors = _symmetric(pooled_descriptors)
            descriptor_labels = [f"{label}_{summary}_radars" for label in descriptor_names for summary in _SUMMARY]
            pooled_pure = pure.reshape(pool_count, pool_frames, -1).mean(axis=1)
            batch = np.concatenate([invariant_descriptors, pooled_pure, quantum], axis=1)
            current_names = descriptor_labels + pure_names + quantum_names
            if names and current_names != names:
                raise RuntimeError("feature schema changed across chunks")
            names = current_names
            if not np.isfinite(batch).all():
                raise FloatingPointError("non-finite quantum features")
            batches.append(batch.astype(np.float32))
            # Last sample, not an earlier nominal midpoint, is the availability timestamp.
            timestamps.append(raw_t[pool_frames - 1::pool_frames])
    features = np.concatenate(batches, axis=0)
    centers = np.concatenate(timestamps).astype(np.float64, copy=False)
    requested_lags = list(_geometry_lags(config))
    raw_actual_lags = [max(1, int(np.floor(value * rate_hz + 0.5))) / rate_hz for value in requested_lags]
    diagnostics: dict[str, Any] = {
        "implementation": "manybody_complex_density_causal_v1",
        "dtype_internal": "complex128", "spin_dimensions": [2, 2, 2], "joint_dimension": 8,
        "feature_count": len(names), "feature_step_s": step_s,
        "timestamp_semantics": "last observed raw sample of each pool; relative to t[0]",
        "availability_delay_from_pool_midpoint_s": (pool_frames - 1) / (2.0 * rate_hz),
        "dropped_final_raw_frames": frame_count - complete_count,
        "maximum_timestamp_grid_error_s": maximum_timestamp_grid_error,
        "unitary_error_max_abs": reservoir.unitary_error,
        "state_norm_error_max_abs": state_norm_error,
        "injection_trace_error_max_abs": injection_trace_error,
        "reservoir_trace_error_max_abs": reservoir.trace_error,
        "reservoir_hermiticity_error_max_abs": reservoir.hermiticity_error,
        "reservoir_min_eigenvalue": reservoir.min_eigenvalue,
        "pure_geometry_requested_lags_s": requested_lags,
        "pure_geometry_actual_lags_s": raw_actual_lags,
        "mixed_geometry_actual_lags_s": [lag * step_s for lag in reservoir.mixed_lags],
        "future_independence_method": "one-sided dynamic EMA and finite sliding FFT endpoint FIR; state reset per stream",
        "analytic_signal": "real residual + i*finite-window circular-Hilbert endpoint over current and past samples only",
        "memory_channel": "rho_t = m U rho_(t-1) U^dagger + (1-m) mean_k |phi_k><phi_k|; CPTP replacement mixture",
        "hamiltonian": "permutation-symmetric all-to-all XXZ plus identical transverse/longitudinal fields",
        "range_translation_invariance": "central moments, center velocity and centered Fourier moments; no absolute center or fixed range bins; assumes translation does not clip support",
        "radar_permutation_invariance": "same local encoding, permutation-commuting Hamiltonian and symmetric local/pair aggregation",
        "quantum_entanglement": "pure input after joint unitary: Schmidt spectra and Meyer-Wallach; mixed reduced pairs: partial-transpose negativity",
        "mixed_entropy_semantics": "mixedness, never asserted to be entanglement",
        "coherence_semantics": "L1 off-diagonal norm in fixed computational basis; basis dependent",
        "qfi_semantics": "pure: 4*(1-fidelity)/lag^2 gauge-projected finite difference; mixed:4*Bures_squared/lag^2 finite difference",
        "bures_semantics": "mixed density uses exact eigendecomposition fidelity; pure uses2*(1-abs(overlap))",
        "initial_geometry_history": "pure: repeat first observed state for unavailable past; mixed: zero distance until lag history exists; reservoir starts maximally mixed",
        "labels_read": False, "composition_read": False, "hardware_quantum_advantage_claimed": False,
    }
    return features, centers, duration_s, names, diagnostics
