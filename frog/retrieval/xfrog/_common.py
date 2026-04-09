"""
Shared infrastructure for the fast XFROG retrievers.

This module isolates the parts that are identical across GPA / PCGP /
Gradient / PIE so each algorithm file can stay focused on its update
rule.  The pieces collected here are:

  * `centered_fft` / `centered_ifft`
        FFT pair that absorbs `fftshift`/`ifftshift` into a (-1)^n sign
        ramp on the input/output.  When N is even, applying that ramp in
        time and frequency yields the same array as
        `fftshift(fft(ifftshift(x)))` up to a global phase that cancels
        in `|S|^2`.  Saves two full-array copies per FFT.

  * `FastWorkspace`
        Built once per (trace, gate) pair.  Holds the precomputed
        constants every retriever needs (shifted gates, sign vector,
        sqrt of the measured intensity, denominators, ...) plus a few
        scratch buffers and a multi-threaded FFT routine.

Memory layout convention
------------------------
All (delay, time) arrays in fast_retrieval use shape **(M, N)** with the
FFT axis = 1 (the contiguous axis).  This is the opposite of the naive
retrieval/ code, which uses (N, M) with FFT axis 0.  Batched FFTs along
the contiguous axis are markedly faster.

Sign-trick FFT (fully baked-in)
-------------------------------
The textbook centered FFT pair is

    S_c = fftshift(fft(ifftshift(x)))

Both shifts can be replaced by a (-1)^n elementwise multiply when N is
even, but doing that multiply per FFT call is itself as expensive as
the FFT.  So we bake the sign vector into the precomputed gate matrix
once and never apply it inside the iteration loop:

    G_signed[m, n] = (-1)^n * G(t_n - tau_m)

Because sign^2 = 1, the sign factors cancel everywhere they would
appear inside the algorithm:

  * Forward signal field: E_sig_signed = E * G_signed produces an array
    that, when fed to plain fft, yields S_signed = sign_w * S_centered.
  * Magnitude / phase split: |S_signed| = |S_centered|, and the sign
    factor in the phase cancels against itself in the projection.
    The intensity projection is therefore simply
        S_signed_new = sqrt(I_meas) * S_signed / |S_signed|
    with no sign multiplication.
  * Inverse FFT yields E_sig_signed_new (not E_sig_new); the sign in
    the time axis gets absorbed into G_signed_conj when we contract
    against it for the field update.

Net effect: the inner loop calls plain `scipy.fft.fft` / `ifft` directly
and never touches a sign vector.  The sign cancellation is exact.

dtype convention
----------------
Single precision (`complex64`) by default.  The phase-retrieval
algorithms are iterative and tolerant of ~1e-7 round-off; complex64
halves the memory bandwidth and roughly doubles FFT throughput.  Pass
`dtype=np.complex128` to a workspace constructor if double precision is
required for validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyfftw

from ...core.field import ElectricField
from ...core.trace import FROGTrace


# Use as many threads as pyfftw reports as "optimal" for the machine.
# Users can override by passing workers= to the workspace.
_DEFAULT_THREADS = max(1, (pyfftw.config.NUM_THREADS or 1))


# ---------------------------------------------------------------------------
# Sign vector for the bake-into-G trick
# ---------------------------------------------------------------------------

def _sign_vector(N: int, dtype) -> np.ndarray:
    """Length-N vector of (-1)^n.  Real, dtype-matched to the workspace."""
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    s = np.ones(N, dtype=real_dtype)
    s[1::2] = -1.0
    return s


# ---------------------------------------------------------------------------
# Fast FROG error metric
# ---------------------------------------------------------------------------

def fast_frog_error(
    I_calc: np.ndarray,
    I_meas: np.ndarray,
    sum_I_meas_sq: float,  # unused; kept for API stability
    peak_I_meas: float,
    scratch: np.ndarray,
) -> float:
    """
    Allocation-free FROG error, equivalent to
    `Retriever.frog_error(I_meas, I_calc)`.

    The closed-form residual `||I_meas||^2 - b^2/a` suffers catastrophic
    cancellation in float32 once the algorithm converges, reading
    spurious 0s.  Instead we form the residual array directly into
    `scratch` and reduce its squared norm.  Two passes over (M, N) plus
    two einsum reductions, all in the input dtype, no allocations.
    """
    a = float(np.einsum("ij,ij->", I_calc, I_calc))   # ||I_calc||^2
    b = float(np.einsum("ij,ij->", I_meas, I_calc))   # <I_meas, I_calc>
    if a <= 0.0:
        return 0.0
    mu = b / a
    # scratch = I_meas - mu * I_calc, then ||scratch||^2.
    np.multiply(I_calc, mu, out=scratch)
    np.subtract(I_meas, scratch, out=scratch)
    sq_residual = float(np.einsum("ij,ij->", scratch, scratch))
    n_elem = I_calc.size
    return float(np.sqrt(sq_residual / n_elem) / (peak_I_meas + 1e-30))


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

@dataclass
class FastWorkspace:
    """
    Precomputed constants and scratch buffers shared by all fast retrievers.

    Build one per (trace, gate, dtype).  Cheap to construct relative to
    a full retrieve() call but not free, so reuse it across retrievals
    on the same dataset where possible.

    Attributes
    ----------
    N, M               : grid sizes
    dtype              : complex dtype (complex64 by default)
    G_shifts           : (M, N) shifted gate matrix with the (-1)^n sign
                         baked in: G_shifts[m, n] = (-1)^n * G(t_n - tau_m)
    G_shifts_conj      : conjugate of the above
    sqrt_I_meas        : (M, N) sqrt of peak-normalized measured intensity
    I_meas             : (M, N) peak-normalized measured intensity
    sum_I_meas_sq      : scalar ||I_meas||^2 (precomputed for fast_frog_error)
    peak_I_meas        : scalar max(I_meas) (= 1.0 since I_meas is normalized,
                         kept explicit for clarity)
    inv_denom_gpa      : (N,) reciprocal of the GPA denominator
    gate_energy        : scalar ||G||^2 (PCGP denominator)
    gate_max_sq        : scalar max|G|^2 (PIE denominator)
    workers            : number of FFT worker threads (-1 = all cores)
    """

    trace: FROGTrace
    gate: ElectricField
    dtype: np.dtype = np.complex64
    workers: int = -1
    plan_effort: str = "FFTW_MEASURE"
    build_2d_plan: bool = False   # GPASquare etc. ask for this at construction

    # Filled by __post_init__
    N: int = field(init=False)
    M: int = field(init=False)
    threads: int = field(init=False)
    G_shifts: np.ndarray = field(init=False)
    G_shifts_conj: np.ndarray = field(init=False)
    I_meas: np.ndarray = field(init=False)
    sqrt_I_meas: np.ndarray = field(init=False)
    sum_I_meas_sq: float = field(init=False)
    peak_I_meas: float = field(init=False)
    inv_denom_gpa: np.ndarray = field(init=False)
    gate_energy: float = field(init=False)
    gate_max_sq: float = field(init=False)

    # pyfftw plans and their pre-allocated input/output buffers.
    # Batched 1D along axis -1 (used by rectangular GPA).
    fft1d_in: np.ndarray = field(init=False)
    fft1d_out: np.ndarray = field(init=False)
    _plan_fft1d: pyfftw.FFTW = field(init=False, repr=False)
    _plan_ifft1d: pyfftw.FFTW = field(init=False, repr=False)
    # Optional 2D plan (square GPA).
    fft2d_in: np.ndarray = field(init=False)
    fft2d_out: np.ndarray = field(init=False)
    _plan_fft2d: object = field(init=False, default=None, repr=False)
    _plan_ifft2d: object = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.trace.grid is not self.gate.grid:
            raise ValueError("trace and gate must share the same Grid instance.")

        grid = self.trace.grid
        self.N = grid.N
        delay_indices = np.asarray(grid.delay_indices)
        self.M = delay_indices.size

        cdtype = np.dtype(self.dtype)
        rdtype = np.float32 if cdtype == np.complex64 else np.float64

        # Shifted gate matrix in (M, N) layout via a single fancy-index op.
        n_idx = np.arange(self.N)
        idx = (n_idx[None, :] - delay_indices[:, None]) % self.N   # (M, N)
        G = np.asarray(self.gate.data, dtype=cdtype)
        G_shifts = np.ascontiguousarray(G[idx])                    # (M, N)

        # Bake the (-1)^n sign vector into G_shifts so the inner loop
        # never touches a sign multiplication.  See module docstring.
        sign_n = _sign_vector(self.N, cdtype)
        G_shifts *= sign_n                                         # in-place
        self.G_shifts = G_shifts
        self.G_shifts_conj = np.conj(G_shifts)

        # Measured intensity in (M, N) layout, peak-normalized.
        I = np.asarray(self.trace.intensity, dtype=rdtype)
        I = I / I.max()
        self.I_meas = np.ascontiguousarray(I.T)                    # (M, N)
        self.sqrt_I_meas = np.sqrt(self.I_meas)

        # Constants for fast_frog_error.
        self.sum_I_meas_sq = float(np.einsum("ij,ij->", self.I_meas, self.I_meas))
        self.peak_I_meas = float(self.I_meas.max())

        # GPA denominator: sum_m |G(t_n - tau_m)|^2  (sign drops out under |.|).
        denom = np.sum(np.abs(self.G_shifts) ** 2, axis=0)         # (N,)
        eps = np.finfo(rdtype).tiny
        self.inv_denom_gpa = (1.0 / np.where(denom > eps, denom, eps)).astype(rdtype)

        # PCGP / PIE scalar denominators.
        self.gate_energy = float(np.sum(np.abs(G) ** 2))
        self.gate_max_sq = float(np.max(np.abs(G) ** 2))

        # Thread count for pyfftw.
        if self.workers is None or self.workers < 0:
            self.threads = _DEFAULT_THREADS
        else:
            self.threads = max(1, int(self.workers))

        # ---- Build the pyfftw plans and their aligned buffers ----
        # Batched 1D plan: input/output are both (M, N), FFT along axis 1.
        self.fft1d_in = pyfftw.empty_aligned((self.M, self.N), dtype=cdtype)
        self.fft1d_out = pyfftw.empty_aligned((self.M, self.N), dtype=cdtype)
        self._plan_fft1d = pyfftw.FFTW(
            self.fft1d_in, self.fft1d_out,
            axes=(1,), direction="FFTW_FORWARD",
            flags=(self.plan_effort, "FFTW_DESTROY_INPUT"),
            threads=self.threads,
        )
        # Reuse the same two buffers for the inverse plan so we can run
        # forward then inverse without copying.
        self._plan_ifft1d = pyfftw.FFTW(
            self.fft1d_out, self.fft1d_in,
            axes=(1,), direction="FFTW_BACKWARD",
            flags=(self.plan_effort, "FFTW_DESTROY_INPUT"),
            threads=self.threads,
        )

        if self.build_2d_plan:
            if self.M != self.N:
                raise ValueError(
                    f"2D plan requires square trace (M == N); got M={self.M}, N={self.N}."
                )
            self.fft2d_in = pyfftw.empty_aligned((self.N, self.N), dtype=cdtype)
            self.fft2d_out = pyfftw.empty_aligned((self.N, self.N), dtype=cdtype)
            self._plan_fft2d = pyfftw.FFTW(
                self.fft2d_in, self.fft2d_out,
                axes=(0, 1), direction="FFTW_FORWARD",
                flags=(self.plan_effort, "FFTW_DESTROY_INPUT"),
                threads=self.threads,
            )
            self._plan_ifft2d = pyfftw.FFTW(
                self.fft2d_out, self.fft2d_in,
                axes=(0, 1), direction="FFTW_BACKWARD",
                flags=(self.plan_effort, "FFTW_DESTROY_INPUT"),
                threads=self.threads,
            )

    # ----- plan execution -----
    # Callers fill self.fft1d_in, call fft1d(), and read self.fft1d_out.
    # pyfftw's FFTW objects normalize the inverse (1/N factor is applied).

    def fft1d(self):
        self._plan_fft1d()

    def ifft1d(self):
        self._plan_ifft1d()

    def fft2d(self):
        self._plan_fft2d()

    def ifft2d(self):
        self._plan_ifft2d()

    # ----- initial guess helper (shared by all retrievers) -----

    def random_initial_field(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        E = (
            rng.standard_normal(self.N)
            + 1j * rng.standard_normal(self.N)
        ).astype(self.dtype)
        E /= np.abs(E).max()
        return E
