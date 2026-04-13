"""
Shared workspace for blind XFROG retrievers.

Differences from ``fast_retrieval._common.FastWorkspace``:

* No gate is supplied at construction — both E and G are unknown.
* The shifted-gate matrix ``G_shifts`` is *not* precomputed: it is rebuilt
  from the current gate estimate every iteration via a fancy-index gather
  against the precomputed ``idx`` matrix.
* A symmetric "inverse" gather ``jdx`` lets the gate update read the
  (M, N) signal-field buffer at positions ``(k + tau_m) mod N`` for each
  delay m, which is exactly what drops out of the change-of-variables
  ``k = n - tau_m`` in the forward model ``psi_m[n] = E[n] * G[n - tau_m]``.

Sign-trick note
---------------
We keep the (-1)^n sign bake trick, but applied *per iteration* on top of
the freshly gathered gate:

    G_shifts[m, n] = sign_n * G_current[(n - tau_m) mod N]

The sign multiplication costs one (M, N) in-place op per iter, still far
cheaper than the FFT pair.  Algebraically it cancels in the same places
as in the known-gate code, so the field/gate update reductions use the
*unsigned* current E/G alongside the signed gathered buffers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import numpy as np
import pyfftw

from ...core.trace import FROGTrace


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

@dataclass
class StoppingCriteria:
    """
    Shared stopping-condition checker.

    Parameters
    ----------
    max_iter : int
        Hard iteration cap.  Always active.
    target_error : float or None
        Stop when FROG error drops below this.  None = off.
    stall_window : int or None
        Number of iterations to look back for stall detection.
        None = off.  Typical value: 10.
    stall_threshold : float
        Minimum |log10(err_now) - log10(err_prev)| over the window.
        If the change is smaller, declare a stall.  Default 0.1
        (i.e. less than one tenth of a decade of progress).
    """
    max_iter: int = 500
    target_error: Optional[float] = None
    stall_window: Optional[int] = None
    stall_threshold: float = 0.1

    @property
    def error_every(self) -> int:
        """How often the error must actually be computed, given active criteria."""
        if self.target_error is not None:
            # Check frequently enough to catch the target without overshooting too much.
            # Every 10 iters is a reasonable default.
            k = 10
            if self.stall_window is not None:
                k = min(k, self.stall_window)
            return k
        if self.stall_window is not None:
            return self.stall_window
        # max_iter only: just compute once at the end
        return self.max_iter

    def should_stop(self, it: int, error_curve: list[float]) -> bool:
        if it >= self.max_iter:
            return True
        if len(error_curve) == 0:
            return False
        err = error_curve[-1]
        if self.target_error is not None and err <= self.target_error:
            return True
        if self.stall_window is not None and len(error_curve) > self.stall_window:
            prev = error_curve[-1 - self.stall_window]
            if prev > 0 and err > 0:
                delta = abs(math.log10(prev) - math.log10(err))
                if delta < self.stall_threshold:
                    return True
        return False


_DEFAULT_THREADS = max(1, (pyfftw.config.NUM_THREADS or 1))


def _sign_vector(N: int, dtype) -> np.ndarray:
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    s = np.ones(N, dtype=real_dtype)
    s[1::2] = -1.0
    return s


def fast_frog_error(I_calc, I_meas, peak_I_meas, scratch) -> float:
    a = float(np.einsum("ij,ij->", I_calc, I_calc))
    b = float(np.einsum("ij,ij->", I_meas, I_calc))
    if a <= 0.0:
        return 0.0
    mu = b / a
    np.multiply(I_calc, mu, out=scratch)
    np.subtract(I_meas, scratch, out=scratch)
    sq = float(np.einsum("ij,ij->", scratch, scratch))
    return float(np.sqrt(sq / I_calc.size) / (peak_I_meas + 1e-30))


@dataclass
class BlindWorkspace:
    trace: FROGTrace
    dtype: np.dtype = np.complex64
    workers: int = -1
    plan_effort: str = "FFTW_MEASURE"

    # Filled by __post_init__
    N: int = field(init=False)
    M: int = field(init=False)
    threads: int = field(init=False)
    sign_n: np.ndarray = field(init=False)
    idx: np.ndarray = field(init=False)         # (M, N) forward shift: (n - tau) % N
    jdx: np.ndarray = field(init=False)         # (M, N) inverse shift: (k + tau) % N
    jdx_flat: np.ndarray = field(init=False)    # flat indices for in_buf.take
    sign_at_j: np.ndarray = field(init=False)   # (M, N) sign_n[jdx]
    I_meas: np.ndarray = field(init=False)
    sqrt_I_meas: np.ndarray = field(init=False)
    peak_I_meas: float = field(init=False)

    # pyfftw plans
    fft1d_in: np.ndarray = field(init=False)
    fft1d_out: np.ndarray = field(init=False)
    _plan_fft1d: pyfftw.FFTW = field(init=False, repr=False)
    _plan_ifft1d: pyfftw.FFTW = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grid = self.trace.grid
        self.N = grid.N
        delay_indices = np.asarray(grid.delay_indices)
        self.M = delay_indices.size

        cdtype = np.dtype(self.dtype)
        rdtype = np.float32 if cdtype == np.complex64 else np.float64

        n_idx = np.arange(self.N)
        self.idx = ((n_idx[None, :] - delay_indices[:, None]) % self.N).astype(np.intp)
        self.jdx = ((n_idx[None, :] + delay_indices[:, None]) % self.N).astype(np.intp)
        self.sign_n = _sign_vector(self.N, cdtype)
        # Precompute (M, N) signed-ramp gather and a flat index for in_buf
        # so the gate update uses one take + one elementwise mul per iter.
        self.sign_at_j = self.sign_n[self.jdx].copy()
        row_off = (np.arange(self.M) * self.N)[:, None]
        self.jdx_flat = (row_off + self.jdx).ravel()

        I = np.asarray(self.trace.intensity, dtype=rdtype)
        I = I / I.max()
        self.I_meas = np.ascontiguousarray(I.T)        # (M, N)
        self.sqrt_I_meas = np.sqrt(self.I_meas)
        self.peak_I_meas = float(self.I_meas.max())

        self.threads = _DEFAULT_THREADS if (self.workers is None or self.workers < 0) \
                                         else max(1, int(self.workers))

        self.fft1d_in = pyfftw.empty_aligned((self.M, self.N), dtype=cdtype)
        self.fft1d_out = pyfftw.empty_aligned((self.M, self.N), dtype=cdtype)
        self._plan_fft1d = pyfftw.FFTW(
            self.fft1d_in, self.fft1d_out,
            axes=(1,), direction="FFTW_FORWARD",
            flags=(self.plan_effort, "FFTW_DESTROY_INPUT"),
            threads=self.threads,
        )
        self._plan_ifft1d = pyfftw.FFTW(
            self.fft1d_out, self.fft1d_in,
            axes=(1,), direction="FFTW_BACKWARD",
            flags=(self.plan_effort, "FFTW_DESTROY_INPUT"),
            threads=self.threads,
        )

    def fft1d(self): self._plan_fft1d()
    def ifft1d(self): self._plan_ifft1d()

    # ---- helpers ----
    def build_G_shifts(self, G: np.ndarray, out: np.ndarray,
                        conj_out: np.ndarray | None = None,
                        absq_out: np.ndarray | None = None) -> None:
        """Fill ``out[m, n] = sign_n * G[(n - tau_m) % N]`` and optionally
        its conjugate and squared magnitude in one pass."""
        np.take(G, self.idx, out=out)
        out *= self.sign_n
        if conj_out is not None:
            np.conjugate(out, out=conj_out)
        if absq_out is not None:
            np.multiply(out.real, out.real, out=absq_out)
            absq_out += out.imag * out.imag

    @staticmethod
    def center_on_E(E: np.ndarray, G: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Post-process blind retrieval to remove trivial ambiguities.

        1. Time-translation: roll both E and G so argmax|E|^2 is at N//2.
        2. Frequency-shift: remove the carrier frequency of G by shifting
           it onto E, so that G is baseband.  Blind FROG has a frequency-
           shift ambiguity: E*exp(iωt), G*exp(-iωt) produce the same
           trace.  We resolve it by forcing G to have zero spectral
           centroid.
        """
        N = E.shape[0]
        # 1. Time centering
        shift = N // 2 - int(np.argmax(np.abs(E) ** 2))
        E = np.roll(E, shift)
        G = np.roll(G, shift)

        # 2. Frequency-shift removal
        # Compute spectral centroid of G
        G_spec = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(G))) ** 2)
        freq = np.fft.fftshift(np.fft.fftfreq(N, dt))
        total = G_spec.sum()
        if total > 0:
            nu_G = np.dot(freq, G_spec) / total
            # Shift G -> baseband, compensate on E
            t = (np.arange(N) - N // 2) * dt
            phase_ramp = np.exp(-2j * np.pi * nu_G * t).astype(E.dtype)
            G = G * phase_ramp
            E = E * np.conj(phase_ramp)

        return E, G

    def random_initial(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        cdtype = self.dtype
        E = (rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N)).astype(cdtype)
        G = (rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N)).astype(cdtype)
        E /= np.abs(E).max(); G /= np.abs(G).max()
        return E, G
