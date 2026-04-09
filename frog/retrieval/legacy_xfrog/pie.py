"""
Ptychographic Iterative Engine (PIE / ePIE) for XFROG with a known gate.

PIE treats XFROG as a ptychography problem in which the gate plays the
role of a known probe and the unknown pulse plays the role of the object.
Each iteration loops over delays in random order; for each delay it
performs an intensity projection on a single spectral slice and applies
the ePIE object-only update

    E(t) <- E(t) + alpha * conj(G(t - tau_m)) / max |G|^2
                  * ( psi'_m(t) - psi_m(t) )

where psi_m(t)  = E(t) * G(t - tau_m)
      Psi_m(w)  = FFT_t psi_m
      Psi'_m(w) = sqrt(I_meas(w, tau_m)) * Psi_m / |Psi_m|
      psi'_m(t) = IFFT_w Psi'_m

Compared to GPA / PCGP this is a stochastic per-delay update rather than
a batch projection.  It usually makes very fast initial progress and
handles noisy / sparse delay grids more gracefully, at the cost of
slightly noisier convergence in the asymptotic regime.

Reference
---------
Maiden, A.M., Rodenburg, J.M., "An improved ptychographical phase
retrieval algorithm for diffractive imaging," Ultramicroscopy 109, 1256
(2009).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Retriever, RetrievalResult
from ...core.field import ElectricField
from ...core.transform import _fft_centered, _ifft_centered


@dataclass
class PIE(Retriever):
    """
    Ptychographic (ePIE) retriever for XFROG with a known gate.

    Parameters
    ----------
    trace : FROGTrace
        Measured XFROG trace.  Internally peak-normalized to 1.
    gate : ElectricField
        Known reference pulse (must share the same Grid as trace).
    alpha : float
        Object update strength.  alpha=1 is the standard ePIE choice.
    eps : float
        Regularization for the gate-magnitude denominator.
    """

    alpha: float = 1.0
    eps: float = 1e-12

    def retrieve(
        self,
        n_iter: int = 200,
        initial_field: Optional[ElectricField] = None,
        seed: int = 0,
    ) -> RetrievalResult:
        grid = self.trace.grid
        N = grid.N

        # Peak-normalized measured trace.  Columns indexed by delay.
        I_meas = self.trace.intensity / self.trace.intensity.max()
        sqrt_I_meas = np.sqrt(I_meas)                        # (N, M)

        delay_indices = np.asarray(grid.delay_indices)
        M = delay_indices.size

        # Shifted gates G(t - tau_m).  Shape (M, N).
        G_shifts = np.array([np.roll(self.gate.data, s) for s in delay_indices])
        G_max_sq = float(np.max(np.abs(self.gate.data) ** 2))
        G_max_sq = max(G_max_sq, self.eps)

        # Initial guess.
        if initial_field is not None:
            E = initial_field.data.astype(np.complex128).copy()
        else:
            rng_init = np.random.default_rng(seed)
            E = (
                rng_init.standard_normal(N)
                + 1j * rng_init.standard_normal(N)
            )
            E /= np.abs(E).max()

        # Separate RNG for delay shuffling so it advances each iteration.
        rng_order = np.random.default_rng(seed + 1)

        def _full_error(E_curr):
            E_sig = E_curr[:, None] * G_shifts.T             # (N, M)
            E_spec = _fft_centered(E_sig)
            I_calc = np.abs(E_spec) ** 2
            return self.frog_error(I_meas, I_calc / (I_calc.max() + 1e-30))

        error_curve: list[float] = [_full_error(E)]

        for _ in range(n_iter):
            order = rng_order.permutation(M)
            for m in order:
                Gm = G_shifts[m]                             # (N,)
                psi = E * Gm                                 # (N,)
                Psi = _fft_centered(psi)                     # (N,)

                # Single-slice intensity projection.
                mag = np.abs(Psi)
                phase = Psi / np.where(mag > 1e-30, mag, 1.0)
                Psi_new = sqrt_I_meas[:, m] * phase

                psi_new = _ifft_centered(Psi_new)            # (N,)

                # ePIE object update (gate held fixed).
                E = E + self.alpha * np.conj(Gm) / G_max_sq * (psi_new - psi)

            error_curve.append(_full_error(E))

        return RetrievalResult(
            field=ElectricField(grid=grid, data=E),
            error_curve=error_curve,
            n_iterations=n_iter,
        )
