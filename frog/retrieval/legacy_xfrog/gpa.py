"""
Generalized Projections Algorithm (GPA) for XFROG with a known gate.

The algorithm alternates between two projections:

  1. Intensity projection (frequency domain)
     Replace the magnitude of the candidate spectrum with sqrt(I_meas),
     keeping the spectral phase.

  2. Field update projection (time domain)
     Since E_sig(t, tau) = E(t) * G(t - tau) and G is known, the
     least-squares estimate of E(t) from the updated signal matrix is:

         E_new(t) = sum_m [ E_sig_new(t, tau_m) * G*(t - tau_m) ]
                    / ( sum_m |G(t - tau_m)|^2 + eps )

     This is a Wiener-filter-like projection identical to the ptychographic
     update used in ePIE when the probe is known exactly.

Reference
---------
Kane, D.J., "Principal components generalized projections: a review"
J. Opt. Soc. Am. B 25, A120 (2008).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Retriever, RetrievalResult
from ...core.field import ElectricField
from ...core.transform import _fft_centered, _ifft_centered


@dataclass
class GPA(Retriever):
    """
    GPA retriever for XFROG with a known gate.

    Parameters
    ----------
    trace : FROGTrace
        Measured (or synthetic) XFROG trace.  Normalized to peak = 1
        internally; the original is not modified.
    gate : ElectricField
        Known reference pulse (must share the same Grid as trace).
    eps : float
        Regularization constant added to the denominator of the field
        update to avoid division by zero where the gate has no power.
    """

    eps: float = 1e-12

    def _retrieve_impl(
        self,
        n_iter: int = 200,
        initial_field: Optional[ElectricField] = None,
        seed: int = 0,
    ) -> RetrievalResult:
        """
        Run GPA for n_iter iterations.

        Parameters
        ----------
        n_iter        : number of iterations
        initial_field : starting guess; random if None
        seed          : RNG seed for the random initial guess
        """
        grid = self.trace.grid

        # Normalize measured trace
        I_meas = self.trace.intensity / self.trace.intensity.max()
        sqrt_I_meas = np.sqrt(I_meas)

        # Precompute all shifted gate arrays G(t - tau_m) once.
        # G_shifts[m, n] = G(t_n - tau_m)  shape (M, N)
        G_shifts = np.array(
            [np.roll(self.gate.data, s) for s in grid.delay_indices]
        )

        # Precompute the denominator of the field update (independent of E).
        # denom[n] = sum_m |G(t_n - tau_m)|^2
        denom = np.sum(np.abs(G_shifts) ** 2, axis=0)          # (N,)
        denom = np.where(denom > self.eps, denom, self.eps)

        # Initial field guess
        if initial_field is not None:
            E = initial_field.data.copy()
        else:
            rng = np.random.default_rng(seed)
            E = (
                rng.standard_normal(grid.N)
                + 1j * rng.standard_normal(grid.N)
            )
            E /= np.abs(E).max()

        error_curve: list[float] = []

        for _ in range(n_iter):
            # ---- Forward: candidate signal field and spectrum ----
            E_sig = E[:, None] * G_shifts.T               # (N, M)
            E_spec = _fft_centered(E_sig)                 # (N, M)

            I_calc = np.abs(E_spec) ** 2
            I_calc_norm = I_calc / (I_calc.max() + 1e-30)
            error_curve.append(self.frog_error(I_meas, I_calc_norm))

            # ---- Intensity projection: replace magnitude, keep phase ----
            E_spec_new = sqrt_I_meas * np.exp(1j * np.angle(E_spec))

            # ---- Back to time domain ----
            E_sig_new = _ifft_centered(E_spec_new)        # (N, M)

            # ---- Field update: least-squares over all delays ----
            # numerator[n] = sum_m E_sig_new[n, m] * G*[m, n]
            numerator = np.sum(E_sig_new * np.conj(G_shifts.T), axis=1)  # (N,)
            E = numerator / denom

        # Compute final error with updated field
        E_sig_final = E[:, None] * G_shifts.T
        E_spec_final = _fft_centered(E_sig_final)
        I_final = np.abs(E_spec_final) ** 2
        I_final_norm = I_final / (I_final.max() + 1e-30)
        error_curve.append(self.frog_error(I_meas, I_final_norm))

        return RetrievalResult(
            field=ElectricField(grid=grid, data=E),
            error_curve=error_curve,
            n_iterations=n_iter,
        )
