"""
Principal Components Generalized Projections (PCGP) for XFROG with a
known gate.

Standard PCGP (Kane 1998, 2008) extracts both pulse and gate from the
signal-field matrix by recasting it into outer-product form and taking
its leading singular vectors via the power method.  For XFROG the gate
is known, so the SVD step collapses to a single matrix-vector product
against the known gate.

Outer-product reshape
---------------------
Define t' = t - tau.  Then

    E_sig(t, tau) = E(t) * G(t - tau)  =  E(t) * G(t')

so in (t, t') coordinates the signal field is the outer product
O(t, t') = E(t) * G(t').  PCGP performs the field update in this
coordinate system:

    1. Form E_sig(t, tau) from the current E and known G.
    2. Spectral intensity projection (replace |FFT_t E_sig| with sqrt(I_meas)).
    3. Map E_sig_new(t, tau)  ->  O(t, t')  via tau = t - t'  (mod N).
    4. Power-method update of E given the known G:

           E_new(t) = sum_{t'} O(t, t') * conj(G(t'))  /  ||G||^2

The crucial difference from GPA is the denominator: PCGP normalizes by
the *global* gate energy ||G||^2 rather than the per-time delay coverage
sum_m |G(t - tau_m)|^2.  This makes the update a true projection onto
the rank-1 subspace spanned by the known gate and tends to be more
stable when the delay grid does not fully cover the pulse support.

References
----------
Kane, D.J., "Real-time measurement of ultrashort laser pulses using
principal component generalized projections," IEEE J. Sel. Top.
Quantum Electron. 4, 278 (1998).

Kane, D.J., "Principal components generalized projections: a review,"
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
class PCGP(Retriever):
    """
    PCGP retriever for XFROG with a known gate.

    Parameters
    ----------
    trace : FROGTrace
        Measured XFROG trace.  Internally peak-normalized to 1.
    gate : ElectricField
        Known reference pulse (must share the same Grid as trace).
    eps : float
        Regularization for the gate-energy denominator.
    """

    eps: float = 1e-12

    def _retrieve_impl(
        self,
        n_iter: int = 200,
        initial_field: Optional[ElectricField] = None,
        seed: int = 0,
    ) -> RetrievalResult:
        grid = self.trace.grid
        N = grid.N

        # Peak-normalized measured trace.
        I_meas = self.trace.intensity / self.trace.intensity.max()
        sqrt_I_meas = np.sqrt(I_meas)

        # Shifted gates G(t - tau_m).  Shape (M, N).
        delay_indices = np.asarray(grid.delay_indices)
        M = delay_indices.size
        G_shifts = np.array([np.roll(self.gate.data, s) for s in delay_indices])
        G_shifts_T = G_shifts.T                              # (N, M)

        # Outer-product reshape index map.
        # For row n and delay column m, the outer-product column is
        #   t'_index = (n - tau_m) mod N
        n_idx = np.arange(N)[:, None]                        # (N, 1)
        tprime_idx = (n_idx - delay_indices[None, :]) % N    # (N, M)

        # Global gate energy (PCGP denominator).
        G = self.gate.data
        gate_energy = float(np.sum(np.abs(G) ** 2))
        gate_energy = max(gate_energy, self.eps)
        G_conj = np.conj(G)                                  # (N,)

        # Initial guess.
        if initial_field is not None:
            E = initial_field.data.astype(np.complex128).copy()
        else:
            rng = np.random.default_rng(seed)
            E = (
                rng.standard_normal(N)
                + 1j * rng.standard_normal(N)
            )
            E /= np.abs(E).max()

        error_curve: list[float] = []

        for _ in range(n_iter):
            # ---- Forward: candidate signal field and spectrum ----
            E_sig = E[:, None] * G_shifts_T                  # (N, M)
            E_spec = _fft_centered(E_sig)                    # (N, M)

            I_calc = np.abs(E_spec) ** 2
            I_calc_norm = I_calc / (I_calc.max() + 1e-30)
            error_curve.append(self.frog_error(I_meas, I_calc_norm))

            # ---- Intensity projection ----
            E_spec_new = sqrt_I_meas * np.exp(1j * np.angle(E_spec))

            # ---- Back to (t, tau) ----
            E_sig_new = _ifft_centered(E_spec_new)           # (N, M)

            # ---- Outer-product reshape: O[n, t'] from E_sig_new[n, m] ----
            # Scatter E_sig_new into O at columns t'_index. Multiple m's may
            # map to the same t' for a given n only if delays repeat, which
            # they don't, so this is a clean assignment.
            O = np.zeros((N, N), dtype=np.complex128)
            np.put_along_axis(O, tprime_idx, E_sig_new, axis=1)

            # ---- Power-method update against the known gate ----
            #   E_new(n) = sum_{t'} O(n, t') * conj(G(t')) / ||G||^2
            E = (O @ G_conj) / gate_energy

        # Final error after the last update.
        E_sig_final = E[:, None] * G_shifts_T
        E_spec_final = _fft_centered(E_sig_final)
        I_final = np.abs(E_spec_final) ** 2
        I_final_norm = I_final / (I_final.max() + 1e-30)
        error_curve.append(self.frog_error(I_meas, I_final_norm))

        return RetrievalResult(
            field=ElectricField(grid=grid, data=E),
            error_curve=error_curve,
            n_iterations=n_iter,
        )
