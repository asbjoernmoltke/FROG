"""
Gradient-descent retrieval for XFROG with a known gate.

Minimizes the intensity-domain loss

    L(E) = 0.5 * sum_{omega, tau} ( |S(omega, tau)|^2 - I_meas(omega, tau) )^2

where  S(omega, tau) = FFT_t [ E(t) * G(t - tau) ]  and the measured trace
I_meas is peak-normalized to 1.  The complex field E(t) is updated via
Wirtinger gradient descent with backtracking line search, which keeps the
step size adaptive without requiring per-problem tuning.

Wirtinger gradient (w.r.t. E*)
------------------------------
With  r(omega, tau) = |S|^2 - I_meas,

    dL/dS*       = r * S
    dL/dE_sig*   = IFFT_t( dL/dS* )                     (adjoint of FFT)
    dL/dE(t)*    = sum_m  dL/dE_sig*(t, tau_m) * conj( G(t - tau_m) )

The numpy IFFT carries a 1/N factor relative to the true adjoint of the
unitary FFT, so the gradient differs from the textbook one by a constant
scale.  Backtracking absorbs this — it does not affect the descent
direction.

Compared to GPA this method is slower per-iteration-of-progress but is
trivially extensible: any extra term added to L (sparsity, smoothness,
bandwidth limits, ...) just contributes an additional gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Retriever, RetrievalResult
from ...core.field import ElectricField
from ...core.transform import _fft_centered, _ifft_centered


@dataclass
class GradientDescent(Retriever):
    """
    Gradient-descent retriever for XFROG with a known gate.

    Parameters
    ----------
    trace : FROGTrace
        Measured XFROG trace.  Internally peak-normalized to 1.
    gate : ElectricField
        Known reference pulse (must share the same Grid as trace).
    step0 : float
        Initial trial step size for backtracking line search.
    bt_shrink : float
        Multiplicative shrink factor used when backtracking rejects a step.
    bt_grow : float
        Multiplicative grow factor applied to the trial step after a
        successful step (lets the step recover after a tight backtrack).
    bt_max_iter : int
        Maximum number of backtracking attempts per iteration before
        accepting the smallest tried step.
    """

    step0: float = 1.0
    bt_shrink: float = 0.5
    bt_grow: float = 1.5
    bt_max_iter: int = 20

    def retrieve(
        self,
        n_iter: int = 200,
        initial_field: Optional[ElectricField] = None,
        seed: int = 0,
    ) -> RetrievalResult:
        grid = self.trace.grid

        # Peak-normalized measured trace.
        I_meas = self.trace.intensity / self.trace.intensity.max()

        # Precompute shifted gate copies G(t - tau_m).  Shape (M, N).
        G_shifts = np.array(
            [np.roll(self.gate.data, s) for s in grid.delay_indices]
        )
        G_shifts_T = G_shifts.T                              # (N, M)
        G_conj_T = np.conj(G_shifts_T)                       # (N, M)

        # Initial guess.
        if initial_field is not None:
            E = initial_field.data.astype(np.complex128).copy()
        else:
            rng = np.random.default_rng(seed)
            E = (
                rng.standard_normal(grid.N)
                + 1j * rng.standard_normal(grid.N)
            )
            E /= np.abs(E).max()

        # Rescale E so that the initial calculated trace has peak ~1.
        # This puts the loss on the same scale as I_meas and lets a single
        # default step size work across problems.
        I0 = np.abs(_fft_centered(E[:, None] * G_shifts_T)) ** 2
        peak0 = I0.max()
        if peak0 > 0:
            E *= np.sqrt(1.0 / peak0)

        def _forward(E_curr):
            E_sig = E_curr[:, None] * G_shifts_T            # (N, M)
            S = _fft_centered(E_sig)                        # (N, M)
            I = np.abs(S) ** 2
            return E_sig, S, I

        def _loss(I):
            return 0.5 * float(np.sum((I - I_meas) ** 2))

        E_sig, S, I = _forward(E)
        loss = _loss(I)

        error_curve: list[float] = []
        I_norm = I / (I.max() + 1e-30)
        error_curve.append(self.frog_error(I_meas, I_norm))

        step = self.step0

        for _ in range(n_iter):
            # ---- Wirtinger gradient w.r.t. E* ----
            r = I - I_meas                                  # (N, M)
            dL_dS_conj = r * S                              # (N, M)
            dL_dEsig_conj = _ifft_centered(dL_dS_conj)      # (N, M)
            grad = np.sum(dL_dEsig_conj * G_conj_T, axis=1) # (N,)

            # ---- Backtracking line search ----
            accepted = False
            for _bt in range(self.bt_max_iter):
                E_trial = E - step * grad
                _, _, I_trial = _forward(E_trial)
                loss_trial = _loss(I_trial)
                if loss_trial < loss:
                    E = E_trial
                    loss = loss_trial
                    I = I_trial
                    _, S, _ = _forward(E)  # refresh S for next gradient
                    step *= self.bt_grow
                    accepted = True
                    break
                step *= self.bt_shrink

            if not accepted:
                # Could not improve — keep current E, leave step shrunken.
                pass

            I_norm = I / (I.max() + 1e-30)
            error_curve.append(self.frog_error(I_meas, I_norm))

        return RetrievalResult(
            field=ElectricField(grid=grid, data=E),
            error_curve=error_curve,
            n_iterations=n_iter,
        )
