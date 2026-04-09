"""
Blind Wirtinger gradient descent on the trace-matching loss

    L(E, G) = sum_{m,n} ( |Psi_{m,n}|^2 - I_meas[m,n] )^2,
    Psi    = FFT_n( E[n] * G[n - tau_m] )

Wirtinger gradients:
    dL/dE*[n] = sum_m conj(G[n - tau_m]) * IFFT_n( 4 * (|Psi|^2 - I) * Psi )[m,n]
    dL/dG*[k] = sum_m conj(E[k + tau_m]) * IFFT_n(...)[m, k + tau_m]

Step selected by backtracking line search on L.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import BlindRetriever, BlindRetrievalResult
from ._common import BlindWorkspace, StoppingCriteria, fast_frog_error
from ...core.field import ElectricField


@dataclass
class BlindGradient(BlindRetriever):
    dtype: np.dtype = np.complex64
    workers: int = -1
    alpha0: float = 1.0
    ls_shrink: float = 0.5
    ls_max: int = 6
    stop_target: Optional[float] = None
    stall_window: Optional[int] = None
    stall_threshold: float = 0.1

    _ws: Optional[BlindWorkspace] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ws = BlindWorkspace(
            trace=self.trace, dtype=self.dtype, workers=self.workers,
        )

    def _loss_and_psi(self, E, G, G_shifts, in_buf, out_buf, I_buf, ws):
        ws.build_G_shifts(G, out=G_shifts)
        np.multiply(E[None, :], G_shifts, out=in_buf)
        ws.fft1d()
        np.multiply(out_buf.real, out_buf.real, out=I_buf)
        I_buf += out_buf.imag * out_buf.imag
        diff = I_buf - ws.I_meas
        return float((diff * diff).sum())

    def retrieve(
        self,
        n_iter: int = 500,
        initial_field: Optional[ElectricField] = None,
        initial_gate: Optional[ElectricField] = None,
        seed: int = 0,
    ) -> BlindRetrievalResult:
        ws = self._ws
        N, M = ws.N, ws.M
        cdtype = ws.dtype
        rdtype = ws.sqrt_I_meas.dtype

        I_meas = ws.I_meas
        peak_I_meas = ws.peak_I_meas
        sign_at_j = ws.sign_at_j
        jdx_flat = ws.jdx_flat
        jdx = ws.jdx

        in_buf = ws.fft1d_in
        out_buf = ws.fft1d_out

        G_shifts = np.empty((M, N), dtype=cdtype)
        G_conj = np.empty((M, N), dtype=cdtype)
        E_shift = np.empty((M, N), dtype=cdtype)
        grad_shift = np.empty((M, N), dtype=cdtype)
        prod_buf = np.empty((M, N), dtype=cdtype)
        I_buf = np.empty((M, N), dtype=rdtype)
        err_scratch = np.empty((M, N), dtype=rdtype)

        if initial_field is not None:
            E = np.asarray(initial_field.data, dtype=cdtype).copy()
        else:
            E = None
        if initial_gate is not None:
            G = np.asarray(initial_gate.data, dtype=cdtype).copy()
        else:
            G = None
        if E is None or G is None:
            E0, G0 = ws.random_initial(seed)
            E = E if E is not None else E0
            G = G if G is not None else G0

        stop = StoppingCriteria(
            max_iter=n_iter,
            target_error=self.stop_target,
            stall_window=self.stall_window,
            stall_threshold=self.stall_threshold,
        )
        err_every = stop.error_every
        error_curve: list[float] = []
        last_err = 0.0
        alpha = self.alpha0
        it = 0

        while not stop.should_stop(it, error_curve):
            # Forward: build psi, get |Psi|^2
            ws.build_G_shifts(G, out=G_shifts, conj_out=G_conj)
            np.multiply(E[None, :], G_shifts, out=in_buf)
            ws.fft1d()
            np.multiply(out_buf.real, out_buf.real, out=I_buf)
            I_buf += out_buf.imag * out_buf.imag

            if it % err_every == 0:
                last_err = fast_frog_error(I_buf, I_meas, peak_I_meas, err_scratch)
            error_curve.append(last_err)

            loss0 = float(((I_buf - I_meas) ** 2).sum())

            # Residual in freq: 4 * (|Psi|^2 - I) * Psi
            diff = I_buf - I_meas
            out_buf *= (4.0 * diff)
            ws.ifft1d()
            # in_buf now holds IFFT of residual spectrum (time domain)

            # Gradient w.r.t. E
            np.multiply(in_buf, G_conj, out=prod_buf)
            grad_E = prod_buf.sum(axis=0)

            # Gradient w.r.t. G (use pre-step E; not yet updated)
            np.take(in_buf, jdx_flat, out=grad_shift.reshape(-1))
            grad_shift *= sign_at_j
            np.take(E, jdx, out=E_shift)
            np.multiply(grad_shift, np.conj(E_shift), out=prod_buf)
            grad_G = prod_buf.sum(axis=0)

            # Normalize gradients so step size is scale-independent
            gE_norm = np.abs(grad_E).max() + 1e-30
            gG_norm = np.abs(grad_G).max() + 1e-30
            dE = (grad_E / gE_norm).astype(cdtype, copy=False)
            dG = (grad_G / gG_norm).astype(cdtype, copy=False)

            # Backtracking line search on L
            step = alpha
            for _ in range(self.ls_max):
                E_try = E - step * dE
                G_try = G - step * dG
                loss_new = self._loss_and_psi(
                    E_try, G_try, G_shifts, in_buf, out_buf, I_buf, ws,
                )
                if loss_new < loss0:
                    break
                step *= self.ls_shrink
            else:
                step = 0.0
                E_try, G_try = E, G

            E = E_try; G = G_try
            alpha = min(self.alpha0, step / max(self.ls_shrink, 1e-6))

            nE2 = float((E.real * E.real).sum() + (E.imag * E.imag).sum())
            nG2 = float((G.real * G.real).sum() + (G.imag * G.imag).sum())
            if nE2 > 0 and nG2 > 0:
                s = (nE2 / nG2) ** 0.25
                E /= s; G *= s
            it += 1

        # Final error
        ws.build_G_shifts(G, out=G_shifts)
        np.multiply(E[None, :], G_shifts, out=in_buf)
        ws.fft1d()
        np.multiply(out_buf.real, out_buf.real, out=I_buf)
        I_buf += out_buf.imag * out_buf.imag
        error_curve.append(fast_frog_error(I_buf, I_meas, peak_I_meas, err_scratch))

        E, G = ws.center_on_E(E, G)

        return BlindRetrievalResult(
            field=ElectricField(grid=self.trace.grid, data=E),
            gate=ElectricField(grid=self.trace.grid, data=G),
            error_curve=error_curve,
            n_iterations=it,
        )
