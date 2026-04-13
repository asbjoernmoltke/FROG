"""
Blind ePIE (batched) retriever.

Batched parallel ePIE on both E and G.  Per iteration:

    psi_m   = E * G_m              (G_m = G shifted by tau_m)
    Psi_m   = FFT(psi_m)
    Psi'_m  = sqrt(I_meas[m]) * Psi_m / |Psi_m|
    psi'_m  = IFFT(Psi'_m)
    delta_m = psi'_m - psi_m

    E <- E + alpha_E * sum_m  conj(G_m) * delta_m  / sum_m |G_m|^2
    G <- G + alpha_G * sum_m  conj(E_{k+tau_m}) * delta_m[k+tau_m]
                               / sum_m |E_{k+tau_m}|^2

With alpha=1 and GPA denominators the batched update is algebraically a
GPA step; use alpha<1 to follow a slower trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import BlindRetriever, BlindRetrievalResult
from ._common import BlindWorkspace, StoppingCriteria, fast_frog_error
from ...core.field import ElectricField


@dataclass
class BlindEPIE(BlindRetriever):
    dtype: np.dtype = np.complex64
    workers: int = -1
    alpha_field: float = 0.5
    alpha_gate: float = 0.5
    stop_target: Optional[float] = None
    stall_window: Optional[int] = None
    stall_threshold: float = 0.1

    _ws: Optional[BlindWorkspace] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ws = BlindWorkspace(
            trace=self.trace, dtype=self.dtype, workers=self.workers,
        )

    def _retrieve_impl(
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

        sqrt_I_meas = ws.sqrt_I_meas
        I_meas = ws.I_meas
        peak_I_meas = ws.peak_I_meas
        sign_n = ws.sign_n
        idx = ws.idx
        jdx = ws.jdx

        in_buf = ws.fft1d_in
        out_buf = ws.fft1d_out

        G_shifts = np.empty((M, N), dtype=cdtype)
        G_conj = np.empty((M, N), dtype=cdtype)
        G_absq = np.empty((M, N), dtype=rdtype)
        E_shift = np.empty((M, N), dtype=cdtype)
        delta_shift = np.empty((M, N), dtype=cdtype)
        psi_buf = np.empty((M, N), dtype=cdtype)
        prod_buf = np.empty((M, N), dtype=cdtype)
        jdx_flat = ws.jdx_flat
        sign_at_j = ws.sign_at_j
        I_buf = np.empty((M, N), dtype=rdtype)
        abs_buf = np.empty((M, N), dtype=rdtype)
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
        eps = np.finfo(rdtype).tiny
        a_E = self.alpha_field
        a_G = self.alpha_gate
        it = 0

        while not stop.should_stop(it, error_curve):
            ws.build_G_shifts(G, out=G_shifts, conj_out=G_conj, absq_out=G_absq)

            np.multiply(E[None, :], G_shifts, out=in_buf)
            psi_buf[...] = in_buf                       # save psi for delta
            ws.fft1d()

            if it % err_every == 0:
                np.multiply(out_buf.real, out_buf.real, out=I_buf)
                I_buf += out_buf.imag * out_buf.imag
                last_err = fast_frog_error(I_buf, I_meas, peak_I_meas, err_scratch)
            error_curve.append(last_err)

            np.abs(out_buf, out=abs_buf)
            np.maximum(abs_buf, 1e-20, out=abs_buf)
            np.divide(sqrt_I_meas, abs_buf, out=abs_buf)
            out_buf *= abs_buf

            ws.ifft1d()                                  # in_buf <- psi'
            np.subtract(in_buf, psi_buf, out=in_buf)     # in_buf now holds delta

            # ---- Field update ----
            np.multiply(in_buf, G_conj, out=prod_buf)
            num_E = prod_buf.sum(axis=0)
            den_E = G_absq.sum(axis=0)
            E = E + a_E * num_E / np.where(den_E > eps, den_E, eps)

            # ---- Gate update ----
            np.take(in_buf, jdx_flat, out=delta_shift.reshape(-1))
            delta_shift *= sign_at_j
            np.take(E, jdx, out=E_shift)
            np.multiply(delta_shift, np.conj(E_shift), out=prod_buf)
            num_G = prod_buf.sum(axis=0)
            den_G = (E_shift.real * E_shift.real).sum(axis=0) \
                  + (E_shift.imag * E_shift.imag).sum(axis=0)
            G = G + a_G * num_G / np.where(den_G > eps, den_G, eps)

            nE2 = float((E.real * E.real).sum() + (E.imag * E.imag).sum())
            nG2 = float((G.real * G.real).sum() + (G.imag * G.imag).sum())
            if nE2 > 0 and nG2 > 0:
                s = (nE2 / nG2) ** 0.25
                E /= s; G *= s
            it += 1

        ws.build_G_shifts(G, out=G_shifts)
        np.multiply(E[None, :], G_shifts, out=in_buf)
        ws.fft1d()
        np.multiply(out_buf.real, out_buf.real, out=I_buf)
        I_buf += out_buf.imag * out_buf.imag
        error_curve.append(fast_frog_error(I_buf, I_meas, peak_I_meas, err_scratch))

        E, G = ws.center_on_E(E, G, self.trace.grid.dt)

        return BlindRetrievalResult(
            field=ElectricField(grid=self.trace.grid, data=E),
            gate=ElectricField(grid=self.trace.grid, data=G),
            error_curve=error_curve,
            n_iterations=it,
        )
