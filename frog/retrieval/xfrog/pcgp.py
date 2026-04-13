"""
Fast PCGP retriever for XFROG with a known gate.

Algorithmically a one-line variant of fast GPA.  In the reference
implementation PCGP performs an explicit outer-product reshape of the
updated signal field into an (N, N) matrix `O[n, t']` and then applies
a power-method update against the known gate.  When G is known, that
whole reshape collapses:

    E_new(n) = sum_{t'} O[n, t'] * conj(G(t'))   /  ||G||^2
             = sum_m   E_sig_new[m, n] * conj(G(t_n - tau_m))   /  ||G||^2
             = sum_m   E_sig_new[m, n] * G_shifts_conj[m, n]    /  ||G||^2

which is the same einsum as GPA, just divided by a *scalar* gate
energy instead of the per-n GPA denominator `sum_m |G(t_n-tau_m)|^2`.

This means fast PCGP shares every line of fast GPA except the field
update.  Per-iteration cost is therefore identical: same FFTs, same
projection, same reduction.  The two algorithms differ only in
convergence behavior.

(There is a separate "real" PCGP formulation built on a 2D FFT of the
sheared outer-product matrix.  With a known gate it does not gain over
the simplified version above, so we don't implement it here.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import Retriever, RetrievalResult
from ._common import FastWorkspace, fast_frog_error
from ...core.field import ElectricField


@dataclass
class PCGP(Retriever):
    """Fast PCGP retriever (XFROG, known gate).  Arbitrary (M, N)."""

    dtype: np.dtype = np.complex64
    workers: int = -1
    error_every: int = 10

    _ws: Optional[FastWorkspace] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._ws = FastWorkspace(
            trace=self.trace,
            gate=self.gate,
            dtype=self.dtype,
            workers=self.workers,
        )

    def _retrieve_impl(
        self,
        n_iter: int = 200,
        initial_field: Optional[ElectricField] = None,
        seed: int = 0,
    ) -> RetrievalResult:
        ws = self._ws
        N, M = ws.N, ws.M
        cdtype = ws.dtype
        rdtype = ws.sqrt_I_meas.dtype

        G = ws.G_shifts                       # (M, N) sign-baked
        G_conj = ws.G_shifts_conj             # (M, N) sign-baked
        sqrt_I_meas = ws.sqrt_I_meas
        I_meas = ws.I_meas
        sum_I_meas_sq = ws.sum_I_meas_sq
        peak_I_meas = ws.peak_I_meas

        # PCGP-specific: scalar reciprocal of the global gate energy.
        inv_gate_energy = 1.0 / max(ws.gate_energy, np.finfo(rdtype).tiny)

        in_buf = ws.fft1d_in
        out_buf = ws.fft1d_out

        I_buf = np.empty((M, N), dtype=rdtype)
        err_scratch = np.empty((M, N), dtype=rdtype)

        if initial_field is not None:
            E = np.asarray(initial_field.data, dtype=cdtype).copy()
        else:
            E = ws.random_initial_field(seed)

        error_curve: list[float] = []
        last_err = 0.0

        for it in range(n_iter):
            # ---- Forward ----
            np.multiply(E[None, :], G, out=in_buf)
            ws.fft1d()

            # ---- Error metric ----
            if it % self.error_every == 0:
                np.multiply(out_buf.real, out_buf.real, out=I_buf)
                I_buf += out_buf.imag * out_buf.imag
                last_err = fast_frog_error(
                    I_buf, I_meas, sum_I_meas_sq, peak_I_meas, err_scratch
                )
            error_curve.append(last_err)

            # ---- Intensity projection ----
            np.abs(out_buf, out=I_buf)
            np.maximum(I_buf, 1e-20, out=I_buf)
            np.divide(sqrt_I_meas, I_buf, out=I_buf)
            out_buf *= I_buf

            # ---- Inverse FFT ----
            ws.ifft1d()

            # ---- PCGP field update: scalar denominator ----
            numerator = np.einsum("mn,mn->n", in_buf, G_conj, optimize=True)
            E = (numerator * inv_gate_energy).astype(cdtype, copy=False)

        # Final error.
        np.multiply(E[None, :], G, out=in_buf)
        ws.fft1d()
        np.multiply(out_buf.real, out_buf.real, out=I_buf)
        I_buf += out_buf.imag * out_buf.imag
        error_curve.append(
            fast_frog_error(I_buf, I_meas, sum_I_meas_sq, peak_I_meas, err_scratch)
        )

        return RetrievalResult(
            field=ElectricField(grid=self.trace.grid, data=E),
            error_curve=error_curve,
            n_iterations=n_iter,
        )
