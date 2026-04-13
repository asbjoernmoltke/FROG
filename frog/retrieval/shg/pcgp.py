"""
SHG-FROG PCGP.  Gate = Field, scalar energy denominators.

The classic PCGP for SHG-FROG: after intensity projection, the outer
product of the signal field is formed and the dominant eigenvector
(power-method approximation) is extracted.  With the G=E constraint the
field and gate roles yield a single averaged update with scalar ||E||^2
denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import SHGRetriever, SHGRetrievalResult
from ..blind_xfrog._common import BlindWorkspace, StoppingCriteria, fast_frog_error
from ...core.field import ElectricField


@dataclass
class SHGPCGP(SHGRetriever):
    dtype: np.dtype = np.complex64
    workers: int = -1
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
        seed: int = 0,
    ) -> SHGRetrievalResult:
        ws = self._ws
        N, M = ws.N, ws.M
        cdtype = ws.dtype
        rdtype = ws.sqrt_I_meas.dtype

        sqrt_I_meas = ws.sqrt_I_meas
        I_meas = ws.I_meas
        peak_I_meas = ws.peak_I_meas

        in_buf = ws.fft1d_in
        out_buf = ws.fft1d_out

        G_shifts = np.empty((M, N), dtype=cdtype)
        prod_buf = np.empty((M, N), dtype=cdtype)
        I_buf = np.empty((M, N), dtype=rdtype)
        abs_buf = np.empty((M, N), dtype=rdtype)
        err_scratch = np.empty((M, N), dtype=rdtype)

        if initial_field is not None:
            E = np.asarray(initial_field.data, dtype=cdtype).copy()
        else:
            E, _ = ws.random_initial(seed)

        stop = StoppingCriteria(
            max_iter=n_iter,
            target_error=self.stop_target,
            stall_window=self.stall_window,
            stall_threshold=self.stall_threshold,
        )
        err_every = stop.error_every
        error_curve: list[float] = []
        last_err = 0.0
        it = 0

        while not stop.should_stop(it, error_curve):
            ws.build_G_shifts(E, out=G_shifts)

            np.multiply(E[None, :], G_shifts, out=in_buf)
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
            ws.ifft1d()

            # Power method: E_new = Σ_m ψ'_m * E_shifted (signs cancel)
            np.multiply(in_buf, G_shifts, out=prod_buf)
            E = prod_buf.sum(axis=0).astype(cdtype, copy=False)

            pk = float(np.abs(E).max())
            if pk > 0:
                E /= pk
            it += 1

        ws.build_G_shifts(E, out=G_shifts)
        np.multiply(E[None, :], G_shifts, out=in_buf)
        ws.fft1d()
        np.multiply(out_buf.real, out_buf.real, out=I_buf)
        I_buf += out_buf.imag * out_buf.imag
        error_curve.append(fast_frog_error(I_buf, I_meas, peak_I_meas, err_scratch))

        shift = N // 2 - int(np.argmax(np.abs(E) ** 2))
        E = np.roll(E, shift)

        return SHGRetrievalResult(
            field=ElectricField(grid=self.trace.grid, data=E),
            error_curve=error_curve,
            n_iterations=it,
        )
