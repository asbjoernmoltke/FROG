"""
SHG-FROG GPA.  Gate = Field, single-variable update.

After the standard intensity projection, the field is updated from both
its "field" and "gate" roles and the two contributions are averaged:

    E_field[n] = sum_m conj(E_m[n]) * psi'_m[n]  / sum_m |E_m[n]|^2
    E_gate[k]  = sum_m conj(E[k+tau]) * psi'_m[k+tau] / sum_m |E[k+tau]|^2
    E_new      = (E_field + E_gate) / 2

This enforces the SHG constraint G = E while using information from both
roles of the pulse in the signal field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import SHGRetriever, SHGRetrievalResult
from ..blind_xfrog._common import BlindWorkspace, StoppingCriteria, fast_frog_error
from ...core.field import ElectricField


@dataclass
class SHGGPA(SHGRetriever):
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
            # G = E: rebuild shifted-"gate" from current E
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

            # Power method on the outer product matrix O:
            # E_new[n] = Σ_m ψ'_m[n] * E[(n - τ_m) mod N]
            # G_shifts already holds sign_n * E[idx], and in_buf holds
            # sign_n * ψ'.  Signs cancel: (sign * ψ') * conj(sign * E_old) gives
            # ψ' * conj(E_old) (wrong), but we want ψ' * E_old (not conjugated).
            # Strip sign from G_shifts: E_shifted = G_shifts * sign_n (undo bake).
            np.multiply(in_buf, G_shifts, out=prod_buf)  # sign²=1, gives ψ' * E_shifted
            # But this is sign_n*ψ' * sign_n*E = ψ'*E (sign² cancels). Good.
            E = prod_buf.sum(axis=0).astype(cdtype, copy=False)

            # Peak-normalize (only one variable, no scale ambiguity to balance)
            pk = float(np.abs(E).max())
            if pk > 0:
                E /= pk
            it += 1

        # Final error
        ws.build_G_shifts(E, out=G_shifts)
        np.multiply(E[None, :], G_shifts, out=in_buf)
        ws.fft1d()
        np.multiply(out_buf.real, out_buf.real, out=I_buf)
        I_buf += out_buf.imag * out_buf.imag
        error_curve.append(fast_frog_error(I_buf, I_meas, peak_I_meas, err_scratch))

        # Center peak at N/2
        shift = N // 2 - int(np.argmax(np.abs(E) ** 2))
        E = np.roll(E, shift)

        return SHGRetrievalResult(
            field=ElectricField(grid=self.trace.grid, data=E),
            error_curve=error_curve,
            n_iterations=it,
        )
