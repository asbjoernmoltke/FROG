"""
Fast blind GPA retriever.  Both E and G are unknown.

Per iteration
-------------
1.  Rebuild shifted gate matrix from the current G estimate.
2.  Forward: psi = E * G_shifts,  Psi = FFT(psi).
3.  Intensity projection: Psi' = sqrt(I_meas) * Psi / |Psi|.
4.  Inverse FFT: psi' = IFFT(Psi').
5.  Field update (GPA, per n):
        E[n] = sum_m conj(G[n - tau_m]) * psi'_m[n]  /  sum_m |G[n - tau_m]|^2
6.  Gate update (GPA, per k, using jdx = (k + tau_m) mod N):
        G[k] = sum_m conj(E[(k + tau_m)]) * psi'_m[(k + tau_m)]
               / sum_m |E[(k + tau_m)]|^2

The gate update mirrors the field update under the change of variables
k = n - tau_m in the forward model psi_m[n] = E[n] * G[n - tau_m].  The
sign factor baked into G_shifts cancels in the field reduction exactly as
in the known-gate code; for the gate reduction the signed psi' is
contracted against the *unsigned* E, so the sign on psi' is absorbed by
the unsigned E's complement (explicit cancellation below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import BlindRetriever, BlindRetrievalResult
from ._common import BlindWorkspace, StoppingCriteria, fast_frog_error
from ...core.field import ElectricField


@dataclass
class BlindGPA(BlindRetriever):
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
        psi_shift = np.empty((M, N), dtype=cdtype)
        prod_buf = np.empty((M, N), dtype=cdtype)
        I_buf = np.empty((M, N), dtype=rdtype)
        abs_buf = np.empty((M, N), dtype=rdtype)
        err_scratch = np.empty((M, N), dtype=rdtype)
        jdx_flat = ws.jdx_flat
        sign_at_j = ws.sign_at_j

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
        it = 0

        while not stop.should_stop(it, error_curve):
            ws.build_G_shifts(G, out=G_shifts, conj_out=G_conj, absq_out=G_absq)

            # Forward: psi = E * G_shifts
            np.multiply(E[None, :], G_shifts, out=in_buf)
            ws.fft1d()

            if it % err_every == 0:
                np.multiply(out_buf.real, out_buf.real, out=I_buf)
                I_buf += out_buf.imag * out_buf.imag
                last_err = fast_frog_error(I_buf, I_meas, peak_I_meas, err_scratch)
            error_curve.append(last_err)

            # Intensity projection
            np.abs(out_buf, out=abs_buf)
            np.maximum(abs_buf, 1e-20, out=abs_buf)
            np.divide(sqrt_I_meas, abs_buf, out=abs_buf)
            out_buf *= abs_buf

            # Inverse: in_buf <- psi' (still sign-carrying on time axis)
            ws.ifft1d()

            # ---- Field update ----
            np.multiply(in_buf, G_conj, out=prod_buf)
            num_E = prod_buf.sum(axis=0)
            den_E = G_absq.sum(axis=0)
            E = num_E / np.where(den_E > eps, den_E, eps)

            # ---- Gate update ----
            np.take(in_buf, jdx_flat, out=psi_shift.reshape(-1))
            psi_shift *= sign_at_j
            np.take(E, jdx, out=E_shift)
            np.multiply(psi_shift, np.conj(E_shift), out=prod_buf)
            num_G = prod_buf.sum(axis=0)
            den_G = (E_shift.real * E_shift.real).sum(axis=0) \
                  + (E_shift.imag * E_shift.imag).sum(axis=0)
            G = num_G / np.where(den_G > eps, den_G, eps)

            nE2 = float((E.real * E.real).sum() + (E.imag * E.imag).sum())
            nG2 = float((G.real * G.real).sum() + (G.imag * G.imag).sum())
            if nE2 > 0 and nG2 > 0:
                s = (nE2 / nG2) ** 0.25
                E /= s; G *= s
            it += 1

        # Final error
        ws.build_G_shifts(G, out=G_shifts)
        E = E.astype(cdtype, copy=False)
        G = G.astype(cdtype, copy=False)
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
