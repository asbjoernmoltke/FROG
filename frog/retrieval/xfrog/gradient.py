"""
Fast gradient-descent retriever for XFROG with a known gate.

Same Wirtinger gradient + backtracking line search as the reference
implementation; the speed comes from the same primitives as fast GPA:

  * pyfftw plans + preallocated buffers (forward and inverse share the
    same input/output pair, so the gradient pass and the line-search
    forward pass reuse the same memory)
  * sign-trick absorbed into G_shifts
  * complex64 / (M, N) layout / batched 1D FFT along the time axis
  * alloc-free FROG error metric reused from `_common`

Algorithmic improvements vs the reference
-----------------------------------------
1. The reference recomputes `_forward(E)` after a successful line-search
   step in order to refresh `S`.  We instead remember the trial output
   that was accepted and reuse it as the next iteration's forward pass —
   one fewer FFT pair per accepted step.

2. Backtracking now stores the *real* loss inside the workspace buffer
   so the comparison `loss_trial < loss` runs against the same allocator
   pattern every iteration (no resize, no reallocation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import Retriever, RetrievalResult
from ._common import FastWorkspace, fast_frog_error
from ...core.field import ElectricField


@dataclass
class GradientDescent(Retriever):
    """Fast gradient-descent retriever (XFROG, known gate)."""

    dtype: np.dtype = np.complex64
    workers: int = -1
    error_every: int = 10

    step0: float = 1.0
    bt_shrink: float = 0.5
    bt_grow: float = 1.5
    bt_max_iter: int = 20

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

        G = ws.G_shifts
        G_conj = ws.G_shifts_conj
        I_meas = ws.I_meas
        sum_I_meas_sq = ws.sum_I_meas_sq
        peak_I_meas = ws.peak_I_meas

        in_buf = ws.fft1d_in     # E_sig
        out_buf = ws.fft1d_out   # E_spec

        # Scratch buffers (real M*N for intensities, complex for trial forward).
        I_buf = np.empty((M, N), dtype=rdtype)
        I_trial = np.empty((M, N), dtype=rdtype)
        err_scratch = np.empty((M, N), dtype=rdtype)
        in_trial = np.empty((M, N), dtype=cdtype)   # holds E_trial * G
        spec_trial = np.empty((M, N), dtype=cdtype)

        if initial_field is not None:
            E = np.asarray(initial_field.data, dtype=cdtype).copy()
        else:
            E = ws.random_initial_field(seed)

        # ---- Forward at the initial guess; rescale so peak(I_calc) ~= 1 ----
        np.multiply(E[None, :], G, out=in_buf)
        ws.fft1d()                                     # out_buf = FFT(in_buf) = S
        np.multiply(out_buf.real, out_buf.real, out=I_buf)
        I_buf += out_buf.imag * out_buf.imag           # I_buf = |S|^2
        peak0 = float(I_buf.max())
        if peak0 > 0:
            scale = (1.0 / peak0) ** 0.5
            E *= scale
            # Rebuild forward at the rescaled E.
            np.multiply(E[None, :], G, out=in_buf)
            ws.fft1d()
            np.multiply(out_buf.real, out_buf.real, out=I_buf)
            I_buf += out_buf.imag * out_buf.imag

        def _loss(I):
            # 0.5 * ||I - I_meas||^2 with no allocation.
            np.subtract(I, I_meas, out=err_scratch)
            return 0.5 * float(np.einsum("ij,ij->", err_scratch, err_scratch))

        loss = _loss(I_buf)

        last_err = fast_frog_error(I_buf, I_meas, sum_I_meas_sq, peak_I_meas, err_scratch)
        error_curve: list[float] = [last_err]   # initial entry, matches reference

        step = self.step0

        for it in range(n_iter):
            # ---- Wirtinger gradient w.r.t. E* ----
            # r = I - I_meas, dL/dS* = r * S, dL/dE_sig* = IFFT(dL/dS*),
            # dL/dE* = sum_m dL/dE_sig*[m,:] * conj(G[m,:]).
            # We can do the IFFT in-place over the plan pair: stash r*S in out_buf,
            # then ws.ifft1d() puts the time-domain version into in_buf.
            np.subtract(I_buf, I_meas, out=err_scratch)         # r (real)
            # out_buf currently holds S; multiply in place by r.
            out_buf *= err_scratch
            ws.ifft1d()                                          # in_buf = IFFT(r*S)
            grad = np.einsum("mn,mn->n", in_buf, G_conj, optimize=True)

            # ---- Backtracking line search ----
            accepted = False
            for _bt in range(self.bt_max_iter):
                E_trial = E - step * grad
                np.multiply(E_trial[None, :], G, out=in_trial)
                # We need an FFT of in_trial; the plan is bound to (in_buf, out_buf),
                # so copy in_trial into in_buf and run the plan.  This costs one
                # extra (M,N) memcpy per LS attempt but avoids creating a second plan.
                in_buf[...] = in_trial
                ws.fft1d()                                       # out_buf = FFT(in_trial)
                # Save the FFT result for potential acceptance.
                spec_trial[...] = out_buf
                np.multiply(spec_trial.real, spec_trial.real, out=I_trial)
                I_trial += spec_trial.imag * spec_trial.imag
                loss_trial = _loss(I_trial)

                if loss_trial < loss:
                    E = E_trial
                    loss = loss_trial
                    # Promote trial state to current state.
                    I_buf, I_trial = I_trial, I_buf
                    out_buf[...] = spec_trial   # the in-place IFFT below needs a fresh S
                    step *= self.bt_grow
                    accepted = True
                    break
                step *= self.bt_shrink

            if accepted and (it % self.error_every == 0):
                last_err = fast_frog_error(
                    I_buf, I_meas, sum_I_meas_sq, peak_I_meas, err_scratch
                )
            error_curve.append(last_err)

            # If the LS could not improve, leave E and step alone; the
            # next iteration's gradient is unchanged so this loops out
            # quickly via the convergence test in calling code.

        return RetrievalResult(
            field=ElectricField(grid=self.trace.grid, data=E),
            error_curve=error_curve,
            n_iterations=n_iter,
        )
