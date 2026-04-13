"""
Fast GPA retriever for XFROG with a known gate.

Uses pyfftw batched 1D plans along the time axis, sign-trick absorbed
into the precomputed shifted-gate matrix, alloc-free FROG error, and
preallocated scratch buffers owned by a `FastWorkspace`.

Note on FFT size
----------------
pyfftw (via FFTW) is dramatically faster when N is a product of small
primes — powers of two give the best performance.  At N=1024 this
retriever runs ~14 ms/iter on a typical desktop; at N=1023 it drops to
~40 ms/iter for the same algorithm because 1023 = 3·11·31 takes a slow
code path.  If you control the measurement grid, pick N = 2^k.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import Retriever, RetrievalResult
from ._common import FastWorkspace, fast_frog_error
from ...core.field import ElectricField


@dataclass
class GPA(Retriever):
    """
    Fast GPA retriever (XFROG, known gate).  Arbitrary (M, N).
    """

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
        sqrt_I_meas = ws.sqrt_I_meas          # (M, N) real
        I_meas = ws.I_meas                    # (M, N) real
        inv_denom = ws.inv_denom_gpa          # (N,)   real
        sum_I_meas_sq = ws.sum_I_meas_sq
        peak_I_meas = ws.peak_I_meas

        # Owned pyfftw plan buffers.  in_buf holds E_sig, out_buf holds E_spec.
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

            # ---- Field update ----
            numerator = np.einsum("mn,mn->n", in_buf, G_conj, optimize=True)
            E = (numerator * inv_denom).astype(cdtype, copy=False)

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
