"""
Fast PIE / batched-PIE retriever for XFROG with a known gate.

Strict ePIE is sequential: each delay's update reads the *current* E
that was just modified by the previous delay's update.  That dependency
chain prevents the (M, N) batch vectorization that drives all the other
fast retrievers, and forces M length-N FFTs per "iteration" instead of
one batched call.

We provide two modes here:

* `mode="batched"` (default for fast_retrieval).  All M delay updates
  are computed from the *same* E and summed (averaged with the gate
  energy at each n).  This is no longer textbook ePIE — it's a parallel
  PIE that converges roughly like GPA but with the ePIE preconditioner.
  Per iteration it costs one forward + one inverse batched FFT, the
  same as fast GPA.

* `mode="strict"`.  The exact ePIE per-delay loop, kept here for
  validation against the reference.  It allocates a single (N,) plan
  pair (no batched FFT) and runs M of them per iteration; this is
  ~M times slower than batched mode and is only useful when you need
  bit-for-bit equivalence with the reference PIE.

Batched-mode update derivation
------------------------------
ePIE update for one delay m (gate held fixed):

    psi_m   = E * G_m
    psi'_m  = IFFT( sqrt(I_meas[:,m]) * Psi_m / |Psi_m| )
    delta_m = psi'_m - psi_m
    E      <- E + alpha * conj(G_m) * delta_m / max|G|^2

Summed over all delays from a single starting E:

    E_new = E + alpha / max|G|^2 * sum_m  conj(G_m) * delta_m

This is exactly what fast GPA computes in its field-update einsum,
except (a) the sum operates on the *delta* rather than the full
projected signal, and (b) the denominator is the scalar max|G|^2 instead
of the per-n GPA sum.  In practice these two changes make the batched
version slightly slower-converging than fast GPA but much more robust
to noisy / sparse delay grids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyfftw

from .base import Retriever, RetrievalResult
from ._common import FastWorkspace, fast_frog_error
from ...core.field import ElectricField


@dataclass
class PIE(Retriever):
    """
    Fast PIE retriever (XFROG, known gate).  Arbitrary (M, N).

    Parameters
    ----------
    mode : {"batched", "strict"}
        "batched" runs the parallel PIE update (one batched FFT pair per
        iteration, the default).  "strict" runs textbook ePIE one delay
        at a time and is provided for validation only.
    alpha : float
        Object update strength.  alpha=1 reproduces standard ePIE.
    """

    dtype: np.dtype = np.complex64
    workers: int = -1
    error_every: int = 10

    mode: str = "batched"
    alpha: float = 0.5

    _ws: Optional[FastWorkspace] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in ("batched", "strict"):
            raise ValueError(f"PIE mode must be 'batched' or 'strict'; got {self.mode!r}")
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
        if self.mode == "batched":
            return self._retrieve_batched(n_iter, initial_field, seed)
        return self._retrieve_strict(n_iter, initial_field, seed)

    # ------------------------------------------------------------------
    # Batched parallel PIE
    # ------------------------------------------------------------------
    def _retrieve_batched(self, n_iter, initial_field, seed) -> RetrievalResult:
        ws = self._ws
        N, M = ws.N, ws.M
        cdtype = ws.dtype
        rdtype = ws.sqrt_I_meas.dtype

        G = ws.G_shifts
        G_conj = ws.G_shifts_conj
        sqrt_I_meas = ws.sqrt_I_meas
        I_meas = ws.I_meas
        sum_I_meas_sq = ws.sum_I_meas_sq
        peak_I_meas = ws.peak_I_meas

        # Batched-PIE step: a single iteration accumulates M ePIE updates
        # at once, so the effective step size is M-fold the per-delay
        # ePIE step.  Using max|G|^2 as the denominator (textbook ePIE)
        # makes the batched update wildly unstable; instead we use the
        # GPA denominator sum_m|G_m(t)|^2, which is exactly the effective
        # gate energy seen at each time point when all M deltas are
        # averaged.  This makes batched PIE numerically equivalent to a
        # GPA step with the residual-form (delta) field update.
        inv_denom = ws.inv_denom_gpa

        in_buf = ws.fft1d_in
        out_buf = ws.fft1d_out

        I_buf = np.empty((M, N), dtype=rdtype)
        err_scratch = np.empty((M, N), dtype=rdtype)
        psi_buf = np.empty((M, N), dtype=cdtype)   # holds psi (E*G) for the delta

        if initial_field is not None:
            E = np.asarray(initial_field.data, dtype=cdtype).copy()
        else:
            E = ws.random_initial_field(seed)

        error_curve: list[float] = []
        last_err = 0.0

        for it in range(n_iter):
            # ---- Forward: psi = E * G, then S = FFT(psi) ----
            np.multiply(E[None, :], G, out=in_buf)
            psi_buf[...] = in_buf                      # save psi for the delta
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

            # ---- Inverse FFT: in_buf <- psi' ----
            ws.ifft1d()

            # ---- Batched ePIE update: delta = psi' - psi, summed ----
            np.subtract(in_buf, psi_buf, out=in_buf)   # in_buf now holds delta
            update = np.einsum("mn,mn->n", in_buf, G_conj, optimize=True)
            E = (E + self.alpha * inv_denom * update).astype(cdtype, copy=False)

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

    # ------------------------------------------------------------------
    # Strict ePIE (sequential).  Same algorithm as the reference PIE,
    # but built on a per-delay pyfftw plan and the precomputed shifted
    # gates from the workspace.  Roughly M times slower than batched
    # mode at large M; provided for parity with the reference.
    # ------------------------------------------------------------------
    def _retrieve_strict(self, n_iter, initial_field, seed) -> RetrievalResult:
        ws = self._ws
        N, M = ws.N, ws.M
        cdtype = ws.dtype
        rdtype = ws.sqrt_I_meas.dtype

        G = ws.G_shifts
        G_conj = ws.G_shifts_conj
        sqrt_I_meas = ws.sqrt_I_meas
        I_meas = ws.I_meas
        sum_I_meas_sq = ws.sum_I_meas_sq
        peak_I_meas = ws.peak_I_meas
        alpha_over_gmax = self.alpha / max(ws.gate_max_sq, np.finfo(rdtype).tiny)

        # Per-delay pyfftw plan (length-N vectors).
        psi_old = pyfftw.empty_aligned(N, dtype=cdtype)
        Psi = pyfftw.empty_aligned(N, dtype=cdtype)
        plan_fwd = pyfftw.FFTW(psi_old, Psi, direction="FFTW_FORWARD",
                               flags=("FFTW_MEASURE",),
                               threads=ws.threads)
        psi_new_buf = pyfftw.empty_aligned(N, dtype=cdtype)
        plan_inv = pyfftw.FFTW(Psi, psi_new_buf, direction="FFTW_BACKWARD",
                               flags=("FFTW_MEASURE", "FFTW_DESTROY_INPUT"),
                               threads=ws.threads)

        I_buf_full = np.empty((M, N), dtype=rdtype)
        err_scratch = np.empty((M, N), dtype=rdtype)
        mag_buf = np.empty(N, dtype=rdtype)
        in_buf_full = ws.fft1d_in
        out_buf_full = ws.fft1d_out

        if initial_field is not None:
            E = np.asarray(initial_field.data, dtype=cdtype).copy()
        else:
            E = ws.random_initial_field(seed)

        rng_order = np.random.default_rng(seed + 1)

        # Note: avoid `+=` on closed-over names; Python would treat them
        # as locals.  We mutate the workspace buffers via `np.add(out=...)`.
        def _full_error(E_curr):
            np.multiply(E_curr[None, :], G, out=in_buf_full)
            ws.fft1d()
            np.multiply(out_buf_full.real, out_buf_full.real, out=I_buf_full)
            np.add(I_buf_full, out_buf_full.imag * out_buf_full.imag, out=I_buf_full)
            return fast_frog_error(
                I_buf_full, I_meas, sum_I_meas_sq, peak_I_meas, err_scratch
            )

        error_curve: list[float] = [_full_error(E)]

        for _ in range(n_iter):
            order = rng_order.permutation(M)
            for m in order:
                Gm = G[m]
                np.multiply(E, Gm, out=psi_old)        # cache psi_old in plan input
                plan_fwd()                              # Psi = FFT(psi_old)

                np.abs(Psi, out=mag_buf)
                np.maximum(mag_buf, 1e-20, out=mag_buf)
                np.divide(sqrt_I_meas[m], mag_buf, out=mag_buf)
                Psi *= mag_buf
                plan_inv()                              # psi_new_buf = IFFT(Psi)

                # ePIE object update.  psi_old is still the original psi
                # because plan_fwd writes its output to Psi, not back to its input.
                E += alpha_over_gmax * G_conj[m] * (psi_new_buf - psi_old)

            error_curve.append(_full_error(E))

        return RetrievalResult(
            field=ElectricField(grid=self.trace.grid, data=E),
            error_curve=error_curve,
            n_iterations=n_iter,
        )
