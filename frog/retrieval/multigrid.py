"""
Multigrid (coarse-to-fine) retrieval and algorithm chaining.

The multigrid wrapper accelerates convergence at large N by solving at
progressively finer resolutions.  Each level warm-starts from the
previous level's solution, interpolated onto the finer grid.

Optional algorithm chaining runs a second retriever at the final
resolution to refine the solution when the primary algorithm stalls.

Example — XFROG with multigrid + gradient refinement::

    from frog.retrieval.xfrog import GPA, GradientDescent
    from frog.retrieval.multigrid import multigrid_retrieve

    result = multigrid_retrieve(
        dataset.trace, GPA,
        gate=dataset.gate,
        n_iter=200,
        refinement_cls=GradientDescent,
        refinement_iter=100,
    )

Example — blind XFROG::

    from frog.retrieval.blind_xfrog import BlindGPA
    from frog.retrieval.multigrid import multigrid_retrieve

    result = multigrid_retrieve(
        dataset.trace, BlindGPA,
        n_iter=300,
    )
"""
from __future__ import annotations

import time
from typing import Optional, Sequence, Union

import numpy as np

from ..core.grid import Grid
from ..core.field import ElectricField
from ..core.trace import FROGTrace


# ======================================================================
# Helpers
# ======================================================================

def _auto_levels(N: int, coarsest: int = 64) -> list[int]:
    """Build power-of-two levels from *coarsest* up to *N*."""
    levels: list[int] = []
    n = min(coarsest, N)
    n = max(8, 2 ** int(np.round(np.log2(n))))
    while n < N:
        levels.append(n)
        n *= 2
    levels.append(N)
    return levels


def _resample_trace(trace: FROGTrace, N_target: int, dt: float) -> FROGTrace:
    """Resample a FROGTrace to a different number of freq points.

    Delays whose shift index exceeds N_target//2 are dropped to avoid
    wrap-around aliasing in np.roll.
    """
    N_src = trace.grid.N

    # Keep only delays valid at the target N
    max_idx = N_target // 2
    valid = np.abs(trace.grid.delay_indices) <= max_idx
    delays = trace.grid.delays[valid]

    # Frequency axes (centered, monotone)
    freq_src = np.fft.fftshift(np.fft.fftfreq(N_src, dt))
    freq_tgt = np.fft.fftshift(np.fft.fftfreq(N_target, dt))

    # Resample intensity along frequency axis for each delay
    I_src = trace.intensity[:, valid]  # (N_src, M_valid)
    M = I_src.shape[1]
    I_tgt = np.empty((N_target, M))
    for m in range(M):
        I_tgt[:, m] = np.interp(freq_tgt, freq_src, I_src[:, m])

    grid = Grid(N=N_target, dt=dt, delays=delays)
    return FROGTrace(grid=grid, intensity=np.maximum(I_tgt, 0.0))


def _resample_field(field: ElectricField, target_grid: Grid) -> ElectricField:
    """Interpolate an ElectricField onto *target_grid*.

    Zero-pads outside the source time window.
    """
    t_src = field.grid.t
    t_tgt = target_grid.t

    re = np.interp(t_tgt, t_src, field.data.real, left=0.0, right=0.0)
    im = np.interp(t_tgt, t_src, field.data.imag, left=0.0, right=0.0)
    return ElectricField(grid=target_grid, data=(re + 1j * im))


# ======================================================================
# Main entry point
# ======================================================================

def multigrid_retrieve(
    trace: FROGTrace,
    retriever_cls: type,
    *,
    gate: Optional[ElectricField] = None,
    levels: Optional[Sequence[int]] = None,
    n_iter: Union[int, Sequence[int]] = 200,
    seed: int = 0,
    refinement_cls: Optional[type] = None,
    refinement_iter: int = 100,
    refinement_kwargs: Optional[dict] = None,
    verbose: bool = True,
    **retriever_kwargs,
):
    """Run coarse-to-fine retrieval with optional algorithm chaining.

    Parameters
    ----------
    trace : FROGTrace
        Measured trace at the target (finest) resolution.
    retriever_cls : type
        Retriever class (e.g. ``GPA``, ``BlindGPA``, ``SHGGPA``).
    gate : ElectricField, optional
        Known gate for XFROG retrievers.
    levels : list of int, optional
        Grid sizes for each multigrid level, ascending.
        Default: powers of two from 64 up to ``trace.grid.N``.
    n_iter : int or list of int
        Iterations per level (single value = same for all levels).
    seed : int
        Random seed for the initial guess at the coarsest level.
    refinement_cls : type, optional
        A second retriever class to run at the final resolution after
        the primary retriever finishes (algorithm chaining).
    refinement_iter : int
        Iterations for the refinement pass.
    refinement_kwargs : dict, optional
        Constructor kwargs for the refinement retriever (overrides
        *retriever_kwargs* for the refinement step).
    verbose : bool
        Print progress per level.
    **retriever_kwargs
        Forwarded to the primary retriever constructor (e.g. ``dtype``,
        ``workers``, ``stop_target``).

    Returns
    -------
    The retrieval result (same type as ``retriever_cls`` produces), with
    ``error_curve`` spanning all levels and ``exec_time`` covering the
    full run.
    """
    N_target = trace.grid.N
    dt = trace.grid.dt

    if levels is None:
        levels = _auto_levels(N_target)
    if isinstance(n_iter, int):
        n_iter_list = [n_iter] * len(levels)
    else:
        n_iter_list = list(n_iter)

    all_errors: list[float] = []
    result = None
    is_blind = False
    t_start = time.perf_counter()

    for i, N in enumerate(levels):
        # ---- Build trace (and gate) at this level ----
        if N == N_target:
            level_trace = trace
            level_gate = gate
        else:
            level_trace = _resample_trace(trace, N, dt)
            level_gate = (
                _resample_field(gate, level_trace.grid)
                if gate is not None else None
            )

        # ---- Create retriever ----
        ctor_kw = dict(trace=level_trace, **retriever_kwargs)
        if level_gate is not None:
            ctor_kw["gate"] = level_gate
        retriever = retriever_cls(**ctor_kw)

        # ---- Warm-start from previous level ----
        ret_kw: dict = dict(n_iter=n_iter_list[i], seed=seed)
        if result is not None:
            ret_kw["initial_field"] = _resample_field(
                result.field, level_trace.grid
            )
            if is_blind:
                ret_kw["initial_gate"] = _resample_field(
                    result.gate, level_trace.grid
                )

        result = retriever.retrieve(**ret_kw)
        is_blind = hasattr(result, "gate")
        all_errors.extend(result.error_curve)

        if verbose:
            print(
                f"  Level {i + 1}/{len(levels)}: N={N:4d}, "
                f"error={result.error_curve[-1]:.2e}, "
                f"{result.exec_time:.2f}s"
            )

    # ---- Optional refinement with a different algorithm ----
    if refinement_cls is not None:
        ref_ctor_kw: dict = dict(trace=trace)
        if gate is not None:
            ref_ctor_kw["gate"] = gate
        if refinement_kwargs:
            ref_ctor_kw.update(refinement_kwargs)
        refiner = refinement_cls(**ref_ctor_kw)

        ref_ret_kw: dict = dict(
            n_iter=refinement_iter, seed=seed,
            initial_field=result.field,
        )
        if is_blind:
            ref_ret_kw["initial_gate"] = result.gate

        result = refiner.retrieve(**ref_ret_kw)
        all_errors.extend(result.error_curve)

        if verbose:
            print(
                f"  Refinement ({refinement_cls.__name__}): "
                f"error={result.error_curve[-1]:.2e}, "
                f"{result.exec_time:.2f}s"
            )

    result.error_curve = all_errors
    result.exec_time = time.perf_counter() - t_start
    return result
