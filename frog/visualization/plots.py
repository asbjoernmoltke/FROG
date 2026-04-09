"""
Visualization utilities for FROG traces and retrieved fields.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..core.trace import FROGTrace
from ..core.field import ElectricField
from ..retrieval.legacy_xfrog.base import RetrievalResult


def plot_trace(
    trace: FROGTrace,
    title: str = "XFROG Trace",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    2-D pseudocolor plot of a FROGTrace.

    The intensity array is already in centered frequency order (as produced
    by forward_model / _fft_centered), so it can be passed to pcolormesh
    directly with grid.freq and grid.delays as axes.
    """
    grid = trace.grid
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.pcolormesh(
        grid.delays,
        grid.freq,
        trace.intensity,
        cmap="inferno",
        shading="auto",
    )
    ax.set_xlabel("Delay [time units]")
    ax.set_ylabel("Frequency [1/time units]")
    ax.set_title(title)

    if standalone:
        plt.tight_layout()
    return ax


def plot_field(
    field: ElectricField,
    label: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the intensity and unwrapped phase of a complex electric field on
    a shared time axis with a twin y-axis for the phase.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 3))

    t = field.grid.t
    ax2 = ax.twinx()
    ax.plot(t, field.norm, label=f"{label} intensity" if label else "intensity")
    ax2.plot(t, field.phase, "--", alpha=0.7, label=f"{label} phase" if label else "phase")

    ax.set_xlabel("Time [time units]")
    ax.set_ylabel("Intensity [arb.]")
    ax2.set_ylabel("Phase [rad]")
    if label:
        ax.set_title(f"Electric field — {label}")

    if standalone:
        plt.tight_layout()
    return ax


def plot_retrieval_summary(
    result: RetrievalResult,
    measured_trace: FROGTrace,
    retrieved_trace: Optional[FROGTrace] = None,
    ground_truth: Optional[ElectricField] = None,
) -> plt.Figure:
    """
    Four-panel summary figure:

      [measured trace]  [retrieved trace]
      [field comparison]  [convergence curve]

    Parameters
    ----------
    result          : output of a Retriever.retrieve() call
    measured_trace  : the FROGTrace that was given to the retriever
    retrieved_trace : FROGTrace computed from result.field (optional;
                      compute with forward_model(result.field, gate) before
                      calling this function)
    ground_truth    : known ground-truth ElectricField for validation (optional)
    """
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_meas = fig.add_subplot(gs[0, 0])
    ax_retr = fig.add_subplot(gs[0, 1])
    ax_field = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1])

    plot_trace(measured_trace, title="Measured trace", ax=ax_meas)

    if retrieved_trace is not None:
        plot_trace(retrieved_trace, title="Retrieved trace", ax=ax_retr)
    else:
        ax_retr.set_title("Retrieved trace\n(pass retrieved_trace= to display)")
        ax_retr.axis("off")

    # Field panel
    plot_field(result.field, label="retrieved", ax=ax_field)
    if ground_truth is not None:
        t = ground_truth.grid.t
        ax_field.plot(t, ground_truth.intensity, "k--", alpha=0.5, label="truth")
        ax_field.legend(fontsize=8)

    # Convergence
    ax_err.semilogy(result.error_curve, linewidth=1.2)
    ax_err.set_xlabel("Iteration")
    ax_err.set_ylabel("FROG error")
    ax_err.set_title("Convergence")
    ax_err.grid(True, which="both", alpha=0.3)

    return fig
