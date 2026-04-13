"""
Visualization utilities for FROG traces and retrieved fields.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..core.trace import FROGTrace
from ..core.field import ElectricField
from ..core.transform import forward_model
from ..retrieval.legacy_xfrog.base import RetrievalResult
from ..retrieval.shg.base import SHGRetrievalResult


# ======================================================================
# Tool functions
# ======================================================================

def crop_phase(
    field: ElectricField,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, phase) restricted to the intensity support.

    Parameters
    ----------
    field : ElectricField
    threshold : fraction of peak intensity below which to mask.

    Returns
    -------
    t_crop, phase_crop : 1-D arrays on the support region.
    """
    mask = field.norm > threshold
    return field.grid.t[mask], field.phase[mask]


def align_phase(
    field: ElectricField,
    reference: Optional[ElectricField] = None,
) -> np.ndarray:
    """Return the field data with global phase aligned.

    If *reference* is given, the global phase is rotated so that
    ``<reference, field>`` is real and positive.  Otherwise the phase is
    zeroed at the intensity peak.

    Returns
    -------
    data : complex array (same shape as ``field.data``).
    """
    data = field.data.copy()
    if reference is not None:
        inner = np.vdot(reference.data, data)
        if abs(inner) > 1e-30:
            data *= np.exp(-1j * np.angle(inner))
    else:
        peak = int(np.argmax(np.abs(data) ** 2))
        data *= np.exp(-1j * np.angle(data[peak]))
    return data


def center_peak(field: ElectricField) -> ElectricField:
    """Return a new ElectricField with argmax|E|^2 shifted to N//2.

    Also useful for removing time-translation ambiguity after blind
    retrieval.
    """
    N = field.grid.N
    peak = int(np.argmax(np.abs(field.data) ** 2))
    shift = N // 2 - peak
    return ElectricField(grid=field.grid, data=np.roll(field.data, shift))


# ======================================================================
# Plot functions
# ======================================================================

# Type that any of our result dataclasses satisfy (duck-typed)
_AnyResult = object  # RetrievalResult | BlindRetrievalResult | SHGRetrievalResult


def _resolve_field_pair(
    first,
    second=None,
    *,
    which: str = "field",
) -> tuple[ElectricField, Optional[ElectricField]]:
    """Unpack flexible (result_or_field, optional_ref) arguments.

    Supports:
        plot_pulse(result)                -> (result.field, None)
        plot_pulse(result, ref=Eref)      -> (result.field, Eref)
        plot_pulse(Erec)                  -> (Erec, None)
        plot_pulse(Erec, Eref)            -> (Erec, Eref)
    """
    if isinstance(first, ElectricField):
        rec = first
        ref = second
    else:
        # Assume result object
        rec = getattr(first, which)
        ref = second
    return rec, ref


def plot_pulse(
    pulse: Union[ElectricField, _AnyResult],
    reference: Optional[ElectricField] = None,
    *,
    which: str = "field",
    phase_threshold: float = 0.001,
    title: Optional[str] = None,
    color: str = "C0",
    ref_color: str = "k",
    lw: float = 1.0,
    ref_lw: float = 1.5,
    label: str = "retrieved",
    ref_label: str = "truth",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a single pulse: intensity on left y-axis, phase on right.

    Parameters
    ----------
    pulse : ElectricField or any retrieval result object.
    reference : optional ground-truth ElectricField for overlay.
    which : attribute name to extract from a result (``"field"`` or ``"gate"``).
    phase_threshold : intensity fraction for phase masking.
    title, color, ref_color, lw, ref_lw, label, ref_label : cosmetics.
    ax : existing axes to draw into (creates a new figure if None).

    Returns the primary (intensity) axes.
    """
    rec, ref = _resolve_field_pair(pulse, reference, which=which)

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 4))

    t = rec.grid.t

    # Intensity
    if ref is not None:
        ax.plot(t, ref.norm, color=ref_color, lw=ref_lw, label=f"|{ref_label}|²")
    ax.plot(t, rec.norm, color=color, lw=lw, label=f"|{label}|²")
    ax.set_xlabel("time")
    ax.set_ylabel("|E|² (norm)")
    ax.grid(True, alpha=0.3)

    # Phase on twin axis
    ax2 = ax.twinx()
    rec_aligned = align_phase(rec, ref)
    t_rec, ph_rec = _cropped_phase(rec_aligned, rec.norm, phase_threshold, t)
    ax2.plot(t_rec, ph_rec, color=color, ls="--", lw=lw, label=f"phase {label}")

    if ref is not None:
        ref_aligned = align_phase(ref)
        t_ref, ph_ref = _cropped_phase(ref_aligned, ref.norm, phase_threshold, t)
        # Zero both at the reference peak
        peak_idx = int(np.argmax(ref.norm))
        ph_ref_full = np.unwrap(np.angle(ref_aligned))
        ph_rec_full = np.unwrap(np.angle(rec_aligned))
        offset_ref = ph_ref_full[peak_idx]
        offset_rec = ph_rec_full[peak_idx]
        ax2.plot(t_ref, ph_ref - offset_ref, color=ref_color, ls="--", lw=ref_lw * 0.8,
                 label=f"phase {ref_label}")
        # Re-plot rec phase with same zero
        ax2.lines[-2].remove()  # remove the un-zeroed rec phase
        ax2.plot(t_rec, ph_rec - offset_rec, color=color, ls="--", lw=lw,
                 label=f"phase {label}")
    ax2.set_ylabel("phase [rad]")

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    if title:
        ax.set_title(title)
    elif standalone:
        ax.set_title(which.capitalize())

    if standalone:
        plt.tight_layout()
    return ax


def _cropped_phase(
    data: np.ndarray,
    norm: np.ndarray,
    threshold: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap phase and crop to support."""
    mask = norm > threshold
    ph = np.unwrap(np.angle(data))
    return t[mask], ph[mask]


def compare_pulse(
    retrieved: Union[ElectricField, _AnyResult],
    reference: ElectricField,
    *,
    which: str = "field",
    phase_threshold: float = 0.05,
    title: Optional[str] = None,
    color: str = "C0",
    ref_color: str = "k",
    lw: float = 1.0,
    ref_lw: float = 1.5,
    label: str = "retrieved",
    ref_label: str = "truth",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Compare a retrieved pulse against a reference (convenience wrapper).

    Identical to ``plot_pulse(retrieved, reference, ...)`` but makes the
    reference argument mandatory for clarity.
    """
    return plot_pulse(
        retrieved, reference,
        which=which,
        phase_threshold=phase_threshold,
        title=title,
        color=color, ref_color=ref_color,
        lw=lw, ref_lw=ref_lw,
        label=label, ref_label=ref_label,
        ax=ax,
    )


def plot_convergence(
    error_curves: Union[list[float], dict[str, list[float]], _AnyResult],
    *,
    title: str = "Convergence",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot FROG error vs iteration on a log-y scale.

    Parameters
    ----------
    error_curves :
        - A single list of floats (one curve).
        - A dict ``{name: [errors]}`` for multiple named curves.
        - Any retrieval result object (uses ``result.error_curve``).
    title : plot title.
    ax : existing axes (creates new figure if None).
    **kwargs : forwarded to ``ax.semilogy()``.

    Returns the axes.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 4))

    if isinstance(error_curves, dict):
        for name, ec in error_curves.items():
            ax.semilogy(ec, label=name, lw=kwargs.pop("lw", 1.3), **kwargs)
        ax.legend(fontsize=8)
    elif isinstance(error_curves, list):
        ax.semilogy(error_curves, lw=kwargs.pop("lw", 1.3), **kwargs)
    else:
        # Result object
        ax.semilogy(error_curves.error_curve, lw=kwargs.pop("lw", 1.3), **kwargs)

    ax.set_xlabel("iteration")
    ax.set_ylabel("FROG error")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    if standalone:
        plt.tight_layout()
    return ax


def plot_traces(
    measured: FROGTrace,
    retrieved: Union[ElectricField, _AnyResult],
    gate: Optional[ElectricField] = None,
    *,
    which: str = "field",
    cmap: str = "inferno",
    diff_cmap: str = "RdBu_r",
    title: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    range: Optional[tuple[float, float] | float] = None,
) -> plt.Figure:
    """Side-by-side measured trace, retrieved trace, and their difference.

    Parameters
    ----------
    measured : the measured FROGTrace.
    retrieved : a retrieval result or the retrieved ElectricField.
    gate : known gate (required for XFROG; for blind/SHG the gate is
        taken from the result object).
    which : attribute name when *retrieved* is a result object
        (``"field"`` is the default).
    cmap : colormap for the measured and retrieved panels.
    diff_cmap : colormap for the difference panel (diverging).
    title : overall figure title.
    fig : existing figure to draw into (creates a new one if None).

    Returns the figure.
    """
    # Resolve field and gate
    if isinstance(retrieved, ElectricField):
        rec_field = retrieved
        rec_gate = gate
    else:
        rec_field = getattr(retrieved, which)
        # Try to get the gate from the result (blind) or explicit arg
        rec_gate = gate or getattr(retrieved, "gate", None)
        # SHG: gate = field (only for actual SHG results)
        if rec_gate is None and isinstance(retrieved, SHGRetrievalResult):
            rec_gate = rec_field

    if rec_gate is None:
        raise ValueError(
            "Cannot compute retrieved trace: pass gate= for XFROG results "
            "(the gate is not stored in the result object)."
        )

    retr_trace = forward_model(rec_field, rec_gate).normalized()
    meas_norm = measured.normalized()

    grid = measured.grid
    diff = 10*np.log10(np.abs(meas_norm.intensity - retr_trace.intensity) + 1e-30)

    retr_trace = 10*np.log10(retr_trace.intensity + 1e-30)
    meas_norm = 10*np.log10(meas_norm.intensity + 1e-30)

    standalone = fig is None
    if standalone:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = fig.subplots(1, 3)

    
    for ax, data, ttl, cm in zip(
        axes,
        [meas_norm, retr_trace, diff],
        ["Measured", "Retrieved", "Difference"],
        [cmap, cmap, diff_cmap],
    ):
        if range is None:
            kw = dict(vmin=data.max() - 30, vmax=data.max())  # show up to 30 dB below peak
        else:
            if isinstance(range, (int, float)):
                range = (data.max() - range, data.max())
            kw = dict(vmin=range[0], vmax=range[1])
        if cm == diff_cmap:
            # Center the diverging colormap at zero
            vmin = kw["vmin"]
            vmax = kw["vmax"]
            vmin = vmax - (abs(vmax - vmin))/2
            kw.update(vmin=vmin, vmax=vmax)

        ax.pcolormesh(grid.delays, grid.freq, data, cmap=cm, shading="auto", **kw)
        ax.set_xlabel("Delay")
        ax.set_ylabel("Frequency")
        ax.set_title(ttl)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig


# ======================================================================
# Legacy functions (kept for backwards compat)
# ======================================================================

def plot_trace(
    trace: FROGTrace,
    title: str = "XFROG Trace",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """2-D pseudocolor plot of a FROGTrace."""
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
    """Plot intensity and unwrapped phase of a field (legacy)."""
    return plot_pulse(field, title=label, label=label or "field", ax=ax)


def plot_retrieval_summary(
    result: RetrievalResult,
    measured_trace: FROGTrace,
    retrieved_trace: Optional[FROGTrace] = None,
    ground_truth: Optional[ElectricField] = None,
) -> plt.Figure:
    """Four-panel summary: measured trace, retrieved trace, field, convergence."""
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

    plot_pulse(result, ground_truth, title="Field", ax=ax_field)
    plot_convergence(result, ax=ax_err)

    return fig
