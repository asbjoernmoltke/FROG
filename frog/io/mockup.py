"""
Offline dataset generator: synthesize an XFROG trace from a known pulse pair.

MockDataset is the primary entry point for algorithm development and
validation.  It creates a self-consistent (grid, field, gate, trace) tuple
from a handful of physical parameters, with optional additive noise.

All time parameters share the unit of `dt` (e.g. fs if dt is in fs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import Grid
from ..core.field import ElectricField
from ..core.trace import FROGTrace
from ..core.transform import forward_model


@dataclass
class MockDataset:
    """
    Synthetic XFROG dataset.

    Attributes
    ----------
    grid  : Grid            — shared discretization for all arrays
    field : ElectricField   — ground-truth unknown pulse (for validation)
    gate  : ElectricField   — known reference gate pulse
    trace : FROGTrace       — noiseless (or noisy) synthetic trace, peak = 1
    """

    grid: Grid
    field: ElectricField
    gate: ElectricField
    trace: FROGTrace

    @classmethod
    def gaussian_pulse(
        cls,
        N: int = 256,
        dt: float = 1.0,
        n_delays: int = 128,
        field_duration: float = 20.0,
        field_chirp: float = 0.0,
        gate_duration: float = 15.0,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> "MockDataset":
        """
        Build a dataset with Gaussian field and gate pulses.

        The delay axis is symmetric around zero with step = dt, which
        trivially satisfies the Grid alignment invariant.

        Parameters
        ----------
        N              : number of time/frequency samples
        dt             : time step [arbitrary unit]
        n_delays       : number of delay points (symmetric, so the actual
                         axis runs from -n_delays//2 to +n_delays//2)
        field_duration : intensity FWHM of the unknown pulse [dt units]
        field_chirp    : quadratic phase coefficient [1/dt^2]
        gate_duration  : intensity FWHM of the gate pulse [dt units]
        noise_level    : additive Gaussian noise amplitude relative to
                         trace peak (0 = noiseless)
        seed           : random seed for reproducible noise
        """
        half = n_delays // 2
        delays = np.arange(-half, half + 1, dtype=float) * dt  # step = dt exactly

        grid = Grid(N=N, dt=dt, delays=delays)

        field = ElectricField.gaussian(
            grid, duration=field_duration, chirp=field_chirp
        )
        gate = ElectricField.gaussian(grid, duration=gate_duration)

        trace = forward_model(field, gate)

        if noise_level > 0.0:
            rng = np.random.default_rng(seed)
            peak = trace.intensity.max()
            noise = rng.normal(0.0, noise_level * peak, trace.intensity.shape)
            noisy = np.clip(trace.intensity + noise, 0.0, None)
            trace = FROGTrace(grid=grid, intensity=noisy)

        return cls(
            grid=grid,
            field=field,
            gate=gate,
            trace=trace.normalized(),
        )

    @classmethod
    def from_field(
        cls,
        field: "ElectricField",
        gate: "ElectricField | None" = None,
        gate_duration: float | None = None,
        n_delays: int | None = None,
        delay_stride: int = 1,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> "MockDataset":
        """
        Build a synthetic XFROG dataset from a user-supplied complex field.

        The field carries its own Grid (e.g. loaded from a CSV via
        `frog.io.load_field_csv`).  The delay axis is constructed
        symmetrically with step `dt`, exactly as in `gaussian_pulse`,
        so the integer-step grid invariant is preserved.

        Parameters
        ----------
        field
            The unknown pulse to use as ground truth.  Its Grid is
            adopted as the dataset Grid.
        gate
            Optional explicit gate field on the same Grid.  If omitted,
            a Gaussian gate is built from `gate_duration` (which then
            becomes mandatory).
        gate_duration
            FWHM of the default Gaussian gate, in `dt` units.  Ignored
            when `gate` is supplied.
        n_delays
            Number of delay points (symmetric).  Defaults to grid.N,
            which gives M = N + 1 — the same convention as
            gaussian_pulse.
        noise_level
            Additive Gaussian noise on the trace, relative to its peak.
        seed
            RNG seed for the noise.
        """
        src_grid = field.grid

        if n_delays is None:
            n_delays = src_grid.N
        half = n_delays // 2
        delays = np.arange(-half, half + 1, dtype=float) * (src_grid.dt * delay_stride)

        # Rebuild the grid with the requested delay axis but the same
        # N / dt, so the field and gate stay sample-compatible.
        grid = Grid(N=src_grid.N, dt=src_grid.dt, delays=delays)
        # Re-bind the field to the new grid (same data, new Grid object).
        field = ElectricField(grid=grid, data=np.asarray(field.data))

        if gate is None:
            if gate_duration is None:
                raise ValueError(
                    "from_field: pass either `gate=` or `gate_duration=` "
                    "(needed to build the default Gaussian gate)."
                )
            gate = ElectricField.gaussian(grid, duration=gate_duration)
        else:
            # Adopt the new grid object so trace/field/gate share identity
            # (Retriever.__post_init__ checks `is`).
            gate = ElectricField(grid=grid, data=np.asarray(gate.data))

        trace = forward_model(field, gate)

        if noise_level > 0.0:
            rng = np.random.default_rng(seed)
            peak = trace.intensity.max()
            noise = rng.normal(0.0, noise_level * peak, trace.intensity.shape)
            noisy = np.clip(trace.intensity + noise, 0.0, None)
            trace = FROGTrace(grid=grid, intensity=noisy)

        return cls(
            grid=grid,
            field=field,
            gate=gate,
            trace=trace.normalized(),
        )
