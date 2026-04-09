"""
Grid: shared discretization for time, frequency, and delay axes.

The central invariant enforced here is that every delay value must be an
exact integer multiple of dt.  This guarantees that the gate G(t - tau_m)
can be computed by a plain np.roll (integer sample shift) with no
interpolation anywhere in the forward model or retrieval algorithms.

Convention: arrays live in *centered* order, i.e. index N//2 corresponds
to t = 0 in the time domain and freq = 0 after FFT.  See transform.py for
the FFT helper that respects this convention.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field as dc_field


@dataclass
class Grid:
    """
    Shared discretization for a FROG/XFROG experiment.

    Parameters
    ----------
    N      : number of time and frequency samples (same, linked by FFT)
    dt     : time step [arbitrary consistent unit, e.g. fs]
    delays : 1-D array of delay values [same unit as dt]

    The constructor validates that every entry in `delays` is within 1e-6
    samples of an integer multiple of dt and stores the rounded integer
    indices as `delay_indices`.  These indices are what the forward model
    uses for np.roll so that the alignment constraint is visible at the
    point where it is actually enforced.
    """

    N: int
    dt: float
    delays: np.ndarray

    delay_indices: np.ndarray = dc_field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.delays = np.asarray(self.delays, dtype=float)

        raw = self.delays / self.dt
        rounded = np.round(raw).astype(int)

        misaligned = ~np.isclose(raw, rounded, atol=1e-6)
        if misaligned.any():
            bad = np.where(misaligned)[0]
            raise ValueError(
                f"All delays must be exact integer multiples of dt={self.dt:.4g}.\n"
                f"  Misaligned indices : {bad}\n"
                f"  Delay values       : {self.delays[bad]}\n"
                f"  Fractional residuals: {(raw - rounded)[bad]}\n"
                f"Use Grid.snap_delays() to round experimental data to the grid."
            )

        self.delay_indices = rounded

    # ------------------------------------------------------------------
    # Derived axes (read-only properties, computed on demand)
    # ------------------------------------------------------------------

    @property
    def M(self) -> int:
        """Number of delay points."""
        return len(self.delays)

    @property
    def t(self) -> np.ndarray:
        """Time axis [dt units], centered at zero. Shape (N,)."""
        return (np.arange(self.N) - self.N // 2) * self.dt

    @property
    def freq(self) -> np.ndarray:
        """
        Ordinary frequency axis [1/dt units], centered and monotonically
        increasing.  Matches the row ordering of FROGTrace.intensity as
        produced by _fft_centered() in transform.py.  Shape (N,).
        """
        return np.fft.fftshift(np.fft.fftfreq(self.N, self.dt))

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dnu(cls, N: int, dnu: float, delays: np.ndarray) -> "Grid":
        """
        Build a Grid from a spectrometer frequency bin spacing dnu.

        The FFT relationship fixes dt = 1 / (N * dnu), which gives a time
        window of T = N*dt = 1/dnu and a frequency window of N*dnu = 1/dt.
        """
        return cls(N=N, dt=1.0 / (N * dnu), delays=delays)

    @classmethod
    def snap_delays(cls, N: int, dt: float, delays_raw: np.ndarray) -> "Grid":
        """
        Build a Grid by rounding arbitrary delay values to the nearest
        integer multiple of dt.

        Useful when loading experimental data where the delay stage step
        is close to dt but not exact due to mechanical imprecision.  The
        caller is responsible for deciding whether the rounding error is
        acceptable for their application.
        """
        snapped = np.round(delays_raw / dt) * dt
        return cls(N=N, dt=dt, delays=snapped)
