"""
FROGTrace: 2-D intensity array (frequency × delay).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .grid import Grid


@dataclass
class FROGTrace:
    """
    XFROG trace, measured or synthetic.

    intensity[k, m] = I(freq[k], delay[m])

    Row index k  → frequency axis, aligned with grid.freq (centered order).
    Column index m → delay axis, aligned with grid.delays.

    Both axes are in the same order as Grid.freq and Grid.delays, so
    intensity can be plotted directly with pcolormesh(grid.delays, grid.freq,
    intensity) without any reordering.
    """

    grid: Grid
    intensity: np.ndarray  # shape (N, M), real, non-negative

    def __post_init__(self) -> None:
        self.intensity = np.asarray(self.intensity, dtype=float)
        expected = (self.grid.N, self.grid.M)
        if self.intensity.shape != expected:
            raise ValueError(
                f"intensity.shape {self.intensity.shape} does not match expected "
                f"{expected} (grid.N={self.grid.N}, grid.M={self.grid.M})."
            )
        if self.intensity.min() < 0:
            raise ValueError("Trace intensities must be non-negative.")

    def normalized(self) -> "FROGTrace":
        """Return a copy scaled so the peak equals 1."""
        return FROGTrace(grid=self.grid, intensity=self.intensity / self.intensity.max())
