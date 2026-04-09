"""
ElectricField: complex envelope of a laser pulse on a Grid.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .grid import Grid


@dataclass
class ElectricField:
    """
    Complex envelope of a laser pulse sampled on a Grid.

    `data[n]` is the complex amplitude at time `grid.t[n]`.  Arrays are
    stored in centered order: data[N//2] corresponds to t = 0.
    """

    grid: Grid
    data: np.ndarray  # complex envelope, shape (N,)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=complex)
        if self.data.shape != (self.grid.N,):
            raise ValueError(
                f"data.shape {self.data.shape} does not match grid.N={self.grid.N}."
            )

    @property
    def intensity(self) -> np.ndarray:
        """Instantaneous intensity |E(t)|^2. Shape (N,)."""
        return np.abs(self.data) ** 2

    @property
    def phase(self) -> np.ndarray:
        """Unwrapped temporal phase [rad]. Shape (N,)."""
        return np.unwrap(np.angle(self.data))
    
    @property
    def norm(self):
        return self.intensity * (1/np.max(self.intensity))

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def gaussian(
        cls,
        grid: Grid,
        duration: float,
        chirp: float = 0.0,
        center: float = 0.0,
    ) -> "ElectricField":
        """
        Chirped Gaussian pulse.

            E(t) = exp(-2 ln2 * (t - center)^2 / duration^2)
                   * exp(i * chirp * (t - center)^2)

        Parameters
        ----------
        duration : FWHM of the intensity profile [same unit as grid.dt]
        chirp    : coefficient of the quadratic temporal phase [1/time^2]
        center   : temporal centre [same unit as grid.dt]
        """
        t = grid.t - center
        sigma = duration / (2.0 * np.sqrt(np.log(2.0)))
        envelope = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
        phase = chirp * t ** 2
        return cls(grid=grid, data=envelope * np.exp(1j * phase))
