"""
One-shot generator for a non-trivial complex E-field test target.

Run once to (re)create `sample_pulse.csv`.  The synthesized pulse is a
coherent superposition of two chirped Gaussians (different durations,
arrival times, and quadratic chirps) plus a small cubic phase, giving
an asymmetric intensity profile and a non-trivial spectral phase.
"""
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frog.core.grid import Grid
from frog.core.field import ElectricField
from frog.io import save_field_csv


def make_pulse(N: int = 512, dt: float = 0.25):
    half = N // 2
    delays = np.arange(-half, half + 1, dtype=float) * dt   # placeholder
    grid = Grid(N=N, dt=dt, delays=delays)
    t = grid.t

    # Pulse 1: short, slightly chirped, centered at -10 fs
    sigma1 = 6.0
    chirp1 = 0.02
    p1 = np.exp(-((t + 10.0) ** 2) / (2 * sigma1 ** 2)) * np.exp(
        1j * (chirp1 * (t + 10.0) ** 2)
    )

    # Pulse 2: longer, more strongly chirped, centered at +6 fs, ~0.7x amplitude
    sigma2 = 10.0
    chirp2 = -0.04
    p2 = 0.7 * np.exp(-((t - 6.0) ** 2) / (2 * sigma2 ** 2)) * np.exp(
        1j * (chirp2 * (t - 6.0) ** 2 + 0.6 * np.pi)
    )

    # Add a global cubic phase (third-order spectral phase mimic in time domain)
    cubic = np.exp(1j * 0.001 * t ** 3)

    data = (p1 + p2) * cubic
    data /= np.max(np.abs(data))
    return ElectricField(grid=grid, data=data)


def main():
    out = Path(__file__).with_name("sample_pulse.csv")
    field = make_pulse()
    save_field_csv(out, field)
    print(f"wrote {out}  (N={field.grid.N}, dt={field.grid.dt})")


if __name__ == "__main__":
    main()
