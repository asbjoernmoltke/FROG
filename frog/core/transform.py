"""
Forward XFROG model: (field, gate) -> FROGTrace.

    I[k, m] = | FFT_t[ E(t) * G(t - tau_m) ] |^2

FFT convention
--------------
All time-domain arrays are stored in *centered* order (index N//2 = t = 0).
The helpers _fft_centered / _ifft_centered apply ifftshift before the FFT
and fftshift after, so the output is also in centered order (freq = 0 at
index N//2).  This means:

  - grid.freq and FROGTrace.intensity rows are in the same monotone order.
  - No manual fftshift is needed anywhere else in the codebase.

Gate shifting
-------------
G(t - tau_m) is computed by np.roll(G, delay_indices[m]).  This is exact
(integer samples, no interpolation) because Grid.__post_init__ guarantees
that every delay is an integer multiple of dt.
"""
from __future__ import annotations

import numpy as np

from .field import ElectricField
from .trace import FROGTrace


# ---------------------------------------------------------------------------
# FFT helpers (centered convention)
# ---------------------------------------------------------------------------

def _fft_centered(x: np.ndarray) -> np.ndarray:
    """
    FFT along axis 0 for arrays in centered time order.

    Input  x[n, ...]  : centered time domain  (index N//2 = t = 0)
    Output X[k, ...]  : centered freq domain  (index N//2 = freq = 0)
    """
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(x, axes=0), axis=0),
        axes=0,
    )


def _ifft_centered(X: np.ndarray) -> np.ndarray:
    """
    IFFT along axis 0 for arrays in centered frequency order.

    Input  X[k, ...]  : centered freq domain  (index N//2 = freq = 0)
    Output x[n, ...]  : centered time domain  (index N//2 = t = 0)
    """
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(X, axes=0), axis=0),
        axes=0,
    )


# ---------------------------------------------------------------------------
# Signal field
# ---------------------------------------------------------------------------

def compute_signal_field(
    field: ElectricField,
    gate: ElectricField,
) -> np.ndarray:
    """
    Build the signal field matrix E_sig[n, m] = E[n] * G[n - shift_m].

    The gate is shifted by an exact integer number of samples for each
    delay, guaranteed by the Grid alignment invariant.

    Returns
    -------
    E_sig : complex ndarray, shape (N, M)
    """
    grid = field.grid
    G_shifts = np.array(
        [np.roll(gate.data, s) for s in grid.delay_indices]
    )  # shape (M, N)
    return field.data[:, None] * G_shifts.T  # (N, 1) * (N, M) -> (N, M)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward_model(
    field: ElectricField,
    gate: ElectricField,
) -> FROGTrace:
    """
    Compute the synthetic XFROG trace.

        I[k, m] = | FFT_t[ E(t) * G(t - tau_m) ] |^2

    Parameters
    ----------
    field : unknown pulse (or candidate during retrieval)
    gate  : known reference pulse; must share the same Grid as field

    Returns
    -------
    FROGTrace with intensity in centered frequency order (matches grid.freq).
    """
    if field.grid is not gate.grid:
        raise ValueError(
            "field and gate must share the same Grid instance. "
            "If the grids are equivalent but different objects, pass the same "
            "Grid to both constructors."
        )
    E_sig = compute_signal_field(field, gate)
    E_spec = _fft_centered(E_sig)
    return FROGTrace(grid=field.grid, intensity=np.abs(E_spec) ** 2)
