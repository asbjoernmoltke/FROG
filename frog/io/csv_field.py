"""
Read / write a complex `ElectricField` from a plain CSV file.

File format
-----------
Two numeric columns (real part, imaginary part), one row per time
sample, in the same centered order as `ElectricField.data` (so row
N//2 corresponds to t = 0).  An optional single-line header is
detected and skipped automatically.

Example::

    real,imag
    -0.001234,0.000000
    -0.001112,0.000045
    ...

The file does not store a time axis: the calling code is responsible
for providing a Grid (or a `dt` from which a default centered grid is
built).  This keeps the format trivially editable in any spreadsheet
and avoids the headaches of round-tripping floating-point time stamps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..core.grid import Grid
from ..core.field import ElectricField


PathLike = Union[str, Path]


def _looks_like_header(first_line: str) -> bool:
    """Return True if the first line is non-numeric (e.g. 'real,imag')."""
    parts = first_line.strip().split(",")
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return False
    except ValueError:
        return True


def load_field_csv(
    path: PathLike,
    grid: Optional[Grid] = None,
    dt: Optional[float] = None,
    delimiter: str = ",",
) -> ElectricField:
    """
    Load a complex E-field from a 2-column (real, imag) CSV.

    Parameters
    ----------
    path
        File to read.
    grid
        Pre-built `Grid` to attach the field to.  Its `N` must equal
        the number of rows in the CSV.  If omitted, a centered grid
        with `dt` is constructed.
    dt
        Time step used when `grid` is None.  Required in that case.
    delimiter
        CSV delimiter (default ",").  Use "\\t" for tab-separated files.

    Returns
    -------
    ElectricField
        On the supplied or constructed grid.
    """
    p = Path(path)
    with p.open("r") as f:
        first = f.readline()
    skip = 1 if _looks_like_header(first) else 0

    arr = np.loadtxt(p, delimiter=delimiter, skiprows=skip, ndmin=2)
    if arr.shape[1] < 2:
        raise ValueError(
            f"{p}: expected at least 2 columns (real, imag); got shape {arr.shape}."
        )
    data = arr[:, 0].astype(np.complex128) + 1j * arr[:, 1]
    N = data.shape[0]

    if grid is None:
        if dt is None:
            raise ValueError(
                "load_field_csv: provide either `grid=` or `dt=` "
                "(needed to construct a default centered Grid)."
            )
        half = N // 2
        delays = np.arange(-half, N - half, dtype=float) * dt
        grid = Grid(N=N, dt=dt, delays=delays)
    else:
        if grid.N != N:
            raise ValueError(
                f"{p}: CSV has {N} rows but grid.N = {grid.N}."
            )

    return ElectricField(grid=grid, data=data)


def load_field_csv_with_time(
    path: PathLike,
    N: int,
    delimiter: str = ",",
    center_pulse: bool = True,
    crop_threshold: Optional[float] = 1e-3,
    crop_padding: float = 2.0,
) -> ElectricField:
    """
    Load a complex E-field from a 3-column (t, real, imag) CSV and
    resample it onto a centered N-point grid.

    The input rows must be sorted by time but may be arbitrarily (non-
    uniformly) spaced; real and imaginary parts are linearly
    interpolated onto a uniform grid spanning the (optionally cropped)
    time window with N samples.

    Parameters
    ----------
    path
        File to read.
    N
        Number of grid points for the resampled field.
    delimiter
        CSV delimiter (default ",").
    center_pulse
        If True, roll the resampled field so that its peak intensity
        sits at index N//2.
    crop_threshold
        Fraction of peak intensity below which the pulse is considered
        zero.  The time window is cropped to the support of the pulse
        (where |E|^2 >= crop_threshold * max|E|^2) plus ``crop_padding``
        times the support width on each side.  Set to ``None`` to
        disable cropping and use the full CSV time range.
    crop_padding
        Extra margin on each side of the detected support, expressed as
        a multiple of the support width.  Default 2.0 gives a total
        window of ~5x the pulse support, which is usually enough to
        capture the tails without wasting resolution on distant zeros.

    Returns
    -------
    ElectricField on a Grid whose ``dt`` is set from the resampled axis
    and whose default delay axis is centered with the same ``dt``.
    """
    p = Path(path)
    with p.open("r") as f:
        first = f.readline()
    skip = 1 if _looks_like_header(first) else 0

    arr = np.loadtxt(p, delimiter=delimiter, skiprows=skip, ndmin=2)
    if arr.shape[1] < 3:
        raise ValueError(
            f"{p}: expected at least 3 columns (t, real, imag); got shape {arr.shape}."
        )
    t_src = arr[:, 0].astype(float)
    re_src = arr[:, 1].astype(float)
    im_src = arr[:, 2].astype(float)

    if np.any(np.diff(t_src) <= 0):
        order = np.argsort(t_src)
        t_src = t_src[order]
        re_src = re_src[order]
        im_src = im_src[order]

    # Crop to the support of the pulse for finer time resolution.
    t_lo, t_hi = t_src[0], t_src[-1]
    if crop_threshold is not None:
        intensity = re_src ** 2 + im_src ** 2
        above = np.nonzero(intensity >= crop_threshold * intensity.max())[0]
        if len(above) > 0:
            support_lo = t_src[above[0]]
            support_hi = t_src[above[-1]]
            support_width = support_hi - support_lo
            margin = crop_padding * support_width
            t_lo = max(t_src[0], support_lo - margin)
            t_hi = min(t_src[-1], support_hi + margin)

    dt = (t_hi - t_lo) / (N - 1)
    t_new = np.linspace(t_lo, t_hi, N)

    re = np.interp(t_new, t_src, re_src)
    im = np.interp(t_new, t_src, im_src)
    data = (re + 1j * im).astype(np.complex128)

    if center_pulse:
        peak_idx = int(np.argmax(np.abs(data) ** 2))
        data = np.roll(data, N // 2 - peak_idx)

    half = N // 2
    delays = np.arange(-half, N - half, dtype=float) * dt
    grid = Grid(N=N, dt=dt, delays=delays)
    return ElectricField(grid=grid, data=data)


def save_field_csv(
    path: PathLike,
    field: ElectricField,
    delimiter: str = ",",
    header: bool = True,
    fmt: str = "%.10e",
) -> None:
    """
    Write an `ElectricField` to a 2-column (real, imag) CSV.

    The grid itself is not written; pair this file with the same `dt`
    when reloading.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(field.data)
    out = np.column_stack([data.real, data.imag])
    np.savetxt(
        p,
        out,
        delimiter=delimiter,
        header="real,imag" if header else "",
        comments="",
        fmt=fmt,
    )
