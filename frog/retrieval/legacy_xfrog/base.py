"""
Abstract base class for XFROG retrieval algorithms.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from ...core.field import ElectricField
from ...core.trace import FROGTrace


@dataclass
class RetrievalResult:
    """Container returned by every Retriever."""

    field: ElectricField    # retrieved pulse
    error_curve: List[float]  # FROG error after each iteration
    n_iterations: int
    exec_time: float = 0.0   # wall-clock seconds for retrieve()


@dataclass
class Retriever(ABC):
    """
    Abstract base for XFROG retrieval algorithms.

    Every concrete retriever receives the measured trace and the known gate
    at construction time, then implements `_retrieve_impl()`.

    The shared `frog_error` static method provides the standard normalized
    RMS intensity error metric used across all algorithms.
    """

    trace: FROGTrace
    gate: ElectricField

    def __post_init__(self) -> None:
        if self.trace.grid is not self.gate.grid:
            raise ValueError(
                "trace and gate must share the same Grid instance."
            )

    def retrieve(self, n_iter: int = 200, **kwargs) -> RetrievalResult:
        """Run retrieval, time it, and return a RetrievalResult."""
        t0 = time.perf_counter()
        result = self._retrieve_impl(n_iter, **kwargs)
        result.exec_time = time.perf_counter() - t0
        return result

    @abstractmethod
    def _retrieve_impl(self, n_iter: int = 200, **kwargs) -> RetrievalResult:
        """Run retrieval and return a RetrievalResult."""
        ...

    @staticmethod
    def frog_error(I_meas: np.ndarray, I_calc: np.ndarray) -> float:
        """
        Normalized FROG error.

        Finds the scalar mu that minimizes ||I_meas - mu * I_calc||^2, then
        returns the normalized RMS residual:

            G = sqrt( mean( (I_meas - mu * I_calc)^2 ) ) / max(I_meas)

        Both inputs are assumed to be already normalized (peak = 1), in which
        case mu ≈ 1.  The explicit minimization keeps the metric well-defined
        even when normalizations differ slightly between iterations.
        """
        denom = float(np.sum(I_calc ** 2))
        mu = float(np.sum(I_meas * I_calc)) / denom if denom > 0 else 1.0
        residual = I_meas - mu * I_calc
        peak = float(I_meas.max())
        return float(np.sqrt(np.mean(residual ** 2))) / (peak + 1e-30)
