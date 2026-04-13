"""
Base classes for blind XFROG retrievers (unknown field AND unknown gate).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ...core.field import ElectricField
from ...core.trace import FROGTrace


@dataclass
class BlindRetrievalResult:
    field: ElectricField          # retrieved unknown pulse
    gate: ElectricField           # retrieved gate
    error_curve: List[float]
    n_iterations: int
    exec_time: float = 0.0       # wall-clock seconds for retrieve()


@dataclass
class BlindRetriever(ABC):
    """Blind-XFROG retriever: trace only, no known gate."""

    trace: FROGTrace

    def retrieve(self, n_iter: int = 500, **kwargs) -> BlindRetrievalResult:
        t0 = time.perf_counter()
        result = self._retrieve_impl(n_iter, **kwargs)
        result.exec_time = time.perf_counter() - t0
        return result

    @abstractmethod
    def _retrieve_impl(self, n_iter: int = 500, **kwargs) -> BlindRetrievalResult: ...
