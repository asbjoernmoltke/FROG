"""
Base classes for SHG-FROG retrievers (gate = field).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ...core.field import ElectricField
from ...core.trace import FROGTrace


@dataclass
class SHGRetrievalResult:
    field: ElectricField          # retrieved pulse (also the gate)
    error_curve: List[float]
    n_iterations: int
    exec_time: float = 0.0       # wall-clock seconds for retrieve()


@dataclass
class SHGRetriever(ABC):
    """SHG-FROG retriever: trace only, gate = field."""

    trace: FROGTrace

    def retrieve(self, n_iter: int = 500, **kwargs) -> SHGRetrievalResult:
        t0 = time.perf_counter()
        result = self._retrieve_impl(n_iter, **kwargs)
        result.exec_time = time.perf_counter() - t0
        return result

    @abstractmethod
    def _retrieve_impl(self, n_iter: int = 500, **kwargs) -> SHGRetrievalResult: ...
