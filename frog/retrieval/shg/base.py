"""
Base classes for SHG-FROG retrievers (gate = field).
"""
from __future__ import annotations

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


@dataclass
class SHGRetriever(ABC):
    """SHG-FROG retriever: trace only, gate = field."""

    trace: FROGTrace

    @abstractmethod
    def retrieve(self, n_iter: int = 500, **kwargs) -> SHGRetrievalResult: ...
