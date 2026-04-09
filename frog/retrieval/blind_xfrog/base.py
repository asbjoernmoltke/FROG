"""
Base classes for blind XFROG retrievers (unknown field AND unknown gate).
"""
from __future__ import annotations

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


@dataclass
class BlindRetriever(ABC):
    """Blind-XFROG retriever: trace only, no known gate."""

    trace: FROGTrace

    @abstractmethod
    def retrieve(self, n_iter: int = 500, **kwargs) -> BlindRetrievalResult: ...
