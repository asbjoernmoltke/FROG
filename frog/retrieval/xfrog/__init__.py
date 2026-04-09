"""
Fast / optimized variants of the XFROG retrievers.

Public API mirrors `frog.retrieval` so callers can swap modules.  Only
the algorithms that have been ported so far are re-exported; the rest
still live as verbatim copies in this folder and will be replaced
incrementally.
"""
from .base import Retriever, RetrievalResult  # noqa: F401
from .gpa import GPA  # noqa: F401
from .pcgp import PCGP  # noqa: F401
from .gradient import GradientDescent  # noqa: F401
from .pie import PIE  # noqa: F401
