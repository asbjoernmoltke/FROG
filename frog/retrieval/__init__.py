"""
FROG retrieval algorithms.

Subpackages
-----------
xfrog         – fast known-gate XFROG (GPA, PCGP, PIE, GradientDescent)
blind_xfrog   – double-blind XFROG  (BlindGPA, BlindPCGP, BlindEPIE, BlindGradient)
shg           – SHG-FROG, gate = field (SHGGPA, SHGPCGP, SHGPIE)
legacy_xfrog  – reference / unoptimised XFROG (same API as xfrog, ~20× slower)

Utilities
---------
multigrid_retrieve – coarse-to-fine wrapper with optional algorithm chaining
"""
from .multigrid import multigrid_retrieve  # noqa: F401
