"""
MWE: multigrid + algorithm chaining for SHG-FROG (gate = field).

Compares direct SHGGPA vs multigrid SHGGPA vs multigrid GPA + PIE refinement.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frog.core.grid import Grid
from frog.core.field import ElectricField
from frog.core.transform import forward_model
from frog.core.trace import FROGTrace
from frog.retrieval.shg import SHGGPA, SHGPIE
from frog.retrieval.multigrid import multigrid_retrieve
from frog.visualization import compare_pulse, plot_convergence

# --- Build synthetic SHG dataset ---
N = 512
dt = 1.0
grid = Grid(N=N, dt=dt, delays=np.arange(-N // 2, N // 2 + 1, dtype=float) * dt)
t = grid.t
E_true = ElectricField(
    grid=grid,
    data=np.exp(-(t ** 2) / (2 * 15.0 ** 2)) * np.exp(1j * 0.02 * t ** 2),
)
trace = forward_model(E_true, E_true).normalized()

# --- Direct SHGGPA ---
print("Direct SHGGPA:")
r_direct = SHGGPA(trace=trace).retrieve(n_iter=1000, seed=0)
print(f"  error={r_direct.error_curve[-1]:.2e}  {r_direct.exec_time:.2f}s\n")

# --- Multigrid SHGGPA ---
print("Multigrid SHGGPA:")
r_multi = multigrid_retrieve(
    trace, SHGGPA, n_iter=200, seed=0,
)
print(f"  error={r_multi.error_curve[-1]:.2e}  {r_multi.exec_time:.2f}s\n")

# --- Multigrid SHGGPA + PIE refinement ---
print("Multigrid SHGGPA + SHGPIE:")
r_chain = multigrid_retrieve(
    trace, SHGGPA, n_iter=200, seed=0,
    refinement_cls=SHGPIE, refinement_iter=200,
    refinement_kwargs=dict(alpha=0.7),
)
print(f"  error={r_chain.error_curve[-1]:.2e}  {r_chain.exec_time:.2f}s\n")

# --- Plots ---
compare_pulse(r_chain, E_true, title="Field (E) — multigrid+PIE")
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(r_direct.error_curve, label=f"direct GPA ({r_direct.exec_time:.1f}s)", alpha=0.7)
ax.semilogy(r_multi.error_curve, label=f"multigrid GPA ({r_multi.exec_time:.1f}s)", alpha=0.7)
ax.semilogy(r_chain.error_curve, label=f"multigrid+PIE ({r_chain.exec_time:.1f}s)", alpha=0.7)
ax.set_xlabel("iteration (cumulative)"); ax.set_ylabel("FROG error")
ax.set_title("Convergence comparison"); ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
