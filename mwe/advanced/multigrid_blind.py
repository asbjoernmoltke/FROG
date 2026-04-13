"""
MWE: multigrid + algorithm chaining for blind XFROG (unknown gate).

Compares direct BlindGPA vs multigrid BlindGPA vs multigrid + gradient.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frog.io import MockDataset, load_field_csv_with_time
from frog.retrieval.blind_xfrog import BlindGPA, BlindGradient
from frog.retrieval.multigrid import multigrid_retrieve
from frog.visualization import compare_pulse, plot_convergence, plot_traces

csv = Path(__file__).resolve().parent.parent / "data" / "E_test_with_t.csv"
truth = load_field_csv_with_time(csv, N=1024, crop_padding=0.5)
dataset = MockDataset.from_field(truth, gate_duration=5e-15, noise_level=0.001)

# --- Direct BlindGPA ---
print("Direct BlindGPA:")
r_direct = BlindGPA(trace=dataset.trace).retrieve(n_iter=1500, seed=0)
print(f"  error={r_direct.error_curve[-1]:.2e}  {r_direct.exec_time:.2f}s\n")

# --- Multigrid BlindGPA ---
print("Multigrid BlindGPA:")
r_multi = multigrid_retrieve(
    dataset.trace, BlindGPA, n_iter=300, seed=0,
)
print(f"  error={r_multi.error_curve[-1]:.2e}  {r_multi.exec_time:.2f}s\n")

# --- Multigrid BlindGPA + Gradient refinement ---
print("Multigrid BlindGPA + BlindGradient:")
r_chain = multigrid_retrieve(
    dataset.trace, BlindGPA, n_iter=300, seed=0,
    refinement_cls=BlindGradient, refinement_iter=100,
)
print(f"  error={r_chain.error_curve[-1]:.2e}  {r_chain.exec_time:.2f}s\n")

# --- Plots ---
compare_pulse(r_chain, dataset.field, which="field", title="Field (E) — multigrid+gradient")
compare_pulse(r_chain, dataset.gate, which="gate", title="Gate (G) — multigrid+gradient")
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(r_direct.error_curve, label=f"direct ({r_direct.exec_time:.1f}s)", alpha=0.7)
ax.semilogy(r_multi.error_curve, label=f"multigrid ({r_multi.exec_time:.1f}s)", alpha=0.7)
ax.semilogy(r_chain.error_curve, label=f"multigrid+gradient ({r_chain.exec_time:.1f}s)", alpha=0.7)
ax.set_xlabel("iteration (cumulative)"); ax.set_ylabel("FROG error")
ax.set_title("Convergence comparison"); ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plot_traces(dataset.trace, r_chain)
plt.show()
