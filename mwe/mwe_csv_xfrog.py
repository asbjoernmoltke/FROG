"""
MWE: known-gate XFROG retrieval from a CSV pulse.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frog.io import MockDataset, load_field_csv_with_time
from frog.retrieval.xfrog import GPA
from frog.visualization import compare_pulse, plot_convergence, plot_traces

csv = Path(__file__).resolve().parent / "data" / "E_test_with_t.csv"
truth = load_field_csv_with_time(csv, N=1024, crop_padding=0.5)
dataset = MockDataset.from_field(truth, gate_duration=5e-15, noise_level=0.001)

retriever = GPA(trace=dataset.trace, gate=dataset.gate)
result = retriever.retrieve(n_iter=200, seed=0)
print(f"FROG error: {result.error_curve[-1]:.2e} after {result.exec_time:.2f} s ({result.n_iterations} iter)")

# -- Plot field and convergence --
compare_pulse(result, dataset.field, title="Field (E)")
plot_convergence(result)
plot_traces(dataset.trace, result, gate=dataset.gate, range=40)

plt.show()
