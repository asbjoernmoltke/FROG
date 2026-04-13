"""
MWE: blind XFROG retrieval (unknown gate) from a CSV pulse.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frog.io import MockDataset, load_field_csv_with_time
from frog.retrieval.blind_xfrog import BlindGPA
from frog.visualization import compare_pulse, plot_convergence, plot_traces

csv = Path(__file__).resolve().parent / "data" / "E_test_with_t.csv"
truth = load_field_csv_with_time(csv, N=1024, crop_padding=0.5)
dataset = MockDataset.from_field(truth, gate_duration=5e-15, noise_level=0.001)

retriever = BlindGPA(trace=dataset.trace, stop_target=1e-7)
result = retriever.retrieve(n_iter=200, seed=0)
print(f"FROG error: {result.error_curve[-1]:.2e} after {result.exec_time:.2f} s ({result.n_iterations} iter)")

# -- Plot field, gate, and convergence --
compare_pulse(result, dataset.field, which="field", title="Field (E)")
compare_pulse(result, dataset.gate, which="gate", title="Gate (G)")
plot_convergence(result)
plot_traces(dataset.trace, result, range=15)
plt.show()
