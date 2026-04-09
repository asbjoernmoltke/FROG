# FROG: XFROG Pulse Characterization

## Getting Started

### Installation
- Python 3.12 is recommended (project tested with Python 3.12.5).
- Core dependencies:
  - `numpy`
  - `matplotlib`
  - `pyfftw`
- Optional development/test dependency:
  - `pytest`
- Install using `pip`:
  ```bash
  pip install numpy matplotlib pyfftw
  ```
- If you want to run tests:
  ```bash
  pip install pytest
  ```
- Install from source by cloning the repo and installing in editable mode:
  ```bash
  git clone https://github.com/asbjoernmoltke/FROG.git
  cd FROG
  pip install -e .
  ```

### Quick Start

**Minimum working example** — known-gate retrieval from synthetic data:

```python
from frog.io import MockDataset
from frog.retrieval.xfrog import GPA

# Generate a synthetic XFROG dataset (chirped Gaussian + Gaussian gate)
dataset = MockDataset.gaussian_pulse(
    N=128, dt=1.0, n_delays=64,
    field_duration=20.0, field_chirp=0.005,
    gate_duration=12.0, noise_level=0.0, seed=42,
)

# Retrieve the pulse from the trace
retriever = GPA(trace=dataset.trace, gate=dataset.gate)
result = retriever.retrieve(n_iter=200, seed=0)

# Result
print(f"Final FROG error: {result.error_curve[-1]:.2e}")
E_retrieved = result.field.data   # complex field array, length N
```

**Blind retrieval** — both pulse and gate unknown:

```python
from frog.retrieval.blind_xfrog import BlindGPA

retriever = BlindGPA(trace=dataset.trace, stop_target=1e-7, stall_window=20)
result = retriever.retrieve(n_iter=800, seed=0)

E_retrieved = result.field.data
G_retrieved = result.gate.data
print(f"Converged in {result.n_iterations} iterations, err {result.error_curve[-1]:.2e}")
```

A full interactive example with plotting is available at `examples/mwe_random_init.py`:

```bash
python examples/mwe_random_init.py --algo fast-gpa --n-iter 200
```

Available algorithm names for the MWE script: `gpa`, `pcgp`, `pie`, `gradient`, `fast-gpa`, `fast-pcgp`, `fast-pie`, `fast-gradient`.

Pass `--no-show` to suppress the plot window.

## Algorithms Overview

### Module 1: Fast XFROG ([frog/retrieval/xfrog/](frog/retrieval/xfrog/))
Fast XFROG methods assume a known gate pulse and optimize the pulse estimate to match the measured XFROG trace.
- **GPA** (Generalized Projections Algorithm): intensity projection followed by a least-squares field update weighted by the per-sample gate energy.
- **PCGP** (Principal Component Generalized Projections): same projection step as GPA but with a scalar gate-energy denominator, equivalent to a rank-1 SVD update.
- **PIE** (Ptychographic Iterative Engine): batched parallel ePIE update using the residual (delta) form with a configurable step size `alpha`.
- **GradientDescent**: Wirtinger gradient descent on the spectrogram-matching loss with backtracking line search.

Fast XFROG is the optimized implementation path; it accelerates Fourier transforms with `pyfftw`, uses `complex64` arrays where possible, bakes the `(-1)^n` sign vector into the shifted-gate matrix to eliminate per-iteration `fftshift`, and reuses preallocated aligned buffers across iterations.

### Module 2: Double-Blind XFROG ([frog/retrieval/blind_xfrog/](frog/retrieval/blind_xfrog/))
Double-blind XFROG recovers both the unknown pulse and gate from the measured XFROG trace, rather than assuming the gate pulse is known.
- **BlindGPA**: alternating GPA-style least-squares updates on E and G, each with per-sample denominators.
- **BlindPCGP**: same as BlindGPA but with scalar energy denominators (`||G||^2` for the field update, `||E||^2` for the gate update).
- **BlindEPIE**: batched parallel ePIE with separate relaxation parameters `alpha_field` and `alpha_gate` (default 0.5 each).
- **BlindGradient**: joint Wirtinger gradient descent on both E and G with backtracking line search and per-variable gradient normalization.

The shifted-gate matrix is rebuilt from the current gate estimate each iteration via a precomputed gather-index matrix. A symmetric inverse-gather index is used for the gate update. Scale ambiguity (`E*c, G/c`) is resolved by balancing `||E||` and `||G||` each iteration; time-translation ambiguity is resolved by centering `argmax|E|^2` at index `N/2` after convergence.

Early stopping is built in via `StoppingCriteria`: target error, stall detection, or iteration cap. Error computation frequency is automatically derived from the active stopping criteria.

### Module 3: SHG-FROG ([frog/retrieval/shg/](frog/retrieval/shg/))
SHG-FROG retrieval for the special case where the gate equals the field (`G = E`). Only a single unknown is recovered.
- **SHGGPA**: power method on the outer product matrix — extracts the dominant eigenvector of `Σ_m ψ'_m ⊗ E_shifted_m`.
- **SHGPCGP**: identical to SHGGPA (both reduce to the same power iteration for the single-variable case).
- **SHGPIE**: relaxed power method — interpolates between the current field and the GPA update with a tunable step size `alpha` (default 0.5).

All three use the same `BlindWorkspace` infrastructure and `StoppingCriteria` as the blind retrievers. The `(-1)^n` sign trick cancels in all products (`sign² = 1`), so the power method operates directly on the baked-in buffers.

### Module 4: Legacy XFROG ([frog/retrieval/legacy_xfrog/](frog/retrieval/legacy_xfrog/))
- A readable, reference implementation of XFROG algorithms
- Supports GPA, PCGP, PIE, and Gradient methods
- Useful for understanding algorithm flow and debugging
- Approximately 20x slower than the fast implementations

### Mathematical Background
In XFROG, the measured spectrogram is generated by gating the unknown pulse with a reference gate and taking the squared magnitude of the Fourier transform.

Notation:
- `G` : assumed known gate pulse.
- `G_t` : the true gate pulse.
- `G_r` : the reconstructed gate pulse.
- `E_t` : the true unknown field.
- `E_r` : the reconstructed field.
- `S_t` : the true XFROG spectrogram.
- `S_r` : the reconstructed XFROG spectrogram.

For a given delay `\tau_m`, the XFROG signal field is:

```math
E_{sig}(t, \tau_m) = E(t) \cdot G(t - \tau_m)
```

The spectrogram intensity is then:

```math
S(k, m) = \left| \mathcal{F}_t\{ E(t) G(t - \tau_m) \} \right|^2
```

In the code, this is implemented as:

```math
S_r[m, k] = \left| \mathcal{F}_t\{ E_r(t) \cdot G_r(t - \tau_m) \} \right|^2
```

For the standard known-gate case, the reconstruction problem is to find `E_r` such that `S_r ≈ S_t` with `G = G_t` fixed.
For blind retrieval, both `E_r` and `G_r` are updated to match `S_r` to `S_t`.

The algorithms in this project differ in how they update estimates:
- `GPA` / `BlindGPA`: intensity projection followed by least-squares field (and gate) updates weighted by per-sample energy denominators.
- `PCGP` / `BlindPCGP`: same projection step but with scalar energy denominators, equivalent to a rank-1 power-method update.
- `PIE` / `BlindEPIE`: batched parallel ePIE using the residual (delta) form with configurable step sizes.
- `GradientDescent` / `BlindGradient`: direct Wirtinger gradient descent on the spectrogram-matching loss with backtracking line search.

#### Core Update Rules
For standard (known-gate) retrieval, the algorithms update the pulse estimate `E_r` to minimize the mismatch with the measured spectrogram `S_t`.

**GPA Field Update:**

After intensity projection `\psi'_m = \mathcal{F}^{-1}\{ \sqrt{S_t} \cdot \Psi_m / |\Psi_m| \}`, the field is updated as:

```math
E_r[n] = \frac{\sum_m \overline{G(t_n - \tau_m)} \cdot \psi'_m[n]}{\sum_m |G(t_n - \tau_m)|^2}
```

This is a per-sample least-squares solution for `E` given the projected signal fields.

**PCGP Field Update:**

Same intensity projection, but with a scalar denominator:

```math
E_r[n] = \frac{\sum_m \overline{G(t_n - \tau_m)} \cdot \psi'_m[n]}{\|G\|^2}
```

**PIE Field Update (batched ePIE):**

```math
E_r \leftarrow E_r + \alpha \cdot \frac{\sum_m \overline{G_m} \cdot \delta_m}{\sum_m |G_m|^2}
```

where `\delta_m = \psi'_m - \psi_m` is the per-delay residual. With `\alpha = 1` this is algebraically identical to the GPA update.

**Gradient Descent:**
```math
E_r \leftarrow E_r - \eta \cdot \nabla_{E_r^*} \left\| S_t - | \mathcal{F}\{E_r \cdot G\} |^2 \right\|^2
```

The step size `\eta` is selected by backtracking line search.

For blind retrieval, the algorithms alternate between updating `E_r` and `G_r`. The gate update mirrors the field update under the change of variables `k = n - \tau_m`:

**Blind GPA Gate Update:**
```math
G_r[k] = \frac{\sum_m \overline{E_r(k + \tau_m)} \cdot \psi'_m[k + \tau_m]}{\sum_m |E_r(k + \tau_m)|^2}
```

**Blind ePIE Gate Update:**
```math
G_r \leftarrow G_r + \alpha_G \cdot \frac{\sum_m \overline{E(k + \tau_m)} \cdot \delta_m[k + \tau_m]}{\sum_m |E(k + \tau_m)|^2}
```

**Blind Gradient Gate Update:**
```math
G_r \leftarrow G_r - \eta \cdot \nabla_{G_r^*} \left\| S_t - | \mathcal{F}\{E_r \cdot G_r\} |^2 \right\|^2
```

#### FROG Error Metric

The FROG error is the normalized RMS residual after optimal intensity scaling:

```math
\mu = \frac{\langle S_t, S_r \rangle}{\|S_r\|^2}, \qquad
\epsilon = \frac{\sqrt{\frac{1}{N_k N_m} \sum_{k,m} \left( S_t[k,m] - \mu \cdot S_r[k,m] \right)^2}}{\max(S_t)}
```

where `S_t` and `S_r` are intensity spectrograms (not their square roots). The explicit `\mu`-scaling makes the metric well-defined even when normalizations differ slightly between iterations.

#### Retrieval Ambiguities
XFROG retrieval has fundamental ambiguities that prevent unique determination of the pulse:

**Intrinsic Phase Ambiguity**: The absolute phase of the pulse cannot be determined from intensity-only measurements.

**Double-Blind XFROG Additional Ambiguities**:
- **Time Translation**: The spectrogram is invariant under a joint time shift of E and G: replacing `E(t) -> E(t - \Delta)` and `G(t) -> G(t - \Delta)` leaves the trace unchanged because the signal `\psi_m(n) = E(n)G(n - \tau_m)` shifts uniformly in `n`, acquiring only a linear spectral phase that vanishes under `|.|^2`.
- **Scale Ambiguity**: `E * c` and `G / c` produce the same trace for any nonzero constant `c`.

By construction, the retrieved results are normalized to resolve these ambiguities:
- The pulse `E_r` is shifted so that `argmax(|E_r|^2)` lands at grid index `N/2`.
- The gate `G_r` is counter-shifted by the same amount, preserving the trace.
- `||E_r||` and `||G_r||` are balanced each iteration via `s = (||E||^2 / ||G||^2)^{1/4}`.

### Parameters

#### Standard XFROG (Known Gate)
For XFROG retrieval with a known reference gate pulse:

```python
from frog.retrieval.xfrog import GPA, PIE, GradientDescent
from frog.core.field import ElectricField
from frog.io import MockDataset
import numpy as np

# Load or generate data
dataset = MockDataset.gaussian_pulse(N=128, n_delays=64, seed=42)
retriever = GPA(trace=dataset.trace, gate=dataset.gate)
result = retriever.retrieve(n_iter=500, seed=42)
```

**Initial Guess**

By default, retrieval begins with a random initial field. You can provide a custom initial guess:

```python
# Option 1: Random initialization (default)
retriever = GPA(trace=dataset.trace, gate=dataset.gate)
result = retriever.retrieve(n_iter=500, seed=42)

# Option 2: Custom initial guess
custom_initial = ElectricField(grid=dataset.trace.grid, data=np.ones(dataset.trace.grid.N, dtype=complex))
result = retriever.retrieve(n_iter=500, initial_field=custom_initial)
```

**Error Update Frequency**

For known-gate retrievers, control how often the FROG error is calculated via the `error_every` parameter:

```python
# Calculate error every 10 iterations (default, less overhead)
retriever = GPA(trace=dataset.trace, gate=dataset.gate, error_every=10)
result = retriever.retrieve(n_iter=500)
```

**Algorithm-Specific Parameters**

Different algorithms support tuning parameters:

```python
from frog.retrieval.xfrog import PIE, GradientDescent

# PIE with step size alpha (default: 0.5)
retriever = PIE(
    trace=dataset.trace,
    gate=dataset.gate,
    alpha=0.3,
)
result = retriever.retrieve(n_iter=200)

# GradientDescent uses backtracking line search (no manual learning rate)
retriever = GradientDescent(
    trace=dataset.trace,
    gate=dataset.gate,
)
result = retriever.retrieve(n_iter=500)
```

#### Double-Blind XFROG (Unknown Pulse and Gate)

```python
from frog.retrieval.blind_xfrog import BlindGPA

# Random initialization (default)
retriever = BlindGPA(trace=dataset.trace)
result = retriever.retrieve(n_iter=800, seed=42)

# Custom initial guess for pulse and gate
custom_E = ElectricField(grid=dataset.trace.grid, data=np.random.randn(dataset.trace.grid.N) + 1j * np.random.randn(dataset.trace.grid.N))
custom_G = ElectricField(grid=dataset.trace.grid, data=np.random.randn(dataset.trace.grid.N) + 1j * np.random.randn(dataset.trace.grid.N))
result = retriever.retrieve(n_iter=800, initial_field=custom_E, initial_gate=custom_G)
```

**Stopping Criteria**

Blind retrievers support early stopping via constructor parameters. Error calculation frequency is automatically derived from the active stopping criteria (no manual `error_every` needed).

```python
from frog.retrieval.blind_xfrog import BlindGPA

retriever = BlindGPA(
    trace=dataset.trace,
    stop_target=1e-6,       # stop if FROG error < 1e-6 (default: None = off)
    stall_window=20,        # detect stall over 20 error samples (default: None = off)
    stall_threshold=0.05,   # stop if |log10(err_now) - log10(err_prev)| < 0.05 decades
)
result = retriever.retrieve(n_iter=1000)  # may stop early
print(result.n_iterations)                # actual iterations run
```

**Algorithm-Specific Parameters for Blind Retrieval**

```python
from frog.retrieval.blind_xfrog import BlindEPIE, BlindGradient

# Blind ePIE with separate step sizes for pulse and gate
retriever = BlindEPIE(
    trace=dataset.trace,
    alpha_field=0.7,        # step size for pulse updates (default: 0.5)
    alpha_gate=0.7,         # step size for gate updates (default: 0.5)
)
result = retriever.retrieve(n_iter=500)

# Blind gradient descent (backtracking line search, no manual learning rate)
retriever = BlindGradient(
    trace=dataset.trace,
    alpha0=1.0,             # initial line search step (default: 1.0)
    ls_shrink=0.5,          # shrink factor per line search step (default: 0.5)
    ls_max=6,               # max line search steps (default: 6)
)
result = retriever.retrieve(n_iter=500)
```

**Accessing Results**

```python
# Convergence history
error_curve = result.error_curve

# Retrieved pulse
retrieved_field = result.field

# For blind retrieval, also access the gate
from frog.retrieval.blind_xfrog import BlindGPA
retriever = BlindGPA(trace=dataset.trace)
result = retriever.retrieve(n_iter=500)
retrieved_gate = result.gate  # BlindRetrievalResult includes the gate
```

### Plotting ([frog/visualization/](frog/visualization/))

The `frog.visualization` module provides three plotting functions for inspecting traces, fields, and retrieval results. All accept an optional `ax` parameter to embed into existing figures.

**`plot_trace`** — 2-D pseudocolor map of a FROG spectrogram:

```python
from frog.visualization import plot_trace

plot_trace(dataset.trace, title="Measured trace")
```

**`plot_field`** — intensity and unwrapped phase on a shared time axis (dual y-axes):

```python
from frog.visualization import plot_field

plot_field(result.field, label="retrieved")
```

**`plot_retrieval_summary`** — four-panel overview (measured trace, retrieved trace, field comparison, convergence curve):

```python
from frog.visualization import plot_retrieval_summary
from frog.core.transform import forward_model

retrieved_trace = forward_model(result.field, dataset.gate)
fig = plot_retrieval_summary(
    result,
    measured_trace=dataset.trace,
    retrieved_trace=retrieved_trace,
    ground_truth=dataset.field,       # optional
)
fig.savefig("retrieval_summary.png", dpi=130)
```

All three functions return the matplotlib `Axes` or `Figure` object for further customization. Pass `ax=existing_axes` to embed a trace or field plot into a larger figure layout.

## Speed

Benchmarked on a desktop CPU (single process, pyfftw with FFTW_MEASURE). Known-gate fast retrievers at N=M=1024, 500 iterations, 1% noise:

| Algorithm | ms/iter | Time to err < 1e-2 |
|---|---|---|
| fast-gpa | ~12 | < 2 s |
| fast-pcgp | ~12 | < 2 s |
| fast-pie | ~13 | < 2 s |
| fast-gradient | ~20 | varies |

Blind retrievers at N=256, 800 iterations (noiseless, two-Gaussian test):

| Algorithm | ms/iter | Typical convergence |
|---|---|---|
| blind-gpa | ~2.5 | ~60 iter to ~1e-8 |
| blind-pcgp | ~2.3 | ~60 iter to ~1e-8 |
| blind-epie | ~2.7 | ~80 iter to ~1e-8 |
| blind-gradient | ~3.0 | stalls ~1e-3 |

Key optimizations:
- `pyfftw` with `FFTW_MEASURE` planned FFTs on preallocated aligned buffers
- `complex64` dtype for 2x FFT throughput and halved memory bandwidth
- `(-1)^n` sign vector baked into the shifted-gate matrix (eliminates per-iteration `fftshift`)
- `(M, N)` memory layout with FFT along the contiguous axis
- Allocation-free FROG error with direct residual computation (avoids float32 catastrophic cancellation)

## Convergence

- GPA, PCGP, and PIE (alpha=1) converge identically on full uniform delay grids (mathematically equivalent updates).
- PIE with alpha < 1 and PCGP diverge from GPA on sparse or non-uniform delay grids.
- Gradient descent is slower per iteration (line search overhead) and may stall on non-convex landscapes.
- Blind retrieval converges to the true solution up to inherent ambiguities (global phase, time translation, scale).

## Important Notes

- The `frog/retrieval/legacy_xfrog/` module is a reference implementation kept for algorithm clarity and validation. Use `frog/retrieval/xfrog/`, `frog/retrieval/blind_xfrog/`, or `frog/retrieval/shg/` for production workloads.
- FFT performance depends strongly on N being a product of small primes. Powers of two (N = 2^k) give the best performance; composite sizes like N=1023 = 3*11*31 can be 3x slower.
- For blind retrieval, the gradient descent variant typically stalls around 1e-3 to 1e-4; prefer GPA, PCGP, or ePIE for blind problems.
- Report issues at https://github.com/<username>/FROG/issues
