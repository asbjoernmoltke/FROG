# Graph Report - .  (2026-04-09)

## Corpus Check
- Corpus is ~44,673 words - fits in a single context window. You may not need a graph.

## Summary
- 306 nodes · 634 edges · 25 communities detected
- Extraction: 53% EXTRACTED · 47% INFERRED · 0% AMBIGUOUS · INFERRED: 300 edges (avg confidence: 0.5)
- Token cost: 8,500 input · 3,200 output

## God Nodes (most connected - your core abstractions)
1. `ElectricField` - 94 edges
2. `FROGTrace` - 55 edges
3. `Retriever` - 40 edges
4. `Grid` - 35 edges
5. `RetrievalResult` - 27 edges
6. `FastWorkspace` - 22 edges
7. `BlindWorkspace` - 19 edges
8. `StoppingCriteria` - 16 edges
9. `TestGrid` - 14 edges
10. `BlindRetriever` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Run retrieval and return a RetrievalResult.` --uses--> `ElectricField`  [INFERRED]
  frog\retrieval\legacy_xfrog\base.py → frog\core\field.py
- `Normalized FROG error.          Finds the scalar mu that minimizes ||I_meas - mu` --uses--> `ElectricField`  [INFERRED]
  frog\retrieval\legacy_xfrog\base.py → frog\core\field.py
- `ElectricField: complex envelope of a laser pulse on a Grid.` --uses--> `Grid`  [INFERRED]
  frog\core\field.py → frog\core\grid.py
- `Complex envelope of a laser pulse sampled on a Grid.      `data[n]` is the compl` --uses--> `Grid`  [INFERRED]
  frog\core\field.py → frog\core\grid.py
- `FROGTrace: 2-D intensity array (frequency × delay).` --uses--> `Grid`  [INFERRED]
  frog\core\trace.py → frog\core\grid.py

## Hyperedges (group relationships)
- **Fast XFROG Algorithms** — readme_gpa_algorithm, readme_pcgp_algorithm, readme_pie_algorithm, readme_gradient_descent_algorithm [EXTRACTED 1.00]
- **Blind XFROG Algorithms** — readme_blind_gpa, readme_blind_pcgp, readme_blind_epie, readme_blind_gradient [EXTRACTED 1.00]
- **SHG-FROG Algorithms** — readme_shg_gpa, readme_shg_pcgp, readme_shg_pie [EXTRACTED 1.00]

## Communities

### Community 0 - "XFROG Retrieval Base"
Cohesion: 0.06
Nodes (42): Re-export of the canonical Retriever base so fast variants share the ABC., Container returned by every Retriever., Abstract base for XFROG retrieval algorithms.      Every concrete retriever rece, RetrievalResult, Retriever, FastWorkspace, Precomputed constants and scratch buffers shared by all fast retrievers.      Bu, ElectricField (+34 more)

### Community 1 - "Blind Workspace & Optimization"
Cohesion: 0.05
Nodes (40): SHG-FROG retriever: trace only, gate = field., Blind-XFROG retriever: trace only, no known gate., Run retrieval and return a RetrievalResult., Normalized FROG error.          Finds the scalar mu that minimizes ||I_meas - mu, Shared infrastructure for the fast XFROG retrievers.  This module isolates the p, Fill ``out[m, n] = sign_n * G[(n - tau_m) % N]`` and optionally         its conj, Roll both E and G so that argmax|E|^2 lands at index N//2.         Joint time-tr, Shared stopping-condition checker.      Parameters     ----------     max_iter : (+32 more)

### Community 2 - "Blind Retrieval Algorithms"
Cohesion: 0.09
Nodes (21): ABC, BlindRetrievalResult, BlindRetriever, SHGRetrievalResult, SHGRetriever, BlindRetriever, BlindWorkspace, StoppingCriteria (+13 more)

### Community 3 - "Electric Field & Core"
Cohesion: 0.09
Nodes (12): ElectricField: complex envelope of a laser pulse on a Grid., Grid: shared discretization for time, frequency, and delay axes.  The central in, FROGTrace: 2-D intensity array (frequency × delay)., compute_signal_field(), _fft_centered(), forward_model(), _ifft_centered(), Forward XFROG model: (field, gate) -> FROGTrace.      I[k, m] = | FFT_t[ E(t) * (+4 more)

### Community 4 - "Documentation & Benchmarks"
Cohesion: 0.1
Nodes (20): BlindEPIE, BlindGPA, BlindGradient, BlindPCGP, BlindWorkspace, Double-Blind XFROG Module, FROG: XFROG Pulse Characterization Project, GPA Algorithm (+12 more)

### Community 5 - "Grid Unit Tests"
Cohesion: 0.18
Nodes (1): TestGrid

### Community 6 - "Scaling Benchmark"
Cohesion: 0.29
Nodes (9): first_index_below(), main(), make_dataset(), _make_plots(), Benchmark the fast XFROG retrievers across grid sizes.  Sweeps N = M = 2^k for k, Build a noisy Gaussian-pulse dataset with N time points and N+1 delays.      The, Return the first iteration index whose error is < threshold, or None., Construct + (optionally warm) + time one retriever on grid size N. (+1 more)

### Community 7 - "CSV Field I/O"
Cohesion: 0.24
Nodes (9): load_field_csv(), load_field_csv_with_time(), _looks_like_header(), Read / write a complex `ElectricField` from a plain CSV file.  File format -----, Load a complex E-field from a 3-column (t, real, imag) CSV and     resample it o, Write an `ElectricField` to a 2-column (real, imag) CSV.      The grid itself is, Return True if the first line is non-numeric (e.g. 'real,imag')., Load a complex E-field from a 2-column (real, imag) CSV.      Parameters     --- (+1 more)

### Community 8 - "Visualization Plots"
Cohesion: 0.32
Nodes (7): plot_field(), plot_retrieval_summary(), plot_trace(), Visualization utilities for FROG traces and retrieved fields., 2-D pseudocolor plot of a FROGTrace.      The intensity array is already in cent, Plot the intensity and unwrapped phase of a complex electric field on     a shar, Four-panel summary figure:        [measured trace]  [retrieved trace]       [fie

### Community 9 - "FROGTrace Unit Tests"
Cohesion: 0.6
Nodes (1): TestFROGTrace

### Community 10 - "Complex Pulse Benchmark"
Cohesion: 0.6
Nodes (4): main(), _make_plots(), parse_args(), Benchmark the fast retrievers on a complex pulse loaded from CSV.  Loads a 2-col

### Community 11 - "Random Init MWE"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Minimum working example: XFROG retrieval from a random-noise initial guess.  Bui

### Community 12 - "Workspace Initialization"
Cohesion: 0.5
Nodes (2): Length-N vector of (-1)^n.  Real, dtype-matched to the workspace., _sign_vector()

### Community 13 - "FROG Error Computation"
Cohesion: 1.0
Nodes (2): fast_frog_error(), Allocation-free FROG error, equivalent to     `Retriever.frog_error(I_meas, I_ca

### Community 14 - "Grid Rationale"
Cohesion: 1.0
Nodes (1): Number of delay points.

### Community 15 - "Grid Rationale"
Cohesion: 1.0
Nodes (1): Time axis [dt units], centered at zero. Shape (N,).

### Community 16 - "Grid Rationale"
Cohesion: 1.0
Nodes (1): Ordinary frequency axis [1/dt units], centered and monotonically         increas

### Community 17 - "Grid Rationale"
Cohesion: 1.0
Nodes (1): Build a Grid from a spectrometer frequency bin spacing dnu.          The FFT rel

### Community 18 - "Grid Rationale"
Cohesion: 1.0
Nodes (1): Build a Grid by rounding arbitrary delay values to the nearest         integer m

### Community 19 - "FROG Error Metric"
Cohesion: 1.0
Nodes (1): FROG Error Metric

### Community 20 - "Retrieval Ambiguities"
Cohesion: 1.0
Nodes (1): Retrieval Ambiguities

### Community 21 - "Visualization Module"
Cohesion: 1.0
Nodes (1): Visualization Module

### Community 22 - "Complex Pulse Chart"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Method Overview"
Cohesion: 1.0
Nodes (1): Method Overview Table

### Community 24 - "Hybrid Method"
Cohesion: 1.0
Nodes (1): Hybrid Method Concept

## Knowledge Gaps
- **36 isolated node(s):** `Benchmark the fast retrievers on a complex pulse loaded from CSV.  Loads a 2-col`, `Benchmark the fast XFROG retrievers across grid sizes.  Sweeps N = M = 2^k for k`, `Build a noisy Gaussian-pulse dataset with N time points and N+1 delays.      The`, `Return the first iteration index whose error is < threshold, or None.`, `Construct + (optionally warm) + time one retriever on grid size N.` (+31 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `FROG Error Computation`** (2 nodes): `fast_frog_error()`, `Allocation-free FROG error, equivalent to     `Retriever.frog_error(I_meas, I_ca`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Grid Rationale`** (1 nodes): `Number of delay points.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Grid Rationale`** (1 nodes): `Time axis [dt units], centered at zero. Shape (N,).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Grid Rationale`** (1 nodes): `Ordinary frequency axis [1/dt units], centered and monotonically         increas`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Grid Rationale`** (1 nodes): `Build a Grid from a spectrometer frequency bin spacing dnu.          The FFT rel`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Grid Rationale`** (1 nodes): `Build a Grid by rounding arbitrary delay values to the nearest         integer m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FROG Error Metric`** (1 nodes): `FROG Error Metric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Retrieval Ambiguities`** (1 nodes): `Retrieval Ambiguities`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Visualization Module`** (1 nodes): `Visualization Module`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Complex Pulse Chart`** (1 nodes): `benchmark_complex_pulse.png`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Method Overview`** (1 nodes): `Method Overview Table`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Hybrid Method`** (1 nodes): `Hybrid Method Concept`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ElectricField` connect `XFROG Retrieval Base` to `Blind Workspace & Optimization`, `Blind Retrieval Algorithms`, `Electric Field & Core`, `Grid Unit Tests`, `CSV Field I/O`, `Visualization Plots`, `FROGTrace Unit Tests`, `Workspace Initialization`, `FROG Error Computation`?**
  _High betweenness centrality (0.393) - this node is a cross-community bridge._
- **Why does `FROGTrace` connect `Blind Workspace & Optimization` to `XFROG Retrieval Base`, `Blind Retrieval Algorithms`, `Electric Field & Core`, `Grid Unit Tests`, `Visualization Plots`, `FROGTrace Unit Tests`, `Workspace Initialization`, `FROG Error Computation`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `Grid` connect `Blind Workspace & Optimization` to `XFROG Retrieval Base`, `Electric Field & Core`, `Grid Unit Tests`, `CSV Field I/O`, `FROGTrace Unit Tests`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Are the 91 inferred relationships involving `ElectricField` (e.g. with `Blind-XFROG MWE: retrieve both field and gate from the trace alone.` and `Remove global phase: multiply E by exp(-i*angle(<E,E_ref>)).`) actually correct?**
  _`ElectricField` has 91 INFERRED edges - model-reasoned connections that need verification._
- **Are the 51 inferred relationships involving `FROGTrace` (e.g. with `Blind-XFROG MWE: retrieve both field and gate from the trace alone.` and `Remove global phase: multiply E by exp(-i*angle(<E,E_ref>)).`) actually correct?**
  _`FROGTrace` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `Retriever` (e.g. with `ElectricField` and `FROGTrace`) actually correct?**
  _`Retriever` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `Grid` (e.g. with `Blind-XFROG MWE: retrieve both field and gate from the trace alone.` and `Remove global phase: multiply E by exp(-i*angle(<E,E_ref>)).`) actually correct?**
  _`Grid` has 32 INFERRED edges - model-reasoned connections that need verification._