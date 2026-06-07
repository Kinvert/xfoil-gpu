# SPEC.md

## Project: CUDA Rewrite of XFOIL in Warp

## 1) Vision
The primary goal is to build a practical CUDA-accelerated approximation of XFOIL’s solver stack, with a clean API surface that can later be used as an oracle in environments such as PufferLib.
For this phase the priority is controlled approximation quality, repeatable runs, and well-characterized speed/quality tradeoffs.
The longer-term integration path is RL (PufferLib), but that is not required in the initial phase.

Warm-start/continuation ideas for downstream sequential workflows are documented in: [warp_xfoil_warm_start_research.md](/home/claude/xfoil/warp_xfoil_warm_start_research.md).

## 2) Problem statement
Given one airfoil geometry and a set of operating conditions, the system should return aerodynamic outputs (`CL`, `CD`, `CM`, and optional surface fields) with stable reproducibility for fixed inputs and a clearly bounded approximation error.

Core requirements:
- Match XFOIL reference behavior with practical error budgets, with explicit tolerances per regime.
- Default accuracy target is roughly 5% for core scalar metrics in normal operating conditions; higher-error regions must be tagged.
- Keep CPU path as the default correctness baseline; add CUDA path where it is numerically stable.
- Emit structured diagnostics for convergence, failures, and performance counters.
- Keep outputs repeatable for repeated runs with fixed seeds/configuration, while accepting that XFOIL itself may exhibit solver-order dependence.

## 3) Scope and phases

### Phase A (now): CUDA rewrite and validator
- Implement a controlled approximation path for XFOIL panel + viscous/inviscid coupled flow in Warp/CUDA where practical.
- Keep a CPU reference path for calibration and fallback.
- Build a reproducible regression harness to compare against trusted XFOIL outputs.
- Confirm baseline dependencies and reproducibility tooling before optimization work.

### Dependency bootstrap work (must be finished before benchmark claims)
- Establish an `uv`-managed Python environment and lockfile strategy.
- Confirm XFOIL installation/availability options, including:
  - Source build availability,
  - binary/runtime availability,
  - command-line invocation stability,
  - and data-input/output formats.
- Audit CUDA/driver/runtime stack assumptions and versioning used by Warp in this environment.

### Phase B (later): API packaging + ecosystem integrations
- Provide stable environment-friendly bindings and wrappers.
- Optional RL integration as an application of the solver API, with a potential PufferLib adapter later.
- Optional continuation/warm-start runtime policy for sequences.

## 4) Architecture

### 4.1 Components
1. `geometry_ingest`
   - Coordinate parsing, normalization, panel redistribution, and geometric sanity validation.

2. `xfoil_cpu_reference`
   - Baseline path using existing compiled/file-backed XFOIL behavior for truth data and regression.

3. `xfoil_cuda_core`
   - Warp/CUDA kernels for batched panel setup, geometry transforms, and solver loops.
   - Deterministic memory layout and explicit kernel launch contracts.

4. `solver_control`
   - Iteration controls equivalent to `ITER` and fallback controls equivalent to `INIT`.
   - Convergence checks and fail-safe exit codes.

5. `validation_bench`
   - Dataset-driven golden tests and benchmarking pipeline.
6. `env_probe`
   - Dependency probing and pinned-runtime manifest generation (warp/cuda/xfoil/uv/python).

### 4.2 Data flow
- input airfoil + conditions -> geometry ingest -> normalized panel data -> GPU/CPU solver dispatch -> outputs + diagnostics -> validation + perf logging.

## 5) Target deliverables
- A rewritten solver module with both CPU baseline and CUDA execution modes.
- A deterministic-enough API mirroring key XFOIL query points (angle sweep, lift sweep, force/moment outputs).
- Regression results package with error/coverage reports and clearly labeled uncertainty regions.
- Benchmark scripts showing speed and accuracy tradeoffs.
- Dependency research notes and reproducibility manifest (versions, environment, runtime knobs, timeouts).

## 6) Accuracy targets
- Nominal attached-flow regime: target practical tolerance around 5% for key scalar outputs.
- Near-stall/transonic and edge geometries: explicit lower-confidence mode with expanded tolerances or conservative rejection.
- Any deviation must be documented by regime and reproducible via seeds/config.

## 7) Performance targets
- Establish baseline wall-clock and throughput on a standard panel-count/condition set.
- Measure performance of each stage (I/O, paneling, solver iteration, post-processing) before asserting overall speed targets.
- Do not set a fixed throughput target yet; treat throughput goals as measurable outcomes after baseline/prototype comparison.
- Do not trade controlled accuracy for opaque optimizations without a labeled tolerance and fail mode.

## 8) Failure and safety behavior
- Standardized failure codes: non-convergence, invalid geometry, conditioning failure, numerical instabilities, timeout.
- Hard fail is explicit and logged; no silent convergence from undefined states.
- Optional fallback to CPU reference mode for debugging and oracle checks.
- Telemetry must include reason code, residual history (when available), iteration count, and timing.
- In this phase, missing external XFOIL binary is an explicit dependency failure mode:
  - with `enable_fallback=True`, evaluator returns bounded deterministic fallback outputs and sets `status` + `fallback_reason`.
  - with `enable_fallback=False`, evaluator emits `XFOIL` dependency error (hard failure).

## 9) Milestones
1. **M0 – Baseline dependency and API plan**
   - Define reproducibility contract (seeding, config schema, deterministic output manifest).
   - Document dependency matrix and `uv` bootstrap path.
   - Add benchmark dataset/version manifest.

2. **M1 – Baseline interfaces**
   - Finalize API contract and regression datasets.
   - Add baseline CLI/benchmark harness.

3. **M2 – CPU parity harness**
   - Implement CPU execution path using trusted reference behavior.
   - Add deterministic result schema and validation scripts.

4. **M3 – CUDA core rewrite**
   - Implement parallel kernels for the highest-cost solver stages.
   - Add batch dispatch, stream-aware execution, and timing instrumentation.

5. **M4 – Convergence controls**
   - Implement `ITER`-style iteration caps and `INIT`-style reinitialization controls in solver control flow.
   - Add robust non-convergence recovery behavior.

6. **M5 – Numerical validation pass**
   - Benchmark error by regime and geometry class.
   - Produce acceptance charts and failure taxonomy.

7. **M6 – Throughput pass**
   - Publish measured speedup and memory scaling curves.
   - Add launch configuration tuning guide.

8. **M7 – API stabilization + future adapter hooks**
   - Prepare clean package API for downstream projects (including RL adapters).

## 10) Acceptance criteria
- Repeatable output for fixed seeds/config on both CPU and CUDA modes (within expected floating-point noise tolerance).
- Regression tests for key operating regimes with documented tolerance bands.
- Clear performance report vs baseline reference at fixed hardware profile.
- Explicitly documented unsupported domains for high-error regions.
- Reproducible benchmark command available in docs.

## 11) Repository layout
- `src/` for solver modules and kernels.
- `configs/` for benchmark and solver settings.
- `data/` for airfoil and test-case datasets.
- `scripts/` for benchmark and validation scripts.
- `tests/` for deterministic regression suites.
- `docs/` for dependency and reproducibility notes.
- `logs/` for perf and failure outputs.

## 12) Non-goals (initial phase)
- Full RL stack in this phase.
- Full stall/transonic fidelity before core rewrite performance/robustness is profiled.
- Hard guarantees of bitwise identity versus legacy XFOIL for all branches.
