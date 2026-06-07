# AGENTS.md

## Purpose
This directory is for a research-to-MVP project: a PufferLib environment that uses XFOIL as an aerodynamics oracle and Warp for GPU-side data processing.

## Environment and tooling
- Work in: `/home/claude/xfoil`
- Use `uv` for Python environment management.
- Use one virtual environment per experiment branch under this directory (e.g. `.venv`).
- Target Python version: 3.11+.
- No assumptions about CUDA availability in every workflow stage; design code to run on CPU first, then enable CUDA where stable.
- Do not run commands that have any chance to alter, degrade, or break CUDA stack, drivers, toolchains, or runtime environment.
- Prohibit system-level changes to CUDA (e.g., driver updates, toolkit installs/removals, environment variable rewrites outside this project).
- Keep CUDA-related work strictly project-scoped and never execute “best effort” recovery/debug commands that touch `/usr`, `/usr/local`, `conda`, `apt`, `pacman`, `dnf`, `brew`, kernel modules, or GPU firmware unless explicitly approved.
- Assume any command is forbidden if it writes outside `/home/claude/xfoil` and can modify CUDA/driver/toolchain/runtime state.
- **Hard rule:** do not run any command with non-trivial risk to CUDA stability, including build/install tooling touching GPU stacks (`nvcc`, `nvidia-smi` config paths, module loads/switches, global CUDA env vars, `/usr/local/cuda`, driver package operations). If a command could touch these, skip unless explicitly approved.

## Mandatory project constraints
- The goal is practical RL environment performance, not bitwise identity with legacy XFOIL.
- Target behavior is controlled approximation with documented tolerances (default planning target: ≤5% error on core scalar metrics).
- Prioritize robust behavior over speed when XFOIL fails to converge.
- Do not overwrite or delete existing markdown files moved into this directory without explicit direction.
- Treat determinism as a first-order requirement: all environment transitions, reward calculations, and geometry transformations must be reproducible from explicit seeds and configuration.
- CUDA is performance-only; define deterministic CPU as the canonical behavioral reference and keep CPU correctness tests mandatory.
- Use strict input validation on airfoil geometry (point count, leading/trailing edge consistency, ordering, bounds).
## Coding rules
- Keep geometry/eval logic deterministic for benchmarking by fixing seeds and logging all control parameters.
- Enforce strict input validation on airfoil geometry (point count, leading/trailing edge consistency, ordering, bounds).
- Maintain a clear separation between:
  - Oracle path (authoritative XFOIL execution)
  - Approximate path (Warp/Python surrogate or pre/post kernels)
  - Failure-handling path
- Every environment transition should include a safe fallback state when solver/IO fails.
- Reward logic must be bounded and saturating; avoid huge spikes from invalid states.
- Prefer explicit config files (YAML/TOML) over hardcoded constants.

## API and architecture constraints
- PufferLib integration should expose the RL step through `c_step`-style stepping where possible.
- Keep evaluator interface explicit and side-effect bounded:
  - input: action/state
  - output: next state, reward, done, info
- If file I/O to XFOIL is used, implement caching and timeout guards.
- Prefer compiled XFOIL bindings for high-throughput benchmarking when available.

## Security and operational boundaries
- Avoid writing outside `/home/claude/xfoil` unless unavoidable.
- Do not run destructive git operations.
- If adding command-line tools that spawn subprocesses, sanitize all filenames and cap execution time.
- Respect the CUDA preservation constraint above for all workflows; never run anything with a non-trivial chance of breaking CUDA stability.
- No CUDA risk:
  - Do not run commands that can alter CUDA installation state or system-level GPU runtime/driver artifacts unless explicitly requested by the user and verified safe (examples: installer scripts, apt/yum/pacman package changes touching `cuda*`, driver toolchain updates, kernel/module rebuilds, manual symlink edits, environment global config for `CUDA_HOME`, or any command that writes to `/usr/local/cuda`, `/usr/share`, `/etc`, `/lib/modules`, or GPU driver paths).
  - Treat CUDA as read-only dependency for now: probing is allowed, installation/configuration changes are not.

## Development conventions
- Keep paths relative to repo root unless user requests absolute paths.
- Use clear module boundaries and include short module docstrings.
- Keep logs lightweight and machine-parseable (JSON preferred).
- Maintain checkpoints and versioned experiment manifests for reproducibility.

## Testing discipline
- Project is TDD-first.
- Write a failing test before each behavior change (including geometry validation, oracle integration, and reward logic updates).
- Keep a minimal deterministic test harness that exercises the canonical CPU path before any CUDA optimization is accepted.
- Add regression tests for:
  - evaluator failure handling (timeout, divergence, invalid geometry),
  - reward invariants across equivalent random seeds,
  - backward-compatible info dict contents.

## Deliverable quality checks (before claiming completion)
- `README`-style run instructions must exist and be accurate.
- Environment should expose a single “single command start” path.
- Known failure modes should be documented with mitigations in `SPEC.md`.

## Working commands (RayLib field render)
- Baseline offscreen check (use this first):
  - `DISPLAY=:0 PYTHONPATH=src ./.venv/bin/python scripts/render_raylib.py --offscreen --offscreen-output logs/field-align/final_sync_fill_bounds.png --naca 2412 --field-mode pressure --alpha-start 4 --alpha-stop 4 --re 1000000 --field-core 0.08 --field-cols 560 --field-rows 360 --field-orientation normal --field-rect-mode panel --field-domain-mode bounds --field-origin-mode leading_edge`
  - This uses the fixed geometry/field basis (`geometry` and `bounds` now share the same view box).

- Live preview (Escape to exit):
  - `DISPLAY=:0 PYTHONPATH=src ./.venv/bin/python scripts/render_raylib.py --naca 2412 --field-mode pressure --alpha-start 2 --alpha-stop 10 --alpha-step 1 --re 1000000 --field-cols 560 --field-rows 360 --field-orientation normal --field-rect-mode panel --field-domain-mode bounds --field-origin-mode leading_edge`

- Wider context (if you want less steep gradients around the body):
  - `DISPLAY=:0 PYTHONPATH=src ./.venv/bin/python scripts/render_raylib.py --offscreen --offscreen-output logs/field-align/final_sync_fill_wider.png --naca 2412 --field-mode pressure --alpha-start 4 --alpha-stop 4 --re 1000000 --field-core 0.08 --field-margin 6.0 --field-cols 560 --field-rows 360 --field-orientation normal --field-rect-mode panel --field-domain-mode padded --field-origin-mode leading_edge --field-strength 1.5`

- Orientation sweep (quick full-screen comparison, no display dependency):
  - `DISPLAY=:0 PYTHONPATH=src ./.venv/bin/python scripts/render_raylib.py --offscreen --offscreen-output logs/field-align/sweep.png --naca 2412 --field-mode pressure --alpha-start 4 --alpha-stop 4 --re 1000000 --field-cols 560 --field-rows 360 --field-orientation-sweep`
