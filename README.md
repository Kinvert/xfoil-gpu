# XFOIL CPU Port

This project contains a CPU-first XFOIL evaluator with a stable backend interface for later GPU-era acceleration.

## Quick status

- CPU-only implementation is available and implemented behind a strict interface.
- CUDA runtime is treated as read-only for now (no CUDA installation/toolchain writes).
- GPU backends can be added behind the same `XfoilEngine` API.
- CPU fallback mode is supported and deterministic when legacy XFOIL is unavailable.

## Installation

```bash
cd /home/claude/xfoil
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv --python 3.11
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e .
```

If package installation is blocked in an environment without internet, run directly with:

```bash
cd /home/claude/xfoil
source .venv/bin/activate
PYTHONPATH=src xfoil-port ...
```

Single-command quickstart (bootstrap + deterministic CPU run):

```bash
cd /home/claude/xfoil
./scripts/quickstart_cpu.sh
```

You can override the quickstart settings (deterministic fallback):

```bash
cd /home/claude/xfoil
./scripts/quickstart_cpu.sh 5 1000000
```

## Runtime dependency (required)

- This project expects `xfoil` (the legacy executable) in your PATH for the CPU backend.
- Install it via your system package manager if available, or build/install from source in a separate environment.
- In this environment, `xfoil` is not installed yet, so the default CLI call path returns a dependency error until the dependency is available.
- For deterministic smoke-testing without XFOIL, use fallback mode:
  - `PYTHONPATH=src python -m xfoil_port.cli /tmp/naca0012.dat --alpha 5.0 --re 1000000 --mach 0.0 --use-mock`

Dependency bootstrap check:

```bash
cd /home/claude/xfoil
PYTHONPATH=src .venv/bin/python scripts/env_probe.py
```

If the manifest reports `"available": false` for xfoil:
- install xfoil system-wide (where network/package access is available), or
- provide a local executable with `--xfoil /path/to/xfoil`,
- set `XFOIL_EXE=/path/to/xfoil` in the environment,
- continue development with `--use-mock` to exercise deterministic CPU fallback behavior.

## Run one evaluation

```bash
cat > /tmp/naca0012.dat <<'EOF'
1.000000 0.000000
0.900000 0.015000
0.800000 0.020000
0.700000 0.019000
0.600000 0.016000
0.500000 0.010000
0.400000 0.000000
0.300000 -0.008000
0.200000 -0.015000
0.100000 -0.020000
0.000000 0.000000
1.000000 0.000000
EOF

xfoil-port /tmp/naca0012.dat --alpha 5.0 --re 1000000 --mach 0.0

PYTHONPATH=src python -m xfoil_port.cli /tmp/naca0012.dat --alpha 5.0 --re 1000000 --mach 0.0 --use-mock
xfoil-port /tmp/naca0012.dat --alpha 5.0 --re 1000000 --mach 0.0 --use-mock
xfoil-port /tmp/naca0012.dat --alpha 5.0 --re 1000000 --mach 0.0 --require-oracle
xfoil-port /home/claude/xfoil/data/naca0012.dat --alpha 5.0 --re 1000000 --mach 0.0
PYTHONPATH=src python -m xfoil_port.cli data/naca0012.dat --config configs/cpu_reference.toml
PYTHONPATH=src python -m xfoil_port.cli data/naca0012.dat --alpha 5.0 --re 1000000 --mach 0.0 --use-mock
```

## RayLib visual renderer (DISPLAY=:0)

Install a RayLib binding in the project venv:

```bash
cd /home/claude/xfoil
UV_CACHE_DIR=/tmp/uv-cache uv pip install raylib
```

Render a standard NACA 0012 sweep:

```bash
cd /home/claude/xfoil
UV_CACHE_DIR=/tmp/uv-cache env PYTHONPATH=src DISPLAY=:0 ./.venv/bin/python scripts/render_raylib.py \
  --geometry data/naca0012.dat \
  --alpha-start 2 \
  --alpha-stop 8 \
  --alpha-step 1 \
  --re 1000000
```

The renderer opens a desktop window with:
- airfoil geometry (left panel),
- `CL` vs `alpha` (upper right),
- `CD` vs `alpha` (lower right).

Aligned pressure-field check for NACA2412 (filled texture, no geometry-y-stretch):

```bash
cd /home/claude/xfoil
UV_CACHE_DIR=/tmp/uv-cache env PYTHONPATH=src DISPLAY=:0 ./.venv/bin/python scripts/render_raylib.py --offscreen --offscreen-output /tmp/airfoil_render.png --naca 2412 --field-mode pressure --alpha-start 4 --alpha-stop 4 --re 1000000 --field-core 0.08 --field-cols 560 --field-rows 360 --field-orientation normal --field-rect-mode panel --field-domain-mode bounds --field-origin-mode leading_edge
```

Headless RayLib render (no display dependency):

```bash
cd /home/claude/xfoil
UV_CACHE_DIR=/tmp/uv-cache env PYTHONPATH=src ./.venv/bin/python scripts/render_raylib.py \
  --offscreen \
  --offscreen-output /tmp/airfoil_render.png \
  --geometry data/naca0012.dat \
  --alpha-start 2 \
  --alpha-stop 8 \
  --alpha-step 1 \
  --re 1000000
```

This writes a portable PNG and proves RayLib rendering worked:

```bash
ls -l /tmp/airfoil_render.png
```

Native CPU mode (no legacy XFOIL needed):

```bash
PYTHONPATH=src python -m xfoil_port.cli data/naca0012.dat --alpha 5.0 --re 1000000 --backend native
./scripts/run_cpu.sh data/naca0012.dat --alpha 5.0 --re 1000000 --backend native
./scripts/bench_cpu.py --backend native --geometry data/naca0012.dat --trials 1 --repeats 1
```

GPU-ready shape note:
- The native approximation is implemented in a small kernel-style module (`backends/native_cpu_kernels.py`) with a stable scalar contract.
- `NativeCpuXfoilEvaluator` attempts to import an optional compiled accelerator module (`backends/native_cpu_cpp`) and will fall back to the pure-Python kernel if it is not present.
- This keeps the immediate CPU path deterministic and runnable while making C++/CUDA swaps a mechanical replacement of the kernel contract.

## Optional native C++ build (CPU-only)

The optional compiled path can be enabled without changing any call sites:

```bash
cd /home/claude/xfoil
python scripts/build_native_cpu_cpp.py
```

After building successfully, `NativeCpuXfoilEvaluator` will prefer the compiled kernel and continue to emit the same fields in its result meta (`source: native_cpu_approx_cpp`).

Single-command local fallback run:

```bash
cd /home/claude/xfoil
./scripts/run_cpu.sh /tmp/naca0012.dat --alpha 5.0 --re 1000000
```
The run prints a machine-parseable launch manifest on `stderr` before evaluation, for example:

```json
{"xfoil_launch":{"mode":"fallback","xfoil":{"requested":"xfoil","resolved":"","available":false},"mode_forced":{"require_oracle":false,"use_mock":false}}}
```

Single-command oracle run (hard-fail if XFOIL is unavailable):

```bash
cd /home/claude/xfoil
./scripts/run_cpu.sh /tmp/naca0012.dat --alpha 5.0 --re 1000000 --require-oracle
```
Deterministic local oracle smoke (no external XFOIL needed):

```bash
cd /home/claude/xfoil
./scripts/run_cpu.sh data/naca0012.dat --alpha 5.0 --re 1000000 --xfoil scripts/fake_xfoil.sh
```

Reproducible local sample geometry:

```bash
cd /home/claude/xfoil
./scripts/run_cpu_fallback.sh data/naca0012.dat --alpha 5.0 --re 1000000
```

Deterministic CPU health check:

```bash
cd /home/claude/xfoil
mkdir -p logs
.venv/bin/python scripts/cpu_smoke.py
.venv/bin/python scripts/cpu_smoke.py --out logs/cpu_smoke.json
.venv/bin/python scripts/cpu_smoke.py --require-oracle
.venv/bin/python scripts/env_probe.py
```

You can also run both scripts without PYTHONPATH because they bootstrap local imports:

```bash
cd /home/claude/xfoil
./scripts/cpu_smoke.py --xfoil scripts/fake_xfoil.sh
./scripts/bench_cpu.py --geometry data/naca0012.dat --xfoil scripts/fake_xfoil.sh --trials 1 --repeats 1
```

CPU baseline benchmark with manifest:

```bash
cd /home/claude/xfoil
mkdir -p logs
./scripts/bench_cpu.py \
  --geometry data/naca0012.dat \
  --xfoil scripts/fake_xfoil.sh \
  --trials 1 \
  --repeats 1 \
  --alpha-start 2 \
  --alpha-stop 6 \
  --alpha-step 2 \
  --out logs/cpu_baseline_manifest.json
```

The manifest includes:
- deterministic run metadata (`benchmark_id`, UTC timestamp)
- runtime/environment snapshot
- geometry signature (`sha256`)
- xfoil executable resolution (`requested`/`resolved`/`available`)
- timing + status summary

Use this manifest as the baseline for future CUDA-side speedup comparisons.

Compare native backend variants (compiled vs pure-Python proxy) with the native
benchmark script:

```bash
cd /home/claude/xfoil
./scripts/bench_native_cpu.py --geometry data/naca0012.dat --backend-mode auto --trials 1 --repeats 1
./scripts/bench_native_cpu.py --geometry data/naca0012.dat --backend-mode python --trials 1 --repeats 1
./scripts/bench_native_cpu.py --geometry data/naca0012.dat --backend-mode compiled --trials 1 --repeats 1
./scripts/bench_native_cpu.py --geometry data/naca0012.dat --compare --trials 1 --repeats 1
```

One-command baseline run script:

```bash
cd /home/claude/xfoil
./scripts/benchmark_cpu.sh --trials 1 --repeats 1 --alpha-start 2 --alpha-stop 8 --alpha-step 2
```

The script emits `logs/cpu_baseline_<timestamp>.json` by default and runs from any
working directory while resolving repo resources from `/home/claude/xfoil`.

You can also drive the benchmark from a TOML config:

```bash
cd /home/claude/xfoil
cat > /tmp/bench_cpu_conf.toml <<'EOF'
[query]
reynolds = 1_000_000

[benchmark]
alpha_start = 2.0
alpha_stop = 8.0
alpha_step = 2.0
repeats = 1
trials = 1

[backend]
xfoil_executable = "scripts/fake_xfoil.sh"
cache_results = false
timeout_seconds = 20.0
EOF

./scripts/bench_cpu.py --config /tmp/bench_cpu_conf.toml --out logs/cpu_baseline_manifest.json
```

## Output

- JSON with `cl`, `cd`, `cm`, and `status`.
- The solver output files and command script are currently written to a temporary work dir by default.
- Geometry is validated for:
  - finite numeric points
  - closed airfoil loop
  - no duplicate consecutive points
  - bounded coordinates (recommended `x ∈ [0, 1]`, `|y| <= 1`) for deterministic handling

## Next steps

- Add geometry auto-close policy controls.
- Add optional subprocess worker pool with caching and hard timeout.
- Replace/augment CPU backend with a CUDA-aware accelerator later without changing external call shape.

## Batch API (CPU now, GPU later)

```python
from xfoil_port import XfoilEngine
from xfoil_port.types import XfoilBatchInput, XfoilQuery

engine = XfoilEngine()
batch = [
    XfoilBatchInput(name="naca0012_5deg", geometry_points=[(0.0, 0.0), (1.0, 0.01), (1.0, 0.0), (0.0, 0.0)], query=XfoilQuery(alpha_deg=5.0, reynolds=1e6)),
    XfoilBatchInput(name="naca0012_2deg", geometry_points=[(0.0, 0.0), (1.0, 0.01), (1.0, 0.0), (0.0, 0.0)], query=XfoilQuery(alpha_deg=2.0, reynolds=1e6)),
]
results = engine.evaluate_many(batch)
```

`evaluate_many` keeps evaluation order deterministic and preserves the same `XfoilResult` contract.

## RL-ready transition API (CPU first)

Use `XfoilStepEnv` to get a deterministic `c_step` contract compatible with later PufferLib integration:

```python
from xfoil_port import XfoilStepEnv
from xfoil_port.types import XfoilQuery

env = XfoilStepEnv()
state = env.reset(
    geometry_points=[(0.0, 0.0), (1.0, 0.01), (1.0, 0.0), (0.0, 0.0)],
    query=XfoilQuery(alpha_deg=5.0, reynolds=1_000_000),
    target_cl=0.5,
    target_cd=0.01,
)
state, reward, done, info = env.c_step(state, {"alpha_delta": -0.25, "reynolds_delta": 0.02})
```

Reward is bounded/saturating and the transition state remains immutable and deterministic.
