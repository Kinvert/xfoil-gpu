from xfoil_port.backends.cpu import CpuXfoilEvaluator, CpuXfoilConfig
from xfoil_port.backends.native_cpu import NativeCpuXfoilConfig, NativeCpuXfoilEvaluator
from xfoil_port.backends.cpu import _build_xfoil_script
from xfoil_port.types import XfoilQuery
from pathlib import Path


def test_cpu_fallback_is_deterministic_and_cacheable():
    evaluator = CpuXfoilEvaluator(
        CpuXfoilConfig(
            xfoil_executable="__definitely_missing__",
            enable_fallback=True,
            cache_results=True,
        )
    )
    query = XfoilQuery(
        alpha_deg=5.0,
        reynolds=1_000_000,
        mach=0.0,
        iterations=100,
    )
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]

    first = evaluator.evaluate(geometry, query, name="fallback").payload
    second = evaluator.evaluate(geometry, query, name="fallback").payload

    assert first.cl == second.cl
    assert first.cd == second.cd
    assert first.cm == second.cm
    assert first.status == second.status
    assert first.status == "fallback_missing_executable"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.meta["source"] == "fallback_heuristic"
    assert first.meta["query"]["alpha_deg"] == "5"
    assert first.meta["query"]["reynolds"] == "1000000"
    assert first.meta["query"]["mach"] == "0"
    assert first.meta["query"]["iterations"] == "100"
    assert first.meta["query"]["n_crit"] == "9"
    assert first.meta["query"]["n_panels"] == "None"
    assert second.meta["query"] == first.meta["query"]
    assert second.meta["cache_hit"] == "true"
    assert first.elapsed_seconds is not None
    assert second.elapsed_seconds is not None


def test_cpu_fallback_miss_without_xfoil_returns_fallback():
    evaluator = CpuXfoilEvaluator(
        CpuXfoilConfig(
            xfoil_executable="__definitely_missing__",
            enable_fallback=True,
            cache_results=False,
        )
    )
    query = XfoilQuery(alpha_deg=-2.0, reynolds=500_000)
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, -0.01), (0.0, 0.0)]

    result = evaluator.evaluate(geometry, query, name="fallback-miss").payload

    assert result.status == "fallback_missing_executable"
    assert "xfoil_unavailable_fallback" in result.warnings


def test_cpu_backend_parses_fake_xfoil_output(tmp_path):
    fake = tmp_path / "fake_xfoil.sh"
    fake.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

while IFS= read -r line; do
  line="$(printf '%s' "$line")"
  if [[ "$line" == "PACC" ]]; then
    read -r maybe_polar
    if [[ -n "$maybe_polar" ]]; then
      printf "  5.0  0.500  0.0100  0.0002  -0.0200\\n" > "$maybe_polar"
    fi
  fi
done
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    evaluator = CpuXfoilEvaluator(
        CpuXfoilConfig(
            xfoil_executable=str(fake),
            enable_fallback=False,
        )
    )
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, mach=0.0)
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]

    result = evaluator.evaluate(geometry, query, name="fake-xfoil").payload

    assert result.status == "ok"
    assert result.reynolds == query.reynolds
    assert result.mach == query.mach
    assert result.cl == 0.5
    assert result.cd == 0.01
    assert result.cm == -0.02
    assert result.meta["query"]["alpha_deg"] == "5"
    assert result.meta["query"]["reynolds"] == "1000000"
    assert result.meta["query"]["n_crit"] == "9"
    assert result.meta["query"]["n_panels"] == "None"


def test_cpu_backend_resolves_relative_executable_from_work_dir(tmp_path):
    fake = tmp_path / "fake_xfoil.sh"
    fake.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

while IFS= read -r line; do
  line="$(printf '%s' "$line")"
  if [[ "$line" == "PACC" ]]; then
    read -r maybe_polar
    if [[ -n "$maybe_polar" ]]; then
      printf "  5.0  0.500  0.0100  0.0002  -0.0200\\n" > "$maybe_polar"
    fi
  fi
done
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    evaluator = CpuXfoilEvaluator(
        CpuXfoilConfig(
            xfoil_executable="fake_xfoil.sh",
            work_dir=str(tmp_path),
            enable_fallback=False,
        )
    )
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, mach=0.0)
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]

    result = evaluator.evaluate(geometry, query, name="fake-oracle-rel").payload

    assert result.status == "ok"
    assert result.cl == 0.5
    assert result.cd == 0.01
    assert result.cm == -0.02


def test_cpu_backend_falls_back_when_parse_fails(tmp_path):
    fake = tmp_path / "fake_xfoil_empty.sh"
    fake.write_text(
        """#!/usr/bin/env bash\nexit 0\n""",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    evaluator = CpuXfoilEvaluator(
        CpuXfoilConfig(
            xfoil_executable=str(fake),
            enable_fallback=True,
            cache_results=False,
        )
    )
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000)
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]

    result = evaluator.evaluate(geometry, query, name="parse-fail").payload
    assert result.status == "fallback_parse"
    assert result.meta["fallback_reason"] == "no_parseable_output"
    assert result.meta["query"]["alpha_deg"] == "5"
    assert result.meta["query"]["reynolds"] == "1000000"
    assert result.elapsed_seconds is not None
    assert result.elapsed_seconds >= 0.0
    assert result.meta["elapsed_seconds"] is not None


def test_cpu_build_script_includes_optional_controls():
    geom_path = Path("/tmp/fake.dat")
    polar_path = Path("/tmp/fake.polar")
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, n_crit=7, n_panels=128, iterations=250)
    script = _build_xfoil_script(geom_path, polar_path, query)

    assert "PPAR\nN 128\n" in script
    assert "VPAR\nN 7\n" in script
    assert "ITER 250" in script
    assert f"VISC {query.reynolds:.6g}" in script


def test_native_cpu_is_deterministic_and_returns_bounds():
    evaluator = NativeCpuXfoilEvaluator(
        NativeCpuXfoilConfig(
            cache_results=True,
        )
    )
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, mach=0.0, n_panels=64, iterations=200)

    first = evaluator.evaluate(geometry, query, name="native").payload
    second = evaluator.evaluate(geometry, query, name="native").payload

    assert first.status in {"ok", "native_warnings"}
    assert second.status == first.status
    assert second.cache_hit is True
    assert first.cl == second.cl
    assert first.cd == second.cd
    assert first.cm == second.cm
    assert first.meta["source"] in {"native_cpu_approx", "native_cpu_approx_cpp"}
    assert first.meta["query"]["alpha_deg"] == "5"
    assert first.meta["query"]["reynolds"] == "1000000"
    assert first.meta["query"]["mach"] == "0"
    assert first.meta["query"]["iterations"] == "200"
    assert first.meta["query"]["n_crit"] == "9"
    assert first.meta["query"]["n_panels"] == "64"
    assert first.elapsed_seconds is not None
    assert second.elapsed_seconds is not None


def test_native_cpu_uses_cpp_kernel_if_available(monkeypatch):
    import types
    import sys

    def fake_estimate_aero(
        points,
        alpha_deg,
        reynolds,
        mach,
        n_panels,
        iterations,
        stall_alpha_deg,
        residual_floor,
        iterations_to_converge,
    ):
        return {
            "cl": 1.23,
            "cd": 0.056,
            "cm": -0.01,
            "status": "ok",
            "residual": 0.001,
            "iterations_used": 10,
            "iterations_failed": False,
            "warnings": [],
            "features": {
                "n": 4.0,
                "curvature": 0.1,
                "camber": 0.01,
                "thickness_ratio": 0.02,
            },
        }

    fake_module = types.ModuleType("xfoil_port.backends.native_cpu_cpp")
    fake_module.HAS_COMPILED = True
    fake_module.estimate_aero = fake_estimate_aero
    monkeypatch.setitem(sys.modules, "xfoil_port.backends.native_cpu_cpp", fake_module)

    evaluator = NativeCpuXfoilEvaluator(config=NativeCpuXfoilConfig(cache_results=False))
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, mach=0.0, n_panels=64, iterations=200)
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]

    result = evaluator.evaluate(geometry, query, name="native-cpp").payload

    assert result.status == "ok"
    assert result.cl == 1.23
    assert result.cd == 0.056
    assert result.cm == -0.01
    assert result.meta["source"] == "native_cpu_approx_cpp"
