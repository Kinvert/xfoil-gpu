"""Runtime checks for CLI scripts without environment assumptions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_script(script: str, args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env_vars = os.environ.copy()
    if env is not None:
        env_vars.update(env)
    env_vars.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        env=env_vars,
        check=False,
    )


def test_cpu_smoke_runs_without_pythonpath(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "smoke.json"
    proc = _run_script(
        str(repo_root / "scripts" / "cpu_smoke.py"),
        [
            "--xfoil",
            str(repo_root / "scripts" / "fake_xfoil.sh"),
            "--out",
            str(out_path),
        ],
    )

    assert proc.returncode == 0
    smoke = json.loads(proc.stdout)

    assert smoke["environment"]["xfoil_executable"] == str(repo_root / "scripts" / "fake_xfoil.sh")
    assert smoke["fallback_smoke"]["status"] == "ok"
    assert smoke["oracle_smoke"]["status"] == "ok"
    assert smoke["oracle_smoke"]["error"] is None
    assert out_path.exists()


def test_bench_cpu_runs_without_pythonpath(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_script(
        str(repo_root / "scripts" / "bench_cpu.py"),
        [
            "--geometry",
            "data/naca0012.dat",
            "--xfoil",
            str(repo_root / "scripts" / "fake_xfoil.sh"),
            "--trials",
            "1",
            "--repeats",
            "1",
            "--alpha-start",
            "2",
            "--alpha-stop",
            "2",
            "--alpha-step",
            "2",
        ],
    )

    assert proc.returncode == 0
    report = json.loads(proc.stdout)

    assert report["config"]["xfoil_executable"] == str(repo_root / "scripts" / "fake_xfoil.sh")
    assert report["config"]["trials"] == 1
    assert report["config"]["repeats"] == 1
    assert report["status_counts"] == {"ok": 1}
    assert report["wall_seconds"]["mean"] > 0.0
    assert report["case_seconds"]["mean"] > 0.0


def test_engine_smoke_runs_without_pythonpath():
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_script(
        str(repo_root / "scripts" / "engine_smoke.py"),
        [
            "--geometry",
            "data/naca0012.dat",
            "--xfoil",
            str(repo_root / "scripts" / "fake_xfoil.sh"),
            "--require-oracle",
        ],
    )

    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["environment"]["backend"] == "cpu"
    assert report["cases"]["oracle"]["status"] == "ok"
    assert len(report["cases"]["batch"]) == 3


def test_bench_cpu_writes_manifest(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "bench_manifest.json"

    proc = _run_script(
        str(repo_root / "scripts" / "bench_cpu.py"),
        [
            "--geometry",
            "data/naca0012.dat",
            "--xfoil",
            str(repo_root / "scripts" / "fake_xfoil.sh"),
            "--trials",
            "1",
            "--repeats",
            "1",
            "--alpha-start",
            "2",
            "--alpha-stop",
            "2",
            "--alpha-step",
            "2",
            "--out",
            str(out_path),
        ],
    )

    assert proc.returncode == 0
    assert out_path.exists()

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["geometry"]["count"] > 0
    assert report["status_counts"] == {"ok": 1}
    assert report["xfoil"]["available"] is True
    assert report["xfoil"]["resolved"] == str(repo_root / "scripts" / "fake_xfoil.sh")


def test_bench_cpu_supports_toml_config(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "bench_cpu.toml"
    out_path = tmp_path / "bench_from_config.json"
    cfg_path.write_text(
        f"""
[query]
reynolds = 1_000_000

[benchmark]
alpha_start = 2.0
alpha_stop = 2.0
alpha_step = 2.0
repeats = 1
trials = 1

[backend]
xfoil_executable = "{(repo_root / 'scripts' / 'fake_xfoil.sh')}"
cache_results = false
timeout_seconds = 20.0
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    proc = _run_script(
        str(repo_root / "scripts" / "bench_cpu.py"),
        [
            "--config",
            str(cfg_path),
            "--out",
            str(out_path),
        ],
    )

    assert proc.returncode == 0
    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["config"]["trials"] == 1
    assert report["config"]["repeats"] == 1
    assert report["config"]["xfoil_executable"] == str(repo_root / "scripts" / "fake_xfoil.sh")
    assert report["config_source"]["backend_from_config"] is True
    assert report["geometry"]["count"] > 0
    assert report["status_counts"] == {"ok": 1}


def test_bench_cpu_supports_benchmark_geometry_override(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "bench_cpu.toml"
    cfg_path.write_text(
        f"""
[benchmark]
geometry = "data/naca0012.dat"
alpha_start = 1.0
alpha_stop = 1.0
alpha_step = 1.0
repeats = 1
trials = 1

[backend]
xfoil_executable = "{repo_root / 'scripts' / 'fake_xfoil.sh'}"
cache_results = false
timeout_seconds = 20.0
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    proc = _run_script(
        str(repo_root / "scripts" / "bench_cpu.py"),
        [
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "bench_geom.json"),
        ],
    )

    assert proc.returncode == 0
    report = json.loads((tmp_path / "bench_geom.json").read_text(encoding="utf-8"))
    assert report["geometry"]["path"] == str(repo_root / "data" / "naca0012.dat")
    assert report["config"]["trials"] == 1
    assert report["config"]["repeats"] == 1


def test_benchmark_cpu_script_writes_default_manifest(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    before = set(p.name for p in (repo_root / "logs").glob("cpu_baseline_*.json"))
    proc = subprocess.run(
        [str(repo_root / "scripts" / "benchmark_cpu.sh"), "--trials", "1", "--repeats", "1"],
        cwd="/tmp",
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Running CPU baseline benchmark" in proc.stderr
    after = set(p.name for p in (repo_root / "logs").glob("cpu_baseline_*.json") if p.is_file())
    assert len(after - before) >= 1


def test_bench_native_cpu_runs_python_mode_without_pythonpath(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_cpu_python.json"
    proc = _run_script(
        str(repo_root / "scripts" / "bench_native_cpu.py"),
        [
            "--geometry",
            "data/naca0012.dat",
            "--backend-mode",
            "python",
            "--trials",
            "1",
            "--repeats",
            "1",
            "--alpha-start",
            "2",
            "--alpha-stop",
            "2",
            "--alpha-step",
            "2",
            "--out",
            str(out_path),
        ],
    )

    assert proc.returncode == 0
    assert out_path.exists()
    report = json.loads(proc.stdout)
    assert report["config"]["backend_mode"] == "python"
    assert report["selected_mode"] == "python"
    assert report["modes"]["active"]["mode"] == "python"
    assert report["modes"]["active"]["backend_source"] == "python"
    assert report["modes"]["active"]["status_counts"] == {"ok": 1}
    assert report["modes"]["active"]["wall_seconds"]["mean"] > 0.0


def test_bench_native_cpu_compare_mode_reports_available_backends(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "native_cpu_compare.json"
    proc = _run_script(
        str(repo_root / "scripts" / "bench_native_cpu.py"),
        [
            "--geometry",
            "data/naca0012.dat",
            "--compare",
            "--trials",
            "1",
            "--repeats",
            "1",
            "--alpha-start",
            "2",
            "--alpha-stop",
            "2",
            "--alpha-step",
            "2",
            "--out",
            str(out_path),
        ],
    )

    assert proc.returncode == 0
    assert out_path.exists()
    report = json.loads(proc.stdout)
    assert report["config"]["compare"] is True
    assert "python" in report["modes"]
    python_mode = report["modes"]["python"]
    assert python_mode["mode"] == "python"
    assert python_mode["backend_source"] == "python"

    compiled_mode = report["modes"]["compiled"]
    if compiled_mode.get("status") == "unavailable":
        assert compiled_mode["kernel_available"] is False
    else:
        assert compiled_mode["backend_source"] == "compiled"
