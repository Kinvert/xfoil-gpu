import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [script, *args],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    return proc


def test_run_cpu_chooses_fallback_and_emits_manifest():
    proc = _run(
        str(_repo_root() / "scripts" / "run_cpu.sh"),
        ["data/naca0012.dat", "--alpha", "2", "--re", "500000"],
    )

    assert proc.returncode == 0
    assert proc.stderr.strip().startswith('{"xfoil_launch"')
    manifest = json.loads(proc.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "fallback"
    assert manifest["xfoil_launch"]["mode_forced"]["require_oracle"] is False
    assert manifest["xfoil_launch"]["mode_forced"]["use_mock"] is False
    assert manifest["xfoil_launch"]["xfoil"]["available"] in (False, True)
    output = json.loads(proc.stdout)
    assert output["status"] in {"fallback_missing_executable", "ok"}


def test_run_cpu_fakes_oracle_with_repo_script():
    proc = _run(
        str(_repo_root() / "scripts" / "run_cpu.sh"),
        [
            "data/naca0012.dat",
            "--alpha",
            "5",
            "--re",
            "1000000",
            "--xfoil",
            str(_repo_root() / "scripts" / "fake_xfoil.sh"),
        ],
    )

    assert proc.returncode == 0
    assert proc.stderr.strip().startswith('{"xfoil_launch"')
    manifest = json.loads(proc.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "oracle"
    assert manifest["xfoil_launch"]["xfoil"]["available"] is True
    output = json.loads(proc.stdout)
    assert output["status"] == "ok"
    assert output["cl"] == 0.5


def test_run_cpu_resolves_repo_relative_geometry_when_run_outside_repo():
    script = str(_repo_root() / "scripts" / "run_cpu.sh")
    proc = _run(
        script,
        ["data/naca0012.dat", "--alpha", "2", "--re", "500000"],
    )
    proc2 = subprocess.run(
        [script, "data/naca0012.dat", "--alpha", "2", "--re", "500000"],
        capture_output=True,
        text=True,
        cwd="/tmp",
        shell=False,
    )

    assert proc.returncode == 0
    assert proc2.returncode == 0
    assert proc2.stderr.strip().startswith('{"xfoil_launch"')
    manifest = json.loads(proc2.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "fallback"
    output = json.loads(proc2.stdout)
    assert output["status"] in {"fallback_missing_executable", "ok"}


def test_run_cpu_resolves_repo_relative_xfoil_arg_when_run_outside_repo():
    script = str(_repo_root() / "scripts" / "run_cpu.sh")
    proc = subprocess.run(
        [script, "data/naca0012.dat", "--alpha", "5", "--re", "1000000", "--xfoil", "scripts/fake_xfoil.sh"],
        capture_output=True,
        text=True,
        cwd="/tmp",
        shell=False,
    )

    assert proc.returncode == 0
    assert proc.stderr.strip().startswith('{"xfoil_launch"')
    manifest = json.loads(proc.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "oracle"
    assert manifest["xfoil_launch"]["xfoil"]["available"] is True
    output = json.loads(proc.stdout)
    assert output["status"] == "ok"
    assert output["cl"] == 0.5


def test_run_cpu_resolves_repo_relative_xfoil_eq_arg_when_run_outside_repo():
    script = str(_repo_root() / "scripts" / "run_cpu.sh")
    proc = subprocess.run(
        [script, "data/naca0012.dat", "--alpha", "5", "--re", "1000000", "--xfoil=scripts/fake_xfoil.sh"],
        capture_output=True,
        text=True,
        cwd="/tmp",
        shell=False,
    )

    assert proc.returncode == 0
    manifest = json.loads(proc.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "oracle"
    assert manifest["xfoil_launch"]["xfoil"]["available"] is True
    output = json.loads(proc.stdout)
    assert output["status"] == "ok"
    assert output["cl"] == 0.5


def test_run_cpu_runs_native_mode_from_wrapper():
    script = str(_repo_root() / "scripts" / "run_cpu.sh")
    proc = subprocess.run(
        [script, "data/naca0012.dat", "--alpha", "5", "--re", "1000000", "--backend", "native"],
        capture_output=True,
        text=True,
        cwd="/tmp",
        shell=False,
    )

    assert proc.returncode == 0
    manifest = json.loads(proc.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "native"
    output = json.loads(proc.stdout)
    assert output["status"] == "ok"
    assert output["meta"]["source"] in {"native_cpu_approx", "native_cpu_approx_cpp"}


def test_run_cpu_native_mode_is_inferred_from_config():
    script = str(_repo_root() / "scripts" / "run_cpu.sh")
    proc = subprocess.run(
        [
            script,
            "data/naca0012.dat",
            "--config",
            "configs/cpu_baseline_native.toml",
            "--alpha",
            "5",
            "--re",
            "1000000",
        ],
        capture_output=True,
        text=True,
        cwd="/tmp",
        shell=False,
    )

    assert proc.returncode == 0
    manifest = json.loads(proc.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "native"
    output = json.loads(proc.stdout)
    assert output["status"] == "ok"
    assert output["meta"]["source"] in {"native_cpu_approx", "native_cpu_approx_cpp"}


def test_run_cpu_explicit_backend_overrides_config(tmp_path):
    script = str(_repo_root() / "scripts" / "run_cpu.sh")
    proc = subprocess.run(
        [
            script,
            "data/naca0012.dat",
            "--config",
            "configs/cpu_baseline_native.toml",
            "--backend",
            "cpu",
            "--alpha",
            "5",
            "--re",
            "1000000",
        ],
        capture_output=True,
        text=True,
        cwd="/tmp",
        shell=False,
    )

    assert proc.returncode == 0
    manifest = json.loads(proc.stderr.strip().splitlines()[-1])
    assert manifest["xfoil_launch"]["mode"] == "fallback"
