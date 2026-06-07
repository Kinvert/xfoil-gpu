import json
import os
import subprocess
import sys
from pathlib import Path


def _run_probe(*, xfoil_exe: str, path: str = "") -> dict:
    env = os.environ.copy()
    env["XFOIL_EXE"] = xfoil_exe
    if path != "":
        env["PATH"] = path
    else:
        env.pop("PATH", None)

    script = Path(__file__).resolve().parents[1] / "scripts" / "env_probe.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_env_probe_reports_absolute_configured_binary(tmp_path):
    exe = tmp_path / "fake_xfoil.sh"
    exe.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)

    manifest = _run_probe(xfoil_exe=str(exe), path="")

    assert manifest["xfoil"]["configured_binary"] == str(exe)
    assert manifest["xfoil"]["configured_available"] is True
    assert manifest["xfoil"]["available"] is True
    assert manifest["xfoil"]["binary"] == str(exe)


def test_env_probe_reports_missing_configured_binary(tmp_path):
    manifest = _run_probe(xfoil_exe="/definitely/not/a/real/xfoil", path="")

    assert manifest["xfoil"]["configured_binary"] == ""
    assert manifest["xfoil"]["configured_available"] is False
    assert manifest["xfoil"]["available"] is False
