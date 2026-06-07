"""CLI-level behavior tests."""

from xfoil_port import cli


def test_cli_rejects_require_oracle_and_use_mock(monkeypatch, tmp_path):
    geom = tmp_path / "naca0012.dat"
    geom.write_text("0.0 0.0\n1.0 0.0\n1.0 0.01\n0.0 0.0\n", encoding="utf-8")

    monkeypatch.setattr(cli.sys, "argv", [
        "xfoil-port",
        str(geom),
        "--alpha",
        "5",
        "--re",
        "1000000",
        "--require-oracle",
        "--use-mock",
    ])
    exit_code = cli.main()

    assert exit_code == 2


def test_cli_fallback_mode_is_executed_with_mock(monkeypatch, tmp_path):
    geom = tmp_path / "naca0012.dat"
    geom.write_text("0.0 0.0\n1.0 0.0\n1.0 0.01\n0.0 0.0\n", encoding="utf-8")

    monkeypatch.setattr(cli.sys, "argv", [
        "xfoil-port",
        str(geom),
        "--alpha",
        "2",
        "--re",
        "500000",
        "--use-mock",
    ])

    # With deterministic fallback enabled, output path should execute and return success.
    exit_code = cli.main()
    assert exit_code == 0


def test_cli_reads_toml_config(tmp_path, monkeypatch):
    geom = tmp_path / "naca0012.dat"
    geom.write_text("0.0 0.0\n1.0 0.0\n1.0 0.01\n0.0 0.0\n", encoding="utf-8")
    cfg = tmp_path / "cpu_reference.toml"
    cfg.write_text(
        """
[query]
alpha = 5.0
reynolds = 1000000

[backend]
use_mock = true
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.sys, "argv", [
        "xfoil-port",
        str(geom),
        "--config",
        str(cfg),
    ])

    exit_code = cli.main()
    assert exit_code == 0


def test_cli_resolves_relative_geometry_from_repo_root(monkeypatch, tmp_path):
    geom = tmp_path / "xfoil-port-test-geo.dat"
    geom.write_text("0.0 0.0\n1.0 0.0\n1.0 0.01\n0.0 0.0\n", encoding="utf-8")

    cwd_before = tmp_path
    monkeypatch.chdir(cwd_before)

    # When running from outside repo root, the CLI should still resolve package
    # geometry from the repo root only for relative repo-layout files.
    monkeypatch.setattr(cli.sys, "argv", [
        "xfoil-port",
        str(cwd_before / "xfoil-port-test-geo.dat"),
        "--alpha",
        "1",
        "--re",
        "1000000",
        "--use-mock",
    ])
    assert cli.main() == 0

    monkeypatch.setattr(cli.sys, "argv", [
        "xfoil-port",
        "data/naca0012.dat",
        "--alpha",
        "5",
        "--re",
        "1000000",
        "--use-mock",
    ])
    assert cli.main() == 0


def test_cli_runs_native_backend_without_xfoil(monkeypatch, tmp_path):
    geom = tmp_path / "naca0012.dat"
    geom.write_text("0.0 0.0\n1.0 0.0\n1.0 0.01\n0.0 0.0\n", encoding="utf-8")

    monkeypatch.setattr(cli.sys, "argv", [
        "xfoil-port",
        str(geom),
        "--alpha",
        "5",
        "--re",
        "1000000",
        "--backend",
        "native",
    ])

    assert cli.main() == 0


def test_cli_runs_native_backend_from_config(monkeypatch, tmp_path):
    geom = tmp_path / "naca0012.dat"
    geom.write_text("0.0 0.0\n1.0 0.0\n1.0 0.02\n0.0 0.0\n", encoding="utf-8")
    cfg = tmp_path / "native_cpu.toml"
    cfg.write_text(
        """
[query]
alpha = 5.0
reynolds = 1000000

[backend]
backend = "native"
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.sys, "argv", [
        "xfoil-port",
        str(geom),
        "--config",
        str(cfg),
        "--xfoil",
        "xfoil",
    ])

    assert cli.main() == 0
