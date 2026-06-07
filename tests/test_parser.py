from pathlib import Path

from xfoil_port.backends.cpu import _parse_polar_file


def test_parse_polar_file_with_header(tmp_path: Path):
    polar = tmp_path / "a.polar"
    polar.write_text(
        """
 alpha   cl    cd   cdp    cm
  5.0  0.5  0.010  0.0002  -0.05
""".strip(),
        encoding="utf-8",
    )
    result = _parse_polar_file(polar, 5.0, reynolds=1_000_000.0, mach=0.0)
    assert result is not None
    assert result.reynolds == 1_000_000.0
    assert result.mach == 0.0
    assert result.alpha_deg == 5.0
    assert abs(result.cl - 0.5) < 1e-12
    assert abs(result.cd - 0.01) < 1e-12
    assert abs(result.cm + 0.05) < 1e-12


def test_parse_polar_file_without_header(tmp_path: Path):
    polar = tmp_path / "b.polar"
    polar.write_text(
        """
   5.0  0.5  0.010  -0.05
""".strip(),
        encoding="utf-8",
    )
    result = _parse_polar_file(polar, 5.0, reynolds=1_000_000.0, mach=0.0)
    assert result is not None
    assert result.reynolds == 1_000_000.0
    assert result.mach == 0.0
    assert result.alpha_deg == 5.0
    assert result.cl == 0.5
    assert result.cd == 0.01
    assert result.cm == -0.05


def test_parse_polar_file_without_header_five_columns(tmp_path: Path):
    polar = tmp_path / "c.polar"
    polar.write_text(
        """
   5.0  0.5  0.010  0.0002  -0.05
""".strip(),
        encoding="utf-8",
    )
    result = _parse_polar_file(polar, 5.0, reynolds=1_000_000.0, mach=0.0)
    assert result is not None
    assert result.reynolds == 1_000_000.0
    assert result.mach == 0.0
    assert result.alpha_deg == 5.0
    assert result.cd == 0.01
    assert result.cm == -0.05
