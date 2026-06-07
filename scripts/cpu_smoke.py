#!/usr/bin/env python3
"""Deterministic CPU smoke check for the XFOIL evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

# Make local package import work when the repo is executed without installation.
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from xfoil_port.backends.cpu import CpuXfoilConfig, CpuXfoilEvaluator
from xfoil_port.types import XfoilQuery


def _read_geometry(path: Path) -> list[tuple[float, float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    points = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            continue
        points.append((float(tokens[0]), float(tokens[1])))
    if not points:
        raise RuntimeError(f"no points read from geometry file {path}")
    return points


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if os.path.sep not in str(value) and not (os.path.altsep and os.path.altsep in str(value)):
        return path if path.exists() else (repo_root / path)
    return path if path.is_absolute() else (repo_root / path)


def _run_case(name: str, config: CpuXfoilConfig, geometry_path: Path) -> Dict[str, object]:
    evaluator = CpuXfoilEvaluator(config=config)
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, mach=0.0)
    result = evaluator.evaluate(_read_geometry(geometry_path), query, name=name).payload
    return {
        "name": name,
        "status": result.status,
        "cl": result.cl,
        "cd": result.cd,
        "cm": result.cm,
        "meta": result.meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU smoke check for XFOIL evaluator.")
    parser.add_argument("--geometry", default="data/naca0012.dat")
    parser.add_argument("--xfoil", default=os.environ.get("XFOIL_EXE", "xfoil"))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--require-oracle",
        action="store_true",
        help="Fail if real xfoil executable path is unavailable or returns non-ok status.",
    )
    parser.add_argument("--out", default="", help="Optional output JSON path for smoke report.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    geometry_path = _resolve_path(args.geometry, repo_root)
    xfoil_exec = _resolve_path(args.xfoil, repo_root)
    if not xfoil_exec.exists() and not xfoil_exec.is_absolute():
        xfoil_exec = Path(args.xfoil)

    if not geometry_path.exists():
        raise RuntimeError(f"geometry file not found: {geometry_path}")

    outcomes = {
        "environment": {
            "xfoil_executable": str(xfoil_exec),
        },
        "fallback_smoke": _run_case(
            "fallback",
            CpuXfoilConfig(
                xfoil_executable=str(xfoil_exec),
                enable_fallback=True,
                timeout_seconds=args.timeout,
            ),
            geometry_path=geometry_path,
        ),
    }

    try:
        outcomes["oracle_smoke"] = _run_case(
            "oracle",
            CpuXfoilConfig(
                xfoil_executable=str(xfoil_exec),
                enable_fallback=False,
                timeout_seconds=args.timeout,
            ),
            geometry_path=geometry_path,
        )
        outcomes["oracle_smoke"]["error"] = None
    except Exception as exc:
        outcomes["oracle_smoke"] = {
            "name": "oracle",
            "status": "error",
            "error": str(exc),
        }

    report = json.dumps(outcomes, indent=2, sort_keys=True)
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")

    fallback_ok = str(outcomes["fallback_smoke"]["status"]).startswith(("ok", "fallback_"))
    oracle_ok = outcomes["oracle_smoke"]["status"] == "ok"
    if args.require_oracle:
        return 0 if fallback_ok and oracle_ok else 1
    if not fallback_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
