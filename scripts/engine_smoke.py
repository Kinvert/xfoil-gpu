#!/usr/bin/env python3
"""Deterministic engine smoke check for backend contract validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from xfoil_port import XfoilBackend, XfoilEngine
from xfoil_port.backends.cpu import CpuXfoilConfig
from xfoil_port.types import XfoilBatchInput, XfoilQuery


@dataclass(frozen=True)
class _CaseResult:
    label: str
    status: str
    cl: float | None
    cd: float | None
    cm: float | None
    backend: str | None
    error: str | None


def _resolve_geometry(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if (repo_root / path).is_file():
        return repo_root / path
    return path


def _read_geometry(path: Path) -> list[tuple[float, float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    points = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            continue
        points.append((float(tokens[0]), float(tokens[1])))
    if not points:
        raise RuntimeError(f"no points read from geometry {path}")
    return points


def _run_case(engine: XfoilEngine, geometry: list[tuple[float, float]], query: XfoilQuery, *, label: str, strict: bool) -> _CaseResult:
    try:
        result = engine.evaluate(geometry, query, name=f"{label}-single")
        return _CaseResult(
            label=label,
            status=result.status,
            cl=result.cl,
            cd=result.cd,
            cm=result.cm,
            backend=result.meta.get("backend"),
            error=None,
        )
    except Exception as exc:
        if strict:
            raise
        return _CaseResult(
            label=label,
            status="error",
            cl=None,
            cd=None,
            cm=None,
            backend=None,
            error=str(exc),
        )


def _run_batch(engine: XfoilEngine, geometry: list[tuple[float, float]]) -> list[_CaseResult]:
    inputs = [
        XfoilBatchInput(
            name=f"batch-{idx}",
            geometry_points=geometry,
            query=XfoilQuery(alpha_deg=2.0 + idx, reynolds=1_000_000 + idx * 10_000),
        )
        for idx in range(3)
    ]
    results = engine.evaluate_many(inputs, meta={"source": "engine_smoke"})
    return [
        _CaseResult(
            label=item.name,
            status=result.status,
            cl=result.cl,
            cd=result.cd,
            cm=result.cm,
            backend=result.meta.get("backend"),
            error=None,
        )
        for item, result in zip(inputs, results)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic CPU engine smoke check.")
    parser.add_argument("--geometry", default="data/naca0012.dat")
    parser.add_argument("--xfoil", default=os.environ.get("XFOIL_EXE", "xfoil"))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--work-dir", default="", help="Override working directory for xfoil runs")
    parser.add_argument(
        "--require-oracle",
        action="store_true",
        help="Fail fast when XFOIL is unavailable.",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Force fallback when oracle missing.",
    )

    args = parser.parse_args()
    if args.require_oracle and args.use_mock:
        raise ValueError("cannot combine --require-oracle and --use-mock")

    geometry_path = _resolve_geometry(args.geometry, _repo_root)
    geometry = _read_geometry(geometry_path)
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, mach=0.0)

    engine = XfoilEngine(
        config=CpuXfoilConfig(
            xfoil_executable=args.xfoil,
            timeout_seconds=args.timeout,
            work_dir=args.work_dir,
            enable_fallback=args.use_mock or not args.require_oracle,
            cache_results=False,
        )
    )

    fallback = _run_case(
        engine=engine,
        geometry=geometry,
        query=query,
        label="fallback",
        strict=False,
    )

    if args.require_oracle:
        oracle = _run_case(
            engine=engine,
            geometry=geometry,
            query=query,
            label="oracle",
            strict=True,
        )
    else:
        oracle = _run_case(
            engine=engine,
            geometry=geometry,
            query=query,
            label="oracle",
            strict=False,
        )

    batch = _run_batch(engine, geometry)

    report = {
        "environment": {
            "python_version": platform.python_version(),
            "engine": "XfoilEngine",
            "backend": engine.backend_id,
        },
        "geometry": {
            "path": str(geometry_path),
            "points": len(geometry),
        },
        "cases": {
            "fallback": fallback.__dict__,
            "oracle": oracle.__dict__,
            "batch_status_counts": dict(Counter(item.status for item in batch)),
            "batch": [item.__dict__ for item in batch],
        },
        "xfoil": {
            "executable": args.xfoil,
        },
        "errors": {
            "fallback_error": fallback.error,
            "oracle_error": oracle.error,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_oracle and oracle.status == "error":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
