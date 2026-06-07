#!/usr/bin/env python3
"""Deterministic CPU benchmark harness for XFOIL evaluator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import hashlib
import sys
import statistics
from datetime import datetime, timezone
import time
import tomllib
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from xfoil_port import XfoilBatchInput, XfoilEngine, XfoilQuery
from xfoil_port.backends.cpu import CpuXfoilConfig
from xfoil_port.backends.native_cpu import NativeCpuXfoilConfig


def _first_non_null(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        raise RuntimeError(f"config file not found: {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"config must be a TOML table: {path}")
    return data


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


def _build_batch(geometry: list[tuple[float, float]], alphas: list[float], reynolds: float, repeats: int) -> list[XfoilBatchInput]:
    items: list[XfoilBatchInput] = []
    for idx in range(repeats):
        for alpha_idx, alpha in enumerate(alphas):
            items.append(
                XfoilBatchInput(
                    name=f"case_{idx:03d}_{alpha_idx:03d}",
                    geometry_points=geometry,
                    query=XfoilQuery(
                        alpha_deg=alpha,
                        reynolds=reynolds,
                        mach=0.0,
                    ),
                )
            )
    return items


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _probe_xfoil(executable: str) -> tuple[str, bool]:
    candidate = Path(executable)
    if os.path.sep in executable and not candidate.is_absolute():
        candidate = _repo_root / candidate
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate), True
    resolved = shutil.which(executable)
    return (resolved or "", bool(resolved))


def _geometry_signature(points: list[tuple[float, float]]) -> str:
    payload = json.dumps(points, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU benchmark harness.")
    parser.add_argument("--config", default=None, help="Optional TOML config path.")
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--xfoil", default=None)
    parser.add_argument("--alpha-start", type=float, default=None)
    parser.add_argument("--alpha-stop", type=float, default=None)
    parser.add_argument("--alpha-step", type=float, default=None)
    parser.add_argument("--re", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--require-oracle", action="store_true")
    parser.add_argument("--enable-cache", action="store_true")
    parser.add_argument(
        "--backend",
        default=None,
        choices=[
            "cpu",
            "cpu-oracle",
            "cpu-oracle-legacy",
            "cpu-native",
            "native-cpu",
            "native",
        ],
        help="Backend to evaluate with (defaults to backend config key, then cpu).",
    )
    parser.add_argument("--out", default="", help="Optional output path for benchmark manifest JSON.")
    args = parser.parse_args()

    config = _load_config(Path(args.config) if args.config else None)
    query_cfg = config.get("query", {})
    benchmark_cfg = config.get("benchmark", {})
    backend_cfg = config.get("backend", {})

    argv_flags = set(sys.argv[1:])
    require_oracle = _first_non_null(
        True if "--require-oracle" in argv_flags else None,
        benchmark_cfg.get("require_oracle"),
        False,
    )
    enable_cache = _first_non_null(
        True if "--enable-cache" in argv_flags else None,
        benchmark_cfg.get("enable_cache"),
        backend_cfg.get("cache_results"),
        backend_cfg.get("enable_cache"),
        False,
    )
    geometry_path = _resolve_path(
        _first_non_null(
            args.geometry,
            config.get("geometry"),
            benchmark_cfg.get("geometry"),
            "data/naca0012.dat",
        ),
        _repo_root,
    )
    backend_name = _first_non_null(
        args.backend,
        backend_cfg.get("backend"),
        backend_cfg.get("backend_name"),
        "cpu",
    )
    backend_key = str(backend_name).lower().replace("_", "-")
    is_native = backend_key in {"cpu-native", "native-cpu", "native"}
    xfoil_candidate = str(
        _first_non_null(
            args.xfoil,
            backend_cfg.get("xfoil_executable"),
            os.environ.get("XFOIL_EXE", "xfoil"),
        )
    )
    if os.path.isabs(xfoil_candidate):
        xfoil_exec = Path(xfoil_candidate)
    elif os.path.sep in xfoil_candidate:
        xfoil_exec = _repo_root / xfoil_candidate
    else:
        xfoil_exec = Path(xfoil_candidate)
    if not geometry_path.exists():
        raise RuntimeError(f"geometry file not found: {geometry_path}")

    alpha_start = float(
        _first_non_null(
            args.alpha_start,
            benchmark_cfg.get("alpha_start"),
            2.0,
        )
    )
    alpha_stop = float(
        _first_non_null(
            args.alpha_stop,
            benchmark_cfg.get("alpha_stop"),
            8.0,
        )
    )
    alpha_step = float(
        _first_non_null(
            args.alpha_step,
            benchmark_cfg.get("alpha_step"),
            2.0,
        )
    )
    re_value = float(
        _first_non_null(
            args.re,
            query_cfg.get("re"),
            query_cfg.get("reynolds"),
            1_000_000.0,
        )
    )
    repeats = int(_first_non_null(args.repeats, benchmark_cfg.get("repeats"), 5))
    trials = int(_first_non_null(args.trials, benchmark_cfg.get("trials"), 3))
    timeout_seconds = float(
        _first_non_null(
            args.timeout,
            backend_cfg.get("timeout"),
            backend_cfg.get("timeout_seconds"),
            20.0,
        )
    )

    geometry = _read_geometry(geometry_path)
    alphas = [
        a
        for a in [alpha_start + idx * alpha_step for idx in range(int((alpha_stop - alpha_start) / alpha_step) + 1)]
    ]

    if is_native:
        engine = XfoilEngine(
            backend="cpu-native",
            config=NativeCpuXfoilConfig(
                cache_results=enable_cache,
                panel_fallback=_first_non_null(backend_cfg.get("panel_fallback"), 80),
                stall_alpha_deg=_first_non_null(backend_cfg.get("stall_alpha_deg"), 18.0),
            ),
        )
    else:
        engine = XfoilEngine(
            config=CpuXfoilConfig(
                xfoil_executable=str(xfoil_exec),
                timeout_seconds=timeout_seconds,
                enable_fallback=not require_oracle,
                cache_results=enable_cache,
            )
        )

    wall_samples: list[float] = []
    per_case_samples: list[float] = []
    statuses: dict[str, int] = {}
    xfoil_probe = _probe_xfoil(str(xfoil_exec))

    for trial in range(trials):
        batch = _build_batch(geometry, alphas, re_value, repeats)
        started = time.perf_counter()
        results = engine.evaluate_many(batch)
        wall = time.perf_counter() - started
        wall_samples.append(wall)

        for r in results:
            if r.elapsed_seconds is not None:
                per_case_samples.append(r.elapsed_seconds)
            statuses[r.status] = statuses.get(r.status, 0) + 1

    report = {
        "benchmark_id": f"cpu-bench-{int(time.time())}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "xfoil_executable": str(xfoil_exec),
            "require_oracle": require_oracle,
            "enable_cache": enable_cache,
            "trials": trials,
            "repeats": repeats,
            "cases_per_trial": len(alphas) * repeats,
            "alphas": alphas,
            "re": re_value,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "wall_seconds": {
            "mean": statistics.fmean(wall_samples) if wall_samples else 0.0,
            "min": min(wall_samples) if wall_samples else 0.0,
            "max": max(wall_samples) if wall_samples else 0.0,
            "p50": _quantile(wall_samples, 0.5),
            "p95": _quantile(wall_samples, 0.95),
            "p99": _quantile(wall_samples, 0.99),
            "throughput_cases_per_sec": (len(alphas) * repeats) / (statistics.fmean(wall_samples) if wall_samples else 1.0),
        },
        "case_seconds": {
            "mean": statistics.fmean(per_case_samples) if per_case_samples else 0.0,
            "p95": _quantile(per_case_samples, 0.95),
            "p99": _quantile(per_case_samples, 0.99),
        },
        "geometry": {
            "path": str(geometry_path),
            "count": len(geometry),
            "sha256": _geometry_signature(geometry),
        },
        "config_source": {
            "config_path": str(Path(args.config) if args.config else ""),
            "query_from_config": bool(query_cfg),
            "benchmark_from_config": bool(benchmark_cfg),
            "backend_from_config": bool(backend_cfg),
        },
    "xfoil": {
            "requested": str(xfoil_exec),
            "resolved": xfoil_probe[0],
            "available": xfoil_probe[1],
            "mode": "native" if is_native else "oracle",
            "backend": backend_key,
        },
        "status_counts": dict(sorted(statuses.items())),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
