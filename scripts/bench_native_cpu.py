#!/usr/bin/env python3
"""Benchmark the native CPU kernel variants (Python-only vs compiled C++ when available)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from typing import Any

import tomllib

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from xfoil_port import XfoilBatchInput, XfoilEngine, XfoilQuery
from xfoil_port.backends.native_cpu import NativeCpuXfoilConfig
from xfoil_port.backends import native_cpu_cpp



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


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() and not ("/" in value or "\\" in value):
        candidate = _repo_root / path
        return candidate if candidate.exists() else path
    return path if path.is_absolute() else _repo_root / path


def _build_batch(geometry: list[tuple[float, float]], alphas: list[float], reynolds: float, repeats: int) -> list[XfoilBatchInput]:
    items: list[XfoilBatchInput] = []
    for idx in range(repeats):
        for alpha_idx, alpha in enumerate(alphas):
            items.append(
                XfoilBatchInput(
                    name=f"case_{idx:03d}_{alpha_idx:03d}",
                    geometry_points=geometry,
                    query=XfoilQuery(alpha_deg=alpha, reynolds=reynolds, mach=0.0),
                )
            )
    return items


def _cpp_available() -> bool:
    return bool(getattr(native_cpu_cpp, "HAS_COMPILED", False))


def _build_engine(use_cpp: bool, cache_results: bool, panel_fallback: int, stall_alpha_deg: float, iterations_to_converge: int) -> XfoilEngine:
    return XfoilEngine(
        backend="cpu-native",
        config=NativeCpuXfoilConfig(
            cache_results=cache_results,
            panel_fallback=panel_fallback,
            stall_alpha_deg=stall_alpha_deg,
            iterations_to_converge=iterations_to_converge,
            use_compiled_kernel=use_cpp,
        ),
    )


def _run_mode(
    *,
    mode_name: str,
    geometry: list[tuple[float, float]],
    alphas: list[float],
    reynolds: float,
    repeats: int,
    trials: int,
    cache_results: bool,
    panel_fallback: int,
    stall_alpha_deg: float,
    iterations_to_converge: int,
    use_cpp: bool,
) -> dict[str, Any]:
    if use_cpp and not _cpp_available():
        raise RuntimeError("compiled native backend not available")

    engine = _build_engine(
        use_cpp=use_cpp,
        cache_results=cache_results,
        panel_fallback=panel_fallback,
        stall_alpha_deg=stall_alpha_deg,
        iterations_to_converge=iterations_to_converge,
    )

    wall_samples: list[float] = []
    per_case_samples: list[float] = []
    statuses: dict[str, int] = {}

    for _ in range(trials):
        batch = _build_batch(geometry, alphas, reynolds, repeats)
        started = time.perf_counter()
        results = engine.evaluate_many(batch)
        wall_samples.append(time.perf_counter() - started)

        for result in results:
            if result.elapsed_seconds is not None:
                per_case_samples.append(result.elapsed_seconds)
            statuses[result.status] = statuses.get(result.status, 0) + 1

    cases_per_trial = len(alphas) * repeats
    mean_wall = statistics.fmean(wall_samples) if wall_samples else 0.0

    source = "python" if not use_cpp else "compiled"
    return {
        "mode": mode_name,
        "backend_source": source,
        "kernel_available": _cpp_available() if use_cpp else True,
        "trials": trials,
        "repeats": repeats,
        "cases_per_trial": cases_per_trial,
        "wall_seconds": {
            "mean": mean_wall,
            "min": min(wall_samples) if wall_samples else 0.0,
            "max": max(wall_samples) if wall_samples else 0.0,
            "p50": _quantile(wall_samples, 0.5),
            "p95": _quantile(wall_samples, 0.95),
            "p99": _quantile(wall_samples, 0.99),
            "throughput_cases_per_sec": cases_per_trial / mean_wall if mean_wall else 0.0,
        },
        "case_seconds": {
            "mean": statistics.fmean(per_case_samples) if per_case_samples else 0.0,
            "p95": _quantile(per_case_samples, 0.95),
            "p99": _quantile(per_case_samples, 0.99),
        },
        "status_counts": dict(sorted(statuses.items())),
    }


def _load_toml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise RuntimeError(f"config file not found: {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"config must be a TOML table: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark native CPU approximations.")
    parser.add_argument("--config", default=None, help="Optional TOML config path.")
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--backend-mode", default="auto", choices=["auto", "python", "compiled"], help="Run python fallback, compiled, or auto mode.")
    parser.add_argument("--compare", action="store_true", help="Run both python and compiled modes and report both in one manifest.")
    parser.add_argument("--alpha-start", type=float, default=None)
    parser.add_argument("--alpha-stop", type=float, default=None)
    parser.add_argument("--alpha-step", type=float, default=None)
    parser.add_argument("--re", type=float, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--enable-cache", action="store_true")
    parser.add_argument("--panel-fallback", type=int, default=None)
    parser.add_argument("--stall-alpha", type=float, default=None)
    parser.add_argument("--iterations-to-converge", type=int, default=None)
    parser.add_argument("--out", default="", help="Optional output manifest path.")
    args = parser.parse_args()

    cfg = _load_toml_config(Path(args.config) if args.config else None)
    query_cfg = cfg.get("query", {})
    backend_cfg = cfg.get("backend", {})
    bench_cfg = cfg.get("benchmark", {})

    geometry_path = _resolve_path(
        str(
            args.geometry
            or cfg.get("geometry")
            or bench_cfg.get("geometry")
            or "data/naca0012.dat"
        )
    )
    if not geometry_path.exists():
        raise RuntimeError(f"geometry file not found: {geometry_path}")

    geometry = _read_geometry(geometry_path)

    alpha_start = args.alpha_start if args.alpha_start is not None else bench_cfg.get("alpha_start", query_cfg.get("alpha_start", 2.0))
    alpha_stop = args.alpha_stop if args.alpha_stop is not None else bench_cfg.get("alpha_stop", query_cfg.get("alpha_stop", 8.0))
    alpha_step = args.alpha_step if args.alpha_step is not None else bench_cfg.get("alpha_step", query_cfg.get("alpha_step", 2.0))
    reynolds = args.re if args.re is not None else bench_cfg.get("re", query_cfg.get("re", query_cfg.get("reynolds", 1_000_000.0)))
    repeats = int(args.repeats if args.repeats is not None else bench_cfg.get("repeats", backend_cfg.get("repeats", 1)))
    trials = int(args.trials if args.trials is not None else bench_cfg.get("trials", backend_cfg.get("trials", 3)))
    panel_fallback = int(args.panel_fallback if args.panel_fallback is not None else backend_cfg.get("panel_fallback", 80))
    stall_alpha_deg = float(args.stall_alpha if args.stall_alpha is not None else backend_cfg.get("stall_alpha_deg", 18.0))
    iterations_to_converge = int(args.iterations_to_converge if args.iterations_to_converge is not None else backend_cfg.get("iterations_to_converge", 120))

    alphas = [
        a
        for a in [alpha_start + idx * alpha_step for idx in range(int((alpha_stop - alpha_start) / alpha_step) + 1)]
    ]

    report: dict[str, Any] = {
        "benchmark_id": f"native-cpu-bench-{int(time.time())}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": {
            "path": str(geometry_path),
            "count": len(geometry),
        },
        "config": {
            "backend_mode": args.backend_mode,
            "compare": bool(args.compare),
            "trials": trials,
            "repeats": repeats,
            "alphas": alphas,
            "re": float(reynolds),
            "panel_fallback": panel_fallback,
            "stall_alpha_deg": stall_alpha_deg,
            "iterations_to_converge": iterations_to_converge,
            "cache_results": bool(args.enable_cache),
            "compiled_available": _cpp_available(),
        },
        "modes": {},
    }

    if args.compare:
        report["modes"]["python"] = _run_mode(
            mode_name="python",
            geometry=geometry,
            alphas=alphas,
            reynolds=float(reynolds),
            repeats=repeats,
            trials=trials,
            cache_results=bool(args.enable_cache),
            panel_fallback=panel_fallback,
            stall_alpha_deg=stall_alpha_deg,
            iterations_to_converge=iterations_to_converge,
            use_cpp=False,
        )
        if _cpp_available():
            report["modes"]["compiled"] = _run_mode(
                mode_name="compiled",
                geometry=geometry,
                alphas=alphas,
                reynolds=float(reynolds),
                repeats=repeats,
                trials=trials,
                cache_results=bool(args.enable_cache),
                panel_fallback=panel_fallback,
                stall_alpha_deg=stall_alpha_deg,
                iterations_to_converge=iterations_to_converge,
                use_cpp=True,
            )
        else:
            report["modes"]["compiled"] = {
                "mode": "compiled",
                "status": "unavailable",
                "backend_source": "compiled",
                "kernel_available": False,
                "message": "compiled native extension not built",
            }
    else:
        mode = args.backend_mode
        if mode == "auto":
            use_cpp = _cpp_available()
            report["config"]["backend_mode"] = "cpp" if use_cpp else "python"
            report["selected_mode"] = "compiled" if use_cpp else "python"
            report["modes"]["active"] = _run_mode(
                mode_name="auto",
                geometry=geometry,
                alphas=alphas,
                reynolds=float(reynolds),
                repeats=repeats,
                trials=trials,
                cache_results=bool(args.enable_cache),
                panel_fallback=panel_fallback,
                stall_alpha_deg=stall_alpha_deg,
                iterations_to_converge=iterations_to_converge,
                use_cpp=use_cpp,
            )
        elif mode == "python":
            report["selected_mode"] = "python"
            report["modes"]["active"] = _run_mode(
                mode_name="python",
                geometry=geometry,
                alphas=alphas,
                reynolds=float(reynolds),
                repeats=repeats,
                trials=trials,
                cache_results=bool(args.enable_cache),
                panel_fallback=panel_fallback,
                stall_alpha_deg=stall_alpha_deg,
                iterations_to_converge=iterations_to_converge,
                use_cpp=False,
            )
        elif mode == "compiled":
            if not _cpp_available():
                raise RuntimeError("compiled native backend requested but native_cpu_cpp is not available")
            report["selected_mode"] = "compiled"
            report["modes"]["active"] = _run_mode(
                mode_name="compiled",
                geometry=geometry,
                alphas=alphas,
                reynolds=float(reynolds),
                repeats=repeats,
                trials=trials,
                cache_results=bool(args.enable_cache),
                panel_fallback=panel_fallback,
                stall_alpha_deg=stall_alpha_deg,
                iterations_to_converge=iterations_to_converge,
                use_cpp=True,
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
