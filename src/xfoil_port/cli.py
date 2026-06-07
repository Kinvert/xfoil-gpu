"""CLI for running a single CPU XFOIL evaluation."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tomllib

from .backends.cpu import CpuXfoilConfig
from .backends.native_cpu import NativeCpuXfoilConfig
from .engine import XfoilEngine
from .types import XfoilQuery
from .errors import XfoilError


def _repo_root() -> pathlib.Path:
    # Local source path layout is /<repo>/src/xfoil_port/cli.py.
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_geometry_path(path: pathlib.Path, repo_root: pathlib.Path) -> pathlib.Path:
    p = path
    if p.is_absolute():
        return p
    if (repo_root / p).is_file():
        return repo_root / p
    return p


def _read_geometry(path: pathlib.Path):
    path = _resolve_geometry_path(path, _repo_root())
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pts = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            continue
        pts.append((float(tokens[0]), float(tokens[1])))
    if not pts:
        raise ValueError(f"no points read from {path}")
    return pts


def _load_config(path: pathlib.Path | None) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"config file not found: {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a TOML table: {path}")
    return data


def _first_non_null(*values):
    for value in values:
        if value is not None:
            return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a single XFOIL condition with CPU backend.")
    parser.add_argument("geometry", help="Path to a whitespace-separated X/Y point file (txt/csv style)")
    parser.add_argument("--config", default=None, help="Optional TOML config path")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--re", type=float, default=None)
    parser.add_argument("--mach", type=float, default=None)
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--n", type=int, default=None, help="Optional panel count")
    parser.add_argument("--n-crit", type=int, default=None, dest="n_crit")
    parser.add_argument(
        "--require-oracle",
        action="store_true",
        default=None,
        help="Disable fallback and fail fast when XFOIL is unavailable or errors.",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        default=None,
        help="Enable deterministic fallback when xfoil is unavailable",
    )
    parser.add_argument(
        "--xfoil",
        default=None,
        help="Path to XFOIL executable (default: $XFOIL_EXE or xfoil)",
    )
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
        help="Evaluator backend to use (defaults to backend section in --config, then cpu)",
    )
    parser.add_argument("--work-dir", default=None, help="Optional working directory")
    parser.add_argument("--timeout", type=float, default=None, help="Seconds")
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        default=None,
        help="Disable in-process memoized cache",
    )
    args = parser.parse_args()

    try:
        config = _load_config(pathlib.Path(args.config) if args.config else None)
        query_config = config.get("query", {})
        backend_config = config.get("backend", {})

        alpha = _first_non_null(args.alpha, query_config.get("alpha"))
        reynolds = _first_non_null(args.re, query_config.get("re", query_config.get("reynolds")))
        mach = _first_non_null(args.mach, query_config.get("mach"), 0.0)
        iterations = _first_non_null(args.iter, query_config.get("iterations"), 200)
        n_crit = _first_non_null(args.n_crit, query_config.get("n_crit"), 9)
        n_panels = _first_non_null(args.n, query_config.get("n_panels"))
        backend_name = _first_non_null(
            args.backend,
            backend_config.get("backend"),
            backend_config.get("backend_name"),
            "cpu",
        )
        backend_key = str(backend_name).lower().replace("_", "-")
        require_oracle = _first_non_null(args.require_oracle, config.get("require_oracle"), backend_config.get("require_oracle"), False)
        use_mock = _first_non_null(args.use_mock, backend_config.get("use_mock"), False)
        xfoil_executable = _first_non_null(
            args.xfoil,
            backend_config.get("xfoil_executable"),
            backend_config.get("xfoil"),
            os.environ.get("XFOIL_EXE", "xfoil"),
        )
        work_dir = _first_non_null(
            args.work_dir,
            backend_config.get("work_dir", ""),
            "",
        )
        timeout_seconds = float(
            _first_non_null(
                args.timeout,
                backend_config.get("timeout_seconds"),
                backend_config.get("timeout"),
                20.0,
            )
        )
        disable_cache = bool(_first_non_null(args.disable_cache, backend_config.get("disable_cache"), False))

        if alpha is None:
            raise ValueError("alpha is required when config does not provide it")
        if reynolds is None:
            raise ValueError("re is required when config does not provide it")

        if require_oracle and use_mock:
            raise ValueError("use-mock cannot be combined with require-oracle")
        if backend_key in {"cpu-native", "native-cpu", "native"} and (require_oracle or use_mock):
            raise ValueError(
                "native backend is orthogonal to xfoil fallback flags; "
                "use --backend=cpu-native without --require-oracle/--use-mock"
            )
        points = _read_geometry(pathlib.Path(args.geometry))
        if backend_key in {"cpu-native", "native-cpu", "native"}:
            engine = XfoilEngine(
                backend="cpu-native",
                config=NativeCpuXfoilConfig(
                    cache_results=not disable_cache,
                    panel_fallback=_first_non_null(backend_config.get("panel_fallback"), 80),
                    stall_alpha_deg=_first_non_null(backend_config.get("stall_alpha_deg"), 18.0),
                ),
            )
        else:
            engine = XfoilEngine(
                config=CpuXfoilConfig(
                    xfoil_executable=xfoil_executable,
                    timeout_seconds=timeout_seconds,
                    work_dir=work_dir,
                    enable_fallback=not require_oracle and use_mock,
                    cache_results=not disable_cache,
                )
            )
        
        query = XfoilQuery(
            alpha_deg=alpha,
            reynolds=reynolds,
            mach=mach,
            iterations=iterations,
            n_crit=n_crit,
            n_panels=n_panels,
        )
        result = engine.evaluate(points, query, name="cli")
        payload = {
            "alpha_deg": result.alpha_deg,
            "reynolds": result.reynolds,
            "mach": result.mach,
            "cl": result.cl,
            "cd": result.cd,
            "cm": result.cm,
            "status": result.status,
            "meta": result.meta,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (XfoilError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
