"""CPU backend using a local XFOIL executable."""

from __future__ import annotations

import hashlib
import json
import math
import re
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..errors import XfoilDependencyError, XfoilRuntimeError
from ..geometry import ensure_closed, format_points, validate_points
from ..types import BackendResult, XfoilQuery, XfoilResult


@dataclass(frozen=True)
class CpuXfoilConfig:
    xfoil_executable: str = "xfoil"
    timeout_seconds: float = 20.0
    work_dir: str = ""
    close_geometry_if_needed: bool = True
    enable_fallback: bool = False
    cache_results: bool = True


class CpuXfoilEvaluator:
    """Evaluate airfoil conditions using XFOIL on CPU.

    Keep this backend deterministic for baseline behavior. A deterministic fallback
    path is available so the API is still runnable if XFOIL is unavailable.
    """
    backend_id: str = "cpu"
    accelerator: str = "cpu"

    def __init__(self, config: Optional[CpuXfoilConfig] = None):
        self.config = config or CpuXfoilConfig()
        self._cache: Dict[str, XfoilResult] = {}

    @staticmethod
    def _resolve_executable(path_or_name: str, work_dir: str = "") -> str:
        """Resolve XFOIL executable path deterministically.

        Resolution order:
          1) absolute path as provided
          2) work dir relative path (when configured)
          3) repository-root relative path (for scripts bundled with project)
          4) PATH lookup
        """
        if not path_or_name:
            return ""

        candidate = Path(path_or_name)
        if candidate.is_absolute():
            return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else ""

        if work_dir:
            candidate_in_work_dir = Path(work_dir) / candidate
            if candidate_in_work_dir.is_file() and os.access(candidate_in_work_dir, os.X_OK):
                return str(candidate_in_work_dir)

        package_root = Path(__file__).resolve().parents[3]
        if (package_root / candidate).is_file() and os.access(package_root / candidate, os.X_OK):
            return str(package_root / candidate)

        resolved = shutil.which(path_or_name)
        return resolved or ""

    @staticmethod
    def _has_executable(path_or_name: str, work_dir: str = "") -> bool:
        return bool(CpuXfoilEvaluator._resolve_executable(path_or_name, work_dir=work_dir))

    def _cache_key(self, geometry_points: Sequence[Tuple[float, float]], query: XfoilQuery, name: str) -> str:
        payload = {
            "name": name,
            "points": [tuple(map(float, p)) for p in geometry_points],
            "alpha": query.alpha_deg,
            "re": query.reynolds,
            "mach": query.mach,
            "ncrit": query.n_crit,
            "iter": query.iterations,
            "n_panels": query.n_panels,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def _fallback_from_geometry(self, geometry_points: Sequence[Tuple[float, float]], query: XfoilQuery) -> XfoilResult:
        # Minimal deterministic proxy for development and smoke testing without XFOIL.
        alpha = math.radians(query.alpha_deg)
        # simple geometric proxy: average camber magnitude
        ys = [p[1] for p in geometry_points]
        camber = max(abs(y) for y in ys) if ys else 0.0
        thickness = (max(ys) - min(ys)) if ys else 0.0

        # Thin-airfoil style baseline + weak geometric modifiers.
        cl = 2.0 * math.pi * alpha * (1.0 + 0.25 * camber)
        cl += 0.05 * camber * math.copysign(1.0, alpha) if alpha != 0 else 0.0
        cd = 0.004 + 0.0025 * thickness + 0.000001 * query.reynolds**0.25 / 2000.0
        cm = -0.02 * (1.0 - query.mach)
        if query.mach > 0:
            cd *= 1.0 + 0.5 * query.mach
            cl *= 1.0 - 0.15 * query.mach
        if query.n_crit:
            cl *= 1.0 - max(0.0, (query.n_crit - 9) * 0.001)

        return XfoilResult(
            alpha_deg=query.alpha_deg,
            reynolds=query.reynolds,
            mach=query.mach,
            cl=float(cl),
            cd=float(cd),
            cm=float(cm),
            status="fallback",
            warnings=["xfoil_unavailable_fallback"],
            meta={
                "source": "fallback_heuristic",
            },
        )

    @staticmethod
    def _with_meta(result: XfoilResult, status: str, reason: str) -> XfoilResult:
        return XfoilResult(
            alpha_deg=result.alpha_deg,
            reynolds=result.reynolds,
            mach=result.mach,
            cl=result.cl,
            cd=result.cd,
            cm=result.cm,
            status=status,
            residual=result.residual,
            iterations_used=result.iterations_used,
            iterations_failed=result.iterations_failed,
            warnings=result.warnings,
            meta={**result.meta, "fallback_reason": reason},
            elapsed_seconds=result.elapsed_seconds,
            cache_hit=result.cache_hit,
        )

    @staticmethod
    def _with_query_meta(
        result: XfoilResult,
        query: XfoilQuery,
    ) -> XfoilResult:
        query_meta = {
            "alpha_deg": f"{query.alpha_deg:.12g}",
            "reynolds": f"{query.reynolds:.12g}",
            "mach": f"{query.mach:.12g}",
            "iterations": str(query.iterations),
            "n_crit": str(query.n_crit),
            "n_panels": str(query.n_panels),
        }
        merged = dict(result.meta)
        merged.update({"query": query_meta})
        return XfoilResult(
            alpha_deg=result.alpha_deg,
            reynolds=result.reynolds,
            mach=result.mach,
            cl=result.cl,
            cd=result.cd,
            cm=result.cm,
            status=result.status,
            residual=result.residual,
            iterations_used=result.iterations_used,
            iterations_failed=result.iterations_failed,
            warnings=result.warnings,
            meta=merged,
            elapsed_seconds=result.elapsed_seconds,
            cache_hit=result.cache_hit,
        )

    @staticmethod
    def _with_timing(result: XfoilResult, elapsed_seconds: float) -> XfoilResult:
        return XfoilResult(
            alpha_deg=result.alpha_deg,
            reynolds=result.reynolds,
            mach=result.mach,
            cl=result.cl,
            cd=result.cd,
            cm=result.cm,
            status=result.status,
            residual=result.residual,
            iterations_used=result.iterations_used,
            iterations_failed=result.iterations_failed,
            warnings=result.warnings,
            meta={
                **result.meta,
                "elapsed_seconds": f"{elapsed_seconds:.9f}",
            },
            elapsed_seconds=elapsed_seconds,
            cache_hit=result.cache_hit,
        )

    def evaluate(
        self,
        geometry_points: Sequence[Tuple[float, float]],
        query: XfoilQuery,
        *,
        name: str = "airfoil",
    ) -> BackendResult:
        if self.config.close_geometry_if_needed:
            geometry_points = ensure_closed(geometry_points)
        validate_points(geometry_points)

        cache_key = self._cache_key(geometry_points, query, name)
        if self.config.cache_results and cache_key in self._cache:
            cached = self._with_timing(
                self._cache[cache_key],
                elapsed_seconds=0.0,
            )
            cached_result = XfoilResult(
                alpha_deg=cached.alpha_deg,
                reynolds=cached.reynolds,
                mach=cached.mach,
                cl=cached.cl,
                cd=cached.cd,
                cm=cached.cm,
                status=cached.status,
                residual=cached.residual,
                iterations_used=cached.iterations_used,
                iterations_failed=cached.iterations_failed,
                warnings=cached.warnings,
                meta={
                    **cached.meta,
                    "cache_hit": "true",
                },
                elapsed_seconds=0.0,
                cache_hit=True,
            )
            return BackendResult(
                ok=True,
                payload=cached_result,
                raw_stdout="",
                raw_stderr="",
            )

        if self.config.enable_fallback and not self._has_executable(
            self.config.xfoil_executable,
            work_dir=self.config.work_dir,
        ):
            started_at = time.perf_counter()
            result = self._with_meta(
                self._fallback_from_geometry(geometry_points, query),
                "fallback_missing_executable",
                f"missing_xfoil:{self.config.xfoil_executable}",
            )
            result = self._with_query_meta(result, query)
            result = self._with_timing(result, time.perf_counter() - started_at)
            if self.config.cache_results:
                self._cache[cache_key] = result
            return BackendResult(ok=True, payload=result, raw_stdout="", raw_stderr="")

        if not self._has_executable(
            self.config.xfoil_executable,
            work_dir=self.config.work_dir,
        ):
            raise XfoilDependencyError(
                f"XFOIL executable not found: {self.config.xfoil_executable}"
            )

        started_at = time.perf_counter()
        xfoil_executable = self._resolve_executable(
            self.config.xfoil_executable,
            work_dir=self.config.work_dir,
        )
        if not xfoil_executable:
            raise XfoilDependencyError(
                f"XFOIL executable resolution failed unexpectedly: {self.config.xfoil_executable}"
            )
        out_dir = Path(self.config.work_dir or tempfile.mkdtemp(prefix="xfoil_cpu_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        geometry_file = out_dir / f"{sanitize_name(name)}.dat"
        geometry_file.write_text(format_points(geometry_points), encoding="utf-8")

        polar_file = out_dir / f"{sanitize_name(name)}.polar"
        cmd_path = out_dir / f"{sanitize_name(name)}.cmd"
        cmd_script = _build_xfoil_script(geometry_file, polar_file, query)
        cmd_path.write_text(cmd_script, encoding="utf-8")

        try:
            proc = subprocess.run(
                [xfoil_executable],
                input=cmd_script,
                cwd=str(out_dir),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            if self.config.enable_fallback:
                result = self._with_meta(
                    self._fallback_from_geometry(geometry_points, query),
                    "fallback_timeout",
                    f"timeout_{self.config.timeout_seconds:.2f}s",
                )
                result = self._with_query_meta(result, query)
                result = self._with_timing(result, time.perf_counter() - started_at)
                if self.config.cache_results:
                    self._cache[cache_key] = result
                return BackendResult(ok=True, payload=result, raw_stdout="", raw_stderr="")
            raise XfoilRuntimeError(
                f"xfoil timed out after {self.config.timeout_seconds:.2f}s for {name}"
            ) from exc
        except FileNotFoundError as exc:
            if self.config.enable_fallback:
                result = self._with_meta(
                    self._fallback_from_geometry(geometry_points, query),
                    "fallback_missing_executable",
                    self.config.xfoil_executable,
                )
                result = self._with_query_meta(result, query)
                result = self._with_timing(result, time.perf_counter() - started_at)
                if self.config.cache_results:
                    self._cache[cache_key] = result
                return BackendResult(ok=True, payload=result, raw_stdout="", raw_stderr="")
            raise XfoilDependencyError(
                f"Failed to execute {self.config.xfoil_executable}"
            ) from exc

        if proc.returncode != 0:
            if self.config.enable_fallback:
                result = self._with_meta(
                    self._fallback_from_geometry(geometry_points, query),
                    "fallback_nonzero_exit",
                    f"xfoil_exit_{proc.returncode}",
                )
                result = self._with_query_meta(result, query)
                result = self._with_timing(result, time.perf_counter() - started_at)
                if self.config.cache_results:
                    self._cache[cache_key] = result
                return BackendResult(ok=True, payload=result, raw_stdout="", raw_stderr="")
            raise XfoilRuntimeError(
                f"xfoil exited with status {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )

        result = _parse_polar_file(
            polar_file,
            query.alpha_deg,
            reynolds=query.reynolds,
            mach=query.mach,
        )
        if result is None:
            result = _regex_parse_stdout(proc.stdout)

        if result is None:
            result = _regex_parse_stdout(proc.stderr)

        if result is None:
            if self.config.enable_fallback:
                result = self._with_meta(
                    self._fallback_from_geometry(geometry_points, query),
                    "fallback_parse",
                    "no_parseable_output",
                )
                result = self._with_query_meta(result, query)
                result = self._with_timing(result, time.perf_counter() - started_at)
                if self.config.cache_results:
                    self._cache[cache_key] = result
                return BackendResult(ok=True, payload=result, raw_stdout=proc.stdout, raw_stderr=proc.stderr)
            raise XfoilRuntimeError(
                "Could not parse XFOIL output. Make sure geometry and settings are supported."
            )

        result = XfoilResult(
            alpha_deg=query.alpha_deg,
            reynolds=query.reynolds,
            mach=query.mach,
            cl=result.cl,
            cd=result.cd,
            cm=result.cm,
            status=result.status,
            residual=result.residual,
            iterations_used=result.iterations_used,
            iterations_failed=result.iterations_failed,
            warnings=result.warnings,
            meta={
                **result.meta,
                "source": result.meta.get("source", "unknown"),
                "command_file": str(cmd_path),
                "polar_file": str(polar_file),
            },
        )
        result = self._with_query_meta(result, query)

        if self.config.cache_results:
            self._cache[cache_key] = result
        return_result = self._with_timing(result, time.perf_counter() - started_at)
        if self.config.cache_results:
            self._cache[cache_key] = return_result
        return BackendResult(ok=True, payload=return_result, raw_stdout=proc.stdout, raw_stderr=proc.stderr)


def _parse_polar_file(
    path: Path,
    target_alpha: float,
    *,
    reynolds: float,
    mach: float,
) -> Optional[XfoilResult]:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header: Optional[List[str]] = None
    rows: List[Dict[str, float]] = []

    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if not tokens:
            continue

        upper = line.upper()
        if (
            "CL" in upper
            and ("CD" in upper or "CDP" in upper)
            and "CM" in upper
            and not any(_is_float(v) for v in tokens[:3])
        ):
            header = [t.strip().upper() for t in tokens]
            continue

        if header is None:
            if len(tokens) >= 4 and _is_float(tokens[0]) and _is_float(tokens[1]):
                try:
                    row = {
                        "ALPHA": float(tokens[0]),
                        "CL": float(tokens[1]),
                        "CD": float(tokens[2]),
                        "CM": float(tokens[4]) if len(tokens) > 4 else float(tokens[3]),
                    }
                    rows.append(row)
                except ValueError:
                    continue
            continue

        if not _is_float(tokens[0]):
            continue

        vals: Dict[str, float] = {}
        for idx, name in enumerate(header):
            if idx >= len(tokens):
                break
            if _is_float(tokens[idx]):
                vals[name] = float(tokens[idx])

        if "CM" not in vals and "CMN" in vals:
            vals["CM"] = vals["CMN"]

        if "CM" not in vals and ("CMQ" in vals or "CMP" in vals):
            vals["CM"] = vals.get("CMQ", vals.get("CMP"))

        if "ALPHA" not in vals or "CL" not in vals or "CD" not in vals or "CM" not in vals:
            continue

        if len(vals) >= 4:
            rows.append(vals)

    if not rows:
        return None

    best = min(rows, key=lambda r: abs(r.get("ALPHA", 0.0) - target_alpha))
    return XfoilResult(
        alpha_deg=float(best.get("ALPHA", target_alpha)),
        reynolds=reynolds,
        mach=mach,
        cl=best.get("CL"),
        cd=best.get("CD"),
        cm=best.get("CM"),
        status="ok",
        warnings=[],
        meta={
            "source": "polar_file",
            "rows_scanned": str(len(rows)),
        },
    )


def _regex_parse_stdout(stdout: str) -> Optional[XfoilResult]:
    for line in stdout.splitlines():
        if "CL" not in line.upper():
            continue
        cl_matches = re.findall(
            r"(?i)CL\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            line,
        )
        cd_matches = re.findall(
            r"(?i)CD\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            line,
        )
        cm_matches = re.findall(
            r"(?i)CM\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            line,
        )
        if cl_matches and cd_matches and cm_matches:
            try:
                cl = float(cl_matches[0])
                cd = float(cd_matches[0])
                cm = float(cm_matches[0])
                return XfoilResult(
                    alpha_deg=0.0,
                    reynolds=0.0,
                    mach=0.0,
                    cl=cl,
                    cd=cd,
                    cm=cm,
                    status="ok",
                    warnings=["stdout_regex"],
                    meta={"source": "stdout_regex"},
                )
            except ValueError:
                continue
    return None


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:120]
    return safe or "airfoil"


def _build_xfoil_script(geometry_file: Path, polar_file: Path, query: XfoilQuery) -> str:
    panel_cmd = ""
    if query.n_panels is not None:
        panel_cmd = f"PPAR\nN {int(query.n_panels)}\n"

    ncrit_cmd = ""
    if query.n_crit:
        ncrit_cmd = f"VPAR\nN {int(query.n_crit)}\n"

    return (
        "PLOP\n"
        "G\n"
        f"LOAD {geometry_file.name}\n"
        f"{panel_cmd}"
        "PANE\n"
        "OPER\n"
        f"VISC {query.reynolds:.6g}\n"
        f"{ncrit_cmd}"
        f"MACH {query.mach:.6g}\n"
        f"ITER {query.iterations}\n"
        "PACC\n"
        f"{polar_file.name}\n"
        "\n"
        f"ALFA {query.alpha_deg:.6f}\n"
        "PACC\n"
        "QUIT\n"
        "\n"
    )
