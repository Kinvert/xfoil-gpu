"""Deterministic native CPU approximation backend for aerodynamic evaluation.

This module provides a fast, pure-Python CPU path that does not invoke the
XFOIL executable. It is intentionally approximate and designed as a stable
reference implementation for early benchmarking and for de-risking later CUDA
rewrites.
"""

from __future__ import annotations

import hashlib
import types
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from ..geometry import ensure_closed, validate_points
from ..types import BackendResult, XfoilQuery, XfoilResult
from .native_cpu_kernels import estimate_aero


@dataclass(frozen=True)
class NativeCpuXfoilConfig:
    """Tunable knobs for the native CPU approximation."""

    cache_results: bool = True
    panel_fallback: int = 80
    stall_alpha_deg: float = 18.0
    residual_floor: float = 1e-10
    iterations_to_converge: int = 120
    use_compiled_kernel: bool = True


class NativeCpuXfoilEvaluator:
    """Evaluate an airfoil + condition set with deterministic CPU math.

    The output format mirrors `CpuXfoilEvaluator` so callers can swap between
    backends without changing post-processing or reward plumbing.
    """

    backend_id = "cpu_native"
    accelerator = "cpu"

    def __init__(self, config: Optional[NativeCpuXfoilConfig] = None):
        self.config = config or NativeCpuXfoilConfig()
        self._cache: Dict[str, XfoilResult] = {}
        self._compiled_kernel: types.ModuleType | None = None
        if self.config.use_compiled_kernel:
            self._compiled_kernel = self._resolve_compiled_kernel()

    @staticmethod
    def _resolve_compiled_kernel() -> types.ModuleType | None:
        """Return compiled accelerator module when available.

        This lookup is best-effort only. If a compiled extension is unavailable,
        the pure Python kernel is used. Keeping import side-effects local avoids
        hard dependency on a compiler/toolchain.
        """

        try:
            from . import native_cpu_cpp  # type: ignore[import-not-found]
            if getattr(native_cpu_cpp, "HAS_COMPILED", False) and callable(
                getattr(native_cpu_cpp, "estimate_aero", None)
            ):
                return native_cpu_cpp
        except Exception:
            return None
        return None

    def _cache_key(self, geometry_points: Sequence[Tuple[float, float]], query: XfoilQuery) -> str:
        payload = {
            "points": [tuple(map(float, p)) for p in geometry_points],
            "alpha": query.alpha_deg,
            "re": query.reynolds,
            "mach": query.mach,
            "n_crit": query.n_crit,
            "iter": query.iterations,
            "n_panels": query.n_panels,
        }
        return hashlib.sha256(
            repr((payload["points"], payload["alpha"], payload["re"], payload["mach"], payload["n_crit"], payload["iter"], payload["n_panels"])).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _query_meta(query: XfoilQuery) -> dict[str, str]:
        return {
            "alpha_deg": f"{query.alpha_deg:.12g}",
            "reynolds": f"{query.reynolds:.12g}",
            "mach": f"{query.mach:.12g}",
            "iterations": str(query.iterations),
            "n_crit": str(query.n_crit),
            "n_panels": str(query.n_panels),
        }

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

    @staticmethod
    def _with_meta(result: XfoilResult, status: str, warnings: Optional[list[str]] = None) -> XfoilResult:
        meta = dict(result.meta)
        meta.setdefault("source", "native_cpu_approx")
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
            warnings=list(dict.fromkeys(warnings or result.warnings)),
            meta=meta,
            elapsed_seconds=result.elapsed_seconds,
            cache_hit=result.cache_hit,
        )

    @staticmethod
    def _geometry_signature(geometry_points: Sequence[Tuple[float, float]]) -> str:
        payload = "|".join(f"{x:.12g},{y:.12g}" for x, y in geometry_points)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _solve_proxy(self, geometry_points: Sequence[Tuple[float, float]], query: XfoilQuery) -> XfoilResult:
        if self._compiled_kernel is not None and hasattr(self._compiled_kernel, "estimate_aero"):
            requested_panels = query.n_panels if query.n_panels is not None else self.config.panel_fallback
            kernel_out = self._compiled_kernel.estimate_aero(
                geometry_points,
                float(query.alpha_deg),
                float(query.reynolds),
                float(query.mach),
                requested_panels,
                max(1, int(query.iterations)),
                self.config.stall_alpha_deg,
                self.config.residual_floor,
                self.config.iterations_to_converge,
            )
            features = kernel_out.get("features") if isinstance(kernel_out, dict) else None
            features = features if isinstance(features, dict) else {}
            return XfoilResult(
                alpha_deg=float(query.alpha_deg),
                reynolds=float(query.reynolds),
                mach=float(query.mach),
                cl=float(kernel_out["cl"]),
                cd=float(kernel_out["cd"]),
                cm=float(kernel_out["cm"]),
                status=str(kernel_out.get("status", "ok")),
                residual=float(kernel_out["residual"]),
                iterations_used=int(kernel_out["iterations_used"]),
                iterations_failed=bool(kernel_out["iterations_failed"]),
                warnings=[str(item) for item in kernel_out.get("warnings", [])],
                meta={
                    "source": "native_cpu_approx_cpp",
                    "geometry_points": str(len(geometry_points)),
                    "geometry_sha": self._geometry_signature(geometry_points),
                    "panel_count": str(int(float(features.get("n", len(geometry_points))) - 1)),
                    "curvature": f'{float(features.get("curvature", float("nan"))):.12g}',
                    "camber": f'{float(features.get("camber", float("nan"))):.12g}',
                    "thickness_ratio": f'{float(features.get("thickness_ratio", float("nan"))):.12g}',
                },
            )

        result = estimate_aero(
            geometry_points=geometry_points,
            query_alpha_deg=float(query.alpha_deg),
            query_reynolds=float(query.reynolds),
            query_mach=float(query.mach),
            n_panels=query.n_panels if query.n_panels is not None else self.config.panel_fallback,
            iterations_requested=max(1, int(query.iterations)),
            stall_alpha_deg=self.config.stall_alpha_deg,
            residual_floor=self.config.residual_floor,
            iterations_to_converge=self.config.iterations_to_converge,
        )
        features = result["features"]
        return XfoilResult(
            alpha_deg=float(query.alpha_deg),
            reynolds=float(query.reynolds),
            mach=float(query.mach),
            cl=float(result["cl"]),
            cd=float(result["cd"]),
            cm=float(result["cm"]),
            status=str(result["status"]),
            residual=float(result["residual"]),
            iterations_used=int(result["iterations_used"]),
            iterations_failed=bool(result["iterations_failed"]),
            warnings=[str(item) for item in result["warnings"]],
            meta={
                "source": "native_cpu_approx",
                "geometry_points": str(len(geometry_points)),
                "geometry_sha": self._geometry_signature(geometry_points),
                "panel_count": str(int(features["n"] - 1)),
                "curvature": f"{features['curvature']:.12g}",
                "camber": f"{features['camber']:.12g}",
                "thickness_ratio": f"{features['thickness_ratio']:.12g}",
            },
        )

    def evaluate(
        self,
        geometry_points: Sequence[Tuple[float, float]],
        query: XfoilQuery,
        *,
        name: str = "airfoil",
    ) -> BackendResult:
        del name
        geometry_points = ensure_closed(geometry_points)
        validate_points(geometry_points)

        cache_key = self._cache_key(geometry_points, query)
        if self.config.cache_results and cache_key in self._cache:
            cached = self._cache[cache_key]
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
                meta={**cached.meta, "cache_hit": "true"},
                elapsed_seconds=0.0,
                cache_hit=True,
            )
            cached_result = self._with_timing(
                XfoilResult(
                    alpha_deg=cached_result.alpha_deg,
                    reynolds=cached_result.reynolds,
                    mach=cached_result.mach,
                    cl=cached_result.cl,
                    cd=cached_result.cd,
                    cm=cached_result.cm,
                    status=cached_result.status,
                    residual=cached_result.residual,
                    iterations_used=cached_result.iterations_used,
                    iterations_failed=cached_result.iterations_failed,
                    warnings=cached_result.warnings,
                    meta=cached_result.meta,
                    elapsed_seconds=0.0,
                    cache_hit=True,
                ),
                0.0,
            )
            return BackendResult(
                ok=True,
                payload=cached_result,
                raw_stdout="",
                raw_stderr="",
            )

        started_at = time.perf_counter()
        result = self._solve_proxy(geometry_points, query)
        result = self._with_meta(
            result,
            status=result.status,
            warnings=result.warnings,
        )
        result = XfoilResult(
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
            meta={**result.meta, "query": self._query_meta(query)},
            elapsed_seconds=result.elapsed_seconds,
            cache_hit=result.cache_hit,
        )
        result = self._with_timing(result, time.perf_counter() - started_at)

        if self.config.cache_results:
            self._cache[cache_key] = result
        return BackendResult(ok=True, payload=result, raw_stdout="", raw_stderr="")
