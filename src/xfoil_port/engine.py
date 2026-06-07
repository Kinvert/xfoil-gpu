"""Main engine abstraction for future multi-backend support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .backends.cpu import CpuXfoilConfig, CpuXfoilEvaluator
from .backends.native_cpu import NativeCpuXfoilConfig, NativeCpuXfoilEvaluator
from .errors import XfoilError
from .types import XfoilBackend, XfoilBatchInput, XfoilQuery, XfoilResult


@dataclass
class XfoilEngine:
    """Facade around a CPU backend with a stable output contract.

    Keep the interface stable so a CUDA-accelerated or compiled backend can be
    plugged in later without changing callers.
    """

    backend: str | XfoilBackend = "cpu"
    config: Optional[CpuXfoilConfig | NativeCpuXfoilConfig] = None
    backend_id: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            backend_name = self.backend
            normalized = str(backend_name).lower().replace("_", "-")

            if normalized in {"cpu", "cpu-oracle", "cpu-oracle-legacy"}:
                self._evaluator = CpuXfoilEvaluator(self.config or CpuXfoilConfig())
                self.backend_id = self._evaluator.backend_id
                return

            if normalized in {"cpu-native", "native-cpu", "native"}:
                self._evaluator = NativeCpuXfoilEvaluator(self.config or NativeCpuXfoilConfig())
                self.backend_id = self._evaluator.backend_id
                return

            if backend_name != "cpu":
                raise XfoilError(
                    f"unsupported backend '{backend_name}'; supported: cpu, cpu-oracle, cpu-native, native-cpu"
                )

            self._evaluator = CpuXfoilEvaluator(self.config or CpuXfoilConfig())
            self.backend_id = self._evaluator.backend_id
            return

        if not hasattr(self.backend, "evaluate"):
            raise XfoilError("backend must implement evaluate(geometry_points, query, name=...)")
        self._evaluator = self.backend
        self.backend_id = getattr(self.backend, "backend_id", self.backend.__class__.__name__)

    def _decorate_result_meta(self, result: XfoilResult) -> XfoilResult:
        meta = dict(result.meta)
        meta.setdefault("backend", self.backend_id)
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
            meta=meta,
            elapsed_seconds=result.elapsed_seconds,
            cache_hit=result.cache_hit,
        )

    def evaluate(
        self,
        geometry_points: Sequence[tuple[float, float]],
        query: XfoilQuery,
        *,
        name: str = "airfoil",
        meta: Optional[Dict[str, str]] = None,
    ) -> XfoilResult:
        result = self._evaluator.evaluate(geometry_points, query, name=name).payload
        if meta:
            merged = dict(result.meta)
            merged.update(meta)
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
                elapsed_seconds=result.elapsed_seconds,
                cache_hit=result.cache_hit,
                meta=merged,
            )
        return self._decorate_result_meta(result)

    def evaluate_many(
        self,
        items: Sequence[XfoilBatchInput],
        *,
        meta: Optional[Dict[str, str]] = None,
    ) -> list[XfoilResult]:
        """Evaluate many geometry/query pairs using the same backend contract.

        This returns a deterministic ordered result vector and is intentionally
        CPU-bound for now. The shape is designed to mirror a future batched GPU
        dispatch.
        """

        results: list[XfoilResult] = []
        for item in items:
            item_meta = dict(meta) if meta else None
            if item_meta is not None:
                item_meta = {**item_meta}
            item_result = self.evaluate(
                item.geometry_points,
                item.query,
                name=item.name,
                meta=item_meta,
            )
            results.append(item_result)
        return results
