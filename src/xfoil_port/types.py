"""Core types for the CPU-first XFOIL wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class XfoilQuery:
    """Single evaluation request."""

    alpha_deg: float
    reynolds: float
    mach: float = 0.0
    n_crit: int = 9
    iterations: int = 200
    n_panels: Optional[int] = None


@dataclass(frozen=True)
class Geometry:
    """Airfoil geometry in counterclockwise order."""

    name: str
    points: Sequence[tuple[float, float]]


@dataclass(frozen=True)
class XfoilResult:
    """Result from one XFOIL call."""

    alpha_deg: float
    reynolds: float
    mach: float
    cl: Optional[float]
    cd: Optional[float]
    cm: Optional[float]
    status: str = "ok"
    residual: Optional[float] = None
    iterations_used: Optional[int] = None
    iterations_failed: bool = False
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)
    elapsed_seconds: Optional[float] = None
    cache_hit: bool = False


@dataclass(frozen=True)
class BackendResult:
    """Raw backend result before any adapter-level normalization."""

    ok: bool
    payload: XfoilResult
    raw_stdout: str = ""
    raw_stderr: str = ""


@runtime_checkable
class XfoilBackend(Protocol):
    """Minimal backend contract for CPU or future GPU implementations."""

    backend_id: str = "custom"

    def evaluate(
        self,
        geometry_points: Sequence[tuple[float, float]],
        query: XfoilQuery,
        *,
        name: str = "airfoil",
    ) -> BackendResult:
        ...


@dataclass(frozen=True)
class XfoilBatchInput:
    """One batched XFOIL request."""

    name: str
    geometry_points: Sequence[tuple[float, float]]
    query: XfoilQuery
