"""Optional C++ accelerator bridge for the native CPU kernel path.

The C++ extension is intentionally optional. When built successfully, this module
exposes a compiled ``estimate_aero`` implementation through ``_native_cpu_cpp``.
When compilation is unavailable, this module falls back to the deterministic
Python kernel implementation.
"""

from __future__ import annotations

from typing import TypedDict

from .native_cpu_kernels import KernelGeometryFeatures, KernelOutput, estimate_aero as _python_estimate_aero


class CppKernelGeometryFeatures(TypedDict):
    """Runtime schema parity for compiled outputs."""

    n: float
    camber: float
    thickness: float
    curvature: float
    thickness_ratio: float
    chord: float


class CppKernelOutput(TypedDict):
    """Runtime schema parity for compiled outputs."""

    cl: float
    cd: float
    cm: float
    status: str
    residual: float
    iterations_used: int
    iterations_failed: bool
    warnings: list[str]
    features: CppKernelGeometryFeatures


HAS_COMPILED = False
_estimate_aero_cpp = None

try:
    from ._native_cpu_cpp import estimate_aero as _estimate_aero_cpp  # type: ignore[import-not-found]
    HAS_COMPILED = True
except Exception:  # pragma: no cover - optional accelerator import path
    _estimate_aero_cpp = None
    HAS_COMPILED = False


def estimate_aero(
    geometry_points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    alpha_deg: float,
    reynolds: float,
    mach: float,
    n_panels: int,
    iterations: int,
    stall_alpha_deg: float,
    residual_floor: float,
    iterations_to_converge: int,
) -> KernelOutput:
    """Return a proxy aero solution from a compiled kernel when present.

    Falls back to the pure Python kernel when compilation is unavailable.
    """

    if HAS_COMPILED:
        return _estimate_aero_cpp(
            geometry_points,
            alpha_deg,
            reynolds,
            mach,
            n_panels,
            iterations,
            stall_alpha_deg,
            residual_floor,
            iterations_to_converge,
        )

    return _python_estimate_aero(
        geometry_points=geometry_points,
        query_alpha_deg=alpha_deg,
        query_reynolds=reynolds,
        query_mach=mach,
        n_panels=n_panels,
        iterations_requested=iterations,
        stall_alpha_deg=stall_alpha_deg,
        residual_floor=residual_floor,
        iterations_to_converge=iterations_to_converge,
    )
