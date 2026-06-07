"""CPU-kernel-style approximation functions for XFOIL surrogate evaluation.

This module isolates the mathematical core used by `NativeCpuXfoilEvaluator`.
The signatures are intentionally simple and deterministic so the same module can
be replaced later by a compiled C++/CUDA implementation without changing call
sites.
"""

from __future__ import annotations

import math
from typing import TypedDict


class KernelGeometryFeatures(TypedDict):
    n: float
    camber: float
    thickness: float
    curvature: float
    thickness_ratio: float
    chord: float


class KernelOutput(TypedDict):
    cl: float
    cd: float
    cm: float
    status: str
    residual: float
    iterations_used: int
    iterations_failed: bool
    warnings: list[str]
    features: KernelGeometryFeatures
from typing import Sequence


def geometry_features(geometry_points: Sequence[tuple[float, float]]) -> KernelGeometryFeatures:
    """Compute deterministic scalar geometry descriptors from ordered points."""

    xs = [float(p[0]) for p in geometry_points]
    ys = [float(p[1]) for p in geometry_points]
    n = len(geometry_points)

    if n < 3:
        return {
            "n": float(n),
            "camber": 0.0,
            "thickness": 0.0,
            "curvature": 0.0,
            "thickness_ratio": 0.0,
            "chord": 0.0,
        }

    x_min = min(xs)
    x_max = max(xs)
    chord = max(x_max - x_min, 1e-12)

    y_min = min(ys)
    y_max = max(ys)
    camber = 0.5 * (y_max + y_min)
    thickness = y_max - y_min

    curv_sum = 0.0
    segments = 0
    for i in range(1, n - 1):
        x0, y0 = geometry_points[i - 1]
        x1, y1 = geometry_points[i]
        x2, y2 = geometry_points[i + 1]

        dx1 = x1 - x0
        dx2 = x2 - x1
        if abs(dx1) < 1e-12 or abs(dx2) < 1e-12:
            continue

        s1 = math.atan2(y1 - y0, dx1)
        s2 = math.atan2(y2 - y1, dx2)
        curv_sum += abs(s2 - s1)
        segments += 1

    curvature = curv_sum / max(1, segments)

    return {
        "n": float(n),
        "camber": camber,
        "thickness": thickness,
        "curvature": curvature,
        "thickness_ratio": thickness / chord,
        "chord": chord,
    }


def estimate_aero(
    geometry_points: Sequence[tuple[float, float]],
    query_alpha_deg: float,
    query_reynolds: float,
    query_mach: float,
    *,
    n_panels: int | None,
    iterations_requested: int,
    stall_alpha_deg: float,
    residual_floor: float,
    iterations_to_converge: int,
) -> KernelOutput:
    """Deterministic closed-form proxy for CL/CD/CM and convergence metadata."""

    features = geometry_features(geometry_points)

    alpha_rad = math.radians(query_alpha_deg)
    camber = features["camber"]
    thickness_ratio = features["thickness_ratio"]
    curvature = features["curvature"]
    stall_alpha = max(1.0, abs(query_alpha_deg))

    stall_ratio = max(0.0, (stall_alpha - stall_alpha_deg) / 20.0)
    stall_ratio = min(stall_ratio, 1.0)

    camber_norm = max(-0.6, min(0.6, camber * 6.0))
    panel_scale = float(n_panels if n_panels is not None else 80.0)
    panel_scale = max(1.0, panel_scale)
    panel_scale = max(1.0, panel_scale / max(20.0, features["n"]))

    cl = 2.0 * math.pi * alpha_rad
    cl *= 1.0 + 0.25 * camber_norm
    cl *= 1.0 + 0.08 * math.tanh(curvature)
    cl *= 1.0 + 0.20 * min(2.0, panel_scale - 1.0) / 4.0
    cl *= 1.0 - 0.70 * stall_ratio

    if query_mach > 0:
        cl *= 1.0 - 0.12 * query_mach

    if query_reynolds > 0:
        re_scale = math.log10(max(query_reynolds, 1.0))
        cl *= 1.0 + 0.002 * max(0.0, re_scale - 5.0)

    cd = 0.004 + 0.018 * thickness_ratio
    cd += 0.0025 * abs(camber_norm)
    cd += 0.0012 * thickness_ratio * (1.0 + curvature)
    cd *= 1.0 + 0.5 * query_mach
    cd *= 1.0 + 0.03 * stall_ratio

    if query_reynolds > 0:
        cd *= 1.0 + 0.15 / math.sqrt(query_reynolds / 1e5 + 1.0)

    cm = -0.02 * (1.0 + camber_norm) - 0.003 * math.sin(alpha_rad * 2.0)
    cm -= 0.01 * thickness_ratio

    iterations_used = min(max(1, iterations_requested), max(1, iterations_to_converge))
    residual = max(residual_floor, math.exp(-0.04 * iterations_used))
    residual *= 1.0 + 0.8 * stall_ratio

    iterations_failed = stall_ratio > 0.9 or iterations_requested < iterations_to_converge * 0.25
    warnings: list[str] = []
    if iterations_failed:
        warnings.append("proxy_low_iterations")
    if stall_ratio > 0.55:
        warnings.append("proxy_stall_like_response")
    if n_panels is not None and n_panels < 16:
        warnings.append("low_panel_resolution")

    status = "ok" if not iterations_failed else "native_warnings"

    return KernelOutput(
        cl=cl,
        cd=cd,
        cm=cm,
        status=status,
        residual=residual,
        iterations_used=iterations_used,
        iterations_failed=iterations_failed,
        warnings=warnings,
        features=features,
    )
