"""Geometry normalization and XFOIL file formatting helpers."""

from __future__ import annotations

from typing import Iterable, Sequence
import math

from .errors import XfoilGeometryError


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def validate_points(points, *, close_tolerance: float = 1e-6) -> None:
    """Validate a geometry point stream for stable XFOIL ingestion.

    The current checks are intentionally conservative and deterministic:
    - two-dimensional finite coordinates
    - at least 3 points
    - no duplicate leading/trailing points if closure is requested
    """

    points = list(points)
    if len(points) < 3:
        raise XfoilGeometryError("at least 3 points are required")

    for idx, p in enumerate(points):
        if not isinstance(p, Iterable):
            raise XfoilGeometryError(f"point {idx} is not a 2D pair: {p!r}")
        coords = tuple(p)
        if len(coords) != 2:
            raise XfoilGeometryError(f"point {idx} is not a 2D pair: {p!r}")
        x, y = float(coords[0]), float(coords[1])
        if not (_is_finite(x) and _is_finite(y)):
            raise XfoilGeometryError(f"point {idx} has non-finite coordinates: {p!r}")
        if not (0.0 - close_tolerance <= x <= 1.0 + close_tolerance):
            raise XfoilGeometryError(f"x coordinate out of expected [0,1] band at index {idx}: {x}")
        if abs(y) > 1.0:
            raise XfoilGeometryError(f"y coordinate magnitude appears invalid at index {idx}: {y}")

    # Basic closed shape sanity; duplicates in consecutive points can produce degenerate
    # solver behavior. Keep strict for deterministic behavior.
    for idx in range(len(points) - 1):
        if (
            abs(points[idx][0] - points[idx + 1][0]) <= close_tolerance
            and abs(points[idx][1] - points[idx + 1][1]) <= close_tolerance
        ):
            raise XfoilGeometryError(f"duplicate consecutive points at index {idx}: {points[idx]!r}")

    x0, y0 = points[0]
    x1, y1 = points[-1]
    closed = abs(x0 - x1) <= close_tolerance and abs(y0 - y1) <= close_tolerance
    if not closed:
        raise XfoilGeometryError(
            "geometry is not closed at end points; append first point at end or enable auto-close explicitly"
        )

    area = 0.0
    for (x_a, y_a), (x_b, y_b) in zip(points, points[1:] + points[:1]):
        area += x_a * y_b - x_b * y_a
    if abs(area) < 1e-12:
        raise XfoilGeometryError("geometry has zero signed area and is likely degenerate")


def format_points(points: Sequence[tuple[float, float]], *, precision: int = 12) -> str:
    """Format points into xfoil-compatible whitespace-separated coordinates."""

    lines = [f"{float(x):.{precision}f} {float(y):.{precision}f}" for x, y in points]
    return "\n".join(lines) + "\n"


def ensure_closed(points: Sequence[tuple[float, float]], *, close_tolerance: float = 1e-6) -> list[tuple[float, float]]:
    """Return a closed copy of the point list."""
    if not points:
        return []
    closed = list(points)
    if abs(points[0][0] - points[-1][0]) <= close_tolerance and abs(points[0][1] - points[-1][1]) <= close_tolerance:
        return closed
    closed.append((float(points[0][0]), float(points[0][1])))
    return closed
