from pathlib import Path

from xfoil_port.geometry import validate_points
from xfoil_port.errors import XfoilGeometryError


def test_validate_points_requires_closed():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.1), (0.0, 0.0)]
    validate_points(pts)  # no raise


def test_validate_points_rejects_unclosed():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.1)]
    try:
        validate_points(pts)
    except XfoilGeometryError:
        assert True
    else:
        raise AssertionError("expected validation error")


def test_validate_points_rejects_out_of_bounds():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.1, 2.0), (0.0, 0.0)]
    try:
        validate_points(pts)
    except XfoilGeometryError:
        assert True
    else:
        raise AssertionError("expected validation error")


def test_validate_points_rejects_duplicate_consecutive_points():
    pts = [(0.0, 0.0), (0.5, 0.05), (0.5, 0.05), (1.0, 0.0), (0.0, 0.0)]
    try:
        validate_points(pts)
    except XfoilGeometryError:
        assert True
    else:
        raise AssertionError("expected validation error")


def test_validate_points_rejects_degenerate_area():
    pts = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.0)]
    try:
        validate_points(pts)
    except XfoilGeometryError:
        assert True
    else:
        raise AssertionError("expected validation error")


def test_validate_points_handles_iterable_points():
    pts = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0))
    validate_points(iter(pts))
