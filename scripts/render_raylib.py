#!/usr/bin/env python3
"""Render native CPU evaluations with RayLib.

This script runs the native CPU approximation path on an airfoil and draws:

- geometry in the left panel
- CL sweep in the top-right chart
- CD sweep in the bottom-right chart

By default this script opens a RayLib window. Use `--offscreen` to render to a PNG
via RayLib image APIs when display is not available:

Usage example:
    UV_CACHE_DIR=/tmp/uv-cache env PYTHONPATH=src ./scripts/render_raylib.py \
        --alpha-start 2 --alpha-stop 10 --alpha-step 2 --re 1000000
    UV_CACHE_DIR=/tmp/uv-cache env PYTHONPATH=src ./scripts/render_raylib.py --offscreen \
        --offscreen-output logs/airfoil_render.png --geometry data/naca0012.dat \
        --alpha-start 2 --alpha-stop 10 --alpha-step 2 --re 1000000
    UV_CACHE_DIR=/tmp/uv-cache env PYTHONPATH=src ./scripts/render_raylib.py --naca 2412 \
        --alpha-start 2 --alpha-stop 10 --alpha-step 2 --re 1000000

Install requirement (one of these bindings):
    uv pip install raylib-py
"""

from __future__ import annotations

import os
import argparse
import importlib
import math
import sys
import socket
import re
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from xfoil_port.backends.native_cpu import NativeCpuXfoilConfig, NativeCpuXfoilEvaluator
from xfoil_port.types import XfoilQuery


_WINDOW_W = 1200
_WINDOW_H = 700
_PADDING = 24
_FPS = 60
_OUTPUT_DEFAULT = "logs/render_raylib.png"
_GEOMETRY_SCREEN_FRACTION = 0.12
_FIELD_ORIENTATIONS = (
    "normal",
    "flip_x",
    "flip_y",
    "flip_xy",
    "rot90_cw",
    "rot90_ccw",
    "rot180",
    "transpose",
    "transpose_rotate90_ccw",
    "transpose_flip",
    "screen_x_mirror_rot90_cw",
    "rot90_cw_flip_x",
)


def _read_geometry(path: Path) -> list[tuple[float, float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    points: list[tuple[float, float]] = []
    for line in lines:
        x, y = map(float, line.split()[:2])
        points.append((x, y))
    if not points:
        raise RuntimeError(f"no geometry points read from {path}")
    return points


def _naca_4points(code: str, *, samples: int = 161) -> list[tuple[float, float]]:
    code = code.strip()
    if not re.fullmatch(r"\d{4}", code):
        raise ValueError("naca code must be exactly 4 digits, e.g. 2412")

    if samples < 6:
        raise ValueError("--naca requires at least 6 samples")

    thickness = float(int(code[2:])) / 100.0
    camber_percent = float(int(code[0]))
    position_percent = float(int(code[1]))
    m = camber_percent / 100.0
    p = position_percent / 10.0

    beta = [math.pi * i / (samples - 1) for i in range(samples)]
    x = [0.5 * (1.0 - math.cos(v)) for v in beta]
    x = [min(1.0, max(0.0, xx)) for xx in x]

    thickness_y = [
        5.0
        * thickness
        * (0.2969 * math.sqrt(v) - 0.1260 * v - 0.3516 * v * v + 0.2843 * v * v * v - 0.1036 * v * v * v * v)
        for v in x
    ]

    camber = [0.0 for _ in x]
    slope = [0.0 for _ in x]
    if m > 0.0 and p > 0.0:
        inv_p2 = 1.0 / (p * p)
        inv_1mp2 = 1.0 / ((1.0 - p) * (1.0 - p))
    for idx, xx in enumerate(x):
        if xx <= p:
            camber[idx] = m / p / p * (2.0 * p * xx - xx * xx)
            slope[idx] = 2.0 * m * inv_p2 * (p - xx)
        else:
            camber[idx] = m / ((1.0 - p) * (1.0 - p)) * ((1.0 - 2.0 * p) + 2.0 * p * xx - xx * xx)
            slope[idx] = 2.0 * m * inv_1mp2 * (p - xx)

    theta = [math.atan(v) for v in slope]
    xu = [xx - t * math.sin(th) for xx, t, th in zip(x, thickness_y, theta)]
    yu = [c + t * math.cos(th) for c, t, th in zip(camber, thickness_y, theta)]
    xl = [xx + t * math.sin(th) for xx, t, th in zip(x, thickness_y, theta)]
    yl = [c - t * math.cos(th) for c, t, th in zip(camber, thickness_y, theta)]

    points = list(zip(reversed(xu), reversed(yu)))
    points.extend(zip(xl[1:], yl[1:]))
    points = [(min(1.0, max(0.0, float(px))), float(py)) for px, py in points]

    cleaned: list[tuple[float, float]] = []
    for px, py in points:
        if cleaned and abs(px - cleaned[-1][0]) <= 1e-6 and abs(py - cleaned[-1][1]) <= 1e-6:
            continue
        cleaned.append((px, py))

    if not cleaned:
        raise ValueError("generated NACA profile has no points")

    # Force deterministic, stable closure at trailing edge.
    cleaned[0] = (1.0, 0.0)
    cleaned[-1] = (1.0, 0.0)
    if len(cleaned) > 1 and abs(cleaned[-1][0] - cleaned[-2][0]) <= 1e-6 and abs(cleaned[-1][1] - cleaned[-2][1]) <= 1e-6:
        cleaned.pop()

    return cleaned


def _resolve_geometry(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / candidate).exists():
        return repo_root / candidate
    return candidate


def _load_raylib() -> tuple[
    object,
    Callable[[int, int, int, int], tuple[int, int, int, int]],
    Callable[[object], None],
]:
    """Import RayLib-like binding and return helpers.

    The script supports `pyray` / `raylib` module names as long as they expose
    the standard RayLib-style functions.
    """

    def _choose_fn(mod, camel_name: str, snake_name: str):
        fn = getattr(mod, camel_name, None)
        if callable(fn):
            return fn
        fn = getattr(mod, snake_name, None)
        if callable(fn):
            return fn
        return None

    def _choose_attr(mod, camel_name: str, snake_name: str | None, default=None):
        if snake_name is None:
            return getattr(mod, camel_name, default)
        return getattr(mod, camel_name, getattr(mod, snake_name, default))

    required = {
        "InitWindow": ("InitWindow", "init_window"),
        "CloseWindow": ("CloseWindow", "close_window"),
        "SetTargetFPS": ("SetTargetFPS", "set_target_fps"),
        "WindowShouldClose": ("WindowShouldClose", "window_should_close"),
        "BeginDrawing": ("BeginDrawing", "begin_drawing"),
        "EndDrawing": ("EndDrawing", "end_drawing"),
        "ClearBackground": ("ClearBackground", "clear_background"),
        "DrawText": ("DrawText", "draw_text"),
        "DrawLine": ("DrawLine", "draw_line"),
        "DrawCircleV": ("DrawCircleV", "draw_circle_v"),
        "DrawTriangleLines": ("DrawTriangleLines", "draw_triangle_lines"),
        "GetFrameTime": ("GetFrameTime", "get_frame_time"),
        "DrawRectangle": ("DrawRectangle", "draw_rectangle"),
        "DrawRectangleLines": ("DrawRectangleLines", "draw_rectangle_lines"),
        "GenImageColor": ("GenImageColor", "gen_image_color"),
        "ImageDrawText": ("ImageDrawText", "image_draw_text"),
        "ImageDrawLine": ("ImageDrawLine", "image_draw_line"),
        "ImageDrawRectangle": ("ImageDrawRectangle", "image_draw_rectangle"),
        "ImageDrawRectangleLines": ("ImageDrawRectangleLines", "image_draw_rectangle_lines"),
        "ExportImage": ("ExportImage", "export_image"),
        "UnloadImage": ("UnloadImage", "unload_image"),
    }

    candidates = ["raylib", "pyray", "raylib_py", "raylibpy", "pyraylib"]
    imported = None
    imported_name = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            missing = [
                name
                for name, (camel_name, snake_name) in required.items()
                if _choose_fn(module, camel_name, snake_name) is None
            ]
            if missing:
                continue
            imported = module
            imported_name = module_name
            break
        except Exception:
            continue

    if imported is None:
        raise RuntimeError(
            "RayLib module not found. Install one of: uv pip install raylib (recommended),"
            " uv pip install raylib-py, pyray, or equivalent package."
        )

    if imported_name:
        print(f"render_raylib: using {imported_name} backend")

    namespace = SimpleNamespace()
    for canonical, (camel_name, snake_name) in required.items():
        setattr(namespace, canonical, _choose_fn(imported, camel_name, snake_name))
    namespace.ffi = getattr(imported, "ffi", None)
    namespace.Image = _choose_attr(imported, "Image", None)

    # Constant aliases where possible.
    namespace.KEY_ESCAPE = _choose_attr(imported, "KEY_ESCAPE", None)
    namespace.KeyboardKey = _choose_attr(imported, "KeyboardKey", None)
    namespace.IsKeyPressed = _choose_attr(imported, "IsKeyPressed", "is_key_pressed", None)
    namespace.IsKeyDown = _choose_attr(imported, "IsKeyDown", "is_key_down", None)

    # Constant names differ in some bindings. Keep a small alias map.
    def _color_from_code(value: int) -> tuple[int, int, int, int]:
        # 0xRRGGBBAA style -> tuple expected by some wrappers.
        a = value & 0xFF
        value >>= 8
        b = value & 0xFF
        value >>= 8
        g = value & 0xFF
        value >>= 8
        r = value & 0xFF
        return r, g, b, a

    def _color_to_vec(value: int) -> tuple[int, int, int, int]:
        # Many `pyray`-style bindings expose Color values as ints in a helper.
        # Fall back to explicit RGBA tuples.
        return _color_from_code(value)

    if hasattr(imported, "Color") and hasattr(imported, "Fade"):
        # typical pyray style
        mkcolor = lambda r, g, b, a=255: imported.Color(r, g, b, a)
    elif namespace.ffi is not None:
        # raylib cffi binding exposes Color as a cdata struct in ffi
        def mkcolor(r: int, g: int, b: int, a: int = 255):
            return namespace.ffi.new(
                "Color *",
                {
                    "r": r,
                    "g": g,
                    "b": b,
                    "a": a,
                },
            )[0]
    else:
        mkcolor = lambda r, g, b, a=255: _color_to_vec((r << 24) | (g << 16) | (b << 8) | a)

    return namespace, mkcolor, namespace.CloseWindow


def _display_available(display: str) -> bool | None:
    if not display:
        return False

    if display.startswith(":"):
        idx = display[1:].split(".")[0]
        if not idx.isdigit():
            return None
        path = f"/tmp/.X11-unix/X{idx}"
        if not os.path.exists(path):
            return False

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(path)
            sock.close()
            return True
        except Exception:
            return False

    if display.startswith("unix:"):
        return _display_available(display[len("unix:"):])

    # Keep non-file based remote displays permissive: they may still work.
    return None


def _escape_requested(rl) -> bool:
    """Return True when user requests closing with Escape, if supported by binding."""
    key_lookup = (
        getattr(rl, "KEY_ESCAPE", None)
        or getattr(getattr(rl, "KeyboardKey", None), "KEY_ESCAPE", None)
    )
    if key_lookup is None:
        return False
    is_key_pressed = getattr(rl, "IsKeyPressed", None)
    if is_key_pressed is not None:
        try:
            if bool(is_key_pressed(key_lookup)):
                return True
        except Exception:
            pass
    is_key_down = getattr(rl, "IsKeyDown", None)
    if is_key_down is None:
        return False
    try:
        return bool(is_key_down(key_lookup))
    except Exception:
        return False


def _should_close_window(rl) -> bool:
    if rl.WindowShouldClose():
        return True
    return _escape_requested(rl)


def _bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        max_x += 1.0
        min_x -= 1.0
    if math.isclose(min_y, max_y):
        max_y += 1.0
        min_y -= 1.0
    dx = max_x - min_x
    dy = max_y - min_y
    pad_x = 0.08 * dx
    pad_y = 0.08 * dy
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _tight_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        raise ValueError("empty geometry provided")
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        max_x += 1.0
        min_x -= 1.0
    if math.isclose(min_y, max_y):
        max_y += 1.0
        min_y -= 1.0
    return min_x, max_x, min_y, max_y


def _geometry_render_bounds(
    points: list[tuple[float, float]],
    *,
    y_span_factor: float = 0.0,
) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = _tight_bounds(points)
    if y_span_factor <= 0.0:
        span_x = max_x - min_x
        span_y = max_y - min_y
        x_pad = 0.02 * span_x if span_x > 0.0 else 0.0
        y_pad = 0.02 * span_y if span_y > 0.0 else 0.0
        return min_x - x_pad, max_x + x_pad, min_y - y_pad, max_y + y_pad

    span_x = max_x - min_x
    span_y = max_y - min_y
    target_span_y = max(span_y, span_x * y_span_factor)
    cy = 0.5 * (min_y + max_y)
    half_span_y = 0.5 * target_span_y
    return min_x, max_x, cy - half_span_y, cy + half_span_y


def _fit_point_to_rect(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    rx, ry, rw, rh = rect
    sx = (x - min_x) / (max_x - min_x)
    sy = (y - min_y) / (max_y - min_y)
    px = int(rx + sx * rw)
    py = int(ry + rh - sy * rh)
    return px, py


def _fit_point_to_rect_equal_scale(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    rx, ry, rw, rh = rect
    dx = max_x - min_x
    dy = max_y - min_y
    if math.isclose(dx, 0.0) or math.isclose(dy, 0.0):
        return _fit_point_to_rect(x, y, bounds, rect)

    scale = min(rw / dx, rh / dy)
    x_offset = (rw - scale * dx) / 2.0
    y_offset = (rh - scale * dy) / 2.0
    px = int(round(rx + x_offset + (x - min_x) * scale))
    py = int(round(ry + y_offset + (max_y - y) * scale))
    return px, py


def _equal_scale_rect(
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    min_x, max_x, min_y, max_y = bounds
    rx, ry, rw, rh = rect
    dx = max_x - min_x
    dy = max_y - min_y
    if math.isclose(dx, 0.0) or math.isclose(dy, 0.0):
        return rect

    scale = min(rw / dx, rh / dy)
    inner_w = max(1, int(round(scale * dx)))
    inner_h = max(1, int(round(scale * dy)))
    inner_w = min(inner_w, rw)
    inner_h = min(inner_h, rh)
    x_offset = int((rw - inner_w) / 2)
    y_offset = int((rh - inner_h) / 2)

    return rx + x_offset, ry + y_offset, inner_w, inner_h


def _equal_scale_rect_with_max_height(
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
    max_height_fraction: float,
) -> tuple[int, int, int, int]:
    base_rect = _equal_scale_rect(bounds, rect)
    if max_height_fraction <= 0.0 or max_height_fraction >= 1.0:
        return base_rect

    min_x, max_x, min_y, max_y = bounds
    dx = max_x - min_x
    dy = max_y - min_y
    if math.isclose(dx, 0.0) or math.isclose(dy, 0.0):
        return base_rect

    rx, ry, rw, rh = rect
    target_height = int(max(1, rh * max_height_fraction))
    scale = min(rw / dx, rh / dy)
    capped_scale = min(scale, target_height / dy)
    if math.isclose(capped_scale, scale):
        return base_rect

    inner_w = max(1, int(round(capped_scale * dx)))
    inner_h = max(1, int(round(capped_scale * dy)))
    inner_w = min(inner_w, rw)
    inner_h = min(inner_h, target_height, rh)

    x_offset = int((rw - inner_w) / 2)
    y_offset = int((rh - inner_h) / 2)
    return rx + x_offset, ry + y_offset, inner_w, inner_h


def _fit_point_from_rect_equal_scale(
    sx: int,
    sy: int,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    rx, ry, rw, rh = rect
    dx = max_x - min_x
    dy = max_y - min_y
    if math.isclose(dx, 0.0) or math.isclose(dy, 0.0):
        return min_x, max_y

    scale = min(rw / dx, rh / dy)
    x_offset = (rw - scale * dx) / 2.0
    y_offset = (rh - scale * dy) / 2.0

    x = min_x + (sx - rx - x_offset) / scale
    y = max_y - (sy - ry - y_offset) / scale
    return x, y


def _fit_point_from_rect(
    sx: int,
    sy: int,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    rx, ry, rw, rh = rect
    dx = max_x - min_x
    dy = max_y - min_y

    if math.isclose(rw, 0.0) or math.isclose(rh, 0.0) or math.isclose(dx, 0.0) or math.isclose(dy, 0.0):
        return min_x, max_y

    x = min_x + (sx - rx) * dx / rw
    y = max_y - (sy - ry) * dy / rh
    return x, y


def _fit_x_to_range(x: float, x_min: float, x_max: float, width: int, left: int) -> int:
    if math.isclose(x_min, x_max):
        return left
    t = (x - x_min) / (x_max - x_min)
    return left + int(t * width)


def _fit_y_to_range(y: float, y_min: float, y_max: float, top: int, height: int) -> int:
    if math.isclose(y_min, y_max):
        return top + int(height / 2)
    t = (y - y_min) / (y_max - y_min)
    return top + int((1.0 - t) * height)


def _cstr(value: str) -> bytes:
    return value.encode("utf-8") + b"\0"


def _draw_text(rl, image_or_window, text: str, x: int, y: int, size: int, color, use_window: bool) -> None:
    if use_window:
        image_or_window.DrawText(_cstr(text), x, y, size, color)
        return

    image_ptr = rl.ffi.addressof(image_or_window)
    rl.ImageDrawText(image_ptr, _cstr(text), x, y, size, color)


def _field_mode_choices() -> tuple[str, str]:
    return ("pressure", "velocity")


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0.0:
        return math.hypot(px - x1, py - y1)

    c2 = vx * vx + vy * vy
    if c2 <= 0.0:
        return math.hypot(px - x1, py - y1)

    if c1 >= c2:
        return math.hypot(px - x2, py - y2)

    b = c1 / c2
    pbx = x1 + b * vx
    pby = y1 + b * vy
    return math.hypot(px - pbx, py - pby)


def _distance_to_polyline(
    px: float,
    py: float,
    points: list[tuple[float, float]],
) -> float:
    if len(points) < 2:
        return float("inf")
    best = float("inf")
    for idx in range(len(points) - 1):
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]
        d = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < best:
            best = d
    return best


def _fill_polygon_mask_window(
    rl,
    make_color,
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    polygon_points: list[tuple[float, float]],
    color,
    field_rect_mode: str = "geometry",
) -> None:
    if not polygon_points:
        return

    eq_rect = _equal_scale_rect(bounds, rect)
    eq_rx, eq_ry, eq_rw, eq_rh = eq_rect
    if eq_rw <= 0 or eq_rh <= 0:
        return

    def sample_world_y(screen_y: int) -> float:
        _, world_y = _fit_point_from_rect_equal_scale(eq_rx, screen_y + 0.5, bounds, eq_rect)
        return world_y

    def sample_screen_x(px: float, py: float) -> int:
        x, _ = _fit_point_to_rect_equal_scale(px, py, bounds, eq_rect)
        return x

    for screen_y in range(eq_ry, eq_ry + eq_rh):
        world_y = sample_world_y(screen_y)
        intersections: list[float] = []
        for idx in range(len(polygon_points) - 1):
            x1, y1 = polygon_points[idx]
            x2, y2 = polygon_points[idx + 1]
            if (y1 > world_y) == (y2 > world_y):
                continue
            if math.isclose(y1, y2):
                continue
            if not ((y1 <= world_y < y2) or (y2 <= world_y < y1)):
                continue
            t = (world_y - y1) / (y2 - y1)
            if 0.0 <= t <= 1.0:
                intersections.append(x1 + t * (x2 - x1))

        if len(intersections) < 2:
            continue
        intersections.sort()
        for i in range(0, len(intersections) - 1, 2):
            x_start = intersections[i]
            x_end = intersections[i + 1]
            screen_x0 = sample_screen_x(x_start, world_y)
            screen_x1 = sample_screen_x(x_end, world_y)
            if screen_x1 <= screen_x0:
                continue
            rl.DrawLine(screen_x0, screen_y, screen_x1, screen_y, color)


def _fill_polygon_mask_image(
    rl,
    make_color,
    image,
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    polygon_points: list[tuple[float, float]],
    color,
    field_rect_mode: str = "geometry",
) -> None:
    if not polygon_points:
        return

    eq_rect = _equal_scale_rect(bounds, rect)
    eq_rx, eq_ry, eq_rw, eq_rh = eq_rect
    if eq_rw <= 0 or eq_rh <= 0:
        return

    def sample_world_y(screen_y: int) -> float:
        _, world_y = _fit_point_from_rect_equal_scale(eq_rx, screen_y + 0.5, bounds, eq_rect)
        return world_y

    def sample_screen_x(px: float, py: float) -> int:
        x, _ = _fit_point_to_rect_equal_scale(px, py, bounds, eq_rect)
        return x

    image_ptr = rl.ffi.addressof(image)
    for screen_y in range(eq_ry, eq_ry + eq_rh):
        world_y = sample_world_y(screen_y)
        intersections: list[float] = []
        for idx in range(len(polygon_points) - 1):
            x1, y1 = polygon_points[idx]
            x2, y2 = polygon_points[idx + 1]
            if (y1 > world_y) == (y2 > world_y):
                continue
            if math.isclose(y1, y2):
                continue
            if not ((y1 <= world_y < y2) or (y2 <= world_y < y1)):
                continue
            t = (world_y - y1) / (y2 - y1)
            if 0.0 <= t <= 1.0:
                intersections.append(x1 + t * (x2 - x1))

        if len(intersections) < 2:
            continue
        intersections.sort()
        for i in range(0, len(intersections) - 1, 2):
            x_start = intersections[i]
            x_end = intersections[i + 1]
            screen_x0 = sample_screen_x(x_start, world_y)
            screen_x1 = sample_screen_x(x_end, world_y)
            if screen_x1 <= screen_x0:
                continue
            rl.ImageDrawLine(image_ptr, screen_x0, screen_y, screen_x1, screen_y, color)


def _field_domain(
    points: list[tuple[float, float]],
    x_margin_factor: float = 2.0,
    y_margin_factor: float | None = None,
) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = _bounds(points)

    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0

    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 0.0:
        span_x = 1.0
    if span_y <= 0.0:
        span_y = 1.0

    # Interpret margin factors as percentage-like padding.
    # 180.0 => +180% padding around both sides.
    extra_scale_x = max(0.0, x_margin_factor) / 100.0
    if y_margin_factor is None:
        y_margin_factor = x_margin_factor
    extra_scale_y = max(0.0, y_margin_factor) / 100.0

    # Keep the field domain anchored to geometry bounds so the pressure map lines up with
    # the airfoil and does not drift toward the chord midpoint.
    half_x = 0.5 * span_x * (1.0 + extra_scale_x)
    half_y = 0.5 * span_y * (1.0 + extra_scale_y)
    half_y = max(half_y, 0.001)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    return cx - half_x, cx + half_x, cy - half_y, cy + half_y


def _resolve_field_domain(
    points: list[tuple[float, float]],
    mode: str,
    margin_factor: float = 2.0,
    y_margin_factor: float | None = None,
) -> tuple[float, float, float, float]:
    if mode == "bounds":
        return _geometry_render_bounds(points)
    if mode == "padded":
        return _field_domain(points, x_margin_factor=margin_factor, y_margin_factor=y_margin_factor)
    raise ValueError(f"unknown field domain mode: {mode}")


def _field_reference_origin(points: list[tuple[float, float]], mode: str) -> tuple[float, float]:
    if mode == "leading_edge":
        x_min = min(x for x, _ in points)
        return x_min, 0.0
    if mode == "bounds_center":
        min_x, max_x, min_y, max_y = _tight_bounds(points)
        return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    if mode == "zero":
        min_x = min(x for x, _ in points)
        return min_x, 0.0
    raise ValueError(f"unknown field origin mode: {mode}")


def _stable_range(values: list[float], *, trim_fraction: float = 0.02) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if len(values) < 8:
        return min(values), max(values)

    sorted_values = sorted(values)
    lo = int(len(sorted_values) * trim_fraction)
    hi = max(lo + 1, int(len(sorted_values) * (1.0 - trim_fraction)) - 1)
    lo = min(max(0, lo), len(sorted_values) - 2)
    hi = min(max(lo + 1, hi), len(sorted_values) - 1)

    vmin = sorted_values[lo]
    vmax = sorted_values[hi]
    if math.isclose(vmin, vmax):
        return min(values), max(values)
    return vmin, vmax


def _field_value_at_orientation(
    field: list[float],
    rows: int,
    cols: int,
    row: int,
    col: int,
    orientation: str,
) -> float:
    if not field or rows <= 0 or cols <= 0:
        return 0.0

    v = 0.0 if rows == 1 else row / (rows - 1)
    u = 0.0 if cols == 1 else col / (cols - 1)

    if orientation == "normal":
        uu = u
        vv = v
    elif orientation == "screen_x_mirror_rot90_cw":
        # Explicitly apply: mirror across screen X axis (y inversion), then rotate
        # 90° clockwise in screen space.
        uu = 1.0 - v
        vv = 1.0 - u
    elif orientation == "flip_x":
        uu = 1.0 - u
        vv = v
    elif orientation == "flip_y":
        uu = u
        vv = 1.0 - v
    elif orientation == "flip_xy":
        uu = 1.0 - u
        vv = 1.0 - v
    elif orientation == "rot90_cw":
        uu = v
        vv = 1.0 - u
    elif orientation == "rot90_ccw":
        uu = 1.0 - v
        vv = u
    elif orientation == "rot180":
        uu = 1.0 - u
        vv = 1.0 - v
    elif orientation == "transpose":
        uu = v
        vv = u
    elif orientation == "transpose_rotate90_ccw":
        # Equivalent to 90° CCW rotation, then mirror around screen-Y axis.
        uu = v
        vv = u
    elif orientation in {"transpose_flip", "rot90_cw_flip_x"}:
        uu = 1.0 - v
        vv = 1.0 - u
    else:
        uu = u
        vv = v

    src_col = int(_clamp(uu, 0.0, 1.0) * (cols - 1)) if cols > 1 else 0
    src_row = int(_clamp(vv, 0.0, 1.0) * (rows - 1)) if rows > 1 else 0
    return field[src_row * cols + src_col]


def _field_value_at_world(
    field: list[float],
    rows: int,
    cols: int,
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    orientation: str,
    *,
    bilinear: bool = True,
) -> float:
    if not field or rows <= 0 or cols <= 0:
        return 0.0

    min_x, max_x, min_y, max_y = bounds
    if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
        return 0.0

    # Map world point into normalized field coordinates.
    u = (x - min_x) / (max_x - min_x)
    v = (max_y - y) / (max_y - min_y)

    if orientation == "normal":
        uu = u
        vv = v
    elif orientation == "screen_x_mirror_rot90_cw":
        # Explicitly apply: mirror across screen X axis (y inversion), then rotate
        # 90° clockwise in screen space.
        uu = 1.0 - v
        vv = 1.0 - u
    elif orientation == "flip_x":
        uu = 1.0 - u
        vv = v
    elif orientation == "flip_y":
        uu = u
        vv = 1.0 - v
    elif orientation == "flip_xy":
        uu = 1.0 - u
        vv = 1.0 - v
    elif orientation == "rot90_cw":
        uu = v
        vv = 1.0 - u
    elif orientation == "rot90_ccw":
        uu = 1.0 - v
        vv = u
    elif orientation == "rot180":
        uu = 1.0 - u
        vv = 1.0 - v
    elif orientation == "transpose":
        uu = v
        vv = u
    elif orientation == "transpose_rotate90_ccw":
        # Equivalent to 90° CCW rotation, then mirror around screen-Y axis.
        uu = v
        vv = u
    elif orientation in {"transpose_flip", "rot90_cw_flip_x"}:
        uu = 1.0 - v
        vv = 1.0 - u
    else:
        uu = u
        vv = v

    uu = _clamp(uu, 0.0, 1.0)
    vv = _clamp(vv, 0.0, 1.0)

    if rows == 1 and cols == 1:
        return field[0]

    if not bilinear or rows == 1 or cols == 1:
        sx = int(uu * (cols - 1)) if cols > 1 else 0
        sy = int(vv * (rows - 1)) if rows > 1 else 0
        return field[sy * cols + sx]

    x_scaled = uu * (cols - 1)
    y_scaled = vv * (rows - 1)
    x0 = int(math.floor(x_scaled))
    y0 = int(math.floor(y_scaled))
    x1 = min(x0 + 1, cols - 1)
    y1 = min(y0 + 1, rows - 1)

    tx = x_scaled - x0
    ty = y_scaled - y0

    v00 = field[y0 * cols + x0]
    v10 = field[y0 * cols + x1]
    v01 = field[y1 * cols + x0]
    v11 = field[y1 * cols + x1]

    v0 = v00 * (1.0 - tx) + v10 * tx
    v1 = v01 * (1.0 - tx) + v11 * tx
    return v0 * (1.0 - ty) + v1 * ty


def _field_color(value: float, vmin: float, vmax: float, mode: str) -> tuple[int, int, int, int]:
    if math.isclose(vmax, vmin):
        t = 0.5
    else:
        t = _clamp((value - vmin) / (vmax - vmin), 0.0, 1.0)

    if mode == "velocity":
        # blue -> white -> red
        if t <= 0.5:
            u = t * 2.0
            return (
                int(255 * u),
                int(255 * u),
                int(255),
                220,
            )
            u = (t - 0.5) * 2.0
            return (
                255,
                int(255 * (1.0 - u)),
                int(255 * (1.0 - u)),
                220,
            )

    # pressure: deep blue -> white -> deep red
    if t <= 0.5:
        u = t * 2.0
        return (
            int(70 * u),
            int(140 * u),
            int(255),
            220,
        )

    u = (t - 0.5) * 2.0
    return (
            255,
            int(255 * (1.0 - 0.4 * u)),
            int(70 * (1.0 - u)),
            220,
        )


def _sample_field_for_alpha(
    points: list[tuple[float, float]],
    alpha_deg: float,
    cl: float,
    mode: str,
    *,
    field_domain: tuple[float, float, float, float],
    grid_cols: int,
    grid_rows: int,
    vortex_scale: float = 1.0,
    core_size: float = 0.04,
    origin_mode: str = "leading_edge",
) -> tuple[list[float], float, float, tuple[float, float, float, float]]:
    x_min, x_max, y_min, y_max = field_domain
    chord = max(max(x for x, _ in points) - min(x for x, _ in points), 1.0e-12)
    # Place the field reference on a known airfoil point to keep geometry/field alignment
    # deterministic even when the airfoil origin differs between files.
    cx, cy = _field_reference_origin(points, origin_mode)

    alpha_rad = math.radians(alpha_deg)
    u_inf = math.cos(alpha_rad)
    v_inf = math.sin(alpha_rad)
    gamma = -cl / (2.0 * math.pi) if cl is not None else 0.0
    gamma *= vortex_scale
    core = max(core_size, 1.0e-4)
    body_influence_scale = max(0.02 * chord, 0.02)

    values: list[float] = []
    vmin = float("inf")
    vmax = float("-inf")

    if grid_cols < 1 or grid_rows < 1:
        return [], 0.0, 0.0, (x_min, x_max, y_min, y_max)

    for row in range(grid_rows):
        fy = 1.0 - row / max(1, grid_rows - 1)
        y = y_min + fy * (y_max - y_min)
        for col in range(grid_cols):
            fx = col / max(1, grid_cols - 1)
            x = x_min + fx * (x_max - x_min)
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy + core * core
            inv_r2 = 1.0 / r2

            # induced vortex component
            u_vortex = -gamma * dy * inv_r2
            v_vortex = gamma * dx * inv_r2

            # keep uniform field dominant away from the body
            body_d = _distance_to_polyline(x, y, points)
            damp = math.exp(-body_d / body_influence_scale)

            u = u_inf + u_vortex * damp
            v = v_inf + v_vortex * damp
            speed = math.hypot(u, v)

            if mode == "velocity":
                sample = speed
            else:
                sample = 1.0 - speed * speed

            values.append(sample)
            vmin = min(vmin, sample)
            vmax = max(vmax, sample)

    if math.isclose(vmin, vmax):
        return values, 0.0, 1.0, (x_min, x_max, y_min, y_max)

    vmin, vmax = _stable_range(values)

    return values, vmin, vmax, (x_min, x_max, y_min, y_max)


def _frame_to_sample(samples: list[Sample], frame: float) -> Sample:
    alpha_min = samples[0].alpha
    alpha_max = samples[-1].alpha
    span = abs(alpha_max - alpha_min)
    alpha_target = alpha_min + (span if span != 0 else 0.0) * (frame % 1.0)
    return min(samples, key=lambda s: abs(s.alpha - alpha_target))


@dataclass(frozen=True)
class Sample:
    alpha: float
    cl: float
    cd: float
    cm: float
    status: str
    field_domain: tuple[float, float, float, float] | None = None
    field: list[float] | None = None
    field_mode: str = "pressure"
    field_min: float = 0.0
    field_max: float = 1.0


def _evaluate_sweep(points: list[tuple[float, float]], args) -> list[Sample]:
    if args.alpha_start > args.alpha_stop and args.alpha_step > 0:
        raise ValueError("alpha-step must be negative when alpha-start > alpha-stop")
    if args.alpha_start < args.alpha_stop and args.alpha_step < 0:
        raise ValueError("alpha-step must be positive when alpha-start < alpha-stop")
    if args.alpha_step == 0:
        raise ValueError("alpha-step cannot be zero")

    evaluator = NativeCpuXfoilEvaluator(
        NativeCpuXfoilConfig(
            cache_results=not args.no_cache,
            panel_fallback=args.panel_fallback,
            iterations_to_converge=args.iterations_to_converge,
        )
    )
    field_mode = args.field_mode
    field_cols = max(1, int(args.field_cols))
    field_rows = max(1, int(args.field_rows))
    sweep_field_domain = _resolve_field_domain(
        points,
        args.field_domain_mode,
        margin_factor=args.field_margin,
        y_margin_factor=args.field_margin_y,
    )

    alpha = args.alpha_start
    samples: list[Sample] = []
    while True:
        if (args.alpha_step > 0 and alpha > args.alpha_stop + 1e-12) or (
            args.alpha_step < 0 and alpha < args.alpha_stop - 1e-12
        ):
            break

        result = evaluator.evaluate(
            points,
            XfoilQuery(
                alpha_deg=alpha,
                reynolds=args.reynolds,
                mach=args.mach,
                iterations=args.iterations,
                n_crit=args.n_crit,
                n_panels=args.n_panels,
            ),
            name="raylib-native",
        ).payload

        cl_val = float(result.cl) if result.cl is not None else 0.0
        field_values: list[float] = []
        field_min = 0.0
        field_max = 1.0
        field_domain = None
        if field_mode != "off":
            field_values, field_min, field_max, field_domain = _sample_field_for_alpha(
                points,
                float(alpha),
                cl_val,
                field_mode,
                field_domain=sweep_field_domain,
                grid_cols=field_cols,
                grid_rows=field_rows,
                vortex_scale=args.field_strength,
                core_size=args.field_core,
                origin_mode=args.field_origin_mode,
            )

        samples.append(
            Sample(
                alpha=float(alpha),
                cl=cl_val,
                cd=float(result.cd) if result.cd is not None else 0.0,
                cm=float(result.cm) if result.cm is not None else 0.0,
                status=result.status,
                field_domain=field_domain,
                field=field_values,
                field_mode=field_mode,
                field_min=field_min,
                field_max=field_max,
            )
        )

        if abs(alpha - args.alpha_stop) < abs(args.alpha_step) * 0.5:
            break
        alpha += args.alpha_step

    if not samples:
        raise RuntimeError("no samples were generated from the provided alpha range")
    return samples


def _draw_field_map_window(
    rl,
    make_color,
    sample: Sample,
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    cols: int,
    rows: int,
    field_orientation: str,
    field_rect_mode: str,
) -> None:
    if not sample.field:
        return

    field_domain = sample.field_domain or bounds
    draw_rect = _equal_scale_rect(field_domain, rect)
    draw_rx, draw_ry, draw_rw, draw_rh = draw_rect
    if draw_rw <= 0 or draw_rh <= 0:
        return

    sample_rows = max(1, rows)
    sample_cols = max(1, cols)
    draw_rows = draw_rh
    draw_cols = draw_rw
    if draw_rows <= 0:
        draw_rows = 1
    if draw_cols <= 0:
        draw_cols = 1
    vmin = sample.field_min
    vmax = sample.field_max
    field = sample.field

    for row in range(draw_rows):
        y = draw_ry + row
        for col in range(draw_cols):
            x = draw_rx + col
            sample_x, sample_y = (
                _fit_point_from_rect_equal_scale(x + 0.5, y + 0.5, field_domain, draw_rect)
            )
            val = _field_value_at_world(
                field,
                sample_rows,
                sample_cols,
                sample_x,
                sample_y,
                field_domain,
                field_orientation,
                bilinear=True,
            )
            color = _field_color(val, vmin, vmax, sample.field_mode)
            if hasattr(rl, "DrawPixel"):
                rl.DrawPixel(x, y, color)
            else:
                rl.DrawRectangle(x, y, 1, 1, color)


def _draw_field_map_image(
    rl,
    make_color,
    image,
    sample: Sample,
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    cols: int,
    rows: int,
    field_orientation: str,
    field_rect_mode: str,
) -> None:
    field = sample.field or []
    if not field:
        return

    field_domain = sample.field_domain or bounds
    draw_rect = _equal_scale_rect(field_domain, rect)
    draw_rx, draw_ry, draw_rw, draw_rh = draw_rect
    if draw_rw <= 0 or draw_rh <= 0:
        return

    sample_rows = max(1, rows)
    sample_cols = max(1, cols)
    image_ptr = rl.ffi.addressof(image)
    vmin = sample.field_min
    vmax = sample.field_max
    draw_rows = draw_rh
    draw_cols = draw_rw
    if draw_rows <= 0:
        draw_rows = 1
    if draw_cols <= 0:
        draw_cols = 1

    for row in range(draw_rows):
        y = draw_ry + row
        for col in range(draw_cols):
            x = draw_rx + col
            sample_x, sample_y = (
                _fit_point_from_rect_equal_scale(x + 0.5, y + 0.5, field_domain, draw_rect)
            )
            val = _field_value_at_world(
                field,
                sample_rows,
                sample_cols,
                sample_x,
                sample_y,
                field_domain,
                field_orientation,
                bilinear=True,
            )
            color = _field_color(val, vmin, vmax, sample.field_mode)
            if hasattr(rl, "ImageDrawPixel"):
                rl.ImageDrawPixel(image_ptr, x, y, color)
            else:
                rl.ImageDrawLine(image_ptr, x, y, x, y, color)


def _draw_geometry(
    rl,
    make_color,
    points: list[tuple[float, float]],
    bounds,
    geometry_bounds,
    sample: Sample | None,
    field_cols: int,
    field_rows: int,
    field_orientation: str,
    field_rect_mode: str,
) -> None:
    gx = _PADDING
    gy = _PADDING
    gw = _WINDOW_W // 2 - 2 * _PADDING
    gh = _WINDOW_H - 2 * _PADDING
    geom_rect = _equal_scale_rect_with_max_height(
        geometry_bounds,
        (gx, gy, gw, gh),
        _GEOMETRY_SCREEN_FRACTION,
    )
    field_bounds = sample.field_domain if sample is not None and sample.field_domain is not None else geometry_bounds
    field_rect = _equal_scale_rect(field_bounds, (gx, gy, gw, gh))

    # panel background
    rl.DrawRectangle(gx, gy, gw, gh, make_color(18, 18, 18, 255))
    rl.DrawRectangleLines(gx, gy, gw, gh, make_color(90, 90, 90, 255))

    if sample is not None and sample.field:
        _draw_field_map_window(
            rl,
            make_color,
            sample,
            field_rect,
            field_bounds,
            max(1, field_cols),
            max(1, field_rows),
            field_orientation,
            field_rect_mode,
        )
        _fill_polygon_mask_window(
            rl,
            make_color,
            field_rect,
            field_bounds,
            points,
            make_color(0, 0, 0, 255),
            field_rect_mode,
        )

    for i in range(len(points) - 1):
        p1 = _fit_point_to_rect_equal_scale(points[i][0], points[i][1], geometry_bounds, geom_rect)
        p2 = _fit_point_to_rect_equal_scale(points[i + 1][0], points[i + 1][1], geometry_bounds, geom_rect)
        rl.DrawLine(p1[0], p1[1], p2[0], p2[1], make_color(40, 220, 220, 255))

    overlay = "Field off" if sample is None or not sample.field else f"{sample.field_mode.title()} field"
    rl.DrawText(_cstr(f"Geometry | {overlay}"), gx + 8, gy + 8, 14, make_color(220, 220, 220, 255))


def _draw_geometry_image(
    rl,
    make_color,
    image,
    points: list[tuple[float, float]],
    bounds,
    geometry_bounds,
    sample: Sample | None,
    field_cols: int,
    field_rows: int,
    field_orientation: str,
    field_rect_mode: str,
) -> None:
    image_ptr = rl.ffi.addressof(image)
    gx = _PADDING
    gy = _PADDING
    gw = _WINDOW_W // 2 - 2 * _PADDING
    gh = _WINDOW_H - 2 * _PADDING
    geom_rect = _equal_scale_rect_with_max_height(
        geometry_bounds,
        (gx, gy, gw, gh),
        _GEOMETRY_SCREEN_FRACTION,
    )
    field_bounds = sample.field_domain if sample is not None and sample.field_domain is not None else geometry_bounds
    field_rect = _equal_scale_rect(field_bounds, (gx, gy, gw, gh))

    # panel background
    rl.ImageDrawRectangle(image_ptr, gx, gy, gw, gh, make_color(18, 18, 18, 255))
    rl.ImageDrawRectangleLines(image_ptr, gx, gy, gw, gh, make_color(90, 90, 90, 255))

    if sample is not None and sample.field:
        _draw_field_map_image(
            rl,
            make_color,
            image,
            sample,
            field_rect,
            field_bounds,
            max(1, field_cols),
            max(1, field_rows),
            field_orientation,
            field_rect_mode,
        )
        _fill_polygon_mask_image(
            rl,
            make_color,
            image,
            field_rect,
            field_bounds,
            points,
            make_color(0, 0, 0, 255),
            field_rect_mode,
        )

    for i in range(len(points) - 1):
        p1 = _fit_point_to_rect_equal_scale(points[i][0], points[i][1], geometry_bounds, geom_rect)
        p2 = _fit_point_to_rect_equal_scale(points[i + 1][0], points[i + 1][1], geometry_bounds, geom_rect)
        rl.ImageDrawLine(image_ptr, p1[0], p1[1], p2[0], p2[1], make_color(40, 220, 220, 255))

    overlay = "Field off" if sample is None or not sample.field else f"{sample.field_mode.title()} field"
    _draw_text(rl, image, f"Geometry | {overlay}", gx + 8, gy + 8, 14, make_color(220, 220, 220, 255), use_window=False)


def _draw_chart(rl, make_color, samples: list[Sample]):
    chart_left = _WINDOW_W // 2 + _PADDING
    chart_top = _PADDING
    chart_w = _WINDOW_W - chart_left - _PADDING
    chart_h = (_WINDOW_H - 3 * _PADDING) // 2

    alpha_min = samples[0].alpha
    alpha_max = samples[-1].alpha
    cl_min = min(s.cl for s in samples)
    cl_max = max(s.cl for s in samples)
    cd_min = min(s.cd for s in samples)
    cd_max = max(s.cd for s in samples)

    span = lambda lo, hi: hi - lo if not math.isclose(hi, lo) else 1.0
    cl_pad = 0.1 * span(cl_min, cl_max)
    cd_pad = 0.1 * span(cd_min, cd_max)
    cl_min -= cl_pad
    cl_max += cl_pad
    cd_min -= cd_pad
    cd_max += cd_pad

    # CL panel
    cx = chart_left
    cy = chart_top
    cw = chart_w
    ch = chart_h
    rl.DrawRectangle(cx, cy, cw, ch, make_color(18, 18, 18, 255))
    rl.DrawRectangleLines(cx, cy, cw, ch, make_color(90, 90, 90, 255))
    rl.DrawText(_cstr("CL vs alpha"), cx + 8, cy + 8, 18, make_color(220, 220, 220, 255))

    for idx in range(1, len(samples)):
        s0 = samples[idx - 1]
        s1 = samples[idx]
        x0 = _fit_x_to_range(s0.alpha, alpha_min, alpha_max, cw, cx)
        x1 = _fit_x_to_range(s1.alpha, alpha_min, alpha_max, cw, cx)
        y0 = _fit_y_to_range(s0.cl, cl_min, cl_max, cy, ch)
        y1 = _fit_y_to_range(s1.cl, cl_min, cl_max, cy, ch)
        rl.DrawLine(x0, y0, x1, y1, make_color(70, 170, 255, 255))

    # CD panel
    cx2 = chart_left
    cy2 = chart_top + ch + _PADDING
    rl.DrawRectangle(cx2, cy2, cw, ch, make_color(18, 18, 18, 255))
    rl.DrawRectangleLines(cx2, cy2, cw, ch, make_color(90, 90, 90, 255))
    rl.DrawText(_cstr("CD vs alpha"), cx2 + 8, cy2 + 8, 18, make_color(220, 220, 220, 255))

    for idx in range(1, len(samples)):
        s0 = samples[idx - 1]
        s1 = samples[idx]
        x0 = _fit_x_to_range(s0.alpha, alpha_min, alpha_max, cw, cx2)
        x1 = _fit_x_to_range(s1.alpha, alpha_min, alpha_max, cw, cx2)
        y0 = _fit_y_to_range(s0.cd, cd_min, cd_max, cy2, ch)
        y1 = _fit_y_to_range(s1.cd, cd_min, cd_max, cy2, ch)
        rl.DrawLine(x0, y0, x1, y1, make_color(255, 140, 80, 255))


def _draw_chart_image(rl, make_color, image, samples: list[Sample]):
    image_ptr = rl.ffi.addressof(image)
    chart_left = _WINDOW_W // 2 + _PADDING
    chart_top = _PADDING
    chart_w = _WINDOW_W - chart_left - _PADDING
    chart_h = (_WINDOW_H - 3 * _PADDING) // 2

    alpha_min = samples[0].alpha
    alpha_max = samples[-1].alpha
    cl_min = min(s.cl for s in samples)
    cl_max = max(s.cl for s in samples)
    cd_min = min(s.cd for s in samples)
    cd_max = max(s.cd for s in samples)

    span = lambda lo, hi: hi - lo if not math.isclose(hi, lo) else 1.0
    cl_pad = 0.1 * span(cl_min, cl_max)
    cd_pad = 0.1 * span(cd_min, cd_max)
    cl_min -= cl_pad
    cl_max += cl_pad
    cd_min -= cd_pad
    cd_max += cd_pad

    # CL panel
    cx = chart_left
    cy = chart_top
    cw = chart_w
    ch = chart_h
    rl.ImageDrawRectangle(image_ptr, cx, cy, cw, ch, make_color(18, 18, 18, 255))
    rl.ImageDrawRectangleLines(image_ptr, cx, cy, cw, ch, make_color(90, 90, 90, 255))
    _draw_text(rl, image, "CL vs alpha", cx + 8, cy + 8, 18, make_color(220, 220, 220, 255), use_window=False)

    for idx in range(1, len(samples)):
        s0 = samples[idx - 1]
        s1 = samples[idx]
        x0 = _fit_x_to_range(s0.alpha, alpha_min, alpha_max, cw, cx)
        x1 = _fit_x_to_range(s1.alpha, alpha_min, alpha_max, cw, cx)
        y0 = _fit_y_to_range(s0.cl, cl_min, cl_max, cy, ch)
        y1 = _fit_y_to_range(s1.cl, cl_min, cl_max, cy, ch)
        rl.ImageDrawLine(image_ptr, x0, y0, x1, y1, make_color(70, 170, 255, 255))

    # CD panel
    cx2 = chart_left
    cy2 = chart_top + ch + _PADDING
    rl.ImageDrawRectangle(image_ptr, cx2, cy2, cw, ch, make_color(18, 18, 18, 255))
    rl.ImageDrawRectangleLines(image_ptr, cx2, cy2, cw, ch, make_color(90, 90, 90, 255))
    _draw_text(rl, image, "CD vs alpha", cx2 + 8, cy2 + 8, 18, make_color(220, 220, 220, 255), use_window=False)

    for idx in range(1, len(samples)):
        s0 = samples[idx - 1]
        s1 = samples[idx]
        x0 = _fit_x_to_range(s0.alpha, alpha_min, alpha_max, cw, cx2)
        x1 = _fit_x_to_range(s1.alpha, alpha_min, alpha_max, cw, cx2)
        y0 = _fit_y_to_range(s0.cd, cd_min, cd_max, cy2, ch)
        y1 = _fit_y_to_range(s1.cd, cd_min, cd_max, cy2, ch)
        rl.ImageDrawLine(image_ptr, x0, y0, x1, y1, make_color(255, 140, 80, 255))


def _draw_overlay_image(
    rl,
    make_color,
    image,
    samples: list[Sample],
    frame: float,
    geometry_name: str,
    field_orientation: str,
) -> None:
    best = _frame_to_sample(samples, frame)
    field_val = "na"
    if best.field:
        center_idx = len(best.field) // 2
        field_val = f"{best.field[center_idx]:.5f}"
    _draw_text(
        rl,
        image,
        f"{geometry_name} native CPU | alpha={best.alpha:+.2f} deg | CL={best.cl:+.4f} | CD={best.cd:.5f} | CM={best.cm:+.5f}",
        _PADDING,
        _WINDOW_H - 2 * _PADDING,
        18,
        make_color(240, 240, 120, 255),
        use_window=False,
    )
    if best.field:
        _draw_text(
            rl,
            image,
            f"{best.field_mode.title()} field center={field_val} | orient={field_orientation}",
            _PADDING,
            _WINDOW_H - 3 * _PADDING // 2,
            14,
            make_color(220, 220, 255, 255),
            use_window=False,
        )
    _draw_text(
        rl,
        image,
        f"status: {best.status}",
        _PADDING,
        _WINDOW_H - _PADDING,
        16,
        make_color(220, 220, 220, 255),
        use_window=False,
    )


def _render_offscreen(rl, make_color, points: list[tuple[float, float]], samples: list[Sample], args) -> int:
    output = Path(args.offscreen_output)
    bounds = _tight_bounds(points)
    geometry_bounds = _geometry_render_bounds(points)
    image = rl.GenImageColor(_WINDOW_W, _WINDOW_H, make_color(12, 12, 12, 255))
    frame_sample = _frame_to_sample(samples, 0.0)

    try:
        _draw_geometry_image(
            rl,
            make_color,
            image,
            points,
            bounds,
            geometry_bounds,
            frame_sample,
            args.field_cols,
            args.field_rows,
            args.field_orientation,
            args.field_rect_mode,
        )
        _draw_chart_image(rl, make_color, image, samples)
        # One frame snapshot is enough for static render artifacts.
        _draw_overlay_image(
            rl,
            make_color,
            image,
            samples,
            0.0,
            args.geometry_name,
            args.field_orientation,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        ok = rl.ExportImage(image, str(output).encode("utf-8"))
        print(f"render_raylib: saved offscreen rendering to {output} (ok={bool(ok)})")
        return 0 if ok else 1
    finally:
        rl.UnloadImage(image)


def _render_offscreen_orientation_sweep(
    rl,
    make_color,
    points: list[tuple[float, float]],
    samples: list[Sample],
    args,
) -> int:
    base = Path(args.offscreen_output)
    stem = base.stem or "field-orientation"
    suffix = base.suffix or ".png"
    parent = base.parent
    parent.mkdir(parents=True, exist_ok=True)

    status = 0
    entries: list[tuple[str, Path]] = []
    for orientation in _FIELD_ORIENTATIONS:
        p = parent / f"{stem}-{orientation}{suffix}"
        run_args = SimpleNamespace(**vars(args))
        run_args.offscreen_output = str(p)
        run_args.field_orientation = orientation
        status = max(status, _render_offscreen(rl, make_color, points, samples, run_args))
        entries.append((orientation, p))

    gallery = parent / f"{stem}-gallery.html"
    _write_field_orientation_gallery(gallery, entries, parent)
    print(f"render_raylib: wrote orientation gallery to {gallery}")

    return status


def _write_field_orientation_gallery(gallery: Path, entries: list[tuple[str, Path]], base_dir: Path) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Field orientation sweep</title></head>",
        "<body style='font-family: monospace; background: #111; color: #eee;'>",
        "<h2>Field orientation sweep</h2>",
        "<div style='display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;'>",
    ]
    for orientation, output in entries:
        rel = output.relative_to(base_dir).as_posix()
        lines.append("<figure>")
        lines.append(f"<figcaption>{orientation}</figcaption>")
        lines.append(f"<img src='{rel}' width='560' />")
        lines.append("</figure>")
    lines.extend(["</div>", "</body>", "</html>"])
    gallery.write_text("\n".join(lines), encoding="utf-8")


def _draw_overlay(
    rl,
    make_color,
    samples: list[Sample],
    frame: float,
    geometry_name: str,
    field_orientation: str,
) -> None:
    # animate a marker across sweep using frame time
    best = _frame_to_sample(samples, frame)
    field_val = "na"
    if best.field:
        field_val = f"{best.field[len(best.field) // 2]:.5f}"
    rl.DrawText(
        _cstr(f"{geometry_name} native CPU | alpha={best.alpha:+.2f} deg | CL={best.cl:+.4f} | CD={best.cd:.5f} | CM={best.cm:+.5f}"),
        _PADDING,
        _WINDOW_H - 2 * _PADDING,
        18,
        make_color(240, 240, 120, 255),
    )
    if best.field:
        rl.DrawText(
            _cstr(f"{best.field_mode.title()} field center={field_val} | orient={field_orientation}"),
            _PADDING,
            _WINDOW_H - int(3.2 * _PADDING),
            14,
            make_color(220, 220, 255, 255),
        )
    rl.DrawText(_cstr(f"status: {best.status}"), _PADDING, _WINDOW_H - _PADDING, 16, make_color(220, 220, 220, 255))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render native CPU evaluations with RayLib.")
    parser.add_argument(
        "--geometry",
        default=None,
        help="Load NACA profile from a geometry file instead of generating from --naca.",
    )
    parser.add_argument(
        "--naca",
        default="2412",
        help="Generate NACA 4-digit geometry instead of loading --geometry file, e.g. 2412",
    )
    parser.add_argument("--alpha-start", type=float, default=2.0)
    parser.add_argument("--alpha-stop", type=float, default=8.0)
    parser.add_argument("--alpha-step", type=float, default=1.0)
    parser.add_argument("--re", dest="reynolds", type=float, default=1_000_000.0)
    parser.add_argument("--mach", type=float, default=0.0)
    parser.add_argument(
        "--field-mode",
        choices=_field_mode_choices() + ("off",),
        default="pressure",
        help="Field rendering mode: pressure (default), velocity, or off.",
    )
    parser.add_argument("--field-cols", type=int, default=560)
    parser.add_argument("--field-rows", type=int, default=360)
    parser.add_argument(
        "--field-orientation",
        choices=_FIELD_ORIENTATIONS,
        default="normal",
        help=(
            "How to rotate/flip the field map: normal, flip_x, flip_y, flip_xy, "
            "rot90_cw, rot90_ccw, rot180, transpose, transpose_rotate90_ccw, transpose_flip, "
            "screen_x_mirror_rot90_cw, rot90_cw_flip_x."
        ),
    )
    parser.add_argument(
        "--field-rect-mode",
        choices=("panel", "geometry"),
        default="panel",
        help="Map field to full panel rectangle; 'geometry' is retained for backward compatibility.",
    )
    parser.add_argument(
        "--field-domain-mode",
        choices=("bounds", "padded"),
        default="bounds",
        help="Use padded airfoil bounds ('bounds') or field-expanded box ('padded') for sampling grid.",
    )
    parser.add_argument(
        "--field-origin-mode",
        choices=("leading_edge", "bounds_center", "zero"),
        default="leading_edge",
        help="Reference origin for synthetic field synthesis: leading edge, bounds center, or x=LE/y=0.",
    )
    parser.add_argument(
        "--field-orientation-sweep",
        action="store_true",
        default=False,
        help="Render all field-orientation modes into separate files with suffix names.",
    )
    parser.add_argument("--field-strength", type=float, default=1.5)
    parser.add_argument("--field-core", type=float, default=0.08)
    parser.add_argument("--field-margin", type=float, default=1.8)
    parser.add_argument(
        "--field-margin-y",
        type=float,
        default=None,
        help="Y-axis field padding in percent while keeping X padding from --field-margin.",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--n-crit", type=int, default=9, dest="n_crit")
    parser.add_argument("--n-panels", type=int, default=None, dest="n_panels")
    parser.add_argument("--panel-fallback", type=int, default=80)
    parser.add_argument("--iterations-to-converge", type=int, default=120)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    parser.add_argument("--offscreen", action="store_true", default=False)
    parser.add_argument(
        "--strict-display",
        action="store_true",
        default=False,
        help="Exit if the display target is not available instead of falling back to offscreen rendering.",
    )
    parser.add_argument("--offscreen-output", default=_OUTPUT_DEFAULT)

    args = parser.parse_args()

    try:
        rl, make_color, close_window = _load_raylib()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.geometry is not None:
        geometry_path = _resolve_geometry(args.geometry)
        points = _read_geometry(geometry_path)
        args.geometry_name = Path(args.geometry).stem
    else:
        try:
            points = _naca_4points(args.naca)
            args.geometry_name = f"NACA-{args.naca}"
        except ValueError as exc:
            print(f"render_raylib: invalid --naca value: {exc}", file=sys.stderr)
            return 2
    samples = _evaluate_sweep(points, args)

    if args.field_orientation_sweep:
        args.offscreen = True

    if args.display:
        os.environ["DISPLAY"] = args.display
        display_available = _display_available(args.display)
        if display_available is False:
            msg = (
                f"DISPLAY '{args.display}' does not appear to be available. "
                "Falling back to offscreen output."
            )
            if args.strict_display:
                print(msg.replace("Falling back to offscreen output.", "Aborting."))
                return 1
            print(f"render_raylib: {msg}")
            args.offscreen = True
        elif display_available is None:
            print(f"render_raylib: could not verify DISPLAY '{args.display}'; attempting window render.")
    else:
        os.environ.setdefault("DISPLAY", ":0")

    if not args.offscreen:
        try:
            rl.InitWindow(_WINDOW_W, _WINDOW_H, b"XFOIL Native CPU RayLib Render")
        except Exception as exc:  # noqa: BLE001
            msg = (
                "raylib window init failed. Check DISPLAY and RayLib binary. "
                f"({type(exc).__name__}: {exc})"
            )
            if args.strict_display:
                print(f"render_raylib: {msg}")
                return 2
            print(f"render_raylib: {msg}")
            print("render_raylib: falling back to offscreen output")
            return _render_offscreen(rl, make_color, points, samples, args)

    if args.field_orientation_sweep:
        return _render_offscreen_orientation_sweep(rl, make_color, points, samples, args)

    if args.offscreen:
        return _render_offscreen(rl, make_color, points, samples, args)
    rl.SetTargetFPS(_FPS)

    bounds = _tight_bounds(points)
    geometry_bounds = _geometry_render_bounds(points)
    elapsed = 0.0

    while not _should_close_window(rl):
        frame = elapsed
        elapsed += rl.GetFrameTime()
        frame_sample = _frame_to_sample(samples, frame)

        rl.BeginDrawing()
        rl.ClearBackground(make_color(12, 12, 12, 255))

        _draw_geometry(
            rl,
            make_color,
            points,
            bounds,
            geometry_bounds,
            frame_sample,
            args.field_cols,
            args.field_rows,
            args.field_orientation,
            args.field_rect_mode,
        )
        _draw_chart(rl, make_color, samples)
        _draw_overlay(
            rl,
            make_color,
            samples,
            frame,
            args.geometry_name,
            args.field_orientation,
        )

        rl.EndDrawing()

    close_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
