"""Step-transition behavior for future RL-style usage."""

from __future__ import annotations

from xfoil_port import XfoilStepEnv
from xfoil_port.backends.cpu import CpuXfoilConfig
from xfoil_port.engine import XfoilEngine
from xfoil_port.types import XfoilQuery


def test_step_env_runs_deterministically_with_fallback():
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]
    query = XfoilQuery(alpha_deg=2.0, reynolds=1_000_000, mach=0.0)
    engine = XfoilEngine(
        config=CpuXfoilConfig(
            xfoil_executable="__definitely_missing__",
            enable_fallback=True,
            cache_results=False,
        )
    )
    env = XfoilStepEnv(engine=engine, max_steps=3)
    state = env.reset(
        geometry_points=geometry,
        query=query,
        seed=123,
    )

    state2, reward1, done1, info1 = env.c_step(state, {"alpha_delta": 1.0, "reynolds_delta": 0.0})
    state3, reward2, done2, info2 = env.c_step(
        state2,
        {"alpha_delta": -1.0, "reynolds_delta": 0.0, "mach_delta": 0.0},
    )

    assert state2.step_index == 1
    assert state3.step_index == 2
    assert state2.query.alpha_deg == 3.0
    assert state3.query.alpha_deg == 2.0
    assert info1["status"] == "fallback_missing_executable"
    assert info2["status"] == "fallback_missing_executable"
    assert reward1 >= -1.0 and reward1 <= 1.0
    assert reward2 >= -1.0 and reward2 <= 1.0
    assert done1 is False
    assert done2 is False


def test_step_env_marks_done_on_max_steps():
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]
    query = XfoilQuery(alpha_deg=2.0, reynolds=1_000_000, mach=0.0)
    engine = XfoilEngine(
        config=CpuXfoilConfig(
            xfoil_executable="__definitely_missing__",
            enable_fallback=True,
            cache_results=False,
        )
    )
    env = XfoilStepEnv(engine=engine, max_steps=1)
    state = env.reset(geometry_points=geometry, query=query)

    state2, reward, done, info = env.c_step(state, {"alpha_delta": 0.1})
    assert state2.step_index == 1
    assert done is True
    assert info["done"] is True


def test_step_env_marks_done_when_target_reached_with_tolerance():
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]
    query = XfoilQuery(alpha_deg=5.0, reynolds=1_000_000, mach=0.0)
    engine = XfoilEngine(
        config=CpuXfoilConfig(
            xfoil_executable="__definitely_missing__",
            enable_fallback=True,
            cache_results=False,
        )
    )
    env = XfoilStepEnv(engine=engine, max_steps=3)
    target_state = env.reset(
        geometry_points=geometry,
        query=query,
        target_cl=0.5520529123941558,
        target_cd=0.004050015811388301,
    )

    _, _, done, _ = env.c_step(target_state, {"alpha_delta": 0.0, "reynolds_delta": 0.0})
    assert done is True


def test_step_env_default_uses_fallback_without_custom_engine():
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]
    query = XfoilQuery(alpha_deg=4.0, reynolds=1_000_000, mach=0.0)
    env = XfoilStepEnv()
    state = env.reset(geometry_points=geometry, query=query, seed=42)

    state2, reward, done, info = env.c_step(state, {"alpha_delta": 0.0, "reynolds_delta": 0.0})

    assert info["status"] == "fallback_missing_executable"
    assert done is False
    assert reward <= 0.0
    assert isinstance(info["result"]["cl"], float)
