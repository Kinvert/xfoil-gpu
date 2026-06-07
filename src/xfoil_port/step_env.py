"""CPU-only step-transition wrapper for future RL-style environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .engine import XfoilEngine
from .backends.cpu import CpuXfoilConfig
from .types import XfoilQuery, XfoilResult


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _bounded_reward(value: float) -> float:
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return value


@dataclass(frozen=True)
class XfoilStepState:
    """Deterministic state container for one environment step."""

    geometry_points: tuple[tuple[float, float], ...]
    query: XfoilQuery
    step_index: int
    last_result: Optional[XfoilResult]
    seed: int
    target_cl: Optional[float]
    target_cd: Optional[float]
    done: bool = False

    @classmethod
    def from_components(
        cls,
        geometry_points: Sequence[tuple[float, float]],
        query: XfoilQuery,
        *,
        seed: int = 0,
        target_cl: Optional[float] = None,
        target_cd: Optional[float] = None,
    ) -> "XfoilStepState":
        return cls(
            geometry_points=tuple(map(tuple, geometry_points)),
            query=query,
            step_index=0,
            last_result=None,
            seed=seed,
            target_cl=target_cl,
            target_cd=target_cd,
            done=False,
        )


class XfoilStepEnv:
    """Simple `c_step`-style adapter around `XfoilEngine`.

    The interface intentionally mirrors a PufferLib-style transition:
    `(state, action) -> (next_state, reward, done, info)`.
    """

    def __init__(
        self,
        engine: Optional[XfoilEngine] = None,
        *,
        max_steps: int = 50,
        enable_fallback: bool = True,
        cl_tolerance: float = 1e-3,
        cd_tolerance: float = 1e-4,
        alpha_clip: Tuple[float, float] = (-90.0, 90.0),
        reynolds_clip: Tuple[float, float] = (1.0, 1e9),
        mach_clip: Tuple[float, float] = (0.0, 2.0),
    ):
        self.engine = engine or XfoilEngine(config=CpuXfoilConfig(enable_fallback=enable_fallback))
        self.max_steps = max_steps
        self.cl_tolerance = abs(cl_tolerance)
        self.cd_tolerance = abs(cd_tolerance)
        self.alpha_clip = alpha_clip
        self.reynolds_clip = reynolds_clip
        self.mach_clip = mach_clip

    def reset(
        self,
        geometry_points: Sequence[tuple[float, float]],
        query: XfoilQuery,
        *,
        seed: int = 0,
        target_cl: Optional[float] = None,
        target_cd: Optional[float] = None,
    ) -> XfoilStepState:
        return XfoilStepState.from_components(
            geometry_points=geometry_points,
            query=query,
            seed=seed,
            target_cl=target_cl,
            target_cd=target_cd,
        )

    def c_step(
        self,
        state: XfoilStepState,
        action: Dict[str, float],
    ) -> tuple[XfoilStepState, float, bool, Dict[str, float | str | bool | int | dict]]:
        if state.done:
            return (
                state,
                0.0,
                True,
                {
                    "status": "already_done",
                    "reward": 0.0,
                    "step_index": state.step_index,
                    "done": True,
                },
            )

        alpha_delta = float(action.get("alpha_delta", 0.0))
        reynolds_delta = float(action.get("reynolds_delta", 0.0))
        mach_delta = float(action.get("mach_delta", 0.0))

        query = state.query
        next_query = XfoilQuery(
            alpha_deg=_clamp(
                query.alpha_deg + alpha_delta,
                self.alpha_clip[0],
                self.alpha_clip[1],
            ),
            reynolds=_clamp(
                query.reynolds * (1.0 + reynolds_delta),
                self.reynolds_clip[0],
                self.reynolds_clip[1],
            ),
            mach=_clamp(
                query.mach + mach_delta,
                self.mach_clip[0],
                self.mach_clip[1],
            ),
            n_crit=int(action.get("n_crit", query.n_crit)),
            iterations=int(action.get("iterations", query.iterations)),
            n_panels=action.get("n_panels", query.n_panels),
        )

        result = self.engine.evaluate(state.geometry_points, next_query, name=f"step_{state.step_index + 1}")

        reward = self._compute_reward(
            result,
            target_cl=state.target_cl,
            target_cd=state.target_cd,
            fallback=result.status.startswith("fallback"),
        )

        done = self._is_done(
            result=result,
            step_index=state.step_index + 1,
            target_cl=state.target_cl,
            target_cd=state.target_cd,
        )

        next_state = XfoilStepState(
            geometry_points=state.geometry_points,
            query=next_query,
            step_index=state.step_index + 1,
            last_result=result,
            seed=state.seed,
            target_cl=state.target_cl,
            target_cd=state.target_cd,
            done=done,
        )

        info = {
            "status": result.status,
            "step_index": next_state.step_index,
            "reward": reward,
            "done": done,
            "query": {
                "alpha_deg": next_query.alpha_deg,
                "reynolds": next_query.reynolds,
                "mach": next_query.mach,
                "n_crit": next_query.n_crit,
                "iterations": next_query.iterations,
                "n_panels": next_query.n_panels,
            },
            "meta": dict(result.meta),
            "result": {
                "cl": result.cl,
                "cd": result.cd,
                "cm": result.cm,
            },
        }

        return next_state, reward, done, info

    def _is_done(
        self,
        *,
        result: XfoilResult,
        step_index: int,
        target_cl: Optional[float],
        target_cd: Optional[float],
    ) -> bool:
        if step_index >= self.max_steps:
            return True
        if target_cl is not None and target_cd is not None:
            if result.cl is not None and result.cd is not None:
                cl_close = abs(result.cl - target_cl) <= self.cl_tolerance
                cd_close = abs(result.cd - target_cd) <= self.cd_tolerance
                if cl_close and cd_close:
                    return True
        return False

    def _compute_reward(
        self,
        result: XfoilResult,
        *,
        target_cl: Optional[float],
        target_cd: Optional[float],
        fallback: bool,
    ) -> float:
        if target_cl is None and target_cd is None:
            # Non-goal shaping keeps policy updates bounded and stable.
            base_reward = -0.0
        else:
            cl_reward = 0.0
            cd_reward = 0.0
            if target_cl is not None and result.cl is not None:
                scale = max(1.0, abs(target_cl))
                cl_reward = 1.0 - min(abs(result.cl - target_cl) / scale, 1.0)
            if target_cd is not None and result.cd is not None:
                scale = max(1.0, abs(target_cd))
                cd_reward = 1.0 - min(abs(result.cd - target_cd) / scale, 1.0)

            weighted = 0.0
            n = 0
            if target_cl is not None:
                weighted += cl_reward
                n += 1
            if target_cd is not None:
                weighted += cd_reward
                n += 1
            if n == 0:
                base_reward = 0.0
            else:
                base_reward = weighted / n

        if fallback:
            base_reward -= 0.25

        return _bounded_reward(base_reward)
