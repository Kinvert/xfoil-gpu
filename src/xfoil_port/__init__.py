"""XFOIL evaluation package.

Public exports for the CPU-first implementation and backend interface.
"""

from .backends.cpu import CpuXfoilConfig, CpuXfoilEvaluator
from .backends.native_cpu import NativeCpuXfoilConfig, NativeCpuXfoilEvaluator
from .engine import XfoilEngine
from .types import XfoilBackend, XfoilBatchInput, XfoilQuery, XfoilResult
from .step_env import XfoilStepEnv, XfoilStepState
from .errors import XfoilError

__all__ = [
    "CpuXfoilConfig",
    "CpuXfoilEvaluator",
    "XfoilEngine",
    "NativeCpuXfoilConfig",
    "NativeCpuXfoilEvaluator",
    "XfoilBackend",
    "XfoilBatchInput",
    "XfoilResult",
    "XfoilQuery",
    "XfoilError",
    "XfoilStepEnv",
    "XfoilStepState",
]
