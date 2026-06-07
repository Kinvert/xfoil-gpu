"""Error classes for deterministic failure handling."""


class XfoilError(RuntimeError):
    """Raised when a step cannot complete with an authoritative result."""


class XfoilDependencyError(XfoilError):
    """Raised when no XFOIL executable or backend toolchain is available."""


class XfoilGeometryError(XfoilError):
    """Raised when geometry validation fails."""


class XfoilRuntimeError(XfoilError):
    """Raised when the solver execution fails or times out."""
