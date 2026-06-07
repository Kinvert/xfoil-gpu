"""Kernel-stage tests for the native CPU approximation path."""

from xfoil_port.backends.native_cpu_kernels import geometry_features, estimate_aero


def test_geometry_features_are_deterministic():
    geometry = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.02),
        (0.0, 0.0),
    ]
    first = geometry_features(geometry)
    second = geometry_features(list(reversed(geometry)))
    assert first["n"] == second["n"]
    assert first["thickness"] == second["thickness"]
    assert first["n"] == 4.0


def test_estimate_aero_returns_stable_fields():
    geometry = [
        (0.0, 0.0),
        (0.5, 0.02),
        (1.0, 0.0),
        (0.0, 0.0),
    ]
    first = estimate_aero(
        geometry_points=geometry,
        query_alpha_deg=5.0,
        query_reynolds=1_000_000,
        query_mach=0.0,
        n_panels=80,
        iterations_requested=200,
        stall_alpha_deg=18.0,
        residual_floor=1e-10,
        iterations_to_converge=120,
    )
    second = estimate_aero(
        geometry_points=geometry,
        query_alpha_deg=5.0,
        query_reynolds=1_000_000,
        query_mach=0.0,
        n_panels=80,
        iterations_requested=200,
        stall_alpha_deg=18.0,
        residual_floor=1e-10,
        iterations_to_converge=120,
    )

    assert first == second
    assert first["status"] in {"ok", "native_warnings"}
    assert first["cl"] > 0
    assert 0.004 < first["cd"] < 0.05
