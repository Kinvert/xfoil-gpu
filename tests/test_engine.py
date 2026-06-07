from xfoil_port.engine import XfoilEngine
from xfoil_port.types import XfoilBatchInput, XfoilQuery
from xfoil_port.backends.cpu import CpuXfoilConfig
from xfoil_port.backends.native_cpu import NativeCpuXfoilConfig
from xfoil_port.types import BackendResult, XfoilResult


def test_engine_batch_returns_ordered_results_with_fallback():
    engine = XfoilEngine(
        config=CpuXfoilConfig(
            xfoil_executable="__definitely_missing__",
            enable_fallback=True,
            cache_results=False,
        )
    )
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.01), (0.0, 0.0)]
    items = [
        XfoilBatchInput(
            name=f"case-{i}",
            geometry_points=geometry,
            query=XfoilQuery(alpha_deg=5.0 + i, reynolds=1_000_000 + i * 100_000),
        )
        for i in range(3)
    ]

    results = engine.evaluate_many(items)

    assert [r.status for r in results] == [
        "fallback_missing_executable",
        "fallback_missing_executable",
        "fallback_missing_executable",
    ]
    assert [r.meta["source"] for r in results] == [
        "fallback_heuristic",
        "fallback_heuristic",
        "fallback_heuristic",
    ]


class _StubBackend:
    backend_id = "stub-gpu"

    def evaluate(self, geometry_points, query, *, name: str = "airfoil") -> BackendResult:
        return BackendResult(
            ok=True,
            raw_stdout="",
            raw_stderr="",
            payload=XfoilResult(
                alpha_deg=query.alpha_deg,
                reynolds=query.reynolds,
                mach=query.mach,
                cl=0.0,
                cd=0.0,
                cm=0.0,
                status="ok",
                meta={"source": "stub"},
            ),
        )


def test_engine_accepts_backend_instance():
    engine = XfoilEngine(backend=_StubBackend(), config=None)
    result = engine.evaluate([(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)], XfoilQuery(alpha_deg=2.0, reynolds=1_000_000))

    assert result.status == "ok"
    assert result.meta["backend"] == "stub-gpu"


def test_engine_accepts_native_backend_name():
    engine = XfoilEngine(
        backend="cpu-native",
        config=NativeCpuXfoilConfig(cache_results=False),
    )
    geometry = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.02), (0.0, 0.0)]
    result = engine.evaluate(geometry, XfoilQuery(alpha_deg=5.0, reynolds=1_000_000))

    assert result.status in {"ok", "native_warnings"}
    assert result.meta["backend"] == "cpu_native"
    assert result.meta["source"] in {"native_cpu_approx", "native_cpu_approx_cpp"}
    assert result.meta["query"]["alpha_deg"] == "5"
