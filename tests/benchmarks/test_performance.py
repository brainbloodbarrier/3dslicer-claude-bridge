"""Performance benchmark tests for Slicer MCP Bridge.

These tests measure actual latencies for common operations when connected
to a running Slicer instance. They are marked as integration tests and
skipped when Slicer is not available.

Run benchmarks:
    pytest tests/benchmarks/ -v --benchmark-only

Or with timing output:
    pytest tests/benchmarks/ -v -s --durations=0
"""

import statistics
import time
from collections.abc import Callable
from typing import Any

import pytest

from slicer_mcp.slicer_client import (
    SlicerClient,
    SlicerConnectionError,
    reset_circuit_breaker,
)


def measure_latency(func: Callable[[], Any], iterations: int = 10) -> dict:
    """Measure latency statistics for a function.

    Args:
        func: Function to measure
        iterations: Number of iterations to run

    Returns:
        Dict with avg, min, max, p95, p99 latencies in milliseconds
    """
    latencies: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    p95_idx = int(len(latencies) * 0.95)
    p99_idx = int(len(latencies) * 0.99)

    return {
        "iterations": iterations,
        "avg_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1],
        "p99_ms": latencies[p99_idx] if p99_idx < len(latencies) else latencies[-1],
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }


@pytest.fixture
def live_client():
    """Create a client and verify Slicer is running."""
    reset_circuit_breaker()
    client = SlicerClient()

    try:
        health = client.health_check(check_version=False)
        if not health.get("connected"):
            pytest.skip("Slicer not connected")
    except SlicerConnectionError:
        pytest.skip("Slicer not running or WebServer not enabled")

    return client


@pytest.mark.integration
@pytest.mark.benchmark
class TestHealthCheckBenchmark:
    """Benchmark health check operation."""

    def test_health_check_latency(self, live_client):
        """Measure health check latency (target: <100ms avg)."""
        reset_circuit_breaker()

        results = measure_latency(
            lambda: live_client.health_check(check_version=False), iterations=20
        )

        print("\n=== Health Check Latency ===")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Average:    {results['avg_ms']:.2f}ms")
        print(f"  Median:     {results['median_ms']:.2f}ms")
        print(f"  Min:        {results['min_ms']:.2f}ms")
        print(f"  Max:        {results['max_ms']:.2f}ms")
        print(f"  P95:        {results['p95_ms']:.2f}ms")
        print(f"  Stdev:      {results['stdev_ms']:.2f}ms")

        # Soft assertion - warn but don't fail
        if results["avg_ms"] > 100:
            print("  WARNING: Average latency exceeds 100ms target")

        assert results["avg_ms"] < 500, "Health check should complete within 500ms"


@pytest.mark.integration
@pytest.mark.benchmark
class TestScreenshotBenchmark:
    """Benchmark screenshot capture operations."""

    def test_slice_screenshot_latency(self, live_client):
        """Measure 2D slice screenshot latency (target: <200ms avg)."""
        reset_circuit_breaker()

        results = measure_latency(lambda: live_client.get_screenshot("Red"), iterations=10)

        print("\n=== 2D Screenshot Latency (Red/Axial) ===")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Average:    {results['avg_ms']:.2f}ms")
        print(f"  Median:     {results['median_ms']:.2f}ms")
        print(f"  Min:        {results['min_ms']:.2f}ms")
        print(f"  Max:        {results['max_ms']:.2f}ms")
        print(f"  P95:        {results['p95_ms']:.2f}ms")

        assert results["avg_ms"] < 1000, "Screenshot should complete within 1s"

    def test_3d_screenshot_latency(self, live_client):
        """Measure 3D view screenshot latency (target: <300ms avg)."""
        reset_circuit_breaker()

        results = measure_latency(lambda: live_client.get_3d_screenshot(), iterations=10)

        print("\n=== 3D Screenshot Latency ===")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Average:    {results['avg_ms']:.2f}ms")
        print(f"  Median:     {results['median_ms']:.2f}ms")
        print(f"  Min:        {results['min_ms']:.2f}ms")
        print(f"  Max:        {results['max_ms']:.2f}ms")
        print(f"  P95:        {results['p95_ms']:.2f}ms")

        assert results["avg_ms"] < 2000, "3D screenshot should complete within 2s"

    def test_full_screenshot_latency(self, live_client):
        """Measure full window screenshot latency (target: <500ms avg)."""
        reset_circuit_breaker()

        results = measure_latency(lambda: live_client.get_full_screenshot(), iterations=10)

        print("\n=== Full Screenshot Latency ===")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Average:    {results['avg_ms']:.2f}ms")
        print(f"  Median:     {results['median_ms']:.2f}ms")
        print(f"  Min:        {results['min_ms']:.2f}ms")
        print(f"  Max:        {results['max_ms']:.2f}ms")
        print(f"  P95:        {results['p95_ms']:.2f}ms")

        assert results["avg_ms"] < 3000, "Full screenshot should complete within 3s"


@pytest.mark.integration
@pytest.mark.benchmark
class TestPythonExecutionBenchmark:
    """Benchmark Python code execution."""

    def test_simple_expression_latency(self, live_client):
        """Measure simple Python expression latency (target: <50ms avg)."""
        reset_circuit_breaker()

        results = measure_latency(lambda: live_client.exec_python("1 + 1"), iterations=20)

        print("\n=== Simple Expression Latency ===")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Average:    {results['avg_ms']:.2f}ms")
        print(f"  Median:     {results['median_ms']:.2f}ms")
        print(f"  Min:        {results['min_ms']:.2f}ms")
        print(f"  Max:        {results['max_ms']:.2f}ms")
        print(f"  P95:        {results['p95_ms']:.2f}ms")

        assert results["avg_ms"] < 500, "Simple expression should complete within 500ms"

    def test_slicer_api_call_latency(self, live_client):
        """Measure Slicer API call latency (target: <100ms avg)."""
        reset_circuit_breaker()

        results = measure_latency(
            lambda: live_client.exec_python(
                "import slicer\n__execResult = slicer.mrmlScene.GetNodesByClass('vtkMRMLVolumeNode').GetNumberOfItems()"
            ),
            iterations=20,
        )

        print("\n=== Slicer API Call Latency ===")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Average:    {results['avg_ms']:.2f}ms")
        print(f"  Median:     {results['median_ms']:.2f}ms")
        print(f"  Min:        {results['min_ms']:.2f}ms")
        print(f"  Max:        {results['max_ms']:.2f}ms")
        print(f"  P95:        {results['p95_ms']:.2f}ms")

        assert results["avg_ms"] < 1000, "API call should complete within 1s"


@pytest.mark.integration
@pytest.mark.benchmark
class TestSceneOperationsBenchmark:
    """Benchmark scene operations."""

    def test_get_scene_nodes_latency(self, live_client):
        """Measure scene node listing latency (target: <200ms avg)."""
        reset_circuit_breaker()

        results = measure_latency(lambda: live_client.get_scene_nodes(), iterations=10)

        print("\n=== Scene Nodes Listing Latency ===")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Average:    {results['avg_ms']:.2f}ms")
        print(f"  Median:     {results['median_ms']:.2f}ms")
        print(f"  Min:        {results['min_ms']:.2f}ms")
        print(f"  Max:        {results['max_ms']:.2f}ms")
        print(f"  P95:        {results['p95_ms']:.2f}ms")

        assert results["avg_ms"] < 1000, "Scene listing should complete within 1s"
