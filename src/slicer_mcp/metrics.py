"""Optional metrics collection for operational visibility.

Provides Prometheus-compatible metrics for monitoring the Slicer MCP Bridge.
Metrics are disabled by default and can be enabled via environment variable.

Usage:
    # Enable metrics:
    export SLICER_METRICS_ENABLED=true

    # In code:
    from slicer_mcp.metrics import track_request, REQUEST_TOTAL

    with track_request("health_check"):
        result = client.health_check()

Metrics exported:
    - slicer_request_duration_seconds: Histogram of request durations
    - slicer_request_total: Counter of total requests by operation and status
    - slicer_retry_total: Counter of retry attempts
    - slicer_circuit_breaker_state: Gauge of circuit breaker state
"""

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger("slicer-mcp")

# Check if metrics are enabled via environment variable
METRICS_ENABLED = os.environ.get("SLICER_METRICS_ENABLED", "").lower() == "true"


class NullMetric:
    """Null object pattern for metrics when disabled.

    Provides the same interface as Prometheus metrics but does nothing.
    This allows code to call metric methods without checking if metrics are enabled.
    """

    def labels(self, *args, **kwargs) -> "NullMetric":
        """Return self for chaining."""
        return self

    def inc(self, amount: float = 1) -> None:
        """No-op increment."""
        pass

    def dec(self, amount: float = 1) -> None:
        """No-op decrement."""
        pass

    def set(self, value: float) -> None:
        """No-op set."""
        pass

    def observe(self, value: float) -> None:
        """No-op observe."""
        pass


# Initialize metrics based on environment
if METRICS_ENABLED:
    try:
        from prometheus_client import Counter, Gauge, Histogram

        logger.info("Metrics collection enabled (prometheus_client)")

        REQUEST_DURATION = Histogram(
            "slicer_request_duration_seconds",
            "Request duration in seconds",
            ["operation"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        REQUEST_TOTAL = Counter("slicer_request_total", "Total requests", ["operation", "status"])

        RETRY_TOTAL = Counter("slicer_retry_total", "Total retry attempts", ["operation"])

        CIRCUIT_BREAKER_STATE = Gauge(
            "slicer_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            ["breaker_name"],
        )

        # Map circuit states to numeric values for Gauge
        CIRCUIT_STATE_VALUES = {
            "closed": 0,
            "half_open": 1,
            "open": 2,
        }

    except ImportError:
        logger.warning(
            "SLICER_METRICS_ENABLED=true but prometheus_client not installed. "
            "Install with: pip install prometheus_client"
        )
        METRICS_ENABLED = False
        REQUEST_DURATION = NullMetric()
        REQUEST_TOTAL = NullMetric()
        RETRY_TOTAL = NullMetric()
        CIRCUIT_BREAKER_STATE = NullMetric()
        CIRCUIT_STATE_VALUES = {}

else:
    # Metrics disabled - use null objects
    REQUEST_DURATION = NullMetric()
    REQUEST_TOTAL = NullMetric()
    RETRY_TOTAL = NullMetric()
    CIRCUIT_BREAKER_STATE = NullMetric()
    CIRCUIT_STATE_VALUES = {}


@contextmanager
def track_request(operation: str) -> Generator[None, None, None]:
    """Context manager to track request duration and status.

    Records the duration of the operation and whether it succeeded or failed.
    Uses Prometheus histogram for duration and counter for total requests.

    Args:
        operation: Name of the operation being tracked (e.g., "health_check")

    Yields:
        None

    Example:
        with track_request("screenshot"):
            result = client.get_screenshot("Red")
    """
    start_time = time.perf_counter()
    status = "success"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start_time
        REQUEST_DURATION.labels(operation=operation).observe(duration)
        REQUEST_TOTAL.labels(operation=operation, status=status).inc()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Metric: {operation} {status} in {duration:.3f}s")


def record_retry(operation: str) -> None:
    """Record a retry attempt for an operation.

    Args:
        operation: Name of the operation being retried
    """
    RETRY_TOTAL.labels(operation=operation).inc()


def update_circuit_breaker_state(breaker_name: str, state: str) -> None:
    """Update circuit breaker state metric.

    Args:
        breaker_name: Name of the circuit breaker
        state: Current state ("closed", "half_open", "open")
    """
    if state in CIRCUIT_STATE_VALUES:
        CIRCUIT_BREAKER_STATE.labels(breaker_name=breaker_name).set(CIRCUIT_STATE_VALUES[state])


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled.

    Returns:
        True if metrics are enabled and prometheus_client is available
    """
    return METRICS_ENABLED
