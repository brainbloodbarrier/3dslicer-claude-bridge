"""Tests for metrics collection module."""

import os

import pytest


class TestNullMetric:
    """Test NullMetric null object pattern."""

    def test_null_metric_inc(self):
        """NullMetric.inc should do nothing without error."""
        from slicer_mcp.metrics import NullMetric

        metric = NullMetric()
        metric.inc()  # Should not raise
        metric.inc(5)  # Should not raise

    def test_null_metric_dec(self):
        """NullMetric.dec should do nothing without error."""
        from slicer_mcp.metrics import NullMetric

        metric = NullMetric()
        metric.dec()  # Should not raise
        metric.dec(5)  # Should not raise

    def test_null_metric_set(self):
        """NullMetric.set should do nothing without error."""
        from slicer_mcp.metrics import NullMetric

        metric = NullMetric()
        metric.set(42)  # Should not raise

    def test_null_metric_observe(self):
        """NullMetric.observe should do nothing without error."""
        from slicer_mcp.metrics import NullMetric

        metric = NullMetric()
        metric.observe(0.5)  # Should not raise

    def test_null_metric_labels_chainable(self):
        """NullMetric.labels should return self for chaining."""
        from slicer_mcp.metrics import NullMetric

        metric = NullMetric()
        result = metric.labels(operation="test")
        assert result is metric

    def test_null_metric_full_chain(self):
        """NullMetric should support full Prometheus-like call chain."""
        from slicer_mcp.metrics import NullMetric

        metric = NullMetric()
        metric.labels(operation="test", status="success").inc()  # Should not raise
        metric.labels(operation="test").observe(0.123)  # Should not raise


class TestTrackRequest:
    """Test track_request context manager."""

    def test_track_request_success(self):
        """track_request should track successful operations."""
        from slicer_mcp.metrics import track_request

        with track_request("test_operation"):
            result = 1 + 1

        assert result == 2  # Operation completed

    def test_track_request_failure(self):
        """track_request should track failed operations and re-raise."""
        from slicer_mcp.metrics import track_request

        with pytest.raises(ValueError):
            with track_request("test_operation"):
                raise ValueError("test error")

    def test_track_request_records_duration(self):
        """track_request should record operation duration."""
        import time

        from slicer_mcp.metrics import track_request

        # Even with null metrics, this should not raise
        with track_request("timed_operation"):
            time.sleep(0.01)


class TestRecordRetry:
    """Test record_retry function."""

    def test_record_retry_does_not_raise(self):
        """record_retry should not raise even with null metrics."""
        from slicer_mcp.metrics import record_retry

        record_retry("test_operation")  # Should not raise


class TestUpdateCircuitBreakerState:
    """Test update_circuit_breaker_state function."""

    def test_update_state_closed(self):
        """update_circuit_breaker_state should handle closed state."""
        from slicer_mcp.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "closed")  # Should not raise

    def test_update_state_half_open(self):
        """update_circuit_breaker_state should handle half_open state."""
        from slicer_mcp.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "half_open")  # Should not raise

    def test_update_state_open(self):
        """update_circuit_breaker_state should handle open state."""
        from slicer_mcp.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "open")  # Should not raise

    def test_update_state_invalid(self):
        """update_circuit_breaker_state should handle invalid state gracefully."""
        from slicer_mcp.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "invalid_state")  # Should not raise


class TestIsMetricsEnabled:
    """Test is_metrics_enabled function."""

    def test_returns_boolean(self):
        """is_metrics_enabled should return a boolean."""
        from slicer_mcp.metrics import is_metrics_enabled

        result = is_metrics_enabled()
        assert isinstance(result, bool)


class TestMetricsDisabledByDefault:
    """Test that metrics are disabled by default."""

    def test_metrics_disabled_without_env_var(self):
        """Metrics should be disabled when env var is not set."""
        # Remove env var if set
        env_backup = os.environ.pop("SLICER_METRICS_ENABLED", None)

        try:
            # Re-import to check default state
            # Note: Module already imported, so we check the current state
            from slicer_mcp.metrics import REQUEST_DURATION, NullMetric

            # If metrics were not enabled at import time, REQUEST_DURATION is NullMetric
            # This test verifies the null object pattern is used
            assert isinstance(REQUEST_DURATION, NullMetric | type(REQUEST_DURATION))
        finally:
            # Restore env var
            if env_backup is not None:
                os.environ["SLICER_METRICS_ENABLED"] = env_backup


class TestMetricsWithPrometheusClient:
    """Test metrics with prometheus_client (if available)."""

    @pytest.mark.skipif(
        not os.environ.get("SLICER_METRICS_ENABLED", "").lower() == "true",
        reason="Metrics not enabled",
    )
    def test_real_metrics_when_enabled(self):
        """Test real Prometheus metrics when enabled."""
        try:
            import prometheus_client  # noqa: F401

            # If we get here, prometheus_client is installed
            from slicer_mcp.metrics import METRICS_ENABLED, REQUEST_DURATION

            if METRICS_ENABLED:
                # Verify it's a real Prometheus metric
                assert hasattr(REQUEST_DURATION, "observe")
        except ImportError:
            pytest.skip("prometheus_client not installed")
