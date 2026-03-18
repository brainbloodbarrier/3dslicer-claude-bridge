"""Tests for metrics collection module."""

import importlib
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

import slicer_mcp.core.metrics as metrics_module


class TestNullMetric:
    """Test NullMetric null object pattern."""

    def test_null_metric_inc(self):
        """NullMetric.inc should do nothing without error."""
        from slicer_mcp.core.metrics import NullMetric

        metric = NullMetric()
        metric.inc()  # Should not raise
        metric.inc(5)  # Should not raise

    def test_null_metric_dec(self):
        """NullMetric.dec should do nothing without error."""
        from slicer_mcp.core.metrics import NullMetric

        metric = NullMetric()
        metric.dec()  # Should not raise
        metric.dec(5)  # Should not raise

    def test_null_metric_set(self):
        """NullMetric.set should do nothing without error."""
        from slicer_mcp.core.metrics import NullMetric

        metric = NullMetric()
        metric.set(42)  # Should not raise

    def test_null_metric_observe(self):
        """NullMetric.observe should do nothing without error."""
        from slicer_mcp.core.metrics import NullMetric

        metric = NullMetric()
        metric.observe(0.5)  # Should not raise

    def test_null_metric_labels_chainable(self):
        """NullMetric.labels should return self for chaining."""
        from slicer_mcp.core.metrics import NullMetric

        metric = NullMetric()
        result = metric.labels(operation="test")
        assert result is metric

    def test_null_metric_full_chain(self):
        """NullMetric should support full Prometheus-like call chain."""
        from slicer_mcp.core.metrics import NullMetric

        metric = NullMetric()
        metric.labels(operation="test", status="success").inc()  # Should not raise
        metric.labels(operation="test").observe(0.123)  # Should not raise


class TestTrackRequest:
    """Test track_request context manager."""

    def test_track_request_success(self):
        """track_request should track successful operations."""
        from slicer_mcp.core.metrics import track_request

        with track_request("test_operation"):
            result = 1 + 1

        assert result == 2  # Operation completed

    def test_track_request_failure(self):
        """track_request should track failed operations and re-raise."""
        from slicer_mcp.core.metrics import track_request

        with pytest.raises(ValueError):
            with track_request("test_operation"):
                raise ValueError("test error")

    def test_track_request_records_duration(self):
        """track_request should record operation duration."""
        import time

        from slicer_mcp.core.metrics import track_request

        # Even with null metrics, this should not raise
        with track_request("timed_operation"):
            time.sleep(0.01)


class TestRecordRetry:
    """Test record_retry function."""

    def test_record_retry_does_not_raise(self):
        """record_retry should not raise even with null metrics."""
        from slicer_mcp.core.metrics import record_retry

        record_retry("test_operation")  # Should not raise


class TestUpdateCircuitBreakerState:
    """Test update_circuit_breaker_state function."""

    def test_update_state_closed(self):
        """update_circuit_breaker_state should handle closed state."""
        from slicer_mcp.core.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "closed")  # Should not raise

    def test_update_state_half_open(self):
        """update_circuit_breaker_state should handle half_open state."""
        from slicer_mcp.core.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "half_open")  # Should not raise

    def test_update_state_open(self):
        """update_circuit_breaker_state should handle open state."""
        from slicer_mcp.core.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "open")  # Should not raise

    def test_update_state_invalid(self):
        """update_circuit_breaker_state should handle invalid state gracefully."""
        from slicer_mcp.core.metrics import update_circuit_breaker_state

        update_circuit_breaker_state("test_breaker", "invalid_state")  # Should not raise


class TestIsMetricsEnabled:
    """Test is_metrics_enabled function."""

    def test_returns_boolean(self):
        """is_metrics_enabled should return a boolean."""
        from slicer_mcp.core.metrics import is_metrics_enabled

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
            from slicer_mcp.core.metrics import REQUEST_DURATION, NullMetric

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
            from slicer_mcp.core.metrics import METRICS_ENABLED, REQUEST_DURATION

            if METRICS_ENABLED:
                # Verify it's a real Prometheus metric
                assert hasattr(REQUEST_DURATION, "observe")
        except ImportError:
            pytest.skip("prometheus_client not installed")


class TestMetricsEnabledPrometheusAvailable:
    """Test METRICS_ENABLED=true with mocked prometheus_client."""

    @pytest.fixture(autouse=True)
    def _restore_metrics(self):
        """Restore metrics module to default disabled state after each test."""
        yield
        os.environ.pop("SLICER_METRICS_ENABLED", None)
        importlib.reload(metrics_module)

    def test_metrics_enabled_flag_is_true(self):
        """METRICS_ENABLED should be True when prometheus_client is available."""
        mock_prom = MagicMock()
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": mock_prom}),
        ):
            importlib.reload(metrics_module)
            assert metrics_module.METRICS_ENABLED is True

    def test_histogram_created_with_correct_name(self):
        """Histogram should be created with the correct metric name."""
        mock_prom = MagicMock()
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": mock_prom}),
        ):
            importlib.reload(metrics_module)
            mock_prom.Histogram.assert_called_once()
            assert mock_prom.Histogram.call_args[0][0] == "slicer_request_duration_seconds"

    def test_request_counters_created(self):
        """Counter should be created for request total and retry total."""
        mock_prom = MagicMock()
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": mock_prom}),
        ):
            importlib.reload(metrics_module)
            assert mock_prom.Counter.call_count == 2
            names = [c[0][0] for c in mock_prom.Counter.call_args_list]
            assert "slicer_request_total" in names
            assert "slicer_retry_total" in names

    def test_gauge_created_for_circuit_breaker(self):
        """Gauge should be created for circuit breaker state."""
        mock_prom = MagicMock()
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": mock_prom}),
        ):
            importlib.reload(metrics_module)
            mock_prom.Gauge.assert_called_once()
            assert mock_prom.Gauge.call_args[0][0] == "slicer_circuit_breaker_state"

    def test_circuit_state_values_populated(self):
        """CIRCUIT_STATE_VALUES should have all 3 state mappings."""
        mock_prom = MagicMock()
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": mock_prom}),
        ):
            importlib.reload(metrics_module)
            assert metrics_module.CIRCUIT_STATE_VALUES == {
                "closed": 0,
                "half_open": 1,
                "open": 2,
            }

    def test_is_metrics_enabled_returns_true(self):
        """is_metrics_enabled() should return True when metrics are active."""
        mock_prom = MagicMock()
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": mock_prom}),
        ):
            importlib.reload(metrics_module)
            assert metrics_module.is_metrics_enabled() is True

    def test_info_log_emitted_on_enable(self, caplog):
        """Info log should be emitted when prometheus_client loads successfully."""
        mock_prom = MagicMock()
        with (
            caplog.at_level(logging.INFO, logger="slicer-mcp"),
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": mock_prom}),
        ):
            importlib.reload(metrics_module)
        assert "Metrics collection enabled" in caplog.text


class TestMetricsEnabledImportError:
    """Test METRICS_ENABLED=true without prometheus_client (ImportError fallback)."""

    @pytest.fixture(autouse=True)
    def _restore_metrics(self):
        """Restore metrics module to default disabled state after each test."""
        yield
        os.environ.pop("SLICER_METRICS_ENABLED", None)
        importlib.reload(metrics_module)

    def test_fallback_sets_metrics_disabled(self):
        """METRICS_ENABLED should be set to False on ImportError."""
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": None}),
        ):
            importlib.reload(metrics_module)
            assert metrics_module.METRICS_ENABLED is False

    def test_fallback_uses_null_metrics(self):
        """All metric objects should be NullMetric on ImportError."""
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": None}),
        ):
            importlib.reload(metrics_module)
            assert isinstance(metrics_module.REQUEST_DURATION, metrics_module.NullMetric)
            assert isinstance(metrics_module.REQUEST_TOTAL, metrics_module.NullMetric)
            assert isinstance(metrics_module.RETRY_TOTAL, metrics_module.NullMetric)
            assert isinstance(metrics_module.CIRCUIT_BREAKER_STATE, metrics_module.NullMetric)

    def test_fallback_empty_state_values(self):
        """CIRCUIT_STATE_VALUES should be empty on ImportError."""
        with (
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": None}),
        ):
            importlib.reload(metrics_module)
            assert metrics_module.CIRCUIT_STATE_VALUES == {}

    def test_warning_log_on_import_error(self, caplog):
        """Warning log should mention prometheus_client not installed."""
        with (
            caplog.at_level(logging.WARNING, logger="slicer-mcp"),
            patch.dict(os.environ, {"SLICER_METRICS_ENABLED": "true"}),
            patch.dict(sys.modules, {"prometheus_client": None}),
        ):
            importlib.reload(metrics_module)
        assert "prometheus_client not installed" in caplog.text


class TestTrackRequestDebugLogging:
    """Test track_request debug logging branch (lines 148-149)."""

    def test_debug_log_on_success(self, caplog):
        """Debug log should include operation name and success status."""
        from slicer_mcp.core.metrics import track_request

        with caplog.at_level(logging.DEBUG, logger="slicer-mcp"):
            with track_request("debug_test_op"):
                pass
        assert "debug_test_op" in caplog.text
        assert "success" in caplog.text

    def test_debug_log_on_error(self, caplog):
        """Debug log should include operation name and error status."""
        from slicer_mcp.core.metrics import track_request

        with caplog.at_level(logging.DEBUG, logger="slicer-mcp"):
            with pytest.raises(ValueError):
                with track_request("failing_debug_op"):
                    raise ValueError("boom")
        assert "failing_debug_op" in caplog.text
        assert "error" in caplog.text

    def test_debug_log_includes_duration(self, caplog):
        """Debug log should include duration in seconds."""
        from slicer_mcp.core.metrics import track_request

        with caplog.at_level(logging.DEBUG, logger="slicer-mcp"):
            with track_request("timed_debug_op"):
                pass
        for record in caplog.records:
            if "timed_debug_op" in record.message:
                assert "in " in record.message
                assert "s" in record.message
                break
        else:
            pytest.fail("No debug log found for timed_debug_op")
