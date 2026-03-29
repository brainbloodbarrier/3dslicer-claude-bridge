"""Tests for runtime configuration validation (slicer_mcp.core.config)."""

import pytest
from pydantic import ValidationError

from slicer_mcp.core.config import SlicerMCPConfig, get_settings

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify that default values match the legacy hardcoded constants."""

    def test_default_url(self):
        cfg = SlicerMCPConfig()
        assert cfg.url == "http://localhost:2016"

    def test_default_timeout(self):
        cfg = SlicerMCPConfig()
        assert cfg.timeout == 30

    def test_default_min_version(self):
        cfg = SlicerMCPConfig()
        assert cfg.min_version == "5.0.0"

    def test_default_retry_max_attempts(self):
        cfg = SlicerMCPConfig()
        assert cfg.retry_max_attempts == 3

    def test_default_retry_backoff_base(self):
        cfg = SlicerMCPConfig()
        assert cfg.retry_backoff_base == 1.0

    def test_default_circuit_breaker_threshold(self):
        cfg = SlicerMCPConfig()
        assert cfg.circuit_breaker_threshold == 5

    def test_default_circuit_breaker_timeout(self):
        cfg = SlicerMCPConfig()
        assert cfg.circuit_breaker_timeout == 30.0


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    """Verify that SLICER_* env vars override defaults."""

    def test_url_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_URL", "http://localhost:9999")
        cfg = get_settings()
        assert cfg.url == "http://localhost:9999"

    def test_timeout_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_TIMEOUT", "120")
        cfg = get_settings()
        assert cfg.timeout == 120

    def test_min_version_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_MIN_VERSION", "5.6")
        cfg = get_settings()
        assert cfg.min_version == "5.6"

    def test_retry_max_attempts_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_RETRY_MAX_ATTEMPTS", "5")
        cfg = get_settings()
        assert cfg.retry_max_attempts == 5

    def test_retry_backoff_base_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_RETRY_BACKOFF_BASE", "2.5")
        cfg = get_settings()
        assert cfg.retry_backoff_base == 2.5

    def test_circuit_breaker_threshold_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_CIRCUIT_BREAKER_THRESHOLD", "10")
        cfg = get_settings()
        assert cfg.circuit_breaker_threshold == 10

    def test_circuit_breaker_timeout_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_CIRCUIT_BREAKER_TIMEOUT", "60.0")
        cfg = get_settings()
        assert cfg.circuit_breaker_timeout == 60.0

    def test_url_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("SLICER_URL", "http://localhost:2016/")
        cfg = get_settings()
        assert cfg.url == "http://localhost:2016"

    def test_https_url_accepted(self, monkeypatch):
        monkeypatch.setenv("SLICER_URL", "https://slicer.example.com:443")
        cfg = get_settings()
        assert cfg.url == "https://slicer.example.com:443"


# ---------------------------------------------------------------------------
# Validation rejects invalid values
# ---------------------------------------------------------------------------


class TestValidationRejects:
    """Verify that invalid values raise ValidationError."""

    def test_empty_url_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_URL", "")
        with pytest.raises(ValidationError, match="url"):
            get_settings()

    def test_non_http_url_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_URL", "ftp://localhost:2016")
        with pytest.raises(ValidationError, match="http or https"):
            get_settings()

    def test_url_without_hostname_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_URL", "http://")
        with pytest.raises(ValidationError, match="hostname"):
            get_settings()

    def test_negative_timeout_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_TIMEOUT", "-1")
        with pytest.raises(ValidationError, match="timeout"):
            get_settings()

    def test_zero_timeout_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_TIMEOUT", "0")
        with pytest.raises(ValidationError, match="timeout"):
            get_settings()

    def test_timeout_too_large_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_TIMEOUT", "999")
        with pytest.raises(ValidationError, match="timeout"):
            get_settings()

    def test_invalid_min_version_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_MIN_VERSION", "not-a-version")
        with pytest.raises(ValidationError, match="min_version"):
            get_settings()

    def test_retry_attempts_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_RETRY_MAX_ATTEMPTS", "0")
        with pytest.raises(ValidationError, match="retry_max_attempts"):
            get_settings()

    def test_retry_attempts_too_large_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_RETRY_MAX_ATTEMPTS", "99")
        with pytest.raises(ValidationError, match="retry_max_attempts"):
            get_settings()

    def test_backoff_too_small_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_RETRY_BACKOFF_BASE", "0.01")
        with pytest.raises(ValidationError, match="retry_backoff_base"):
            get_settings()

    def test_backoff_too_large_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_RETRY_BACKOFF_BASE", "99.0")
        with pytest.raises(ValidationError, match="retry_backoff_base"):
            get_settings()

    def test_cb_threshold_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_CIRCUIT_BREAKER_THRESHOLD", "0")
        with pytest.raises(ValidationError, match="circuit_breaker_threshold"):
            get_settings()

    def test_cb_threshold_too_large_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_CIRCUIT_BREAKER_THRESHOLD", "100")
        with pytest.raises(ValidationError, match="circuit_breaker_threshold"):
            get_settings()

    def test_cb_timeout_too_small_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_CIRCUIT_BREAKER_TIMEOUT", "1.0")
        with pytest.raises(ValidationError, match="circuit_breaker_timeout"):
            get_settings()

    def test_cb_timeout_too_large_rejected(self, monkeypatch):
        monkeypatch.setenv("SLICER_CIRCUIT_BREAKER_TIMEOUT", "999.0")
        with pytest.raises(ValidationError, match="circuit_breaker_timeout"):
            get_settings()


# ---------------------------------------------------------------------------
# Boundary values (edge of valid range should be accepted)
# ---------------------------------------------------------------------------


class TestBoundaryValues:
    """Verify that edge-of-range values are accepted."""

    def test_timeout_min_boundary(self):
        cfg = SlicerMCPConfig(timeout=1)
        assert cfg.timeout == 1

    def test_timeout_max_boundary(self):
        cfg = SlicerMCPConfig(timeout=600)
        assert cfg.timeout == 600

    def test_retry_max_attempts_min_boundary(self):
        cfg = SlicerMCPConfig(retry_max_attempts=1)
        assert cfg.retry_max_attempts == 1

    def test_retry_max_attempts_max_boundary(self):
        cfg = SlicerMCPConfig(retry_max_attempts=10)
        assert cfg.retry_max_attempts == 10

    def test_backoff_min_boundary(self):
        cfg = SlicerMCPConfig(retry_backoff_base=0.1)
        assert cfg.retry_backoff_base == pytest.approx(0.1)

    def test_backoff_max_boundary(self):
        cfg = SlicerMCPConfig(retry_backoff_base=10.0)
        assert cfg.retry_backoff_base == pytest.approx(10.0)

    def test_cb_threshold_min_boundary(self):
        cfg = SlicerMCPConfig(circuit_breaker_threshold=1)
        assert cfg.circuit_breaker_threshold == 1

    def test_cb_threshold_max_boundary(self):
        cfg = SlicerMCPConfig(circuit_breaker_threshold=50)
        assert cfg.circuit_breaker_threshold == 50

    def test_cb_timeout_min_boundary(self):
        cfg = SlicerMCPConfig(circuit_breaker_timeout=5.0)
        assert cfg.circuit_breaker_timeout == pytest.approx(5.0)

    def test_cb_timeout_max_boundary(self):
        cfg = SlicerMCPConfig(circuit_breaker_timeout=300.0)
        assert cfg.circuit_breaker_timeout == pytest.approx(300.0)

    def test_min_version_two_part(self):
        cfg = SlicerMCPConfig(min_version="5.6")
        assert cfg.min_version == "5.6"


# ---------------------------------------------------------------------------
# constants.py backward compatibility
# ---------------------------------------------------------------------------


class TestConstantsBackwardCompat:
    """Verify that constants.py still exports the expected names with correct values."""

    def test_default_slicer_url_exported(self):
        from slicer_mcp.core.constants import DEFAULT_SLICER_URL

        assert DEFAULT_SLICER_URL == "http://localhost:2016"

    def test_default_timeout_exported(self):
        from slicer_mcp.core.constants import DEFAULT_TIMEOUT_SECONDS

        assert DEFAULT_TIMEOUT_SECONDS == 30

    def test_retry_max_attempts_exported(self):
        from slicer_mcp.core.constants import RETRY_MAX_ATTEMPTS

        assert RETRY_MAX_ATTEMPTS == 3

    def test_retry_backoff_base_exported(self):
        from slicer_mcp.core.constants import RETRY_BACKOFF_BASE

        assert RETRY_BACKOFF_BASE == 1.0

    def test_slicer_min_version_exported(self):
        from slicer_mcp.core.constants import SLICER_MIN_VERSION

        assert SLICER_MIN_VERSION == "5.0.0"

    def test_circuit_breaker_threshold_exported(self):
        from slicer_mcp.core.constants import CIRCUIT_BREAKER_FAILURE_THRESHOLD

        assert CIRCUIT_BREAKER_FAILURE_THRESHOLD == 5

    def test_circuit_breaker_timeout_exported(self):
        from slicer_mcp.core.constants import CIRCUIT_BREAKER_RECOVERY_TIMEOUT

        assert CIRCUIT_BREAKER_RECOVERY_TIMEOUT == 30.0

    def test_static_constants_unchanged(self):
        """Verify that static (non-config) constants are still present."""
        from slicer_mcp.core.constants import (
            MAX_NODE_ID_LENGTH,
            MAX_PYTHON_CODE_LENGTH,
            VALID_LAYOUTS,
            VIEW_AXIAL,
            VIEW_MAP,
        )

        assert MAX_NODE_ID_LENGTH == 256
        assert MAX_PYTHON_CODE_LENGTH == 100000
        assert VIEW_AXIAL == "Red"
        assert "axial" in VIEW_MAP
        assert "FourUp" in VALID_LAYOUTS


# ---------------------------------------------------------------------------
# get_settings() factory
# ---------------------------------------------------------------------------


class TestGetSettings:
    """Verify that get_settings() returns a fresh instance."""

    def test_get_settings_returns_config(self):
        cfg = get_settings()
        assert isinstance(cfg, SlicerMCPConfig)

    def test_get_settings_reads_current_env(self, monkeypatch):
        monkeypatch.setenv("SLICER_TIMEOUT", "42")
        cfg = get_settings()
        assert cfg.timeout == 42
