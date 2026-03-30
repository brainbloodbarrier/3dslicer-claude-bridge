"""Runtime configuration with Pydantic Settings validation.

Provides a validated, type-safe configuration object that reads from
environment variables (SLICER_* prefix) with sensible defaults matching
the existing constants in constants.py.

Usage:
    from slicer_mcp.core.config import settings

    print(settings.url)      # "http://localhost:2016"
    print(settings.timeout)  # 30
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "SlicerMCPConfig",
    "get_settings",
    "settings",
]


class SlicerMCPConfig(BaseSettings):
    """Validated runtime configuration for Slicer MCP Bridge.

    All fields can be overridden via environment variables with the SLICER_ prefix.
    For example, SLICER_URL overrides the url field, SLICER_TIMEOUT overrides timeout, etc.
    """

    model_config = SettingsConfigDict(
        env_prefix="SLICER_",
        # Reject unknown fields to catch typos early
        extra="ignore",
    )

    # -- Connection --
    url: str = "http://localhost:2016"
    timeout: int = 30

    # -- Version --
    min_version: str = "5.0.0"

    # -- Retry --
    retry_max_attempts: int = 3
    retry_backoff_base: float = 1.0

    # -- Circuit breaker --
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """URL must be non-empty and use http/https scheme."""
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        from urllib.parse import urlparse

        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"url must use http or https scheme, got '{parsed.scheme}'")
        if not parsed.hostname:
            raise ValueError("url must include a hostname")
        return v.rstrip("/")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Timeout must be between 1 and 600 seconds."""
        if v < 1 or v > 600:
            raise ValueError(f"timeout must be between 1 and 600, got {v}")
        return v

    @field_validator("min_version")
    @classmethod
    def validate_min_version(cls, v: str) -> str:
        """Version string must look like a semver (digits and dots)."""
        import re

        if not re.match(r"^\d+\.\d+(\.\d+)?$", v):
            raise ValueError(
                f"min_version must be a valid version string (e.g. '5.0.0'), got '{v}'"
            )
        return v

    @field_validator("retry_max_attempts")
    @classmethod
    def validate_retry_max_attempts(cls, v: int) -> int:
        """Retry attempts must be between 1 and 10."""
        if v < 1 or v > 10:
            raise ValueError(f"retry_max_attempts must be between 1 and 10, got {v}")
        return v

    @field_validator("retry_backoff_base")
    @classmethod
    def validate_retry_backoff_base(cls, v: float) -> float:
        """Backoff base must be between 0.1 and 10.0 seconds."""
        if v < 0.1 or v > 10.0:
            raise ValueError(f"retry_backoff_base must be between 0.1 and 10.0, got {v}")
        return v

    @field_validator("circuit_breaker_threshold")
    @classmethod
    def validate_circuit_breaker_threshold(cls, v: int) -> int:
        """Circuit breaker threshold must be between 1 and 50."""
        if v < 1 or v > 50:
            raise ValueError(f"circuit_breaker_threshold must be between 1 and 50, got {v}")
        return v

    @field_validator("circuit_breaker_timeout")
    @classmethod
    def validate_circuit_breaker_timeout(cls, v: float) -> float:
        """Circuit breaker timeout must be between 5.0 and 300.0 seconds."""
        if v < 5.0 or v > 300.0:
            raise ValueError(f"circuit_breaker_timeout must be between 5.0 and 300.0, got {v}")
        return v


def get_settings() -> SlicerMCPConfig:
    """Create a fresh SlicerMCPConfig from the current environment.

    Useful in tests where you need a new instance after modifying env vars.
    For production code, use the module-level `settings` singleton.
    """
    return SlicerMCPConfig()


# Module-level singleton — import this for production use
settings = SlicerMCPConfig()
