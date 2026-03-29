"""Async HTTP client for 3D Slicer WebServer API.

Drop-in async replacement for ``SlicerClient`` using ``httpx``.
Enables non-blocking I/O when used with async MCP tool handlers,
allowing FastMCP to report progress during long operations.

Usage::

    from slicer_mcp.core.async_client import get_async_client

    async def my_tool():
        client = get_async_client()
        result = await client.exec_python("__execResult = 42")
        return result
"""

import asyncio
import json
import logging
import time
from functools import wraps
from typing import Any, NoReturn, TypeVar

import httpx

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.constants import (
    DEFAULT_SLICER_URL,
    DEFAULT_TIMEOUT_SECONDS,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_ATTEMPTS,
    SLICER_MIN_VERSION,
    SLICER_TESTED_VERSIONS,
)
from slicer_mcp.core.metrics import (
    record_retry,
    track_request,
    update_circuit_breaker_state,
)
from slicer_mcp.core.slicer_client import SlicerConnectionError, SlicerTimeoutError

__all__ = [
    "SlicerAsyncClient",
    "async_with_retry",
    "get_async_client",
    "reset_async_client",
]

T = TypeVar("T")
logger = logging.getLogger("slicer-mcp")

# ── Singleton management ─────────────────────────────────────────────
_async_client_instance: "SlicerAsyncClient | None" = None


def get_async_client() -> "SlicerAsyncClient":
    """Get the singleton async SlicerClient instance.

    Creates the client on first call, reuses it for subsequent calls.
    Async-safe: asyncio is single-threaded so no lock is needed.
    """
    global _async_client_instance
    if _async_client_instance is None:
        _async_client_instance = SlicerAsyncClient()
        logger.info("Created singleton SlicerAsyncClient instance")
    return _async_client_instance


def reset_async_client() -> None:
    """Reset the async client singleton. Useful for testing."""
    global _async_client_instance
    _async_client_instance = None
    logger.info("Reset singleton SlicerAsyncClient instance")


# ── Async retry decorator ────────────────────────────────────────────


def async_with_retry(
    max_retries: int = RETRY_MAX_ATTEMPTS,
    backoff_base: float = RETRY_BACKOFF_BASE,
    retryable_exceptions: tuple[type[Exception], ...] = (SlicerConnectionError,),
):
    """Async decorator for retry with exponential backoff.

    Same semantics as ``with_retry`` but uses ``asyncio.sleep``
    instead of ``time.sleep``, keeping the event loop unblocked.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = backoff_base * (2**attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {sleep_time}s delay. Error: {e}"
                        )
                        record_retry(func.__name__)
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}. "
                            f"Final error: {e}"
                        )
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Retry logic error in {func.__name__}")

        return wrapper

    return decorator


# ── Shared circuit breaker (reuse from sync client) ──────────────────
from slicer_mcp.core.slicer_client import (  # noqa: E402
    get_circuit_breaker,
)


class SlicerAsyncClient:
    """Async HTTP client for 3D Slicer WebServer API.

    Mirror of ``SlicerClient`` using ``httpx.AsyncClient`` for
    non-blocking I/O.  Shares the same circuit breaker instance
    with the sync client for coordinated protection.
    """

    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        import os
        from urllib.parse import urlparse

        if base_url is None:
            base_url = os.environ.get("SLICER_URL", DEFAULT_SLICER_URL)
        if timeout is None:
            timeout_str = os.environ.get("SLICER_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
            try:
                timeout = int(timeout_str)
                if timeout <= 0:
                    timeout = DEFAULT_TIMEOUT_SECONDS
            except ValueError:
                timeout = DEFAULT_TIMEOUT_SECONDS

        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"SLICER_URL must use http or https scheme, got '{parsed.scheme}'")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cb = get_circuit_breaker()

        logger.info(
            f"Initialized SlicerAsyncClient with base_url={self.base_url}, "
            f"timeout={self.timeout}s"
        )

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows the request."""
        if not self._cb.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker 'slicer' is OPEN. " f"Will retry in {self._cb.recovery_timeout}s",
                breaker_name="slicer",
                recovery_timeout=self._cb.recovery_timeout,
            )

    def _record_success(self) -> None:
        self._cb.record_success()
        update_circuit_breaker_state("slicer", "closed")

    def _handle_request_error(
        self, operation: str, error: Exception, extra_details: dict[str, Any] | None = None
    ) -> NoReturn:
        details = {"url": self.base_url}
        if extra_details:
            details.update(extra_details)

        if isinstance(error, httpx.TimeoutException | httpx.ConnectError):
            self._cb.record_failure()
            update_circuit_breaker_state("slicer", self._cb.state.value)

        if isinstance(error, httpx.TimeoutException):
            raise SlicerTimeoutError(
                f"{operation} timed out after {self.timeout}s",
                details={**details, "timeout": self.timeout},
            )
        elif isinstance(error, httpx.ConnectError):
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={**details, "suggestion": "Ensure Slicer is running"},
            )
        else:
            raise SlicerConnectionError(f"{operation} failed: {error}", details=details)

    @async_with_retry(retryable_exceptions=(SlicerConnectionError,))
    async def health_check(self, check_version: bool = True) -> dict[str, Any]:
        """Async health check against Slicer WebServer."""
        self._check_circuit_breaker()

        with track_request("health_check"):
            try:
                start = time.time()
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self.base_url}/slicer/mrml",
                        timeout=self.timeout,
                    )
                elapsed_ms = int((time.time() - start) * 1000)
                resp.raise_for_status()
                self._record_success()

                result: dict[str, Any] = {
                    "connected": True,
                    "webserver_url": self.base_url,
                    "response_time_ms": elapsed_ms,
                }
                if check_version:
                    try:
                        version_info = await self.check_version_compatibility()
                        result["slicer_version"] = version_info["version"]
                        result["version_compatible"] = version_info["compatible"]
                    except Exception:
                        pass
                return result

            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
                self._handle_request_error("Health check", e)

    async def exec_python(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute Python code in Slicer — async, non-blocking.

        NOT retried because Python execution is not idempotent.
        """
        self._check_circuit_breaker()
        effective_timeout = timeout if timeout is not None else self.timeout

        with track_request("exec_python"):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self.base_url}/slicer/exec",
                        content=code,
                        headers={"Content-Type": "text/plain"},
                        timeout=effective_timeout,
                    )
                resp.raise_for_status()
                self._record_success()
                return {"success": True, "result": resp.text}

            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
                self._handle_request_error("Python execution", e, {"code_preview": code[:100]})

    async def get_slicer_version(self) -> str:
        """Get Slicer version string — async."""
        code = "import json, slicer; __execResult = json.dumps(slicer.app.applicationVersion)"
        result = await self.exec_python(code)
        raw = result.get("result", "").strip()
        try:
            return json.loads(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return raw.strip("'\"")

    async def check_version_compatibility(self) -> dict[str, Any]:
        """Check version compatibility — async."""
        import re

        from packaging import version as pkg_version

        current = await self.get_slicer_version()

        def extract_ver(s: str) -> str:
            m = re.match(r"^(\d+\.\d+\.\d+)", s)
            return m.group(1) if m else s

        try:
            is_compatible = pkg_version.parse(extract_ver(current)) >= pkg_version.parse(
                SLICER_MIN_VERSION
            )
        except (ValueError, TypeError):
            is_compatible = False

        return {
            "version": current,
            "compatible": is_compatible,
            "tested": current in SLICER_TESTED_VERSIONS,
            "minimum_required": SLICER_MIN_VERSION,
            "warning": None if is_compatible else f"Below minimum {SLICER_MIN_VERSION}",
        }
