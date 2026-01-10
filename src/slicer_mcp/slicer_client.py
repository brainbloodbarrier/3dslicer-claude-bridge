"""HTTP client for 3D Slicer WebServer API."""

import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, NoReturn, Optional, TypeVar

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

from slicer_mcp.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
)
from slicer_mcp.constants import (
    DEFAULT_SLICER_URL,
    DEFAULT_TIMEOUT_SECONDS,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_ATTEMPTS,
    SLICER_MIN_VERSION,
    SLICER_TESTED_VERSIONS,
)
from slicer_mcp.metrics import (
    record_retry,
    track_request,
    update_circuit_breaker_state,
)

# Type variable for generic return type
T = TypeVar("T")

# Configure logging to stderr (stdout reserved for MCP)
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    stream=sys.stderr,
)
logger = logging.getLogger("slicer-mcp")


class SlicerConnectionError(Exception):
    """Raised when connection to Slicer WebServer fails (retryable)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class SlicerTimeoutError(Exception):
    """Raised when request to Slicer times out (not retryable - Slicer may be frozen)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


def with_retry(
    max_retries: int = RETRY_MAX_ATTEMPTS,
    backoff_base: float = RETRY_BACKOFF_BASE,
    retryable_exceptions: tuple[type[Exception], ...] = (ConnectionError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry with exponential backoff.

    Implements retry logic as specified in SPECIFICATION.md:
    - Connection errors: Retry up to 3 times with exponential backoff (1s, 2s, 4s)
    - Timeout errors: No retry (likely Slicer is frozen)
    - Other errors: No retry

    Also records retry metrics for operational visibility.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base delay in seconds for exponential backoff (default: 1.0)
        retryable_exceptions: Tuple of exception types that should trigger retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = backoff_base * (2**attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {sleep_time}s delay. Error: {e}"
                        )
                        # Record retry metric
                        record_retry(func.__name__)
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}. "
                            f"Final error: {e}"
                        )
            raise last_exception

        return wrapper

    return decorator


# Singleton instance management
# Lock initialized at module load time to avoid race conditions
# (module loading is guaranteed to be single-threaded in Python)
_client_instance: Optional["SlicerClient"] = None
_client_lock: threading.Lock = threading.Lock()

# Global circuit breaker for Slicer connection protection
# Opens after 5 consecutive failures, tests recovery after 30s
_slicer_circuit_breaker = CircuitBreaker(name="slicer")


def get_client() -> "SlicerClient":
    """Get the singleton SlicerClient instance.

    Creates the client on first call, reuses it for subsequent calls.
    Thread-safe initialization using double-checked locking pattern.

    Returns:
        The singleton SlicerClient instance configured from environment variables.
    """
    global _client_instance

    # Fast path: if already initialized, return immediately (no lock needed)
    if _client_instance is not None:
        return _client_instance

    # Slow path: acquire lock and check again (double-checked locking)
    with _client_lock:
        # Re-check after acquiring lock (another thread may have initialized)
        if _client_instance is None:
            _client_instance = SlicerClient()
            logger.info("Created singleton SlicerClient instance")

    return _client_instance


def reset_client() -> None:
    """Reset the singleton client instance.

    Useful for testing or when connection parameters change.
    """
    global _client_instance
    _client_instance = None
    logger.info("Reset singleton SlicerClient instance")


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global Slicer circuit breaker.

    Returns:
        The CircuitBreaker instance protecting Slicer connections
    """
    return _slicer_circuit_breaker


def reset_circuit_breaker() -> None:
    """Reset the Slicer circuit breaker to closed state.

    Useful for testing or manual recovery intervention.
    """
    _slicer_circuit_breaker.reset()
    update_circuit_breaker_state("slicer", "closed")


class SlicerClient:
    """HTTP client for 3D Slicer WebServer API.

    This client provides methods to interact with Slicer's REST API endpoints
    including Python execution, screenshot capture, scene inspection, and data loading.

    For most use cases, use the `get_client()` function to get the singleton instance
    rather than creating new instances directly.
    """

    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        """Initialize Slicer HTTP client.

        Args:
            base_url: Base URL of Slicer WebServer. If None, reads from SLICER_URL
                      environment variable (default: http://localhost:2016)
            timeout: Request timeout in seconds. If None, reads from SLICER_TIMEOUT
                     environment variable (default: 30)
        """
        import os

        if base_url is None:
            base_url = os.environ.get("SLICER_URL", DEFAULT_SLICER_URL)
        if timeout is None:
            timeout = int(os.environ.get("SLICER_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Note: We use direct requests.get/post instead of Session to avoid
        # connection reuse issues with Slicer's WebServer (it closes connections
        # immediately after response, causing "Connection reset by peer" errors)

        logger.info(
            f"Initialized SlicerClient with base_url={self.base_url}, timeout={self.timeout}s"
        )

    def _handle_request_error(
        self, operation: str, error: Exception, extra_details: dict[str, Any] | None = None
    ) -> NoReturn:
        """Handle HTTP request errors with consistent logging and exception raising.

        Also records failure with the circuit breaker and updates metrics.

        Args:
            operation: Description of the failed operation (for logging)
            error: The caught exception
            extra_details: Additional details to include in the exception

        Raises:
            SlicerConnectionError: Always raised with standardized details
            SlicerTimeoutError: If the error is a Timeout
        """
        details = {"url": self.base_url}
        if extra_details:
            details.update(extra_details)

        # Record failure with circuit breaker (for connection-related errors)
        if isinstance(error, (Timeout, ConnectionError)):
            _slicer_circuit_breaker.record_failure()
            update_circuit_breaker_state("slicer", _slicer_circuit_breaker.state.value)

        if isinstance(error, Timeout):
            logger.error(f"{operation} timeout: {error}")
            raise SlicerTimeoutError(
                f"{operation} timed out after {self.timeout}s",
                details={
                    **details,
                    "timeout": self.timeout,
                    "suggestion": "Slicer may be frozen. Try restarting Slicer.",
                },
            )
        elif isinstance(error, ConnectionError):
            logger.error(f"{operation} connection failed: {error}")
            raise SlicerConnectionError(
                f"Could not connect to Slicer WebServer at {self.base_url}",
                details={
                    **details,
                    "suggestion": "Ensure Slicer is running with WebServer extension enabled",
                },
            )
        else:
            logger.error(f"{operation} failed: {error}")
            raise SlicerConnectionError(f"{operation} failed: {str(error)}", details=details)

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows the request.

        Raises:
            CircuitOpenError: If circuit breaker is open
        """
        if not _slicer_circuit_breaker.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker 'slicer' is OPEN. "
                f"Slicer connection appears to be failing. "
                f"Will retry in {_slicer_circuit_breaker.recovery_timeout}s",
                breaker_name="slicer",
                recovery_timeout=_slicer_circuit_breaker.recovery_timeout,
            )

    def _record_success(self) -> None:
        """Record a successful operation with circuit breaker."""
        _slicer_circuit_breaker.record_success()
        update_circuit_breaker_state("slicer", "closed")

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def health_check(self, check_version: bool = True) -> dict[str, Any]:
        """Check connection to Slicer WebServer.

        Args:
            check_version: If True, also check version compatibility (default: True)

        Returns:
            Dict with connection status, response time, and optional version info

        Raises:
            SlicerConnectionError: If Slicer is not reachable (will retry)
            SlicerTimeoutError: If request times out (no retry - Slicer may be frozen)
            CircuitOpenError: If circuit breaker is open
        """
        # Check circuit breaker first
        self._check_circuit_breaker()

        with track_request("health_check"):
            try:
                start_time = time.time()

                response = requests.get(f"{self.base_url}/slicer/mrml", timeout=self.timeout)

                elapsed_ms = int((time.time() - start_time) * 1000)

                response.raise_for_status()

                # Record success with circuit breaker
                self._record_success()

                logger.info(f"Health check successful: {elapsed_ms}ms")

                result = {
                    "connected": True,
                    "webserver_url": self.base_url,
                    "response_time_ms": elapsed_ms,
                    "circuit_breaker_state": _slicer_circuit_breaker.state.value,
                }

                # Optionally check version compatibility
                if check_version:
                    try:
                        version_info = self.check_version_compatibility()
                        result["slicer_version"] = version_info["version"]
                        result["version_compatible"] = version_info["compatible"]
                        result["version_tested"] = version_info["tested"]
                        if version_info["warning"]:
                            result["version_warning"] = version_info["warning"]
                    except Exception as e:
                        # Version check is optional, don't fail health check
                        logger.debug(f"Version check failed (non-critical): {e}")

                return result

            except (Timeout, ConnectionError, RequestException) as e:
                # Timeout errors are NOT retried - Slicer may be frozen
                # Connection errors ARE retried
                self._handle_request_error("Health check", e, {"timeout": self.timeout})

    def get_slicer_version(self) -> str:
        """Get the Slicer application version string.

        Returns:
            Version string (e.g., "5.6.2")

        Raises:
            SlicerConnectionError: If Slicer is not reachable
        """
        python_code = "import slicer; slicer.app.applicationVersion"
        result = self.exec_python(python_code)
        # Result comes back quoted, strip quotes and whitespace
        version_str = result.get("result", "").strip().strip("'\"")
        logger.info(f"Slicer version: {version_str}")
        return version_str

    def check_version_compatibility(self) -> dict[str, Any]:
        """Check if the connected Slicer version is compatible.

        Queries Slicer for its version and compares against known compatible
        versions. Returns compatibility information including warnings for
        untested versions.

        Returns:
            Dict with keys:
                - version: The Slicer version string
                - compatible: True if version >= minimum required
                - tested: True if version is in the tested versions list
                - minimum_required: Minimum supported version string
                - warning: Warning message if version has issues, None otherwise

        Raises:
            SlicerConnectionError: If Slicer is not reachable
        """
        from packaging import version as pkg_version

        current = self.get_slicer_version()

        # Parse versions for comparison
        # Slicer dev versions like "5.7.0-2024-01-01" aren't PEP 440 compliant
        # Extract major.minor.patch for comparison
        def extract_version(ver_str: str) -> str:
            """Extract semver-like version from Slicer version string."""
            import re

            match = re.match(r"^(\d+\.\d+\.\d+)", ver_str)
            return match.group(1) if match else ver_str

        try:
            current_parsed = pkg_version.parse(extract_version(current))
            min_parsed = pkg_version.parse(SLICER_MIN_VERSION)
            is_compatible = current_parsed >= min_parsed
        except Exception as e:
            logger.warning(f"Could not parse version '{current}': {e}")
            is_compatible = False

        is_tested = current in SLICER_TESTED_VERSIONS

        result = {
            "version": current,
            "compatible": is_compatible,
            "tested": is_tested,
            "minimum_required": SLICER_MIN_VERSION,
            "warning": None,
        }

        if not is_compatible:
            result["warning"] = (
                f"Slicer {current} is below minimum required version {SLICER_MIN_VERSION}. "
                f"Some features may not work correctly."
            )
            logger.warning(result["warning"])
        elif not is_tested:
            result["warning"] = (
                f"Slicer {current} has not been tested with this bridge. "
                f"Tested versions: {', '.join(sorted(SLICER_TESTED_VERSIONS))}"
            )
            logger.info(result["warning"])
        else:
            logger.info(f"Slicer {current} is compatible and tested")

        return result

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def exec_python(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute Python code in Slicer's Python environment.

        Args:
            code: Python code to execute
            timeout: Optional timeout override in seconds. If None, uses the client's
                     default timeout. Use for long-running operations like brain extraction.

        Returns:
            Dict with success status and result/output

        Raises:
            SlicerConnectionError: If connection fails (will retry)
            SlicerTimeoutError: If request times out (no retry)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        # Use provided timeout or fall back to default
        effective_timeout = timeout if timeout is not None else self.timeout

        with track_request("exec_python"):
            try:
                logger.debug(
                    f"Executing Python code (timeout={effective_timeout}s): {code[:100]}..."
                )

                response = requests.post(
                    f"{self.base_url}/slicer/exec",
                    data=code,
                    headers={"Content-Type": "text/plain"},
                    timeout=effective_timeout,
                )

                response.raise_for_status()

                result = response.text

                self._record_success()

                logger.info(f"Python execution successful, result length: {len(result)}")

                return {"success": True, "result": result, "stdout": "", "stderr": ""}

            except (Timeout, ConnectionError, RequestException) as e:
                self._handle_request_error(
                    "Python execution",
                    e,
                    {"code_preview": code[:100], "timeout": effective_timeout},
                )

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def get_screenshot(self, view: str = "Red", scroll_to: float | None = None) -> bytes:
        """Capture screenshot from a slice view.

        Args:
            view: View name - Red (axial), Yellow (sagittal), Green (coronal)
            scroll_to: Slice position from 0.0 to 1.0 (optional)

        Returns:
            PNG image as bytes

        Raises:
            SlicerConnectionError: If request fails (will retry up to 3 times)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        with track_request("get_screenshot"):
            try:
                url = f"{self.base_url}/slicer/slice?view={view}"
                if scroll_to is not None:
                    url += f"&scrollTo={scroll_to}"

                logger.debug(f"Capturing screenshot: view={view}, scroll_to={scroll_to}")

                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                image_bytes = response.content

                self._record_success()

                logger.info(f"Screenshot captured: {len(image_bytes)} bytes")

                return image_bytes

            except (ConnectionError, Timeout, RequestException) as e:
                self._handle_request_error(
                    "Screenshot capture", e, {"view": view, "scroll_to": scroll_to}
                )

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def get_3d_screenshot(self, look_from_axis: str | None = None) -> bytes:
        """Capture screenshot from 3D view.

        Args:
            look_from_axis: Camera axis - left, right, anterior, posterior, superior, inferior (optional)

        Returns:
            PNG image as bytes

        Raises:
            SlicerConnectionError: If request fails (will retry up to 3 times)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        with track_request("get_3d_screenshot"):
            try:
                url = f"{self.base_url}/slicer/threeD"
                if look_from_axis:
                    url += f"?lookFromAxis={look_from_axis}"

                logger.debug(f"Capturing 3D screenshot: look_from_axis={look_from_axis}")

                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                image_bytes = response.content

                self._record_success()

                logger.info(f"3D screenshot captured: {len(image_bytes)} bytes")

                return image_bytes

            except (ConnectionError, Timeout, RequestException) as e:
                self._handle_request_error(
                    "3D screenshot capture", e, {"look_from_axis": look_from_axis}
                )

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def get_full_screenshot(self) -> bytes:
        """Capture screenshot of full Slicer window.

        Returns:
            PNG image as bytes

        Raises:
            SlicerConnectionError: If request fails (will retry up to 3 times)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        with track_request("get_full_screenshot"):
            try:
                logger.debug("Capturing full window screenshot")

                response = requests.get(f"{self.base_url}/slicer/screenshot", timeout=self.timeout)
                response.raise_for_status()

                image_bytes = response.content

                self._record_success()

                logger.info(f"Full screenshot captured: {len(image_bytes)} bytes")

                return image_bytes

            except (ConnectionError, Timeout, RequestException) as e:
                self._handle_request_error("Full screenshot capture", e)

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def get_scene_nodes(self) -> list[dict[str, Any]]:
        """Get list of all MRML scene nodes with IDs and names.

        Returns:
            List of node dictionaries with id, name, type

        Raises:
            SlicerConnectionError: If request fails (will retry)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        with track_request("get_scene_nodes"):
            try:
                logger.debug("Fetching scene nodes")

                # Get node names
                names_response = requests.get(
                    f"{self.base_url}/slicer/mrml/names", timeout=self.timeout
                )
                names_response.raise_for_status()

                # Get node IDs
                ids_response = requests.get(
                    f"{self.base_url}/slicer/mrml/ids", timeout=self.timeout
                )
                ids_response.raise_for_status()

                # Parse responses - Slicer returns JSON arrays
                try:
                    names = json.loads(names_response.text)
                    ids = json.loads(ids_response.text)
                except json.JSONDecodeError as e:
                    raise SlicerConnectionError(f"Failed to parse scene nodes response: {e}")

                # Combine into node list
                nodes = []
                for node_id, node_name in zip(ids, names):
                    # Extract node type from ID (e.g., vtkMRMLScalarVolumeNode1 -> vtkMRMLScalarVolumeNode)
                    node_type = "".join(c for c in node_id if not c.isdigit())
                    nodes.append({"id": node_id, "name": node_name, "type": node_type})

                logger.info(f"Fetched {len(nodes)} scene nodes")

                self._record_success()

                return nodes

            except (ConnectionError, Timeout, RequestException) as e:
                self._handle_request_error("Scene nodes fetch", e)

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def get_node_properties(self, node_id: str) -> dict[str, Any]:
        """Get properties of a specific MRML node.

        Args:
            node_id: MRML node ID (e.g., vtkMRMLScalarVolumeNode1)

        Returns:
            Dict with node properties

        Raises:
            SlicerConnectionError: If request fails (will retry up to 3 times)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        with track_request("get_node_properties"):
            try:
                logger.debug(f"Fetching properties for node: {node_id}")

                response = requests.get(
                    f"{self.base_url}/slicer/mrml/properties",
                    params={"id": node_id},
                    timeout=self.timeout,
                )
                response.raise_for_status()

                # Properties are returned as text, parse as needed
                properties_text = response.text

                # Return as dict (Slicer returns key=value pairs)
                properties: dict[str, Any] = {}
                for line in properties_text.strip().split("\n"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        properties[key.strip()] = value.strip()

                self._record_success()

                logger.info(f"Fetched properties for node {node_id}")

                return properties

            except (ConnectionError, Timeout, RequestException) as e:
                self._handle_request_error("Node properties fetch", e, {"node_id": node_id})

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def load_sample_data(self, name: str) -> dict[str, Any]:
        """Load a sample dataset into Slicer.

        Args:
            name: Sample dataset name (e.g., MRHead, CTChest)

        Returns:
            Dict with success status and message

        Raises:
            SlicerConnectionError: If request fails (will retry)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        with track_request("load_sample_data"):
            try:
                logger.debug(f"Loading sample data: {name}")

                response = requests.get(
                    f"{self.base_url}/slicer/sampledata",
                    params={"name": name},
                    timeout=self.timeout,
                )
                response.raise_for_status()

                logger.info(f"Sample data '{name}' loaded successfully")

                self._record_success()

                return {
                    "success": True,
                    "dataset_name": name,
                    "message": f"Sample data '{name}' loaded successfully",
                }

            except (ConnectionError, Timeout, RequestException) as e:
                extra_details = {
                    "dataset_name": name,
                    "suggestion": "Check dataset name is valid (MRHead, CTChest, CTACardio, DTIBrain, MRBrainTumor1, MRBrainTumor2)",
                }
                self._handle_request_error(f"Sample data load '{name}'", e, extra_details)

    @with_retry(
        max_retries=RETRY_MAX_ATTEMPTS,
        backoff_base=RETRY_BACKOFF_BASE,
        retryable_exceptions=(SlicerConnectionError,),
    )
    def set_layout(self, layout: str, gui_mode: str = "full") -> dict[str, Any]:
        """Set Slicer viewer layout and GUI mode.

        Args:
            layout: Layout name (FourUp, OneUp3D, OneUpRedSlice, Conventional, SideBySide)
            gui_mode: GUI mode - full (complete GUI) or viewers (viewers only)

        Returns:
            Dict with success status and message

        Raises:
            SlicerConnectionError: If request fails (will retry)
            CircuitOpenError: If circuit breaker is open
        """
        self._check_circuit_breaker()

        with track_request("set_layout"):
            try:
                logger.debug(f"Setting layout: {layout}, gui_mode: {gui_mode}")

                response = requests.get(
                    f"{self.base_url}/slicer/gui",
                    params={"contents": gui_mode, "viewersLayout": layout},
                    timeout=self.timeout,
                )
                response.raise_for_status()

                logger.info(f"Layout changed to {layout} with {gui_mode} GUI mode")

                self._record_success()

                return {
                    "success": True,
                    "layout": layout,
                    "gui_mode": gui_mode,
                    "message": f"Layout changed to {layout}",
                }

            except (ConnectionError, Timeout, RequestException) as e:
                extra_details = {
                    "layout": layout,
                    "gui_mode": gui_mode,
                    "suggestion": "Check layout name is valid (FourUp, OneUp3D, OneUpRedSlice, Conventional, SideBySide)",
                }
                self._handle_request_error("Layout change", e, extra_details)
