"""Circuit breaker pattern for preventing cascading failures.

The circuit breaker pattern prevents an application from repeatedly trying to
execute an operation that's likely to fail. It wraps calls to external services
and monitors for failures. After a threshold of failures, it "opens" the circuit
and fails fast for subsequent requests, then periodically allows test requests
to check if the underlying issue has resolved.

States:
    CLOSED: Normal operation, requests pass through
    OPEN: Too many failures, requests fail fast
    HALF_OPEN: Testing if service has recovered

Usage:
    breaker = CircuitBreaker(name="slicer", failure_threshold=5, recovery_timeout=30)

    @with_circuit_breaker(breaker)
    def call_external_service():
        # ... code that might fail
"""

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import TypeVar

from slicer_mcp.constants import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
)

logger = logging.getLogger("slicer-mcp")

# Type variable for generic return type
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing fast, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery, allowing one request


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    def __init__(self, message: str, breaker_name: str, recovery_timeout: float):
        self.message = message
        self.breaker_name = breaker_name
        self.recovery_timeout = recovery_timeout
        super().__init__(self.message)


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    Monitors for failures and "trips" the circuit after a threshold is reached,
    causing subsequent requests to fail fast. After a recovery timeout, allows
    a test request to check if the service has recovered.

    Thread-safe implementation using locks for state management.

    Attributes:
        name: Identifier for this circuit breaker (for logging)
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for logging and error messages
            failure_threshold: Open circuit after this many consecutive failures
            recovery_timeout: Seconds to wait in OPEN state before testing recovery
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

        logger.debug(
            f"CircuitBreaker '{name}' initialized: "
            f"threshold={failure_threshold}, recovery={recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout.

        If circuit is OPEN and recovery timeout has elapsed, transitions to HALF_OPEN.

        Returns:
            Current CircuitState
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        f"CircuitBreaker '{self.name}' transitioning OPEN -> HALF_OPEN "
                        f"after {elapsed:.1f}s"
                    )
                    self._state = CircuitState.HALF_OPEN
            return self._state

    @property
    def failure_count(self) -> int:
        """Get current consecutive failure count."""
        with self._lock:
            return self._failure_count

    def record_success(self) -> None:
        """Record a successful operation, resetting failure count and closing circuit.

        Call this after a successful operation to reset the circuit breaker.
        If circuit was HALF_OPEN, this closes it (service has recovered).
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"CircuitBreaker '{self.name}' transitioning HALF_OPEN -> CLOSED")
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed operation, potentially opening the circuit.

        Call this after a failed operation. Increments failure count and opens
        the circuit if threshold is reached. If circuit was HALF_OPEN (test
        request failed), immediately reopens the circuit.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Test request failed, reopen circuit
                logger.warning(
                    f"CircuitBreaker '{self.name}' transitioning HALF_OPEN -> OPEN "
                    f"(test request failed)"
                )
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                # Threshold reached, open circuit
                logger.warning(
                    f"CircuitBreaker '{self.name}' transitioning CLOSED -> OPEN "
                    f"after {self._failure_count} failures"
                )
                self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request should proceed, False if circuit is open
        """
        current_state = self.state  # Property handles timeout check
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            # Allow test request in half-open state
            return True
        else:  # OPEN
            return False

    def reset(self) -> None:
        """Reset circuit breaker to initial state.

        Useful for testing or manual intervention.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            logger.info(f"CircuitBreaker '{self.name}' manually reset")


def with_circuit_breaker(
    breaker: CircuitBreaker,
    failure_exceptions: tuple[type[Exception], ...] = (ConnectionError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap a function with circuit breaker protection.

    If the circuit is open, raises CircuitOpenError immediately without calling
    the wrapped function. Records successes and failures to manage circuit state.

    Only exceptions specified in failure_exceptions will trip the circuit.
    Other exceptions (validation errors, timeouts, etc.) pass through without
    affecting circuit state.

    Args:
        breaker: CircuitBreaker instance to use
        failure_exceptions: Tuple of exception types that should trip the circuit.
            Defaults to (ConnectionError,). Only these exceptions count as failures.

    Returns:
        Decorator function

    Example:
        breaker = CircuitBreaker(name="api", failure_threshold=5)

        @with_circuit_breaker(breaker, failure_exceptions=(ConnectionError, IOError))
        def call_api():
            return requests.get("http://api.example.com")
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not breaker.allow_request():
                raise CircuitOpenError(
                    f"Circuit breaker '{breaker.name}' is OPEN. "
                    f"Service unavailable, will retry in {breaker.recovery_timeout}s",
                    breaker_name=breaker.name,
                    recovery_timeout=breaker.recovery_timeout,
                )

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except failure_exceptions:
                # Only connection-related exceptions trip the circuit
                breaker.record_failure()
                raise
            # Other exceptions pass through without affecting circuit state

        return wrapper

    return decorator
