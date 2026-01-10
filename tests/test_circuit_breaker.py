"""Tests for circuit breaker pattern implementation."""

import time

import pytest

from slicer_mcp.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    with_circuit_breaker,
)


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""

    def test_starts_in_closed_state(self):
        """Circuit breaker should start in CLOSED state."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_stays_closed_below_threshold(self):
        """Circuit should remain CLOSED if failures below threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2

    def test_opens_at_failure_threshold(self):
        """Circuit should OPEN when failure threshold is reached."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_success_resets_failure_count(self):
        """Successful operation should reset failure count."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_success_closes_half_open_circuit(self):
        """Success in HALF_OPEN state should close circuit."""
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        # Success closes the circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in HALF_OPEN state should reopen circuit."""
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        # Failure reopens circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Before timeout
        time.sleep(0.02)
        assert cb.state == CircuitState.OPEN

        # After timeout
        time.sleep(0.05)
        assert cb.state == CircuitState.HALF_OPEN


class TestCircuitBreakerAllowRequest:
    """Test allow_request behavior."""

    def test_allows_request_when_closed(self):
        """Should allow requests when circuit is CLOSED."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.allow_request() is True

    def test_rejects_request_when_open(self):
        """Should reject requests when circuit is OPEN."""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_allows_test_request_when_half_open(self):
        """Should allow test request when circuit is HALF_OPEN."""
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True


class TestCircuitBreakerReset:
    """Test manual reset functionality."""

    def test_reset_closes_open_circuit(self):
        """Manual reset should close an OPEN circuit."""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_reset_clears_failure_count(self):
        """Manual reset should clear failure count."""
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 3

        cb.reset()
        assert cb.failure_count == 0


class TestCircuitBreakerDecorator:
    """Test the with_circuit_breaker decorator."""

    def test_decorator_passes_through_on_closed(self):
        """Decorator should pass through calls when circuit is CLOSED."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        @with_circuit_breaker(cb)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_decorator_records_success(self):
        """Decorator should record success after successful call."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()  # Start with one failure
        assert cb.failure_count == 1

        @with_circuit_breaker(cb)
        def successful_function():
            return "success"

        successful_function()
        assert cb.failure_count == 0  # Reset after success

    def test_decorator_records_failure(self):
        """Decorator should record failure when function raises a failure exception."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        @with_circuit_breaker(cb, failure_exceptions=(ConnectionError,))
        def failing_function():
            raise ConnectionError("test error")

        with pytest.raises(ConnectionError):
            failing_function()

        assert cb.failure_count == 1

    def test_decorator_ignores_non_failure_exceptions(self):
        """Decorator should not trip circuit for non-failure exceptions."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        @with_circuit_breaker(cb, failure_exceptions=(ConnectionError,))
        def failing_function():
            raise ValueError("validation error")

        with pytest.raises(ValueError):
            failing_function()

        assert cb.failure_count == 0  # ValueError shouldn't trip the circuit

    def test_decorator_raises_circuit_open_error(self):
        """Decorator should raise CircuitOpenError when circuit is OPEN."""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()  # Open the circuit

        @with_circuit_breaker(cb)
        def some_function():
            return "should not reach"

        with pytest.raises(CircuitOpenError) as exc_info:
            some_function()

        assert "OPEN" in str(exc_info.value)
        assert exc_info.value.breaker_name == "test"

    def test_decorator_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring."""
        cb = CircuitBreaker(name="test")

        @with_circuit_breaker(cb)
        def documented_function():
            """This is the docstring."""
            return True

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."

    def test_decorator_opens_circuit_after_threshold(self):
        """Decorator should open circuit after failure threshold reached."""
        cb = CircuitBreaker(name="test", failure_threshold=2)

        @with_circuit_breaker(cb, failure_exceptions=(ConnectionError,))
        def failing_function():
            raise ConnectionError("failure")

        # First failure
        with pytest.raises(ConnectionError):
            failing_function()
        assert cb.state == CircuitState.CLOSED

        # Second failure - opens circuit
        with pytest.raises(ConnectionError):
            failing_function()
        assert cb.state == CircuitState.OPEN

        # Third call - circuit is open
        with pytest.raises(CircuitOpenError):
            failing_function()


class TestCircuitBreakerThreadSafety:
    """Test thread safety of circuit breaker."""

    def test_concurrent_failures(self):
        """Multiple threads recording failures should be thread-safe."""
        import threading

        cb = CircuitBreaker(name="test", failure_threshold=100)
        barrier = threading.Barrier(10)

        def record_failures():
            barrier.wait()
            for _ in range(10):
                cb.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 threads * 10 failures = 100 failures
        assert cb.failure_count == 100
        assert cb.state == CircuitState.OPEN

    def test_concurrent_state_checks(self):
        """Multiple threads checking state should be thread-safe."""
        import threading

        cb = CircuitBreaker(name="test", failure_threshold=5, recovery_timeout=0.01)

        # Open the circuit
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for half-open
        time.sleep(0.02)

        results = []
        barrier = threading.Barrier(10)

        def check_state():
            barrier.wait()
            results.append(cb.state)

        threads = [threading.Thread(target=check_state) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be HALF_OPEN (state check is atomic)
        assert all(r == CircuitState.HALF_OPEN for r in results)


class TestCircuitOpenError:
    """Test CircuitOpenError exception."""

    def test_error_contains_breaker_info(self):
        """Error should contain breaker name and recovery timeout."""
        error = CircuitOpenError(
            message="Circuit is open", breaker_name="test_breaker", recovery_timeout=30.0
        )

        assert error.breaker_name == "test_breaker"
        assert error.recovery_timeout == 30.0
        assert "Circuit is open" in str(error)
