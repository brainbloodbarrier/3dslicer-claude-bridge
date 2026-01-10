"""Shared pytest fixtures for Slicer MCP Bridge tests."""

from unittest.mock import Mock

import pytest

from slicer_mcp.slicer_client import SlicerClient


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker before each test to ensure test isolation.

    This fixture runs automatically for all tests to prevent state leakage
    between tests that might trip the circuit breaker.
    """
    from slicer_mcp.slicer_client import reset_circuit_breaker as reset_cb

    reset_cb()
    yield
    reset_cb()


@pytest.fixture
def slicer_client():
    """Create a SlicerClient instance for testing.

    Returns:
        SlicerClient configured with default test settings
    """
    return SlicerClient(base_url="http://localhost:2016", timeout=30)


@pytest.fixture
def mock_response():
    """Create a mock HTTP response for testing.

    Returns:
        Mock object configured as a successful HTTP response
    """
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.text = "{}"
    return response


@pytest.fixture
def mock_slicer_exec_result():
    """Create a mock result from Slicer Python execution.

    Returns:
        Dict mimicking successful Python execution result
    """
    return {
        "success": True,
        "result": "{}",
        "stdout": "",
        "stderr": "",
    }
