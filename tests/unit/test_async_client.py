"""Unit tests for SlicerAsyncClient — async HTTP client for 3D Slicer.

Tests mirror test_slicer_client.py patterns but use pytest-asyncio
and httpx mock responses.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from slicer_mcp.core.async_client import (
    SlicerAsyncClient,
    get_async_client,
    reset_async_client,
)
from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import SlicerConnectionError, SlicerTimeoutError


@pytest.fixture(autouse=True)
def _reset_async_singleton():
    """Reset async client singleton before each test."""
    reset_async_client()
    yield
    reset_async_client()


class TestSlicerAsyncClientInit:
    """Test initialization and configuration."""

    def test_default_config(self):
        client = SlicerAsyncClient()
        assert client.base_url == "http://localhost:2016"
        assert client.timeout == 30

    def test_custom_config(self):
        client = SlicerAsyncClient(base_url="http://localhost:3000", timeout=60)
        assert client.base_url == "http://localhost:3000"
        assert client.timeout == 60

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("SLICER_URL", "http://localhost:9999")
        monkeypatch.setenv("SLICER_TIMEOUT", "120")
        client = SlicerAsyncClient()
        assert client.base_url == "http://localhost:9999"
        assert client.timeout == 120

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="http or https"):
            SlicerAsyncClient(base_url="ftp://localhost:2016")

    def test_invalid_timeout_uses_default(self, monkeypatch):
        monkeypatch.setenv("SLICER_TIMEOUT", "not_a_number")
        client = SlicerAsyncClient()
        assert client.timeout == 30

    def test_negative_timeout_uses_default(self, monkeypatch):
        monkeypatch.setenv("SLICER_TIMEOUT", "-5")
        client = SlicerAsyncClient()
        assert client.timeout == 30

    def test_trailing_slash_stripped(self):
        client = SlicerAsyncClient(base_url="http://localhost:2016/")
        assert client.base_url == "http://localhost:2016"


class TestSingleton:
    """Test get_async_client singleton behavior."""

    def test_returns_same_instance(self):
        a = get_async_client()
        b = get_async_client()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_async_client()
        reset_async_client()
        b = get_async_client()
        assert a is not b


def _mock_httpx_response(status: int = 200, text: str = "") -> httpx.Response:
    """Create an httpx.Response with a request object set (required for raise_for_status)."""
    request = httpx.Request("POST", "http://localhost:2016/slicer/exec")
    return httpx.Response(status, text=text, request=request)


class TestExecPython:
    """Test async exec_python method."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        client = SlicerAsyncClient()
        mock_resp = _mock_httpx_response(200, '{"key": "value"}')

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await client.exec_python("__execResult = 42")

        assert result["success"] is True
        assert result["result"] == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        client = SlicerAsyncClient()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ReadTimeout("timeout")
            with pytest.raises(SlicerTimeoutError):
                await client.exec_python("code")

    @pytest.mark.asyncio
    async def test_connection_error_raises(self):
        client = SlicerAsyncClient()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("refused")
            with pytest.raises(SlicerConnectionError):
                await client.exec_python("code")

    @pytest.mark.asyncio
    async def test_custom_timeout_passed(self):
        client = SlicerAsyncClient()
        mock_resp = _mock_httpx_response(200, "ok")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await client.exec_python("code", timeout=300)
            _, kwargs = mock_post.call_args
            assert kwargs["timeout"] == 300

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self):
        client = SlicerAsyncClient()
        with patch.object(client._cb, "allow_request", return_value=False):
            with pytest.raises(CircuitOpenError):
                await client.exec_python("code")


class TestHealthCheck:
    """Test async health_check method."""

    @pytest.mark.asyncio
    async def test_successful_check(self):
        client = SlicerAsyncClient()
        request = httpx.Request("GET", "http://localhost:2016/slicer/mrml")
        mock_resp = httpx.Response(200, text="<mrml>scene</mrml>", request=request)

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            result = await client.health_check(check_version=False)

        assert result["connected"] is True
        assert result["webserver_url"] == "http://localhost:2016"
        assert "response_time_ms" in result

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        client = SlicerAsyncClient()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ReadTimeout("timeout")
            with pytest.raises(SlicerTimeoutError):
                await client.health_check(check_version=False)
