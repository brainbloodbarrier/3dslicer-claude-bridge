"""Unit tests for SlicerClient with mocked HTTP responses."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout

from slicer_mcp.slicer_client import SlicerClient, SlicerConnectionError, SlicerTimeoutError


@pytest.fixture
def slicer_client():
    """Create a SlicerClient instance for testing."""
    return SlicerClient(base_url="http://localhost:2016", timeout=30)


class TestSlicerClientInit:
    """Test SlicerClient initialization."""

    def test_default_initialization(self):
        """Test client initializes with default values."""
        client = SlicerClient()
        assert client.base_url == "http://localhost:2016"
        assert client.timeout == 30

    def test_custom_initialization(self):
        """Test client initializes with custom values."""
        client = SlicerClient(base_url="http://localhost:3000", timeout=60)
        assert client.base_url == "http://localhost:3000"
        assert client.timeout == 60

    def test_base_url_trailing_slash_removed(self):
        """Test trailing slash is removed from base URL."""
        client = SlicerClient(base_url="http://localhost:2016/")
        assert client.base_url == "http://localhost:2016"

    def test_environment_variable_slicer_url(self, monkeypatch):
        """Test SLICER_URL environment variable is respected."""
        monkeypatch.setenv('SLICER_URL', 'http://custom-slicer:8080')
        client = SlicerClient()
        assert client.base_url == "http://custom-slicer:8080"

    def test_environment_variable_slicer_timeout(self, monkeypatch):
        """Test SLICER_TIMEOUT environment variable is respected."""
        monkeypatch.setenv('SLICER_TIMEOUT', '120')
        client = SlicerClient()
        assert client.timeout == 120

    def test_explicit_params_override_env_vars(self, monkeypatch):
        """Test explicit parameters override environment variables."""
        monkeypatch.setenv('SLICER_URL', 'http://env-url:9999')
        monkeypatch.setenv('SLICER_TIMEOUT', '999')
        client = SlicerClient(base_url="http://explicit:1234", timeout=45)
        assert client.base_url == "http://explicit:1234"
        assert client.timeout == 45


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_success(self, slicer_client):
        """Test successful health check."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.health_check()

            assert result["connected"] is True
            assert result["webserver_url"] == "http://localhost:2016"
            assert "response_time_ms" in result
            mock_get.assert_called_once()

    def test_health_check_connection_error(self, slicer_client):
        """Test health check with connection error."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError) as exc_info:
                slicer_client.health_check()

            assert "Could not connect" in str(exc_info.value)

    def test_health_check_timeout(self, slicer_client):
        """Test health check with timeout."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_get.side_effect = Timeout("Request timeout")

            with pytest.raises(SlicerTimeoutError) as exc_info:
                slicer_client.health_check()

            assert "timed out" in str(exc_info.value)


class TestExecPython:
    """Test exec_python method."""

    def test_exec_python_success(self, slicer_client):
        """Test successful Python code execution."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "42"
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = slicer_client.exec_python("print('Hello')")

            assert result["success"] is True
            assert result["result"] == "42"
            mock_post.assert_called_once()

    def test_exec_python_connection_error(self, slicer_client):
        """Test Python execution with connection error."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_post.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError) as exc_info:
                slicer_client.exec_python("print('Hello')")

            assert "Could not connect" in str(exc_info.value)


class TestGetScreenshot:
    """Test get_screenshot method."""

    def test_get_screenshot_default(self, slicer_client):
        """Test screenshot capture with default parameters."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'\x89PNG\r\n\x1a\n...'  # PNG magic bytes
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.get_screenshot()

            assert result.startswith(b'\x89PNG')
            mock_get.assert_called_once()

    def test_get_screenshot_with_scroll(self, slicer_client):
        """Test screenshot capture with scroll position."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'\x89PNG\r\n\x1a\n...'
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.get_screenshot(view="Yellow", scroll_to=0.5)

            assert result.startswith(b'\x89PNG')
            call_args = mock_get.call_args
            assert "view=Yellow" in call_args[0][0]
            assert "scrollTo=0.5" in call_args[0][0]

    def test_get_screenshot_connection_error(self, slicer_client):
        """Test screenshot capture with connection error."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError) as exc_info:
                slicer_client.get_screenshot()

            assert "Could not connect" in str(exc_info.value)


class TestGet3DScreenshot:
    """Test get_3d_screenshot method."""

    def test_get_3d_screenshot_default(self, slicer_client):
        """Test 3D screenshot capture with default parameters."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'\x89PNG\r\n\x1a\n...'
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.get_3d_screenshot()

            assert result.startswith(b'\x89PNG')
            assert "/slicer/threeD" in mock_get.call_args[0][0]

    def test_get_3d_screenshot_with_axis(self, slicer_client):
        """Test 3D screenshot capture with look_from_axis."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'\x89PNG\r\n\x1a\n...'
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.get_3d_screenshot(look_from_axis="left")

            assert result.startswith(b'\x89PNG')
            assert "lookFromAxis=left" in mock_get.call_args[0][0]


class TestGetSceneNodes:
    """Test get_scene_nodes method."""

    def test_get_scene_nodes_success(self, slicer_client):
        """Test successful scene nodes retrieval."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            # Mock names response (JSON array)
            mock_names_response = Mock()
            mock_names_response.status_code = 200
            mock_names_response.text = json.dumps(["Brain-MRI", "Segmentation"])
            mock_names_response.raise_for_status = Mock()

            # Mock IDs response (JSON array)
            mock_ids_response = Mock()
            mock_ids_response.status_code = 200
            mock_ids_response.text = json.dumps(["vtkMRMLScalarVolumeNode1", "vtkMRMLSegmentationNode1"])
            mock_ids_response.raise_for_status = Mock()

            mock_get.side_effect = [mock_names_response, mock_ids_response]

            result = slicer_client.get_scene_nodes()

            assert len(result) == 2
            assert result[0]["id"] == "vtkMRMLScalarVolumeNode1"
            assert result[0]["name"] == "Brain-MRI"
            assert result[0]["type"] == "vtkMRMLScalarVolumeNode"

    def test_get_scene_nodes_empty(self, slicer_client):
        """Test scene nodes retrieval with empty scene."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = json.dumps([])
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.get_scene_nodes()

            assert result == []


class TestLoadSampleData:
    """Test load_sample_data method."""

    def test_load_sample_data_success(self, slicer_client):
        """Test successful sample data loading."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.load_sample_data("MRHead")

            assert result["success"] is True
            assert result["dataset_name"] == "MRHead"
            assert "MRHead" in result["message"]

    def test_load_sample_data_connection_error(self, slicer_client):
        """Test sample data loading with connection error."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError) as exc_info:
                slicer_client.load_sample_data("MRHead")

            assert "Could not connect" in str(exc_info.value)


class TestSetLayout:
    """Test set_layout method."""

    def test_set_layout_success(self, slicer_client):
        """Test successful layout change."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.set_layout("FourUp", "full")

            assert result["success"] is True
            assert result["layout"] == "FourUp"
            assert result["gui_mode"] == "full"

    def test_set_layout_viewers_only(self, slicer_client):
        """Test layout change with viewers only mode."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = slicer_client.set_layout("OneUp3D", "viewers")

            assert result["success"] is True
            assert result["layout"] == "OneUp3D"
            assert result["gui_mode"] == "viewers"
            # Verify params were passed correctly
            call_args = mock_get.call_args
            assert call_args.kwargs["params"]["contents"] == "viewers"
            assert call_args.kwargs["params"]["viewersLayout"] == "OneUp3D"


# =============================================================================
# Thread Safety Tests (Batch 1 Fix 1.1)
# =============================================================================

class TestSingletonThreadSafety:
    """Test thread-safe singleton pattern for SlicerClient."""

    def test_concurrent_get_client_returns_same_instance(self):
        """Test multiple threads calling get_client() get the same instance."""
        import threading
        from slicer_mcp.slicer_client import get_client, reset_client

        # Reset to ensure clean state
        reset_client()

        instances = []
        errors = []
        barrier = threading.Barrier(10)  # Ensure all threads start together

        def get_instance():
            try:
                barrier.wait()  # Synchronize thread start
                client = get_client()
                instances.append(id(client))
            except Exception as e:
                errors.append(e)

        # Spawn 10 threads to get the client simultaneously
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert not errors, f"Errors occurred: {errors}"

        # Verify all threads got the same instance (all IDs should be identical)
        assert len(set(instances)) == 1, f"Got {len(set(instances))} different instances"

    def test_singleton_reset_allows_new_instance(self):
        """Test reset_client() allows creation of a new singleton."""
        from slicer_mcp.slicer_client import get_client, reset_client

        client1 = get_client()
        reset_client()
        client2 = get_client()

        # Should be different instances after reset
        assert id(client1) != id(client2)


# =============================================================================
# Retry Decorator Tests (Batch 1 Fix 1.2)
# =============================================================================

class TestScreenshotRetry:
    """Test retry decorator on screenshot methods."""

    def test_get_screenshot_retries_on_connection_error(self, slicer_client):
        """Test get_screenshot retries on SlicerConnectionError."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            # First 3 calls fail with ConnectionError, 4th succeeds
            mock_success = Mock()
            mock_success.status_code = 200
            mock_success.content = b'\x89PNG\r\n\x1a\n...'
            mock_success.raise_for_status = Mock()

            mock_get.side_effect = [
                ConnectionError("Connection refused"),
                ConnectionError("Connection refused"),
                ConnectionError("Connection refused"),
                mock_success,
            ]

            result = slicer_client.get_screenshot()

            assert result.startswith(b'\x89PNG')
            # Should have called 4 times (initial + 3 retries = 4 attempts)
            assert mock_get.call_count == 4
            # Should have slept 3 times (between retries)
            assert mock_sleep.call_count == 3

    def test_get_3d_screenshot_retries_on_connection_error(self, slicer_client):
        """Test get_3d_screenshot retries on SlicerConnectionError."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            mock_success = Mock()
            mock_success.status_code = 200
            mock_success.content = b'\x89PNG\r\n\x1a\n...'
            mock_success.raise_for_status = Mock()

            # Fail twice, then succeed
            mock_get.side_effect = [
                ConnectionError("Connection refused"),
                ConnectionError("Connection refused"),
                mock_success,
            ]

            result = slicer_client.get_3d_screenshot()

            assert result.startswith(b'\x89PNG')
            assert mock_get.call_count == 3

    def test_get_full_screenshot_retries_on_connection_error(self, slicer_client):
        """Test get_full_screenshot retries on SlicerConnectionError."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            mock_success = Mock()
            mock_success.status_code = 200
            mock_success.content = b'\x89PNG\r\n\x1a\n...'
            mock_success.raise_for_status = Mock()

            # Fail once, then succeed
            mock_get.side_effect = [
                ConnectionError("Connection refused"),
                mock_success,
            ]

            result = slicer_client.get_full_screenshot()

            assert result.startswith(b'\x89PNG')
            assert mock_get.call_count == 2

    def test_get_screenshot_exhausts_retries(self, slicer_client):
        """Test get_screenshot fails after exhausting all retries."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            # All calls fail
            mock_get.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError):
                slicer_client.get_screenshot()

            # Should have tried 4 times total (initial + 3 retries)
            assert mock_get.call_count == 4


# Integration Tests (require running Slicer instance)
# ====================================================

# =============================================================================
# Retry Exhaustion Tests (Batch 6 Fix 6.1)
# =============================================================================

class TestRetryExhaustion:
    """Test retry behavior when all attempts fail."""

    def test_health_check_exhausts_all_retries(self, slicer_client):
        """Test health check fails after exhausting all retries."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            mock_get.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError):
                slicer_client.health_check()

            # Should have been called 4 times (initial + 3 retries)
            assert mock_get.call_count == 4
            # Should have slept 3 times with exponential backoff
            assert mock_sleep.call_count == 3

    def test_exec_python_exhausts_all_retries(self, slicer_client):
        """Test exec_python fails after exhausting all retries."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            mock_post.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError):
                slicer_client.exec_python("print('test')")

            assert mock_post.call_count == 4

    def test_retry_exponential_backoff_timing(self, slicer_client):
        """Test exponential backoff delays are correct."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            mock_get.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError):
                slicer_client.health_check()

            # Verify backoff delays: 1s, 2s, 4s (exponential)
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls == [1.0, 2.0, 4.0]


# =============================================================================
# Error Handler Edge Cases (Batch 6 Fix 6.4)
# =============================================================================

class TestErrorHandlerEdgeCases:
    """Test edge cases in error handling."""

    def test_timeout_not_retried(self, slicer_client):
        """Test that Timeout errors are NOT retried (Slicer may be frozen)."""
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get:
            mock_get.side_effect = Timeout("Request timeout")

            with pytest.raises(SlicerTimeoutError):
                slicer_client.health_check()

            # Should only be called once - no retries for timeout
            assert mock_get.call_count == 1

    def test_generic_request_exception_retried(self, slicer_client):
        """Test that generic RequestException is converted to SlicerConnectionError and retried."""
        from requests.exceptions import RequestException
        with patch('slicer_mcp.slicer_client.requests.get') as mock_get, \
             patch('slicer_mcp.slicer_client.time.sleep') as mock_sleep:
            mock_get.side_effect = RequestException("Generic error")

            with pytest.raises(SlicerConnectionError):
                slicer_client.health_check()

            # Should be called 4 times (initial + 3 retries)
            # Generic RequestException is converted to SlicerConnectionError which is retried
            assert mock_get.call_count == 4
            assert mock_sleep.call_count == 3


# =============================================================================
# Version Checking Tests (Batch 9: Slicer Version Compatibility)
# =============================================================================

class TestVersionChecking:
    """Test Slicer version checking functionality."""

    def test_get_slicer_version_returns_string(self, slicer_client):
        """Test get_slicer_version returns a version string."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "'5.6.2'"  # Slicer returns quoted string
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            version = slicer_client.get_slicer_version()

            assert version == "5.6.2"
            mock_post.assert_called_once()

    def test_get_slicer_version_strips_quotes(self, slicer_client):
        """Test version string is cleaned of quotes."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '"5.4.0"'  # Double-quoted
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            version = slicer_client.get_slicer_version()

            assert version == "5.4.0"

    def test_check_version_compatibility_compatible_tested(self, slicer_client):
        """Test compatible and tested version returns correct status."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "'5.6.2'"
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = slicer_client.check_version_compatibility()

            assert result["version"] == "5.6.2"
            assert result["compatible"] is True
            assert result["tested"] is True
            assert result["warning"] is None
            assert result["minimum_required"] == "5.0.0"

    def test_check_version_compatibility_compatible_untested(self, slicer_client):
        """Test compatible but untested version returns warning."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "'5.8.0'"  # Not in tested versions
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = slicer_client.check_version_compatibility()

            assert result["version"] == "5.8.0"
            assert result["compatible"] is True
            assert result["tested"] is False
            assert result["warning"] is not None
            assert "not been tested" in result["warning"]

    def test_check_version_compatibility_incompatible(self, slicer_client):
        """Test incompatible version returns warning."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "'4.11.0'"  # Below minimum
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = slicer_client.check_version_compatibility()

            assert result["version"] == "4.11.0"
            assert result["compatible"] is False
            assert result["tested"] is False
            assert result["warning"] is not None
            assert "below minimum" in result["warning"]

    def test_check_version_compatibility_dev_version(self, slicer_client):
        """Test development version (e.g., 5.7.0-2024-01-01) is handled."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "'5.7.0-2024-01-01'"
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = slicer_client.check_version_compatibility()

            assert result["version"] == "5.7.0-2024-01-01"
            assert result["compatible"] is True  # 5.7.0 > 5.0.0
            assert result["tested"] is False
            assert result["warning"] is not None

    def test_check_version_compatibility_connection_error(self, slicer_client):
        """Test version check raises error if Slicer not connected."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_post.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError):
                slicer_client.check_version_compatibility()

    def test_get_slicer_version_whitespace_handling(self, slicer_client):
        """Test version string handles whitespace."""
        with patch('slicer_mcp.slicer_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "  '5.6.1'  \n"  # Extra whitespace
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            version = slicer_client.get_slicer_version()

            assert version == "5.6.1"


@pytest.mark.integration
class TestSlicerIntegration:
    """Integration tests requiring a running Slicer instance."""

    def test_real_connection(self):
        """Test real connection to Slicer WebServer.

        Requires: Slicer running with WebServer extension on localhost:2016
        """
        client = SlicerClient()

        try:
            health = client.health_check()
            assert health["connected"] is True
            assert health["response_time_ms"] > 0
        except SlicerConnectionError:
            pytest.skip("Slicer not running or WebServer not enabled")

    def test_real_python_execution(self):
        """Test real Python code execution in Slicer.

        Requires: Slicer running with WebServer extension on localhost:2016
        Note: Slicer's /slicer/exec returns result as text, expressions may return {}
        """
        client = SlicerClient()

        try:
            # Use a statement that produces output Slicer can return
            result = client.exec_python("import slicer; slicer.app.applicationVersion")
            assert result["success"] is True
            # Slicer returns {} for most expressions, check we got a response
            assert "result" in result
        except SlicerConnectionError:
            pytest.skip("Slicer not running or WebServer not enabled")
