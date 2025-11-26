"""Unit tests for SlicerClient with mocked HTTP responses."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout

from slicer_mcp.slicer_client import SlicerClient, SlicerConnectionError


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

            with pytest.raises(SlicerConnectionError) as exc_info:
                slicer_client.health_check()

            assert "Could not connect" in str(exc_info.value)


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


# Integration Tests (require running Slicer instance)
# ====================================================

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
