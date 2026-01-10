"""Tests for MCP protocol layer - tools/resources listing and schema validation."""

import json
from unittest.mock import Mock, patch

import pytest

# Import the server module to test tool/resource registration
from slicer_mcp.server import mcp


class TestMCPToolsDiscovery:
    """Test MCP tools are properly registered and discoverable."""

    def test_server_has_tools_registered(self):
        """Verify tools are registered with the MCP server."""
        # FastMCP stores tools in _tool_manager
        assert hasattr(mcp, "_tool_manager") or hasattr(mcp, "tools")

    def test_capture_screenshot_tool_exists(self):
        """Verify capture_screenshot tool is registered."""
        # Check tool registration by looking at server internals
        # FastMCP registers tools via decorators
        from slicer_mcp.server import capture_screenshot

        assert callable(capture_screenshot)

    def test_list_scene_nodes_tool_exists(self):
        """Verify list_scene_nodes tool is registered."""
        from slicer_mcp.server import list_scene_nodes

        assert callable(list_scene_nodes)

    def test_execute_python_tool_exists(self):
        """Verify execute_python tool is registered."""
        from slicer_mcp.server import execute_python

        assert callable(execute_python)

    def test_measure_volume_tool_exists(self):
        """Verify measure_volume tool is registered."""
        from slicer_mcp.server import measure_volume

        assert callable(measure_volume)

    def test_list_sample_data_tool_exists(self):
        """Verify list_sample_data tool is registered."""
        from slicer_mcp.server import list_sample_data

        assert callable(list_sample_data)

    def test_load_sample_data_tool_exists(self):
        """Verify load_sample_data tool is registered."""
        from slicer_mcp.server import load_sample_data

        assert callable(load_sample_data)

    def test_set_layout_tool_exists(self):
        """Verify set_layout tool is registered."""
        from slicer_mcp.server import set_layout

        assert callable(set_layout)


class TestMCPResourcesDiscovery:
    """Test MCP resources are properly registered and discoverable."""

    def test_scene_resource_exists(self):
        """Verify slicer://scene resource is registered."""
        from slicer_mcp.server import get_scene

        assert callable(get_scene)

    def test_volumes_resource_exists(self):
        """Verify slicer://volumes resource is registered."""
        from slicer_mcp.server import get_volumes

        assert callable(get_volumes)

    def test_status_resource_exists(self):
        """Verify slicer://status resource is registered."""
        from slicer_mcp.server import get_status

        assert callable(get_status)


class TestToolParameterValidation:
    """Test tool parameter validation without Slicer connection."""

    def test_capture_screenshot_validates_view_type(self):
        """Test capture_screenshot rejects invalid view_type."""
        from slicer_mcp.tools import capture_screenshot

        with pytest.raises(ValueError) as exc_info:
            capture_screenshot(view_type="invalid_view")

        assert "Invalid view_type" in str(exc_info.value)
        assert "axial" in str(exc_info.value)  # Shows valid options

    def test_load_sample_data_validates_dataset(self):
        """Test load_sample_data rejects invalid dataset names."""
        from slicer_mcp.tools import FALLBACK_SAMPLE_DATASETS, load_sample_data

        with pytest.raises(ValueError) as exc_info:
            load_sample_data(dataset_name="InvalidDataset")

        assert "Invalid dataset_name" in str(exc_info.value)
        # Should show available datasets in error
        for dataset in FALLBACK_SAMPLE_DATASETS[:2]:  # Check at least some
            assert dataset in str(exc_info.value)

    def test_set_layout_validates_layout(self):
        """Test set_layout rejects invalid layout names."""
        from slicer_mcp.tools import set_layout

        with pytest.raises(ValueError) as exc_info:
            set_layout(layout="InvalidLayout")

        assert "Invalid layout" in str(exc_info.value)
        assert "FourUp" in str(exc_info.value)  # Shows valid options

    def test_set_layout_validates_gui_mode(self):
        """Test set_layout rejects invalid gui_mode."""
        from slicer_mcp.tools import set_layout

        with pytest.raises(ValueError) as exc_info:
            set_layout(layout="FourUp", gui_mode="invalid")

        assert "Invalid gui_mode" in str(exc_info.value)


class TestToolResponseFormat:
    """Test tools return correctly formatted responses (mocked Slicer)."""

    def test_capture_screenshot_response_format(self):
        """Test capture_screenshot returns expected response structure."""
        from slicer_mcp.tools import capture_screenshot

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            # Return PNG magic bytes
            mock_client.get_screenshot.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            mock_get_client.return_value = mock_client

            result = capture_screenshot(view_type="axial")

            assert "success" in result
            assert result["success"] is True
            assert "image_base64" in result
            assert "view_type" in result
            assert result["view_type"] == "axial"
            assert "content_type" in result
            assert result["content_type"] == "image/png"

    def test_list_scene_nodes_response_format(self):
        """Test list_scene_nodes returns expected response structure."""
        from slicer_mcp.tools import list_scene_nodes

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_scene_nodes.return_value = [
                {
                    "id": "vtkMRMLScalarVolumeNode1",
                    "name": "TestVolume",
                    "type": "vtkMRMLScalarVolumeNode",
                }
            ]
            mock_get_client.return_value = mock_client

            result = list_scene_nodes()

            assert "nodes" in result
            assert "total_count" in result
            assert result["total_count"] == 1
            assert len(result["nodes"]) == 1

    def test_execute_python_response_format(self):
        """Test execute_python returns expected response structure."""
        from slicer_mcp.tools import execute_python

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "42",
                "stdout": "",
                "stderr": "",
            }
            mock_get_client.return_value = mock_client

            result = execute_python(code="1 + 1")

            assert "success" in result
            assert result["success"] is True
            assert "result" in result

    def test_list_sample_data_dynamic_response_format(self):
        """Test list_sample_data returns expected response structure with dynamic discovery."""
        from slicer_mcp.slicer_client import reset_client
        from slicer_mcp.tools import list_sample_data

        reset_client()

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "MRHead",
                                "category": "BuiltIn",
                                "description": "MR head scan",
                            },
                            {
                                "name": "CTChest",
                                "category": "BuiltIn",
                                "description": "CT chest scan",
                            },
                        ],
                        "total_count": 2,
                        "source": "dynamic",
                    }
                ),
            }
            mock_get_client.return_value = mock_client

            result = list_sample_data()

            assert "datasets" in result
            assert "total_count" in result
            assert "source" in result
            assert result["source"] == "dynamic"
            assert len(result["datasets"]) == 2
            assert result["datasets"][0]["name"] == "MRHead"

    def test_list_sample_data_fallback_on_connection_error(self):
        """Test list_sample_data falls back to static list when Slicer disconnected."""
        from slicer_mcp.slicer_client import SlicerConnectionError, reset_client
        from slicer_mcp.tools import FALLBACK_SAMPLE_DATASETS, list_sample_data

        reset_client()

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")
            mock_get_client.return_value = mock_client

            result = list_sample_data()

            assert "datasets" in result
            assert "source" in result
            assert result["source"] == "fallback"
            assert "error" in result
            assert result["total_count"] == len(FALLBACK_SAMPLE_DATASETS)


class TestAuditLogging:
    """Test audit logging for Python code execution."""

    def test_audit_log_entry_on_success(self):
        """Test audit log entry is created on successful execution."""
        from slicer_mcp.slicer_client import reset_client
        from slicer_mcp.tools import audit_logger, execute_python

        reset_client()

        with (
            patch("slicer_mcp.tools.get_client") as mock_get_client,
            patch.object(audit_logger, "info") as mock_audit,
        ):
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "42",
                "stdout": "",
                "stderr": "",
            }
            mock_get_client.return_value = mock_client

            execute_python(code="1 + 1")

            # Verify audit log was called
            mock_audit.assert_called_once()
            audit_entry = json.loads(mock_audit.call_args[0][0])

            assert audit_entry["event"] == "python_execution"
            assert audit_entry["success"] is True
            assert "request_id" in audit_entry
            assert "code_hash" in audit_entry
            assert "timestamp" in audit_entry
            assert audit_entry["code_preview"] == "1 + 1"

    def test_audit_log_entry_on_failure(self):
        """Test audit log entry is created on failed execution."""
        from slicer_mcp.slicer_client import SlicerConnectionError, reset_client
        from slicer_mcp.tools import audit_logger, execute_python

        reset_client()

        with (
            patch("slicer_mcp.tools.get_client") as mock_get_client,
            patch.object(audit_logger, "info") as mock_audit,
        ):
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                execute_python(code="bad_code()")

            # Verify audit log was called with failure
            mock_audit.assert_called_once()
            audit_entry = json.loads(mock_audit.call_args[0][0])

            assert audit_entry["event"] == "python_execution"
            assert audit_entry["success"] is False
            assert "error" in audit_entry
            assert audit_entry["code_preview"] == "bad_code()"

    def test_audit_log_truncates_large_code(self):
        """Test audit log truncates large code blocks."""
        from slicer_mcp.tools import _audit_log_execution, audit_logger

        with patch.object(audit_logger, "info") as mock_audit:
            large_code = "x = 1\n" * 400  # Create large code block (2400 chars > 2000 limit)
            _audit_log_execution(large_code, "test123", success=True)

            audit_entry = json.loads(mock_audit.call_args[0][0])

            # Code preview should be truncated
            assert len(audit_entry["code_preview"]) < len(large_code)
            assert "truncated" in audit_entry["code_preview"]
            assert audit_entry["code_length"] == len(large_code)

    def test_audit_log_includes_code_hash(self):
        """Test audit log includes consistent code hash."""
        import hashlib

        from slicer_mcp.tools import _audit_log_execution, audit_logger

        with patch.object(audit_logger, "info") as mock_audit:
            code = "print('hello world')"
            expected_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

            _audit_log_execution(code, "test456", success=True)

            audit_entry = json.loads(mock_audit.call_args[0][0])
            assert audit_entry["code_hash"] == expected_hash


class TestResourceResponseFormat:
    """Test resources return correctly formatted responses (mocked Slicer)."""

    def test_status_resource_disconnected_format(self):
        """Test status resource returns valid JSON even when disconnected."""
        from slicer_mcp.resources import get_status_resource
        from slicer_mcp.slicer_client import SlicerConnectionError, reset_client

        # Reset client to ensure fresh state
        reset_client()

        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.health_check.side_effect = SlicerConnectionError("Connection failed")
            mock_client.base_url = "http://localhost:2016"
            mock_get_client.return_value = mock_client

            result_json = get_status_resource()
            result = json.loads(result_json)

            assert "connected" in result
            assert result["connected"] is False
            assert "error" in result
            assert "last_check" in result

    def test_scene_resource_format(self):
        """Test scene resource returns valid JSON structure."""
        from slicer_mcp.resources import get_scene_resource
        from slicer_mcp.slicer_client import reset_client

        reset_client()

        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_scene_nodes.return_value = []
            mock_get_client.return_value = mock_client

            result_json = get_scene_resource()
            result = json.loads(result_json)

            assert "scene_id" in result
            assert "node_count" in result
            assert "nodes" in result
            assert "modified_time" in result


# =============================================================================
# Server Tool Error Handling Tests (Bug Fix 1)
# =============================================================================


class TestServerToolErrorHandling:
    """Test that server tool wrappers handle errors correctly."""

    def test_capture_screenshot_handles_connection_error(self):
        """Test capture_screenshot returns error dict on connection failure."""
        from slicer_mcp.server import capture_screenshot
        from slicer_mcp.slicer_client import SlicerConnectionError

        with patch("slicer_mcp.tools.capture_screenshot") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection failed")

            result = capture_screenshot(view_type="axial")

            assert result["success"] is False
            assert result["error_type"] == "connection"
            assert "error" in result

    def test_capture_screenshot_handles_timeout_error(self):
        """Test capture_screenshot returns error dict on timeout."""
        from slicer_mcp.server import capture_screenshot
        from slicer_mcp.slicer_client import SlicerTimeoutError

        with patch("slicer_mcp.tools.capture_screenshot") as mock_tool:
            mock_tool.side_effect = SlicerTimeoutError("Timeout occurred")

            result = capture_screenshot(view_type="axial")

            assert result["success"] is False
            assert result["error_type"] == "timeout"

    def test_capture_screenshot_handles_circuit_open_error(self):
        """Test capture_screenshot returns error dict when circuit is open."""
        from slicer_mcp.circuit_breaker import CircuitOpenError
        from slicer_mcp.server import capture_screenshot

        with patch("slicer_mcp.tools.capture_screenshot") as mock_tool:
            mock_tool.side_effect = CircuitOpenError("Circuit open", "slicer", 30)

            result = capture_screenshot(view_type="axial")

            assert result["success"] is False
            assert result["error_type"] == "circuit_open"

    def test_list_scene_nodes_handles_error(self):
        """Test list_scene_nodes returns error dict on failure."""
        from slicer_mcp.server import list_scene_nodes
        from slicer_mcp.slicer_client import SlicerConnectionError

        with patch("slicer_mcp.tools.list_scene_nodes") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection failed")

            result = list_scene_nodes()

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_execute_python_handles_error(self):
        """Test execute_python returns error dict on failure."""
        from slicer_mcp.server import execute_python
        from slicer_mcp.slicer_client import SlicerConnectionError

        with patch("slicer_mcp.tools.execute_python") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection failed")

            result = execute_python(code="print('hello')")

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_measure_volume_handles_error(self):
        """Test measure_volume returns error dict on failure."""
        from slicer_mcp.server import measure_volume
        from slicer_mcp.slicer_client import SlicerConnectionError

        with patch("slicer_mcp.tools.measure_volume") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection failed")

            result = measure_volume(node_id="vtkMRMLSegmentationNode1")

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_list_sample_data_handles_error(self):
        """Test list_sample_data returns error dict on failure."""
        from slicer_mcp.server import list_sample_data
        from slicer_mcp.slicer_client import SlicerConnectionError

        with patch("slicer_mcp.tools.list_sample_data") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection failed")

            result = list_sample_data()

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_load_sample_data_handles_error(self):
        """Test load_sample_data returns error dict on failure."""
        from slicer_mcp.server import load_sample_data
        from slicer_mcp.slicer_client import SlicerConnectionError

        with patch("slicer_mcp.tools.load_sample_data") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection failed")

            result = load_sample_data(dataset_name="MRHead")

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_set_layout_handles_error(self):
        """Test set_layout returns error dict on failure."""
        from slicer_mcp.server import set_layout
        from slicer_mcp.slicer_client import SlicerConnectionError

        with patch("slicer_mcp.tools.set_layout") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection failed")

            result = set_layout(layout="FourUp")

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_handle_tool_error_unexpected_exception(self):
        """Test _handle_tool_error handles unexpected exceptions."""
        from slicer_mcp.server import capture_screenshot

        with patch("slicer_mcp.tools.capture_screenshot") as mock_tool:
            mock_tool.side_effect = RuntimeError("Unexpected error")

            result = capture_screenshot(view_type="axial")

            assert result["success"] is False
            assert result["error_type"] == "unexpected"
            assert "Unexpected error" in result["error"]
