"""Unit tests for MCP resource implementations."""

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.resources import (
    _iso_timestamp,
    get_scene_resource,
    get_status_resource,
    get_volumes_resource,
)
from slicer_mcp.slicer_client import SlicerConnectionError


class TestIsoTimestamp:
    """Test _iso_timestamp helper function."""

    def test_iso_timestamp_returns_string(self):
        """_iso_timestamp should return a string."""
        ts = _iso_timestamp()
        assert isinstance(ts, str)

    def test_iso_timestamp_ends_with_z_suffix(self):
        """_iso_timestamp should end with 'Z' suffix for UTC."""
        ts = _iso_timestamp()
        assert ts.endswith("Z"), f"Expected timestamp to end with 'Z', got: {ts}"

    def test_iso_timestamp_contains_t_separator(self):
        """_iso_timestamp should contain 'T' separator between date and time."""
        ts = _iso_timestamp()
        assert "T" in ts, f"Expected 'T' separator in timestamp, got: {ts}"

    def test_iso_timestamp_is_parseable(self):
        """_iso_timestamp should return a parseable ISO 8601 timestamp."""
        ts = _iso_timestamp()

        # Replace 'Z' with '+00:00' for Python's fromisoformat
        parseable_ts = ts.replace("Z", "+00:00")

        # This should not raise an exception
        parsed = datetime.fromisoformat(parseable_ts)
        assert parsed is not None

    def test_iso_timestamp_format_structure(self):
        """_iso_timestamp should follow YYYY-MM-DDTHH:MM:SSZ format."""
        ts = _iso_timestamp()

        # Should be exactly 20 characters: 2024-01-15T14:30:00Z
        assert len(ts) == 20, f"Expected 20 characters, got {len(ts)}: {ts}"

        # Check structure
        assert ts[4] == "-", f"Expected '-' at position 4, got: {ts}"
        assert ts[7] == "-", f"Expected '-' at position 7, got: {ts}"
        assert ts[10] == "T", f"Expected 'T' at position 10, got: {ts}"
        assert ts[13] == ":", f"Expected ':' at position 13, got: {ts}"
        assert ts[16] == ":", f"Expected ':' at position 16, got: {ts}"
        assert ts[19] == "Z", f"Expected 'Z' at position 19, got: {ts}"

    def test_iso_timestamp_is_utc(self):
        """_iso_timestamp should return UTC time (verified by 'Z' suffix)."""
        ts = _iso_timestamp()

        # The 'Z' suffix indicates Zulu time (UTC)
        assert ts.endswith("Z")

        # Parse and verify timezone is UTC
        parseable_ts = ts.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(parseable_ts)

        # The parsed datetime should have tzinfo indicating UTC
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_iso_timestamp_seconds_precision(self):
        """_iso_timestamp should have seconds precision (not milliseconds)."""
        ts = _iso_timestamp()

        # Should NOT contain a decimal point (no sub-second precision)
        assert "." not in ts, f"Expected no decimal point in timestamp, got: {ts}"


class TestGetSceneResource:
    """Test get_scene_resource function."""

    def test_returns_json_with_nodes(self):
        """get_scene_resource should return JSON with node count and nodes."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_scene_nodes.return_value = [
                {
                    "id": "vtkMRMLScalarVolumeNode1",
                    "name": "MRHead",
                    "type": "vtkMRMLScalarVolumeNode",
                }
            ]
            mock_get_client.return_value = mock_client
            result = json.loads(get_scene_resource())
            assert result["node_count"] == 1
            assert result["nodes"][0]["name"] == "MRHead"

    def test_connection_error_raises(self):
        """get_scene_resource should propagate SlicerConnectionError."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_scene_nodes.side_effect = SlicerConnectionError("fail")
            mock_get_client.return_value = mock_client
            with pytest.raises(SlicerConnectionError):
                get_scene_resource()


class TestGetVolumesResource:
    """Test get_volumes_resource function."""

    def test_returns_json_with_volumes(self):
        """get_volumes_resource should return JSON with volume data."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"volumes": [], "total_count": 0}',
            }
            mock_get_client.return_value = mock_client
            result = json.loads(get_volumes_resource())
            assert result["total_count"] == 0

    def test_connection_error_raises(self):
        """get_volumes_resource should propagate SlicerConnectionError."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("fail")
            mock_get_client.return_value = mock_client
            with pytest.raises(SlicerConnectionError):
                get_volumes_resource()


class TestGetStatusResource:
    """Test get_status_resource function."""

    def test_connected_status(self):
        """get_status_resource should return connected status with Slicer info."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.health_check.return_value = {
                "connected": True,
                "webserver_url": "http://localhost:2016",
                "response_time_ms": 10,
            }
            mock_client.exec_python.return_value = {
                "success": True,
                "result": (
                    '{"slicer_version": "5.6.2",'
                    ' "scene_loaded": true,'
                    ' "python_available": true}'
                ),
            }
            mock_get_client.return_value = mock_client
            result = json.loads(get_status_resource())
            assert result["connected"] is True
            assert result["slicer_version"] == "5.6.2"

    def test_disconnected_status(self):
        """get_status_resource should return disconnected status on connection error."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.base_url = "http://localhost:2016"
            mock_client.health_check.side_effect = SlicerConnectionError("fail")
            mock_get_client.return_value = mock_client
            result = json.loads(get_status_resource())
            assert result["connected"] is False
            assert result["error"] == "fail"
