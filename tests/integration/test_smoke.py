"""Integration smoke tests for the Slicer MCP bridge.

These tests verify that the real MCP bridge works end-to-end against a
running 3D Slicer instance with WebServer enabled.

Requirements:
    - 3D Slicer running with the SlicerWeb extension active
    - WebServer listening on SLICER_URL (default http://localhost:2016)

Run with:
    uv run pytest tests/integration/ -v -m integration
"""

import json

import pytest

from slicer_mcp.core.slicer_client import SlicerClient

pytestmark = pytest.mark.integration


class TestSlicerConnection:
    """Tests verifying basic connectivity to the Slicer WebServer."""

    def test_slicer_connection(self, live_client: SlicerClient) -> None:
        """Verify we can connect and get a health check response.

        The health_check method hits /slicer/mrml and returns a dict
        with 'connected': True on success.
        """
        result = live_client.health_check(check_version=False)

        assert result["connected"] is True
        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0
        assert result["webserver_url"] == live_client.base_url

    def test_get_slicer_version(self, live_client: SlicerClient) -> None:
        """Verify we can retrieve the Slicer application version string.

        Executes Python inside Slicer to read slicer.app.applicationVersion
        and confirms the result looks like a version string (digits and dots).
        """
        version = live_client.get_slicer_version()

        assert isinstance(version, str)
        assert len(version) > 0
        # Slicer versions look like "5.6.2" or "5.7.0-2024-01-01"
        parts = version.split("-")[0].split(".")
        assert len(parts) >= 2, f"Expected semver-like version, got: {version}"
        assert all(
            part.isdigit() for part in parts
        ), f"Version parts should be numeric, got: {version}"


class TestExecPython:
    """Tests verifying Python code execution inside Slicer."""

    def test_exec_python_simple(self, live_client: SlicerClient) -> None:
        """Execute a trivial assignment and verify the result comes back.

        Uses the __execResult protocol: setting __execResult in the executed
        code causes Slicer to return that value in the response.
        """
        result = live_client.exec_python("__execResult = 42")

        assert result["success"] is True
        # The result field contains the string representation
        raw = result["result"].strip()
        assert "42" in raw

    def test_exec_python_json_result(self, live_client: SlicerClient) -> None:
        """Execute code that sets __execResult to a JSON-serializable dict.

        Verifies that the bridge correctly transports structured data
        from Slicer back to the caller.
        """
        code = 'import json; __execResult = json.dumps({"key": "value", "count": 3})'
        result = live_client.exec_python(code)

        assert result["success"] is True
        raw = result["result"].strip()
        # The result may be double-encoded; try parsing
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            pytest.fail(f"Could not parse JSON from exec result: {raw!r}")

        assert parsed["key"] == "value"
        assert parsed["count"] == 3

    def test_exec_python_list_result(self, live_client: SlicerClient) -> None:
        """Execute code that returns a list via __execResult.

        Confirms list data survives the round-trip through the bridge.
        """
        code = 'import json; __execResult = json.dumps([1, 2, 3, "four"])'
        result = live_client.exec_python(code)

        assert result["success"] is True
        raw = result["result"].strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            pytest.fail(f"Could not parse list from exec result: {raw!r}")

        assert isinstance(parsed, list)
        assert len(parsed) == 4
        assert parsed[0] == 1
        assert parsed[3] == "four"


class TestSceneOperations:
    """Tests verifying scene inspection via the real Slicer bridge."""

    def test_list_scene_nodes(self, live_client: SlicerClient) -> None:
        """Call get_scene_nodes on a live Slicer and verify the response shape.

        Even an empty scene has default MRML nodes (views, layouts, etc.),
        so we expect a non-empty list.
        """
        nodes = live_client.get_scene_nodes()

        assert isinstance(nodes, list)
        # A fresh Slicer scene always has built-in nodes
        assert len(nodes) > 0

        # Verify node structure
        first_node = nodes[0]
        assert "id" in first_node
        assert "name" in first_node
        assert "type" in first_node
        assert isinstance(first_node["id"], str)
        assert len(first_node["id"]) > 0

    def test_list_scene_nodes_via_feature(self, slicer_available: bool, slicer_url: str) -> None:
        """Call the feature-level list_scene_nodes function end-to-end.

        This exercises the full path: feature function -> get_client() ->
        SlicerClient -> Slicer WebServer, verifying the integration at
        the feature layer (not just the client layer).
        """
        import os

        from slicer_mcp.core.slicer_client import reset_client
        from slicer_mcp.features.base_tools import list_scene_nodes

        # Ensure the singleton picks up the correct URL
        old_url = os.environ.get("SLICER_URL")
        os.environ["SLICER_URL"] = slicer_url
        reset_client()

        try:
            result = list_scene_nodes()

            assert "nodes" in result
            assert "total_count" in result
            assert isinstance(result["nodes"], list)
            assert result["total_count"] == len(result["nodes"])
            assert result["total_count"] > 0
        finally:
            # Restore environment
            reset_client()
            if old_url is not None:
                os.environ["SLICER_URL"] = old_url
            elif "SLICER_URL" in os.environ:
                del os.environ["SLICER_URL"]
