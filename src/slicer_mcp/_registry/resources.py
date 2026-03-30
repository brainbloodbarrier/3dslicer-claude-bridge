"""MCP resource registrations — slicer:// URI scheme."""

from typing import Any

from slicer_mcp.core import resources


def register_resources(mcp: Any) -> tuple[list[str], dict[str, Any]]:
    """Register MCP resources that expose Slicer state.

    Returns:
        Tuple of (resource URI list, dict mapping name → handler function).
    """

    @mcp.resource("slicer://scene")
    def get_scene() -> str:
        """Get the current MRML scene structure with all nodes and connections.

        Returns:
            JSON string with scene_id, modified_time, node_count, nodes list, and connections
        """
        return resources.get_scene_resource()

    @mcp.resource("slicer://volumes")
    def get_volumes() -> str:
        """Get all loaded imaging volumes with metadata.

        Returns:
            JSON string with volumes list (id, name, type, dimensions,
                spacing, origin, scalar_range, file_path) and total_count
        """
        return resources.get_volumes_resource()

    @mcp.resource("slicer://status")
    def get_status() -> str:
        """Get health status and connection information for 3D Slicer.

        Returns:
            JSON string with connected status, slicer_version,
                webserver_url, response_time_ms, scene_loaded,
                python_available, and last_check timestamp
        """
        return resources.get_status_resource()

    @mcp.resource("slicer://workflows")
    def get_workflows() -> str:
        """List available workflow tools with required inputs and clinical use cases.

        Returns:
            JSON string with workflows list containing name, status,
                description, required_modalities, clinical_indication,
                tools_orchestrated, and estimated_runtime for each workflow
        """
        return resources.get_workflows_resource()

    uris = [
        "slicer://scene",
        "slicer://volumes",
        "slicer://status",
        "slicer://workflows",
    ]
    handlers = {
        "get_scene": get_scene,
        "get_volumes": get_volumes,
        "get_status": get_status,
        "get_workflows": get_workflows,
    }
    return uris, handlers
