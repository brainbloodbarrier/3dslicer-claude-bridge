"""MCP resource implementations for Slicer Bridge."""

import json
import logging
from datetime import datetime, timezone

from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.tools import _parse_json_result

logger = logging.getLogger("slicer-mcp")


def _iso_timestamp() -> str:
    """Generate ISO 8601 timestamp in UTC with 'Z' suffix.

    Returns:
        Timestamp string like '2024-01-15T14:30:00Z'
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_scene_resource() -> str:
    """Get the current MRML scene structure as JSON.

    Returns:
        JSON string with scene structure and nodes

    Raises:
        SlicerConnectionError: If Slicer is not reachable
    """
    client = get_client()

    try:
        nodes = client.get_scene_nodes()

        # Build scene resource
        scene_data = {
            "scene_id": "vtkMRMLScene",
            "modified_time": _iso_timestamp(),
            "node_count": len(nodes),
            "nodes": nodes,
            "connections": [],  # Future: add node connections if needed
        }

        logger.info(f"Scene resource retrieved: {len(nodes)} nodes")

        return json.dumps(scene_data, indent=2)

    except SlicerConnectionError as e:
        logger.error(f"Scene resource retrieval failed: {e.message}")
        raise


def get_volumes_resource() -> str:
    """Get all loaded imaging volumes with metadata.

    Returns:
        JSON string with volumes list and metadata

    Raises:
        SlicerConnectionError: If Slicer is not reachable
    """
    client = get_client()

    # Python code to get volume information
    python_code = """
import slicer
import json

volumes = []

# Get all scalar volume nodes
scalarVolumeNodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
for node in scalarVolumeNodes:
    imageData = node.GetImageData()
    if imageData:
        dimensions = imageData.GetDimensions()
        spacing = node.GetSpacing()
        origin = node.GetOrigin()
        scalarRange = imageData.GetScalarRange()

        volume_info = {
            'id': node.GetID(),
            'name': node.GetName(),
            'type': 'scalar',
            'dimensions': list(dimensions),
            'spacing': list(spacing),
            'origin': list(origin),
            'scalar_range': list(scalarRange),
            'file_path': node.GetStorageNode().GetFileName() if node.GetStorageNode() else None
        }
        volumes.append(volume_info)

# Get all vector volume nodes
vectorVolumeNodes = slicer.util.getNodesByClass('vtkMRMLVectorVolumeNode')
for node in vectorVolumeNodes:
    imageData = node.GetImageData()
    if imageData:
        dimensions = imageData.GetDimensions()
        spacing = node.GetSpacing()
        origin = node.GetOrigin()

        volume_info = {
            'id': node.GetID(),
            'name': node.GetName(),
            'type': 'vector',
            'dimensions': list(dimensions),
            'spacing': list(spacing),
            'origin': list(origin),
            'file_path': node.GetStorageNode().GetFileName() if node.GetStorageNode() else None
        }
        volumes.append(volume_info)

result = {
    'volumes': volumes,
    'total_count': len(volumes)
}

__execResult = result
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result using shared helper
        volumes_data = _parse_json_result(exec_result.get("result", ""), "volumes resource")

        logger.info(f"Volumes resource retrieved: {volumes_data['total_count']} volumes")

        return json.dumps(volumes_data, indent=2)

    except SlicerConnectionError as e:
        logger.error(f"Volumes resource retrieval failed: {e.message}")
        raise


def get_status_resource() -> str:
    """Get health status and connection information for 3D Slicer.

    Returns:
        JSON string with connection status and system info

    Raises:
        SlicerConnectionError: If Slicer is not reachable
    """
    client = get_client()

    try:
        # Perform health check
        health = client.health_check()

        # Get Slicer version and scene status
        python_code = """
import slicer
import json

result = {
    'slicer_version': slicer.app.applicationVersion,
    'scene_loaded': slicer.mrmlScene.GetNumberOfNodes() > 0,
    'python_available': True
}

__execResult = result
"""

        exec_result = client.exec_python(python_code)

        # Parse JSON result using shared helper
        slicer_info = _parse_json_result(exec_result.get("result", ""), "status resource")

        # Combine health check and Slicer info
        status_data = {
            "connected": health["connected"],
            "slicer_version": slicer_info["slicer_version"],
            "webserver_url": health["webserver_url"],
            "response_time_ms": health["response_time_ms"],
            "scene_loaded": slicer_info["scene_loaded"],
            "python_available": slicer_info["python_available"],
            "last_check": _iso_timestamp(),
        }

        logger.info(f"Status resource retrieved: connected={status_data['connected']}")

        return json.dumps(status_data, indent=2)

    except SlicerConnectionError as e:
        # Return disconnected status if connection fails
        logger.warning(f"Status resource: Slicer not connected: {e.message}")

        status_data = {
            "connected": False,
            "slicer_version": None,
            "webserver_url": client.base_url,
            "response_time_ms": None,
            "scene_loaded": False,
            "python_available": False,
            "last_check": _iso_timestamp(),
            "error": e.message,
        }

        return json.dumps(status_data, indent=2)
