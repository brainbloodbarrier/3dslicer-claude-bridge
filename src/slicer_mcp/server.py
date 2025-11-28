"""MCP Slicer Bridge server entry point."""

import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Import tools and resources
from slicer_mcp import tools, resources

# Configure logging to stderr (stdout reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    stream=sys.stderr
)
logger = logging.getLogger("slicer-mcp")

# Initialize FastMCP server
mcp = FastMCP("slicer-bridge")

logger.info("Initializing MCP Slicer Bridge server")


# Register Tools
# ==============

@mcp.tool()
def capture_screenshot(
    view_type: str,
    scroll_position: Optional[float] = None,
    look_from_axis: Optional[str] = None
) -> dict:
    """Capture a screenshot from a specific 3D Slicer viewport and return as base64 PNG.

    Args:
        view_type: Viewport type - "axial" (Red slice), "sagittal" (Yellow slice), "coronal" (Green slice), "3d" (3D view), "full" (complete window)
        scroll_position: Slice position from 0.0 to 1.0 (only for axial/sagittal/coronal views)
        look_from_axis: Camera axis for 3D view - "left", "right", "anterior", "posterior", "superior", "inferior" (only for 3d view)

    Returns:
        Dict with success status, base64-encoded PNG image, view type, and metadata
    """
    return tools.capture_screenshot(view_type, scroll_position, look_from_axis)


@mcp.tool()
def list_scene_nodes() -> dict:
    """List all nodes in the current MRML scene with metadata.

    Returns:
        Dict with nodes list (id, name, type) and total count
    """
    return tools.list_scene_nodes()


@mcp.tool()
def execute_python(code: str) -> dict:
    """Execute arbitrary Python code in Slicer's Python environment.

    Security Warning: This executes code directly in Slicer's Python interpreter.
    Use only with trusted code in controlled environments.

    Args:
        code: Python code to execute (has access to slicer, vtk, qt modules)

    Returns:
        Dict with success status, result, stdout, and stderr
    """
    return tools.execute_python(code)


@mcp.tool()
def measure_volume(node_id: str, segment_name: Optional[str] = None) -> dict:
    """Calculate the volume of a segmentation node or specific segment in cubic millimeters.

    Args:
        node_id: MRML node ID of segmentation (e.g., "vtkMRMLSegmentationNode1")
        segment_name: Specific segment to measure (if None, measures all segments)

    Returns:
        Dict with node_id, node_name, total_volume_mm3, total_volume_ml, and per-segment breakdown
    """
    return tools.measure_volume(node_id, segment_name)


@mcp.tool()
def list_sample_data() -> dict:
    """List all available sample datasets from 3D Slicer's SampleData module.

    Dynamically queries Slicer to discover available sample datasets.
    Falls back to a known list if dynamic discovery fails.

    Returns:
        Dict with datasets list (name, category, description), total_count, and source (dynamic/fallback)
    """
    return tools.list_sample_data()


@mcp.tool()
def load_sample_data(dataset_name: str) -> dict:
    """Load a sample dataset into 3D Slicer for testing and demonstration.

    Use list_sample_data() first to see available datasets.

    Args:
        dataset_name: Name of sample dataset (e.g., "MRHead", "CTChest", "CTACardio")

    Returns:
        Dict with success status, dataset_name, loaded_node_id, loaded_node_name, and message
    """
    return tools.load_sample_data(dataset_name)


@mcp.tool()
def set_layout(layout: str, gui_mode: str = "full") -> dict:
    """Set the viewer layout and GUI mode in 3D Slicer.

    Args:
        layout: Layout name - "FourUp" (standard 4-panel), "OneUp3D" (single 3D view), "OneUpRedSlice" (single axial), "Conventional" (traditional radiology), "SideBySide" (comparison view)
        gui_mode: GUI display mode - "full" (complete GUI) or "viewers" (viewers only, minimal chrome)

    Returns:
        Dict with success status, layout, gui_mode, and message
    """
    return tools.set_layout(layout, gui_mode)


# Register Resources
# ==================

@mcp.resource("slicer://scene")
def get_scene() -> str:
    """Get the current MRML scene structure with all nodes and connections.

    Returns:
        JSON string with scene_id, modified_time, node_count, nodes list, and connections
    """
    return resources.get_scene_resource()


@mcp.resource("slicer://volumes")
def get_volumes() -> str:
    """Get all loaded imaging volumes with metadata including dimensions, spacing, and file paths.

    Returns:
        JSON string with volumes list (id, name, type, dimensions, spacing, origin, scalar_range, file_path) and total_count
    """
    return resources.get_volumes_resource()


@mcp.resource("slicer://status")
def get_status() -> str:
    """Get health status and connection information for 3D Slicer.

    Returns:
        JSON string with connected status, slicer_version, webserver_url, response_time_ms, scene_loaded, python_available, and last_check timestamp
    """
    return resources.get_status_resource()


# Main Entry Point
# ================

def main():
    """Run the MCP Slicer Bridge server with stdio transport."""
    logger.info("Starting MCP Slicer Bridge server")
    logger.info("Registered 7 tools: capture_screenshot, list_scene_nodes, execute_python, measure_volume, list_sample_data, load_sample_data, set_layout")
    logger.info("Registered 3 resources: slicer://scene, slicer://volumes, slicer://status")

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
