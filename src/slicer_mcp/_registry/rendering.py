"""Rendering & 3D model export tool registrations."""

from typing import Any

from slicer_mcp._registry._common import register_tool
from slicer_mcp.features import rendering as rendering_tools


def register_rendering_tools(mcp: Any) -> dict[str, Any]:
    """Register volume rendering and model export tools.

    Returns:
        Dict mapping tool name → wrapper function.
    """
    wrappers: dict[str, Any] = {}

    def _reg(fn_name: str, doc: str) -> None:
        wrappers[fn_name] = register_tool(mcp, rendering_tools, fn_name, doc)

    _reg(
        "enable_volume_rendering",
        """Enable volume rendering visualization on a volume node.

    Creates or updates volume rendering display for the given volume.
    Optionally applies a named preset (e.g., 'CT-Bone', 'MR-Default').

    Args:
        node_id: MRML node ID of the scalar volume to render
        preset: Optional volume rendering preset name
        visible: Whether the volume rendering should be visible (default: True)

    Returns:
        Dict with success status, volume_node_id, display_node_id, preset, visible
    """,
    )

    _reg(
        "set_volume_rendering_property",
        """Adjust volume rendering display properties on a volume node.

    Modifies opacity, window/level, or visibility of an existing volume
    rendering display. Volume rendering must be enabled first.

    Args:
        node_id: MRML node ID of the volume with active volume rendering
        opacity_scale: Multiplier for all opacity values (0.0 to 10.0)
        window: Window width for window/level adjustment
        level: Center level for window/level adjustment
        visible: Whether the volume rendering should be visible

    Returns:
        Dict with success status, volume_node_id, display_node_id, changes_applied
    """,
    )

    _reg(
        "export_model",
        """Export a model node to a 3D mesh file.

    Saves the model's polygon data to STL, OBJ, PLY, or VTK format.

    Args:
        node_id: MRML node ID of the model node to export
        output_directory: Directory where the file will be saved
        filename: Output filename without extension
        file_format: Export format - 'STL', 'OBJ', 'PLY', or 'VTK'

    Returns:
        Dict with success status, model_node_id, output_path, format,
            file_size_bytes, point_count, cell_count
    """,
    )

    _reg(
        "segmentation_to_models",
        """Convert segmentation segments to individual model nodes.

    Creates a vtkMRMLModelNode for each segment by extracting its closed
    surface representation. Useful for exporting segmentations to 3D files.

    Args:
        segmentation_node_id: MRML node ID of the segmentation node
        segment_ids: Optional list of specific segment IDs to convert.
            If None, all visible segments are converted.

    Returns:
        Dict with success status, segmentation_node_id, models list, model_count
    """,
    )

    _reg(
        "capture_3d_view",
        """Capture a screenshot of a 3D view to an image file.

    Renders the current 3D view and saves it as PNG, JPG, BMP, or TIFF.

    Args:
        output_path: Full output file path (supports .png, .jpg, .bmp, .tiff)
        width: Optional capture width in pixels (default: current view size)
        height: Optional capture height in pixels (default: current view size)
        view_index: Index of the 3D view to capture (default: 0)

    Returns:
        Dict with success status, output_path, file_size_bytes, view_index
    """,
    )

    return wrappers
