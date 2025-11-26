"""MCP tool implementations for Slicer Bridge."""

import base64
import logging
from typing import Optional, Dict, Any, List

from slicer_mcp.slicer_client import SlicerClient, SlicerConnectionError

logger = logging.getLogger("slicer-mcp")


def capture_screenshot(
    view_type: str,
    scroll_position: Optional[float] = None,
    look_from_axis: Optional[str] = None
) -> Dict[str, Any]:
    """Capture a screenshot from a specific 3D Slicer viewport.

    Args:
        view_type: Viewport type to capture - "axial", "sagittal", "coronal", "3d", "full"
        scroll_position: Slice position from 0.0 to 1.0 (only for 2D views)
        look_from_axis: Camera axis for 3D view - "left", "right", "anterior", "posterior", "superior", "inferior"

    Returns:
        Dict with success status, base64-encoded PNG image, and metadata

    Raises:
        ValueError: If view_type is invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    # Map view types to Slicer view names
    view_map = {
        "axial": "Red",
        "sagittal": "Yellow",
        "coronal": "Green",
        "3d": "3d",
        "full": "full"
    }

    if view_type not in view_map:
        raise ValueError(
            f"Invalid view_type '{view_type}'. "
            f"Must be one of: {', '.join(view_map.keys())}"
        )

    client = SlicerClient()

    try:
        # Capture screenshot based on view type
        if view_type == "full":
            image_bytes = client.get_full_screenshot()
        elif view_type == "3d":
            image_bytes = client.get_3d_screenshot(look_from_axis)
        else:
            # 2D slice views (axial, sagittal, coronal)
            slicer_view = view_map[view_type]
            image_bytes = client.get_screenshot(slicer_view, scroll_position)

        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        logger.info(f"Screenshot captured: view_type={view_type}, size={len(image_base64)} chars")

        result = {
            "success": True,
            "image_base64": image_base64,
            "view_type": view_type,
            "content_type": "image/png"
        }

        if scroll_position is not None:
            result["scroll_position"] = scroll_position

        if look_from_axis is not None:
            result["look_from_axis"] = look_from_axis

        return result

    except SlicerConnectionError as e:
        logger.error(f"Screenshot capture failed: {e.message}")
        raise


def list_scene_nodes() -> Dict[str, Any]:
    """List all nodes in the current MRML scene.

    Returns:
        Dict with nodes list and total count

    Raises:
        SlicerConnectionError: If Slicer is not reachable
    """
    client = SlicerClient()

    try:
        nodes = client.get_scene_nodes()

        logger.info(f"Listed {len(nodes)} scene nodes")

        return {
            "nodes": nodes,
            "total_count": len(nodes)
        }

    except SlicerConnectionError as e:
        logger.error(f"Scene nodes listing failed: {e.message}")
        raise


def execute_python(code: str) -> Dict[str, Any]:
    """Execute arbitrary Python code in Slicer's Python environment.

    Security Warning: This executes code directly in Slicer. Use only with trusted code.

    Args:
        code: Python code to execute

    Returns:
        Dict with success status and execution result

    Raises:
        SlicerConnectionError: If Slicer is not reachable or execution fails
    """
    client = SlicerClient()

    try:
        result = client.exec_python(code)

        logger.info(f"Python code executed successfully")

        return result

    except SlicerConnectionError as e:
        logger.error(f"Python execution failed: {e.message}")
        raise


def measure_volume(node_id: str, segment_name: Optional[str] = None) -> Dict[str, Any]:
    """Calculate the volume of a segmentation node or specific segment.

    Args:
        node_id: MRML node ID of segmentation (e.g., vtkMRMLSegmentationNode1)
        segment_name: Specific segment to measure (if None, measures all segments)

    Returns:
        Dict with volume measurements in mm3 and ml

    Raises:
        SlicerConnectionError: If Slicer is not reachable or calculation fails
    """
    client = SlicerClient()

    # Build Python code to calculate volumes using SegmentStatistics
    if segment_name:
        # Measure specific segment
        python_code = f"""
import slicer
from SegmentStatistics import SegmentStatisticsLogic

segmentationNode = slicer.mrmlScene.GetNodeByID('{node_id}')
if not segmentationNode:
    raise ValueError('Node not found: {node_id}')

# Get segment
segmentation = segmentationNode.GetSegmentation()
segment = segmentation.GetSegment('{segment_name}')
if not segment:
    raise ValueError('Segment not found: {segment_name}')

# Calculate statistics
statsLogic = SegmentStatisticsLogic()
statsLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
statsLogic.computeStatistics()
stats = statsLogic.getStatistics()

# Get volume in cc (cubic cm = ml)
volume_cc = stats['{segment_name}', 'SegmentStatistics.volume_cc']
volume_mm3 = volume_cc * 1000  # Convert cc to mm3

result = {{
    'node_id': '{node_id}',
    'node_name': segmentationNode.GetName(),
    'total_volume_mm3': volume_mm3,
    'total_volume_ml': volume_cc,
    'segments': [
        {{
            'name': '{segment_name}',
            'volume_mm3': volume_mm3,
            'volume_ml': volume_cc
        }}
    ]
}}

import json
json.dumps(result)
"""
    else:
        # Measure all segments
        python_code = f"""
import slicer
from SegmentStatistics import SegmentStatisticsLogic
import json

segmentationNode = slicer.mrmlScene.GetNodeByID('{node_id}')
if not segmentationNode:
    raise ValueError('Node not found: {node_id}')

# Calculate statistics
statsLogic = SegmentStatisticsLogic()
statsLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
statsLogic.computeStatistics()
stats = statsLogic.getStatistics()

# Get all segments
segmentation = segmentationNode.GetSegmentation()
segments = []
total_volume_mm3 = 0
total_volume_ml = 0

for i in range(segmentation.GetNumberOfSegments()):
    segment = segmentation.GetNthSegment(i)
    segment_name = segment.GetName()

    # Get volume in cc (cubic cm = ml)
    volume_cc = stats[segment_name, 'SegmentStatistics.volume_cc']
    volume_mm3 = volume_cc * 1000  # Convert cc to mm3

    segments.append({{
        'name': segment_name,
        'volume_mm3': volume_mm3,
        'volume_ml': volume_cc
    }})

    total_volume_mm3 += volume_mm3
    total_volume_ml += volume_cc

result = {{
    'node_id': '{node_id}',
    'node_name': segmentationNode.GetName(),
    'total_volume_mm3': total_volume_mm3,
    'total_volume_ml': total_volume_ml,
    'segments': segments
}}

json.dumps(result)
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result
        import json
        volume_data = json.loads(exec_result["result"])

        logger.info(f"Volume measured for node {node_id}")

        return volume_data

    except SlicerConnectionError as e:
        logger.error(f"Volume measurement failed: {e.message}")
        raise
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Volume measurement result parsing failed: {e}")
        raise SlicerConnectionError(
            f"Failed to parse volume measurement result: {str(e)}",
            details={"node_id": node_id, "segment_name": segment_name}
        )


def load_sample_data(dataset_name: str) -> Dict[str, Any]:
    """Load a sample dataset into 3D Slicer.

    Args:
        dataset_name: Name of sample dataset - "MRHead", "CTChest", "CTACardio", "DTIBrain", "MRBrainTumor1", "MRBrainTumor2"

    Returns:
        Dict with success status and loaded node information

    Raises:
        ValueError: If dataset_name is invalid
        SlicerConnectionError: If Slicer is not reachable or load fails
    """
    valid_datasets = ["MRHead", "CTChest", "CTACardio", "DTIBrain", "MRBrainTumor1", "MRBrainTumor2"]

    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Invalid dataset_name '{dataset_name}'. "
            f"Must be one of: {', '.join(valid_datasets)}"
        )

    client = SlicerClient()

    try:
        result = client.load_sample_data(dataset_name)

        # Get loaded node info via Python
        python_code = f"""
import slicer
import json

# Get most recently added volume node (sample data creates new volume)
nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
if nodes:
    latest_node = nodes[-1]
    result = {{
        'loaded_node_id': latest_node.GetID(),
        'loaded_node_name': latest_node.GetName()
    }}
else:
    result = {{
        'loaded_node_id': None,
        'loaded_node_name': None
    }}

json.dumps(result)
"""
        exec_result = client.exec_python(python_code)

        import json
        node_info = json.loads(exec_result["result"])

        result.update(node_info)

        logger.info(f"Sample data '{dataset_name}' loaded successfully")

        return result

    except SlicerConnectionError as e:
        logger.error(f"Sample data load failed: {e.message}")
        raise


def set_layout(layout: str, gui_mode: str = "full") -> Dict[str, Any]:
    """Set the viewer layout and GUI mode in 3D Slicer.

    Args:
        layout: Layout name - "FourUp", "OneUp3D", "OneUpRedSlice", "Conventional", "SideBySide"
        gui_mode: GUI mode - "full" (complete GUI) or "viewers" (viewers only)

    Returns:
        Dict with success status and layout information

    Raises:
        ValueError: If layout or gui_mode is invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    valid_layouts = ["FourUp", "OneUp3D", "OneUpRedSlice", "Conventional", "SideBySide"]
    valid_gui_modes = ["full", "viewers"]

    if layout not in valid_layouts:
        raise ValueError(
            f"Invalid layout '{layout}'. "
            f"Must be one of: {', '.join(valid_layouts)}"
        )

    if gui_mode not in valid_gui_modes:
        raise ValueError(
            f"Invalid gui_mode '{gui_mode}'. "
            f"Must be one of: {', '.join(valid_gui_modes)}"
        )

    client = SlicerClient()

    try:
        result = client.set_layout(layout, gui_mode)

        logger.info(f"Layout set to {layout} with {gui_mode} GUI mode")

        return result

    except SlicerConnectionError as e:
        logger.error(f"Layout change failed: {e.message}")
        raise
