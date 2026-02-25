"""Spine-specific MCP tool implementations for Slicer Bridge.

Provides tools for spine segmentation, morphometry, and classification
using TotalSegmentator and established clinical grading systems.
"""

import json
import logging
from typing import Any

from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    SPINE_REGIONS,
    SPINE_SEGMENTATION_TIMEOUT,
    TOTALSEG_TASK_FULL,
    TOTALSEG_TASK_VERTEBRAE,
)
from slicer_mcp.tools import ValidationError, _parse_json_result, validate_mrml_node_id

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# Code Generation Helpers
# =============================================================================


def _build_spine_segmentation_code(
    safe_node_id: str,
    safe_region: str,
    include_discs: bool,
    include_spinal_cord: bool,
) -> str:
    """Build Python code for TotalSegmentator spine segmentation.

    Generates Slicer-Python code that:
    1. Validates the input volume exists
    2. Runs TotalSegmentator with the appropriate task
    3. Filters results to the requested region
    4. Returns structured JSON with vertebrae list

    Args:
        safe_node_id: JSON-escaped MRML node ID string
        safe_region: JSON-escaped region string
        include_discs: Whether to include intervertebral discs
        include_spinal_cord: Whether to include spinal cord segmentation

    Returns:
        Python code string to execute in Slicer
    """
    task = TOTALSEG_TASK_FULL if (include_discs or include_spinal_cord) else TOTALSEG_TASK_VERTEBRAE
    safe_task = json.dumps(task)
    safe_include_discs = json.dumps(include_discs)
    safe_include_spinal_cord = json.dumps(include_spinal_cord)

    return f"""
import slicer
import time
import json

input_node_id = {safe_node_id}
region = {safe_region}
task = {safe_task}
include_discs = {safe_include_discs}
include_spinal_cord = {safe_include_spinal_cord}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

# Create output segmentation node
outputSeg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
outputSeg.SetName(inputVolume.GetName() + "_spine_seg")

# Run TotalSegmentator
start_time = time.time()

try:
    import TotalSegmentator
    logic = TotalSegmentator.TotalSegmentatorLogic()
    logic.process(
        inputVolume=inputVolume,
        outputSegmentation=outputSeg,
        task=task,
        fast=False,
    )
except Exception as e:
    slicer.mrmlScene.RemoveNode(outputSeg)
    raise ValueError(f"TotalSegmentator failed: {{e}}")

elapsed = time.time() - start_time

# Region vertebrae mapping
REGION_VERTEBRAE = {{
    "cervical": ["C1","C2","C3","C4","C5","C6","C7"],
    "thoracic": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12"],
    "lumbar": ["L1","L2","L3","L4","L5"],
    "full": [
        "C1","C2","C3","C4","C5","C6","C7",
        "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
        "L1","L2","L3","L4","L5",
    ],
}}

VERTEBRA_MAP = {{
    "vertebrae_C1":"C1","vertebrae_C2":"C2","vertebrae_C3":"C3",
    "vertebrae_C4":"C4","vertebrae_C5":"C5","vertebrae_C6":"C6","vertebrae_C7":"C7",
    "vertebrae_T1":"T1","vertebrae_T2":"T2","vertebrae_T3":"T3",
    "vertebrae_T4":"T4","vertebrae_T5":"T5","vertebrae_T6":"T6",
    "vertebrae_T7":"T7","vertebrae_T8":"T8","vertebrae_T9":"T9",
    "vertebrae_T10":"T10","vertebrae_T11":"T11","vertebrae_T12":"T12",
    "vertebrae_L1":"L1","vertebrae_L2":"L2","vertebrae_L3":"L3",
    "vertebrae_L4":"L4","vertebrae_L5":"L5",
}}

DISC_MAP = {{
    "disc_L5_S1":"L5-S1","disc_L4_L5":"L4-L5","disc_L3_L4":"L3-L4",
    "disc_L2_L3":"L2-L3","disc_L1_L2":"L1-L2","disc_T12_L1":"T12-L1",
}}

expected = set(REGION_VERTEBRAE.get(region, []))
seg = outputSeg.GetSegmentation()
found_vertebrae = []
found_discs = []
found_other = []

# Iterate over all segments
for i in range(seg.GetNumberOfSegments()):
    seg_id = seg.GetNthSegmentID(i)
    seg_name = seg.GetNthSegment(i).GetName()

    if seg_name in VERTEBRA_MAP:
        label = VERTEBRA_MAP[seg_name]
        if label in expected:
            found_vertebrae.append({{"segment_id": seg_id, "label": label, "name": seg_name}})
        else:
            # Remove segments outside requested region
            seg.RemoveSegment(seg_id)
    elif seg_name in DISC_MAP:
        if include_discs:
            disc_label = DISC_MAP[seg_name]
            found_discs.append({{"segment_id": seg_id, "label": disc_label, "name": seg_name}})
        else:
            seg.RemoveSegment(seg_id)
    elif seg_name == "spinal_cord":
        if include_spinal_cord:
            found_other.append({{"segment_id": seg_id, "label": "spinal_cord", "name": seg_name}})
        else:
            seg.RemoveSegment(seg_id)
    else:
        # Remove non-spine segments
        seg.RemoveSegment(seg_id)

# Sort vertebrae by anatomical order
order = REGION_VERTEBRAE.get(region, [])
order_map = {{v: idx for idx, v in enumerate(order)}}
found_vertebrae.sort(key=lambda x: order_map.get(x["label"], 999))

result = {{
    "success": True,
    "input_node_id": input_node_id,
    "region": region,
    "output_segmentation_id": outputSeg.GetID(),
    "output_segmentation_name": outputSeg.GetName(),
    "vertebrae_count": len(found_vertebrae),
    "vertebrae": found_vertebrae,
    "discs": found_discs,
    "other_structures": found_other,
    "processing_time_seconds": round(elapsed, 2),
}}

print(json.dumps(result))
"""


# =============================================================================
# Spine Segmentation Tool
# =============================================================================


def segment_spine(
    input_node_id: str,
    region: str = "full",
    include_discs: bool = False,
    include_spinal_cord: bool = False,
) -> dict[str, Any]:
    """Segment spine structures from a CT volume using TotalSegmentator.

    LONG OPERATION: This may take 1-5 minutes depending on hardware and region.

    Identifies and labels individual vertebrae, optionally including
    intervertebral discs and spinal cord. Uses TotalSegmentator AI
    segmentation with region-based filtering.

    Expected durations:
    - GPU: ~1-2 minutes
    - CPU: ~3-5 minutes

    Args:
        input_node_id: MRML node ID of input CT volume
        region: Spine region to segment - "cervical", "thoracic",
            "lumbar", or "full" (all vertebrae)
        include_discs: If True, also segment intervertebral discs
            (requires "total" task, may increase processing time)
        include_spinal_cord: If True, also segment the spinal cord
            (requires "total" task, may increase processing time)

    Returns:
        Dict with:
        - output_segmentation_id: MRML node ID of output segmentation
        - vertebrae_count: Number of vertebrae found
        - vertebrae: List of dicts with segment_id, label, name
        - discs: List of disc segments (if include_discs=True)
        - other_structures: List of other structures (if include_spinal_cord=True)
        - processing_time_seconds: Actual processing time
        - long_operation: Metadata about this being a long operation

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or processing fails
    """
    # Validate input node ID
    input_node_id = validate_mrml_node_id(input_node_id)

    # Validate region
    if region not in SPINE_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(SPINE_REGIONS))}",
            "region",
            region,
        )

    client = get_client()

    # JSON-escape all values for safe injection into Python code
    safe_node_id = json.dumps(input_node_id)
    safe_region = json.dumps(region)

    python_code = _build_spine_segmentation_code(
        safe_node_id, safe_region, include_discs, include_spinal_cord
    )

    try:
        exec_result = client.exec_python(python_code, timeout=SPINE_SEGMENTATION_TIMEOUT)

        result_data = _parse_json_result(
            exec_result.get("result", ""), f"spine segmentation ({region})"
        )

        # Add long_operation metadata
        result_data["long_operation"] = {
            "type": "spine_segmentation",
            "region": region,
            "timeout_seconds": SPINE_SEGMENTATION_TIMEOUT,
            "typical_duration": "1-2 minutes (GPU), 3-5 minutes (CPU)",
        }

        logger.info(
            f"Spine segmentation completed: region={region}, "
            f"vertebrae={result_data.get('vertebrae_count', 0)}, "
            f"time={result_data.get('processing_time_seconds', 0)}s"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Spine segmentation failed: {e.message}")
        raise
