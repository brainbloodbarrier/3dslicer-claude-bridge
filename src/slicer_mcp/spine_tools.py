"""Spine-specific MCP tool implementations: vertebral artery segmentation and bone quality."""

import json
import logging
from typing import Any

from slicer_mcp.constants import SEGMENTATION_TIMEOUT
from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    PICKHARDT_HU_THRESHOLDS,
    SPINE_SEGMENTATION_TIMEOUT,
)
from slicer_mcp.tools import ValidationError, _parse_json_result, validate_mrml_node_id

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# Valid Parameters
# =============================================================================

VALID_ARTERY_SIDES = frozenset(["left", "right", "both"])
"""Valid side parameters for vertebral artery segmentation."""

VALID_BONE_REGIONS = frozenset(["cervical", "thoracic", "lumbar", "full"])
"""Valid spine region parameters for bone quality analysis."""


# =============================================================================
# Vertebral Artery Segmentation (SlicerVMTK)
# =============================================================================


def _build_vesselness_code(safe_node_id: str, safe_side: str) -> str:
    """Build Python code for vesselness filter + level set segmentation.

    Uses SlicerVMTK's vesselness filtering and level set evolution
    to segment vertebral arteries from CTA volumes.

    Args:
        safe_node_id: JSON-escaped MRML node ID of input CTA volume
        safe_side: JSON-escaped side parameter ("left", "right", or "both")

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json
import time

input_node_id = {safe_node_id}
side = {safe_side}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

start_time = time.time()

# --- Step 1: Vesselness filtering via SlicerVMTK ---
try:
    import VesselnessFiltering
except ImportError:
    raise ValueError(
        "SlicerVMTK extension not installed. "
        "Install via Extension Manager: SlicerVMTK"
    )

vesselnessLogic = VesselnessFiltering.VesselnessFilteringLogic()

# Create output vesselness volume
vesselnessVolume = slicer.mrmlScene.AddNewNodeByClass(
    'vtkMRMLScalarVolumeNode'
)
vesselnessVolume.SetName(inputVolume.GetName() + "_vesselness")

# Configure vesselness filter for vertebral arteries
# Typical VA diameter: 2-5 mm
vesselnessLogic.computeVesselnessVolume(
    inputVolume, vesselnessVolume,
    minimumDiameterMm=1.5,
    maximumDiameterMm=6.0,
    alpha=0.3,
    beta=0.3,
    contrastMeasure=200
)

# --- Step 2: Level set segmentation ---
import LevelSetSegmentation
lsLogic = LevelSetSegmentation.LevelSetSegmentationLogic()

# Create segmentation node for results
segNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
segNode.SetName(inputVolume.GetName() + "_VA_seg")

# Use vesselness volume to initialise level set evolution
lsLogic.performEvolution(
    vesselnessVolume, segNode,
    inflationWeight=1.0,
    curvatureWeight=0.7,
    attractionWeight=1.0,
    iterationCount=20
)

# --- Step 3: Centerline extraction ---
import ExtractCenterline
clLogic = ExtractCenterline.ExtractCenterlineLogic()

# Create centerline model
centerlineModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
centerlineModel.SetName(inputVolume.GetName() + "_VA_centerline")

# Extract centerline from segmentation
clLogic.extractCenterline(segNode, centerlineModel)

# --- Step 4: Compute diameters along centerline ---
diameters = []
if centerlineModel.GetPolyData():
    pd = centerlineModel.GetPolyData()
    radiusArray = pd.GetPointData().GetArray("Radius")
    if radiusArray:
        n_points = radiusArray.GetNumberOfTuples()
        step = max(1, n_points // 20)  # Sample ~20 points
        for i in range(0, n_points, step):
            r = radiusArray.GetValue(i)
            diameters.append(round(r * 2.0, 2))  # diameter = 2 * radius

elapsed = time.time() - start_time

# Build result
result = {{
    "success": True,
    "input_node_id": input_node_id,
    "side": side,
    "model_node_id": segNode.GetID(),
    "model_node_name": segNode.GetName(),
    "centerline_node_id": centerlineModel.GetID(),
    "centerline_node_name": centerlineModel.GetName(),
    "vesselness_node_id": vesselnessVolume.GetID(),
    "diameters_mm": diameters,
    "mean_diameter_mm": round(sum(diameters) / len(diameters), 2) if diameters else 0.0,
    "processing_time_seconds": round(elapsed, 2)
}}

print(json.dumps(result))
"""


def _build_vesselness_with_seeds_code(safe_node_id: str, safe_side: str, safe_seeds: str) -> str:
    """Build Python code for seeded vertebral artery segmentation.

    Uses user-supplied seed points to guide the level set initialization,
    improving accuracy over the automatic approach.

    Args:
        safe_node_id: JSON-escaped MRML node ID of input CTA volume
        safe_side: JSON-escaped side parameter
        safe_seeds: JSON-escaped list of [x, y, z] seed point coordinates

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json
import time
import vtk

input_node_id = {safe_node_id}
side = {safe_side}
seed_points = {safe_seeds}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

start_time = time.time()

# --- Step 1: Vesselness filtering ---
try:
    import VesselnessFiltering
except ImportError:
    raise ValueError(
        "SlicerVMTK extension not installed. "
        "Install via Extension Manager: SlicerVMTK"
    )

vesselnessLogic = VesselnessFiltering.VesselnessFilteringLogic()

vesselnessVolume = slicer.mrmlScene.AddNewNodeByClass(
    'vtkMRMLScalarVolumeNode'
)
vesselnessVolume.SetName(inputVolume.GetName() + "_vesselness")

vesselnessLogic.computeVesselnessVolume(
    inputVolume, vesselnessVolume,
    minimumDiameterMm=1.5,
    maximumDiameterMm=6.0,
    alpha=0.3,
    beta=0.3,
    contrastMeasure=200
)

# --- Step 2: Create seed fiducials ---
seedNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
seedNode.SetName("VA_seeds")
for pt in seed_points:
    seedNode.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))

# --- Step 3: Level set segmentation with seeds ---
import LevelSetSegmentation
lsLogic = LevelSetSegmentation.LevelSetSegmentationLogic()

segNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
segNode.SetName(inputVolume.GetName() + "_VA_seg")

lsLogic.performEvolution(
    vesselnessVolume, segNode,
    seedNode=seedNode,
    inflationWeight=1.0,
    curvatureWeight=0.7,
    attractionWeight=1.0,
    iterationCount=20
)

# --- Step 4: Centerline extraction ---
import ExtractCenterline
clLogic = ExtractCenterline.ExtractCenterlineLogic()

centerlineModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
centerlineModel.SetName(inputVolume.GetName() + "_VA_centerline")

clLogic.extractCenterline(segNode, centerlineModel)

# --- Step 5: Compute diameters along centerline ---
diameters = []
if centerlineModel.GetPolyData():
    pd = centerlineModel.GetPolyData()
    radiusArray = pd.GetPointData().GetArray("Radius")
    if radiusArray:
        n_points = radiusArray.GetNumberOfTuples()
        step = max(1, n_points // 20)
        for i in range(0, n_points, step):
            r = radiusArray.GetValue(i)
            diameters.append(round(r * 2.0, 2))

# Clean up seed node
slicer.mrmlScene.RemoveNode(seedNode)

elapsed = time.time() - start_time

result = {{
    "success": True,
    "input_node_id": input_node_id,
    "side": side,
    "model_node_id": segNode.GetID(),
    "model_node_name": segNode.GetName(),
    "centerline_node_id": centerlineModel.GetID(),
    "centerline_node_name": centerlineModel.GetName(),
    "vesselness_node_id": vesselnessVolume.GetID(),
    "diameters_mm": diameters,
    "mean_diameter_mm": round(sum(diameters) / len(diameters), 2) if diameters else 0.0,
    "seed_count": len(seed_points),
    "processing_time_seconds": round(elapsed, 2)
}}

print(json.dumps(result))
"""


def _validate_seed_points(seed_points: list[list[float]]) -> list[list[float]]:
    """Validate seed point coordinates for vertebral artery segmentation.

    Each seed point must be a list of exactly 3 floats (RAS coordinates).

    Args:
        seed_points: List of [x, y, z] coordinate lists

    Returns:
        Validated seed points

    Raises:
        ValidationError: If seed points are malformed
    """
    if not seed_points:
        raise ValidationError("Seed points list cannot be empty", "seed_points", "[]")

    if len(seed_points) > 50:
        raise ValidationError(
            f"Too many seed points ({len(seed_points)}). Maximum is 50.",
            "seed_points",
            f"[{len(seed_points)} points]",
        )

    for i, pt in enumerate(seed_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 3:
            raise ValidationError(
                f"Seed point {i} must be [x, y, z] coordinates, got: {pt}",
                "seed_points",
                str(pt),
            )
        for j, coord in enumerate(pt):
            if not isinstance(coord, (int, float)):
                raise ValidationError(
                    f"Seed point {i} coordinate {j} must be numeric, got: {type(coord).__name__}",
                    "seed_points",
                    str(coord),
                )

    return seed_points


def segment_vertebral_artery(
    input_node_id: str,
    side: str = "both",
    seed_points: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Segment vertebral arteries from a CTA volume using SlicerVMTK.

    LONG OPERATION: This tool may take 1-5 minutes depending on volume size.

    Pipeline: vesselness filter -> level set segmentation -> centerline extraction.
    Requires the SlicerVMTK extension to be installed in 3D Slicer.

    Args:
        input_node_id: MRML node ID of input CTA volume
        side: Which artery to segment - "left", "right", or "both"
        seed_points: Optional list of [x, y, z] RAS coordinates to guide segmentation.
            Each point should be placed inside the artery lumen.

    Returns:
        Dict with model_node_id, centerline_node_id, diameters along trajectory,
        mean_diameter_mm, processing_time_seconds, and long_operation metadata

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or processing fails
    """
    # Validate input node ID
    input_node_id = validate_mrml_node_id(input_node_id)

    # Validate side parameter
    if side not in VALID_ARTERY_SIDES:
        raise ValidationError(
            f"Invalid side '{side}'. Must be one of: {', '.join(sorted(VALID_ARTERY_SIDES))}",
            "side",
            side,
        )

    # Validate seed points if provided
    if seed_points is not None:
        seed_points = _validate_seed_points(seed_points)

    client = get_client()

    # Safe strings for Python code (defense-in-depth)
    safe_node_id = json.dumps(input_node_id)
    safe_side = json.dumps(side)

    if seed_points is not None:
        safe_seeds = json.dumps(seed_points)
        python_code = _build_vesselness_with_seeds_code(safe_node_id, safe_side, safe_seeds)
    else:
        python_code = _build_vesselness_code(safe_node_id, safe_side)

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)

        # Parse JSON result
        result_data = _parse_json_result(
            exec_result.get("result", ""), "vertebral artery segmentation"
        )

        # Add long_operation metadata
        result_data["long_operation"] = {
            "type": "vertebral_artery_segmentation",
            "timeout_seconds": SEGMENTATION_TIMEOUT,
            "typical_duration": "1-5 minutes",
        }

        logger.info(
            f"Vertebral artery segmentation completed: side={side}, "
            f"mean_diameter={result_data.get('mean_diameter_mm', 0)}mm, "
            f"time={result_data.get('processing_time_seconds', 0)}s"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Vertebral artery segmentation failed: {e.message}")
        raise


# =============================================================================
# Bone Quality Analysis (BoneTexture Extension)
# =============================================================================


def _build_bone_quality_code(safe_node_id: str, safe_seg_id: str, safe_region: str) -> str:
    """Build Python code for per-vertebra bone quality analysis.

    Uses the BoneTexture extension for trabecular metrics and Hounsfield
    unit analysis for osteoporosis classification (Pickhardt criteria).

    Args:
        safe_node_id: JSON-escaped MRML node ID of input CT volume
        safe_seg_id: JSON-escaped MRML node ID of vertebra segmentation
        safe_region: JSON-escaped spine region string

    Returns:
        Python code string for execution in Slicer
    """
    pickhardt_normal = PICKHARDT_HU_THRESHOLDS["normal_min"]
    pickhardt_osteopenia = PICKHARDT_HU_THRESHOLDS["osteopenia_min"]

    return f"""
import slicer
import json
import time
import numpy as np

input_node_id = {safe_node_id}
seg_node_id = {safe_seg_id}
region = {safe_region}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

segNode = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not segNode:
    raise ValueError(f"Segmentation node not found: {{seg_node_id}}")

start_time = time.time()

# Check for BoneTexture extension
try:
    import BoneTexture
    has_bone_texture = True
except ImportError:
    has_bone_texture = False

segmentation = segNode.GetSegmentation()
n_segments = segmentation.GetNumberOfSegments()

# Pickhardt HU thresholds for osteoporosis classification
NORMAL_MIN_HU = {pickhardt_normal}
OSTEOPENIA_MIN_HU = {pickhardt_osteopenia}

vertebrae_results = []

for i in range(n_segments):
    segment = segmentation.GetNthSegment(i)
    seg_name = segment.GetName()
    seg_id = segmentation.GetNthSegmentID(i)

    # Create labelmap for this segment
    labelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    labelNode.SetName(seg_name + "_label")

    seg_ids = vtk.vtkStringArray()
    seg_ids.InsertNextValue(seg_id)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segNode, seg_ids, labelNode, inputVolume
    )

    # Extract voxel values within the segment ROI
    import vtk.util.numpy_support as vtk_np

    labelArray = vtk_np.vtk_to_numpy(
        labelNode.GetImageData().GetPointData().GetScalars()
    )
    volumeArray = vtk_np.vtk_to_numpy(
        inputVolume.GetImageData().GetPointData().GetScalars()
    )

    mask = labelArray > 0
    roi_values = volumeArray[mask].astype(float)

    vertebra_data = {{
        "name": seg_name,
        "segment_id": seg_id,
        "voxel_count": int(mask.sum()),
    }}

    if len(roi_values) > 0:
        mean_hu = float(np.mean(roi_values))
        std_hu = float(np.std(roi_values))

        vertebra_data["mean_hu"] = round(mean_hu, 1)
        vertebra_data["std_hu"] = round(std_hu, 1)
        vertebra_data["min_hu"] = round(float(np.min(roi_values)), 1)
        vertebra_data["max_hu"] = round(float(np.max(roi_values)), 1)

        # Pickhardt classification based on mean trabecular HU
        if mean_hu >= NORMAL_MIN_HU:
            classification = "normal"
        elif mean_hu >= OSTEOPENIA_MIN_HU:
            classification = "osteopenia"
        else:
            classification = "osteoporosis"
        vertebra_data["classification"] = classification

        # BoneTexture metrics if extension is available
        if has_bone_texture:
            try:
                btLogic = BoneTexture.BoneTextureLogic()
                glcm = btLogic.computeGLCM(inputVolume, labelNode)

                if glcm:
                    vertebra_data["bv_tv"] = round(glcm.get("BV/TV", 0.0), 4)
                    vertebra_data["tb_th_mm"] = round(glcm.get("Tb.Th", 0.0), 3)
                    vertebra_data["tb_sp_mm"] = round(glcm.get("Tb.Sp", 0.0), 3)
            except Exception as bt_err:
                vertebra_data["bone_texture_error"] = str(bt_err)
    else:
        vertebra_data["mean_hu"] = None
        vertebra_data["classification"] = "insufficient_data"

    vertebrae_results.append(vertebra_data)

    # Clean up temporary label node
    slicer.mrmlScene.RemoveNode(labelNode)

elapsed = time.time() - start_time

# Summary statistics
hu_values = [v["mean_hu"] for v in vertebrae_results if v["mean_hu"] is not None]
classifications = [v["classification"] for v in vertebrae_results]

result = {{
    "success": True,
    "input_node_id": input_node_id,
    "segmentation_node_id": seg_node_id,
    "region": region,
    "vertebrae": vertebrae_results,
    "vertebrae_count": len(vertebrae_results),
    "summary": {{
        "mean_hu_overall": round(sum(hu_values) / len(hu_values), 1) if hu_values else None,
        "normal_count": classifications.count("normal"),
        "osteopenia_count": classifications.count("osteopenia"),
        "osteoporosis_count": classifications.count("osteoporosis"),
        "has_bone_texture_metrics": has_bone_texture,
    }},
    "processing_time_seconds": round(elapsed, 2)
}}

print(json.dumps(result))
"""


def analyze_bone_quality(
    input_node_id: str,
    segmentation_node_id: str,
    region: str = "lumbar",
) -> dict[str, Any]:
    """Analyze bone quality per vertebra using CT Hounsfield units and BoneTexture metrics.

    LONG OPERATION: This tool may take 30 seconds to 3 minutes depending on vertebra count.

    Extracts per-vertebra ROIs from a segmentation, computes mean HU for
    osteoporosis classification (Pickhardt criteria), and optionally computes
    trabecular bone metrics (BV/TV, Tb.Th, Tb.Sp) via the BoneTexture extension.

    Pickhardt classification (L1 trabecular HU on non-contrast CT):
    - Normal: >= 135 HU
    - Osteopenia: 90-134 HU
    - Osteoporosis: < 90 HU

    Args:
        input_node_id: MRML node ID of input CT volume
        segmentation_node_id: MRML node ID of vertebra segmentation
            (e.g., from TotalSegmentator)
        region: Spine region to analyze - "cervical", "thoracic", "lumbar", or "full"

    Returns:
        Dict with per-vertebra metrics (mean_hu, classification, optional BV/TV,
        Tb.Th, Tb.Sp), summary counts, and processing_time_seconds

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or analysis fails
    """
    # Validate inputs
    input_node_id = validate_mrml_node_id(input_node_id)
    segmentation_node_id = validate_mrml_node_id(segmentation_node_id)

    if region not in VALID_BONE_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(VALID_BONE_REGIONS))}",
            "region",
            region,
        )

    client = get_client()

    # Safe strings for Python code (defense-in-depth)
    safe_node_id = json.dumps(input_node_id)
    safe_seg_id = json.dumps(segmentation_node_id)
    safe_region = json.dumps(region)

    python_code = _build_bone_quality_code(safe_node_id, safe_seg_id, safe_region)

    try:
        exec_result = client.exec_python(python_code, timeout=SPINE_SEGMENTATION_TIMEOUT)

        # Parse JSON result
        result_data = _parse_json_result(exec_result.get("result", ""), "bone quality analysis")

        # Add long_operation metadata
        result_data["long_operation"] = {
            "type": "bone_quality_analysis",
            "timeout_seconds": SPINE_SEGMENTATION_TIMEOUT,
            "typical_duration": "30s - 3 minutes",
        }

        logger.info(
            f"Bone quality analysis completed: region={region}, "
            f"vertebrae={result_data.get('vertebrae_count', 0)}, "
            f"time={result_data.get('processing_time_seconds', 0)}s"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Bone quality analysis failed: {e.message}")
        raise
