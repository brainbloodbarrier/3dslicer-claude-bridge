"""Bone quality analysis and clinical spine visualization tools.

Provides per-vertebra Hounsfield unit analysis with Pickhardt osteoporosis
classification, optional BoneTexture trabecular metrics, and high-quality
sagittal spine visualization with color-coded vertebra overlays.
"""

import json
import logging
from typing import Any

from slicer_mcp.core.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.features.base_tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)
from slicer_mcp.features.spine.constants import (
    PICKHARDT_HU_THRESHOLDS,
    REGION_VERTEBRAE,
    SPINE_REGIONS,
    SPINE_SEGMENTATION_TIMEOUT,
    TOTALSEGMENTATOR_VERTEBRA_MAP,
    VALID_BONE_REGIONS,
)

__all__ = [
    "analyze_bone_quality",
    "visualize_spine_segmentation",
]

logger = logging.getLogger("slicer-mcp")


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

__execResult = result
"""


# =============================================================================
# Clinical Spine Visualization
# =============================================================================


def _build_clinical_spine_visualization_code(
    safe_seg_node_id: str,
    safe_volume_node_id: str,
    safe_output_path: str,
    safe_region: str,
) -> str:
    """Build Python code to create clinical spine visualization in Slicer.

    Generates a sagittal screenshot with:
    - Distinct color per vertebra (gradient from yellow to red)
    - Segmentation in outline mode (thick contour, no fill)
    - CT bone window (W:2000 L:400)
    - Navigation to median sagittal plane of vertebral bodies
    - Markup fiducial labels at each vertebra centroid
    - High resolution capture (3x)

    Args:
        safe_seg_node_id: JSON-escaped segmentation node ID
        safe_volume_node_id: JSON-escaped volume node ID for background CT
        safe_output_path: JSON-escaped output file path
        safe_region: JSON-escaped region string

    Returns:
        Python code string for Slicer execution
    """
    return f"""
import slicer
import json
import numpy as np

seg_node_id = {safe_seg_node_id}
volume_node_id = {safe_volume_node_id}
output_path = {safe_output_path}
region = {safe_region}

segNode = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not segNode:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

volNode = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volNode:
    raise ValueError('Volume node not found: ' + volume_node_id)

segmentation = segNode.GetSegmentation()
if not segmentation:
    raise ValueError('No segmentation data in node: ' + seg_node_id)

# TotalSegmentator vertebra label map
ts_map = {json.dumps(TOTALSEGMENTATOR_VERTEBRA_MAP)}
region_vertebrae = {json.dumps(dict(REGION_VERTEBRAE))}

target_vertebrae = region_vertebrae.get(region, region_vertebrae['full'])

# --- Vertebra Color Gradient (yellow -> red) ---
n_verts = len(target_vertebrae)
vertebra_colors = {{}}
for i, vert_name in enumerate(target_vertebrae):
    t = i / max(n_verts - 1, 1)
    r = 1.0
    g = 1.0 - t
    b = 0.0
    vertebra_colors[vert_name] = (r, g, b)

# --- Apply colors and collect centroids ---
centroids = {{}}
available_segment_ids = set()
for i in range(segmentation.GetNumberOfSegments()):
    available_segment_ids.add(segmentation.GetNthSegmentID(i))

for vert_name in target_vertebrae:
    # Find segment ID (TotalSegmentator format or direct name)
    seg_id = None
    for ts_key, std_name in ts_map.items():
        if std_name == vert_name and ts_key in available_segment_ids:
            seg_id = ts_key
            break
    if seg_id is None and vert_name in available_segment_ids:
        seg_id = vert_name
    if seg_id is None:
        continue

    segment = segmentation.GetSegment(seg_id)
    if not segment:
        continue

    # Apply color
    color = vertebra_colors.get(vert_name, (1.0, 1.0, 0.0))
    segment.SetColor(color[0], color[1], color[2])

    # Extract centroid via labelmap
    import vtk
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segNode, [seg_id], labelmapNode, None
    )
    ijkToRas = vtk.vtkMatrix4x4()
    labelmapNode.GetIJKToRASMatrix(ijkToRas)
    imageData = labelmapNode.GetImageData()
    dims = imageData.GetDimensions()

    points_ras = []
    for k in range(dims[2]):
        for j in range(dims[1]):
            for ii in range(dims[0]):
                val = imageData.GetScalarComponentAsFloat(ii, j, k, 0)
                if val > 0:
                    ijk = [ii, j, k, 1]
                    ras = [0, 0, 0, 1]
                    ijkToRas.MultiplyPoint(ijk, ras)
                    points_ras.append(ras[:3])

    slicer.mrmlScene.RemoveNode(labelmapNode)

    if points_ras:
        pts = np.array(points_ras)
        centroid = pts.mean(axis=0).tolist()
        centroids[vert_name] = centroid

# --- Configure segmentation display: outline mode ---
displayNode = segNode.GetDisplayNode()
if displayNode:
    # 2D outline with thick border
    displayNode.SetVisibility2DOutline(True)
    displayNode.SetVisibility2DFill(False)
    displayNode.SetSliceIntersectionThickness(3)
    # 2D visibility ON
    displayNode.SetVisibility2D(True)

# --- CT bone window (W:2000 L:400) ---
volDisplayNode = volNode.GetDisplayNode()
if volDisplayNode:
    volDisplayNode.SetAutoWindowLevel(False)
    volDisplayNode.SetWindow(2000)
    volDisplayNode.SetLevel(400)

# --- Navigate to sagittal view at median R coordinate ---
if centroids:
    r_values = [c[0] for c in centroids.values()]
    median_r = float(np.median(r_values))
else:
    median_r = 0.0

lm = slicer.app.layoutManager()

# Switch to sagittal-only layout for clean capture
lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView)
slicer.app.processEvents()

sliceWidget = lm.sliceWidget('Yellow')
sliceLogic = sliceWidget.sliceLogic()
sliceNode = sliceLogic.GetSliceNode()
compositeNode = sliceLogic.GetSliceCompositeNode()

# Set background volume
compositeNode.SetBackgroundVolumeID(volume_node_id)

# Set sagittal orientation
sliceNode.SetOrientation('Sagittal')

# Navigate to median R coordinate
sliceNode.SetSliceOffset(median_r)

# Set FOV ~300mm for L1-S1 coverage with margin
sliceNode.SetFieldOfView(300, 300, 1)

# --- Create fiducial labels at centroids ---
fiducialNodes = []
if centroids:
    fidNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
    fidNode.SetName('SpineLabels')
    fidNode.GetDisplayNode().SetTextScale(4.0)
    fidNode.GetDisplayNode().SetGlyphScale(2.0)
    fidNode.GetDisplayNode().SetColor(1.0, 1.0, 1.0)
    fidNode.GetDisplayNode().SetSelectedColor(1.0, 1.0, 1.0)
    fidNode.GetDisplayNode().SetActiveColor(1.0, 1.0, 1.0)
    # Offset labels slightly anterior for visibility
    for vert_name, centroid in centroids.items():
        n = fidNode.AddControlPoint(centroid[0], centroid[1] + 15, centroid[2])
        fidNode.SetNthControlPointLabel(n, vert_name)
    fiducialNodes.append(fidNode)

# Force render
slicer.util.forceRenderAllViews()
slicer.app.processEvents()

# --- Capture at 3x resolution ---
import vtk as vtk_lib

renderWindow = sliceWidget.sliceView().renderWindow()
windowToImage = vtk_lib.vtkWindowToImageFilter()
windowToImage.SetInput(renderWindow)
windowToImage.SetScale(3)
windowToImage.Update()

import os
writer = vtk_lib.vtkPNGWriter()
writer.SetFileName(output_path)
writer.SetInputConnection(windowToImage.GetOutputPort())
writer.Write()

if not os.path.exists(output_path):
    raise ValueError('Screenshot was not saved: ' + output_path)

file_size = os.path.getsize(output_path)

# --- Cleanup: remove fiducials ---
cleanup_ids = []
for fid in fiducialNodes:
    cleanup_ids.append(fid.GetID())

result = {{
    'success': True,
    'output_path': output_path,
    'file_size_bytes': file_size,
    'vertebrae_colored': list(centroids.keys()),
    'centroids': centroids,
    'median_r_mm': median_r,
    'view': 'sagittal',
    'window_width': 2000,
    'window_level': 400,
    'resolution_scale': 3,
    'fiducial_node_ids': cleanup_ids,
}}

__execResult = result
"""


# =============================================================================
# Public Tool Functions — Bone Quality & Visualization
# =============================================================================


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


def visualize_spine_segmentation(
    segmentation_node_id: str,
    volume_node_id: str,
    output_path: str,
    region: str = "lumbar",
) -> dict[str, Any]:
    """Create clinical visualization of spine segmentation.

    Generates a high-quality sagittal screenshot with distinct colors
    per vertebra, outline-mode segmentation overlay, bone-window CT,
    and anatomical labels at each vertebra centroid.

    Args:
        segmentation_node_id: MRML node ID of the segmentation containing
            vertebral segments (e.g., from TotalSegmentator)
        volume_node_id: MRML node ID of the background CT volume
        output_path: File path for the output PNG screenshot
        region: Spine region to visualize - "cervical", "thoracic",
            "lumbar", or "full"

    Returns:
        Dict with success status, output_path, file_size_bytes,
        vertebrae_colored, centroids, and view parameters

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    volume_node_id = validate_mrml_node_id(volume_node_id)

    if region not in SPINE_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(SPINE_REGIONS))}",
            "region",
            region,
        )

    if not output_path or not output_path.strip():
        raise ValidationError("Output path cannot be empty", "output_path", "")

    client = get_client()

    safe_seg_id = json.dumps(segmentation_node_id)
    safe_vol_id = json.dumps(volume_node_id)
    safe_output = json.dumps(output_path)
    safe_region = json.dumps(region)

    python_code = _build_clinical_spine_visualization_code(
        safe_seg_id, safe_vol_id, safe_output, safe_region
    )

    try:
        exec_result = client.exec_python(python_code, timeout=SPINE_SEGMENTATION_TIMEOUT)

        result_data = _parse_json_result(exec_result.get("result", ""), "spine visualization")

        logger.info(
            f"Spine visualization completed: region={region}, "
            f"vertebrae={len(result_data.get('vertebrae_colored', []))}, "
            f"output={output_path}"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Spine visualization failed: {e.message}")
        raise
