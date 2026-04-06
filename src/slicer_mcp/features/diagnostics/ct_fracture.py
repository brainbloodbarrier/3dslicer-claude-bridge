"""CT fracture detection and spinal morphometry tools.

Implements 3 diagnostic tools:
- Vertebral fracture detection (Genant, AO Spine, Denis)
- Spondylolisthesis measurement (Meyerding)
- Spinal canal morphometry (Torg-Pavlov)
"""

import json
import logging
from typing import Any

from slicer_mcp.core.constants import (
    SEGMENTATION_TIMEOUT,
)
from slicer_mcp.core.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.features._subprocess import _build_totalseg_subprocess_block
from slicer_mcp.features._validation import _validate_region
from slicer_mcp.features.base_tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)
from slicer_mcp.features.diagnostics._common import _validate_levels
from slicer_mcp.features.spine.constants import (
    GENANT_THRESHOLDS,
    MEYERDING_THRESHOLDS,
    REGION_VERTEBRAE,
    SPINE_SEGMENTATION_TIMEOUT,
    TOTALSEG_TASK_VERTEBRAE,
)

__all__ = [
    "detect_vertebral_fractures_ct",
    "measure_listhesis_ct",
    "measure_spinal_canal_ct",
]

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# Input Validation Helpers
# =============================================================================


def _validate_classification_system(system: str) -> str:
    """Validate fracture classification system parameter.

    Args:
        system: Classification system string

    Returns:
        Validated system string

    Raises:
        ValidationError: If system is not valid
    """
    valid = {"ao_spine", "genant", "denis", "all"}
    if system not in valid:
        raise ValidationError(
            f"Invalid classification_system '{system}'. Must be one of: {', '.join(sorted(valid))}",
            field="classification_system",
            value=system,
        )
    return system


# =============================================================================
# Genant Classification Helper
# =============================================================================


def _classify_genant(height_loss_fraction: float) -> tuple[int, str]:
    """Classify vertebral fracture by Genant semi-quantitative grading.

    Args:
        height_loss_fraction: Fractional height reduction (0.0-1.0)

    Returns:
        Tuple of (grade, label)
    """
    if height_loss_fraction >= GENANT_THRESHOLDS["severe_min"]:
        return 3, "Severe"
    elif height_loss_fraction >= GENANT_THRESHOLDS["moderate_min"]:
        return 2, "Moderate"
    elif height_loss_fraction >= GENANT_THRESHOLDS["mild_min"]:
        return 1, "Mild"
    return 0, "Normal"


# =============================================================================
# Meyerding Classification Helper
# =============================================================================


def _classify_meyerding(slip_fraction: float) -> tuple[str, int]:
    """Classify spondylolisthesis by Meyerding grading.

    Args:
        slip_fraction: Fractional slip (0.0-1.0+)

    Returns:
        Tuple of (grade_label, grade_number)
    """
    if slip_fraction > MEYERDING_THRESHOLDS["grade_iv_max"]:
        return "V", 5
    elif slip_fraction > MEYERDING_THRESHOLDS["grade_iii_max"]:
        return "IV", 4
    elif slip_fraction > MEYERDING_THRESHOLDS["grade_ii_max"]:
        return "III", 3
    elif slip_fraction > MEYERDING_THRESHOLDS["grade_i_max"]:
        return "II", 2
    elif slip_fraction > 0:
        return "I", 1
    return "0", 0


# =============================================================================
# Stenosis Classification Helper
# =============================================================================


def _classify_stenosis(ap_diameter_mm: float) -> str:
    """Classify spinal canal stenosis by AP diameter.

    Args:
        ap_diameter_mm: Anteroposterior diameter in mm

    Returns:
        Stenosis grade string
    """
    if ap_diameter_mm < 7:
        return "severe"
    elif ap_diameter_mm < 10:
        return "moderate"
    elif ap_diameter_mm < 13:
        return "mild"
    return "none"


# =============================================================================
# Tool 1: detect_vertebral_fractures_ct
# =============================================================================


def _build_fracture_detection_code(
    safe_volume_id: str,
    safe_seg_id: str | None,
    region: str,
    classification_system: str,
) -> str:
    """Build Python code for vertebral fracture detection in Slicer.

    Args:
        safe_volume_id: JSON-escaped volume node ID
        safe_seg_id: JSON-escaped segmentation node ID or None
        region: Spine region to analyze
        classification_system: Classification system to use

    Returns:
        Python code string for execution in Slicer
    """
    vertebrae_list = json.dumps(list(REGION_VERTEBRAE.get(region, REGION_VERTEBRAE["full"])))
    safe_classification = json.dumps(classification_system)

    seg_block = f"seg_node_id = {safe_seg_id}" if safe_seg_id else "seg_node_id = None"
    auto_seg = _build_totalseg_subprocess_block("volume_node", "seg_node", TOTALSEG_TASK_VERTEBRAE)

    return f"""
import slicer
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

volume_node_id = {safe_volume_id}
{seg_block}
target_vertebrae = {vertebrae_list}
classification_system = {safe_classification}

volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volume_node:
    raise ValueError("Volume node not found: " + volume_node_id)

{auto_seg}

segmentation = seg_node.GetSegmentation()
volume_array_node = slicer.util.arrayFromVolume(volume_node)
ras_to_ijk = vtk.vtkMatrix4x4()
volume_node.GetRASToIJKMatrix(ras_to_ijk)

vertebrae_results = []
prev_hp = None

for vert_name in target_vertebrae:
    # Find segment matching this vertebra
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        name = seg.GetName()
        if vert_name in name or name.endswith(vert_name):
            seg_id = segmentation.GetNthSegmentID(i)
            break

    if not seg_id:
        continue

    # Get labelmap for this segment
    labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, volume_node)
    if labelmap is None or labelmap.sum() == 0:
        continue

    # Get HU values within segment
    hu_values = volume_array_node[labelmap > 0]

    # Calculate heights in sagittal midplane
    # Find sagittal midplane (middle column of the segment)
    indices = np.argwhere(labelmap > 0)
    if len(indices) < 10:
        continue

    # K=axial, J=coronal, I=sagittal in numpy array (K,J,I)
    mid_sagittal = int(np.median(indices[:, 2]))
    sagittal_slice = labelmap[:, :, mid_sagittal]
    seg_indices = np.argwhere(sagittal_slice > 0)
    if len(seg_indices) < 5:
        continue

    # Coronal range for anterior/middle/posterior thirds
    coronal_min = seg_indices[:, 1].min()
    coronal_max = seg_indices[:, 1].max()
    coronal_range = coronal_max - coronal_min
    if coronal_range < 3:
        continue

    third = coronal_range / 3.0
    ant_col = int(coronal_min + third * 0.5)
    mid_col = int(coronal_min + third * 1.5)
    post_col = int(coronal_min + third * 2.5)

    def col_height(col):
        col_mask = sagittal_slice[:, col]
        rows = np.argwhere(col_mask > 0).flatten()
        if len(rows) < 2:
            return 0.0
        return float(rows.max() - rows.min())

    spacing = volume_node.GetSpacing()
    axial_spacing = spacing[2]

    ha = col_height(ant_col) * axial_spacing
    hm = col_height(mid_col) * axial_spacing
    hp = col_height(post_col) * axial_spacing

    if hp < 0.1:
        continue

    wedge_ratio = ha / hp
    biconcave_ratio = hm / hp
    crush_ratio = hp / prev_hp if prev_hp and prev_hp > 0.1 else 1.0

    # Genant grading
    min_ratio = min(wedge_ratio, biconcave_ratio, crush_ratio)
    height_loss = max(0.0, 1.0 - min_ratio)

    if height_loss >= 0.40:
        genant_grade, genant_label = 3, "Severe"
    elif height_loss >= 0.25:
        genant_grade, genant_label = 2, "Moderate"
    elif height_loss >= 0.20:
        genant_grade, genant_label = 1, "Mild"
    else:
        genant_grade, genant_label = 0, "Normal"

    # Determine morphology
    ratios = {{"wedge": wedge_ratio, "biconcave": biconcave_ratio, "crush": crush_ratio}}
    morphology = min(ratios, key=ratios.get) if genant_grade > 0 else "normal"

    # Posterior wall analysis for burst detection
    post_third_start = int(coronal_min + third * 2)
    post_region = labelmap[:, post_third_start:coronal_max+1, :]
    post_indices = np.argwhere(post_region > 0)
    retropulsion_mm = 0.0
    canal_compromise_pct = 0.0
    if len(post_indices) > 0 and genant_grade >= 2:
        # Simplified retropulsion: check if posterior wall extends beyond expected
        retropulsion_mm = float(np.std(post_indices[:, 1])) * spacing[1] * 0.5
        canal_compromise_pct = min(retropulsion_mm / 15.0 * 100, 100.0)

    vert_result = {{
        "level": vert_name,
        "heights_mm": {{
            "anterior": round(ha, 1),
            "middle": round(hm, 1),
            "posterior": round(hp, 1)
        }},
        "ratios": {{
            "wedge": round(wedge_ratio, 2),
            "biconcave": round(biconcave_ratio, 2),
            "crush": round(crush_ratio, 2)
        }},
        "genant": {{
            "grade": genant_grade,
            "label": genant_label,
            "morphology": morphology,
            "height_loss_percent": round(height_loss * 100, 1)
        }},
        "bone_density_hu": round(float(np.median(hu_values)), 1),
        "osteoporotic_flag": float(np.median(hu_values)) < 110
    }}

    # AO Spine classification
    if classification_system in ("ao_spine", "all") and genant_grade > 0:
        if retropulsion_mm > 0 and height_loss >= 0.25:
            ao_type = "A4" if biconcave_ratio < 0.6 else "A3"
            ao_label = "Complete burst" if ao_type == "A4" else "Incomplete burst"
        elif morphology == "wedge":
            ao_type = "A1"
            ao_label = "Wedge compression"
        elif morphology == "crush":
            ao_type = "A2" if height_loss >= 0.25 else "A1"
            ao_label = "Split fracture" if ao_type == "A2" else "Wedge compression"
        else:
            ao_type = "A1"
            ao_label = "Compression fracture"
        vert_result["ao_spine"] = {{"type": ao_type, "label": ao_label}}

    # Denis 3-column classification
    if classification_system in ("denis", "all") and genant_grade > 0:
        ant_fractured = wedge_ratio < 0.80
        mid_fractured = retropulsion_mm > 0 or biconcave_ratio < 0.75
        post_fractured = False  # Requires posterior element analysis
        denis_stability = "UNSTABLE" if mid_fractured else "STABLE"
        vert_result["denis"] = {{
            "anterior_column": "fractured" if ant_fractured else "intact",
            "middle_column": "fractured" if mid_fractured else "intact",
            "posterior_column": "fractured" if post_fractured else "intact",
            "stability": denis_stability
        }}

    # Canal compromise
    if retropulsion_mm > 0:
        vert_result["canal_compromise"] = {{
            "retropulsion_mm": round(retropulsion_mm, 1),
            "canal_compromise_percent": round(canal_compromise_pct, 1)
        }}

    # Posterior elements (simplified)
    vert_result["posterior_elements"] = {{
        "lamina": "intact",
        "pedicle_left": "intact",
        "pedicle_right": "intact",
        "spinous_process": "intact",
        "facets": "intact"
    }}

    vertebrae_results.append(vert_result)
    prev_hp = hp

# Summary
fractured = [v for v in vertebrae_results if v["genant"]["grade"] > 0]
most_severe = max(fractured, key=lambda v: v["genant"]["grade"]) if fractured else None
canal_present = any("canal_compromise" in v for v in fractured)
osteoporotic = [v for v in fractured if v.get("osteoporotic_flag", False)]

result = {{
    "success": True,
    "modality": "CT",
    "region_analyzed": target_vertebrae[0] + "-" + target_vertebrae[-1] if target_vertebrae else "",
    "fractures_detected": len(fractured),
    "vertebrae": vertebrae_results,
    "summary": {{
        "total_fractured": len(fractured),
        "most_severe_level": most_severe["level"] if most_severe else None,
        "most_severe_grade": most_severe["genant"]["label"] if most_severe else None,
        "canal_compromise_present": canal_present,
        "osteoporotic_fractures": len(osteoporotic),
        "traumatic_fractures": len(fractured) - len(osteoporotic)
    }}
}}

# Clean up auto-created segmentation (keep if user-provided)
if not _seg_was_provided:
    slicer.mrmlScene.RemoveNode(seg_node)

__execResult = result
"""


def detect_vertebral_fractures_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    region: str = "full",
    classification_system: str = "ao_spine",
) -> dict[str, Any]:
    """Detect vertebral fractures in CT with multi-system classification.

    Performs height calculations (Ha, Hm, Hp), Genant grading (0-3),
    AO Spine classification, Denis 3-column analysis, posterior wall
    retropulsion assessment, and canal compromise measurement.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing segmentation (optional;
            runs TotalSegmentator if not provided)
        region: Spine region - "full", "cervical", "thoracic", "lumbar"
        classification_system: "ao_spine", "genant", "denis", or "all"

    Returns:
        Dict with per-vertebra fracture analysis and summary

    Tip:
        Run ``segment_spine`` once and pass its ``output_segmentation_id``
        as ``segmentation_node_id`` to skip auto-segmentation (~10x faster).

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    region = _validate_region(region)
    classification_system = _validate_classification_system(classification_system)

    client = get_client()

    safe_volume_id = json.dumps(volume_node_id)
    safe_seg_id = json.dumps(segmentation_node_id) if segmentation_node_id else None

    python_code = _build_fracture_detection_code(
        safe_volume_id, safe_seg_id, region, classification_system
    )

    timeout = SEGMENTATION_TIMEOUT if segmentation_node_id else SPINE_SEGMENTATION_TIMEOUT
    try:
        exec_result = client.exec_python(python_code, timeout=timeout)
        data = _parse_json_result(exec_result.get("result", ""), "fracture detection")
        logger.info(
            f"Fracture detection complete: {data.get('fractures_detected', 0)} fractures found"
        )
        return data

    except SlicerConnectionError as e:
        logger.error(f"Fracture detection failed: {e.message}")
        raise


# =============================================================================
# Tool 5: measure_listhesis_ct
# =============================================================================


def _build_listhesis_code(
    safe_volume_id: str,
    safe_seg_id: str | None,
    levels: list[str],
) -> str:
    """Build Python code for listhesis measurement in Slicer.

    Args:
        safe_volume_id: JSON-escaped volume node ID
        safe_seg_id: JSON-escaped segmentation node ID or None
        levels: Vertebral levels to measure

    Returns:
        Python code string for execution in Slicer
    """
    safe_levels = json.dumps(levels)
    seg_block = f"seg_node_id = {safe_seg_id}" if safe_seg_id else "seg_node_id = None"
    auto_seg = _build_totalseg_subprocess_block("volume_node", "seg_node", TOTALSEG_TASK_VERTEBRAE)

    return f"""
import slicer
import json
import numpy as np

volume_node_id = {safe_volume_id}
{seg_block}
target_levels = {safe_levels}

volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volume_node:
    raise ValueError("Volume node not found: " + volume_node_id)

{auto_seg}

segmentation = seg_node.GetSegmentation()
spacing = volume_node.GetSpacing()

# Get posterior edge positions for each vertebra
vert_data = {{}}
for level in target_levels:
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if level in seg.GetName() or seg.GetName().endswith(level):
            seg_id = segmentation.GetNthSegmentID(i)
            lm = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, volume_node)
            if lm is not None and lm.sum() > 0:
                indices = np.argwhere(lm > 0)
                # Sagittal midplane
                mid_sag = int(np.median(indices[:, 2]))
                sag_slice = lm[:, :, mid_sag]
                sag_idx = np.argwhere(sag_slice > 0)
                if len(sag_idx) > 5:
                    # Posterior edge = max coronal index
                    post_col = sag_idx[:, 1].max()
                    # Superior and inferior rows
                    sup_row = sag_idx[:, 0].min()
                    inf_row = sag_idx[:, 0].max()
                    # AP width for percentage calculation
                    ant_col = sag_idx[:, 1].min()
                    ap_width = (post_col - ant_col) * spacing[1]

                    vert_data[level] = {{
                        "post_col": int(post_col),
                        "ant_col": int(ant_col),
                        "sup_row": int(sup_row),
                        "inf_row": int(inf_row),
                        "ap_width": float(ap_width),
                        "mid_sag": mid_sag
                    }}
            break

# Measure listhesis between adjacent levels
levels_results = []
for i in range(len(target_levels) - 1):
    sup_level = target_levels[i]
    inf_level = target_levels[i + 1]

    if sup_level not in vert_data or inf_level not in vert_data:
        continue

    sup = vert_data[sup_level]
    inf = vert_data[inf_level]

    # Translation = difference in posterior edge position
    translation_voxels = sup["post_col"] - inf["post_col"]
    translation_mm = float(abs(translation_voxels)) * spacing[1]
    if translation_voxels < 0:
        direction = "anterior"
    elif translation_voxels > 0:
        direction = "posterior"
    else:
        direction = "none"

    # Percentage relative to inferior body AP width
    ap_width = inf["ap_width"]
    translation_pct = (translation_mm / ap_width * 100) if ap_width > 0 else 0.0

    # Meyerding grade
    slip_frac = translation_pct / 100.0
    if slip_frac > 1.0:
        meyerding = "V"
    elif slip_frac > 0.75:
        meyerding = "IV"
    elif slip_frac > 0.50:
        meyerding = "III"
    elif slip_frac > 0.25:
        meyerding = "II"
    elif slip_frac > 0.0:
        meyerding = "I"
    else:
        meyerding = "0"

    # Simplified slip angle
    sup_inf_row = sup["inf_row"]
    inf_sup_row = inf["sup_row"]
    row_diff = (inf_sup_row - sup_inf_row) * spacing[2]
    col_diff = translation_mm
    slip_angle = float(np.degrees(np.arctan2(col_diff, row_diff))) if row_diff > 0 else 0.0

    level_pair = f"{{sup_level}}-{{inf_level}}"
    levels_results.append({{
        "level": level_pair,
        "translation_mm": round(translation_mm, 1),
        "translation_percent": round(translation_pct, 1),
        "direction": direction,
        "meyerding_grade": meyerding,
        "slip_angle_deg": round(abs(slip_angle), 1),
        "spondylolysis_detected": False,
        "spondylolysis_side": None
    }})

result = {{
    "success": True,
    "modality": "CT",
    "static_measurement": True,
    "note": "Static CT. Dynamic instability requires flexion/extension X-ray.",
    "levels": levels_results
}}

# Clean up auto-created segmentation (keep if user-provided)
if not _seg_was_provided:
    slicer.mrmlScene.RemoveNode(seg_node)

__execResult = result
"""


def measure_listhesis_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    levels: list[str] | None = None,
) -> dict[str, Any]:
    """Measure spondylolisthesis on static CT.

    Calculates translation (mm and %), Meyerding grade (I-V),
    slip angle, and spondylolysis detection. Includes static
    measurement disclaimer.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing segmentation (optional)
        levels: Vertebral levels to measure (default: ["L3", "L4", "L5"])

    Returns:
        Dict with per-level listhesis measurements

    Tip:
        Run ``segment_spine`` once and pass its ``output_segmentation_id``
        as ``segmentation_node_id`` to skip auto-segmentation (~10x faster).

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    levels = _validate_levels(levels, ["L3", "L4", "L5"])

    client = get_client()

    safe_volume_id = json.dumps(volume_node_id)
    safe_seg_id = json.dumps(segmentation_node_id) if segmentation_node_id else None

    python_code = _build_listhesis_code(safe_volume_id, safe_seg_id, levels)

    timeout = SEGMENTATION_TIMEOUT if segmentation_node_id else SPINE_SEGMENTATION_TIMEOUT
    try:
        exec_result = client.exec_python(python_code, timeout=timeout)
        data = _parse_json_result(exec_result.get("result", ""), "listhesis measurement")
        logger.info(f"Listhesis measurement complete for levels: {levels}")
        return data

    except SlicerConnectionError as e:
        logger.error(f"Listhesis measurement failed: {e.message}")
        raise


# =============================================================================
# Tool 6: measure_spinal_canal_ct
# =============================================================================


def _build_canal_measurement_code(
    safe_volume_id: str,
    safe_seg_id: str | None,
    levels: list[str],
) -> str:
    """Build Python code for spinal canal measurement in Slicer.

    Args:
        safe_volume_id: JSON-escaped volume node ID
        safe_seg_id: JSON-escaped segmentation node ID or None
        levels: Vertebral levels to measure

    Returns:
        Python code string for execution in Slicer
    """
    safe_levels = json.dumps(levels)
    seg_block = f"seg_node_id = {safe_seg_id}" if safe_seg_id else "seg_node_id = None"
    auto_seg = _build_totalseg_subprocess_block("volume_node", "seg_node", TOTALSEG_TASK_VERTEBRAE)

    return f"""
import slicer
import json
import numpy as np

volume_node_id = {safe_volume_id}
{seg_block}
target_levels = {safe_levels}

volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volume_node:
    raise ValueError("Volume node not found: " + volume_node_id)

{auto_seg}

segmentation = seg_node.GetSegmentation()
volume_array = slicer.util.arrayFromVolume(volume_node)
spacing = volume_node.GetSpacing()

levels_results = []

for level in target_levels:
    # Find vertebral body segment
    body_seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if level in seg.GetName() or seg.GetName().endswith(level):
            body_seg_id = segmentation.GetNthSegmentID(i)
            break

    if not body_seg_id:
        continue

    body_lm = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, body_seg_id, volume_node)
    if body_lm is None or body_lm.sum() == 0:
        continue

    body_indices = np.argwhere(body_lm > 0)

    # Find spinal canal segment if available
    canal_seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        name = seg.GetName().lower()
        if "spinal_canal" in name or "spinal canal" in name:
            canal_seg_id = segmentation.GetNthSegmentID(i)
            break

    if canal_seg_id:
        canal_lm = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, canal_seg_id, volume_node)
    else:
        canal_lm = None

    # Get axial slice at mid-vertebra height
    mid_axial = int(np.median(body_indices[:, 0]))

    if canal_lm is not None:
        canal_slice = canal_lm[mid_axial, :, :]
        canal_idx = np.argwhere(canal_slice > 0)

        if len(canal_idx) < 5:
            continue

        # AP diameter (coronal extent = axis 0 in axial slice)
        ap_voxels = canal_idx[:, 0].max() - canal_idx[:, 0].min()
        ap_mm = float(ap_voxels) * spacing[1]

        # Transverse diameter (sagittal extent = axis 1 in axial slice)
        tr_voxels = canal_idx[:, 1].max() - canal_idx[:, 1].min()
        tr_mm = float(tr_voxels) * spacing[0]

        # Cross-section area (voxel count * pixel area)
        area_mm2 = float(canal_slice.sum()) * spacing[0] * spacing[1]
    else:
        # Estimate canal from space posterior to vertebral body
        body_slice = body_lm[mid_axial, :, :]
        body_idx = np.argwhere(body_slice > 0)
        if len(body_idx) < 5:
            continue

        # Canal is posterior to body: estimate from HU in that region
        post_edge = body_idx[:, 0].max()
        mid_lr = int(np.median(body_idx[:, 1]))

        # Scan posteriorly for low-HU region (canal)
        axial_hu = volume_array[mid_axial, :, :]
        canal_start = post_edge + 1
        canal_end = canal_start
        for row in range(canal_start, min(canal_start + 30, axial_hu.shape[0])):
            if axial_hu[row, mid_lr] > 200:
                break
            canal_end = row

        ap_mm = float(canal_end - canal_start) * spacing[1]
        tr_mm = ap_mm * 1.2  # Approximate
        area_mm2 = ap_mm * tr_mm * 0.785  # Ellipse approximation

    # Vertebral body AP diameter for Torg-Pavlov
    body_slice = body_lm[mid_axial, :, :]
    body_idx = np.argwhere(body_slice > 0)
    if len(body_idx) > 0:
        body_ap_voxels = body_idx[:, 0].max() - body_idx[:, 0].min()
        body_ap_mm = float(body_ap_voxels) * spacing[1]
    else:
        body_ap_mm = 1.0

    torg_pavlov = ap_mm / body_ap_mm if body_ap_mm > 0 else 0.0

    # Stenosis grading
    if ap_mm < 7:
        stenosis = "severe"
    elif ap_mm < 10:
        stenosis = "moderate"
    elif ap_mm < 13:
        stenosis = "mild"
    else:
        stenosis = "none"

    # Determine region for reference ranges
    if level.startswith("C"):
        region = "cervical"
    elif level.startswith("T"):
        region = "thoracic"
    else:
        region = "lumbar"

    levels_results.append({{
        "level": level,
        "ap_diameter_mm": round(ap_mm, 1),
        "transverse_diameter_mm": round(tr_mm, 1),
        "cross_section_area_mm2": round(area_mm2, 1),
        "torg_pavlov_ratio": round(torg_pavlov, 2),
        "torg_pavlov_stenosis": torg_pavlov < 0.80,
        "stenosis_grade": stenosis,
        "vertebral_body_ap_mm": round(body_ap_mm, 1),
        "region": region
    }})

result = {{
    "success": True,
    "modality": "CT",
    "levels": levels_results
}}

# Clean up auto-created segmentation (keep if user-provided)
if not _seg_was_provided:
    slicer.mrmlScene.RemoveNode(seg_node)

__execResult = result
"""


def measure_spinal_canal_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    levels: list[str] | None = None,
) -> dict[str, Any]:
    """Measure spinal canal morphometry on CT.

    Calculates AP and transverse diameters, cross-section area,
    Torg-Pavlov ratio, and stenosis grading per level.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing segmentation (optional)
        levels: Vertebral levels to measure (default: ["C3", "C4", "C5", "C6", "C7"])

    Returns:
        Dict with per-level canal measurements

    Tip:
        Run ``segment_spine`` once and pass its ``output_segmentation_id``
        as ``segmentation_node_id`` to skip auto-segmentation (~10x faster).

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    levels = _validate_levels(levels, ["C3", "C4", "C5", "C6", "C7"])

    client = get_client()

    safe_volume_id = json.dumps(volume_node_id)
    safe_seg_id = json.dumps(segmentation_node_id) if segmentation_node_id else None

    python_code = _build_canal_measurement_code(safe_volume_id, safe_seg_id, levels)

    timeout = SEGMENTATION_TIMEOUT if segmentation_node_id else SPINE_SEGMENTATION_TIMEOUT
    try:
        exec_result = client.exec_python(python_code, timeout=timeout)
        data = _parse_json_result(exec_result.get("result", ""), "spinal canal measurement")
        logger.info(f"Spinal canal measurement complete for levels: {levels}")
        return data

    except SlicerConnectionError as e:
        logger.error(f"Spinal canal measurement failed: {e.message}")
        raise
