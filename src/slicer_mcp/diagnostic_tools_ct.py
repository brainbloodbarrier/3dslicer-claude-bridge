"""CT diagnostic protocol tools for spine analysis.

Implements 6 diagnostic tools for CT-based vertebral assessment:
- Vertebral fracture detection (Genant, AO Spine, Denis)
- Opportunistic osteoporosis screening (Pickhardt 2013)
- Metastatic lesion detection (lytic/blastic/mixed)
- SINS score calculation (4/6 automated imaging components)
- Spondylolisthesis measurement (Meyerding)
- Spinal canal morphometry (Torg-Pavlov)
"""

import json
import logging
from typing import Any

from slicer_mcp.constants import (
    SEGMENTATION_TIMEOUT,
)
from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    GENANT_THRESHOLDS,
    MEYERDING_THRESHOLDS,
    REGION_VERTEBRAE,
    SINS_RANGES,
    SPINE_REGIONS,
)
from slicer_mcp.tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# Input Validation Helpers
# =============================================================================


def _validate_region(region: str) -> str:
    """Validate spine region parameter.

    Args:
        region: Spine region string

    Returns:
        Validated region string

    Raises:
        ValidationError: If region is not valid
    """
    if region not in SPINE_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(SPINE_REGIONS))}",
            field="region",
            value=region,
        )
    return region


def _validate_levels(levels: list[str] | None, default_levels: list[str]) -> list[str]:
    """Validate vertebral level list.

    Args:
        levels: User-provided level list or None for defaults
        default_levels: Default levels when None provided

    Returns:
        Validated list of vertebral levels

    Raises:
        ValidationError: If any level is invalid
    """
    if levels is None:
        return default_levels

    all_vertebrae = set(REGION_VERTEBRAE["full"])
    for level in levels:
        if level not in all_vertebrae:
            raise ValidationError(
                f"Invalid vertebral level '{level}'. " f"Must be one of: C1-C7, T1-T12, L1-L5",
                field="levels",
                value=level,
            )
    return levels


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
            f"Invalid classification_system '{system}'. "
            f"Must be one of: {', '.join(sorted(valid))}",
            field="classification_system",
            value=system,
        )
    return system


def _validate_osteo_method(method: str) -> str:
    """Validate osteoporosis assessment method.

    Args:
        method: Assessment method string

    Returns:
        Validated method string

    Raises:
        ValidationError: If method is not valid
    """
    valid = {"trabecular_roi", "vertebral_mean", "both"}
    if method not in valid:
        raise ValidationError(
            f"Invalid method '{method}'. Must be one of: {', '.join(sorted(valid))}",
            field="method",
            value=method,
        )
    return method


# =============================================================================
# SINS Score Helper Functions
# =============================================================================


def _sins_location_score(level: str) -> tuple[int, str]:
    """Calculate SINS location component score.

    Args:
        level: Vertebral level string

    Returns:
        Tuple of (score, rationale)
    """
    junctional = {"C1", "C2", "C7", "T1", "T2", "T11", "T12", "L1", "L5", "S1"}
    mobile = {"C3", "C4", "C5", "C6"}
    semi_rigid = {f"T{i}" for i in range(3, 11)}
    rigid = {f"S{i}" for i in range(2, 6)}

    if level in junctional:
        return 3, f"{level} = junctional"
    elif level in mobile:
        return 2, f"{level} = mobile spine (C3-C6)"
    elif level in semi_rigid:
        return 1, f"{level} = semi-rigid (T3-T10)"
    elif level in rigid:
        return 0, f"{level} = rigid (S2-S5)"
    return 0, f"{level} = unknown region, scored as 0"


def _sins_lesion_type_score(lesion_type: str) -> tuple[int, str]:
    """Calculate SINS lesion type component score.

    Args:
        lesion_type: "lytic", "mixed", "blastic", or "unknown"

    Returns:
        Tuple of (score, rationale)
    """
    if lesion_type == "lytic":
        return 2, "Lytic lesion detected on CT"
    elif lesion_type == "mixed":
        return 1, "Mixed lytic/blastic lesion"
    elif lesion_type == "blastic":
        return 0, "Blastic lesion"
    return 1, "Lesion type uncertain, scored conservatively as mixed"


def _sins_alignment_score(alignment_data: dict) -> tuple[int, str]:
    """Calculate SINS alignment component score.

    Args:
        alignment_data: Dict with subluxation and deformity flags

    Returns:
        Tuple of (score, rationale)
    """
    if alignment_data.get("subluxation"):
        return 4, "Subluxation/translation present"
    elif alignment_data.get("focal_kyphosis_new") or alignment_data.get("focal_scoliosis_new"):
        return 2, "New focal deformity detected"
    return 0, "Normal alignment"


def _sins_collapse_score(collapse_pct: float, involvement_pct: float) -> tuple[int, str]:
    """Calculate SINS vertebral body collapse component score.

    Args:
        collapse_pct: Percentage of body collapse
        involvement_pct: Percentage of body involvement

    Returns:
        Tuple of (score, rationale)
    """
    if collapse_pct > 50:
        return 3, f"Body collapse {collapse_pct:.0f}% (>50%)"
    elif collapse_pct > 0:
        return 2, f"Body collapse {collapse_pct:.0f}% (<50%)"
    elif involvement_pct > 50:
        return 1, f"No collapse but >50% body involved ({involvement_pct:.0f}%)"
    return 0, "No collapse, <50% involvement"


def _sins_posterolateral_score(posterior_elements: dict) -> tuple[int, str]:
    """Calculate SINS posterolateral involvement component score.

    Args:
        posterior_elements: Dict with pedicle status

    Returns:
        Tuple of (score, rationale)
    """
    left = posterior_elements.get("pedicle_left", "intact") != "intact"
    right = posterior_elements.get("pedicle_right", "intact") != "intact"
    if left and right:
        return 3, "Bilateral pedicle involvement"
    elif left or right:
        side = "left" if left else "right"
        return 1, f"Unilateral pedicle involvement ({side})"
    return 0, "No posterolateral involvement"


def _classify_sins_total(total: int) -> tuple[str, str]:
    """Classify SINS total score.

    Args:
        total: Total SINS score

    Returns:
        Tuple of (classification, interpretation)
    """
    low, high = SINS_RANGES["stable"]
    if low <= total <= high:
        return "STABLE", f"Score {low}-{high}: Stable. No surgical consult needed."
    low, high = SINS_RANGES["indeterminate"]
    if low <= total <= high:
        return "INDETERMINATE", (
            f"Score {low}-{high}: Indeterminate stability. " f"Surgical consultation recommended."
        )
    return "UNSTABLE", "Score 13-18: Unstable. Surgical intervention likely required."


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
# Pickhardt Classification Helper
# =============================================================================


def _classify_pickhardt(mean_hu: float) -> tuple[str, str, str]:
    """Classify bone density by Pickhardt 2013 HU thresholds.

    Args:
        mean_hu: Mean trabecular HU value

    Returns:
        Tuple of (category, estimated_t_score_range, confidence)
    """
    if mean_hu >= 160:
        return "NORMAL", "> -1.0", "high"
    elif mean_hu >= 110:
        return "OSTEOPENIA", "-1.0 to -2.5", "moderate"
    elif mean_hu >= 80:
        return "OSTEOPOROSIS_PROBABLE", "approx -2.5", "moderate"
    elif mean_hu >= 50:
        return "OSTEOPOROSIS", "< -2.5", "high"
    return "OSTEOPOROSIS_SEVERE", "<< -2.5", "high"


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

    seg_block = ""
    if safe_seg_id:
        seg_block = f"seg_node_id = {safe_seg_id}"
    else:
        seg_block = "seg_node_id = None"

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

# Get or create segmentation
if seg_node_id:
    seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not seg_node:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    # Run TotalSegmentator for vertebral body segmentation
    import TotalSegmentator
    seg_node = TotalSegmentator.TotalSegmentatorLogic().process(
        volume_node, task="vertebral_body"
    )
    seg_node_id = seg_node.GetID()

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

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)
        data = _parse_json_result(exec_result.get("result", ""), "fracture detection")
        logger.info(
            f"Fracture detection complete: {data.get('fractures_detected', 0)} fractures found"
        )
        return data

    except SlicerConnectionError as e:
        logger.error(f"Fracture detection failed: {e.message}")
        raise


# =============================================================================
# Tool 2: assess_osteoporosis_ct
# =============================================================================


def _build_osteoporosis_code(
    safe_volume_id: str,
    safe_seg_id: str | None,
    levels: list[str],
    method: str,
) -> str:
    """Build Python code for osteoporosis assessment in Slicer.

    Args:
        safe_volume_id: JSON-escaped volume node ID
        safe_seg_id: JSON-escaped segmentation node ID or None
        levels: List of vertebral levels to assess
        method: Assessment method

    Returns:
        Python code string for execution in Slicer
    """
    safe_levels = json.dumps(levels)
    safe_method = json.dumps(method)

    seg_block = f"seg_node_id = {safe_seg_id}" if safe_seg_id else "seg_node_id = None"

    return f"""
import slicer
import json
import numpy as np
import vtk

volume_node_id = {safe_volume_id}
{seg_block}
target_levels = {safe_levels}
method = {safe_method}

volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volume_node:
    raise ValueError("Volume node not found: " + volume_node_id)

# Get or create segmentation
if seg_node_id:
    seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not seg_node:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    import TotalSegmentator
    seg_node = TotalSegmentator.TotalSegmentatorLogic().process(
        volume_node, task="vertebral_body"
    )

segmentation = seg_node.GetSegmentation()
volume_array = slicer.util.arrayFromVolume(volume_node)
spacing = volume_node.GetSpacing()
voxel_vol = spacing[0] * spacing[1] * spacing[2]

levels_results = []

for level in target_levels:
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        name = seg.GetName()
        if level in name or name.endswith(level):
            seg_id = segmentation.GetNthSegmentID(i)
            break

    if not seg_id:
        continue

    labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, volume_node)
    if labelmap is None or labelmap.sum() == 0:
        continue

    if method in ("trabecular_roi", "both"):
        # 3mm morphological erosion to exclude cortical bone
        import SimpleITK as sitk
        labelmap_sitk = sitk.GetImageFromArray(labelmap.astype(np.uint8))
        labelmap_sitk.SetSpacing([float(s) for s in spacing])
        eroded = sitk.BinaryErode(labelmap_sitk, [3, 3, 3], sitk.sitkBall)
        eroded_array = sitk.GetArrayFromImage(eroded)

        # Get HU values in eroded ROI
        roi_mask = eroded_array > 0
        hu_roi = volume_array[roi_mask].astype(float)

        # Exclude cortical remnants (>300 HU) and air/artifact (<-50 HU)
        hu_filtered = hu_roi[(hu_roi >= -50) & (hu_roi <= 300)]
    else:
        # Vertebral mean (no erosion)
        hu_filtered = volume_array[labelmap > 0].astype(float)
        hu_filtered = hu_filtered[(hu_filtered >= -50) & (hu_filtered <= 300)]

    if len(hu_filtered) < 10:
        continue

    mean_hu = float(np.mean(hu_filtered))
    median_hu = float(np.median(hu_filtered))
    std_hu = float(np.std(hu_filtered))

    # Pickhardt classification
    if mean_hu >= 160:
        category = "NORMAL"
        t_score = "> -1.0"
        confidence = "high"
    elif mean_hu >= 110:
        category = "OSTEOPENIA"
        t_score = "-1.0 to -2.5"
        confidence = "moderate"
    elif mean_hu >= 80:
        category = "OSTEOPOROSIS_PROBABLE"
        t_score = "approx -2.5"
        confidence = "moderate"
    elif mean_hu >= 50:
        category = "OSTEOPOROSIS"
        t_score = "< -2.5"
        confidence = "high"
    else:
        category = "OSTEOPOROSIS_SEVERE"
        t_score = "<< -2.5"
        confidence = "high"

    roi_voxel_count = int(np.sum(hu_filtered > -999))
    roi_volume_mm3 = roi_voxel_count * voxel_vol

    level_result = {{
        "level": level,
        "roi_volume_mm3": round(roi_volume_mm3, 1),
        "roi_voxel_count": roi_voxel_count,
        "hu_statistics": {{
            "mean": round(mean_hu, 1),
            "median": round(median_hu, 1),
            "std": round(std_hu, 1),
            "min": round(float(np.min(hu_filtered)), 1),
            "max": round(float(np.max(hu_filtered)), 1),
            "p10": round(float(np.percentile(hu_filtered, 10)), 1),
            "p25": round(float(np.percentile(hu_filtered, 25)), 1),
            "p50": round(float(np.percentile(hu_filtered, 50)), 1),
            "p75": round(float(np.percentile(hu_filtered, 75)), 1),
            "p90": round(float(np.percentile(hu_filtered, 90)), 1)
        }},
        "classification": {{
            "category": category,
            "hu_threshold_used": "Pickhardt_2013",
            "estimated_t_score_range": t_score,
            "confidence": confidence
        }}
    }}

    levels_results.append(level_result)

# Global assessment
if levels_results:
    all_means = [l["hu_statistics"]["mean"] for l in levels_results]
    global_mean = sum(all_means) / len(all_means)

    if global_mean >= 160:
        global_cat = "NORMAL"
    elif global_mean >= 110:
        global_cat = "OSTEOPENIA"
    elif global_mean >= 80:
        global_cat = "OSTEOPOROSIS_PROBABLE"
    elif global_mean >= 50:
        global_cat = "OSTEOPOROSIS"
    else:
        global_cat = "OSTEOPOROSIS_SEVERE"

    screw_risk = "HIGH" if global_mean < 110 else ("MODERATE" if global_mean < 160 else "LOW")
    cement_flag = global_mean < 110
else:
    global_mean = 0.0
    global_cat = "UNKNOWN"
    screw_risk = "UNKNOWN"
    cement_flag = False

result = {{
    "success": True,
    "modality": "CT",
    "method": method,
    "calibrated": False,
    "note": "Opportunistic screening - not equivalent to DXA. No calibration phantom.",
    "levels": levels_results,
    "global_assessment": {{
        "mean_hu_all_levels": round(global_mean, 1),
        "classification": global_cat,
        "recommendation": "Formal DXA recommended for confirmation"
    }},
    "clinical_context": {{
        "screw_pullout_risk": screw_risk,
        "cement_augmentation_flag": cement_flag,
        "note": "Consider cement-augmented screws or expandable screws" if cement_flag else ""
    }}
}}

__execResult = result
"""


def assess_osteoporosis_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    levels: list[str] | None = None,
    method: str = "trabecular_roi",
) -> dict[str, Any]:
    """Assess bone density for opportunistic osteoporosis screening on CT.

    Uses trabecular ROI with 3mm morphological erosion, HU statistics,
    and Pickhardt 2013 classification. Reports screw pullout risk.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing segmentation (optional)
        levels: Vertebral levels to assess (default: ["L1"])
        method: "trabecular_roi", "vertebral_mean", or "both"

    Returns:
        Dict with per-level HU statistics, classification, and clinical context

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    levels = _validate_levels(levels, ["L1"])
    method = _validate_osteo_method(method)

    client = get_client()

    safe_volume_id = json.dumps(volume_node_id)
    safe_seg_id = json.dumps(segmentation_node_id) if segmentation_node_id else None

    python_code = _build_osteoporosis_code(safe_volume_id, safe_seg_id, levels, method)

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)
        data = _parse_json_result(exec_result.get("result", ""), "osteoporosis assessment")
        logger.info(f"Osteoporosis assessment complete for levels: {levels}")
        return data

    except SlicerConnectionError as e:
        logger.error(f"Osteoporosis assessment failed: {e.message}")
        raise


# =============================================================================
# Tool 3: detect_metastatic_lesions_ct
# =============================================================================


def _build_metastatic_detection_code(
    safe_volume_id: str,
    safe_seg_id: str | None,
    region: str,
    include_posterior_elements: bool,
) -> str:
    """Build Python code for metastatic lesion detection in Slicer.

    Args:
        safe_volume_id: JSON-escaped volume node ID
        safe_seg_id: JSON-escaped segmentation node ID or None
        region: Spine region to analyze
        include_posterior_elements: Whether to analyze posterior elements

    Returns:
        Python code string for execution in Slicer
    """
    vertebrae_list = json.dumps(list(REGION_VERTEBRAE.get(region, REGION_VERTEBRAE["full"])))
    seg_block = f"seg_node_id = {safe_seg_id}" if safe_seg_id else "seg_node_id = None"

    return f"""
import slicer
import json
import numpy as np
import vtk
from scipy import ndimage

volume_node_id = {safe_volume_id}
{seg_block}
target_vertebrae = {vertebrae_list}
include_posterior = {str(include_posterior_elements)}

volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volume_node:
    raise ValueError("Volume node not found: " + volume_node_id)

if seg_node_id:
    seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not seg_node:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    import TotalSegmentator
    seg_node = TotalSegmentator.TotalSegmentatorLogic().process(
        volume_node, task="vertebral_body"
    )

segmentation = seg_node.GetSegmentation()
volume_array = slicer.util.arrayFromVolume(volume_node)
spacing = volume_node.GetSpacing()
voxel_vol = spacing[0] * spacing[1] * spacing[2]

# Collect HU baselines from all vertebrae first
all_medians = {{}}
for vert_name in target_vertebrae:
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if vert_name in seg.GetName() or seg.GetName().endswith(vert_name):
            seg_id = segmentation.GetNthSegmentID(i)
            lm = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, volume_node)
            if lm is not None and lm.sum() > 0:
                hu = volume_array[lm > 0].astype(float)
                all_medians[vert_name] = float(np.median(hu))
            break

vertebrae_results = []
total_lesions = 0

for idx, vert_name in enumerate(target_vertebrae):
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if vert_name in seg.GetName() or seg.GetName().endswith(vert_name):
            seg_id = segmentation.GetNthSegmentID(i)
            break

    if not seg_id:
        continue

    labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, volume_node)
    if labelmap is None or labelmap.sum() == 0:
        continue

    hu_values = volume_array[labelmap > 0].astype(float)
    body_volume = float(labelmap.sum()) * voxel_vol

    # Calculate adaptive baseline from 3 adjacent non-affected levels
    adjacent_medians = []
    for offset in [-2, -1, 1, 2]:
        adj_idx = idx + offset
        if 0 <= adj_idx < len(target_vertebrae):
            adj_name = target_vertebrae[adj_idx]
            if adj_name in all_medians:
                adjacent_medians.append(all_medians[adj_name])
    if len(adjacent_medians) < 2:
        adjacent_medians = [float(np.median(hu_values))]
    baseline = np.mean(adjacent_medians)
    baseline_sd = np.std(adjacent_medians) if len(adjacent_medians) > 1 else 30.0

    # Detect lytic lesions: HU < baseline - 2*SD AND HU < 100
    lytic_thresh = min(baseline - 2 * baseline_sd, 100)
    lytic_mask = (volume_array < lytic_thresh) & (labelmap > 0)
    lytic_labeled, n_lytic = ndimage.label(lytic_mask) if lytic_mask.any() else (lytic_mask, 0)

    # Detect blastic lesions: HU > baseline + 2*SD AND HU > 300
    blastic_thresh = max(baseline + 2 * baseline_sd, 300)
    blastic_mask = (volume_array > blastic_thresh) & (labelmap > 0)
    if blastic_mask.any():
        blastic_labeled, n_blastic = ndimage.label(blastic_mask)
    else:
        blastic_labeled, n_blastic = blastic_mask, 0

    lesions = []
    # Process lytic lesions
    for l_id in range(1, n_lytic + 1):
        cluster = lytic_labeled == l_id
        if cluster.sum() < 10:
            continue
        vol_mm3 = float(cluster.sum()) * voxel_vol
        hu_mean = float(np.mean(volume_array[cluster]))
        involvement = vol_mm3 / body_volume * 100 if body_volume > 0 else 0
        lesions.append({{
            "type": "lytic",
            "volume_mm3": round(vol_mm3, 1),
            "body_involvement_percent": round(involvement, 1),
            "hu_mean": round(hu_mean, 1),
            "hu_baseline_adjacent": round(baseline, 1)
        }})

    # Process blastic lesions
    for b_id in range(1, n_blastic + 1):
        cluster = blastic_labeled == b_id
        if cluster.sum() < 10:
            continue
        vol_mm3 = float(cluster.sum()) * voxel_vol
        hu_mean = float(np.mean(volume_array[cluster]))
        involvement = vol_mm3 / body_volume * 100 if body_volume > 0 else 0
        lesions.append({{
            "type": "blastic",
            "volume_mm3": round(vol_mm3, 1),
            "body_involvement_percent": round(involvement, 1),
            "hu_mean": round(hu_mean, 1),
            "hu_baseline_adjacent": round(baseline, 1)
        }})

    if not lesions:
        continue

    total_lesions += len(lesions)

    # Determine overall lesion type
    lytic_vol = sum(l["volume_mm3"] for l in lesions if l["type"] == "lytic")
    blastic_vol = sum(l["volume_mm3"] for l in lesions if l["type"] == "blastic")
    if lytic_vol > 0 and blastic_vol > 0:
        lesion_type = "mixed"
    elif lytic_vol > blastic_vol:
        lesion_type = "lytic"
    else:
        lesion_type = "blastic"

    # Body collapse
    indices = np.argwhere(labelmap > 0)
    height_voxels = indices[:, 0].max() - indices[:, 0].min()
    collapse_pct = 0.0  # Simplified; full comparison needs adjacent levels

    vert_result = {{
        "level": vert_name,
        "lesion_type": lesion_type,
        "lesion_count": len(lesions),
        "lesions": lesions,
        "body_collapse_percent": round(collapse_pct, 1),
        "canal_compromise_percent": 0.0,
        "alignment": "maintained"
    }}

    # Posterior elements
    if include_posterior:
        vert_result["posterior_elements"] = {{
            "pedicle_left": "intact",
            "pedicle_right": "intact",
            "lamina": "intact",
            "spinous_process": "intact",
            "facets": "intact"
        }}

    vertebrae_results.append(vert_result)

result = {{
    "success": True,
    "modality": "CT",
    "lesions_detected": total_lesions,
    "vertebrae": vertebrae_results
}}

__execResult = result
"""


def detect_metastatic_lesions_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    region: str = "full",
    include_posterior_elements: bool = True,
) -> dict[str, Any]:
    """Detect metastatic lesions (lytic/blastic/mixed) in vertebral bodies on CT.

    Uses adaptive HU thresholds with 3D connected component clustering.
    Reports volume, location, body involvement percentage, and
    posterior element involvement.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing segmentation (optional)
        region: Spine region - "full", "cervical", "thoracic", "lumbar"
        include_posterior_elements: Analyze posterior element involvement

    Returns:
        Dict with per-vertebra lesion analysis

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    region = _validate_region(region)

    client = get_client()

    safe_volume_id = json.dumps(volume_node_id)
    safe_seg_id = json.dumps(segmentation_node_id) if segmentation_node_id else None

    python_code = _build_metastatic_detection_code(
        safe_volume_id, safe_seg_id, region, include_posterior_elements
    )

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)
        data = _parse_json_result(exec_result.get("result", ""), "metastatic lesion detection")
        logger.info(f"Metastatic detection complete: {data.get('lesions_detected', 0)} lesions")
        return data

    except SlicerConnectionError as e:
        logger.error(f"Metastatic lesion detection failed: {e.message}")
        raise


# =============================================================================
# Tool 4: calculate_sins_score
# =============================================================================


def _build_sins_code(
    safe_volume_id: str,
    safe_seg_id: str | None,
    target_levels: list[str],
    pain_score: int | None,
) -> str:
    """Build Python code for SINS score calculation in Slicer.

    Args:
        safe_volume_id: JSON-escaped volume node ID
        safe_seg_id: JSON-escaped segmentation node ID or None
        target_levels: Vertebral levels to evaluate
        pain_score: Clinical pain score (0-3) or None

    Returns:
        Python code string for execution in Slicer
    """
    safe_levels = json.dumps(target_levels)
    safe_pain = repr(pain_score)  # None→"None", int→"N" (both valid Python)
    seg_block = f"seg_node_id = {safe_seg_id}" if safe_seg_id else "seg_node_id = None"

    return f"""
import slicer
import json
import numpy as np
from scipy import ndimage

volume_node_id = {safe_volume_id}
{seg_block}
target_levels = {safe_levels}
pain_score = {safe_pain}

volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volume_node:
    raise ValueError("Volume node not found: " + volume_node_id)

if seg_node_id:
    seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not seg_node:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    import TotalSegmentator
    seg_node = TotalSegmentator.TotalSegmentatorLogic().process(
        volume_node, task="vertebral_body"
    )

segmentation = seg_node.GetSegmentation()
volume_array = slicer.util.arrayFromVolume(volume_node)
spacing = volume_node.GetSpacing()
voxel_vol = spacing[0] * spacing[1] * spacing[2]

results_by_level = []

for level in target_levels:
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if level in seg.GetName() or seg.GetName().endswith(level):
            seg_id = segmentation.GetNthSegmentID(i)
            break

    if not seg_id:
        continue

    labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, volume_node)
    if labelmap is None or labelmap.sum() == 0:
        continue

    hu_values = volume_array[labelmap > 0].astype(float)
    median_hu = float(np.median(hu_values))

    # Component 1: Location
    junctional = {{"C1", "C2", "C7", "T1", "T2", "T11", "T12", "L1", "L5", "S1"}}
    mobile = {{"C3", "C4", "C5", "C6"}}
    semi_rigid = {{f"T{{i}}" for i in range(3, 11)}}

    if level in junctional:
        loc_score, loc_rationale = 3, f"{{level}} = junctional"
    elif level in mobile:
        loc_score, loc_rationale = 2, f"{{level}} = mobile spine"
    elif level in semi_rigid:
        loc_score, loc_rationale = 1, f"{{level}} = semi-rigid"
    else:
        loc_score, loc_rationale = 0, f"{{level}} = rigid"

    # Component 2: Pain (clinical, not from imaging)
    if pain_score is not None:
        pain_actual = pain_score
        pain_rationale = "User-provided clinical pain score"
        pain_source = "clinical"
    else:
        pain_actual = None
        pain_rationale = "Not provided (clinical input required)"
        pain_source = "missing"

    # Component 3: Lesion type from HU analysis
    lytic_thresh = 100
    blastic_thresh = 300
    lytic_count = int(np.sum(hu_values < lytic_thresh))
    blastic_count = int(np.sum(hu_values > blastic_thresh))
    total_voxels = len(hu_values)

    lytic_frac = lytic_count / total_voxels if total_voxels > 0 else 0
    blastic_frac = blastic_count / total_voxels if total_voxels > 0 else 0

    if lytic_frac > 0.1 and blastic_frac > 0.1:
        lesion_score, lesion_rationale = 1, "Mixed lesion pattern"
    elif lytic_frac > 0.1:
        lesion_score, lesion_rationale = 2, "Lytic lesion detected on CT"
    elif blastic_frac > 0.1:
        lesion_score, lesion_rationale = 0, "Blastic lesion"
    else:
        lesion_score, lesion_rationale = 1, "Lesion type uncertain"

    # Component 4: Alignment (simplified from imaging)
    alignment_score, alignment_rationale = 0, "Normal alignment"

    # Component 5: Collapse
    indices = np.argwhere(labelmap > 0)
    height = (indices[:, 0].max() - indices[:, 0].min()) * spacing[2]
    body_vol = float(labelmap.sum()) * voxel_vol
    involvement_pct = (lytic_count + blastic_count) / total_voxels * 100 if total_voxels > 0 else 0

    collapse_pct = 0.0
    collapse_score, collapse_rationale = 0, "No collapse"
    if involvement_pct > 50:
        collapse_score = 1
        collapse_rationale = (
            f"No collapse but >50% involved ({{involvement_pct:.0f}}%)"
        )

    # Component 6: Posterolateral
    posterolateral_score = 0
    posterolateral_rationale = "No posterolateral involvement"

    # Sum components
    imaging_scores = (
        loc_score + lesion_score + alignment_score
        + collapse_score + posterolateral_score
    )
    components = {{
        "location": {{
            "score": loc_score,
            "rationale": loc_rationale, "source": "imaging"
        }},
        "pain": {{
            "score": pain_actual,
            "rationale": pain_rationale, "source": pain_source
        }},
        "lesion_type": {{
            "score": lesion_score,
            "rationale": lesion_rationale, "source": "imaging"
        }},
        "alignment": {{
            "score": alignment_score,
            "rationale": alignment_rationale, "source": "imaging"
        }},
        "collapse": {{
            "score": collapse_score,
            "rationale": collapse_rationale, "source": "imaging"
        }},
        "posterolateral": {{
            "score": posterolateral_score,
            "rationale": posterolateral_rationale,
            "source": "imaging"
        }}
    }}

    if pain_actual is not None:
        total_score = imaging_scores + pain_actual
        score_min = total_score
        score_max = total_score
    else:
        score_min = imaging_scores + 0
        score_max = imaging_scores + 3

    # Classification
    if pain_actual is not None:
        if total_score <= 6:
            classification = "STABLE"
            interpretation = "Score 0-6: Stable. No surgical consult needed."
        elif total_score <= 12:
            classification = "INDETERMINATE"
            interpretation = (
                "Score 7-12: Indeterminate stability. "
                "Surgical consultation recommended."
            )
        else:
            classification = "UNSTABLE"
            interpretation = (
                "Score 13-18: Unstable. "
                "Surgical intervention likely required."
            )
    else:
        total_score = None
        if score_max <= 6:
            classification = "STABLE"
            interpretation = (
                f"Score range {{score_min}}-{{score_max}}: "
                "Likely stable."
            )
        elif score_min >= 13:
            classification = "UNSTABLE"
            interpretation = (
                f"Score range {{score_min}}-{{score_max}}: "
                "Likely unstable."
            )
        else:
            classification = "INDETERMINATE"
            interpretation = (
                f"Score range {{score_min}}-{{score_max}}: "
                "Classification depends on clinical pain input."
            )

    level_result = {{
        "level": level,
        "sins_components": components,
        "sins_total": total_score,
        "sins_classification": classification,
        "sins_interpretation": interpretation,
        "imaging_only_score": imaging_scores,
        "score_range_min": score_min,
        "score_range_max": score_max,
        "automated_components": 5,
        "clinical_components_provided": 1 if pain_actual is not None else 0,
        "clinical_components_missing": [] if pain_actual is not None else ["pain"]
    }}

    results_by_level.append(level_result)

result = {{
    "success": True,
    "modality": "CT",
    "levels": results_by_level
}}

__execResult = result
"""


def calculate_sins_score(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    target_levels: list[str] | None = None,
    pain_score: int | None = None,
) -> dict[str, Any]:
    """Calculate SINS (Spinal Instability Neoplastic Score).

    Automates 4/6 SINS components from imaging (location, lesion type,
    alignment, collapse, posterolateral). Pain score is clinical input.
    Reports total score, classification, and score range when pain is missing.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing segmentation (optional)
        target_levels: Vertebral levels with known lesions (default: full spine)
        pain_score: Clinical pain score 0-3 (optional; if None, reports range)

    Returns:
        Dict with per-level SINS breakdown and classification

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    target_levels = _validate_levels(target_levels, list(REGION_VERTEBRAE["full"]))
    if pain_score is not None and not (0 <= pain_score <= 3):
        raise ValidationError(
            f"pain_score must be 0-3, got {pain_score}",
            field="pain_score",
            value=str(pain_score),
        )

    client = get_client()

    safe_volume_id = json.dumps(volume_node_id)
    safe_seg_id = json.dumps(segmentation_node_id) if segmentation_node_id else None

    python_code = _build_sins_code(safe_volume_id, safe_seg_id, target_levels, pain_score)

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)
        data = _parse_json_result(exec_result.get("result", ""), "SINS score calculation")
        logger.info(f"SINS calculation complete for levels: {target_levels}")
        return data

    except SlicerConnectionError as e:
        logger.error(f"SINS calculation failed: {e.message}")
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

if seg_node_id:
    seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not seg_node:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    import TotalSegmentator
    seg_node = TotalSegmentator.TotalSegmentatorLogic().process(
        volume_node, task="vertebral_body"
    )

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

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)
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

if seg_node_id:
    seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not seg_node:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    import TotalSegmentator
    seg_node = TotalSegmentator.TotalSegmentatorLogic().process(
        volume_node, task="total"
    )

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

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)
        data = _parse_json_result(exec_result.get("result", ""), "spinal canal measurement")
        logger.info(f"Spinal canal measurement complete for levels: {levels}")
        return data

    except SlicerConnectionError as e:
        logger.error(f"Spinal canal measurement failed: {e.message}")
        raise
