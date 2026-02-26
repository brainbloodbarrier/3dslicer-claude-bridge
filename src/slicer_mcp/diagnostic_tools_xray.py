"""X-ray diagnostic protocol tools for spine analysis.

All measurements are 2D projected. DICOM X-rays load as single-slice volumes
in Slicer. The semi-automatic pipeline: Claude sees screenshot -> positions
fiducials via Markups API -> computes.

Angles are NOT affected by magnification. Distances include magnification
disclaimer.

References:
    - Schwab F et al. Spine 2012 (SRS-Schwab classification)
    - Roussouly P et al. Spine 2005 (lordosis types)
    - White AA, Panjabi MM. Clinical Biomechanics of the Spine. 1990
    - Genant HK et al. JBMR 1993;8(9):1137-48
    - Meyerding HW. Surg Gynecol Obstet. 1932;54:371-377
"""

import json
import logging
import math
import re
from typing import Any

from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    CORONAL_C7_CSVL_THRESHOLD_MM,
    CORONAL_COBB_MILD_THRESHOLD_DEG,
    CORONAL_COBB_MODERATE_THRESHOLD_DEG,
    DYNAMIC_INSTABILITY_THRESHOLDS,
    GENANT_THRESHOLDS,
    MEYERDING_THRESHOLDS,
    ROUSSOULY_SS_THRESHOLDS,
    SCHWAB_PI_LL_THRESHOLDS,
    SCHWAB_PT_THRESHOLDS,
    SCHWAB_SVA_THRESHOLDS,
    VERTEBRA_LABEL_PATTERN,
    VERTEBRA_LEVEL_PATTERN,
)
from slicer_mcp.tools import ValidationError, _parse_json_result, validate_mrml_node_id

logger = logging.getLogger("slicer-mcp")

# =============================================================================
# X-ray Constants
# =============================================================================

# Valid X-ray view types
VALID_XRAY_VIEWS = frozenset(["lateral", "ap"])

# Maximum number of landmarks per tool (safety limit)
MAX_LANDMARKS = 100

# Magnification disclaimer for distance measurements
MAGNIFICATION_DISCLAIMER = (
    "Distance measurements on projected X-ray are subject to magnification error. "
    "Use known-size calibration marker or specify magnification factor for accurate "
    "absolute distances. Angles are NOT affected by magnification."
)

# =============================================================================
# Geometry Helpers (2D)
# =============================================================================


def _angle_between_lines_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> float:
    """Compute angle between two 2D lines defined by point pairs.

    Line 1: p1->p2, Line 2: p3->p4.
    Returns angle in degrees (0-180).
    """
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = p4[0] - p3[0], p4[1] - p3[1]

    dot = dx1 * dx2 + dy1 * dy2
    mag1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    mag2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _distance_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def _signed_angle_2d(vec: tuple[float, float], ref: tuple[float, float]) -> float:
    """Compute signed angle from reference vector to vec using atan2.

    Returns angle in degrees (-180 to +180).
    Positive = counter-clockwise from reference.
    """
    # Cross product (z-component) and dot product
    cross = ref[0] * vec[1] - ref[1] * vec[0]
    dot = ref[0] * vec[0] + ref[1] * vec[1]
    if abs(cross) < 1e-12 and abs(dot) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(cross, dot))


def _cobb_angle_2d(
    sup_left: tuple[float, float],
    sup_right: tuple[float, float],
    inf_left: tuple[float, float],
    inf_right: tuple[float, float],
) -> float:
    """Compute Cobb angle from superior and inferior endplate endpoints.

    The Cobb angle is the angle between the superior endplate of the upper
    end vertebra and the inferior endplate of the lower end vertebra.
    """
    return _angle_between_lines_2d(sup_left, sup_right, inf_left, inf_right)


def _classify_schwab_pi_ll(pi_ll_mismatch: float) -> str:
    """Classify PI-LL mismatch per SRS-Schwab.

    Args:
        pi_ll_mismatch: Absolute PI - LL value in degrees.

    Returns:
        Classification string: "matched", "moderate", or "marked".
    """
    abs_val = abs(pi_ll_mismatch)
    if abs_val < SCHWAB_PI_LL_THRESHOLDS["matched"]:
        return "matched"
    elif abs_val < SCHWAB_PI_LL_THRESHOLDS["moderate"]:
        return "moderate"
    return "marked"


def _classify_schwab_sva(sva_mm: float) -> str:
    """Classify SVA per SRS-Schwab.

    Args:
        sva_mm: SVA in millimeters.

    Returns:
        Classification: "0" (aligned), "+" (moderate), "++" (marked).
    """
    if sva_mm < SCHWAB_SVA_THRESHOLDS["grade_0"]:
        return "0"
    elif sva_mm < SCHWAB_SVA_THRESHOLDS["grade_1"]:
        return "+"
    return "++"


def _classify_schwab_pt(pt_deg: float) -> str:
    """Classify pelvic tilt per SRS-Schwab.

    Args:
        pt_deg: Pelvic tilt in degrees.

    Returns:
        Classification: "0" (normal), "+" (moderate), "++" (marked).
    """
    if pt_deg < SCHWAB_PT_THRESHOLDS["grade_0"]:
        return "0"
    elif pt_deg < SCHWAB_PT_THRESHOLDS["grade_1"]:
        return "+"
    return "++"


def _classify_roussouly(ss_deg: float, ll_deg: float, apex_level: str) -> str:
    """Classify Roussouly lordosis type.

    Args:
        ss_deg: Sacral slope in degrees.
        ll_deg: Lumbar lordosis in degrees.
        apex_level: Apex of lumbar lordosis (e.g. "L4", "L5").

    Returns:
        Roussouly type string: "Type 1", "Type 2", "Type 3", or "Type 4".
    """
    if ss_deg < ROUSSOULY_SS_THRESHOLDS["type_1_max"]:
        # Low SS: distinguish Type 1 (short, low lordosis) vs Type 2 (flat back)
        if abs(ll_deg) < 45.0 and apex_level in ("L5", "S1"):
            return "Type 1"
        return "Type 2"
    elif ss_deg <= ROUSSOULY_SS_THRESHOLDS["type_3_max"]:
        return "Type 3"
    return "Type 4"


def _classify_genant(height_reduction: float) -> dict[str, Any]:
    """Classify vertebral fracture per Genant semi-quantitative grading.

    Args:
        height_reduction: Fractional height reduction (0.0 = no loss, 1.0 = complete loss).

    Returns:
        Dict with grade (int 0-3), label, and description.
    """
    if height_reduction >= GENANT_THRESHOLDS["severe_min"]:
        return {"grade": 3, "label": "severe", "description": ">40% height reduction"}
    elif height_reduction >= GENANT_THRESHOLDS["moderate_min"]:
        return {"grade": 2, "label": "moderate", "description": "25-40% height reduction"}
    elif height_reduction >= GENANT_THRESHOLDS["mild_min"]:
        return {"grade": 1, "label": "mild", "description": "20-25% height reduction"}
    return {"grade": 0, "label": "normal", "description": "<20% height reduction"}


def _classify_meyerding(slip_fraction: float) -> dict[str, Any]:
    """Classify spondylolisthesis per Meyerding grading.

    Args:
        slip_fraction: Fraction of vertebral body slip (0.0-1.0+).

    Returns:
        Dict with grade (int 1-5) and label.
    """
    if slip_fraction > MEYERDING_THRESHOLDS["grade_iv_max"]:
        return {"grade": 5, "label": "spondyloptosis"}
    elif slip_fraction > MEYERDING_THRESHOLDS["grade_iii_max"]:
        return {"grade": 4, "label": "Grade IV (75-100%)"}
    elif slip_fraction > MEYERDING_THRESHOLDS["grade_ii_max"]:
        return {"grade": 3, "label": "Grade III (50-75%)"}
    elif slip_fraction > MEYERDING_THRESHOLDS["grade_i_max"]:
        return {"grade": 2, "label": "Grade II (25-50%)"}
    return {"grade": 1, "label": "Grade I (0-25%)"}


def _is_dynamic_unstable(
    translation_mm: float, angulation_deg: float, region: str
) -> dict[str, Any]:
    """Check White & Panjabi dynamic instability criteria.

    Args:
        translation_mm: Sagittal translation in mm.
        angulation_deg: Sagittal angulation in degrees.
        region: Spine region ("cervical" or "lumbar").

    Returns:
        Dict with unstable flag and criteria details.
    """
    if region == "cervical":
        trans_thresh = DYNAMIC_INSTABILITY_THRESHOLDS["cervical_translation_mm"]
        ang_thresh = DYNAMIC_INSTABILITY_THRESHOLDS["cervical_angulation_deg"]
    else:
        trans_thresh = DYNAMIC_INSTABILITY_THRESHOLDS["lumbar_translation_mm"]
        ang_thresh = DYNAMIC_INSTABILITY_THRESHOLDS["lumbar_angulation_deg"]

    translation_unstable = translation_mm > trans_thresh
    angulation_unstable = angulation_deg > ang_thresh

    return {
        "unstable": translation_unstable or angulation_unstable,
        "translation_mm": round(translation_mm, 2),
        "translation_threshold_mm": trans_thresh,
        "translation_exceeds": translation_unstable,
        "angulation_deg": round(angulation_deg, 2),
        "angulation_threshold_deg": ang_thresh,
        "angulation_exceeds": angulation_unstable,
        "criteria": "White & Panjabi 1990",
    }


# =============================================================================
# Landmark Validation
# =============================================================================


def _validate_landmarks(
    landmarks: dict[str, list[float]], expected_keys: list[str], tool_name: str
) -> dict[str, tuple[float, float]]:
    """Validate landmark dict: each key must map to [x, y] coordinate.

    Args:
        landmarks: Dict mapping landmark names to [x, y] coordinates.
        expected_keys: List of required landmark names.
        tool_name: Tool name for error messages.

    Returns:
        Dict mapping landmark names to (x, y) tuples.

    Raises:
        ValidationError: If landmarks are missing, malformed, or exceed limits.
    """
    if not landmarks:
        raise ValidationError(
            f"{tool_name}: landmarks dict cannot be empty",
            field="landmarks",
            value="{}",
        )

    if len(landmarks) > MAX_LANDMARKS:
        raise ValidationError(
            f"{tool_name}: too many landmarks ({len(landmarks)} > {MAX_LANDMARKS})",
            field="landmarks",
            value=str(len(landmarks)),
        )

    missing = [k for k in expected_keys if k not in landmarks]
    if missing:
        raise ValidationError(
            f"{tool_name}: missing required landmarks: {', '.join(missing)}",
            field="landmarks",
            value=str(missing),
        )

    result: dict[str, tuple[float, float]] = {}
    for key in expected_keys:
        coord = landmarks[key]
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise ValidationError(
                f"{tool_name}: landmark '{key}' must be [x, y], got: {coord}",
                field=f"landmarks.{key}",
                value=str(coord),
            )
        try:
            result[key] = (float(coord[0]), float(coord[1]))
        except (TypeError, ValueError):
            raise ValidationError(
                f"{tool_name}: landmark '{key}' coordinates must be numeric",
                field=f"landmarks.{key}",
                value=str(coord),
            )

    return result


def _validate_vertebra_label(label: str, tool_name: str) -> str:
    """Validate a vertebra label (e.g., 'T12') or level (e.g., 'L4-L5').

    Args:
        label: Vertebra label or level string.
        tool_name: Tool name for error messages.

    Returns:
        The validated label string.

    Raises:
        ValidationError: If label doesn't match expected pattern.
    """
    if re.match(VERTEBRA_LABEL_PATTERN, label) or re.match(VERTEBRA_LEVEL_PATTERN, label):
        return label
    raise ValidationError(
        f"{tool_name}: invalid vertebra label '{label}'. "
        f"Must match pattern like 'T12' or 'L4-L5'",
        field="vertebra_label",
        value=label,
    )


# =============================================================================
# Slicer Python Code Builders
# =============================================================================


def _build_place_landmarks_code(
    volume_node_id: str, landmarks: dict[str, tuple[float, float]], list_name: str
) -> str:
    """Build Slicer Python code to place fiducial landmarks on an X-ray.

    Args:
        volume_node_id: MRML node ID of the X-ray volume.
        landmarks: Dict of landmark_name -> (x, y) coordinates.
        list_name: Name for the markup fiducial list node.

    Returns:
        Python code string for Slicer execution.
    """
    safe_node_id = json.dumps(volume_node_id)
    safe_list_name = json.dumps(list_name)

    # Build landmark placement lines
    landmark_lines = []
    for name, (x, y) in landmarks.items():
        safe_name = json.dumps(name)
        landmark_lines.append(
            f"    idx = markupsNode.AddControlPoint(vtk.vtkVector3d({x}, {y}, 0.0))\n"
            f"    markupsNode.SetNthControlPointLabel(idx, {safe_name})"
        )

    landmarks_code = "\n".join(landmark_lines)

    return f"""
import slicer
import vtk
import json

volume_node_id = {safe_node_id}
list_name = {safe_list_name}

# Verify volume exists
volumeNode = slicer.mrmlScene.GetNodeByID(volume_node_id)
if not volumeNode:
    raise ValueError('Volume node not found: ' + volume_node_id)

# Create markup fiducial list
markupsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', list_name)
markupsNode.CreateDefaultDisplayNodes()

# Place landmarks
{landmarks_code}

result = {{
    'success': True,
    'markups_node_id': markupsNode.GetID(),
    'markups_node_name': markupsNode.GetName(),
    'num_landmarks': markupsNode.GetNumberOfControlPoints()
}}

print(json.dumps(result))
"""


# =============================================================================
# Tool 1: Sagittal Balance (Lateral X-ray)
# =============================================================================

# Required landmarks for sagittal balance analysis
SAGITTAL_BALANCE_LANDMARKS = [
    # C2 body centroid (for C2-C7 SVA, C2 plumb line)
    "C2_centroid",
    # C7 body centroid (for SVA, C2-C7 SVA)
    "C7_centroid",
    # C2 superior endplate endpoints (for cervical lordosis)
    "C2_sup_ant",
    "C2_sup_post",
    # C7 inferior endplate endpoints (for cervical lordosis, T1 slope reference)
    "C7_inf_ant",
    "C7_inf_post",
    # T1 superior endplate endpoints (for T1 slope)
    "T1_sup_ant",
    "T1_sup_post",
    # T4 superior endplate endpoints (for thoracic kyphosis)
    "T4_sup_ant",
    "T4_sup_post",
    # T12 inferior endplate endpoints (for thoracic kyphosis + lumbar lordosis)
    "T12_inf_ant",
    "T12_inf_post",
    # L1 superior endplate endpoints (for lumbar lordosis)
    "L1_sup_ant",
    "L1_sup_post",
    # S1 superior endplate endpoints (for lumbar lordosis, SS, PI)
    "S1_sup_ant",
    "S1_sup_post",
    # Sacral endplate midpoint (for SS line)
    "S1_endplate_mid",
    # Femoral head centers (for SVA reference, PT, PI)
    "femoral_head_center_L",
    "femoral_head_center_R",
    # Posterior superior corner of S1 (for PT line)
    "S1_post_sup",
]


def measure_sagittal_balance_xray(
    volume_node_id: str,
    landmarks: dict[str, list[float]],
    magnification_factor: float = 1.0,
) -> dict[str, Any]:
    """Measure sagittal spinal balance parameters from lateral standing X-ray.

    Computes SVA, C2-C7 SVA, T1 slope, TPA, cervical lordosis, thoracic
    kyphosis, lumbar lordosis (all Cobb method), pelvic parameters (PI, PT, SS),
    PI-LL mismatch, SRS-Schwab classification, and Roussouly type.

    Args:
        volume_node_id: MRML node ID of the lateral X-ray volume.
        landmarks: Dict mapping landmark names to [x, y] coordinates.
            Required landmarks (20): C2_centroid, C7_centroid, C2_sup_ant,
            C2_sup_post, C7_inf_ant, C7_inf_post, T1_sup_ant, T1_sup_post,
            T4_sup_ant, T4_sup_post, T12_inf_ant, T12_inf_post, L1_sup_ant,
            L1_sup_post, S1_sup_ant, S1_sup_post, S1_endplate_mid,
            femoral_head_center_L, femoral_head_center_R, S1_post_sup.
        magnification_factor: X-ray magnification correction factor (default 1.0).

    Returns:
        Dict with sagittal balance parameters, classifications, and metadata.

    Raises:
        ValidationError: If inputs are invalid or landmarks are missing.
        SlicerConnectionError: If Slicer communication fails.
    """
    # Validate inputs
    volume_node_id = validate_mrml_node_id(volume_node_id)
    pts = _validate_landmarks(landmarks, SAGITTAL_BALANCE_LANDMARKS, "sagittal_balance")

    if magnification_factor <= 0:
        raise ValidationError(
            "magnification_factor must be positive",
            field="magnification_factor",
            value=str(magnification_factor),
        )

    # Place landmarks in Slicer
    client = get_client()
    place_code = _build_place_landmarks_code(volume_node_id, pts, "SagittalBalance_Landmarks")

    try:
        exec_result = client.exec_python(place_code)
        _parse_json_result(exec_result.get("result", ""), "sagittal balance landmark placement")
    except SlicerConnectionError:
        logger.error("Failed to place sagittal balance landmarks in Slicer")
        raise

    # Compute measurements locally from 2D coordinates
    # Femoral head bicoxal midpoint
    fem_mid = (
        (pts["femoral_head_center_L"][0] + pts["femoral_head_center_R"][0]) / 2,
        (pts["femoral_head_center_L"][1] + pts["femoral_head_center_R"][1]) / 2,
    )

    # SVA: horizontal distance from C7 centroid to posterior superior S1
    # Positive = anterior displacement (forward imbalance)
    sva_mm = (pts["C7_centroid"][0] - pts["S1_post_sup"][0]) / magnification_factor

    # C2-C7 SVA: horizontal distance from C2 plumb to C7
    c2_c7_sva_mm = (pts["C2_centroid"][0] - pts["C7_centroid"][0]) / magnification_factor

    # T1 slope: angle between T1 superior endplate and horizontal (signed)
    t1_vec = (
        pts["T1_sup_ant"][0] - pts["T1_sup_post"][0],
        pts["T1_sup_ant"][1] - pts["T1_sup_post"][1],
    )
    t1_slope = _signed_angle_2d(t1_vec, (1, 0))

    # Cervical Lordosis (CL): C2 sup endplate to C7 inf endplate (Cobb)
    cl = _cobb_angle_2d(
        pts["C2_sup_ant"],
        pts["C2_sup_post"],
        pts["C7_inf_ant"],
        pts["C7_inf_post"],
    )

    # Thoracic Kyphosis (TK): T4 sup endplate to T12 inf endplate (Cobb)
    tk = _cobb_angle_2d(
        pts["T4_sup_ant"],
        pts["T4_sup_post"],
        pts["T12_inf_ant"],
        pts["T12_inf_post"],
    )

    # Lumbar Lordosis (LL): L1 sup endplate to S1 sup endplate (Cobb)
    ll = _cobb_angle_2d(
        pts["L1_sup_ant"],
        pts["L1_sup_post"],
        pts["S1_sup_ant"],
        pts["S1_sup_post"],
    )

    # Sacral Slope (SS): signed angle between S1 endplate and horizontal
    s1_vec = (
        pts["S1_sup_ant"][0] - pts["S1_sup_post"][0],
        pts["S1_sup_ant"][1] - pts["S1_sup_post"][1],
    )
    ss = _signed_angle_2d(s1_vec, (1, 0))

    # Pelvic Tilt (PT): signed angle between S1_mid->fem_head and vertical
    pt_vec = (
        fem_mid[0] - pts["S1_endplate_mid"][0],
        fem_mid[1] - pts["S1_endplate_mid"][1],
    )
    pt_val = _signed_angle_2d(pt_vec, (0, 1))

    # Pelvic Incidence (PI): angle between perpendicular to S1 endplate
    # and line from S1_mid to femoral head center. PI is always positive.
    # Perpendicular to S1 endplate: rotate s1_vec 90deg CCW
    s1_perp = (-s1_vec[1], s1_vec[0])
    s1_to_fem = (
        fem_mid[0] - pts["S1_endplate_mid"][0],
        fem_mid[1] - pts["S1_endplate_mid"][1],
    )
    pi_val = abs(_signed_angle_2d(s1_to_fem, s1_perp))

    # PI-LL mismatch
    pi_ll = pi_val - ll

    # T1 Pelvic Angle (TPA): angle between line from T1 centroid to femoral
    # head center and line from S1 endplate midpoint to femoral head center
    # TPA is unsigned
    t1_centroid = (
        (pts["T1_sup_ant"][0] + pts["T1_sup_post"][0]) / 2,
        (pts["T1_sup_ant"][1] + pts["T1_sup_post"][1]) / 2,
    )
    tpa = _angle_between_lines_2d(
        t1_centroid,
        fem_mid,
        pts["S1_endplate_mid"],
        fem_mid,
    )

    # SRS-Schwab classification
    schwab = {
        "PI_LL": _classify_schwab_pi_ll(pi_ll),
        "SVA": _classify_schwab_sva(abs(sva_mm)),
        "PT": _classify_schwab_pt(pt_val),
    }

    # Roussouly type (estimate apex from L4/L5 region)
    roussouly = _classify_roussouly(ss, ll, "L4")

    result = {
        "success": True,
        "tool": "measure_sagittal_balance_xray",
        "parameters": {
            "SVA_mm": round(sva_mm, 2),
            "C2_C7_SVA_mm": round(c2_c7_sva_mm, 2),
            "T1_slope_deg": round(t1_slope, 2),
            "TPA_deg": round(tpa, 2),
            "cervical_lordosis_deg": round(cl, 2),
            "thoracic_kyphosis_deg": round(tk, 2),
            "lumbar_lordosis_deg": round(ll, 2),
            "pelvic_incidence_deg": round(pi_val, 2),
            "pelvic_tilt_deg": round(pt_val, 2),
            "sacral_slope_deg": round(ss, 2),
            "PI_LL_mismatch_deg": round(pi_ll, 2),
        },
        "classifications": {
            "SRS_Schwab": schwab,
            "Roussouly_type": roussouly,
        },
        "magnification_factor": magnification_factor,
        "disclaimer": MAGNIFICATION_DISCLAIMER,
        "num_landmarks": len(pts),
        "references": [
            "Schwab F et al. Spine 2012 (SRS-Schwab classification)",
            "Roussouly P et al. Spine 2005 (lordosis type classification)",
        ],
    }

    logger.info(
        f"Sagittal balance measured: SVA={sva_mm:.1f}mm, PI-LL={pi_ll:.1f}°, "
        f"Schwab PI-LL={schwab['PI_LL']}"
    )

    return result


# =============================================================================
# Tool 2: Coronal Balance (AP X-ray)
# =============================================================================

CORONAL_BALANCE_LANDMARKS = [
    # C7 centroid (for C7 plumb line)
    "C7_centroid",
    # CSVL reference: center of sacrum
    "sacrum_center",
    # Trunk shift: T1 centroid or clavicle midpoint
    "T1_centroid",
    # Shoulder landmarks
    "shoulder_L",
    "shoulder_R",
    # Pelvis landmarks (iliac crests)
    "iliac_crest_L",
    "iliac_crest_R",
    # Coronal Cobb: upper end vertebra endplate
    "upper_end_vertebra_L",
    "upper_end_vertebra_R",
    # Coronal Cobb: lower end vertebra endplate
    "lower_end_vertebra_L",
    "lower_end_vertebra_R",
]


def measure_coronal_balance_xray(
    volume_node_id: str,
    landmarks: dict[str, list[float]],
    magnification_factor: float = 1.0,
) -> dict[str, Any]:
    """Measure coronal spinal balance from AP standing X-ray.

    Computes C7 plumb line offset to CSVL, trunk shift, pelvic obliquity,
    shoulder balance, and coronal Cobb angle.

    Args:
        volume_node_id: MRML node ID of the AP X-ray volume.
        landmarks: Dict mapping landmark names to [x, y] coordinates.
            Required landmarks (11): C7_centroid, sacrum_center, T1_centroid,
            shoulder_L, shoulder_R, iliac_crest_L, iliac_crest_R,
            upper_end_vertebra_L, upper_end_vertebra_R,
            lower_end_vertebra_L, lower_end_vertebra_R.
        magnification_factor: X-ray magnification correction factor (default 1.0).

    Returns:
        Dict with coronal balance parameters and metadata.

    Raises:
        ValidationError: If inputs are invalid or landmarks are missing.
        SlicerConnectionError: If Slicer communication fails.
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    pts = _validate_landmarks(landmarks, CORONAL_BALANCE_LANDMARKS, "coronal_balance")

    if magnification_factor <= 0:
        raise ValidationError(
            "magnification_factor must be positive",
            field="magnification_factor",
            value=str(magnification_factor),
        )

    # Place landmarks in Slicer
    client = get_client()
    place_code = _build_place_landmarks_code(volume_node_id, pts, "CoronalBalance_Landmarks")

    try:
        exec_result = client.exec_python(place_code)
        _parse_json_result(exec_result.get("result", ""), "coronal balance landmark placement")
    except SlicerConnectionError:
        logger.error("Failed to place coronal balance landmarks in Slicer")
        raise

    # C7 plumb line to CSVL offset (horizontal distance)
    c7_csvl_offset_mm = (pts["C7_centroid"][0] - pts["sacrum_center"][0]) / magnification_factor

    # Trunk shift: T1 centroid offset from CSVL
    trunk_shift_mm = (pts["T1_centroid"][0] - pts["sacrum_center"][0]) / magnification_factor

    # Pelvic obliquity: angle of iliac crest line from horizontal
    pelvic_obliquity = _angle_between_lines_2d(
        pts["iliac_crest_L"],
        pts["iliac_crest_R"],
        (0, 0),
        (1, 0),
    )

    # Shoulder balance: height difference (Y-axis)
    shoulder_balance_mm = (pts["shoulder_R"][1] - pts["shoulder_L"][1]) / magnification_factor

    # Coronal Cobb angle
    coronal_cobb = _cobb_angle_2d(
        pts["upper_end_vertebra_L"],
        pts["upper_end_vertebra_R"],
        pts["lower_end_vertebra_L"],
        pts["lower_end_vertebra_R"],
    )

    result = {
        "success": True,
        "tool": "measure_coronal_balance_xray",
        "parameters": {
            "C7_CSVL_offset_mm": round(c7_csvl_offset_mm, 2),
            "trunk_shift_mm": round(trunk_shift_mm, 2),
            "pelvic_obliquity_deg": round(pelvic_obliquity, 2),
            "shoulder_balance_mm": round(shoulder_balance_mm, 2),
            "coronal_cobb_angle_deg": round(coronal_cobb, 2),
        },
        "interpretation": {
            "C7_CSVL": (
                "balanced"
                if abs(c7_csvl_offset_mm) < CORONAL_C7_CSVL_THRESHOLD_MM
                else "imbalanced"
            ),
            "coronal_cobb_severity": (
                "mild"
                if coronal_cobb < CORONAL_COBB_MILD_THRESHOLD_DEG
                else "moderate" if coronal_cobb < CORONAL_COBB_MODERATE_THRESHOLD_DEG else "severe"
            ),
        },
        "magnification_factor": magnification_factor,
        "disclaimer": MAGNIFICATION_DISCLAIMER,
        "num_landmarks": len(pts),
    }

    logger.info(
        f"Coronal balance measured: C7-CSVL={c7_csvl_offset_mm:.1f}mm, " f"Cobb={coronal_cobb:.1f}°"
    )

    return result


# =============================================================================
# Tool 3: Dynamic Listhesis (3 X-rays: neutral + flex + ext)
# =============================================================================

# Per-level landmarks for listhesis measurement
LISTHESIS_LEVEL_LANDMARKS = [
    "sup_post_inf",  # Superior vertebra posterior-inferior corner
    "inf_post_sup",  # Inferior vertebra posterior-superior corner
    "sup_ant_inf",  # Superior vertebra anterior-inferior corner
    "inf_ant_sup",  # Inferior vertebra anterior-superior corner
    "sup_inf_endplate_ant",  # Superior vertebra inferior endplate anterior
    "sup_inf_endplate_post",  # Superior vertebra inferior endplate posterior
    "inf_sup_endplate_ant",  # Inferior vertebra superior endplate anterior
    "inf_sup_endplate_post",  # Inferior vertebra superior endplate posterior
]


def measure_listhesis_dynamic_xray(
    volume_node_ids: dict[str, str],
    landmarks_per_position: dict[str, dict[str, dict[str, list[float]]]],
    levels: list[str],
    region: str = "lumbar",
    magnification_factor: float = 1.0,
) -> dict[str, Any]:
    """Measure dynamic listhesis from neutral, flexion, and extension X-rays.

    Processes 3 lateral X-rays simultaneously. Computes translation and angular
    motion per position per level, applies White & Panjabi instability criteria,
    and performs Meyerding grading at worst position.

    Args:
        volume_node_ids: Dict mapping position to MRML node ID.
            Required keys: "neutral", "flexion", "extension".
        landmarks_per_position: Nested dict: position -> level -> landmark -> [x, y].
            Each level requires 8 landmarks (see LISTHESIS_LEVEL_LANDMARKS).
        levels: List of spinal levels to assess (e.g., ["L4-L5", "L5-S1"]).
        region: Spine region for instability thresholds ("cervical" or "lumbar").
        magnification_factor: X-ray magnification correction factor (default 1.0).

    Returns:
        Dict with per-level measurements, instability assessment, and Meyerding grading.

    Raises:
        ValidationError: If inputs are invalid or landmarks are missing.
        SlicerConnectionError: If Slicer communication fails.
    """
    # Validate volume node IDs
    required_positions = ["neutral", "flexion", "extension"]
    for pos in required_positions:
        if pos not in volume_node_ids:
            raise ValidationError(
                f"listhesis_dynamic: missing volume_node_id for position '{pos}'",
                field="volume_node_ids",
                value=str(list(volume_node_ids.keys())),
            )
        validate_mrml_node_id(volume_node_ids[pos])

    if region not in ("cervical", "lumbar"):
        raise ValidationError(
            f"listhesis_dynamic: region must be 'cervical' or 'lumbar', got '{region}'",
            field="region",
            value=region,
        )

    if not levels:
        raise ValidationError(
            "listhesis_dynamic: levels list cannot be empty",
            field="levels",
            value="[]",
        )

    for level in levels:
        _validate_vertebra_label(level, "listhesis_dynamic")

    if magnification_factor <= 0:
        raise ValidationError(
            "magnification_factor must be positive",
            field="magnification_factor",
            value=str(magnification_factor),
        )

    # Validate landmarks structure
    for pos in required_positions:
        if pos not in landmarks_per_position:
            raise ValidationError(
                f"listhesis_dynamic: missing landmarks for position '{pos}'",
                field="landmarks_per_position",
                value=str(list(landmarks_per_position.keys())),
            )
        for level in levels:
            if level not in landmarks_per_position[pos]:
                raise ValidationError(
                    f"listhesis_dynamic: missing landmarks for {pos}/{level}",
                    field=f"landmarks_per_position.{pos}",
                    value=str(list(landmarks_per_position[pos].keys())),
                )

    # Place landmarks in Slicer for each position, caching validated results
    validated_landmarks: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    client = get_client()
    for pos in required_positions:
        validated_landmarks[pos] = {}
        all_pos_landmarks: dict[str, tuple[float, float]] = {}
        for level in levels:
            level_lm = _validate_landmarks(
                landmarks_per_position[pos][level],
                LISTHESIS_LEVEL_LANDMARKS,
                f"listhesis_{pos}_{level}",
            )
            validated_landmarks[pos][level] = level_lm
            for name, coord in level_lm.items():
                all_pos_landmarks[f"{level}_{name}"] = coord

        place_code = _build_place_landmarks_code(
            volume_node_ids[pos], all_pos_landmarks, f"Listhesis_{pos}_Landmarks"
        )

        try:
            exec_result = client.exec_python(place_code)
            _parse_json_result(
                exec_result.get("result", ""),
                f"listhesis {pos} landmark placement",
            )
        except SlicerConnectionError:
            logger.error(f"Failed to place listhesis landmarks for {pos}")
            raise

    # Compute per-level, per-position measurements
    level_results = []
    worst_meyerding: dict[str, Any] = {"grade": 0, "label": "none", "level": "", "position": ""}

    for level in levels:
        level_data: dict[str, Any] = {"level": level, "positions": {}}

        for pos in required_positions:
            lm = validated_landmarks[pos][level]

            # Translation: posterior wall offset
            translation_mm = (
                abs(lm["sup_post_inf"][0] - lm["inf_post_sup"][0]) / magnification_factor
            )

            # Slip fraction for Meyerding: anterior offset / inferior body width
            inf_body_width = _distance_2d(lm["inf_ant_sup"], lm["inf_post_sup"])
            if inf_body_width > 1e-6:
                slip_fraction = abs(lm["sup_ant_inf"][0] - lm["inf_ant_sup"][0]) / inf_body_width
            else:
                slip_fraction = 0.0

            # Angular motion: angle between endplates
            angulation = _cobb_angle_2d(
                lm["sup_inf_endplate_ant"],
                lm["sup_inf_endplate_post"],
                lm["inf_sup_endplate_ant"],
                lm["inf_sup_endplate_post"],
            )

            meyerding = _classify_meyerding(slip_fraction)

            level_data["positions"][pos] = {
                "translation_mm": round(translation_mm, 2),
                "slip_fraction": round(slip_fraction, 3),
                "angulation_deg": round(angulation, 2),
                "meyerding": meyerding,
            }

            # Track worst Meyerding
            if meyerding["grade"] > worst_meyerding["grade"]:
                worst_meyerding = {
                    **meyerding,
                    "level": level,
                    "position": pos,
                }

        # Dynamic instability: max translation and angular range across positions
        translations = [level_data["positions"][p]["translation_mm"] for p in required_positions]
        angulations = [level_data["positions"][p]["angulation_deg"] for p in required_positions]

        max_translation = max(translations)
        angular_range = max(angulations) - min(angulations)

        instability = _is_dynamic_unstable(max_translation, angular_range, region)
        level_data["dynamic_instability"] = instability
        level_data["max_translation_mm"] = round(max_translation, 2)
        level_data["angular_range_deg"] = round(angular_range, 2)

        level_results.append(level_data)

    any_unstable = any(lr["dynamic_instability"]["unstable"] for lr in level_results)

    result = {
        "success": True,
        "tool": "measure_listhesis_dynamic_xray",
        "levels": level_results,
        "worst_meyerding": worst_meyerding,
        "any_dynamic_instability": any_unstable,
        "instability_pattern": ("unstable" if any_unstable else "stable"),
        "region": region,
        "magnification_factor": magnification_factor,
        "disclaimer": MAGNIFICATION_DISCLAIMER,
        "references": [
            "White AA, Panjabi MM. Clinical Biomechanics of the Spine. 1990",
            "Meyerding HW. Surg Gynecol Obstet. 1932;54:371-377",
        ],
    }

    logger.info(
        f"Dynamic listhesis measured: {len(levels)} levels, "
        f"worst Meyerding={worst_meyerding['grade']}, "
        f"unstable={any_unstable}"
    )

    return result


# =============================================================================
# Tool 4: Vertebral Fracture Detection (Lateral X-ray)
# =============================================================================

# 6 points per vertebral body for Genant semi-quantitative
FRACTURE_VERTEBRA_LANDMARKS = [
    "ant_sup",  # Anterior superior corner
    "ant_inf",  # Anterior inferior corner
    "mid_sup",  # Middle superior point
    "mid_inf",  # Middle inferior point
    "post_sup",  # Posterior superior corner
    "post_inf",  # Posterior inferior corner
]


def detect_vertebral_fractures_xray(
    volume_node_id: str,
    landmarks_per_vertebra: dict[str, dict[str, list[float]]],
    magnification_factor: float = 1.0,
) -> dict[str, Any]:
    """Detect vertebral fractures using Genant semi-quantitative method.

    6 points per vertebral body define anterior, middle, and posterior heights.
    Height reduction relative to expected (adjacent vertebra) determines grade.

    Args:
        volume_node_id: MRML node ID of the lateral X-ray volume.
        landmarks_per_vertebra: Dict: vertebra_label -> landmark_name -> [x, y].
            Each vertebra requires 6 landmarks (ant_sup, ant_inf, mid_sup,
            mid_inf, post_sup, post_inf).
        magnification_factor: X-ray magnification correction factor (default 1.0).

    Returns:
        Dict with per-vertebra fracture assessment, Genant grades, and summary.

    Raises:
        ValidationError: If inputs are invalid or landmarks are missing.
        SlicerConnectionError: If Slicer communication fails.
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)

    if not landmarks_per_vertebra:
        raise ValidationError(
            "detect_fractures: landmarks_per_vertebra cannot be empty",
            field="landmarks_per_vertebra",
            value="{}",
        )

    for vert_label in landmarks_per_vertebra:
        _validate_vertebra_label(vert_label, "detect_fractures")

    if magnification_factor <= 0:
        raise ValidationError(
            "magnification_factor must be positive",
            field="magnification_factor",
            value=str(magnification_factor),
        )

    # Validate and collect all landmarks
    all_landmarks: dict[str, tuple[float, float]] = {}
    vertebra_pts: dict[str, dict[str, tuple[float, float]]] = {}

    for vert_label, vert_landmarks in landmarks_per_vertebra.items():
        validated = _validate_landmarks(
            vert_landmarks,
            FRACTURE_VERTEBRA_LANDMARKS,
            f"detect_fractures_{vert_label}",
        )
        vertebra_pts[vert_label] = validated
        for name, coord in validated.items():
            all_landmarks[f"{vert_label}_{name}"] = coord

    # Place landmarks in Slicer
    client = get_client()
    place_code = _build_place_landmarks_code(
        volume_node_id, all_landmarks, "VertebralFractures_Landmarks"
    )

    try:
        exec_result = client.exec_python(place_code)
        _parse_json_result(exec_result.get("result", ""), "fracture landmark placement")
    except SlicerConnectionError:
        logger.error("Failed to place fracture detection landmarks in Slicer")
        raise

    # Compute heights and fracture grades
    vertebra_labels = list(vertebra_pts.keys())
    vertebra_results = []
    fracture_count = 0

    for i, vert_label in enumerate(vertebra_labels):
        vp = vertebra_pts[vert_label]

        # Compute 3 heights (anterior, middle, posterior)
        anterior_height = _distance_2d(vp["ant_sup"], vp["ant_inf"])
        middle_height = _distance_2d(vp["mid_sup"], vp["mid_inf"])
        posterior_height = _distance_2d(vp["post_sup"], vp["post_inf"])

        # Reference height: maximum of posterior height of this vertebra
        # and adjacent vertebrae (if available)
        ref_heights = [posterior_height]
        if i > 0:
            prev_vp = vertebra_pts[vertebra_labels[i - 1]]
            ref_heights.append(_distance_2d(prev_vp["post_sup"], prev_vp["post_inf"]))
        if i < len(vertebra_labels) - 1:
            next_vp = vertebra_pts[vertebra_labels[i + 1]]
            ref_heights.append(_distance_2d(next_vp["post_sup"], next_vp["post_inf"]))

        reference_height = max(ref_heights) if ref_heights else posterior_height

        # Height reductions (fraction)
        if reference_height > 1e-6:
            ant_reduction = max(0.0, 1.0 - anterior_height / reference_height)
            mid_reduction = max(0.0, 1.0 - middle_height / reference_height)
            post_reduction = max(0.0, 1.0 - posterior_height / reference_height)
        else:
            ant_reduction = 0.0
            mid_reduction = 0.0
            post_reduction = 0.0

        # Worst reduction determines grade
        worst_reduction = max(ant_reduction, mid_reduction, post_reduction)
        genant = _classify_genant(worst_reduction)

        # Determine fracture morphology
        if worst_reduction < GENANT_THRESHOLDS["mild_min"]:
            morphology = "normal"
        elif ant_reduction >= mid_reduction and ant_reduction >= post_reduction:
            morphology = "wedge"
        elif mid_reduction >= ant_reduction and mid_reduction >= post_reduction:
            morphology = "biconcave"
        else:
            morphology = "crush"

        # Confidence based on height consistency
        height_std = (
            (anterior_height - middle_height) ** 2
            + (middle_height - posterior_height) ** 2
            + (anterior_height - posterior_height) ** 2
        ) ** 0.5
        confidence = max(0.0, min(1.0, 1.0 - height_std / (reference_height + 1e-6) * 0.5))

        if genant["grade"] > 0:
            fracture_count += 1

        vertebra_results.append(
            {
                "vertebra": vert_label,
                "anterior_height_px": round(anterior_height, 2),
                "middle_height_px": round(middle_height, 2),
                "posterior_height_px": round(posterior_height, 2),
                "reference_height_px": round(reference_height, 2),
                "height_reductions": {
                    "anterior": round(ant_reduction, 3),
                    "middle": round(mid_reduction, 3),
                    "posterior": round(post_reduction, 3),
                },
                "worst_reduction": round(worst_reduction, 3),
                "genant_grade": genant["grade"],
                "genant_label": genant["label"],
                "genant_description": genant["description"],
                "morphology": morphology,
                "confidence": round(confidence, 3),
            }
        )

    result = {
        "success": True,
        "tool": "detect_vertebral_fractures_xray",
        "vertebrae": vertebra_results,
        "summary": {
            "total_vertebrae_assessed": len(vertebra_results),
            "fracture_count": fracture_count,
            "max_genant_grade": max((v["genant_grade"] for v in vertebra_results), default=0),
        },
        "magnification_factor": magnification_factor,
        "disclaimer": (
            "Heights in pixels (px); absolute mm values require calibration. "
            + MAGNIFICATION_DISCLAIMER
        ),
        "references": [
            "Genant HK et al. JBMR 1993;8(9):1137-48",
        ],
    }

    logger.info(
        f"Vertebral fractures assessed: {len(vertebra_results)} vertebrae, "
        f"{fracture_count} fractures detected"
    )

    return result


# =============================================================================
# Tool 5: Cobb Angle (AP X-ray)
# =============================================================================

COBB_ANGLE_LANDMARKS = [
    # Upper end vertebra: superior endplate endpoints
    "upper_end_sup_L",
    "upper_end_sup_R",
    # Lower end vertebra: inferior endplate endpoints
    "lower_end_inf_L",
    "lower_end_inf_R",
    # Apical vertebra: centroid
    "apex_centroid",
]


def measure_cobb_angle_xray(
    volume_node_id: str,
    landmarks: dict[str, list[float]],
    upper_end_vertebra: str = "",
    lower_end_vertebra: str = "",
    curve_type: str = "primary",
) -> dict[str, Any]:
    """Measure Cobb angle for scoliosis from AP standing X-ray.

    Identifies the endvertebrae (most tilted), computes Cobb angle between
    superior endplate of upper end vertebra and inferior endplate of lower
    end vertebra, and identifies apical vertebra.

    Args:
        volume_node_id: MRML node ID of the AP X-ray volume.
        landmarks: Dict mapping landmark names to [x, y] coordinates.
            Required landmarks (5): upper_end_sup_L, upper_end_sup_R,
            lower_end_inf_L, lower_end_inf_R, apex_centroid.
        upper_end_vertebra: Label of upper end vertebra (e.g. "T6").
        lower_end_vertebra: Label of lower end vertebra (e.g. "L1").
        curve_type: Curve classification: "primary", "secondary", "compensatory".

    Returns:
        Dict with Cobb angle, endvertebrae identification, and curve classification.

    Raises:
        ValidationError: If inputs are invalid or landmarks are missing.
        SlicerConnectionError: If Slicer communication fails.
    """
    volume_node_id = validate_mrml_node_id(volume_node_id)
    pts = _validate_landmarks(landmarks, COBB_ANGLE_LANDMARKS, "cobb_angle")

    valid_curve_types = ("primary", "secondary", "compensatory")
    if curve_type not in valid_curve_types:
        raise ValidationError(
            f"cobb_angle: invalid curve_type '{curve_type}'. "
            f"Must be one of: {', '.join(valid_curve_types)}",
            field="curve_type",
            value=curve_type,
        )

    # Place landmarks in Slicer
    client = get_client()
    place_code = _build_place_landmarks_code(volume_node_id, pts, "CobbAngle_Landmarks")

    try:
        exec_result = client.exec_python(place_code)
        _parse_json_result(exec_result.get("result", ""), "Cobb angle landmark placement")
    except SlicerConnectionError:
        logger.error("Failed to place Cobb angle landmarks in Slicer")
        raise

    # Compute Cobb angle
    cobb_angle = _cobb_angle_2d(
        pts["upper_end_sup_L"],
        pts["upper_end_sup_R"],
        pts["lower_end_inf_L"],
        pts["lower_end_inf_R"],
    )

    # Curve direction: based on apex position relative to midline
    upper_mid_x = (pts["upper_end_sup_L"][0] + pts["upper_end_sup_R"][0]) / 2
    lower_mid_x = (pts["lower_end_inf_L"][0] + pts["lower_end_inf_R"][0]) / 2
    midline_x = (upper_mid_x + lower_mid_x) / 2

    if pts["apex_centroid"][0] < midline_x:
        curve_direction = "left"
    else:
        curve_direction = "right"

    # Severity classification (Lenke)
    if cobb_angle < 10.0:
        severity = "within normal limits"
    elif cobb_angle < 25.0:
        severity = "mild"
    elif cobb_angle < 45.0:
        severity = "moderate"
    elif cobb_angle < 70.0:
        severity = "severe"
    else:
        severity = "very severe"

    result = {
        "success": True,
        "tool": "measure_cobb_angle_xray",
        "cobb_angle_deg": round(cobb_angle, 2),
        "curve_direction": curve_direction,
        "curve_type": curve_type,
        "severity": severity,
        "endvertebrae": {
            "upper": upper_end_vertebra,
            "lower": lower_end_vertebra,
        },
        "apex": {
            "centroid_x": round(pts["apex_centroid"][0], 2),
            "centroid_y": round(pts["apex_centroid"][1], 2),
        },
        "num_landmarks": len(pts),
        "references": [
            "Cobb JR. Am Acad Orthop Surg Instr Course Lect. 1948;5:261-75",
        ],
    }

    logger.info(f"Cobb angle measured: {cobb_angle:.1f}° {curve_direction}, severity={severity}")

    return result
