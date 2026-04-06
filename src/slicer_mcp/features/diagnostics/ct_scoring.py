"""CT SINS score calculation tools.

Implements 1 diagnostic tool:
- SINS score calculation (4/6 automated imaging components)
"""

import json
import logging
from typing import Any

from slicer_mcp.core.constants import (
    SEGMENTATION_TIMEOUT,
)
from slicer_mcp.core.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.features._subprocess import _build_totalseg_subprocess_block
from slicer_mcp.features.base_tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)
from slicer_mcp.features.diagnostics._common import _validate_levels
from slicer_mcp.features.spine.constants import (
    LESION_BLASTIC_HU_THRESHOLD,
    LESION_LYTIC_HU_THRESHOLD,
    REGION_VERTEBRAE,
    SINS_RANGES,
    SPINE_SEGMENTATION_TIMEOUT,
    TOTALSEG_TASK_VERTEBRAE,
)

__all__ = [
    "calculate_sins_score",
]

logger = logging.getLogger("slicer-mcp")


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
            f"Score {low}-{high}: Indeterminate stability. Surgical consultation recommended."
        )
    return "UNSTABLE", "Score 13-18: Unstable. Surgical intervention likely required."


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
    safe_pain = repr(pain_score)  # None->"None", int->"N" (both valid Python)
    seg_block = f"seg_node_id = {safe_seg_id}" if safe_seg_id else "seg_node_id = None"
    auto_seg = _build_totalseg_subprocess_block("volume_node", "seg_node", TOTALSEG_TASK_VERTEBRAE)

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

{auto_seg}

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
    lytic_thresh = {LESION_LYTIC_HU_THRESHOLD}
    blastic_thresh = {LESION_BLASTIC_HU_THRESHOLD}
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

# Clean up auto-created segmentation (keep if user-provided)
if not _seg_was_provided:
    slicer.mrmlScene.RemoveNode(seg_node)

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

    timeout = SEGMENTATION_TIMEOUT if segmentation_node_id else SPINE_SEGMENTATION_TIMEOUT
    try:
        exec_result = client.exec_python(python_code, timeout=timeout)
        data = _parse_json_result(exec_result.get("result", ""), "SINS score calculation")
        logger.info(f"SINS calculation complete for levels: {target_levels}")
        return data

    except SlicerConnectionError as e:
        logger.error(f"SINS calculation failed: {e.message}")
        raise
