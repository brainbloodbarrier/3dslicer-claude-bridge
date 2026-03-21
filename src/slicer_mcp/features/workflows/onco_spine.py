"""Oncologic spine assessment workflow.

Orchestrates segment_spine, detect_metastatic_lesions_ct,
calculate_sins_score, measure_listhesis_ct, measure_spinal_canal_ct,
assess_osteoporosis_ct, detect_metastatic_lesions_mri (if MRI),
and capture_screenshot into a single workflow tool for
comprehensive oncologic spine evaluation.

References:
    Fisher CG et al. Spine. 2010;35(22):E1221-9 (SINS).
    Tokuhashi H et al. Spine. 2005;30(19):2186-91.
"""

import logging
from typing import Any

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import (
    SlicerConnectionError,
    SlicerTimeoutError,
)
from slicer_mcp.features.base_tools import (
    ValidationError,
    capture_screenshot,
    validate_mrml_node_id,
)
from slicer_mcp.features.diagnostics.ct import (
    assess_osteoporosis_ct,
    calculate_sins_score,
    detect_metastatic_lesions_ct,
    measure_listhesis_ct,
    measure_spinal_canal_ct,
)
from slicer_mcp.features.diagnostics.mri import (
    detect_metastatic_lesions_mri,
)
from slicer_mcp.features.spine.constants import (
    SINS_PAIN_SCORES,
    SPINE_REGIONS,
)
from slicer_mcp.features.spine.tools import segment_spine

__all__ = ["workflow_onco_spine"]

logger = logging.getLogger("slicer-mcp")


def _validate_region(region: str) -> str:
    """Validate region parameter for onco-spine workflow.

    Args:
        region: Spine region to analyze

    Returns:
        Validated region string

    Raises:
        ValidationError: If region is invalid
    """
    if region not in SPINE_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: "
            f"{', '.join(sorted(SPINE_REGIONS))}",
            "region",
            region,
        )
    return region


def _validate_pain_type(
    pain_type: str | None,
) -> str | None:
    """Validate pain_type parameter for onco-spine workflow.

    Args:
        pain_type: Clinical pain type or None. Valid values:
            "mechanical", "occasional_non_mechanical", "pain_free"

    Returns:
        Validated pain_type string or None

    Raises:
        ValidationError: If pain_type is invalid
    """
    if pain_type is not None and pain_type not in SINS_PAIN_SCORES:
        raise ValidationError(
            f"Invalid pain_type '{pain_type}'. Must be one of: "
            f"{', '.join(sorted(SINS_PAIN_SCORES))}",
            "pain_type",
            pain_type,
        )
    return pain_type


def workflow_onco_spine(
    ct_volume_id: str,
    region: str = "full",
    t1_volume_id: str | None = None,
    t2_volume_id: str | None = None,
    segmentation_node_id: str | None = None,
    pain_type: str | None = None,
) -> dict[str, Any]:
    """Run oncologic spine assessment.

    Orchestrates: segment_spine, detect_metastatic_lesions_ct,
    calculate_sins_score, measure_listhesis_ct,
    measure_spinal_canal_ct, assess_osteoporosis_ct,
    detect_metastatic_lesions_mri (if MRI), capture_screenshot.

    LONG OPERATION: May take 5-15 minutes depending on hardware
    and whether segmentation is pre-computed.

    Args:
        ct_volume_id: MRML node ID of CT volume
        region: Spine region - "cervical", "thoracic", "lumbar",
            or "full" (default: "full")
        t1_volume_id: MRML node ID of T1 MRI volume (optional)
        t2_volume_id: MRML node ID of T2/STIR MRI volume
            (optional)
        segmentation_node_id: MRML node ID of existing
            segmentation (optional; runs segment_spine if not
            provided)
        pain_type: "mechanical", "occasional_non_mechanical",
            or "pain_free" for SINS scoring (optional)

    Returns:
        Dict with:
        - segmentation_node_id: MRML node ID of segmentation
        - metastatic_lesions_ct: CT lesion detection results
        - sins_scores: Per-level SINS breakdown
        - listhesis: Spondylolisthesis measurements
        - canal_stenosis: Spinal canal morphometry
        - bone_quality: Osteoporosis screening results
        - metastatic_lesions_mri: MRI lesion results (or None)
        - screenshots: List of captured screenshots
        - region: Region analyzed
        - pain_type: Pain type used for SINS
        - steps_completed: Workflow steps that completed

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or
            any analysis step fails
    """
    # -- Step 0: Validate inputs ---------------------------------
    ct_volume_id = validate_mrml_node_id(ct_volume_id)
    region = _validate_region(region)
    pain_type = _validate_pain_type(pain_type)
    if t1_volume_id is not None:
        t1_volume_id = validate_mrml_node_id(t1_volume_id)
    if t2_volume_id is not None:
        t2_volume_id = validate_mrml_node_id(t2_volume_id)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)

    steps_completed: list[str] = []
    screenshots: list[dict[str, Any]] = []

    # -- Step 1: Segment spine (if needed) -----------------------
    if segmentation_node_id is None:
        logger.info("workflow_onco_spine: running segment_spine")
        seg_result = segment_spine(
            input_node_id=ct_volume_id,
            region=region,
            include_discs=True,
            include_spinal_cord=False,
        )
        segmentation_node_id = seg_result["output_segmentation_id"]
        steps_completed.append("segment_spine")
    else:
        logger.info(
            "workflow_onco_spine: using existing " "segmentation %s",
            segmentation_node_id,
        )
        steps_completed.append("segment_spine_skipped")

    # At this point segmentation_node_id is guaranteed non-None
    if segmentation_node_id is None:
        raise RuntimeError(
            "segmentation_node_id should have been set by "
            "segment_spine or provided by caller"
        )

    # -- Step 2: Detect metastatic lesions on CT -----------------
    logger.info("workflow_onco_spine: detecting metastatic " "lesions on CT")
    metastatic_ct = detect_metastatic_lesions_ct(
        volume_node_id=ct_volume_id,
        segmentation_node_id=segmentation_node_id,
        region=region,
    )
    steps_completed.append("detect_metastatic_lesions_ct")

    # -- Step 3: Calculate SINS score ----------------------------
    logger.info("workflow_onco_spine: calculating SINS score")
    sins_kwargs: dict[str, Any] = {
        "volume_node_id": ct_volume_id,
        "segmentation_node_id": segmentation_node_id,
    }
    if pain_type is not None:
        sins_kwargs["pain_score"] = SINS_PAIN_SCORES[pain_type]
    sins_result = calculate_sins_score(**sins_kwargs)
    steps_completed.append("calculate_sins_score")

    # -- Step 4: Measure listhesis -------------------------------
    logger.info("workflow_onco_spine: measuring listhesis")
    listhesis_result = measure_listhesis_ct(
        volume_node_id=ct_volume_id,
        segmentation_node_id=segmentation_node_id,
    )
    steps_completed.append("measure_listhesis_ct")

    # -- Step 5: Measure spinal canal ----------------------------
    logger.info("workflow_onco_spine: measuring spinal canal")
    canal_result = measure_spinal_canal_ct(
        volume_node_id=ct_volume_id,
        segmentation_node_id=segmentation_node_id,
    )
    steps_completed.append("measure_spinal_canal_ct")

    # -- Step 6: Assess osteoporosis -----------------------------
    logger.info("workflow_onco_spine: assessing osteoporosis")
    bone_result = assess_osteoporosis_ct(
        volume_node_id=ct_volume_id,
        segmentation_node_id=segmentation_node_id,
    )
    steps_completed.append("assess_osteoporosis_ct")

    # -- Step 7: MRI lesion detection (optional) -----------------
    mri_result: dict[str, Any] | None = None
    if t1_volume_id is not None and t2_volume_id is not None:
        logger.info("workflow_onco_spine: detecting metastatic " "lesions on MRI")
        mri_result = detect_metastatic_lesions_mri(
            t1_node_id=t1_volume_id,
            t2_stir_node_id=t2_volume_id,
            region=region,
            segmentation_node_id=segmentation_node_id,
        )
        steps_completed.append("detect_metastatic_lesions_mri")

    # -- Step 8: Capture screenshot (non-fatal) ------------------
    logger.info("workflow_onco_spine: capturing sagittal screenshot")
    try:
        screenshot = capture_screenshot(view_type="sagittal")
        screenshots.append(screenshot)
        steps_completed.append("capture_screenshot")
    except (
        SlicerConnectionError,
        SlicerTimeoutError,
        CircuitOpenError,
        ValueError,
    ) as e:
        logger.warning(
            "workflow_onco_spine: screenshot failed " "(non-fatal): %s",
            e,
        )

    # -- Assemble result -----------------------------------------
    return {
        "segmentation_node_id": segmentation_node_id,
        "metastatic_lesions_ct": metastatic_ct,
        "sins_scores": sins_result,
        "listhesis": listhesis_result,
        "canal_stenosis": canal_result,
        "bone_quality": bone_result,
        "metastatic_lesions_mri": mri_result,
        "screenshots": screenshots,
        "region": region,
        "pain_type": pain_type,
        "steps_completed": steps_completed,
    }
