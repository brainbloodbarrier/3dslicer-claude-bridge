"""Modic endplate and disc degeneration workflow.

Orchestrates segment_spine, classify_modic_changes, assess_disc_degeneration_mri,
detect_cord_compression_mri, and capture_screenshot into a single workflow tool
for comprehensive MRI-based spine assessment.

References:
    Modic MT et al. Radiology. 1988;166(1):193-199.
    Pfirrmann CW et al. Spine. 2001;26(17):1873-8.
    Fehlings MG et al. Spine. 2013;38(22 Suppl 1):S9-S18.
"""

import logging
from typing import Any

from slicer_mcp.core.slicer_client import SlicerConnectionError
from slicer_mcp.features.base_tools import (
    ValidationError,
    capture_screenshot,
    validate_mrml_node_id,
)
from slicer_mcp.features.diagnostics.mri import (
    VALID_MRI_REGIONS,
    assess_disc_degeneration_mri,
    classify_modic_changes,
    detect_cord_compression_mri,
)
from slicer_mcp.features.spine.tools import segment_spine

logger = logging.getLogger("slicer-mcp")

# Regions where cord compression screening is clinically relevant
CORD_SCREENING_REGIONS = frozenset(["cervical", "thoracic"])


def _validate_region(region: str) -> str:
    """Validate region parameter for Modic workflow.

    Args:
        region: Spine region to analyze

    Returns:
        Validated region string

    Raises:
        ValidationError: If region is invalid
    """
    if region not in VALID_MRI_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(VALID_MRI_REGIONS))}",
            "region",
            region,
        )
    return region


def workflow_modic_eval(
    t1_volume_id: str,
    t2_volume_id: str,
    region: str = "lumbar",
    segmentation_node_id: str | None = None,
    include_cord_screening: bool = True,
) -> dict[str, Any]:
    """Run Modic endplate and disc degeneration assessment from MRI.

    Orchestrates: segment_spine, classify_modic_changes, assess_disc_degeneration_mri,
    detect_cord_compression_mri (if cervical/thoracic), capture_screenshot.

    LONG OPERATION: May take 2-10 minutes depending on hardware and whether
    segmentation is pre-computed.

    Args:
        t1_volume_id: MRML node ID of T1-weighted MRI volume
        t2_volume_id: MRML node ID of T2-weighted MRI volume
        region: Spine region - "cervical", "thoracic", or "lumbar"
        segmentation_node_id: MRML node ID of existing segmentation (optional;
            runs segment_spine on T2 volume if not provided)
        include_cord_screening: If True, run cord compression detection for
            cervical/thoracic regions (default: True)

    Returns:
        Dict with:
        - segmentation_node_id: MRML node ID of segmentation used
        - modic_changes: Per-level Modic classification results
        - pfirrmann_grades: Per-disc Pfirrmann grading results
        - cord_compression: Cord compression results (if screened, else None)
        - screenshots: List of captured screenshot results
        - region: Region analyzed
        - steps_completed: List of workflow steps that completed

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or any analysis step fails
    """
    # ── Step 0: Validate inputs ──────────────────────────────────────────
    t1_volume_id = validate_mrml_node_id(t1_volume_id)
    t2_volume_id = validate_mrml_node_id(t2_volume_id)
    region = _validate_region(region)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(segmentation_node_id)

    steps_completed: list[str] = []
    screenshots: list[dict[str, Any]] = []

    # ── Step 1: Segment spine (if needed) ────────────────────────────────
    if segmentation_node_id is None:
        logger.info("workflow_modic_eval: running segment_spine on T2 volume")
        seg_result = segment_spine(
            input_node_id=t2_volume_id,
            region=region,
            include_discs=True,
            include_spinal_cord=True,
        )
        segmentation_node_id = seg_result["output_segmentation_id"]
        steps_completed.append("segment_spine")
    else:
        logger.info("workflow_modic_eval: using existing segmentation %s", segmentation_node_id)
        steps_completed.append("segment_spine_skipped")

    # ── Step 2: Classify Modic changes ───────────────────────────────────
    logger.info("workflow_modic_eval: classifying Modic changes")
    modic_result = classify_modic_changes(
        t1_node_id=t1_volume_id,
        t2_node_id=t2_volume_id,
        region=region,
        segmentation_node_id=segmentation_node_id,
    )
    steps_completed.append("classify_modic_changes")

    # ── Step 3: Assess disc degeneration (Pfirrmann) ─────────────────────
    logger.info("workflow_modic_eval: assessing disc degeneration")
    pfirrmann_result = assess_disc_degeneration_mri(
        t2_node_id=t2_volume_id,
        region=region,
        segmentation_node_id=segmentation_node_id,
    )
    steps_completed.append("assess_disc_degeneration_mri")

    # ── Step 4: Cord compression screening (cervical/thoracic only) ──────
    cord_result: dict[str, Any] | None = None
    if include_cord_screening and region in CORD_SCREENING_REGIONS:
        logger.info("workflow_modic_eval: screening for cord compression (%s)", region)
        cord_result = detect_cord_compression_mri(
            t2_node_id=t2_volume_id,
            t1_node_id=t1_volume_id,
            region=region,
            segmentation_node_id=segmentation_node_id,
        )
        steps_completed.append("detect_cord_compression_mri")

    # ── Step 5: Capture screenshots ──────────────────────────────────────
    logger.info("workflow_modic_eval: capturing sagittal screenshot")
    try:
        screenshot = capture_screenshot(view_type="sagittal")
        screenshots.append(screenshot)
        steps_completed.append("capture_screenshot")
    except (SlicerConnectionError, ValueError) as e:
        logger.warning("workflow_modic_eval: screenshot failed: %s", e)
        # Screenshot failure is non-fatal for the workflow

    # ── Assemble result ──────────────────────────────────────────────────
    return {
        "segmentation_node_id": segmentation_node_id,
        "modic_changes": modic_result,
        "pfirrmann_grades": pfirrmann_result,
        "cord_compression": cord_result,
        "screenshots": screenshots,
        "region": region,
        "steps_completed": steps_completed,
    }
