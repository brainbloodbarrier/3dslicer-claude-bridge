"""Craniocervical junction (CCJ) assessment workflow.

Orchestrates segment_spine, measure_ccj_angles,
segment_vertebral_artery, analyze_bone_quality, and
capture_screenshot into a single workflow tool for
comprehensive CT-based CCJ assessment.

References:
    Joaquim AF et al. World Neurosurg. 2019;127:e633-e644.
    Harris JH Jr et al. Radiology. 2002;223(2):514-520.
    Powers B et al. Neurosurgery. 1979;4(6):542-551.
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
from slicer_mcp.features.spine.tools import (
    analyze_bone_quality,
    measure_ccj_angles,
    segment_spine,
    segment_vertebral_artery,
)

logger = logging.getLogger("slicer-mcp")

__all__ = ["workflow_ccj_protocol"]

VALID_POPULATIONS = {"adult", "child"}


def _validate_population(population: str) -> str:
    """Validate population parameter for CCJ workflow.

    Args:
        population: Population group for thresholds

    Returns:
        Validated population string

    Raises:
        ValidationError: If population is invalid
    """
    if population not in VALID_POPULATIONS:
        raise ValidationError(
            f"Invalid population '{population}'. "
            f"Must be one of: "
            f"{', '.join(sorted(VALID_POPULATIONS))}",
            "population",
            population,
        )
    return population


def workflow_ccj_protocol(
    ct_volume_id: str,
    segmentation_node_id: str | None = None,
    cta_volume_id: str | None = None,
    population: str = "adult",
    include_bone_quality: bool = True,
) -> dict[str, Any]:
    """Run craniocervical junction assessment protocol.

    Orchestrates: segment_spine, measure_ccj_angles,
    segment_vertebral_artery (if CTA provided),
    analyze_bone_quality, capture_screenshot.

    LONG OPERATION: May take 2-10 minutes depending on
    hardware and whether segmentation is pre-computed.

    Args:
        ct_volume_id: MRML node ID of CT volume
        segmentation_node_id: MRML node ID of existing
            segmentation (optional; runs segment_spine
            if not provided)
        cta_volume_id: MRML node ID of CTA volume for VA
            assessment (optional)
        population: "adult" or "child" (affects ADI
            threshold)
        include_bone_quality: Include C1-C2 HU analysis
            (default: True)

    Returns:
        Dict with:
        - segmentation_node_id: MRML node ID of
            segmentation used
        - ccj_angles: Craniometry measurement results
        - vertebral_artery: VA assessment (or None)
        - bone_quality: Bone quality results (or None)
        - screenshots: List of captured screenshot results
        - population: Population group used
        - steps_completed: List of workflow steps that
            completed

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable
            or any analysis step fails
    """
    # ── Step 0: Validate inputs ─────────────────────────
    ct_volume_id = validate_mrml_node_id(ct_volume_id)
    population = _validate_population(population)
    if segmentation_node_id is not None:
        segmentation_node_id = validate_mrml_node_id(
            segmentation_node_id,
        )
    if cta_volume_id is not None:
        cta_volume_id = validate_mrml_node_id(cta_volume_id)

    steps_completed: list[str] = []
    screenshots: list[dict[str, Any]] = []

    # ── Step 1: Segment spine (if needed) ───────────────
    if segmentation_node_id is None:
        logger.info(
            "workflow_ccj_protocol: running segment_spine" " on CT volume",
        )
        seg_result = segment_spine(
            input_node_id=ct_volume_id,
            region="cervical",
            include_discs=False,
            include_spinal_cord=False,
        )
        segmentation_node_id = seg_result["output_segmentation_id"]
        steps_completed.append("segment_spine")
    else:
        logger.info(
            "workflow_ccj_protocol: using existing " "segmentation %s",
            segmentation_node_id,
        )
        steps_completed.append("segment_spine_skipped")

    # At this point segmentation_node_id is guaranteed non-None
    # (either provided by caller or set in step 1)
    assert segmentation_node_id is not None  # noqa: S101

    # ── Step 2: Measure CCJ angles ──────────────────────
    logger.info(
        "workflow_ccj_protocol: measuring CCJ angles",
    )
    ccj_result = measure_ccj_angles(
        segmentation_node_id=segmentation_node_id,
        population=population,
    )
    steps_completed.append("measure_ccj_angles")

    # ── Step 3: Segment vertebral artery (if CTA) ───────
    va_result: dict[str, Any] | None = None
    if cta_volume_id is not None:
        logger.info(
            "workflow_ccj_protocol: segmenting " "vertebral artery from CTA",
        )
        va_result = segment_vertebral_artery(
            input_node_id=cta_volume_id,
        )
        steps_completed.append(
            "segment_vertebral_artery",
        )

    # ── Step 4: Bone quality analysis (if requested) ────
    bone_result: dict[str, Any] | None = None
    if include_bone_quality:
        logger.info(
            "workflow_ccj_protocol: analyzing bone " "quality",
        )
        bone_result = analyze_bone_quality(
            input_node_id=ct_volume_id,
            segmentation_node_id=segmentation_node_id,
            region="cervical",
        )
        steps_completed.append("analyze_bone_quality")

    # ── Step 5: Capture screenshots ─────────────────────
    logger.info(
        "workflow_ccj_protocol: capturing sagittal " "screenshot",
    )
    try:
        screenshot = capture_screenshot(
            view_type="sagittal",
        )
        screenshots.append(screenshot)
        steps_completed.append("capture_screenshot")
    except (
        SlicerConnectionError,
        SlicerTimeoutError,
        CircuitOpenError,
        ValueError,
    ) as e:
        logger.warning(
            "workflow_ccj_protocol: screenshot failed " "(non-fatal): %s",
            e,
        )
        # Screenshot failure is non-fatal for the workflow

    # ── Assemble result ─────────────────────────────────
    return {
        "segmentation_node_id": segmentation_node_id,
        "ccj_angles": ccj_result,
        "vertebral_artery": va_result,
        "bone_quality": bone_result,
        "screenshots": screenshots,
        "population": population,
        "steps_completed": steps_completed,
    }
