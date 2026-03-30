"""Workflow tool registrations — orchestrated clinical protocols."""

from typing import Any

from slicer_mcp._registry._common import register_tool
from slicer_mcp.features.workflows import ccj as workflow_ccj
from slicer_mcp.features.workflows import modic as workflow_modic
from slicer_mcp.features.workflows import onco_spine as workflow_onco


def register_workflow_tools(mcp: Any) -> dict[str, Any]:
    """Register orchestrated clinical workflow tools.

    Returns:
        Dict mapping tool name → wrapper function.
    """
    wrappers: dict[str, Any] = {}

    def _reg(module: Any, fn_name: str, doc: str) -> None:
        wrappers[fn_name] = register_tool(mcp, module, fn_name, doc)

    _reg(
        workflow_modic,
        "workflow_modic_eval",
        """Run Modic endplate and disc degeneration assessment from MRI.

    LONG OPERATION: May take 2-10 minutes depending on hardware and whether
    segmentation is pre-computed.

    Orchestrates: segment_spine, classify_modic_changes, assess_disc_degeneration_mri,
    detect_cord_compression_mri (if cervical/thoracic), capture_screenshot.

    Requires both T1 and T2 weighted MRI sequences loaded in the scene.

    Args:
        t1_volume_id: MRML node ID of T1-weighted MRI volume
        t2_volume_id: MRML node ID of T2-weighted MRI volume
        region: Spine region - "cervical", "thoracic", or "lumbar"
        segmentation_node_id: MRML node ID of existing segmentation (optional;
            runs segment_spine on T2 volume if not provided)
        include_cord_screening: If True, run cord compression detection for
            cervical/thoracic regions (default: True)

    Returns:
        Dict with segmentation_node_id, modic_changes (per level/endplate),
        pfirrmann_grades, cord_compression (if screened), screenshots,
        region, and steps_completed
    """,
    )

    _reg(
        workflow_ccj,
        "workflow_ccj_protocol",
        """Run craniocervical junction assessment protocol.

    LONG OPERATION: May take 2-10 minutes depending on hardware and whether
    segmentation is pre-computed.

    Orchestrates: segment_spine, measure_ccj_angles, segment_vertebral_artery
    (if CTA provided), analyze_bone_quality, capture_screenshot.

    Args:
        ct_volume_id: MRML node ID of CT volume
        segmentation_node_id: MRML node ID of existing segmentation
            (optional; runs segment_spine if not provided)
        cta_volume_id: MRML node ID of CTA volume for VA assessment
            (optional)
        population: "adult" or "child" (affects ADI threshold)
        include_bone_quality: Include C1-C2 HU analysis (default: True)

    Returns:
        Dict with ccj_angles (craniometry measurements), vertebral_artery
        assessment, bone_quality, screenshots, population, and
        steps_completed
    """,
    )

    _reg(
        workflow_onco,
        "workflow_onco_spine",
        """Run oncologic spine assessment: lesion detection, SINS, stability.

    LONG OPERATION: May take 5-15 minutes depending on hardware and whether
    segmentation is pre-computed.

    Orchestrates: segment_spine, detect_metastatic_lesions_ct,
    calculate_sins_score, measure_listhesis_ct, measure_spinal_canal_ct,
    assess_osteoporosis_ct, detect_metastatic_lesions_mri (if MRI),
    capture_screenshot.

    Args:
        ct_volume_id: MRML node ID of CT volume
        region: Spine region - "cervical", "thoracic", "lumbar", or
            "full" (default: "full")
        t1_volume_id: MRML node ID of T1 MRI volume (optional, for
            enhanced lesion detection)
        t2_volume_id: MRML node ID of T2/STIR MRI volume (optional)
        segmentation_node_id: MRML node ID of existing segmentation
            (optional; runs segment_spine if not provided)
        pain_type: "mechanical" or "non_mechanical" for SINS scoring
            (optional)

    Returns:
        Dict with metastatic_lesions_ct, sins_scores, listhesis,
        canal_stenosis, bone_quality, metastatic_lesions_mri (if MRI),
        screenshots, region, pain_type, and steps_completed
    """,
    )

    return wrappers
