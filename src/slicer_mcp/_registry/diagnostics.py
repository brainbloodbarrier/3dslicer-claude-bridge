"""Diagnostic tool registrations — CT, MRI, and X-ray protocols."""

from typing import Any

from slicer_mcp._registry._common import register_tool
from slicer_mcp.features.diagnostics import ct as diagnostic_tools_ct
from slicer_mcp.features.diagnostics import mri as diagnostic_tools_mri
from slicer_mcp.features.diagnostics import xray as diagnostic_tools_xray


def register_diagnostic_tools(mcp: Any) -> dict[str, Any]:
    """Register all diagnostic protocol tools (CT + MRI + X-ray).

    Returns:
        Dict mapping tool name → wrapper function.
    """
    wrappers: dict[str, Any] = {}

    def _reg(module: Any, fn_name: str, doc: str) -> None:
        wrappers[fn_name] = register_tool(mcp, module, fn_name, doc)

    # ── X-ray ────────────────────────────────────────────────────────────

    _reg(
        diagnostic_tools_xray,
        "measure_sagittal_balance_xray",
        """Measure sagittal spinal balance from lateral standing X-ray.

    Computes SVA, C2-C7 SVA, T1 slope, TPA, cervical lordosis, thoracic
    kyphosis, lumbar lordosis (Cobb), pelvic parameters (PI, PT, SS),
    PI-LL mismatch, SRS-Schwab classification, and Roussouly type.

    Args:
        volume_node_id: MRML node ID of the lateral X-ray volume
        landmarks: Dict mapping landmark names to [x, y] coordinates.
            Required landmarks (20): C2_centroid, C7_centroid, C2_sup_ant,
            C2_sup_post, C7_inf_ant, C7_inf_post, T1_sup_ant, T1_sup_post,
            T4_sup_ant, T4_sup_post, T12_inf_ant, T12_inf_post, L1_sup_ant,
            L1_sup_post, S1_sup_ant, S1_sup_post, S1_endplate_mid,
            femoral_head_center_L, femoral_head_center_R, S1_post_sup
        magnification_factor: X-ray magnification correction (default 1.0)

    Returns:
        Dict with sagittal balance parameters, classifications, and metadata
    """,
    )

    _reg(
        diagnostic_tools_xray,
        "measure_coronal_balance_xray",
        """Measure coronal spinal balance from AP standing X-ray.

    Computes C7 plumb line to CSVL offset, trunk shift, pelvic obliquity,
    shoulder balance, and coronal Cobb angle.

    Args:
        volume_node_id: MRML node ID of the AP X-ray volume
        landmarks: Dict mapping landmark names to [x, y] coordinates.
            Required landmarks (11): C7_centroid, sacrum_center, T1_centroid,
            shoulder_L, shoulder_R, iliac_crest_L, iliac_crest_R,
            upper_end_vertebra_L, upper_end_vertebra_R,
            lower_end_vertebra_L, lower_end_vertebra_R
        magnification_factor: X-ray magnification correction (default 1.0)

    Returns:
        Dict with coronal balance parameters and metadata
    """,
    )

    _reg(
        diagnostic_tools_xray,
        "measure_listhesis_dynamic_xray",
        """Measure dynamic listhesis from neutral, flexion, and extension X-rays.

    Processes 3 lateral X-rays simultaneously. Computes translation and angular
    motion per position per level, applies White & Panjabi instability criteria,
    and performs Meyerding grading at worst position.

    Args:
        volume_node_ids: Dict mapping position to MRML node ID.
            Required keys: "neutral", "flexion", "extension"
        landmarks_per_position: Nested dict: position -> level -> landmark -> [x, y].
            Each level requires 8 landmarks per position
        levels: List of spinal levels to assess (e.g., ["L4-L5", "L5-S1"])
        region: Spine region for instability thresholds ("cervical" or "lumbar")
        magnification_factor: X-ray magnification correction (default 1.0)

    Returns:
        Dict with per-level measurements, instability assessment, and Meyerding grading
    """,
    )

    _reg(
        diagnostic_tools_xray,
        "detect_vertebral_fractures_xray",
        """Detect vertebral fractures using Genant semi-quantitative method on lateral X-ray.

    6 points per vertebral body define anterior, middle, and posterior heights.
    Height reduction relative to expected (adjacent) height determines fracture grade.

    Args:
        volume_node_id: MRML node ID of the lateral X-ray volume
        landmarks_per_vertebra: Dict: vertebra_label -> landmark_name -> [x, y].
            Each vertebra requires 6 landmarks (ant_sup, ant_inf, mid_sup,
            mid_inf, post_sup, post_inf)
        magnification_factor: X-ray magnification correction (default 1.0)

    Returns:
        Dict with per-vertebra fracture assessment, Genant grades, and summary
    """,
    )

    _reg(
        diagnostic_tools_xray,
        "measure_cobb_angle_xray",
        """Measure Cobb angle for scoliosis from AP standing X-ray.

    Computes angle between superior endplate of upper end vertebra and inferior
    endplate of lower end vertebra. Identifies curve direction and severity.

    Args:
        volume_node_id: MRML node ID of the AP X-ray volume
        landmarks: Dict mapping landmark names to [x, y] coordinates.
            Required landmarks (5): upper_end_sup_L, upper_end_sup_R,
            lower_end_inf_L, lower_end_inf_R, apex_centroid
        upper_end_vertebra: Label of upper end vertebra (e.g. "T6")
        lower_end_vertebra: Label of lower end vertebra (e.g. "L1")
        curve_type: "primary", "secondary", or "compensatory"

    Returns:
        Dict with Cobb angle, curve direction, severity, and metadata
    """,
    )

    _reg(
        diagnostic_tools_xray,
        "classify_disc_degeneration_xray",
        """Classify disc degeneration from lateral X-ray using height and osteophyte analysis.

    Adapted grading combining disc height loss with osteophyte evaluation.
    Each disc level requires 8 landmarks: disc margins (anterior/middle/posterior
    sup and inf) plus osteophyte tips (anterior and posterior).

    Args:
        volume_node_id: MRML node ID of the lateral X-ray volume
        landmarks_per_disc: Dict: disc_level -> landmark_name -> [x, y].
            Each disc needs 8 landmarks.
        reference_disc_height_mm: Known normal disc height in mm (optional)
        magnification_factor: X-ray magnification correction factor (default 1.0)

    Returns:
        Dict with per-disc grades (1-5), height measurements, osteophyte
            assessment, and summary
    """,
    )

    # ── CT ────────────────────────────────────────────────────────────────

    _reg(
        diagnostic_tools_ct,
        "detect_vertebral_fractures_ct",
        """Detect vertebral fractures in CT with Genant, AO Spine, and Denis classification.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Calculates vertebral body heights (Ha, Hm, Hp), Genant grading (0-3),
    AO Spine classification (Type A/B/C), Denis 3-column analysis, posterior
    wall retropulsion, and canal compromise percentage.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        region: Spine region - "full", "cervical", "thoracic", "lumbar"
        classification_system: "ao_spine", "genant", "denis", or "all"

    Returns:
        Dict with per-vertebra fracture analysis including heights, ratios,
            classification grades, canal compromise, and summary
    """,
    )

    _reg(
        diagnostic_tools_ct,
        "assess_osteoporosis_ct",
        """Assess bone density for opportunistic osteoporosis screening on CT.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Uses trabecular ROI with 3mm morphological erosion and Pickhardt 2013
    classification. Not equivalent to DXA -- opportunistic screening only.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        levels: Vertebral levels to assess (default: ["L1"])
        method: "trabecular_roi", "vertebral_mean", or "both"

    Returns:
        Dict with per-level HU statistics, Pickhardt classification,
            screw pullout risk, and cement augmentation flags
    """,
    )

    _reg(
        diagnostic_tools_ct,
        "detect_metastatic_lesions_ct",
        """Detect metastatic lesions (lytic/blastic/mixed) in vertebral bodies on CT.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Uses adaptive HU thresholds relative to adjacent-level baseline with
    3D connected component clustering. Reports volume, body involvement
    percentage, and posterior element status.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        region: Spine region - "full", "cervical", "thoracic", "lumbar"
        include_posterior_elements: Whether to analyze posterior elements

    Returns:
        Dict with per-vertebra lesion analysis including type, volume,
            body involvement, and canal compromise
    """,
    )

    _reg(
        diagnostic_tools_ct,
        "calculate_sins_score",
        """Calculate SINS (Spinal Instability Neoplastic Score) from CT.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Automates 4/6 SINS components from imaging: location, lesion type,
    alignment, collapse, posterolateral involvement. Pain is clinical input.
    When pain_score is not provided, reports possible score range.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        target_levels: Vertebral levels with known lesions (default: full spine scan)
        pain_score: Clinical pain score 0=pain-free, 1=occasional, 3=mechanical (optional)

    Returns:
        Dict with per-level SINS component breakdown, total score,
            classification (stable/indeterminate/unstable), and score range
    """,
    )

    _reg(
        diagnostic_tools_ct,
        "measure_listhesis_ct",
        """Measure spondylolisthesis on static CT.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Calculates translation (mm and percentage), Meyerding grade (I-V),
    slip angle, and spondylolysis detection. Static measurement only --
    dynamic instability requires flexion/extension X-ray.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        levels: Vertebral levels to measure (default: ["L3", "L4", "L5"])

    Returns:
        Dict with per-level translation, Meyerding grade, slip angle,
            and static measurement disclaimer
    """,
    )

    _reg(
        diagnostic_tools_ct,
        "measure_spinal_canal_ct",
        """Measure spinal canal morphometry on CT.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Calculates AP and transverse diameters, cross-section area,
    Torg-Pavlov ratio (canal/body AP), and stenosis grading per level.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        levels: Vertebral levels to measure (default: ["C3", "C4", "C5", "C6", "C7"])

    Returns:
        Dict with per-level canal AP/transverse diameters, area,
            Torg-Pavlov ratio, and stenosis grade
    """,
    )

    # ── MRI ───────────────────────────────────────────────────────────────

    _reg(
        diagnostic_tools_mri,
        "classify_modic_changes",
        """Classify Modic endplate changes using T1 and T2 MRI sequences.

    LONG OPERATION: Requires registration check + TotalSegmentator segmentation + analysis.

    Analyzes vertebral endplate signal patterns to classify:
    - Type 0: Normal
    - Type I: Edema/inflammation (T1 low, T2 high)
    - Type II: Fatty degeneration (T1 high, T2 iso/high)
    - Type III: Sclerosis (T1 low, T2 low)

    Uses ratio normalization against reference vertebral body (MRI signals are not absolute).

    Args:
        t1_node_id: MRML node ID of T1-weighted volume
        t2_node_id: MRML node ID of T2-weighted volume
        region: Spine region - "cervical", "thoracic", or "lumbar"
        segmentation_node_id: MRML node ID of existing segmentation (optional;
            runs TotalSegmentator if not provided)

    Returns:
        Dict with per-level Modic type, signal ratios, mixed patterns, and summary counts
    """,
    )

    _reg(
        diagnostic_tools_mri,
        "assess_disc_degeneration_mri",
        """Assess intervertebral disc degeneration using Pfirrmann grading on T2 MRI.

    LONG OPERATION: Requires TotalSegmentator segmentation + per-disc analysis.

    Evaluates T2 signal intensity (normalized to CSF), homogeneity (CV),
    disc height, and nucleus-annulus distinction to assign Pfirrmann grades I-V.

    Args:
        t2_node_id: MRML node ID of T2-weighted sagittal volume
        region: Spine region - "cervical", "thoracic", or "lumbar"
        segmentation_node_id: MRML node ID of existing segmentation (optional;
            runs TotalSegmentator if not provided)

    Returns:
        Dict with per-disc Pfirrmann grade, signal ratio to CSF,
            homogeneity, height loss, and grade summary
    """,
    )

    _reg(
        diagnostic_tools_mri,
        "detect_cord_compression_mri",
        """Detect spinal cord compression on MRI with optional myelopathy assessment.

    LONG OPERATION: Requires TotalSegmentator segmentation + per-level analysis.

    Measures cord AP/transverse diameters, compression ratio, cross-section area,
    MSCC ratio, and stenosis grading. Detects T2 hyperintensity (myelopathy sign).
    If T1 is provided, assesses reversibility (T1 normal=reversible, T1 hypo=irreversible).

    Args:
        t2_node_id: MRML node ID of T2-weighted volume
        t1_node_id: MRML node ID of T1-weighted volume (optional, for reversibility)
        region: Spine region - "cervical", "thoracic", or "lumbar"
        segmentation_node_id: MRML node ID of existing segmentation (optional;
            runs TotalSegmentator if not provided)

    Returns:
        Dict with per-level compression metrics, stenosis grades,
            myelopathy status, and MSCC assessment
    """,
    )

    _reg(
        diagnostic_tools_mri,
        "detect_metastatic_lesions_mri",
        """Detect metastatic lesions in the spine using T1 and T2/STIR MRI.

    LONG OPERATION: Requires registration + TotalSegmentator segmentation + per-vertebra analysis.

    Classifies lesion signal patterns:
    - Lytic: T1 low, T2/STIR high
    - Blastic: T1 low, T2/STIR low
    - Mixed: T1 low, T2/STIR intermediate

    Also evaluates posterior element involvement and benign vs malignant
    fracture differentiation features.

    Args:
        t1_node_id: MRML node ID of T1-weighted volume
        t2_stir_node_id: MRML node ID of T2/STIR-weighted volume
        region: Spine region - "cervical", "thoracic", "lumbar", or "full"
        segmentation_node_id: MRML node ID of existing segmentation (optional;
            runs TotalSegmentator if not provided)

    Returns:
        Dict with per-vertebra signal analysis, suspicious lesions,
            lesion type summary, and posterior element involvement
    """,
    )

    return wrappers
