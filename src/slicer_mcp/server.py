"""MCP Slicer Bridge server entry point."""

import functools
import logging
import sys
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from slicer_mcp.core import resources

# Import tools and resources from canonical locations
from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import SlicerConnectionError, SlicerTimeoutError
from slicer_mcp.features import base_tools as tools
from slicer_mcp.features import registration as registration_tools
from slicer_mcp.features import rendering as rendering_tools
from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.features.diagnostics import ct as diagnostic_tools_ct
from slicer_mcp.features.diagnostics import mri as diagnostic_tools_mri
from slicer_mcp.features.diagnostics import xray as diagnostic_tools_xray
from slicer_mcp.features.spine import instrumentation as instrumentation_tools
from slicer_mcp.features.spine import tools as spine_tools
from slicer_mcp.features.workflows import modic as workflow_modic

# Configure logging to stderr (stdout reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    stream=sys.stderr,
)
logger = logging.getLogger("slicer-mcp")

# Initialize FastMCP server
mcp = FastMCP("slicer-bridge")

logger.info("Initializing MCP Slicer Bridge server")


# =============================================================================
# Error Handling
# =============================================================================


def _handle_tool_error(error: Exception, tool_name: str) -> dict:
    """Handle tool errors and return standardized error response.

    Args:
        error: The caught exception
        tool_name: Name of the tool that failed

    Returns:
        Dict with error information
    """
    if isinstance(error, ValidationError):
        logger.warning(f"Tool {tool_name}: Validation error - {error.message}")
        return {
            "success": False,
            "error": error.message,
            "error_type": "validation",
            "field": error.field,
            "value": error.value,
        }
    elif isinstance(error, CircuitOpenError):
        logger.warning(f"Tool {tool_name}: Circuit breaker open - {error}")
        return {"success": False, "error": str(error), "error_type": "circuit_open"}
    elif isinstance(error, SlicerTimeoutError):
        logger.error(f"Tool {tool_name}: Timeout - {error.message}")
        return {
            "success": False,
            "error": error.message,
            "error_type": "timeout",
            "details": error.details,
        }
    elif isinstance(error, SlicerConnectionError):
        logger.error(f"Tool {tool_name}: Connection error - {error.message}")
        return {
            "success": False,
            "error": error.message,
            "error_type": "connection",
            "details": error.details,
        }
    else:
        logger.error(f"Tool {tool_name}: Unexpected error - {error}", exc_info=True)
        return {"success": False, "error": str(error), "error_type": "unexpected"}


# =============================================================================
# Tool Registration Helper
# =============================================================================


def _register_tool(
    module: Any,
    fn_name: str,
    doc: str,
) -> Callable[..., dict]:
    """Register a feature function as an MCP tool with error handling.

    Creates a wrapper that:
    1. Preserves the original function's signature (for MCP schema generation)
    2. Delegates to the feature function via late-bound getattr() lookup
       (so unittest.mock.patch on the module attribute works correctly)
    3. Catches all exceptions and routes them through _handle_tool_error

    Args:
        module: The feature module object (as imported in this file, e.g. ``tools``)
        fn_name: Attribute name of the function on *module*
        doc: Docstring for the MCP tool (used by MCP for tool descriptions)

    Returns:
        The registered MCP tool wrapper function
    """
    # Resolve once at import time only to copy the signature for MCP schema.
    tool_fn = getattr(module, fn_name)

    @mcp.tool()
    @functools.wraps(tool_fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            # Late-bound lookup: picks up mocks installed by unittest.mock.patch.
            fn = getattr(module, fn_name)
            return fn(*args, **kwargs)
        except Exception as e:
            return _handle_tool_error(e, fn_name)

    wrapper.__doc__ = doc
    return wrapper


# =============================================================================
# Base Tools
# =============================================================================

capture_screenshot = _register_tool(
    tools, "capture_screenshot",
    """Capture a screenshot from a specific 3D Slicer viewport and return as base64 PNG.

    Args:
        view_type: Viewport type - "axial" (Red slice),
            "sagittal" (Yellow slice), "coronal" (Green slice),
            "3d" (3D view), "full" (complete window)
        scroll_position: Slice position from 0.0 to 1.0 (only for axial/sagittal/coronal views)
        look_from_axis: Camera axis for 3D view - "left", "right",
            "anterior", "posterior", "superior", "inferior"
            (only for 3d view)

    Returns:
        Dict with success status, base64-encoded PNG image, view type, and metadata
    """,
)

list_scene_nodes = _register_tool(
    tools, "list_scene_nodes",
    """List all nodes in the current MRML scene with metadata.

    Returns:
        Dict with nodes list (id, name, type) and total count
    """,
)

execute_python = _register_tool(
    tools, "execute_python",
    """Execute arbitrary Python code in Slicer's Python environment.

    Security Warning: This executes code directly in Slicer's Python interpreter.
    Use only with trusted code in controlled environments.

    Args:
        code: Python code to execute (has access to slicer, vtk, qt modules)

    Returns:
        Dict with success status, result, stdout, and stderr
    """,
)

measure_volume = _register_tool(
    tools, "measure_volume",
    """Calculate the volume of a segmentation node or specific segment in cubic millimeters.

    Args:
        node_id: MRML node ID of segmentation (e.g., "vtkMRMLSegmentationNode1")
        segment_name: Specific segment to measure (if None, measures all segments)

    Returns:
        Dict with node_id, node_name, total_volume_mm3, total_volume_ml, and per-segment breakdown
    """,
)

list_sample_data = _register_tool(
    tools, "list_sample_data",
    """List all available sample datasets from 3D Slicer's SampleData module.

    Dynamically queries Slicer to discover available sample datasets.
    Falls back to a known list if dynamic discovery fails.

    Returns:
        Dict with datasets list (name, category, description),
            total_count, and source (dynamic/fallback)
    """,
)

load_sample_data = _register_tool(
    tools, "load_sample_data",
    """Load a sample dataset into 3D Slicer for testing and demonstration.

    Use list_sample_data() first to see available datasets.

    Args:
        dataset_name: Name of sample dataset (e.g., "MRHead", "CTChest", "CTACardio")

    Returns:
        Dict with success status, dataset_name, loaded_node_id, loaded_node_name, and message
    """,
)

set_layout = _register_tool(
    tools, "set_layout",
    """Set the viewer layout and GUI mode in 3D Slicer.

    Args:
        layout: Layout name - "FourUp" (standard 4-panel),
            "OneUp3D" (single 3D view),
            "OneUpRedSlice" (single axial),
            "Conventional" (traditional radiology),
            "SideBySide" (comparison view)
        gui_mode: GUI display mode - "full" (complete GUI)
            or "viewers" (viewers only, minimal chrome)

    Returns:
        Dict with success status, layout, gui_mode, and message
    """,
)

import_dicom = _register_tool(
    tools, "import_dicom",
    """Import DICOM files from a folder into Slicer's DICOM database.

    Args:
        folder_path: Path to folder containing DICOM files (can be nested)

    Returns:
        Dict with success status, patients_count, studies_count,
            series_count, and new_patients count
    """,
)

list_dicom_studies = _register_tool(
    tools, "list_dicom_studies",
    """List all studies in the DICOM database with patient and study metadata.

    Returns:
        Dict with studies list (patient_id, patient_name, study_uid,
            study_date, modalities, series_count) and total_count
    """,
)

list_dicom_series = _register_tool(
    tools, "list_dicom_series",
    """List all series within a DICOM study.

    Args:
        study_uid: DICOM Study UID (from list_dicom_studies)

    Returns:
        Dict with series list (series_uid, series_number,
            series_description, modality, file_count) and total_count
    """,
)

load_dicom_series = _register_tool(
    tools, "load_dicom_series",
    """Load a DICOM series as a volume into the scene.

    Args:
        series_uid: DICOM Series UID (from list_dicom_series)

    Returns:
        Dict with success status, node_id, node_name, dimensions, spacing, origin, and scalar_range
    """,
)

run_brain_extraction = _register_tool(
    tools, "run_brain_extraction",
    """Extract brain from MRI/CT scan (skull stripping).

    LONG OPERATION: This tool may take 20 seconds to 5 minutes depending on method and hardware.

    Removes skull and non-brain tissue from brain imaging using AI or atlas-based methods.

    Expected durations:
    - hd-bet with GPU: ~20-30 seconds
    - hd-bet with CPU: ~3-5 minutes
    - swiss (atlas-based): ~2-3 minutes

    Args:
        input_node_id: MRML node ID of input brain MRI or CT volume
        method: Extraction method - "hd-bet" (AI, faster with GPU)
            or "swiss" (atlas-based, CPU only)
        device: For hd-bet only - "auto" (detect GPU),
            "cpu" (force CPU), or GPU index ("0", "1", etc.)

    Returns:
        Dict with:
        - output_volume_id: MRML node ID of extracted brain volume
        - output_segmentation_id: MRML node ID of brain mask segmentation
        - brain_volume_ml: Calculated brain volume in milliliters
        - processing_time_seconds: Actual processing time
        - long_operation: Metadata about this being a long operation
    """,
)


# =============================================================================
# X-ray Diagnostic Protocol Tools
# =============================================================================

measure_sagittal_balance_xray = _register_tool(
    diagnostic_tools_xray, "measure_sagittal_balance_xray",
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

measure_coronal_balance_xray = _register_tool(
    diagnostic_tools_xray, "measure_coronal_balance_xray",
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

measure_listhesis_dynamic_xray = _register_tool(
    diagnostic_tools_xray, "measure_listhesis_dynamic_xray",
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

detect_vertebral_fractures_xray = _register_tool(
    diagnostic_tools_xray, "detect_vertebral_fractures_xray",
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

measure_cobb_angle_xray = _register_tool(
    diagnostic_tools_xray, "measure_cobb_angle_xray",
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

classify_disc_degeneration_xray = _register_tool(
    diagnostic_tools_xray, "classify_disc_degeneration_xray",
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


# =============================================================================
# CT Diagnostic Protocol Tools
# =============================================================================

detect_vertebral_fractures_ct = _register_tool(
    diagnostic_tools_ct, "detect_vertebral_fractures_ct",
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

assess_osteoporosis_ct = _register_tool(
    diagnostic_tools_ct, "assess_osteoporosis_ct",
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

detect_metastatic_lesions_ct = _register_tool(
    diagnostic_tools_ct, "detect_metastatic_lesions_ct",
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

calculate_sins_score = _register_tool(
    diagnostic_tools_ct, "calculate_sins_score",
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

measure_listhesis_ct = _register_tool(
    diagnostic_tools_ct, "measure_listhesis_ct",
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

measure_spinal_canal_ct = _register_tool(
    diagnostic_tools_ct, "measure_spinal_canal_ct",
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


# =============================================================================
# Spine Surgery Planning Tools
# =============================================================================

plan_cervical_screws = _register_tool(
    instrumentation_tools, "plan_cervical_screws",
    """Plan cervical screw placement using one of 6 instrumentation techniques.

    Generates patient-specific screw entry points, trajectory vectors, safety
    assessments (VA, canal, nerve root), and 3D visualization in Slicer.

    Techniques:
        - pedicle: Cervical pedicle screws (C2-C7)
        - lateral_mass: Lateral mass screws (C3-C7) with 4 variants
            (roy_camille, magerl, an, anderson)
        - transarticular: C1-C2 Magerl transarticular screws (VA REQUIRED)
        - c1_lateral_mass: C1 lateral mass Harms/Goel screws
        - c2_pars: C2 pars interarticularis screws
        - occipital: Occipital screws with thickness mapping
        - auto: Analyze anatomy and recommend best technique

    Args:
        technique: Instrumentation technique name
        level: Vertebral level (e.g. "C5", "C1C2", "Occiput")
        segmentation_node_id: MRML node ID of vertebral segmentation
        side: "left", "right", or "bilateral"
        va_node_id: MRML node ID of vertebral artery segmentation
            (REQUIRED for transarticular, recommended for C1-C2)
        variant: Lateral mass variant (only for lateral_mass technique)
        screw_diameter_mm: Override default screw diameter (mm)
        screw_length_mm: Override default screw length (mm)

    Returns:
        Dict with technique details, screw parameters, safety assessment,
        visualization node IDs, warnings, and recommendations
    """,
)

measure_ccj_angles = _register_tool(
    spine_tools, "measure_ccj_angles",
    """Measure craniocervical junction (CCJ) angles and distances.

    Computes CXA, ADI, Powers ratio, BDI, BAI, Ranawat, McGregor,
    Chamberlain, and Wackenheim measurements from a spine segmentation.
    Requires C1/C2 vertebral segments (e.g., from TotalSegmentator).

    LONG OPERATION: May take 30-60 seconds depending on segmentation size.

    Args:
        segmentation_node_id: MRML node ID of segmentation containing
            vertebral segments (e.g., "vtkMRMLSegmentationNode1")
        population: Patient population - "adult" (ADI <= 3mm)
            or "child" (ADI <= 5mm)

    Returns:
        Dict with landmarks, measurements (CXA, ADI, Powers, BDI, BAI,
        Ranawat, McGregor, Chamberlain, Wackenheim), reference_ranges,
        statuses, and coordinate_system (RAS)
    """,
)

measure_spine_alignment = _register_tool(
    spine_tools, "measure_spine_alignment",
    """Measure sagittal spinal alignment parameters.

    Computes cervical lordosis, thoracic kyphosis, lumbar lordosis,
    SVA, C2-C7 SVA, T1 slope, pelvic incidence, pelvic tilt,
    sacral slope, PI-LL mismatch, Roussouly type, and SRS-Schwab
    classification from vertebral body segmentations.

    LONG OPERATION: May take 1-3 minutes depending on region and segmentation size.

    Args:
        segmentation_node_id: MRML node ID of segmentation containing
            vertebral segments (e.g., "vtkMRMLSegmentationNode1")
        region: Spine region - "cervical" (C1-C7), "thoracic" (T1-T12),
            "lumbar" (L1-L5), or "full" (all vertebrae)

    Returns:
        Dict with vertebrae_found, vertebrae_data (centroids, endplates),
        measurements (CL, TK, LL, SVA, T1 slope, PI, PT, SS, etc.),
        reference_ranges, statuses, Roussouly type, and Schwab classification
    """,
)

segment_spine = _register_tool(
    spine_tools, "segment_spine",
    """Segment spine structures from a CT volume using TotalSegmentator AI.

    LONG OPERATION: This tool may take 1-5 minutes depending on hardware and region.

    Identifies and labels individual vertebrae from C1-L5, optionally
    including intervertebral discs and spinal cord.

    Expected durations:
    - GPU: ~1-2 minutes
    - CPU: ~3-5 minutes

    Args:
        input_node_id: MRML node ID of input CT volume
        region: Spine region - "cervical" (C1-C7), "thoracic" (T1-T12),
            "lumbar" (L1-L5), or "full" (all vertebrae)
        include_discs: If True, also segment intervertebral discs
        include_spinal_cord: If True, also segment the spinal cord

    Returns:
        Dict with output_segmentation_id, vertebrae_count, vertebrae list,
        discs list, processing_time_seconds, and long_operation metadata
    """,
)

visualize_spine_segmentation = _register_tool(
    spine_tools, "visualize_spine_segmentation",
    """Create clinical visualization of spine segmentation.

    Generates a high-quality sagittal screenshot with distinct colors
    per vertebra (yellow-to-red gradient), outline-mode segmentation
    overlay, bone-window CT (W:2000 L:400), and anatomical labels
    at each vertebra centroid.

    Args:
        segmentation_node_id: MRML node ID of segmentation with vertebral
            segments (e.g., from segment_spine or TotalSegmentator)
        volume_node_id: MRML node ID of background CT volume
        output_path: File path for the output PNG screenshot
        region: Spine region - "cervical", "thoracic", "lumbar", or "full"

    Returns:
        Dict with output_path, file_size_bytes, vertebrae_colored,
        centroids (RAS), view parameters, and fiducial_node_ids
    """,
)

segment_vertebral_artery = _register_tool(
    spine_tools, "segment_vertebral_artery",
    """Segment vertebral arteries from a CTA volume using SlicerVMTK.

    LONG OPERATION: This tool may take 1-5 minutes depending on volume size.

    Pipeline: vesselness filter -> level set segmentation -> centerline extraction.
    Requires the SlicerVMTK extension to be installed in 3D Slicer.

    Args:
        input_node_id: MRML node ID of input CTA volume
        side: Which artery to segment - "left", "right", or "both"
        seed_points: Optional list of [x, y, z] RAS coordinates to guide segmentation.
            Each point should be placed inside the artery lumen.

    Returns:
        Dict with model_node_id, centerline_node_id, diameters along trajectory,
        mean_diameter_mm, processing_time_seconds, and long_operation metadata
    """,
)

analyze_bone_quality = _register_tool(
    spine_tools, "analyze_bone_quality",
    """Analyze bone quality per vertebra using CT HU and BoneTexture metrics.

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
        region: Spine region - "cervical", "thoracic", "lumbar", or "full"

    Returns:
        Dict with per-vertebra metrics (mean_hu, classification, optional BV/TV,
        Tb.Th, Tb.Sp), summary counts, and processing_time_seconds
    """,
)


# =============================================================================
# Workflow Tools
# =============================================================================

workflow_modic_eval = _register_tool(
    workflow_modic, "workflow_modic_eval",
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


# =============================================================================
# MRI Diagnostic Protocol Tools
# =============================================================================

classify_modic_changes = _register_tool(
    diagnostic_tools_mri, "classify_modic_changes",
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

assess_disc_degeneration_mri = _register_tool(
    diagnostic_tools_mri, "assess_disc_degeneration_mri",
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

detect_cord_compression_mri = _register_tool(
    diagnostic_tools_mri, "detect_cord_compression_mri",
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

detect_metastatic_lesions_mri = _register_tool(
    diagnostic_tools_mri, "detect_metastatic_lesions_mri",
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


# =============================================================================
# Registration & Landmark Tools
# =============================================================================

place_landmarks = _register_tool(
    registration_tools, "place_landmarks",
    """Create a markup fiducial node with named control points in 3D Slicer.

    Places a set of RAS-coordinate landmarks as a vtkMRMLMarkupsFiducialNode.
    Useful for defining anatomical landmarks for registration or measurement.

    Args:
        name: Display name for the markup node (non-empty, max 64 chars)
        points: List of [x, y, z] RAS coordinates for each control point
        labels: Optional list of labels for each point (must match points length).
            Labels must match the pattern [A-Za-z0-9_-]+.

    Returns:
        Dict with success status, node_id, node_name, and point_count
    """,
)

get_landmarks = _register_tool(
    registration_tools, "get_landmarks",
    """Retrieve all control points from a markup fiducial node.

    Args:
        node_id: MRML node ID of the markups fiducial node
            (e.g., "vtkMRMLMarkupsFiducialNode1")

    Returns:
        Dict with success status, node_id, node_name, point_count,
            and points list (index, label, position_ras)
    """,
)

register_volumes = _register_tool(
    registration_tools, "register_volumes",
    """Perform intensity-based volume registration using BRAINSFit.

    LONG OPERATION: May take up to 5 minutes depending on volume size and transform type.

    Aligns a moving volume to a fixed volume using intensity-based optimization.
    Supports rigid, affine, and deformable (BSpline) transforms.

    Args:
        fixed_node_id: MRML node ID of the fixed (reference) volume
        moving_node_id: MRML node ID of the moving volume to be registered
        transform_type: Registration transform - "Rigid", "ScaleVersor3D",
            "ScaleSkewVersor3D", "Affine", or "BSpline"
        init_mode: Initialization method - "useMomentsAlign",
            "useCenterOfHeadAlign", "useGeometryAlign", or "Off"
        sampling_percentage: Fraction of voxels to sample (0.0-1.0, default 0.01)
        histogram_match: Whether to match histograms before registration
        create_resampled: Whether to create a resampled output volume

    Returns:
        Dict with success status, transform_node_id, transform_node_name,
            transform_type, and optional resampled_node_id
    """,
)

register_landmarks = _register_tool(
    registration_tools, "register_landmarks",
    """Perform landmark-based registration using paired fiducial points.

    Computes a spatial transform that aligns moving landmarks to fixed landmarks.
    Requires matching control points in both markup nodes.

    Args:
        fixed_landmarks_id: MRML node ID of the fixed (reference) landmarks
        moving_landmarks_id: MRML node ID of the moving landmarks
        transform_type: Registration transform - "Rigid", "Similarity", or "Affine"

    Returns:
        Dict with success status, transform_node_id, transform_node_name,
            and transform_type
    """,
)

apply_transform = _register_tool(
    registration_tools, "apply_transform",
    """Apply a spatial transform to any transformable node.

    Sets the transform on the node. If harden is True, bakes the transform
    into the node's data so the node moves to its transformed position permanently.

    Args:
        node_id: MRML node ID of the node to transform
        transform_node_id: MRML node ID of the transform to apply
        harden: If True, permanently bake the transform into the node data

    Returns:
        Dict with success status, node_id, transform_node_id, and hardened flag
    """,
)


# =============================================================================
# Volume Rendering & 3D Model Export Tools
# =============================================================================

enable_volume_rendering = _register_tool(
    rendering_tools, "enable_volume_rendering",
    """Enable volume rendering visualization on a volume node.

    Creates or updates volume rendering display for the given volume.
    Optionally applies a named preset (e.g., 'CT-Bone', 'MR-Default').

    Args:
        node_id: MRML node ID of the scalar volume to render
        preset: Optional volume rendering preset name
        visible: Whether the volume rendering should be visible (default: True)

    Returns:
        Dict with success status, volume_node_id, display_node_id, preset, visible
    """,
)

set_volume_rendering_property = _register_tool(
    rendering_tools, "set_volume_rendering_property",
    """Adjust volume rendering display properties on a volume node.

    Modifies opacity, window/level, or visibility of an existing volume
    rendering display. Volume rendering must be enabled first.

    Args:
        node_id: MRML node ID of the volume with active volume rendering
        opacity_scale: Multiplier for all opacity values (0.0 to 10.0)
        window: Window width for window/level adjustment
        level: Center level for window/level adjustment
        visible: Whether the volume rendering should be visible

    Returns:
        Dict with success status, volume_node_id, display_node_id, changes_applied
    """,
)

export_model = _register_tool(
    rendering_tools, "export_model",
    """Export a model node to a 3D mesh file.

    Saves the model's polygon data to STL, OBJ, PLY, or VTK format.

    Args:
        node_id: MRML node ID of the model node to export
        output_directory: Directory where the file will be saved
        filename: Output filename without extension
        file_format: Export format - 'STL', 'OBJ', 'PLY', or 'VTK'

    Returns:
        Dict with success status, model_node_id, output_path, format,
            file_size_bytes, point_count, cell_count
    """,
)

segmentation_to_models = _register_tool(
    rendering_tools, "segmentation_to_models",
    """Convert segmentation segments to individual model nodes.

    Creates a vtkMRMLModelNode for each segment by extracting its closed
    surface representation. Useful for exporting segmentations to 3D files.

    Args:
        segmentation_node_id: MRML node ID of the segmentation node
        segment_ids: Optional list of specific segment IDs to convert.
            If None, all visible segments are converted.

    Returns:
        Dict with success status, segmentation_node_id, models list, model_count
    """,
)

capture_3d_view = _register_tool(
    rendering_tools, "capture_3d_view",
    """Capture a screenshot of a 3D view to an image file.

    Renders the current 3D view and saves it as PNG, JPG, BMP, or TIFF.

    Args:
        output_path: Full output file path (supports .png, .jpg, .bmp, .tiff)
        width: Optional capture width in pixels (default: current view size)
        height: Optional capture height in pixels (default: current view size)
        view_index: Index of the 3D view to capture (default: 0)

    Returns:
        Dict with success status, output_path, file_size_bytes, view_index
    """,
)


# =============================================================================
# Register Resources
# =============================================================================


@mcp.resource("slicer://scene")
def get_scene() -> str:
    """Get the current MRML scene structure with all nodes and connections.

    Returns:
        JSON string with scene_id, modified_time, node_count, nodes list, and connections
    """
    return resources.get_scene_resource()


@mcp.resource("slicer://volumes")
def get_volumes() -> str:
    """Get all loaded imaging volumes with metadata including dimensions, spacing, and file paths.

    Returns:
        JSON string with volumes list (id, name, type, dimensions,
            spacing, origin, scalar_range, file_path) and total_count
    """
    return resources.get_volumes_resource()


@mcp.resource("slicer://status")
def get_status() -> str:
    """Get health status and connection information for 3D Slicer.

    Returns:
        JSON string with connected status, slicer_version,
            webserver_url, response_time_ms, scene_loaded,
            python_available, and last_check timestamp
    """
    return resources.get_status_resource()


@mcp.resource("slicer://workflows")
def get_workflows() -> str:
    """List available workflow tools with required inputs and clinical use cases.

    Returns:
        JSON string with workflows list containing name, status,
            description, required_modalities, clinical_indication,
            tools_orchestrated, and estimated_runtime for each workflow
    """
    return resources.get_workflows_resource()


# =============================================================================
# Main Entry Point
# =============================================================================

# Collect tool names for startup log
_TOOL_NAMES = [
    name for name, obj in sorted(locals().items())
    if callable(obj) and not name.startswith("_") and name not in {
        "get_scene", "get_volumes", "get_status", "get_workflows", "main",
    }
]


def main():
    """Run the MCP Slicer Bridge server with stdio transport."""
    logger.info("Starting MCP Slicer Bridge server")
    logger.info("Registered %d tools: %s", len(_TOOL_NAMES), ", ".join(_TOOL_NAMES))
    logger.info(
        "Registered 4 resources: slicer://scene, slicer://volumes, "
        "slicer://status, slicer://workflows"
    )

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
