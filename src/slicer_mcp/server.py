"""MCP Slicer Bridge server entry point."""

import logging
import sys

from mcp.server.fastmcp import FastMCP

# Import tools and resources
from slicer_mcp import diagnostic_tools_ct, instrumentation_tools, resources, spine_tools, tools
from slicer_mcp.circuit_breaker import CircuitOpenError
from slicer_mcp.slicer_client import SlicerConnectionError, SlicerTimeoutError

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


# Register Tools
# ==============


def _handle_tool_error(error: Exception, tool_name: str) -> dict:
    """Handle tool errors and return standardized error response.

    Args:
        error: The caught exception
        tool_name: Name of the tool that failed

    Returns:
        Dict with error information
    """
    if isinstance(error, CircuitOpenError):
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


@mcp.tool()
def capture_screenshot(
    view_type: str, scroll_position: float | None = None, look_from_axis: str | None = None
) -> dict:
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
    """
    try:
        return tools.capture_screenshot(view_type, scroll_position, look_from_axis)
    except Exception as e:
        return _handle_tool_error(e, "capture_screenshot")


@mcp.tool()
def list_scene_nodes() -> dict:
    """List all nodes in the current MRML scene with metadata.

    Returns:
        Dict with nodes list (id, name, type) and total count
    """
    try:
        return tools.list_scene_nodes()
    except Exception as e:
        return _handle_tool_error(e, "list_scene_nodes")


@mcp.tool()
def execute_python(code: str) -> dict:
    """Execute arbitrary Python code in Slicer's Python environment.

    Security Warning: This executes code directly in Slicer's Python interpreter.
    Use only with trusted code in controlled environments.

    Args:
        code: Python code to execute (has access to slicer, vtk, qt modules)

    Returns:
        Dict with success status, result, stdout, and stderr
    """
    try:
        return tools.execute_python(code)
    except Exception as e:
        return _handle_tool_error(e, "execute_python")


@mcp.tool()
def measure_volume(node_id: str, segment_name: str | None = None) -> dict:
    """Calculate the volume of a segmentation node or specific segment in cubic millimeters.

    Args:
        node_id: MRML node ID of segmentation (e.g., "vtkMRMLSegmentationNode1")
        segment_name: Specific segment to measure (if None, measures all segments)

    Returns:
        Dict with node_id, node_name, total_volume_mm3, total_volume_ml, and per-segment breakdown
    """
    try:
        return tools.measure_volume(node_id, segment_name)
    except Exception as e:
        return _handle_tool_error(e, "measure_volume")


@mcp.tool()
def list_sample_data() -> dict:
    """List all available sample datasets from 3D Slicer's SampleData module.

    Dynamically queries Slicer to discover available sample datasets.
    Falls back to a known list if dynamic discovery fails.

    Returns:
        Dict with datasets list (name, category, description),
            total_count, and source (dynamic/fallback)
    """
    try:
        return tools.list_sample_data()
    except Exception as e:
        return _handle_tool_error(e, "list_sample_data")


@mcp.tool()
def load_sample_data(dataset_name: str) -> dict:
    """Load a sample dataset into 3D Slicer for testing and demonstration.

    Use list_sample_data() first to see available datasets.

    Args:
        dataset_name: Name of sample dataset (e.g., "MRHead", "CTChest", "CTACardio")

    Returns:
        Dict with success status, dataset_name, loaded_node_id, loaded_node_name, and message
    """
    try:
        return tools.load_sample_data(dataset_name)
    except Exception as e:
        return _handle_tool_error(e, "load_sample_data")


@mcp.tool()
def set_layout(layout: str, gui_mode: str = "full") -> dict:
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
    """
    try:
        return tools.set_layout(layout, gui_mode)
    except Exception as e:
        return _handle_tool_error(e, "set_layout")


@mcp.tool()
def import_dicom(folder_path: str) -> dict:
    """Import DICOM files from a folder into Slicer's DICOM database.

    Args:
        folder_path: Path to folder containing DICOM files (can be nested)

    Returns:
        Dict with success status, patients_count, studies_count,
            series_count, and new_patients count
    """
    try:
        return tools.import_dicom(folder_path)
    except Exception as e:
        return _handle_tool_error(e, "import_dicom")


@mcp.tool()
def list_dicom_studies() -> dict:
    """List all studies in the DICOM database with patient and study metadata.

    Returns:
        Dict with studies list (patient_id, patient_name, study_uid,
            study_date, modalities, series_count) and total_count
    """
    try:
        return tools.list_dicom_studies()
    except Exception as e:
        return _handle_tool_error(e, "list_dicom_studies")


@mcp.tool()
def list_dicom_series(study_uid: str) -> dict:
    """List all series within a DICOM study.

    Args:
        study_uid: DICOM Study UID (from list_dicom_studies)

    Returns:
        Dict with series list (series_uid, series_number,
            series_description, modality, file_count) and total_count
    """
    try:
        return tools.list_dicom_series(study_uid)
    except Exception as e:
        return _handle_tool_error(e, "list_dicom_series")


@mcp.tool()
def load_dicom_series(series_uid: str) -> dict:
    """Load a DICOM series as a volume into the scene.

    Args:
        series_uid: DICOM Series UID (from list_dicom_series)

    Returns:
        Dict with success status, node_id, node_name, dimensions, spacing, origin, and scalar_range
    """
    try:
        return tools.load_dicom_series(series_uid)
    except Exception as e:
        return _handle_tool_error(e, "load_dicom_series")


@mcp.tool()
def run_brain_extraction(input_node_id: str, method: str = "hd-bet", device: str = "auto") -> dict:
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
    """
    try:
        return tools.run_brain_extraction(input_node_id, method, device)
    except Exception as e:
        return _handle_tool_error(e, "run_brain_extraction")


# CT Diagnostic Protocol Tools
# ============================


@mcp.tool()
def detect_vertebral_fractures_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    region: str = "full",
    classification_system: str = "ao_spine",
) -> dict:
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
    """
    try:
        return diagnostic_tools_ct.detect_vertebral_fractures_ct(
            volume_node_id, segmentation_node_id, region, classification_system
        )
    except Exception as e:
        return _handle_tool_error(e, "detect_vertebral_fractures_ct")


@mcp.tool()
def assess_osteoporosis_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    levels: list[str] | None = None,
    method: str = "trabecular_roi",
) -> dict:
    """Assess bone density for opportunistic osteoporosis screening on CT.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Uses trabecular ROI with 3mm morphological erosion and Pickhardt 2013
    classification. Not equivalent to DXA — opportunistic screening only.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        levels: Vertebral levels to assess (default: ["L1"])
        method: "trabecular_roi", "vertebral_mean", or "both"

    Returns:
        Dict with per-level HU statistics, Pickhardt classification,
            screw pullout risk, and cement augmentation flags
    """
    try:
        return diagnostic_tools_ct.assess_osteoporosis_ct(
            volume_node_id, segmentation_node_id, levels, method
        )
    except Exception as e:
        return _handle_tool_error(e, "assess_osteoporosis_ct")


@mcp.tool()
def detect_metastatic_lesions_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    region: str = "full",
    include_posterior_elements: bool = True,
) -> dict:
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
    """
    try:
        return diagnostic_tools_ct.detect_metastatic_lesions_ct(
            volume_node_id, segmentation_node_id, region, include_posterior_elements
        )
    except Exception as e:
        return _handle_tool_error(e, "detect_metastatic_lesions_ct")


@mcp.tool()
def calculate_sins_score(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    target_levels: list[str] | None = None,
    pain_score: int | None = None,
) -> dict:
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
    """
    try:
        return diagnostic_tools_ct.calculate_sins_score(
            volume_node_id, segmentation_node_id, target_levels, pain_score
        )
    except Exception as e:
        return _handle_tool_error(e, "calculate_sins_score")


@mcp.tool()
def measure_listhesis_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    levels: list[str] | None = None,
) -> dict:
    """Measure spondylolisthesis on static CT.

    LONG OPERATION: Runs TotalSegmentator if no segmentation provided.

    Calculates translation (mm and percentage), Meyerding grade (I-V),
    slip angle, and spondylolysis detection. Static measurement only —
    dynamic instability requires flexion/extension X-ray.

    Args:
        volume_node_id: MRML node ID of the CT volume
        segmentation_node_id: MRML node ID of existing spine segmentation (optional)
        levels: Vertebral levels to measure (default: ["L3", "L4", "L5"])

    Returns:
        Dict with per-level translation, Meyerding grade, slip angle,
            and static measurement disclaimer
    """
    try:
        return diagnostic_tools_ct.measure_listhesis_ct(
            volume_node_id, segmentation_node_id, levels
        )
    except Exception as e:
        return _handle_tool_error(e, "measure_listhesis_ct")


@mcp.tool()
def measure_spinal_canal_ct(
    volume_node_id: str,
    segmentation_node_id: str | None = None,
    levels: list[str] | None = None,
) -> dict:
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
    """
    try:
        return diagnostic_tools_ct.measure_spinal_canal_ct(
            volume_node_id, segmentation_node_id, levels
        )
    except Exception as e:
        return _handle_tool_error(e, "measure_spinal_canal_ct")


# Spine Surgery Planning Tools
# ============================


@mcp.tool()
def plan_cervical_screws(
    technique: str,
    level: str,
    segmentation_node_id: str,
    side: str = "bilateral",
    va_node_id: str | None = None,
    variant: str | None = None,
    screw_diameter_mm: float | None = None,
    screw_length_mm: float | None = None,
) -> dict:
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
    """
    try:
        return instrumentation_tools.plan_cervical_screws(
            technique=technique,
            level=level,
            segmentation_node_id=segmentation_node_id,
            side=side,
            va_node_id=va_node_id,
            variant=variant,
            screw_diameter_mm=screw_diameter_mm,
            screw_length_mm=screw_length_mm,
        )
    except Exception as e:
        return _handle_tool_error(e, "plan_cervical_screws")


@mcp.tool()
def measure_ccj_angles(segmentation_node_id: str, population: str = "adult") -> dict:
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
    """
    try:
        return spine_tools.measure_ccj_angles(segmentation_node_id, population)
    except Exception as e:
        return _handle_tool_error(e, "measure_ccj_angles")


@mcp.tool()
def measure_spine_alignment(segmentation_node_id: str, region: str = "full") -> dict:
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
    """
    try:
        return spine_tools.measure_spine_alignment(segmentation_node_id, region)
    except Exception as e:
        return _handle_tool_error(e, "measure_spine_alignment")


@mcp.tool()
def segment_spine(
    input_node_id: str,
    region: str = "full",
    include_discs: bool = False,
    include_spinal_cord: bool = False,
) -> dict:
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
    """
    try:
        return spine_tools.segment_spine(input_node_id, region, include_discs, include_spinal_cord)
    except Exception as e:
        return _handle_tool_error(e, "segment_spine")


@mcp.tool()
def segment_vertebral_artery(
    input_node_id: str,
    side: str = "both",
    seed_points: list[list[float]] | None = None,
) -> dict:
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
    """
    try:
        return spine_tools.segment_vertebral_artery(input_node_id, side, seed_points)
    except Exception as e:
        return _handle_tool_error(e, "segment_vertebral_artery")


@mcp.tool()
def analyze_bone_quality(
    input_node_id: str,
    segmentation_node_id: str,
    region: str = "lumbar",
) -> dict:
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
    """
    try:
        return spine_tools.analyze_bone_quality(input_node_id, segmentation_node_id, region)
    except Exception as e:
        return _handle_tool_error(e, "analyze_bone_quality")


# Register Resources
# ==================


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


# Main Entry Point
# ================


def main():
    """Run the MCP Slicer Bridge server with stdio transport."""
    logger.info("Starting MCP Slicer Bridge server")
    logger.info(
        "Registered 24 tools: capture_screenshot, list_scene_nodes, "
        "execute_python, measure_volume, list_sample_data, load_sample_data, "
        "set_layout, import_dicom, list_dicom_studies, list_dicom_series, "
        "load_dicom_series, run_brain_extraction, detect_vertebral_fractures_ct, "
        "assess_osteoporosis_ct, detect_metastatic_lesions_ct, calculate_sins_score, "
        "measure_listhesis_ct, measure_spinal_canal_ct, plan_cervical_screws, "
        "measure_ccj_angles, measure_spine_alignment, segment_spine, "
        "segment_vertebral_artery, analyze_bone_quality"
    )
    logger.info("Registered 3 resources: slicer://scene, slicer://volumes, slicer://status")

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
