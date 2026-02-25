"""MCP Slicer Bridge server entry point."""

import logging
import sys

from mcp.server.fastmcp import FastMCP

# Import tools and resources
from slicer_mcp import diagnostic_tools_ct, resources, tools
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
        "Registered 18 tools: capture_screenshot, list_scene_nodes, "
        "execute_python, measure_volume, list_sample_data, load_sample_data, "
        "set_layout, import_dicom, list_dicom_studies, list_dicom_series, "
        "load_dicom_series, run_brain_extraction, detect_vertebral_fractures_ct, "
        "assess_osteoporosis_ct, detect_metastatic_lesions_ct, calculate_sins_score, "
        "measure_listhesis_ct, measure_spinal_canal_ct"
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
