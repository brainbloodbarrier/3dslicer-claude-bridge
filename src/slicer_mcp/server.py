"""MCP Slicer Bridge server entry point."""

import logging
import sys

from mcp.server.fastmcp import FastMCP

# Import tools and resources
from slicer_mcp import diagnostic_tools_xray, resources, tools
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


# X-ray Diagnostic Protocol Tools
# ================================


@mcp.tool()
def measure_sagittal_balance_xray(
    volume_node_id: str,
    landmarks: dict[str, list[float]],
    magnification_factor: float = 1.0,
) -> dict:
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
    """
    try:
        return diagnostic_tools_xray.measure_sagittal_balance_xray(
            volume_node_id, landmarks, magnification_factor
        )
    except Exception as e:
        return _handle_tool_error(e, "measure_sagittal_balance_xray")


@mcp.tool()
def measure_coronal_balance_xray(
    volume_node_id: str,
    landmarks: dict[str, list[float]],
    magnification_factor: float = 1.0,
) -> dict:
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
    """
    try:
        return diagnostic_tools_xray.measure_coronal_balance_xray(
            volume_node_id, landmarks, magnification_factor
        )
    except Exception as e:
        return _handle_tool_error(e, "measure_coronal_balance_xray")


@mcp.tool()
def measure_listhesis_dynamic_xray(
    volume_node_ids: dict[str, str],
    landmarks_per_position: dict[str, dict[str, dict[str, list[float]]]],
    levels: list[str],
    region: str = "lumbar",
    magnification_factor: float = 1.0,
) -> dict:
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
    """
    try:
        return diagnostic_tools_xray.measure_listhesis_dynamic_xray(
            volume_node_ids, landmarks_per_position, levels, region, magnification_factor
        )
    except Exception as e:
        return _handle_tool_error(e, "measure_listhesis_dynamic_xray")


@mcp.tool()
def detect_vertebral_fractures_xray(
    volume_node_id: str,
    landmarks_per_vertebra: dict[str, dict[str, list[float]]],
    magnification_factor: float = 1.0,
) -> dict:
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
    """
    try:
        return diagnostic_tools_xray.detect_vertebral_fractures_xray(
            volume_node_id, landmarks_per_vertebra, magnification_factor
        )
    except Exception as e:
        return _handle_tool_error(e, "detect_vertebral_fractures_xray")


@mcp.tool()
def measure_cobb_angle_xray(
    volume_node_id: str,
    landmarks: dict[str, list[float]],
    upper_end_vertebra: str = "",
    lower_end_vertebra: str = "",
    curve_type: str = "primary",
) -> dict:
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
    """
    try:
        return diagnostic_tools_xray.measure_cobb_angle_xray(
            volume_node_id, landmarks, upper_end_vertebra, lower_end_vertebra, curve_type
        )
    except Exception as e:
        return _handle_tool_error(e, "measure_cobb_angle_xray")


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
        "Registered 17 tools: capture_screenshot, list_scene_nodes, "
        "execute_python, measure_volume, list_sample_data, load_sample_data, "
        "set_layout, import_dicom, list_dicom_studies, list_dicom_series, "
        "load_dicom_series, run_brain_extraction, measure_sagittal_balance_xray, "
        "measure_coronal_balance_xray, measure_listhesis_dynamic_xray, "
        "detect_vertebral_fractures_xray, measure_cobb_angle_xray"
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
