"""MCP tool implementations for Slicer Bridge."""

import base64
import hashlib
import json
import logging
import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any

from slicer_mcp.constants import (
    AUDIT_CODE_MAX_LENGTH,
    AUDIT_RESULT_MAX_LENGTH,
    DICOM_UID_PATTERN,
    FALLBACK_SAMPLE_DATASETS,
    MAX_DICOM_UID_LENGTH,
    MAX_NODE_ID_LENGTH,
    MAX_PYTHON_CODE_LENGTH,
    MAX_SEGMENT_NAME_LENGTH,
    VALID_GUI_MODES,
    VALID_LAYOUTS,
    VIEW_MAP,
)
from slicer_mcp.slicer_client import SlicerConnectionError, get_client

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# JSON Parsing Helper (Error Handling)
# =============================================================================


def _parse_json_result(result: str, context: str) -> Any:
    """Parse JSON result with null/empty handling.

    Args:
        result: JSON string to parse
        context: Description for error messages

    Returns:
        Parsed JSON data

    Raises:
        SlicerConnectionError: If result is empty, null, or malformed
    """
    if not result or result.strip() in ("", "null", "None"):
        raise SlicerConnectionError(
            f"Empty result from {context}", details={"result": result[:100] if result else "None"}
        )

    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        raise SlicerConnectionError(
            f"Failed to parse {context} result: {str(e)}",
            details={"result_preview": result[:100] if result else "None"},
        )


# =============================================================================
# Audit Logging (Security)
# =============================================================================

# Forbidden directories for audit log (security measure)
FORBIDDEN_AUDIT_PATHS = frozenset(
    [
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/var",
        "/root",
        "/lib",
        "/System",
        "/Library",
        "/Applications",  # macOS
        "/Windows",
        "/Program Files",
        "/Program Files (x86)",  # Windows
    ]
)


def _validate_audit_log_path(path: str) -> str:
    """Validate audit log path is safe to write to.

    Args:
        path: Proposed audit log file path

    Returns:
        Validated absolute path

    Raises:
        ValueError: If path is in a forbidden directory
    """
    # Expand user home directory and resolve to absolute path
    abs_path = os.path.realpath(os.path.expanduser(path))

    # Check against forbidden directories
    for forbidden in FORBIDDEN_AUDIT_PATHS:
        if abs_path.lower().startswith(forbidden.lower()):
            raise ValueError(
                f"Audit log path '{path}' is in forbidden directory '{forbidden}'. "
                f"Use a path in your home directory or project directory."
            )

    return abs_path


# Dedicated audit logger for Python code execution
audit_logger = logging.getLogger("slicer-mcp.audit")

# Configure audit logging based on environment
_audit_file = os.environ.get("SLICER_AUDIT_LOG")
if _audit_file:
    try:
        validated_path = _validate_audit_log_path(_audit_file)
        _audit_handler = logging.FileHandler(validated_path)
        _audit_handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(_audit_handler)
        audit_logger.setLevel(logging.INFO)
        logger.info(f"Audit logging enabled: {validated_path}")
    except ValueError as e:
        logger.warning(f"Invalid audit log path, audit logging disabled: {e}")

# Maximum code length to log in full (larger code is truncated with hash)
# Note: AUDIT_CODE_MAX_LENGTH is imported from constants


def _audit_log_execution(
    code: str,
    request_id: str,
    success: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Log Python code execution to audit log.

    Creates structured JSON audit entries for security monitoring.

    Args:
        code: The Python code that was executed
        request_id: Unique identifier for this execution request
        success: Whether execution succeeded
        result: Execution result (if successful)
        error: Error message (if failed)
    """
    # Compute hash for code identification (useful for large code blocks)
    code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

    # Truncate code for logging if too long
    code_preview = code[:AUDIT_CODE_MAX_LENGTH]
    if len(code) > AUDIT_CODE_MAX_LENGTH:
        code_preview += f"... [truncated, {len(code)} chars total]"

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "python_execution",
        "request_id": request_id,
        "code_hash": code_hash,
        "code_length": len(code),
        "code_preview": code_preview,
        "success": success,
    }

    if success and result:
        # Log result preview (truncated if needed)
        result_str = str(result.get("result", ""))
        if len(result_str) > AUDIT_RESULT_MAX_LENGTH:
            result_str = result_str[:AUDIT_RESULT_MAX_LENGTH] + "..."
        audit_entry["result_preview"] = result_str
        audit_entry["has_stdout"] = bool(result.get("stdout"))
        audit_entry["has_stderr"] = bool(result.get("stderr"))
    elif error:
        audit_entry["error"] = error[:500] if len(error) > 500 else error

    audit_logger.info(json.dumps(audit_entry))


# =============================================================================
# Input Validation Functions (Security)
# =============================================================================

# Pattern for valid MRML node IDs: starts with letter, alphanumeric + underscore
MRML_ID_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

# Pattern for valid segment names: Unicode word chars, spaces, underscores, hyphens
# Uses re.UNICODE to support medical terminology with Greek letters (α, β, μ),
# accented characters (é, ñ, ü), and other international alphabets.
# Security: Still blocks shell metacharacters (;`$|&), quotes, and control characters.
SEGMENT_NAME_PATTERN = re.compile(r"^[\w\s\-]+$", re.UNICODE)

# Pattern for DICOM UIDs - compiled for efficiency
DICOM_UID_COMPILED = re.compile(DICOM_UID_PATTERN)

# Zero-width and invisible characters that could be used for attacks
# NFKC normalization doesn't remove these, so we filter explicitly
INVISIBLE_CHARS = frozenset(
    [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\ufeff",  # BOM / Zero-width no-break space
        "\u00ad",  # Soft hyphen
        "\u2060",  # Word joiner
        "\u2061",  # Function application
        "\u2062",  # Invisible times
        "\u2063",  # Invisible separator
        "\u2064",  # Invisible plus
    ]
)


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str, value: str):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


def validate_mrml_node_id(node_id: str) -> str:
    """Validate MRML node ID format to prevent code injection.

    Valid format: starts with letter, contains only alphanumeric and underscore.
    Examples: vtkMRMLScalarVolumeNode1, vtkMRMLSegmentationNode2

    Args:
        node_id: The MRML node ID to validate

    Returns:
        The validated node_id (unchanged if valid)

    Raises:
        ValidationError: If node_id format is invalid
    """
    if not node_id:
        raise ValidationError("Node ID cannot be empty", "node_id", node_id or "")

    if len(node_id) > MAX_NODE_ID_LENGTH:
        raise ValidationError(
            f"Node ID exceeds maximum length ({MAX_NODE_ID_LENGTH})",
            "node_id",
            node_id[:50] + "...",
        )

    if not MRML_ID_PATTERN.match(node_id):
        raise ValidationError(
            f"Invalid node_id format. Must start with letter and contain only "
            f"alphanumeric characters and underscores. Got: '{node_id[:50]}'",
            "node_id",
            node_id,
        )

    return node_id


def validate_segment_name(segment_name: str) -> str:
    """Validate and normalize segment name format to prevent code injection.

    Normalization:
    - Apply NFKC Unicode normalization (prevents homoglyph and zero-width attacks)
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space

    Valid format: Unicode word characters, spaces, underscores, and hyphens.
    Examples: Tumor, Left Lung, Segment_1, Brain-Stem, α-fetoprotein, Müller cells

    Security: NFKC normalization converts lookalike characters to canonical forms,
    removes zero-width characters, and normalizes compatibility characters.

    Args:
        segment_name: The segment name to validate

    Returns:
        The validated and normalized segment_name

    Raises:
        ValidationError: If segment_name format is invalid
    """
    if not segment_name:
        raise ValidationError("Segment name cannot be empty", "segment_name", segment_name or "")

    # Apply NFKC Unicode normalization first
    # - Converts lookalike characters to canonical forms (homoglyph protection)
    # - Normalizes compatibility characters (fullwidth -> ASCII)
    normalized = unicodedata.normalize("NFKC", segment_name)

    # Remove zero-width and invisible characters that could be used for attacks
    # NFKC doesn't remove these, so we filter explicitly
    normalized = "".join(c for c in normalized if c not in INVISIBLE_CHARS)

    # Normalize whitespace: strip and collapse multiple spaces
    normalized = " ".join(normalized.split())

    # Check if normalization resulted in empty string (was only whitespace)
    if not normalized:
        raise ValidationError(
            "Segment name cannot be only whitespace", "segment_name", segment_name
        )

    if len(normalized) > MAX_SEGMENT_NAME_LENGTH:
        raise ValidationError(
            f"Segment name exceeds maximum length ({MAX_SEGMENT_NAME_LENGTH})",
            "segment_name",
            normalized[:50] + "...",
        )

    if not SEGMENT_NAME_PATTERN.match(normalized):
        raise ValidationError(
            f"Invalid segment_name format. Must contain only Unicode word characters, "
            f"spaces, underscores, and hyphens. Got: '{normalized[:50]}'",
            "segment_name",
            normalized,
        )

    return normalized


def validate_folder_path(path: str) -> str:
    """Validate folder path for DICOM import.

    Security checks:
    - Path must exist and be a directory
    - No path traversal attacks (..)
    - Reasonable path length

    Args:
        path: Folder path to validate

    Returns:
        Validated absolute path

    Raises:
        ValidationError: If path is invalid or unsafe
    """
    from slicer_mcp.constants import FORBIDDEN_PATH_COMPONENTS, MAX_FOLDER_PATH_LENGTH

    if not path:
        raise ValidationError("Folder path cannot be empty", "folder_path", "")

    if len(path) > MAX_FOLDER_PATH_LENGTH:
        raise ValidationError(
            f"Folder path exceeds maximum length ({MAX_FOLDER_PATH_LENGTH})",
            "folder_path",
            path[:50] + "...",
        )

    # Check for path traversal attacks
    path_parts = path.replace("\\", "/").split("/")
    for part in path_parts:
        if part in FORBIDDEN_PATH_COMPONENTS:
            raise ValidationError(
                f"Path contains forbidden component: '{part}'", "folder_path", path
            )

    # Resolve to absolute path
    abs_path = os.path.abspath(os.path.expanduser(path))

    # Check path exists and is directory
    if not os.path.exists(abs_path):
        raise ValidationError(f"Path does not exist: {abs_path}", "folder_path", path)

    if not os.path.isdir(abs_path):
        raise ValidationError(f"Path is not a directory: {abs_path}", "folder_path", path)

    return abs_path


def validate_dicom_uid(uid: str, field_name: str = "uid") -> str:
    """Validate DICOM UID format.

    DICOM UIDs contain only digits and dots, e.g., "1.2.840.113619.2.55.3.604688"

    Args:
        uid: DICOM UID to validate
        field_name: Field name for error messages

    Returns:
        Validated UID (unchanged if valid)

    Raises:
        ValidationError: If UID format is invalid
    """
    if not uid:
        raise ValidationError(f"DICOM {field_name} cannot be empty", field_name, "")

    if len(uid) > MAX_DICOM_UID_LENGTH:
        raise ValidationError(
            f"DICOM {field_name} exceeds maximum length ({MAX_DICOM_UID_LENGTH})",
            field_name,
            uid[:50] + "...",
        )

    if not DICOM_UID_COMPILED.match(uid):
        raise ValidationError(
            f"Invalid DICOM {field_name} format. Must contain only digits and dots. Got: '{uid[:30]}'",
            field_name,
            uid,
        )

    return uid


def capture_screenshot(
    view_type: str, scroll_position: float | None = None, look_from_axis: str | None = None
) -> dict[str, Any]:
    """Capture a screenshot from a specific 3D Slicer viewport.

    Args:
        view_type: Viewport type to capture - "axial", "sagittal", "coronal", "3d", "full"
        scroll_position: Slice position from 0.0 to 1.0 (only for 2D views)
        look_from_axis: Camera axis for 3D view - "left", "right", "anterior", "posterior", "superior", "inferior"

    Returns:
        Dict with success status, base64-encoded PNG image, and metadata

    Raises:
        ValueError: If view_type is invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    # Map view types to Slicer view names (using centralized VIEW_MAP)
    if view_type not in VIEW_MAP:
        raise ValueError(
            f"Invalid view_type '{view_type}'. Must be one of: {', '.join(VIEW_MAP.keys())}"
        )

    client = get_client()

    try:
        # Capture screenshot based on view type
        if view_type == "full":
            image_bytes = client.get_full_screenshot()
        elif view_type == "3d":
            image_bytes = client.get_3d_screenshot(look_from_axis)
        else:
            # 2D slice views (axial, sagittal, coronal)
            slicer_view = VIEW_MAP[view_type]
            image_bytes = client.get_screenshot(slicer_view, scroll_position)

        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        logger.info(f"Screenshot captured: view_type={view_type}, size={len(image_base64)} chars")

        result = {
            "success": True,
            "image_base64": image_base64,
            "view_type": view_type,
            "content_type": "image/png",
        }

        if scroll_position is not None:
            result["scroll_position"] = scroll_position

        if look_from_axis is not None:
            result["look_from_axis"] = look_from_axis

        return result

    except SlicerConnectionError as e:
        logger.error(f"Screenshot capture failed: {e.message}")
        raise


def list_scene_nodes() -> dict[str, Any]:
    """List all nodes in the current MRML scene.

    Returns:
        Dict with nodes list and total count

    Raises:
        SlicerConnectionError: If Slicer is not reachable
    """
    client = get_client()

    try:
        nodes = client.get_scene_nodes()

        logger.info(f"Listed {len(nodes)} scene nodes")

        return {"nodes": nodes, "total_count": len(nodes)}

    except SlicerConnectionError as e:
        logger.error(f"Scene nodes listing failed: {e.message}")
        raise


def execute_python(code: str) -> dict[str, Any]:
    """Execute arbitrary Python code in Slicer's Python environment.

    Security Warning: This executes code directly in Slicer. Use only with trusted code.
    All executions are logged to the audit log for security monitoring.

    Args:
        code: Python code to execute

    Returns:
        Dict with success status and execution result

    Raises:
        ValidationError: If code exceeds maximum size limit
        SlicerConnectionError: If Slicer is not reachable or execution fails
    """
    # Validate code size
    if len(code) > MAX_PYTHON_CODE_LENGTH:
        raise ValidationError(
            f"Code exceeds maximum length of {MAX_PYTHON_CODE_LENGTH} bytes",
            field="code",
            value=f"<{len(code)} bytes>",
        )

    # Generate unique request ID for audit trail
    request_id = str(uuid.uuid4())[:8]

    client = get_client()

    try:
        result = client.exec_python(code)

        # Audit log successful execution
        _audit_log_execution(code, request_id, success=True, result=result)

        logger.info(f"Python code executed successfully (request_id={request_id})")

        return result

    except SlicerConnectionError as e:
        # Audit log failed execution
        _audit_log_execution(code, request_id, success=False, error=e.message)

        logger.error(f"Python execution failed (request_id={request_id}): {e.message}")
        raise


def _build_single_segment_volume_code(safe_node_id: str, safe_segment_name: str) -> str:
    """Build Python code to measure volume of a single segment.

    Args:
        safe_node_id: JSON-escaped node ID string
        safe_segment_name: JSON-escaped segment name string

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json
from SegmentStatistics import SegmentStatisticsLogic

# Use JSON-escaped values for safety
node_id = {safe_node_id}
segment_name = {safe_segment_name}

segmentationNode = slicer.mrmlScene.GetNodeByID(node_id)
if not segmentationNode:
    raise ValueError('Node not found: ' + node_id)

# Get segment
segmentation = segmentationNode.GetSegmentation()
segment = segmentation.GetSegment(segment_name)
if not segment:
    raise ValueError('Segment not found: ' + segment_name)

# Calculate statistics
statsLogic = SegmentStatisticsLogic()
statsLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
statsLogic.computeStatistics()
stats = statsLogic.getStatistics()

# Get volume in cc (cubic cm = ml)
volume_cc = stats[segment_name, 'SegmentStatistics.volume_cc']
volume_mm3 = volume_cc * 1000  # Convert cc to mm3

result = {{
    'node_id': node_id,
    'node_name': segmentationNode.GetName(),
    'total_volume_mm3': volume_mm3,
    'total_volume_ml': volume_cc,
    'segments': [
        {{
            'name': segment_name,
            'volume_mm3': volume_mm3,
            'volume_ml': volume_cc
        }}
    ]
}}

print(json.dumps(result))
"""


def _build_all_segments_volume_code(safe_node_id: str) -> str:
    """Build Python code to measure volume of all segments in a segmentation node.

    Args:
        safe_node_id: JSON-escaped node ID string

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json
from SegmentStatistics import SegmentStatisticsLogic

# Use JSON-escaped value for safety
node_id = {safe_node_id}

segmentationNode = slicer.mrmlScene.GetNodeByID(node_id)
if not segmentationNode:
    raise ValueError('Node not found: ' + node_id)

# Calculate statistics
statsLogic = SegmentStatisticsLogic()
statsLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
statsLogic.computeStatistics()
stats = statsLogic.getStatistics()

# Get all segments
segmentation = segmentationNode.GetSegmentation()
segments = []
total_volume_mm3 = 0
total_volume_ml = 0

for i in range(segmentation.GetNumberOfSegments()):
    segment = segmentation.GetNthSegment(i)
    segment_name = segment.GetName()

    # Get volume in cc (cubic cm = ml)
    volume_cc = stats[segment_name, 'SegmentStatistics.volume_cc']
    volume_mm3 = volume_cc * 1000  # Convert cc to mm3

    segments.append({{
        'name': segment_name,
        'volume_mm3': volume_mm3,
        'volume_ml': volume_cc
    }})

    total_volume_mm3 += volume_mm3
    total_volume_ml += volume_cc

result = {{
    'node_id': node_id,
    'node_name': segmentationNode.GetName(),
    'total_volume_mm3': total_volume_mm3,
    'total_volume_ml': total_volume_ml,
    'segments': segments
}}

print(json.dumps(result))
"""


def measure_volume(node_id: str, segment_name: str | None = None) -> dict[str, Any]:
    """Calculate the volume of a segmentation node or specific segment.

    Args:
        node_id: MRML node ID of segmentation (e.g., vtkMRMLSegmentationNode1)
        segment_name: Specific segment to measure (if None, measures all segments)

    Returns:
        Dict with volume measurements in mm3 and ml

    Raises:
        ValidationError: If node_id or segment_name format is invalid
        SlicerConnectionError: If Slicer is not reachable or calculation fails
    """
    # Validate inputs to prevent code injection
    node_id = validate_mrml_node_id(node_id)
    if segment_name is not None:
        segment_name = validate_segment_name(segment_name)

    client = get_client()

    # Use json.dumps for safe string escaping (defense-in-depth)
    # Even though validation exists, this prevents code injection if validation is ever bypassed
    safe_node_id = json.dumps(node_id)
    safe_segment_name = json.dumps(segment_name) if segment_name else None

    # Build Python code to calculate volumes using SegmentStatistics
    if segment_name:
        python_code = _build_single_segment_volume_code(safe_node_id, safe_segment_name)
    else:
        python_code = _build_all_segments_volume_code(safe_node_id)

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result using helper
        volume_data = _parse_json_result(exec_result.get("result", ""), "volume measurement")

        logger.info(f"Volume measured for node {node_id}")

        return volume_data

    except SlicerConnectionError as e:
        logger.error(f"Volume measurement failed: {e.message}")
        raise


def list_sample_data() -> dict[str, Any]:
    """List all available sample datasets from 3D Slicer's SampleData module.

    Dynamically queries Slicer to discover available sample datasets.
    Falls back to a known list if Slicer is not connected.

    Returns:
        Dict with available datasets list and metadata

    Raises:
        SlicerConnectionError: If Slicer is not reachable
    """
    client = get_client()

    # Python code to get available sample datasets from SampleData module
    python_code = """
import slicer
import json

try:
    import SampleData

    # Get all registered sample data sources
    sampleDataLogic = SampleData.SampleDataLogic()
    sources = sampleDataLogic.builtInSources() if hasattr(sampleDataLogic, 'builtInSources') else []

    # Also check for registered sources in the module
    registeredSources = []
    if hasattr(SampleData, 'SampleDataSources'):
        for category, sourceList in SampleData.SampleDataSources.items():
            for source in sourceList:
                registeredSources.append({
                    'name': source.get('sampleName', source.get('loadFileType', 'Unknown')),
                    'category': category,
                    'description': source.get('uris', [''])[0] if source.get('uris') else ''
                })

    # Fallback: query the SampleData module widget for available items
    if not registeredSources:
        # Try to get sample names from the logic
        sampleNames = []
        for attr in dir(sampleDataLogic):
            if attr.startswith('download') and callable(getattr(sampleDataLogic, attr)):
                # Extract dataset name from method name (e.g., downloadMRHead -> MRHead)
                name = attr.replace('download', '')
                if name and name[0].isupper():
                    sampleNames.append(name)

        registeredSources = [{'name': name, 'category': 'BuiltIn', 'description': ''} for name in sampleNames]

    result = {
        'datasets': registeredSources,
        'total_count': len(registeredSources),
        'source': 'dynamic'
    }
except Exception as e:
    # Fallback to known common datasets
    result = {
        'datasets': [
            {'name': 'MRHead', 'category': 'BuiltIn', 'description': 'MR head scan'},
            {'name': 'CTChest', 'category': 'BuiltIn', 'description': 'CT chest scan'},
            {'name': 'CTACardio', 'category': 'BuiltIn', 'description': 'CTA cardiac scan'},
            {'name': 'DTIBrain', 'category': 'BuiltIn', 'description': 'DTI brain scan'},
            {'name': 'MRBrainTumor1', 'category': 'BuiltIn', 'description': 'MR brain tumor case 1'},
            {'name': 'MRBrainTumor2', 'category': 'BuiltIn', 'description': 'MR brain tumor case 2'}
        ],
        'total_count': 6,
        'source': 'fallback',
        'error': str(e)
    }

print(json.dumps(result))
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result using helper
        sample_data = _parse_json_result(exec_result.get("result", ""), "sample data list")

        logger.info(
            f"Sample data list retrieved: {sample_data['total_count']} datasets ({sample_data['source']})"
        )

        return sample_data

    except SlicerConnectionError as e:
        logger.warning(f"Dynamic sample data discovery failed, using fallback: {e.message}")
        # Return fallback list when Slicer is not connected
        return {
            "datasets": [
                {"name": name, "category": "BuiltIn", "description": ""}
                for name in FALLBACK_SAMPLE_DATASETS
            ],
            "total_count": len(FALLBACK_SAMPLE_DATASETS),
            "source": "fallback",
            "error": e.message,
        }


def _get_valid_datasets() -> list[str]:
    """Get list of valid dataset names, trying dynamic discovery first.

    Returns:
        List of valid dataset names
    """
    try:
        sample_data = list_sample_data()
        return [d["name"] for d in sample_data["datasets"]]
    except (SlicerConnectionError, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Dynamic dataset discovery failed, using fallback: {e}")
        return FALLBACK_SAMPLE_DATASETS


def load_sample_data(dataset_name: str) -> dict[str, Any]:
    """Load a sample dataset into 3D Slicer.

    Args:
        dataset_name: Name of sample dataset (use list_sample_data() to see available options)

    Returns:
        Dict with success status and loaded node information

    Raises:
        ValueError: If dataset_name is invalid
        SlicerConnectionError: If Slicer is not reachable or load fails
    """
    # Try dynamic discovery first, fall back to static list
    valid_datasets = _get_valid_datasets()

    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Invalid dataset_name '{dataset_name}'. "
            f"Available datasets: {', '.join(valid_datasets)}"
        )

    client = get_client()

    try:
        result = client.load_sample_data(dataset_name)

        # Get loaded node info via Python
        python_code = """
import slicer
import json

# Get most recently added volume node (sample data creates new volume)
nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
if nodes:
    latest_node = nodes[-1]
    result = {
        'loaded_node_id': latest_node.GetID(),
        'loaded_node_name': latest_node.GetName()
    }
else:
    result = {
        'loaded_node_id': None,
        'loaded_node_name': None
    }

print(json.dumps(result))
"""
        exec_result = client.exec_python(python_code)

        # Parse JSON result using helper
        node_info = _parse_json_result(exec_result.get("result", ""), "loaded node info")

        result.update(node_info)

        logger.info(f"Sample data '{dataset_name}' loaded successfully")

        return result

    except SlicerConnectionError as e:
        logger.error(f"Sample data load failed: {e.message}")
        raise


def set_layout(layout: str, gui_mode: str = "full") -> dict[str, Any]:
    """Set the viewer layout and GUI mode in 3D Slicer.

    Args:
        layout: Layout name - "FourUp", "OneUp3D", "OneUpRedSlice", "Conventional", "SideBySide"
        gui_mode: GUI mode - "full" (complete GUI) or "viewers" (viewers only)

    Returns:
        Dict with success status and layout information

    Raises:
        ValueError: If layout or gui_mode is invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Invalid layout '{layout}'. Must be one of: {', '.join(VALID_LAYOUTS)}")

    if gui_mode not in VALID_GUI_MODES:
        raise ValueError(
            f"Invalid gui_mode '{gui_mode}'. Must be one of: {', '.join(VALID_GUI_MODES)}"
        )

    client = get_client()

    try:
        result = client.set_layout(layout, gui_mode)

        logger.info(f"Layout set to {layout} with {gui_mode} GUI mode")

        return result

    except SlicerConnectionError as e:
        logger.error(f"Layout change failed: {e.message}")
        raise


def list_dicom_series(study_uid: str) -> dict[str, Any]:
    """List all series within a DICOM study.

    Args:
        study_uid: DICOM Study UID

    Returns:
        Dict with series list and metadata

    Raises:
        ValidationError: If study_uid format is invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    # Validate UID format
    study_uid = validate_dicom_uid(study_uid, "study_uid")

    client = get_client()

    # Safe string for Python code
    safe_uid = json.dumps(study_uid)

    python_code = f"""
import slicer
import json

study_uid = {safe_uid}
db = slicer.dicomDatabase

if not db or not db.isOpen:
    result = {{
        'success': False,
        'error': 'DICOM database is not open',
        'series': [],
        'total_count': 0
    }}
else:
    series_list = db.seriesForStudy(study_uid)

    if not series_list:
        result = {{
            'success': False,
            'error': f'Study not found: {{study_uid}}',
            'series': [],
            'total_count': 0
        }}
    else:
        series = []

        for series_uid in series_list:
            # Get series metadata
            modality = db.fieldForSeries('Modality', series_uid) or ''
            series_desc = db.fieldForSeries('SeriesDescription', series_uid) or ''
            series_num = db.fieldForSeries('SeriesNumber', series_uid) or ''

            # Count files in series
            files = db.filesForSeries(series_uid)

            series.append({{
                'series_uid': series_uid,
                'series_number': series_num,
                'series_description': series_desc,
                'modality': modality,
                'file_count': len(files)
            }})

        # Sort by series number
        series.sort(key=lambda x: int(x['series_number']) if x['series_number'].isdigit() else 0)

        result = {{
            'success': True,
            'study_uid': study_uid,
            'series': series,
            'total_count': len(series)
        }}

print(json.dumps(result))
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result
        series_data = _parse_json_result(exec_result.get("result", ""), "DICOM series list")

        logger.info(f"Listed {series_data['total_count']} series for study {study_uid[:20]}...")

        return series_data

    except SlicerConnectionError as e:
        logger.error(f"DICOM series list failed: {e.message}")
        raise


def load_dicom_series(series_uid: str) -> dict[str, Any]:
    """Load a DICOM series as a volume into the scene.

    Args:
        series_uid: DICOM Series UID to load

    Returns:
        Dict with loaded node information including dimensions and spacing

    Raises:
        ValidationError: If series_uid format is invalid
        SlicerConnectionError: If Slicer is not reachable or load fails
    """
    # Validate UID format
    series_uid = validate_dicom_uid(series_uid, "series_uid")

    client = get_client()

    # Safe string for Python code
    safe_uid = json.dumps(series_uid)

    python_code = f"""
import slicer
import json

series_uid = {safe_uid}

from DICOMLib import DICOMUtils

# Load the series
loadedNodeIDs = DICOMUtils.loadSeriesByUID([series_uid])

if not loadedNodeIDs:
    result = {{
        'success': False,
        'error': f'Failed to load series: {{series_uid}}',
        'node_id': None
    }}
else:
    node_id = loadedNodeIDs[0]
    node = slicer.mrmlScene.GetNodeByID(node_id)

    if node is None:
        result = {{
            'success': False,
            'error': f'Node not found after load: {{node_id}}',
            'node_id': node_id
        }}
    else:
        # Get volume information
        result = {{
            'success': True,
            'series_uid': series_uid,
            'node_id': node_id,
            'node_name': node.GetName(),
            'node_class': node.GetClassName()
        }}

        # Add dimensions and spacing if it's a volume node
        if hasattr(node, 'GetImageData') and node.GetImageData():
            dims = node.GetImageData().GetDimensions()
            spacing = node.GetSpacing()
            origin = node.GetOrigin()

            result.update({{
                'dimensions': list(dims),
                'spacing': list(spacing),
                'origin': list(origin)
            }})

            # Get scalar range if available
            scalar_range = node.GetImageData().GetScalarRange()
            result['scalar_range'] = list(scalar_range)

print(json.dumps(result))
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result
        load_data = _parse_json_result(exec_result.get("result", ""), "DICOM series load")

        if load_data.get("success"):
            logger.info(f"Loaded DICOM series {series_uid[:20]}... as {load_data['node_name']}")
        else:
            logger.warning(f"Failed to load DICOM series: {load_data.get('error')}")

        return load_data

    except SlicerConnectionError as e:
        logger.error(f"DICOM series load failed: {e.message}")
        raise


def import_dicom(folder_path: str) -> dict[str, Any]:
    """Import DICOM files from a folder into Slicer's DICOM database.

    Args:
        folder_path: Path to folder containing DICOM files

    Returns:
        Dict with import results including patient/study/series counts

    Raises:
        ValidationError: If folder_path is invalid
        SlicerConnectionError: If Slicer is not reachable or import fails
    """
    # Validate folder path
    validated_path = validate_folder_path(folder_path)

    client = get_client()

    # Safe string for Python code
    safe_path = json.dumps(validated_path)

    python_code = f"""
import slicer
import json

folder_path = {safe_path}

# Import DICOM files
from DICOMLib import DICOMUtils

# Get initial counts
db = slicer.dicomDatabase
initial_patients = len(db.patients()) if db else 0

# Import the folder
DICOMUtils.importDicom(folder_path)

# Get new counts
final_patients = len(db.patients()) if db else 0
all_studies = []
all_series = []

for patient in db.patients():
    studies = db.studiesForPatient(patient)
    all_studies.extend(studies)
    for study in studies:
        series = db.seriesForStudy(study)
        all_series.extend(series)

result = {{
    'success': True,
    'folder_path': folder_path,
    'patients_count': len(db.patients()) if db else 0,
    'studies_count': len(all_studies),
    'series_count': len(all_series),
    'new_patients': final_patients - initial_patients
}}

print(json.dumps(result))
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result
        import_data = _parse_json_result(exec_result.get("result", ""), "DICOM import")

        logger.info(f"DICOM import completed: {import_data['series_count']} series")

        return import_data

    except SlicerConnectionError as e:
        logger.error(f"DICOM import failed: {e.message}")
        raise


def list_dicom_studies() -> dict[str, Any]:
    """List all studies in the DICOM database.

    Returns:
        Dict with studies list and metadata

    Raises:
        SlicerConnectionError: If Slicer is not reachable
    """
    client = get_client()

    python_code = """
import slicer
import json

db = slicer.dicomDatabase
if not db or not db.isOpen:
    result = {
        'success': False,
        'error': 'DICOM database is not open',
        'studies': [],
        'total_count': 0
    }
else:
    studies = []

    for patient in db.patients():
        patient_name = db.nameForPatient(patient)

        for study_uid in db.studiesForPatient(patient):
            # Get study metadata
            study_date = db.fieldForStudy('StudyDate', study_uid) or ''
            study_desc = db.fieldForStudy('StudyDescription', study_uid) or ''

            # Get modalities from series
            modalities = set()
            series_list = db.seriesForStudy(study_uid)
            for series_uid in series_list:
                modality = db.fieldForSeries('Modality', series_uid)
                if modality:
                    modalities.add(modality)

            studies.append({
                'patient_id': patient,
                'patient_name': patient_name,
                'study_uid': study_uid,
                'study_date': study_date,
                'study_description': study_desc,
                'modalities': list(modalities),
                'series_count': len(series_list)
            })

    result = {
        'success': True,
        'studies': studies,
        'total_count': len(studies)
    }

print(json.dumps(result))
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result
        studies_data = _parse_json_result(exec_result.get("result", ""), "DICOM studies list")

        logger.info(f"Listed {studies_data['total_count']} DICOM studies")

        return studies_data

    except SlicerConnectionError as e:
        logger.error(f"DICOM studies list failed: {e.message}")
        raise


def _build_segment_statistics_code(seg_node_var: str) -> str:
    """Build Python code snippet for SegmentStatistics volume calculation.

    Args:
        seg_node_var: Variable name of the segmentation node in the generated code

    Returns:
        Python code snippet that calculates brain_vol_cc from the segmentation
    """
    return f"""
# Calculate brain volume using SegmentStatistics
brain_vol_cc = 0.0
try:
    from SegmentStatistics import SegmentStatisticsLogic
    statsLogic = SegmentStatisticsLogic()
    statsLogic.getParameterNode().SetParameter("Segmentation", {seg_node_var}.GetID())
    statsLogic.computeStatistics()
    stats = statsLogic.getStatistics()
    # Try to get volume - segment name may vary
    for key in stats.keys():
        if 'volume_cc' in str(key):
            brain_vol_cc = stats[key]
            break
except Exception as e:
    print(f"Volume calculation warning: {{e}}")
"""


def _build_hdbet_code(safe_node_id: str, safe_device: str) -> str:
    """Build Python code for HD-BET brain extraction.

    Args:
        safe_node_id: JSON-escaped node ID string
        safe_device: JSON-escaped device string ("auto", "cpu", or GPU index)

    Returns:
        Python code string to execute in Slicer for HD-BET brain extraction
    """
    setup_code = f"""
import slicer
import time
import json

input_node_id = {safe_node_id}
device = {safe_device}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

# Create output nodes
brainVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
brainVolume.SetName(inputVolume.GetName() + "_brain")

brainSeg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
brainSeg.SetName(inputVolume.GetName() + "_brain_mask")

# Run HD-BET
import HDBrainExtractionTool
logic = HDBrainExtractionTool.HDBrainExtractionToolLogic()

start_time = time.time()
logic.process(inputVolume, brainVolume, brainSeg, device=device)
elapsed = time.time() - start_time
"""

    result_code = """
result = {
    "success": True,
    "input_node_id": input_node_id,
    "method": "hd-bet",
    "output_volume_id": brainVolume.GetID(),
    "output_volume_name": brainVolume.GetName(),
    "output_segmentation_id": brainSeg.GetID(),
    "output_segmentation_name": brainSeg.GetName(),
    "brain_volume_ml": round(brain_vol_cc, 2),
    "processing_time_seconds": round(elapsed, 2)
}

print(json.dumps(result))
"""

    return setup_code + _build_segment_statistics_code("brainSeg") + result_code


def _build_swiss_code(safe_node_id: str) -> str:
    """Build Python code for SwissSkullStripper brain extraction.

    Args:
        safe_node_id: JSON-escaped node ID string

    Returns:
        Python code string to execute in Slicer for SwissSkullStripper brain extraction
    """
    setup_code = f"""
import slicer
import time
import json

input_node_id = {safe_node_id}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

# Create output nodes
outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
outputVolume.SetName(inputVolume.GetName() + "_brain")

maskLabel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
maskLabel.SetName(inputVolume.GetName() + "_brain_label")

# Load default atlas from SwissSkullStripper sample data
import SampleData
try:
    atlasVolume = SampleData.downloadSample("SwissSkullStripperAtlas")
    atlasMask = SampleData.downloadSample("SwissSkullStripperAtlasMask")
except Exception as e:
    raise ValueError(f"Failed to download SwissSkullStripper atlas: {{e}}")

# Run SwissSkullStripper CLI module
parameters = {{
    "atlasMRIVolume": atlasVolume.GetID(),
    "atlasMaskVolume": atlasMask.GetID(),
    "patientVolume": inputVolume.GetID(),
    "patientOutputVolume": outputVolume.GetID(),
    "patientMaskLabel": maskLabel.GetID()
}}

start_time = time.time()
cliNode = slicer.cli.runSync(slicer.modules.swissskullstripper, None, parameters)
elapsed = time.time() - start_time

# Check for errors
if cliNode.GetStatus() & cliNode.ErrorsMask:
    raise ValueError(f"SwissSkullStripper failed: {{cliNode.GetErrorText()}}")

# Convert label map to segmentation for consistency
brainSeg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
brainSeg.SetName(inputVolume.GetName() + "_brain_mask")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(maskLabel, brainSeg)
"""

    result_code = """
# Clean up atlas nodes (optional)
slicer.mrmlScene.RemoveNode(atlasVolume)
slicer.mrmlScene.RemoveNode(atlasMask)
slicer.mrmlScene.RemoveNode(maskLabel)

result = {
    "success": True,
    "input_node_id": input_node_id,
    "method": "swiss",
    "output_volume_id": outputVolume.GetID(),
    "output_volume_name": outputVolume.GetName(),
    "output_segmentation_id": brainSeg.GetID(),
    "output_segmentation_name": brainSeg.GetName(),
    "brain_volume_ml": round(brain_vol_cc, 2),
    "processing_time_seconds": round(elapsed, 2)
}

print(json.dumps(result))
"""

    return setup_code + _build_segment_statistics_code("brainSeg") + result_code


def run_brain_extraction(
    input_node_id: str,
    method: str = "hd-bet",
    device: str = "auto",
) -> dict[str, Any]:
    """Run brain extraction (skull stripping) on a brain MRI or CT scan.

    LONG OPERATION: This may take 20 seconds to 5 minutes depending on method and hardware.

    Extracts the brain from surrounding skull and non-brain tissue using
    either AI-based (HD-BET) or atlas-based (SwissSkullStripper) methods.

    Expected durations:
    - hd-bet with GPU: ~20-30 seconds
    - hd-bet with CPU: ~3-5 minutes
    - swiss (atlas-based): ~2-3 minutes

    Args:
        input_node_id: MRML node ID of input brain MRI/CT volume
        method: Extraction method - "hd-bet" (AI, faster with GPU) or "swiss" (atlas-based)
        device: For HD-BET only - "auto", "cpu", or GPU index ("0", "1", etc.)

    Returns:
        Dict with output_volume_id, output_segmentation_id, brain_volume_ml,
        processing_time_seconds, and long_operation metadata

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or processing fails
    """
    from slicer_mcp.constants import (
        BRAIN_EXTRACTION_TIMEOUT,
        VALID_BRAIN_EXTRACTION_METHODS,
        VALID_HDBET_DEVICES,
    )

    # Validate input node ID
    input_node_id = validate_mrml_node_id(input_node_id)

    # Validate method
    if method not in VALID_BRAIN_EXTRACTION_METHODS:
        raise ValidationError(
            f"Invalid method '{method}'. Must be one of: {', '.join(VALID_BRAIN_EXTRACTION_METHODS)}",
            "method",
            method,
        )

    # Validate device (only applies to HD-BET)
    if method == "hd-bet" and device not in VALID_HDBET_DEVICES:
        raise ValidationError(
            f"Invalid device '{device}'. Must be one of: {', '.join(VALID_HDBET_DEVICES)}",
            "device",
            device,
        )

    client = get_client()

    # Safe strings for Python code
    safe_node_id = json.dumps(input_node_id)
    safe_device = json.dumps(device)

    if method == "hd-bet":
        python_code = _build_hdbet_code(safe_node_id, safe_device)
    else:  # swiss method
        python_code = _build_swiss_code(safe_node_id)

    try:
        # Use extended timeout for brain extraction (can take 5+ minutes on CPU)
        exec_result = client.exec_python(python_code, timeout=BRAIN_EXTRACTION_TIMEOUT)

        # Parse JSON result
        result_data = _parse_json_result(
            exec_result.get("result", ""), f"brain extraction ({method})"
        )

        # Add long_operation metadata to inform clients about operation characteristics
        result_data["long_operation"] = {
            "type": "brain_extraction",
            "method": method,
            "timeout_seconds": BRAIN_EXTRACTION_TIMEOUT,
            "typical_duration": {
                "hd-bet-gpu": "20-30 seconds",
                "hd-bet-cpu": "3-5 minutes",
                "swiss": "2-3 minutes",
            }.get(f"{method}-{device}" if method == "hd-bet" else method, "variable"),
        }

        logger.info(
            f"Brain extraction completed: method={method}, "
            f"volume={result_data.get('brain_volume_ml', 0)}ml, "
            f"time={result_data.get('processing_time_seconds', 0)}s"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Brain extraction failed: {e.message}")
        raise
