"""MCP tool implementations for Slicer Bridge."""

import base64
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from slicer_mcp.slicer_client import get_client, SlicerConnectionError
from slicer_mcp.constants import (
    VIEW_MAP,
    VALID_LAYOUTS,
    VALID_GUI_MODES,
    MAX_NODE_ID_LENGTH,
    MAX_SEGMENT_NAME_LENGTH,
    AUDIT_CODE_MAX_LENGTH,
    AUDIT_RESULT_MAX_LENGTH,
)

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
    if not result or result.strip() in ('', 'null', 'None'):
        raise SlicerConnectionError(
            f"Empty result from {context}",
            details={"result": result[:100] if result else "None"}
        )

    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        raise SlicerConnectionError(
            f"Failed to parse {context} result: {str(e)}",
            details={"result_preview": result[:100] if result else "None"}
        )


# =============================================================================
# Audit Logging (Security)
# =============================================================================

# Forbidden directories for audit log (security measure)
FORBIDDEN_AUDIT_PATHS = frozenset([
    '/etc', '/usr', '/bin', '/sbin', '/var', '/root', '/lib',
    '/System', '/Library', '/Applications',  # macOS
    '/Windows', '/Program Files', '/Program Files (x86)',  # Windows
])


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
    abs_path = os.path.abspath(os.path.expanduser(path))

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
        _audit_handler.setFormatter(logging.Formatter('%(message)s'))
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
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
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
MRML_ID_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')

# Pattern for valid segment names: Unicode word chars, spaces, underscores, hyphens
# Uses re.UNICODE to support medical terminology with Greek letters (α, β, μ),
# accented characters (é, ñ, ü), and other international alphabets.
# Security: Still blocks shell metacharacters (;`$|&), quotes, and control characters.
SEGMENT_NAME_PATTERN = re.compile(r'^[\w\s\-]+$', re.UNICODE)


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
        raise ValidationError(f"Node ID exceeds maximum length ({MAX_NODE_ID_LENGTH})", "node_id", node_id[:50] + "...")

    if not MRML_ID_PATTERN.match(node_id):
        raise ValidationError(
            f"Invalid node_id format. Must start with letter and contain only "
            f"alphanumeric characters and underscores. Got: '{node_id[:50]}'",
            "node_id",
            node_id
        )

    return node_id


def validate_segment_name(segment_name: str) -> str:
    """Validate and normalize segment name format to prevent code injection.

    Normalization:
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space

    Valid format: alphanumeric characters, spaces, underscores, and hyphens.
    Examples: Tumor, Left Lung, Segment_1, Brain-Stem

    Args:
        segment_name: The segment name to validate

    Returns:
        The validated and normalized segment_name

    Raises:
        ValidationError: If segment_name format is invalid
    """
    if not segment_name:
        raise ValidationError("Segment name cannot be empty", "segment_name", segment_name or "")

    # Normalize whitespace: strip and collapse multiple spaces
    normalized = ' '.join(segment_name.split())

    # Check if normalization resulted in empty string (was only whitespace)
    if not normalized:
        raise ValidationError("Segment name cannot be only whitespace", "segment_name", segment_name)

    if len(normalized) > MAX_SEGMENT_NAME_LENGTH:
        raise ValidationError(
            f"Segment name exceeds maximum length ({MAX_SEGMENT_NAME_LENGTH})",
            "segment_name",
            normalized[:50] + "..."
        )

    if not SEGMENT_NAME_PATTERN.match(normalized):
        raise ValidationError(
            f"Invalid segment_name format. Must contain only Unicode word characters, "
            f"spaces, underscores, and hyphens. Got: '{normalized[:50]}'",
            "segment_name",
            normalized
        )

    return normalized


def capture_screenshot(
    view_type: str,
    scroll_position: Optional[float] = None,
    look_from_axis: Optional[str] = None
) -> Dict[str, Any]:
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
            f"Invalid view_type '{view_type}'. "
            f"Must be one of: {', '.join(VIEW_MAP.keys())}"
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
            "content_type": "image/png"
        }

        if scroll_position is not None:
            result["scroll_position"] = scroll_position

        if look_from_axis is not None:
            result["look_from_axis"] = look_from_axis

        return result

    except SlicerConnectionError as e:
        logger.error(f"Screenshot capture failed: {e.message}")
        raise


def list_scene_nodes() -> Dict[str, Any]:
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

        return {
            "nodes": nodes,
            "total_count": len(nodes)
        }

    except SlicerConnectionError as e:
        logger.error(f"Scene nodes listing failed: {e.message}")
        raise


def execute_python(code: str) -> Dict[str, Any]:
    """Execute arbitrary Python code in Slicer's Python environment.

    Security Warning: This executes code directly in Slicer. Use only with trusted code.
    All executions are logged to the audit log for security monitoring.

    Args:
        code: Python code to execute

    Returns:
        Dict with success status and execution result

    Raises:
        SlicerConnectionError: If Slicer is not reachable or execution fails
    """
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


def measure_volume(node_id: str, segment_name: Optional[str] = None) -> Dict[str, Any]:
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
        # Measure specific segment
        python_code = f"""
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

json.dumps(result)
"""
    else:
        # Measure all segments
        python_code = f"""
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

json.dumps(result)
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result using helper
        volume_data = _parse_json_result(exec_result.get("result", ""), "volume measurement")

        logger.info(f"Volume measured for node {node_id}")

        return volume_data

    except SlicerConnectionError as e:
        logger.error(f"Volume measurement failed: {e.message}")
        raise


# Fallback list of common sample datasets (used when dynamic discovery fails)
FALLBACK_SAMPLE_DATASETS = [
    "MRHead", "CTChest", "CTACardio", "DTIBrain", "MRBrainTumor1", "MRBrainTumor2"
]


def list_sample_data() -> Dict[str, Any]:
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

json.dumps(result)
"""

    try:
        exec_result = client.exec_python(python_code)

        # Parse JSON result using helper
        sample_data = _parse_json_result(exec_result.get("result", ""), "sample data list")

        logger.info(f"Sample data list retrieved: {sample_data['total_count']} datasets ({sample_data['source']})")

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
            "error": e.message
        }


def _get_valid_datasets() -> List[str]:
    """Get list of valid dataset names, trying dynamic discovery first.

    Returns:
        List of valid dataset names
    """
    try:
        sample_data = list_sample_data()
        return [d["name"] for d in sample_data["datasets"]]
    except Exception:
        return FALLBACK_SAMPLE_DATASETS


def load_sample_data(dataset_name: str) -> Dict[str, Any]:
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
        python_code = f"""
import slicer
import json

# Get most recently added volume node (sample data creates new volume)
nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
if nodes:
    latest_node = nodes[-1]
    result = {{
        'loaded_node_id': latest_node.GetID(),
        'loaded_node_name': latest_node.GetName()
    }}
else:
    result = {{
        'loaded_node_id': None,
        'loaded_node_name': None
    }}

json.dumps(result)
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


def set_layout(layout: str, gui_mode: str = "full") -> Dict[str, Any]:
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
        raise ValueError(
            f"Invalid layout '{layout}'. "
            f"Must be one of: {', '.join(VALID_LAYOUTS)}"
        )

    if gui_mode not in VALID_GUI_MODES:
        raise ValueError(
            f"Invalid gui_mode '{gui_mode}'. "
            f"Must be one of: {', '.join(VALID_GUI_MODES)}"
        )

    client = get_client()

    try:
        result = client.set_layout(layout, gui_mode)

        logger.info(f"Layout set to {layout} with {gui_mode} GUI mode")

        return result

    except SlicerConnectionError as e:
        logger.error(f"Layout change failed: {e.message}")
        raise
