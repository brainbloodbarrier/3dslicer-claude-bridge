"""MCP tool implementations for volume rendering and 3D model export."""

import json
import logging
import os
import re

from slicer_mcp.constants import (
    EXPORT_FILENAME_PATTERN,
    EXPORT_FORMAT_EXTENSIONS,
    MAX_EXPORT_FILENAME_LENGTH,
    MAX_FOLDER_PATH_LENGTH,
    MODEL_EXPORT_TIMEOUT,
    VALID_EXPORT_FORMATS,
    VALID_VR_PRESETS,
    VOLUME_RENDERING_TIMEOUT,
)
from slicer_mcp.slicer_client import get_client
from slicer_mcp.tools import ValidationError, _parse_json_result, validate_mrml_node_id

logger = logging.getLogger("slicer-mcp")

# Compiled pattern for export filenames
EXPORT_FILENAME_COMPILED = re.compile(EXPORT_FILENAME_PATTERN)


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_export_filename(filename: str) -> str:
    """Validate an export filename against the allowed pattern.

    Args:
        filename: Filename (without extension) to validate

    Returns:
        The validated filename (unchanged if valid)

    Raises:
        ValidationError: If filename is empty, too long, or contains invalid characters
    """
    if not filename:
        raise ValidationError("Export filename cannot be empty", "filename", filename or "")

    if len(filename) > MAX_EXPORT_FILENAME_LENGTH:
        raise ValidationError(
            f"Filename exceeds maximum length ({MAX_EXPORT_FILENAME_LENGTH})",
            "filename",
            filename[:50] + "...",
        )

    if not EXPORT_FILENAME_COMPILED.match(filename):
        raise ValidationError(
            "Filename contains invalid characters (allowed: alphanumeric, _ - . space ())",
            "filename",
            filename[:50],
        )

    return filename


def validate_export_directory(directory: str) -> str:
    """Validate an export directory path.

    Args:
        directory: Directory path to validate

    Returns:
        Validated absolute path

    Raises:
        ValidationError: If path is empty, too long, contains traversal, or doesn't exist
    """
    if not directory:
        raise ValidationError("Export directory cannot be empty", "directory", directory or "")

    if len(directory) > MAX_FOLDER_PATH_LENGTH:
        raise ValidationError(
            f"Directory path exceeds maximum length ({MAX_FOLDER_PATH_LENGTH})",
            "directory",
            directory[:50] + "...",
        )

    # Check for path traversal
    path_parts = directory.replace("\\", "/").split("/")
    for part in path_parts:
        if part == "..":
            raise ValidationError(
                "Directory path contains forbidden component: '..'",
                "directory",
                directory,
            )

    abs_path = os.path.realpath(os.path.expanduser(directory))

    if not os.path.exists(abs_path):
        raise ValidationError(f"Directory does not exist: {abs_path}", "directory", directory)

    if not os.path.isdir(abs_path):
        raise ValidationError(f"Path is not a directory: {abs_path}", "directory", directory)

    return abs_path


# =============================================================================
# Codegen Functions
# =============================================================================


def _build_enable_volume_rendering_code(
    safe_node_id: str, safe_preset: str, safe_visible: str
) -> str:
    """Build Python code to enable volume rendering on a volume node.

    Args:
        safe_node_id: JSON-escaped volume node ID string
        safe_preset: JSON-escaped preset name string (or "None")
        safe_visible: JSON-escaped boolean string

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer

node_id = {safe_node_id}
preset_name = {safe_preset}
visible = {safe_visible}

volumeNode = slicer.mrmlScene.GetNodeByID(node_id)
if not volumeNode:
    raise ValueError('Volume node not found: ' + node_id)

volRenLogic = slicer.modules.volumerendering.logic()

# Get or create volume rendering display node
displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
if not displayNode:
    displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)

# Apply preset if specified
if preset_name is not None:
    presetNode = volRenLogic.GetPresetByName(preset_name)
    if not presetNode:
        available = []
        for i in range(volRenLogic.GetPresetsScene().GetNumberOfNodes()):
            n = volRenLogic.GetPresetsScene().GetNthNode(i)
            if n and hasattr(n, 'GetName'):
                available.append(n.GetName())
        raise ValueError(
            'Preset not found: ' + preset_name
            + '. Available: ' + ', '.join(available[:20])
        )
    displayNode.GetVolumePropertyNode().Copy(presetNode)

displayNode.SetVisibility(visible)

result = {{
    'success': True,
    'volume_node_id': volumeNode.GetID(),
    'volume_node_name': volumeNode.GetName(),
    'display_node_id': displayNode.GetID(),
    'preset': preset_name,
    'visible': visible,
}}

__execResult = result
"""


def _build_set_volume_rendering_property_code(
    safe_node_id: str,
    safe_opacity_scale: str,
    safe_window: str,
    safe_level: str,
    safe_visible: str,
) -> str:
    """Build Python code to adjust volume rendering display properties.

    Args:
        safe_node_id: JSON-escaped volume node ID string
        safe_opacity_scale: JSON-escaped opacity scale float string (or "None")
        safe_window: JSON-escaped window float string (or "None")
        safe_level: JSON-escaped level float string (or "None")
        safe_visible: JSON-escaped visible boolean string (or "None")

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer

node_id = {safe_node_id}
opacity_scale = {safe_opacity_scale}
window = {safe_window}
level = {safe_level}
visible = {safe_visible}

volumeNode = slicer.mrmlScene.GetNodeByID(node_id)
if not volumeNode:
    raise ValueError('Volume node not found: ' + node_id)

volRenLogic = slicer.modules.volumerendering.logic()
displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
if not displayNode:
    raise ValueError('No volume rendering display node found for: ' + node_id
                     + '. Enable volume rendering first.')

changes = []

if opacity_scale is not None:
    vpNode = displayNode.GetVolumePropertyNode()
    opacity = vpNode.GetVolumeProperty().GetScalarOpacity()
    # Scale all opacity values
    for i in range(opacity.GetSize()):
        val = [0.0, 0.0, 0.0, 0.0]
        opacity.GetNodeValue(i, val)
        val[1] = val[1] * opacity_scale
        opacity.SetNodeValue(i, val)
    vpNode.Modified()
    changes.append('opacity_scale')

if window is not None and level is not None:
    displayNode.SetWindowLevel(window, level)
    changes.append('window_level')

if visible is not None:
    displayNode.SetVisibility(visible)
    changes.append('visibility')

result = {{
    'success': True,
    'volume_node_id': volumeNode.GetID(),
    'display_node_id': displayNode.GetID(),
    'changes_applied': changes,
}}

__execResult = result
"""


def _build_export_model_code(safe_node_id: str, safe_output_path: str, safe_format: str) -> str:
    """Build Python code to export a model node to STL/OBJ/PLY/VTK.

    Args:
        safe_node_id: JSON-escaped model node ID string
        safe_output_path: JSON-escaped full output file path string
        safe_format: JSON-escaped format string (STL, OBJ, PLY, VTK)

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import os

node_id = {safe_node_id}
output_path = {safe_output_path}
export_format = {safe_format}

modelNode = slicer.mrmlScene.GetNodeByID(node_id)
if not modelNode:
    raise ValueError('Model node not found: ' + node_id)

# Ensure polydata exists
polydata = modelNode.GetPolyData()
if not polydata or polydata.GetNumberOfPoints() == 0:
    raise ValueError('Model node has no mesh data: ' + node_id)

# Ensure directory exists
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

success = slicer.util.exportNode(modelNode, output_path)
if not success:
    raise ValueError('Failed to export model to: ' + output_path)

# Verify file was created
if not os.path.exists(output_path):
    raise ValueError('Export file was not created: ' + output_path)

file_size = os.path.getsize(output_path)

result = {{
    'success': True,
    'model_node_id': modelNode.GetID(),
    'model_node_name': modelNode.GetName(),
    'output_path': output_path,
    'format': export_format,
    'file_size_bytes': file_size,
    'point_count': polydata.GetNumberOfPoints(),
    'cell_count': polydata.GetNumberOfCells(),
}}

__execResult = result
"""


def _build_segmentation_to_models_code(safe_segmentation_id: str, safe_segment_ids: str) -> str:
    """Build Python code to convert segmentation segments to model nodes.

    Args:
        safe_segmentation_id: JSON-escaped segmentation node ID string
        safe_segment_ids: JSON-escaped list of segment IDs (or "None" for all visible)

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json

segmentation_id = {safe_segmentation_id}
segment_ids = {safe_segment_ids}

segNode = slicer.mrmlScene.GetNodeByID(segmentation_id)
if not segNode:
    raise ValueError('Segmentation node not found: ' + segmentation_id)

segmentation = segNode.GetSegmentation()
if not segmentation or segmentation.GetNumberOfSegments() == 0:
    raise ValueError('Segmentation has no segments: ' + segmentation_id)

# Determine which segments to export
if segment_ids is not None:
    export_ids = segment_ids
    for sid in export_ids:
        if not segmentation.GetSegment(sid):
            raise ValueError('Segment not found: ' + sid)
else:
    export_ids = []
    for i in range(segmentation.GetNumberOfSegments()):
        seg_id = segmentation.GetNthSegmentID(i)
        segment = segmentation.GetSegment(seg_id)
        if segment.GetTag('Segmentation.Status') != 'Removed':
            export_ids.append(seg_id)

if not export_ids:
    raise ValueError('No segments to export in: ' + segmentation_id)

# Create a folder node for organization
shFolderNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLFolderDisplayNode')
folderName = segNode.GetName() + '_models'
shFolderNode.SetName(folderName)

# Export segments to model nodes
models = []
for seg_id in export_ids:
    segment = segmentation.GetSegment(seg_id)

    # Ensure closed surface representation exists
    if not segmentation.ContainsRepresentation(
        slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
    ):
        segmentation.CreateRepresentation(
            slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
        )

    # Create model node
    modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    modelNode.SetName(segment.GetName())

    # Copy polydata from segment
    polydata = segmentation.GetSegment(seg_id).GetRepresentation(
        slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
    )
    if polydata:
        import vtk
        copiedPolyData = vtk.vtkPolyData()
        copiedPolyData.DeepCopy(polydata)
        modelNode.SetAndObservePolyData(copiedPolyData)
        modelNode.CreateDefaultDisplayNodes()

        # Apply segment color
        color = segment.GetColor()
        displayNode = modelNode.GetDisplayNode()
        if displayNode and color:
            displayNode.SetColor(color[0], color[1], color[2])

        models.append({{
            'segment_id': seg_id,
            'segment_name': segment.GetName(),
            'model_node_id': modelNode.GetID(),
            'model_node_name': modelNode.GetName(),
            'point_count': copiedPolyData.GetNumberOfPoints(),
            'cell_count': copiedPolyData.GetNumberOfCells(),
        }})
    else:
        # Clean up empty model
        slicer.mrmlScene.RemoveNode(modelNode)
        models.append({{
            'segment_id': seg_id,
            'segment_name': segment.GetName(),
            'model_node_id': None,
            'error': 'No closed surface representation available',
        }})

result = {{
    'success': True,
    'segmentation_node_id': segNode.GetID(),
    'segmentation_node_name': segNode.GetName(),
    'models': models,
    'model_count': len([m for m in models if m.get('model_node_id')]),
}}

__execResult = result
"""


def _build_capture_3d_view_code(
    safe_output_path: str, safe_width: str, safe_height: str, safe_view_id: str
) -> str:
    """Build Python code to capture a 3D view as an image.

    Args:
        safe_output_path: JSON-escaped output file path string
        safe_width: JSON-escaped width int string (or "None" for current)
        safe_height: JSON-escaped height int string (or "None" for current)
        safe_view_id: JSON-escaped view index int string (0 for first 3D view)

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import os

output_path = {safe_output_path}
width = {safe_width}
height = {safe_height}
view_id = {safe_view_id}

lm = slicer.app.layoutManager()
threeDWidget = lm.threeDWidget(view_id)
if not threeDWidget:
    raise ValueError('3D view not found at index: ' + str(view_id))

threeDView = threeDWidget.threeDView()
renderWindow = threeDView.renderWindow()

# Force render
slicer.util.forceRenderAllViews()

# Capture screenshot
import vtk
windowToImage = vtk.vtkWindowToImageFilter()
windowToImage.SetInput(renderWindow)

if width is not None and height is not None:
    windowToImage.SetScale(1)
    renderWindow.SetSize(width, height)
    slicer.util.forceRenderAllViews()

windowToImage.Update()

# Determine writer from extension
ext = os.path.splitext(output_path)[1].lower()
if ext == '.png':
    writer = vtk.vtkPNGWriter()
elif ext == '.jpg' or ext == '.jpeg':
    writer = vtk.vtkJPEGWriter()
elif ext == '.bmp':
    writer = vtk.vtkBMPWriter()
elif ext == '.tiff' or ext == '.tif':
    writer = vtk.vtkTIFFWriter()
else:
    writer = vtk.vtkPNGWriter()
    if not output_path.endswith('.png'):
        output_path = output_path + '.png'

# Ensure directory exists
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

writer.SetFileName(output_path)
writer.SetInputConnection(windowToImage.GetOutputPort())
writer.Write()

if not os.path.exists(output_path):
    raise ValueError('Screenshot was not saved: ' + output_path)

file_size = os.path.getsize(output_path)

result = {{
    'success': True,
    'output_path': output_path,
    'file_size_bytes': file_size,
    'view_index': view_id,
}}

__execResult = result
"""


# =============================================================================
# Public Tool Functions
# =============================================================================


def enable_volume_rendering(
    node_id: str,
    preset: str | None = None,
    visible: bool = True,
) -> dict:
    """Enable volume rendering on a volume node with optional preset.

    Args:
        node_id: MRML node ID of the scalar volume to render
        preset: Optional volume rendering preset name (e.g., "CT-Bone", "MR-Default")
        visible: Whether the volume rendering should be visible

    Returns:
        Dict with success, volume_node_id, display_node_id, preset, visible

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    node_id = validate_mrml_node_id(node_id)

    if preset is not None and preset not in VALID_VR_PRESETS:
        raise ValidationError(
            f"Invalid volume rendering preset '{preset}'. "
            f"Must be one of: {', '.join(sorted(VALID_VR_PRESETS))}",
            "preset",
            preset,
        )

    client = get_client()

    safe_node_id = json.dumps(node_id)
    safe_preset = json.dumps(preset) if preset is not None else "None"
    safe_visible = json.dumps(visible)

    python_code = _build_enable_volume_rendering_code(safe_node_id, safe_preset, safe_visible)

    try:
        exec_result = client.exec_python(python_code, timeout=VOLUME_RENDERING_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "enable volume rendering")

        logger.info(f"Volume rendering enabled: node={node_id}, preset={preset}, visible={visible}")

        return result

    except Exception:
        logger.error(f"Enable volume rendering failed for {node_id}")
        raise


def set_volume_rendering_property(
    node_id: str,
    opacity_scale: float | None = None,
    window: float | None = None,
    level: float | None = None,
    visible: bool | None = None,
) -> dict:
    """Adjust volume rendering display properties on a volume node.

    Args:
        node_id: MRML node ID of the volume with active volume rendering
        opacity_scale: Multiplier for all opacity values (0.0 to 10.0). 1.0 = no change.
        window: Window width for window/level adjustment
        level: Center level for window/level adjustment
        visible: Whether the volume rendering should be visible

    Returns:
        Dict with success, volume_node_id, display_node_id, changes_applied

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    node_id = validate_mrml_node_id(node_id)

    if opacity_scale is not None and not (0.0 <= opacity_scale <= 10.0):
        raise ValidationError(
            f"opacity_scale must be in range [0.0, 10.0], got {opacity_scale}",
            "opacity_scale",
            str(opacity_scale),
        )

    if (window is None) != (level is None):
        raise ValidationError(
            "window and level must both be provided or both omitted",
            "window/level",
            f"window={window}, level={level}",
        )

    if opacity_scale is None and window is None and visible is None:
        raise ValidationError(
            "At least one property must be specified (opacity_scale, window/level, or visible)",
            "properties",
            "none specified",
        )

    client = get_client()

    safe_node_id = json.dumps(node_id)
    safe_opacity_scale = json.dumps(opacity_scale) if opacity_scale is not None else "None"
    safe_window = json.dumps(window) if window is not None else "None"
    safe_level = json.dumps(level) if level is not None else "None"
    safe_visible = json.dumps(visible) if visible is not None else "None"

    python_code = _build_set_volume_rendering_property_code(
        safe_node_id, safe_opacity_scale, safe_window, safe_level, safe_visible
    )

    try:
        exec_result = client.exec_python(python_code, timeout=VOLUME_RENDERING_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "set volume rendering property")

        logger.info(f"Volume rendering property updated: node={node_id}")

        return result

    except Exception:
        logger.error(f"Set volume rendering property failed for {node_id}")
        raise


def export_model(
    node_id: str,
    output_directory: str,
    filename: str,
    file_format: str = "STL",
) -> dict:
    """Export a model node to a 3D mesh file (STL, OBJ, PLY, VTK).

    Args:
        node_id: MRML node ID of the model node to export
        output_directory: Directory where the file will be saved
        filename: Output filename (without extension)
        file_format: Export format - "STL", "OBJ", "PLY", or "VTK"

    Returns:
        Dict with success, model_node_id, output_path, format, file_size_bytes,
        point_count, cell_count

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    node_id = validate_mrml_node_id(node_id)
    validate_export_filename(filename)
    output_directory = validate_export_directory(output_directory)

    file_format_upper = file_format.upper()
    if file_format_upper not in VALID_EXPORT_FORMATS:
        raise ValidationError(
            f"Invalid export format '{file_format}'. "
            f"Must be one of: {', '.join(sorted(VALID_EXPORT_FORMATS))}",
            "file_format",
            file_format,
        )

    extension = EXPORT_FORMAT_EXTENSIONS[file_format_upper]
    output_path = os.path.join(output_directory, filename + extension)

    client = get_client()

    safe_node_id = json.dumps(node_id)
    safe_output_path = json.dumps(output_path)
    safe_format = json.dumps(file_format_upper)

    python_code = _build_export_model_code(safe_node_id, safe_output_path, safe_format)

    try:
        exec_result = client.exec_python(python_code, timeout=MODEL_EXPORT_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "export model")

        logger.info(
            f"Model exported: node={node_id}, path={output_path}, format={file_format_upper}"
        )

        return result

    except Exception:
        logger.error(f"Export model failed for {node_id}")
        raise


def segmentation_to_models(
    segmentation_node_id: str,
    segment_ids: list[str] | None = None,
) -> dict:
    """Convert segmentation segments to individual model nodes.

    Creates a vtkMRMLModelNode for each segment, copying the closed surface
    representation. If no segment_ids provided, exports all non-removed segments.

    Args:
        segmentation_node_id: MRML node ID of the segmentation node
        segment_ids: Optional list of specific segment IDs to convert.
            If None, all visible segments are converted.

    Returns:
        Dict with success, segmentation_node_id, models list, model_count

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    segmentation_node_id = validate_mrml_node_id(segmentation_node_id)

    client = get_client()

    safe_segmentation_id = json.dumps(segmentation_node_id)
    safe_segment_ids = json.dumps(segment_ids) if segment_ids is not None else "None"

    python_code = _build_segmentation_to_models_code(safe_segmentation_id, safe_segment_ids)

    try:
        exec_result = client.exec_python(python_code, timeout=MODEL_EXPORT_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "segmentation to models")

        logger.info(
            f"Segmentation converted to models: "
            f"segmentation={segmentation_node_id}, "
            f"model_count={result.get('model_count', 0)}"
        )

        return result

    except Exception:
        logger.error(f"Segmentation to models failed for {segmentation_node_id}")
        raise


def capture_3d_view(
    output_path: str,
    width: int | None = None,
    height: int | None = None,
    view_index: int = 0,
) -> dict:
    """Capture a screenshot of a 3D view to an image file.

    Args:
        output_path: Full output file path (supports .png, .jpg, .bmp, .tiff)
        width: Optional capture width in pixels (default: current view size)
        height: Optional capture height in pixels (default: current view size)
        view_index: Index of the 3D view to capture (default: 0 for first)

    Returns:
        Dict with success, output_path, file_size_bytes, view_index

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    if not output_path:
        raise ValidationError("Output path cannot be empty", "output_path", output_path or "")

    if len(output_path) > MAX_FOLDER_PATH_LENGTH:
        raise ValidationError(
            f"Output path exceeds maximum length ({MAX_FOLDER_PATH_LENGTH})",
            "output_path",
            output_path[:50] + "...",
        )

    # Check for path traversal
    path_parts = output_path.replace("\\", "/").split("/")
    for part in path_parts:
        if part == "..":
            raise ValidationError(
                "Output path contains forbidden component: '..'",
                "output_path",
                output_path,
            )

    if (width is None) != (height is None):
        raise ValidationError(
            "width and height must both be provided or both omitted",
            "dimensions",
            f"width={width}, height={height}",
        )

    if width is not None:
        if not (1 <= width <= 8192):
            raise ValidationError(
                f"width must be in range [1, 8192], got {width}",
                "width",
                str(width),
            )
        if not (1 <= height <= 8192):  # type: ignore[operator]
            raise ValidationError(
                f"height must be in range [1, 8192], got {height}",
                "height",
                str(height),
            )

    if not (0 <= view_index <= 10):
        raise ValidationError(
            f"view_index must be in range [0, 10], got {view_index}",
            "view_index",
            str(view_index),
        )

    client = get_client()

    safe_output_path = json.dumps(output_path)
    safe_width = json.dumps(width) if width is not None else "None"
    safe_height = json.dumps(height) if height is not None else "None"
    safe_view_id = json.dumps(view_index)

    python_code = _build_capture_3d_view_code(
        safe_output_path, safe_width, safe_height, safe_view_id
    )

    try:
        exec_result = client.exec_python(python_code, timeout=VOLUME_RENDERING_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "capture 3D view")

        logger.info(f"3D view captured: path={output_path}")

        return result

    except Exception:
        logger.error(f"Capture 3D view failed: path={output_path}")
        raise
