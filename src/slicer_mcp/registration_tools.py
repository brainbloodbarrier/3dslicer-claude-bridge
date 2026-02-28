"""MCP tool implementations for registration and landmark management."""

import json
import logging
import re

from slicer_mcp.constants import (
    DEFAULT_SAMPLING_PERCENTAGE,
    LANDMARK_LABEL_PATTERN,
    MAX_LANDMARK_LABEL_LENGTH,
    MAX_LANDMARKS,
    REGISTRATION_TIMEOUT,
    VALID_INIT_MODES,
    VALID_LANDMARK_TRANSFORM_TYPES,
    VALID_TRANSFORM_TYPES,
)
from slicer_mcp.slicer_client import get_client
from slicer_mcp.tools import ValidationError, _parse_json_result, validate_mrml_node_id

logger = logging.getLogger("slicer-mcp")

# Compiled pattern for landmark labels
LANDMARK_LABEL_COMPILED = re.compile(LANDMARK_LABEL_PATTERN)


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_landmark_label(label: str) -> str:
    """Validate a landmark label against the allowed pattern.

    Args:
        label: Landmark label string to validate

    Returns:
        The validated label (unchanged if valid)

    Raises:
        ValidationError: If label is empty, too long, or contains invalid characters
    """
    if not label:
        raise ValidationError("Landmark label cannot be empty", "label", label or "")

    if len(label) > MAX_LANDMARK_LABEL_LENGTH:
        raise ValidationError(
            f"Landmark label exceeds maximum length ({MAX_LANDMARK_LABEL_LENGTH})",
            "label",
            label[:50] + "...",
        )

    if not LANDMARK_LABEL_COMPILED.match(label):
        raise ValidationError(
            f"Invalid landmark label format. Must match pattern {LANDMARK_LABEL_PATTERN}. "
            f"Got: '{label[:50]}'",
            "label",
            label,
        )

    return label


def validate_points(points: list[list[float]]) -> list[list[float]]:
    """Validate a list of 3D point coordinates.

    Each point must be a list of exactly 3 numeric values (floats or ints).

    Args:
        points: List of [x, y, z] coordinate triples

    Returns:
        The validated points list

    Raises:
        ValidationError: If points list is empty, too large, or contains invalid entries
    """
    if not points:
        raise ValidationError("Points list cannot be empty", "points", "[]")

    if len(points) > MAX_LANDMARKS:
        raise ValidationError(
            f"Too many points ({len(points)}). Maximum is {MAX_LANDMARKS}",
            "points",
            f"<{len(points)} points>",
        )

    for i, point in enumerate(points):
        if not isinstance(point, (list, tuple)):
            raise ValidationError(
                f"Point at index {i} must be a list of 3 floats, got {type(point).__name__}",
                "points",
                str(point)[:50],
            )
        if len(point) != 3:
            raise ValidationError(
                f"Point at index {i} must have exactly 3 coordinates (x, y, z), got {len(point)}",
                "points",
                str(point)[:50],
            )
        for j, coord in enumerate(point):
            if not isinstance(coord, (int, float)):
                raise ValidationError(
                    f"Coordinate {j} of point {i} must be numeric, " f"got {type(coord).__name__}",
                    "points",
                    str(coord)[:50],
                )

    return points


# =============================================================================
# Codegen Functions
# =============================================================================


def _build_place_landmarks_code(safe_name: str, safe_points: str, safe_labels: str) -> str:
    """Build Python code to create a markups fiducial node with control points.

    Args:
        safe_name: JSON-escaped name string
        safe_points: JSON-escaped points list string
        safe_labels: JSON-escaped labels list string (or "None")

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json

name = {safe_name}
points = {safe_points}
labels = {safe_labels}

fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
fiducialNode.SetName(name)

for i, pt in enumerate(points):
    fiducialNode.AddControlPoint(pt[0], pt[1], pt[2])
    if labels is not None:
        fiducialNode.SetNthControlPointLabel(i, labels[i])

result = {{
    'success': True,
    'node_id': fiducialNode.GetID(),
    'node_name': fiducialNode.GetName(),
    'point_count': fiducialNode.GetNumberOfControlPoints()
}}

__execResult = result
"""


def _build_get_landmarks_code(safe_node_id: str) -> str:
    """Build Python code to retrieve control points from a markup node.

    Args:
        safe_node_id: JSON-escaped node ID string

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json

node_id = {safe_node_id}

node = slicer.mrmlScene.GetNodeByID(node_id)
if not node:
    raise ValueError('Markup node not found: ' + node_id)

points = []
for i in range(node.GetNumberOfControlPoints()):
    pos = [0.0, 0.0, 0.0]
    node.GetNthControlPointPosition(i, pos)
    points.append({{
        'index': i,
        'label': node.GetNthControlPointLabel(i),
        'position_ras': list(pos)
    }})

result = {{
    'success': True,
    'node_id': node.GetID(),
    'node_name': node.GetName(),
    'point_count': node.GetNumberOfControlPoints(),
    'points': points
}}

__execResult = result
"""


def _build_register_volumes_code(
    safe_fixed_id: str,
    safe_moving_id: str,
    safe_transform_type: str,
    safe_init_mode: str,
    safe_sampling: str,
    safe_histogram: str,
    safe_create_resampled: str,
) -> str:
    """Build Python code for BRAINSFit intensity-based registration.

    Args:
        safe_fixed_id: JSON-escaped fixed volume node ID
        safe_moving_id: JSON-escaped moving volume node ID
        safe_transform_type: JSON-escaped transform type string
        safe_init_mode: JSON-escaped initialization mode string
        safe_sampling: JSON-escaped sampling percentage string
        safe_histogram: JSON-escaped histogram match boolean string
        safe_create_resampled: JSON-escaped create resampled boolean string

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json

fixed_id = {safe_fixed_id}
moving_id = {safe_moving_id}
transform_type = {safe_transform_type}
init_mode = {safe_init_mode}
sampling_pct = {safe_sampling}
histogram_match = {safe_histogram}
create_resampled = {safe_create_resampled}

fixedVolume = slicer.mrmlScene.GetNodeByID(fixed_id)
if not fixedVolume:
    raise ValueError('Fixed volume not found: ' + fixed_id)

movingVolume = slicer.mrmlScene.GetNodeByID(moving_id)
if not movingVolume:
    raise ValueError('Moving volume not found: ' + moving_id)

# Create transform node (BSpline vs Linear)
if transform_type == 'BSpline':
    transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLBSplineTransformNode')
else:
    transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
transformNode.SetName(f'{{movingVolume.GetName()}}_to_{{fixedVolume.GetName()}}_{{transform_type}}')

# Build BRAINSFit parameters
parameters = {{
    'fixedVolume': fixedVolume.GetID(),
    'movingVolume': movingVolume.GetID(),
    'outputTransform': transformNode.GetID(),
    'initializeTransformMode': init_mode,
    'samplingPercentage': sampling_pct,
    'histogramMatch': histogram_match,
}}

# Set transform-specific flags
if transform_type == 'Rigid':
    parameters['useRigid'] = True
elif transform_type == 'ScaleVersor3D':
    parameters['useScaleVersor3D'] = True
elif transform_type == 'ScaleSkewVersor3D':
    parameters['useScaleSkewVersor3D'] = True
elif transform_type == 'Affine':
    parameters['useAffine'] = True
elif transform_type == 'BSpline':
    parameters['useBSpline'] = True

# Optionally create resampled output volume
resampledNode = None
if create_resampled:
    resampledNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    resampledNode.SetName(movingVolume.GetName() + '_resampled')
    parameters['outputVolume'] = resampledNode.GetID()

cliNode = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)

if cliNode.GetStatus() & cliNode.ErrorsMask:
    error_text = cliNode.GetErrorText()
    slicer.mrmlScene.RemoveNode(cliNode)
    raise ValueError('BRAINSFit registration failed: ' + error_text)

slicer.mrmlScene.RemoveNode(cliNode)

result = {{
    'success': True,
    'transform_node_id': transformNode.GetID(),
    'transform_node_name': transformNode.GetName(),
    'transform_type': transform_type,
    'resampled_node_id': resampledNode.GetID() if resampledNode else None,
}}

__execResult = result
"""


def _build_register_landmarks_code(
    safe_fixed_fids: str, safe_moving_fids: str, safe_transform_type: str
) -> str:
    """Build Python code for landmark-based registration.

    Args:
        safe_fixed_fids: JSON-escaped fixed landmarks node ID
        safe_moving_fids: JSON-escaped moving landmarks node ID
        safe_transform_type: JSON-escaped transform type string

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json

fixed_fids_id = {safe_fixed_fids}
moving_fids_id = {safe_moving_fids}
transform_type = {safe_transform_type}

fixedFids = slicer.mrmlScene.GetNodeByID(fixed_fids_id)
if not fixedFids:
    raise ValueError('Fixed landmarks not found: ' + fixed_fids_id)

movingFids = slicer.mrmlScene.GetNodeByID(moving_fids_id)
if not movingFids:
    raise ValueError('Moving landmarks not found: ' + moving_fids_id)

# Create output transform
transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
transformNode.SetName(f'{{movingFids.GetName()}}_to_{{fixedFids.GetName()}}_{{transform_type}}')

parameters = {{
    'fixedLandmarks': fixedFids.GetID(),
    'movingLandmarks': movingFids.GetID(),
    'saveTransform': transformNode.GetID(),
    'transformType': transform_type,
}}

cliNode = slicer.cli.runSync(slicer.modules.fiducialregistration, None, parameters)

if cliNode.GetStatus() & cliNode.ErrorsMask:
    error_text = cliNode.GetErrorText()
    slicer.mrmlScene.RemoveNode(cliNode)
    raise ValueError('Fiducial registration failed: ' + error_text)

slicer.mrmlScene.RemoveNode(cliNode)

result = {{
    'success': True,
    'transform_node_id': transformNode.GetID(),
    'transform_node_name': transformNode.GetName(),
    'transform_type': transform_type,
}}

__execResult = result
"""


def _build_apply_transform_code(safe_node_id: str, safe_transform_id: str, harden: bool) -> str:
    """Build Python code to apply a transform to a node.

    Args:
        safe_node_id: JSON-escaped target node ID
        safe_transform_id: JSON-escaped transform node ID
        harden: Whether to harden the transform after applying

    Returns:
        Python code string for execution in Slicer
    """
    safe_harden = json.dumps(harden)
    return f"""
import slicer
import json

node_id = {safe_node_id}
transform_id = {safe_transform_id}
harden = {safe_harden}

node = slicer.mrmlScene.GetNodeByID(node_id)
if not node:
    raise ValueError('Node not found: ' + node_id)

transformNode = slicer.mrmlScene.GetNodeByID(transform_id)
if not transformNode:
    raise ValueError('Transform node not found: ' + transform_id)

node.SetAndObserveTransformNodeID(transformNode.GetID())

hardened = False
if harden:
    node.HardenTransform()
    hardened = True

result = {{
    'success': True,
    'node_id': node.GetID(),
    'transform_node_id': transformNode.GetID(),
    'hardened': hardened,
}}

__execResult = result
"""


# =============================================================================
# Public Tool Functions
# =============================================================================


def place_landmarks(
    name: str,
    points: list[list[float]],
    labels: list[str] | None = None,
) -> dict:
    """Create a vtkMRMLMarkupsFiducialNode with the given control points.

    Args:
        name: Name for the new fiducial node
        points: List of [x, y, z] coordinate triples
        labels: Optional list of labels for each point (must match points length)

    Returns:
        Dict with success, node_id, node_name, point_count

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    # Validate name
    if not name:
        raise ValidationError("Landmark name cannot be empty", "name", name or "")

    if len(name) > MAX_LANDMARK_LABEL_LENGTH:
        raise ValidationError(
            f"Landmark name exceeds maximum length ({MAX_LANDMARK_LABEL_LENGTH})",
            "name",
            name[:50] + "...",
        )

    # Validate points
    validate_points(points)

    # Validate labels if provided
    if labels is not None:
        if len(labels) != len(points):
            raise ValidationError(
                f"Labels length ({len(labels)}) must match points length ({len(points)})",
                "labels",
                f"<{len(labels)} labels>",
            )
        for label in labels:
            validate_landmark_label(label)

    client = get_client()

    safe_name = json.dumps(name)
    safe_points = json.dumps(points)
    safe_labels = json.dumps(labels) if labels is not None else "None"

    python_code = _build_place_landmarks_code(safe_name, safe_points, safe_labels)

    try:
        exec_result = client.exec_python(python_code, timeout=REGISTRATION_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "place landmarks")

        logger.info(f"Landmarks placed: name={name}, points={len(points)}")

        return result

    except Exception:
        logger.error(f"Place landmarks failed for '{name}'")
        raise


def get_landmarks(node_id: str) -> dict:
    """Retrieve all control points from a markup node.

    Args:
        node_id: MRML node ID of the markup fiducial node

    Returns:
        Dict with success, node_id, node_name, point_count, points

    Raises:
        ValidationError: If node_id format is invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    node_id = validate_mrml_node_id(node_id)

    client = get_client()

    safe_node_id = json.dumps(node_id)

    python_code = _build_get_landmarks_code(safe_node_id)

    try:
        exec_result = client.exec_python(python_code, timeout=REGISTRATION_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "get landmarks")

        logger.info(f"Retrieved landmarks from {node_id}: {result.get('point_count', 0)} points")

        return result

    except Exception:
        logger.error(f"Get landmarks failed for {node_id}")
        raise


def register_volumes(
    fixed_node_id: str,
    moving_node_id: str,
    transform_type: str = "Rigid",
    init_mode: str = "useMomentsAlign",
    sampling_percentage: float = DEFAULT_SAMPLING_PERCENTAGE,
    histogram_match: bool = False,
    create_resampled: bool = False,
) -> dict:
    """Run BRAINSFit intensity-based registration between two volumes.

    Args:
        fixed_node_id: MRML node ID of the fixed (reference) volume
        moving_node_id: MRML node ID of the moving volume to register
        transform_type: Type of transform to compute
        init_mode: Initialization mode for alignment
        sampling_percentage: Fraction of voxels to sample (0 < x <= 1.0)
        histogram_match: Whether to apply histogram matching
        create_resampled: Whether to create a resampled output volume

    Returns:
        Dict with success, transform_node_id, transform_node_name,
        transform_type, resampled_node_id

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    fixed_node_id = validate_mrml_node_id(fixed_node_id)
    moving_node_id = validate_mrml_node_id(moving_node_id)

    if transform_type not in VALID_TRANSFORM_TYPES:
        raise ValidationError(
            f"Invalid transform_type '{transform_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSFORM_TYPES))}",
            "transform_type",
            transform_type,
        )

    if init_mode not in VALID_INIT_MODES:
        raise ValidationError(
            f"Invalid init_mode '{init_mode}'. "
            f"Must be one of: {', '.join(sorted(VALID_INIT_MODES))}",
            "init_mode",
            init_mode,
        )

    if not (0 < sampling_percentage <= 1.0):
        raise ValidationError(
            f"sampling_percentage must be in range (0, 1.0], got {sampling_percentage}",
            "sampling_percentage",
            str(sampling_percentage),
        )

    client = get_client()

    safe_fixed_id = json.dumps(fixed_node_id)
    safe_moving_id = json.dumps(moving_node_id)
    safe_transform_type = json.dumps(transform_type)
    safe_init_mode = json.dumps(init_mode)
    safe_sampling = json.dumps(sampling_percentage)
    safe_histogram = json.dumps(histogram_match)
    safe_create_resampled = json.dumps(create_resampled)

    python_code = _build_register_volumes_code(
        safe_fixed_id,
        safe_moving_id,
        safe_transform_type,
        safe_init_mode,
        safe_sampling,
        safe_histogram,
        safe_create_resampled,
    )

    try:
        exec_result = client.exec_python(python_code, timeout=REGISTRATION_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "register volumes")

        logger.info(
            f"Volume registration completed: type={transform_type}, "
            f"transform={result.get('transform_node_id')}"
        )

        return result

    except Exception:
        logger.error(
            f"Volume registration failed: fixed={fixed_node_id}, " f"moving={moving_node_id}"
        )
        raise


def register_landmarks(
    fixed_landmarks_id: str,
    moving_landmarks_id: str,
    transform_type: str = "Rigid",
) -> dict:
    """Run landmark-based registration between two fiducial node sets.

    Args:
        fixed_landmarks_id: MRML node ID of the fixed landmarks
        moving_landmarks_id: MRML node ID of the moving landmarks
        transform_type: Type of transform to compute

    Returns:
        Dict with success, transform_node_id, transform_node_name, transform_type

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    fixed_landmarks_id = validate_mrml_node_id(fixed_landmarks_id)
    moving_landmarks_id = validate_mrml_node_id(moving_landmarks_id)

    if transform_type not in VALID_LANDMARK_TRANSFORM_TYPES:
        raise ValidationError(
            f"Invalid transform_type '{transform_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_LANDMARK_TRANSFORM_TYPES))}",
            "transform_type",
            transform_type,
        )

    client = get_client()

    safe_fixed_fids = json.dumps(fixed_landmarks_id)
    safe_moving_fids = json.dumps(moving_landmarks_id)
    safe_transform_type = json.dumps(transform_type)

    python_code = _build_register_landmarks_code(
        safe_fixed_fids, safe_moving_fids, safe_transform_type
    )

    try:
        exec_result = client.exec_python(python_code, timeout=REGISTRATION_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "register landmarks")

        logger.info(
            f"Landmark registration completed: type={transform_type}, "
            f"transform={result.get('transform_node_id')}"
        )

        return result

    except Exception:
        logger.error(
            f"Landmark registration failed: fixed={fixed_landmarks_id}, "
            f"moving={moving_landmarks_id}"
        )
        raise


def apply_transform(
    node_id: str,
    transform_node_id: str,
    harden: bool = False,
) -> dict:
    """Apply (or harden) a transform to any transformable node.

    Args:
        node_id: MRML node ID of the node to transform
        transform_node_id: MRML node ID of the transform to apply
        harden: If True, harden the transform after applying

    Returns:
        Dict with success, node_id, transform_node_id, hardened

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable
    """
    node_id = validate_mrml_node_id(node_id)
    transform_node_id = validate_mrml_node_id(transform_node_id)

    client = get_client()

    safe_node_id = json.dumps(node_id)
    safe_transform_id = json.dumps(transform_node_id)

    python_code = _build_apply_transform_code(safe_node_id, safe_transform_id, harden)

    try:
        exec_result = client.exec_python(python_code, timeout=REGISTRATION_TIMEOUT)

        result = _parse_json_result(exec_result.get("result", ""), "apply transform")

        logger.info(
            f"Transform applied: node={node_id}, transform={transform_node_id}, "
            f"hardened={harden}"
        )

        return result

    except Exception:
        logger.error(f"Apply transform failed: node={node_id}, transform={transform_node_id}")
        raise
