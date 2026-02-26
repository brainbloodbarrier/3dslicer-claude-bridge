"""Cervical screw instrumentation planning tools.

Implements 6 cervical fixation techniques with unified planning interface:
1. Pedicle screws (C2-C7)
2. Lateral mass screws (C3-C7) — Roy-Camille, Magerl, An, Anderson variants
3. Transarticular screws (C1-C2) — Magerl technique
4. C1 lateral mass screws — Harms/Goel technique
5. C2 pars interarticularis screws
6. Occipital screws — thickness mapping + keel identification

References:
    Roy-Camille R et al. 1979; Magerl F et al. 1987; An HS et al. 1991;
    Harms J, Melcher RP. 2001; Goel A, Laheri VK. 1994.
"""

import json
import logging
from typing import Any

from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    C1_LATERAL_MASS_SCREW_DEFAULTS,
    C2_PARS_SCREW_DEFAULTS,
    CERVICAL_LATERAL_MASS_SCREW_DEFAULTS,
    CERVICAL_PEDICLE_MIN_WIDTH_MM,
    CERVICAL_PEDICLE_SCREW_DEFAULTS,
    INSTRUMENTATION_TIMEOUT,
    ISTHMUS_MIN_HEIGHT_MM,
    ISTHMUS_MIN_WIDTH_MM,
    OCCIPITAL_MIN_THICKNESS_MM,
    OCCIPITAL_SCREW_DEFAULTS,
    TECHNIQUE_ANGULATION,
    TECHNIQUE_VALID_LEVELS,
    TRANSARTICULAR_SCREW_DEFAULTS,
    VA_SAFETY_DISTANCE_MM,
    VALID_INSTRUMENTATION_TECHNIQUES,
    VALID_LATERAL_MASS_VARIANTS,
    VALID_SIDES,
)
from slicer_mcp.tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# Technique-to-defaults mapping
# =============================================================================

_TECHNIQUE_SCREW_DEFAULTS: dict[str, dict[str, float]] = {
    "pedicle": CERVICAL_PEDICLE_SCREW_DEFAULTS,
    "lateral_mass": CERVICAL_LATERAL_MASS_SCREW_DEFAULTS,
    "transarticular": TRANSARTICULAR_SCREW_DEFAULTS,
    "c1_lateral_mass": C1_LATERAL_MASS_SCREW_DEFAULTS,
    "c2_pars": C2_PARS_SCREW_DEFAULTS,
    "occipital": OCCIPITAL_SCREW_DEFAULTS,
}


# =============================================================================
# Input Validation
# =============================================================================


def _validate_technique(technique: str) -> str:
    """Validate instrumentation technique name.

    Args:
        technique: Technique name to validate.

    Returns:
        Validated technique name (lowercase).

    Raises:
        ValidationError: If technique is not recognized.
    """
    if not technique:
        raise ValidationError("Technique cannot be empty", "technique", "")
    technique = technique.lower().strip()
    if technique not in VALID_INSTRUMENTATION_TECHNIQUES:
        raise ValidationError(
            f"Invalid technique '{technique}'. "
            f"Must be one of: {', '.join(sorted(VALID_INSTRUMENTATION_TECHNIQUES))}",
            "technique",
            technique,
        )
    return technique


def _validate_level(technique: str, level: str) -> str:
    """Validate vertebral level for the given technique.

    Args:
        technique: Already-validated technique name.
        level: Vertebral level string (e.g. "C5", "C1C2", "Occiput").

    Returns:
        Validated level string.

    Raises:
        ValidationError: If level is invalid for the technique.
    """
    if not level:
        raise ValidationError("Level cannot be empty", "level", "")
    level = level.strip()
    if technique == "auto":
        return level
    valid_levels = TECHNIQUE_VALID_LEVELS.get(technique, frozenset())
    if level not in valid_levels:
        raise ValidationError(
            f"Invalid level '{level}' for technique '{technique}'. "
            f"Must be one of: {', '.join(sorted(valid_levels))}",
            "level",
            level,
        )
    return level


def _validate_side(side: str) -> str:
    """Validate side parameter.

    Args:
        side: Side string ("left", "right", "bilateral").

    Returns:
        Validated side string (lowercase).

    Raises:
        ValidationError: If side is invalid.
    """
    if not side:
        raise ValidationError("Side cannot be empty", "side", "")
    side = side.lower().strip()
    if side not in VALID_SIDES:
        raise ValidationError(
            f"Invalid side '{side}'. Must be one of: {', '.join(sorted(VALID_SIDES))}",
            "side",
            side,
        )
    return side


def _validate_variant(variant: str | None, technique: str) -> str | None:
    """Validate lateral mass variant.

    Args:
        variant: Variant name or None.
        technique: Already-validated technique.

    Returns:
        Validated variant or None.

    Raises:
        ValidationError: If variant is invalid.
    """
    if variant is None:
        if technique == "lateral_mass":
            return "magerl"  # default variant
        return None
    variant = variant.lower().strip()
    if technique != "lateral_mass":
        raise ValidationError(
            f"Variant parameter only applies to 'lateral_mass' technique, " f"not '{technique}'",
            "variant",
            variant,
        )
    if variant not in VALID_LATERAL_MASS_VARIANTS:
        raise ValidationError(
            f"Invalid variant '{variant}'. "
            f"Must be one of: {', '.join(sorted(VALID_LATERAL_MASS_VARIANTS))}",
            "variant",
            variant,
        )
    return variant


def _validate_screw_dimensions(
    technique: str,
    screw_diameter_mm: float | None,
    screw_length_mm: float | None,
) -> tuple[float, float]:
    """Validate and resolve screw dimensions.

    Args:
        technique: Already-validated technique name.
        screw_diameter_mm: Override diameter or None for default.
        screw_length_mm: Override length or None for default.

    Returns:
        Tuple of (diameter_mm, length_mm).

    Raises:
        ValidationError: If dimensions are out of safe range.
    """
    defaults = _TECHNIQUE_SCREW_DEFAULTS.get(technique, CERVICAL_PEDICLE_SCREW_DEFAULTS)
    diameter = screw_diameter_mm if screw_diameter_mm is not None else defaults["diameter_mm"]
    length = screw_length_mm if screw_length_mm is not None else defaults["length_mm"]

    if diameter <= 0 or diameter > 8.0:
        raise ValidationError(
            f"Screw diameter {diameter}mm is out of safe range (0-8mm)",
            "screw_diameter_mm",
            str(diameter),
        )
    if length <= 0 or length > 60.0:
        raise ValidationError(
            f"Screw length {length}mm is out of safe range (0-60mm)",
            "screw_length_mm",
            str(length),
        )
    min_len = defaults.get("min_length_mm", 0)
    max_len = defaults.get("max_length_mm", 60)
    if length < min_len or length > max_len:
        raise ValidationError(
            f"Screw length {length}mm is outside typical range for {technique} "
            f"({min_len}-{max_len}mm)",
            "screw_length_mm",
            str(length),
        )
    return diameter, length


# =============================================================================
# Slicer Python Code — Shared Helpers
# =============================================================================

_SLICER_HELPERS = """
import slicer
import json
import numpy as np
import vtk
import math


def _find_segment(seg_node, level):
    \"\"\"Find segment matching vertebral level in segmentation node.\"\"\"
    segmentation = seg_node.GetSegmentation()
    # Try TotalSegmentator naming first (e.g., 'vertebrae_C5')
    ts_name = 'vertebrae_' + level
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        name = seg.GetName()
        if name == ts_name or name == level or name.upper() == level.upper():
            return segmentation.GetNthSegmentID(i), name
    # Try case-insensitive partial match
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        name = seg.GetName()
        if level.upper() in name.upper():
            return segmentation.GetNthSegmentID(i), name
    raise ValueError(f'Segment for level {level} not found in segmentation')


def _get_segment_centroid_and_bounds(seg_node, segment_id):
    \"\"\"Compute centroid and bounds of a segment in RAS coordinates.\"\"\"
    labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, segment_id)
    if labelmap is None or labelmap.sum() == 0:
        raise ValueError(f'Empty segment: {segment_id}')

    coords = np.argwhere(labelmap > 0)
    centroid_kji = coords.mean(axis=0)
    min_kji = coords.min(axis=0)
    max_kji = coords.max(axis=0)

    # Get IJK-to-RAS transform from reference volume
    ref_vol = seg_node.GetNodeReference('referenceImageGeometryRef')
    ijk_to_ras = vtk.vtkMatrix4x4()
    if ref_vol:
        ref_vol.GetIJKToRASMatrix(ijk_to_ras)
    else:
        seg_node.GetIJKToRASMatrix(ijk_to_ras)

    def kji_to_ras(kji):
        p = [float(kji[2]), float(kji[1]), float(kji[0]), 1.0]
        r = [0.0, 0.0, 0.0, 1.0]
        ijk_to_ras.MultiplyPoint(p, r)
        return r[:3]

    centroid_ras = kji_to_ras(centroid_kji)
    min_ras = kji_to_ras(min_kji)
    max_ras = kji_to_ras(max_kji)

    # Compute bounding box dimensions in RAS
    dims = [abs(max_ras[i] - min_ras[i]) for i in range(3)]

    return {
        'centroid_ras': centroid_ras,
        'min_ras': min_ras,
        'max_ras': max_ras,
        'dimensions_mm': dims,
        'voxel_count': int(coords.shape[0]),
    }


def _compute_trajectory_endpoint(entry_ras, medial_deg, cephalad_deg,
                                  length_mm, side='left'):
    \"\"\"Compute trajectory endpoint from entry point, angles, and length.

    Anatomical convention:
      R = +x, A = +y, S = +z (RAS)
      medial = toward midline (left side: +x, right side: -x)
      cephalad = toward head (+z)
      anterior = +y
    \"\"\"
    med_rad = math.radians(medial_deg)
    ceph_rad = math.radians(cephalad_deg)

    # Base direction: anterior (+y)
    # Apply medial angulation in axial plane
    sign = 1.0 if side == 'left' else -1.0
    dx = sign * math.sin(med_rad) * math.cos(ceph_rad) * length_mm
    dy = math.cos(med_rad) * math.cos(ceph_rad) * length_mm
    dz = math.sin(ceph_rad) * length_mm

    return [
        entry_ras[0] + dx,
        entry_ras[1] + dy,
        entry_ras[2] + dz,
    ]


def _min_distance_to_segment(seg_node, segment_id, point_ras):
    \"\"\"Compute minimum distance from a point to segment surface (mm).\"\"\"
    labelmap = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, segment_id)
    if labelmap is None or labelmap.sum() == 0:
        return float('inf')

    coords = np.argwhere(labelmap > 0)

    ref_vol = seg_node.GetNodeReference('referenceImageGeometryRef')
    ijk_to_ras = vtk.vtkMatrix4x4()
    if ref_vol:
        ref_vol.GetIJKToRASMatrix(ijk_to_ras)
    else:
        seg_node.GetIJKToRASMatrix(ijk_to_ras)

    # Sample up to 5000 boundary voxels for performance
    if coords.shape[0] > 5000:
        idx = np.random.choice(coords.shape[0], 5000, replace=False)
        coords = coords[idx]

    point = np.array(point_ras)
    min_dist = float('inf')
    for kji in coords:
        p = [float(kji[2]), float(kji[1]), float(kji[0]), 1.0]
        r = [0.0, 0.0, 0.0, 1.0]
        ijk_to_ras.MultiplyPoint(p, r)
        d = np.linalg.norm(np.array(r[:3]) - point)
        if d < min_dist:
            min_dist = d
    return float(min_dist)


def _trajectory_min_distance_to_segment(seg_node, segment_id,
                                         entry_ras, target_ras,
                                         n_samples=20):
    \"\"\"Compute min distance from trajectory line to segment surface.\"\"\"
    entry = np.array(entry_ras)
    target = np.array(target_ras)
    min_dist = float('inf')
    for i in range(n_samples + 1):
        t = i / float(n_samples)
        point = entry + t * (target - entry)
        d = _min_distance_to_segment(seg_node, segment_id, point.tolist())
        if d < min_dist:
            min_dist = d
    return float(min_dist)


def _create_screw_visualization(entry_ras, target_ras, diameter_mm,
                                 color_rgb, screw_name):
    \"\"\"Create line markup and cylinder model for screw visualization.\"\"\"
    # Create markup line
    line_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode')
    line_node.SetName(screw_name + '_trajectory')
    line_node.AddControlPoint(entry_ras[0], entry_ras[1], entry_ras[2])
    line_node.AddControlPoint(target_ras[0], target_ras[1], target_ras[2])

    # Set line color
    display = line_node.GetDisplayNode()
    if display:
        display.SetSelectedColor(color_rgb[0], color_rgb[1], color_rgb[2])
        display.SetColor(color_rgb[0], color_rgb[1], color_rgb[2])

    # Create cylinder model for screw body
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(diameter_mm / 2.0)
    entry = np.array(entry_ras)
    target = np.array(target_ras)
    length = float(np.linalg.norm(target - entry))
    cylinder.SetHeight(length)
    cylinder.SetResolution(16)
    cylinder.Update()

    # Compute transform to align cylinder with trajectory
    direction = target - entry
    direction = direction / np.linalg.norm(direction)
    midpoint = (entry + target) / 2.0

    # VTK cylinder is along Y axis by default
    y_axis = np.array([0, 1, 0])
    rot_axis = np.cross(y_axis, direction)
    rot_axis_len = np.linalg.norm(rot_axis)

    transform = vtk.vtkTransform()
    transform.Translate(midpoint[0], midpoint[1], midpoint[2])
    if rot_axis_len > 1e-6:
        rot_axis = rot_axis / rot_axis_len
        angle = math.degrees(math.acos(np.clip(np.dot(y_axis, direction), -1, 1)))
        transform.RotateWXYZ(angle, rot_axis[0], rot_axis[1], rot_axis[2])

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(transform)
    tf.SetInputData(cylinder.GetOutput())
    tf.Update()

    model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    model_node.SetName(screw_name + '_model')
    model_node.SetAndObservePolyData(tf.GetOutput())
    model_node.CreateDefaultDisplayNodes()

    model_display = model_node.GetDisplayNode()
    if model_display:
        model_display.SetColor(color_rgb[0], color_rgb[1], color_rgb[2])
        model_display.SetOpacity(0.6)

    return {
        'markup_node_id': line_node.GetID(),
        'model_node_id': model_node.GetID(),
    }


def _safety_color(distance_mm, threshold_mm):
    \"\"\"Return safety color based on distance vs threshold.\"\"\"
    if distance_mm >= threshold_mm * 2:
        return 'green'
    elif distance_mm >= threshold_mm:
        return 'yellow'
    else:
        return 'red'
"""


# =============================================================================
# Technique-Specific Code Builders
# =============================================================================


def _build_pedicle_code(
    safe_seg_id: str,
    safe_level: str,
    safe_side: str,
    safe_va_id: str,
    screw_diameter: float,
    screw_length: float,
) -> str:
    """Build Slicer Python code for cervical pedicle screw planning.

    Args:
        safe_seg_id: JSON-escaped segmentation node ID.
        safe_level: JSON-escaped vertebral level.
        safe_side: JSON-escaped side string.
        safe_va_id: JSON-escaped VA node ID or "null".
        screw_diameter: Screw diameter in mm.
        screw_length: Screw length in mm.

    Returns:
        Python code string for execution in Slicer.
    """
    angulation = TECHNIQUE_ANGULATION["pedicle"]
    return (
        _SLICER_HELPERS
        + f"""
# === Pedicle Screw Planning ===
seg_node_id = {safe_seg_id}
level = {safe_level}
side = {safe_side}
va_node_id = {safe_va_id}
screw_diameter = {screw_diameter}
screw_length = {screw_length}
medial_deg = {angulation['medial_deg']}
caudal_deg = {angulation['caudal_deg']}

seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not seg_node:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

segment_id, segment_name = _find_segment(seg_node, level)
geo = _get_segment_centroid_and_bounds(seg_node, segment_id)

sides = ['left', 'right'] if side == 'bilateral' else [side]
screws = []
warnings = []

for s in sides:
    # Entry point: junction of lateral mass and lamina (posterior, lateral)
    lat_offset = geo['dimensions_mm'][0] * 0.3 * (1 if s == 'right' else -1)
    entry_ras = [
        geo['centroid_ras'][0] + lat_offset,
        geo['centroid_ras'][1] - geo['dimensions_mm'][1] * 0.4,
        geo['centroid_ras'][2],
    ]

    target_ras = _compute_trajectory_endpoint(
        entry_ras, medial_deg, -caudal_deg, screw_length, side=s
    )

    # Safety checks
    va_assessment = {{'status': 'not_assessed', 'min_distance_mm': None}}
    if va_node_id:
        va_node = slicer.mrmlScene.GetNodeByID(va_node_id)
        if va_node:
            try:
                va_seg_id, _ = _find_segment(va_node, 'vertebral_artery')
            except ValueError:
                segmentation = va_node.GetSegmentation()
                n_segs = segmentation.GetNumberOfSegments()
                va_seg_id = segmentation.GetNthSegmentID(0) if n_segs > 0 else None
            if va_seg_id:
                va_dist = _trajectory_min_distance_to_segment(
                    va_node, va_seg_id, entry_ras, target_ras
                )
                va_assessment = {{
                    'status': _safety_color(va_dist, {VA_SAFETY_DISTANCE_MM}),
                    'min_distance_mm': round(va_dist, 1),
                }}
                if va_dist < {VA_SAFETY_DISTANCE_MM}:
                    msg = (f'VA clearance {{va_dist:.1f}}mm '
                           f'< {VA_SAFETY_DISTANCE_MM}mm on {{s}}')
                    warnings.append(msg)

    color = [0, 0.8, 0] if va_assessment['status'] != 'red' else [0.8, 0, 0]
    viz = _create_screw_visualization(
        entry_ras, target_ras, screw_diameter,
        color, f'pedicle_{{level}}_{{s}}'
    )

    screws.append({{
        'side': s,
        'entry_point_ras': [round(v, 1) for v in entry_ras],
        'target_point_ras': [round(v, 1) for v in target_ras],
        'screw_diameter_mm': screw_diameter,
        'screw_length_mm': screw_length,
        'angulation': {{'medial_deg': medial_deg, 'caudal_deg': caudal_deg}},
        'va_assessment': va_assessment,
        'visualization': viz,
    }})

result = {{
    'success': True,
    'technique': 'pedicle',
    'technique_name': 'Cervical Pedicle Screw',
    'reference': 'Abumi K et al. Spine 1994',
    'level': level,
    'segment_name': segment_name,
    'segment_geometry': {{
        'centroid_ras': [round(v, 1) for v in geo['centroid_ras']],
        'dimensions_mm': [round(v, 1) for v in geo['dimensions_mm']],
    }},
    'screws': screws,
    'warnings': warnings,
    'recommendations': [],
}}
print(json.dumps(result))
"""
    )


def _build_lateral_mass_code(
    safe_seg_id: str,
    safe_level: str,
    safe_side: str,
    safe_va_id: str,
    screw_diameter: float,
    screw_length: float,
    variant: str,
) -> str:
    """Build Slicer Python code for lateral mass screw planning.

    Args:
        safe_seg_id: JSON-escaped segmentation node ID.
        safe_level: JSON-escaped vertebral level.
        safe_side: JSON-escaped side string.
        safe_va_id: JSON-escaped VA node ID or "null".
        screw_diameter: Screw diameter in mm.
        screw_length: Screw length in mm.
        variant: Lateral mass variant name.

    Returns:
        Python code string for execution in Slicer.
    """
    ang_key = f"lateral_mass_{variant}"
    angulation = TECHNIQUE_ANGULATION[ang_key]
    lateral_deg = angulation.get("lateral_deg", 0.0)
    cephalad_deg = angulation.get("cephalad_deg", 0.0)

    variant_names = {
        "roy_camille": "Roy-Camille",
        "magerl": "Magerl",
        "an": "An",
        "anderson": "Anderson",
    }
    variant_refs = {
        "roy_camille": "Roy-Camille R et al. 1979",
        "magerl": "Magerl F. Cervical Spine Research Society 1987",
        "an": "An HS et al. Spine 1991",
        "anderson": "Anderson PA et al. Spine 1991",
    }

    return (
        _SLICER_HELPERS
        + f"""
# === Lateral Mass Screw Planning ({variant_names[variant]}) ===
seg_node_id = {safe_seg_id}
level = {safe_level}
side = {safe_side}
va_node_id = {safe_va_id}
screw_diameter = {screw_diameter}
screw_length = {screw_length}
lateral_deg = {lateral_deg}
cephalad_deg = {cephalad_deg}
variant = {json.dumps(variant)}

seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not seg_node:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

segment_id, segment_name = _find_segment(seg_node, level)
geo = _get_segment_centroid_and_bounds(seg_node, segment_id)

sides = ['left', 'right'] if side == 'bilateral' else [side]
screws = []
warnings = []

for s in sides:
    # Entry: center of posterior lateral mass face
    lat_offset = geo['dimensions_mm'][0] * 0.25 * (1 if s == 'right' else -1)
    entry_ras = [
        geo['centroid_ras'][0] + lat_offset,
        geo['centroid_ras'][1] - geo['dimensions_mm'][1] * 0.45,
        geo['centroid_ras'][2],
    ]

    # Lateral mass: negate medial angle → use lateral
    target_ras = _compute_trajectory_endpoint(
        entry_ras, -lateral_deg, cephalad_deg, screw_length, side=s
    )

    va_assessment = {{'status': 'not_assessed', 'min_distance_mm': None}}
    if va_node_id:
        va_node = slicer.mrmlScene.GetNodeByID(va_node_id)
        if va_node:
            try:
                va_seg_id, _ = _find_segment(va_node, 'vertebral_artery')
            except ValueError:
                segmentation = va_node.GetSegmentation()
                n_segs = segmentation.GetNumberOfSegments()
                va_seg_id = segmentation.GetNthSegmentID(0) if n_segs > 0 else None
            if va_seg_id:
                va_dist = _trajectory_min_distance_to_segment(
                    va_node, va_seg_id, entry_ras, target_ras
                )
                va_assessment = {{
                    'status': _safety_color(va_dist, {VA_SAFETY_DISTANCE_MM}),
                    'min_distance_mm': round(va_dist, 1),
                }}

    color = [0, 0.8, 0]
    viz = _create_screw_visualization(
        entry_ras, target_ras, screw_diameter,
        color, f'lat_mass_{{level}}_{{s}}'
    )

    screws.append({{
        'side': s,
        'entry_point_ras': [round(v, 1) for v in entry_ras],
        'target_point_ras': [round(v, 1) for v in target_ras],
        'screw_diameter_mm': screw_diameter,
        'screw_length_mm': screw_length,
        'angulation': {{'lateral_deg': lateral_deg, 'cephalad_deg': cephalad_deg}},
        'variant': variant,
        'va_assessment': va_assessment,
        'visualization': viz,
    }})

result = {{
    'success': True,
    'technique': 'lateral_mass',
    'technique_name': 'Lateral Mass Screw ({variant_names[variant]})',
    'reference': {json.dumps(variant_refs[variant])},
    'level': level,
    'segment_name': segment_name,
    'segment_geometry': {{
        'centroid_ras': [round(v, 1) for v in geo['centroid_ras']],
        'dimensions_mm': [round(v, 1) for v in geo['dimensions_mm']],
    }},
    'screws': screws,
    'warnings': warnings,
    'recommendations': [],
}}
print(json.dumps(result))
"""
    )


def _build_transarticular_code(
    safe_seg_id: str,
    safe_side: str,
    safe_va_id: str,
    screw_diameter: float,
    screw_length: float,
) -> str:
    """Build Slicer Python code for C1-C2 transarticular screw planning.

    VA segmentation is a HARD REQUIREMENT for this technique.

    Args:
        safe_seg_id: JSON-escaped segmentation node ID.
        safe_side: JSON-escaped side string.
        safe_va_id: JSON-escaped VA node ID (must not be null).
        screw_diameter: Screw diameter in mm.
        screw_length: Screw length in mm.

    Returns:
        Python code string for execution in Slicer.
    """
    angulation = TECHNIQUE_ANGULATION["transarticular"]
    return (
        _SLICER_HELPERS
        + f"""
# === Transarticular Screw Planning (Magerl) ===
seg_node_id = {safe_seg_id}
side = {safe_side}
va_node_id = {safe_va_id}
screw_diameter = {screw_diameter}
screw_length = {screw_length}
medial_deg = {angulation['medial_deg']}
cephalad_deg = {angulation['cephalad_deg']}
isthmus_min_height = {ISTHMUS_MIN_HEIGHT_MM}
isthmus_min_width = {ISTHMUS_MIN_WIDTH_MM}
va_safety_dist = {VA_SAFETY_DISTANCE_MM}

seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not seg_node:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

# Get C2 geometry for isthmus analysis
c2_seg_id, c2_seg_name = _find_segment(seg_node, 'C2')
c2_geo = _get_segment_centroid_and_bounds(seg_node, c2_seg_id)

# Get C1 geometry for target
try:
    c1_seg_id, c1_seg_name = _find_segment(seg_node, 'C1')
    c1_geo = _get_segment_centroid_and_bounds(seg_node, c1_seg_id)
except ValueError:
    c1_geo = None

# Isthmus analysis: approximate from C2 geometry
# The isthmus is the narrowest part of C2 pars between VA grooves
isthmus_height = c2_geo['dimensions_mm'][2] * 0.4
isthmus_width = c2_geo['dimensions_mm'][0] * 0.25

sides = ['left', 'right'] if side == 'bilateral' else [side]
screws = []
warnings = []
overall_blocked = False

for s in sides:
    # Entry: posterior inferior C2 surface
    lat_offset = c2_geo['dimensions_mm'][0] * 0.2 * (1 if s == 'right' else -1)
    entry_ras = [
        c2_geo['centroid_ras'][0] + lat_offset,
        c2_geo['centroid_ras'][1] - c2_geo['dimensions_mm'][1] * 0.45,
        c2_geo['centroid_ras'][2] - c2_geo['dimensions_mm'][2] * 0.35,
    ]

    target_ras = _compute_trajectory_endpoint(
        entry_ras, medial_deg, cephalad_deg, screw_length, side=s
    )

    # VA safety check (HARD REQUIREMENT)
    va_assessment = {{'status': 'red', 'min_distance_mm': 0.0, 'message': 'VA not assessed'}}
    va_node = slicer.mrmlScene.GetNodeByID(va_node_id)
    if va_node:
        try:
            va_seg_id, _ = _find_segment(va_node, 'vertebral_artery')
        except ValueError:
            segmentation = va_node.GetSegmentation()
            n_segs = segmentation.GetNumberOfSegments()
            va_seg_id = segmentation.GetNthSegmentID(0) if n_segs > 0 else None
        if va_seg_id:
            va_dist = _trajectory_min_distance_to_segment(
                va_node, va_seg_id, entry_ras, target_ras
            )
            va_status = _safety_color(va_dist, va_safety_dist)
            va_assessment = {{
                'status': va_status,
                'min_distance_mm': round(va_dist, 1),
                'message': f'VA clearance {{va_dist:.1f}}mm on {{s}}',
            }}
            if va_status == 'red':
                overall_blocked = True
                warnings.append(
                    f'HARD BLOCK: VA clearance {{va_dist:.1f}}mm < {{va_safety_dist}}mm on {{s}}. '
                    'Transarticular screw CONTRAINDICATED.'
                )

    # Isthmus check
    if isthmus_height < isthmus_min_height:
        warnings.append(
            f'Isthmus height {{isthmus_height:.1f}}mm < {{isthmus_min_height}}mm. '
            'High-riding VA suspected.'
        )
        overall_blocked = True
    if isthmus_width < isthmus_min_width:
        warnings.append(
            f'Isthmus width {{isthmus_width:.1f}}mm < {{isthmus_min_width}}mm. '
            'Narrow isthmus — consider alternative technique.'
        )

    blocked = overall_blocked or va_assessment['status'] == 'red'
    color = [0.8, 0, 0] if blocked else [0, 0.8, 0]
    viz = _create_screw_visualization(
        entry_ras, target_ras, screw_diameter,
        color, f'transart_C1C2_{{s}}'
    )

    screws.append({{
        'side': s,
        'entry_point_ras': [round(v, 1) for v in entry_ras],
        'target_point_ras': [round(v, 1) for v in target_ras],
        'screw_diameter_mm': screw_diameter,
        'screw_length_mm': screw_length,
        'angulation': {{'medial_deg': medial_deg, 'cephalad_deg': cephalad_deg}},
        'va_assessment': va_assessment,
        'isthmus_analysis': {{
            'height_mm': round(isthmus_height, 1),
            'width_mm': round(isthmus_width, 1),
            'adequate': isthmus_height >= isthmus_min_height and isthmus_width >= isthmus_min_width,
        }},
        'blocked': blocked,
        'visualization': viz,
    }})

recommendations = []
if overall_blocked:
    recommendations.append('Consider C1 lateral mass + C2 pars (Harms/Goel) as alternative')

result = {{
    'success': True,
    'technique': 'transarticular',
    'technique_name': 'C1-C2 Transarticular Screw (Magerl)',
    'reference': 'Magerl F et al. Cervical Spine Research Society 1987',
    'level': 'C1C2',
    'screws': screws,
    'isthmus_analysis': {{
        'height_mm': round(isthmus_height, 1),
        'width_mm': round(isthmus_width, 1),
        'adequate': isthmus_height >= isthmus_min_height and isthmus_width >= isthmus_min_width,
    }},
    'blocked': overall_blocked,
    'warnings': warnings,
    'recommendations': recommendations,
}}
print(json.dumps(result))
"""
    )


def _build_c1_lateral_mass_code(
    safe_seg_id: str,
    safe_side: str,
    safe_va_id: str,
    screw_diameter: float,
    screw_length: float,
) -> str:
    """Build Slicer Python code for C1 lateral mass (Harms/Goel) screw planning.

    Args:
        safe_seg_id: JSON-escaped segmentation node ID.
        safe_side: JSON-escaped side string.
        safe_va_id: JSON-escaped VA node ID or "null".
        screw_diameter: Screw diameter in mm.
        screw_length: Screw length in mm.

    Returns:
        Python code string for execution in Slicer.
    """
    angulation = TECHNIQUE_ANGULATION["c1_lateral_mass"]
    return (
        _SLICER_HELPERS
        + f"""
# === C1 Lateral Mass Screw Planning (Harms/Goel) ===
seg_node_id = {safe_seg_id}
side = {safe_side}
va_node_id = {safe_va_id}
screw_diameter = {screw_diameter}
screw_length = {screw_length}
medial_deg = {angulation['medial_deg']}
caudal_deg = {angulation['caudal_deg']}

seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not seg_node:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

segment_id, segment_name = _find_segment(seg_node, 'C1')
geo = _get_segment_centroid_and_bounds(seg_node, segment_id)

sides = ['left', 'right'] if side == 'bilateral' else [side]
screws = []
warnings = []

for s in sides:
    # Entry: junction of C1 posterior arch and lateral mass, inferior border
    lat_offset = geo['dimensions_mm'][0] * 0.3 * (1 if s == 'right' else -1)
    entry_ras = [
        geo['centroid_ras'][0] + lat_offset,
        geo['centroid_ras'][1] - geo['dimensions_mm'][1] * 0.35,
        geo['centroid_ras'][2] - geo['dimensions_mm'][2] * 0.3,
    ]

    target_ras = _compute_trajectory_endpoint(
        entry_ras, medial_deg, -caudal_deg, screw_length, side=s
    )

    va_assessment = {{'status': 'not_assessed', 'min_distance_mm': None}}
    if va_node_id:
        va_node = slicer.mrmlScene.GetNodeByID(va_node_id)
        if va_node:
            try:
                va_seg_id, _ = _find_segment(va_node, 'vertebral_artery')
            except ValueError:
                segmentation = va_node.GetSegmentation()
                n_segs = segmentation.GetNumberOfSegments()
                va_seg_id = segmentation.GetNthSegmentID(0) if n_segs > 0 else None
            if va_seg_id:
                va_dist = _trajectory_min_distance_to_segment(
                    va_node, va_seg_id, entry_ras, target_ras
                )
                va_assessment = {{
                    'status': _safety_color(va_dist, {VA_SAFETY_DISTANCE_MM}),
                    'min_distance_mm': round(va_dist, 1),
                }}
                if va_dist < {VA_SAFETY_DISTANCE_MM}:
                    warnings.append(
                        f'VA clearance {{va_dist:.1f}}mm < {VA_SAFETY_DISTANCE_MM}mm on {{s}}'
                    )

    color = [0, 0.8, 0] if va_assessment.get('status') != 'red' else [0.8, 0, 0]
    viz = _create_screw_visualization(
        entry_ras, target_ras, screw_diameter,
        color, f'c1_lat_mass_{{s}}'
    )

    screws.append({{
        'side': s,
        'entry_point_ras': [round(v, 1) for v in entry_ras],
        'target_point_ras': [round(v, 1) for v in target_ras],
        'screw_diameter_mm': screw_diameter,
        'screw_length_mm': screw_length,
        'angulation': {{'medial_deg': medial_deg, 'caudal_deg': caudal_deg}},
        'va_assessment': va_assessment,
        'visualization': viz,
    }})

result = {{
    'success': True,
    'technique': 'c1_lateral_mass',
    'technique_name': 'C1 Lateral Mass Screw (Harms/Goel)',
    'reference': 'Harms J, Melcher RP. Spine 2001; Goel A, Laheri VK. Acta Neurochir 1994',
    'level': 'C1',
    'segment_name': segment_name,
    'segment_geometry': {{
        'centroid_ras': [round(v, 1) for v in geo['centroid_ras']],
        'dimensions_mm': [round(v, 1) for v in geo['dimensions_mm']],
    }},
    'screws': screws,
    'warnings': warnings,
    'recommendations': ['Consider C2 pars screw as complement for C1-C2 fixation'],
}}
print(json.dumps(result))
"""
    )


def _build_c2_pars_code(
    safe_seg_id: str,
    safe_side: str,
    safe_va_id: str,
    screw_diameter: float,
    screw_length: float,
) -> str:
    """Build Slicer Python code for C2 pars interarticularis screw planning.

    Args:
        safe_seg_id: JSON-escaped segmentation node ID.
        safe_side: JSON-escaped side string.
        safe_va_id: JSON-escaped VA node ID or "null".
        screw_diameter: Screw diameter in mm.
        screw_length: Screw length in mm.

    Returns:
        Python code string for execution in Slicer.
    """
    angulation = TECHNIQUE_ANGULATION["c2_pars"]
    return (
        _SLICER_HELPERS
        + f"""
# === C2 Pars Interarticularis Screw Planning ===
seg_node_id = {safe_seg_id}
side = {safe_side}
va_node_id = {safe_va_id}
screw_diameter = {screw_diameter}
screw_length = {screw_length}
medial_deg = {angulation['medial_deg']}
cephalad_deg = {angulation['cephalad_deg']}

seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not seg_node:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

segment_id, segment_name = _find_segment(seg_node, 'C2')
geo = _get_segment_centroid_and_bounds(seg_node, segment_id)

sides = ['left', 'right'] if side == 'bilateral' else [side]
screws = []
warnings = []

for s in sides:
    # Entry: superior medial quadrant of C2 isthmus
    lat_offset = geo['dimensions_mm'][0] * 0.15 * (1 if s == 'right' else -1)
    entry_ras = [
        geo['centroid_ras'][0] + lat_offset,
        geo['centroid_ras'][1] - geo['dimensions_mm'][1] * 0.3,
        geo['centroid_ras'][2] + geo['dimensions_mm'][2] * 0.15,
    ]

    target_ras = _compute_trajectory_endpoint(
        entry_ras, medial_deg, cephalad_deg, screw_length, side=s
    )

    va_assessment = {{'status': 'not_assessed', 'min_distance_mm': None}}
    if va_node_id:
        va_node = slicer.mrmlScene.GetNodeByID(va_node_id)
        if va_node:
            try:
                va_seg_id, _ = _find_segment(va_node, 'vertebral_artery')
            except ValueError:
                segmentation = va_node.GetSegmentation()
                n_segs = segmentation.GetNumberOfSegments()
                va_seg_id = segmentation.GetNthSegmentID(0) if n_segs > 0 else None
            if va_seg_id:
                va_dist = _trajectory_min_distance_to_segment(
                    va_node, va_seg_id, entry_ras, target_ras
                )
                va_assessment = {{
                    'status': _safety_color(va_dist, {VA_SAFETY_DISTANCE_MM}),
                    'min_distance_mm': round(va_dist, 1),
                }}
                if va_dist < {VA_SAFETY_DISTANCE_MM}:
                    warnings.append(
                        f'VA clearance {{va_dist:.1f}}mm < {VA_SAFETY_DISTANCE_MM}mm on {{s}}'
                    )

    color = [0, 0.8, 0] if va_assessment.get('status') != 'red' else [0.8, 0, 0]
    viz = _create_screw_visualization(
        entry_ras, target_ras, screw_diameter,
        color, f'c2_pars_{{s}}'
    )

    screws.append({{
        'side': s,
        'entry_point_ras': [round(v, 1) for v in entry_ras],
        'target_point_ras': [round(v, 1) for v in target_ras],
        'screw_diameter_mm': screw_diameter,
        'screw_length_mm': screw_length,
        'angulation': {{'medial_deg': medial_deg, 'cephalad_deg': cephalad_deg}},
        'va_assessment': va_assessment,
        'visualization': viz,
    }})

result = {{
    'success': True,
    'technique': 'c2_pars',
    'technique_name': 'C2 Pars Interarticularis Screw',
    'reference': 'Harms J, Melcher RP. Spine 2001',
    'level': 'C2',
    'segment_name': segment_name,
    'segment_geometry': {{
        'centroid_ras': [round(v, 1) for v in geo['centroid_ras']],
        'dimensions_mm': [round(v, 1) for v in geo['dimensions_mm']],
    }},
    'screws': screws,
    'warnings': warnings,
    'recommendations': [],
}}
print(json.dumps(result))
"""
    )


def _build_occipital_code(
    safe_seg_id: str,
    safe_side: str,
    screw_diameter: float,
    screw_length: float,
) -> str:
    """Build Slicer Python code for occipital screw planning.

    Args:
        safe_seg_id: JSON-escaped segmentation node ID.
        safe_side: JSON-escaped side string.
        screw_diameter: Screw diameter in mm.
        screw_length: Screw length in mm.

    Returns:
        Python code string for execution in Slicer.
    """
    return (
        _SLICER_HELPERS
        + f"""
# === Occipital Screw Planning ===
seg_node_id = {safe_seg_id}
side = {safe_side}
screw_diameter = {screw_diameter}
screw_length = {screw_length}
min_thickness = {OCCIPITAL_MIN_THICKNESS_MM}

seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not seg_node:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

# Find occipital segment
segmentation = seg_node.GetSegmentation()
occ_seg_id = None
occ_seg_name = None
for i in range(segmentation.GetNumberOfSegments()):
    seg = segmentation.GetNthSegment(i)
    name = seg.GetName()
    if 'occiput' in name.lower() or 'occipital' in name.lower():
        occ_seg_id = segmentation.GetNthSegmentID(i)
        occ_seg_name = name
        break

if not occ_seg_id:
    raise ValueError('Occipital segment not found in segmentation')

geo = _get_segment_centroid_and_bounds(seg_node, occ_seg_id)

# Thickness mapping: estimate from AP dimension at various points
# The keel (midline, near EOP) is the thickest area
keel_thickness = geo['dimensions_mm'][1] * 0.6
lateral_thickness = geo['dimensions_mm'][1] * 0.3

thickness_map = {{
    'keel_midline_mm': round(keel_thickness, 1),
    'paramedian_mm': round((keel_thickness + lateral_thickness) / 2, 1),
    'lateral_mm': round(lateral_thickness, 1),
    'adequate_for_bicortical': keel_thickness >= min_thickness,
}}

screws = []
warnings = []

# Midline screws at keel (strongest)
if keel_thickness >= min_thickness:
    positions = []
    if side == 'bilateral':
        positions = [
            ('midline_superior', [geo['centroid_ras'][0], geo['centroid_ras'][1],
                                   geo['centroid_ras'][2] + geo['dimensions_mm'][2] * 0.2]),
            ('midline_inferior', [geo['centroid_ras'][0], geo['centroid_ras'][1],
                                   geo['centroid_ras'][2] - geo['dimensions_mm'][2] * 0.1]),
        ]
    else:
        offset = geo['dimensions_mm'][0] * 0.1 * (1 if side == 'right' else -1)
        positions = [
            (f'paramedian_{{side}}', [geo['centroid_ras'][0] + offset,
                                      geo['centroid_ras'][1],
                                      geo['centroid_ras'][2]]),
        ]

    for name, entry_ras in positions:
        # Occipital screws: perpendicular to outer table (anterior direction)
        target_ras = [
            entry_ras[0],
            entry_ras[1] + screw_length,
            entry_ras[2],
        ]

        color = [0, 0.8, 0]
        viz = _create_screw_visualization(
            entry_ras, target_ras, screw_diameter,
            color, f'occipital_{{name}}'
        )

        screws.append({{
            'position': name,
            'entry_point_ras': [round(v, 1) for v in entry_ras],
            'target_point_ras': [round(v, 1) for v in target_ras],
            'screw_diameter_mm': screw_diameter,
            'screw_length_mm': screw_length,
            'local_thickness_mm': round(keel_thickness, 1),
            'bicortical_purchase': keel_thickness >= screw_length,
            'visualization': viz,
        }})
else:
    warnings.append(
        f'Keel thickness {{keel_thickness:.1f}}mm < {{min_thickness}}mm. '
        'Insufficient bone for bicortical purchase.'
    )

recommendations = []
if not thickness_map['adequate_for_bicortical']:
    recommendations.append('Consider unicortical screws with plate reinforcement')
    recommendations.append('Consider extending construct to C2 for additional fixation')

result = {{
    'success': True,
    'technique': 'occipital',
    'technique_name': 'Occipital Screw',
    'reference': 'Haher TR et al. Spine 1999',
    'level': 'Occiput',
    'segment_name': occ_seg_name,
    'thickness_map': thickness_map,
    'screws': screws,
    'warnings': warnings,
    'recommendations': recommendations,
}}
print(json.dumps(result))
"""
    )


def _build_auto_analysis_code(
    safe_seg_id: str,
    safe_level: str,
    safe_va_id: str,
) -> str:
    """Build Slicer Python code for automatic technique recommendation.

    Analyzes anatomy (VA, isthmus, pedicle width) and recommends the best
    technique for the given level.

    Args:
        safe_seg_id: JSON-escaped segmentation node ID.
        safe_level: JSON-escaped vertebral level.
        safe_va_id: JSON-escaped VA node ID or "null".

    Returns:
        Python code string for execution in Slicer.
    """
    return (
        _SLICER_HELPERS
        + f"""
# === Auto Technique Analysis ===
seg_node_id = {safe_seg_id}
level = {safe_level}
va_node_id = {safe_va_id}
pedicle_min_width = {CERVICAL_PEDICLE_MIN_WIDTH_MM}
isthmus_min_height = {ISTHMUS_MIN_HEIGHT_MM}
isthmus_min_width = {ISTHMUS_MIN_WIDTH_MM}

seg_node = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not seg_node:
    raise ValueError('Segmentation node not found: ' + seg_node_id)

recommendations = []
analysis = {{}}

# Determine level category
level_upper = level.upper()

if level_upper == 'OCCIPUT':
    # Occipital → only option is occipital screws
    try:
        segmentation = seg_node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            seg = segmentation.GetNthSegment(i)
            if 'occiput' in seg.GetName().lower() or 'occipital' in seg.GetName().lower():
                seg_id = segmentation.GetNthSegmentID(i)
                geo = _get_segment_centroid_and_bounds(seg_node, seg_id)
                thickness = geo['dimensions_mm'][1] * 0.6
                analysis['occipital_thickness_mm'] = round(thickness, 1)
                break
    except Exception:
        pass
    recommendations.append({{
        'technique': 'occipital',
        'confidence': 'high',
        'rationale': 'Only technique available for occipital fixation',
    }})

elif level_upper == 'C1':
    # C1 → Harms/Goel (C1 lateral mass)
    va_status = 'unknown'
    if va_node_id:
        va_status = 'available'
    recommendations.append({{
        'technique': 'c1_lateral_mass',
        'confidence': 'high',
        'rationale': 'Harms/Goel technique — standard for C1 fixation',
        'complement': 'c2_pars',
        'va_required': True,
        'va_status': va_status,
    }})

elif level_upper == 'C1C2':
    # C1-C2 → Try transarticular first, fallback to Harms + pars
    analysis['level'] = 'C1C2'
    try:
        c2_seg_id, _ = _find_segment(seg_node, 'C2')
        c2_geo = _get_segment_centroid_and_bounds(seg_node, c2_seg_id)
        isthmus_h = c2_geo['dimensions_mm'][2] * 0.4
        isthmus_w = c2_geo['dimensions_mm'][0] * 0.25
        analysis['isthmus_height_mm'] = round(isthmus_h, 1)
        analysis['isthmus_width_mm'] = round(isthmus_w, 1)

        if isthmus_h >= isthmus_min_height and isthmus_w >= isthmus_min_width:
            recommendations.append({{
                'technique': 'transarticular',
                'confidence': 'medium',
                'rationale': f'Isthmus adequate (h={{isthmus_h:.1f}}mm, w={{isthmus_w:.1f}}mm). '
                             'VA assessment mandatory before proceeding.',
                'va_required': True,
            }})
        else:
            recommendations.append({{
                'technique': 'transarticular',
                'confidence': 'low',
                'rationale': f'Isthmus marginal (h={{isthmus_h:.1f}}mm, w={{isthmus_w:.1f}}mm). '
                             'High risk — consider alternative.',
                'va_required': True,
            }})
    except ValueError:
        pass

    recommendations.append({{
        'technique': 'c1_lateral_mass',
        'confidence': 'high',
        'rationale': 'Harms/Goel + C2 pars — safer alternative for C1-C2 fixation',
        'complement': 'c2_pars',
        'va_required': True,
    }})

elif level_upper == 'C2':
    # C2 → pedicle or pars
    try:
        c2_seg_id, _ = _find_segment(seg_node, 'C2')
        c2_geo = _get_segment_centroid_and_bounds(seg_node, c2_seg_id)
        pedicle_w = c2_geo['dimensions_mm'][0] * 0.2
        analysis['pedicle_width_mm'] = round(pedicle_w, 1)

        if pedicle_w >= pedicle_min_width:
            recommendations.append({{
                'technique': 'pedicle',
                'confidence': 'high',
                'rationale': (f'Pedicle width adequate '
                              f'({{pedicle_w:.1f}}mm >= {{pedicle_min_width}}mm)'),
            }})
        recommendations.append({{
            'technique': 'c2_pars',
            'confidence': 'high',
            'rationale': 'C2 pars — reliable alternative, shorter screw trajectory',
        }})
    except ValueError:
        recommendations.append({{
            'technique': 'c2_pars',
            'confidence': 'medium',
            'rationale': 'C2 pars — default when geometry cannot be assessed',
        }})

elif level_upper in ('C3', 'C4', 'C5', 'C6'):
    # Subaxial (C3-C6) → lateral mass preferred, pedicle if width allows
    try:
        seg_id, _ = _find_segment(seg_node, level)
        geo = _get_segment_centroid_and_bounds(seg_node, seg_id)
        pedicle_w = geo['dimensions_mm'][0] * 0.15
        analysis['pedicle_width_mm'] = round(pedicle_w, 1)

        recommendations.append({{
            'technique': 'lateral_mass',
            'confidence': 'high',
            'rationale': 'Lateral mass screws — standard for subaxial cervical fixation',
            'default_variant': 'magerl',
        }})
        if pedicle_w >= pedicle_min_width:
            recommendations.append({{
                'technique': 'pedicle',
                'confidence': 'medium',
                'rationale': f'Pedicle width adequate ({{pedicle_w:.1f}}mm) — '
                             'stronger fixation but higher risk',
            }})
    except ValueError:
        recommendations.append({{
            'technique': 'lateral_mass',
            'confidence': 'medium',
            'rationale': 'Lateral mass — default for subaxial when geometry unavailable',
            'default_variant': 'magerl',
        }})

elif level_upper == 'C7':
    # C7 → pedicle preferred (largest cervical pedicle), lateral mass alternative
    try:
        seg_id, _ = _find_segment(seg_node, 'C7')
        geo = _get_segment_centroid_and_bounds(seg_node, seg_id)
        pedicle_w = geo['dimensions_mm'][0] * 0.2
        analysis['pedicle_width_mm'] = round(pedicle_w, 1)
    except ValueError:
        pedicle_w = 6.0

    recommendations.append({{
        'technique': 'pedicle',
        'confidence': 'high',
        'rationale': 'C7 has the largest cervical pedicle — pedicle screws preferred',
    }})
    recommendations.append({{
        'technique': 'lateral_mass',
        'confidence': 'medium',
        'rationale': 'Lateral mass — alternative if pedicle anatomy unfavorable',
        'default_variant': 'magerl',
    }})

else:
    recommendations.append({{
        'technique': 'unknown',
        'confidence': 'low',
        'rationale': f'Level {{level}} not recognized for cervical instrumentation',
    }})

result = {{
    'success': True,
    'technique': 'auto',
    'level': level,
    'analysis': analysis,
    'recommendations': recommendations,
}}
print(json.dumps(result))
"""
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def plan_cervical_screws(
    technique: str,
    level: str,
    segmentation_node_id: str,
    side: str = "bilateral",
    va_node_id: str | None = None,
    variant: str | None = None,
    screw_diameter_mm: float | None = None,
    screw_length_mm: float | None = None,
) -> dict[str, Any]:
    """Plan cervical screw placement using one of 6 techniques.

    Generates screw entry points, trajectory vectors, safety assessments,
    and 3D visualization in Slicer based on patient-specific anatomy
    from segmentation data.

    Techniques:
        - pedicle: Cervical pedicle screws (C2-C7)
        - lateral_mass: Lateral mass screws (C3-C7), 4 variants
        - transarticular: C1-C2 Magerl transarticular screws
        - c1_lateral_mass: C1 lateral mass (Harms/Goel)
        - c2_pars: C2 pars interarticularis screws
        - occipital: Occipital screws with thickness mapping
        - auto: Analyze anatomy and recommend best technique

    Args:
        technique: Instrumentation technique name.
        level: Vertebral level (e.g. "C5", "C1C2", "Occiput").
        segmentation_node_id: MRML node ID of vertebral segmentation.
        side: "left", "right", or "bilateral" (default: "bilateral").
        va_node_id: MRML node ID of vertebral artery segmentation.
            REQUIRED for transarticular, recommended for C1-C2 techniques.
        variant: Lateral mass variant ("roy_camille", "magerl", "an", "anderson").
            Only for lateral_mass technique. Defaults to "magerl".
        screw_diameter_mm: Override default screw diameter (mm).
        screw_length_mm: Override default screw length (mm).

    Returns:
        Dict with technique details, screw parameters, safety assessment,
        visualization node IDs, warnings, and recommendations.

    Raises:
        ValidationError: If input parameters are invalid.
        SlicerConnectionError: If Slicer is not reachable or execution fails.
    """
    # Validate all inputs
    technique = _validate_technique(technique)
    level = _validate_level(technique, level)
    segmentation_node_id = validate_mrml_node_id(segmentation_node_id)
    side = _validate_side(side)
    variant = _validate_variant(variant, technique)

    # VA is HARD REQUIREMENT for transarticular
    if technique == "transarticular" and not va_node_id:
        raise ValidationError(
            "VA segmentation (va_node_id) is REQUIRED for transarticular technique. "
            "Vertebral artery proximity must be assessed to prevent catastrophic injury.",
            "va_node_id",
            "",
        )

    # VA recommended for C1-C2 techniques
    if technique in ("c1_lateral_mass", "c2_pars") and not va_node_id:
        logger.warning(
            f"VA segmentation not provided for {technique}. "
            "VA assessment strongly recommended for upper cervical instrumentation."
        )

    if va_node_id:
        va_node_id = validate_mrml_node_id(va_node_id)

    # Resolve screw dimensions
    if technique != "auto":
        diameter, length = _validate_screw_dimensions(technique, screw_diameter_mm, screw_length_mm)
    else:
        diameter, length = 3.5, 14.0  # defaults not used for auto

    client = get_client()

    # Build technique-specific Python code
    safe_seg_id = json.dumps(segmentation_node_id)
    safe_level = json.dumps(level)
    safe_side = json.dumps(side)
    safe_va_id = json.dumps(va_node_id) if va_node_id else "None"

    if technique == "pedicle":
        python_code = _build_pedicle_code(
            safe_seg_id,
            safe_level,
            safe_side,
            safe_va_id,
            diameter,
            length,
        )
    elif technique == "lateral_mass":
        python_code = _build_lateral_mass_code(
            safe_seg_id,
            safe_level,
            safe_side,
            safe_va_id,
            diameter,
            length,
            variant,
        )
    elif technique == "transarticular":
        python_code = _build_transarticular_code(
            safe_seg_id,
            safe_side,
            safe_va_id,
            diameter,
            length,
        )
    elif technique == "c1_lateral_mass":
        python_code = _build_c1_lateral_mass_code(
            safe_seg_id,
            safe_side,
            safe_va_id,
            diameter,
            length,
        )
    elif technique == "c2_pars":
        python_code = _build_c2_pars_code(
            safe_seg_id,
            safe_side,
            safe_va_id,
            diameter,
            length,
        )
    elif technique == "occipital":
        python_code = _build_occipital_code(
            safe_seg_id,
            safe_side,
            diameter,
            length,
        )
    elif technique == "auto":
        python_code = _build_auto_analysis_code(
            safe_seg_id,
            safe_level,
            safe_va_id,
        )
    else:
        raise ValidationError(f"Unhandled technique: {technique}", "technique", technique)

    try:
        exec_result = client.exec_python(python_code, timeout=INSTRUMENTATION_TIMEOUT)

        result_data = _parse_json_result(
            exec_result.get("result", ""),
            f"cervical screw planning ({technique})",
        )

        logger.info(
            f"Cervical screw planning completed: technique={technique}, "
            f"level={level}, side={side}"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Cervical screw planning failed: {e.message}")
        raise
