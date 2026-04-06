"""Craniocervical junction (CCJ) measurement tools.

Provides landmark extraction, angle/distance computation (CXA, ADI, Powers
ratio, BDI, BAI, Ranawat, McGregor, Chamberlain, Wackenheim), and the
public ``measure_ccj_angles`` tool function.

Clinical references:
- Joaquim 2019, Harris 2002, Powers 1979
"""

import json
import logging
from typing import Any

from slicer_mcp.core.constants import LONG_OPERATION_TIMEOUT
from slicer_mcp.core.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.features._codegen_snippets import (
    _render_classify_preamble,
    _render_geometry_preamble,
)
from slicer_mcp.features.base_tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)
from slicer_mcp.features.spine.constants import (
    CCJ_NORMAL_RANGES,
    TOTALSEGMENTATOR_VERTEBRA_MAP,
    VALID_POPULATIONS,
)

__all__ = [
    "measure_ccj_angles",
]

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# CCJ Measurement Python Code Builders
# =============================================================================


def _build_ccj_landmark_extraction_code(safe_node_id: str) -> str:
    """Build Python code to extract CCJ landmarks from segmentation.

    Extracts centroids and extremity points of relevant vertebral
    segments (C1, C2) and skull base structures from a TotalSegmentator
    segmentation in RAS coordinate space.

    Args:
        safe_node_id: JSON-escaped segmentation node ID

    Returns:
        Python code string for Slicer execution
    """
    return f"""
import slicer
import json
import numpy as np

node_id = {safe_node_id}

segNode = slicer.mrmlScene.GetNodeByID(node_id)
if not segNode:
    raise ValueError('Segmentation node not found: ' + node_id)

segmentation = segNode.GetSegmentation()
if not segmentation:
    raise ValueError('No segmentation data in node: ' + node_id)

# Helper: get segment centroid and bounding box extremities in RAS
def get_segment_points(seg_node, segment_name):
    \"\"\"Return centroid and extremity points for a segment in RAS coords.\"\"\"
    import vtk

    seg = seg_node.GetSegmentation()
    seg_id = None
    for i in range(seg.GetNumberOfSegments()):
        s = seg.GetNthSegment(i)
        if s.GetName() == segment_name:
            seg_id = seg.GetNthSegmentID(i)
            break

    if seg_id is None:
        return None

    # Get labelmap representation
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        seg_node, [seg_id], labelmapNode, None
    )

    # Get IJK to RAS matrix
    ijkToRas = vtk.vtkMatrix4x4()
    labelmapNode.GetIJKToRASMatrix(ijkToRas)

    imageData = labelmapNode.GetImageData()
    dims = imageData.GetDimensions()

    # Collect all voxel positions in RAS
    points_ras = []
    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                val = imageData.GetScalarComponentAsFloat(i, j, k, 0)
                if val > 0:
                    ijk = [i, j, k, 1]
                    ras = [0, 0, 0, 1]
                    ijkToRas.MultiplyPoint(ijk, ras)
                    points_ras.append(ras[:3])

    slicer.mrmlScene.RemoveNode(labelmapNode)

    if not points_ras:
        return None

    pts = np.array(points_ras)
    centroid = pts.mean(axis=0).tolist()
    superior = pts[pts[:, 2].argmax()].tolist()   # max S
    inferior = pts[pts[:, 2].argmin()].tolist()    # min S
    anterior = pts[pts[:, 1].argmax()].tolist()    # max A (positive A = anterior in RAS)
    posterior = pts[pts[:, 1].argmin()].tolist()    # min A

    return {{
        'centroid': centroid,
        'superior': superior,
        'inferior': inferior,
        'anterior': anterior,
        'posterior': posterior,
        'n_voxels': len(points_ras)
    }}

# Map TotalSegmentator segment names to look for
ts_names = {{}}
for i in range(segmentation.GetNumberOfSegments()):
    s = segmentation.GetNthSegment(i)
    ts_names[s.GetName()] = s.GetName()

# Extract C1 and C2 points
c1_data = None
c2_data = None

cervical_ts_map = {json.dumps(dict(list(TOTALSEGMENTATOR_VERTEBRA_MAP.items())[:7]))}
for ts_label, standard in cervical_ts_map.items():
    if ts_label in ts_names:
        pts = get_segment_points(segNode, ts_label)
        if pts is not None:
            if standard == 'C1':
                c1_data = pts
            elif standard == 'C2':
                c2_data = pts

# Also try standard names
if c1_data is None and 'C1' in ts_names:
    c1_data = get_segment_points(segNode, 'C1')
if c2_data is None and 'C2' in ts_names:
    c2_data = get_segment_points(segNode, 'C2')

# Derive CCJ landmarks from vertebral geometry
landmarks = {{}}

if c1_data:
    landmarks['c1_anterior_arch'] = c1_data['anterior']
    landmarks['c1_posterior_arch'] = c1_data['posterior']
    landmarks['c1_centroid'] = c1_data['centroid']

if c2_data:
    landmarks['dens_tip'] = c2_data['superior']
    landmarks['dens_base'] = c2_data['centroid']
    landmarks['c2_posteroinferior'] = c2_data['posterior']
    landmarks['c2_anteroinferior'] = c2_data['anterior']
    landmarks['c2_centroid'] = c2_data['centroid']

# Basion/opisthion: approximate from skull base segment if available
skull_names = ['skull', 'skull_base', 'cranium']
skull_data = None
for name in skull_names:
    if name in ts_names:
        skull_data = get_segment_points(segNode, name)
        if skull_data:
            break

if skull_data:
    landmarks['basion'] = skull_data['inferior']  # most inferior skull = basion approx
    landmarks['opisthion'] = skull_data['posterior']  # posterior skull base
else:
    # Estimate basion from C1 superior + offset
    if c1_data:
        c1_sup = np.array(c1_data['superior'])
        landmarks['basion'] = (c1_sup + np.array([0, 0, 5])).tolist()
        landmarks['opisthion'] = (c1_sup + np.array([0, -15, 5])).tolist()

# Foramen magnum approximation
if 'basion' in landmarks and 'opisthion' in landmarks:
    landmarks['foramen_magnum_anterior'] = landmarks['basion']
    landmarks['foramen_magnum_posterior'] = landmarks['opisthion']

result = {{
    'success': True,
    'landmarks': landmarks,
    'segments_found': {{
        'C1': c1_data is not None,
        'C2': c2_data is not None,
        'skull_base': skull_data is not None
    }},
    'coordinate_system': 'RAS',
    'node_id': node_id,
    'node_name': segNode.GetName()
}}

__execResult = result
"""


def _build_ccj_angles_code(safe_node_id: str, safe_population: str) -> str:
    """Build Python code to compute all CCJ measurements.

    Computes CXA, ADI, Powers ratio, Ranawat, McGregor, Chamberlain,
    and Wackenheim line measurements from extracted landmarks.

    Args:
        safe_node_id: JSON-escaped segmentation node ID
        safe_population: JSON-escaped population string ("adult" or "child")

    Returns:
        Python code string for Slicer execution
    """
    landmark_code = _build_ccj_landmark_extraction_code(safe_node_id)

    measurement_code = f"""
# --- CCJ Angle & Distance Measurements ---
population = {safe_population}

{_render_geometry_preamble()}

lm = landmarks
measurements = {{}}
reference_ranges = {{}}
statuses = {{}}

{_render_classify_preamble()}

# --- ADI (Atlantodental Interval) ---
if 'c1_anterior_arch' in lm and 'dens_tip' in lm:
    adi = point_dist(lm['c1_anterior_arch'], lm['dens_tip'])
    measurements['ADI_mm'] = round(adi, 2)
    adi_key = 'ADI_child' if population == 'child' else 'ADI_adult'
    ref = {json.dumps(CCJ_NORMAL_RANGES)}
    if adi_key in ref:
        reference_ranges['ADI'] = {{'min': ref[adi_key][0], 'max': ref[adi_key][1], 'unit': 'mm'}}
        classify(adi, ref[adi_key][0], ref[adi_key][1], 'ADI')

# --- BDI (Basion-Dens Interval) ---
if 'basion' in lm and 'dens_tip' in lm:
    bdi = point_dist(lm['basion'], lm['dens_tip'])
    measurements['BDI_mm'] = round(bdi, 2)
    ref = {json.dumps(CCJ_NORMAL_RANGES)}
    if 'BDI' in ref:
        reference_ranges['BDI'] = {{'min': ref['BDI'][0], 'max': ref['BDI'][1], 'unit': 'mm'}}
        classify(bdi, ref['BDI'][0], ref['BDI'][1], 'BDI')

# --- BAI (Basion-Axis Interval) ---
if 'basion' in lm and 'c2_posteroinferior' in lm:
    bai = point_dist(lm['basion'], lm['c2_posteroinferior'])
    measurements['BAI_mm'] = round(bai, 2)
    ref = {json.dumps(CCJ_NORMAL_RANGES)}
    if 'BAI' in ref:
        reference_ranges['BAI'] = {{'min': ref['BAI'][0], 'max': ref['BAI'][1], 'unit': 'mm'}}
        classify(bai, ref['BAI'][0], ref['BAI'][1], 'BAI')

# --- Powers Ratio ---
# BC / OA where B=basion, C=C1 posterior arch, O=opisthion, A=C1 anterior arch
if all(k in lm for k in ['basion', 'c1_posterior_arch', 'opisthion', 'c1_anterior_arch']):
    bc = point_dist(lm['basion'], lm['c1_posterior_arch'])
    oa = point_dist(lm['opisthion'], lm['c1_anterior_arch'])
    if oa > 1e-6:
        powers = bc / oa
        measurements['powers_ratio'] = round(powers, 3)
        ref = {json.dumps(CCJ_NORMAL_RANGES)}
        if 'powers_ratio' in ref:
            reference_ranges['powers_ratio'] = {{
                'min': ref['powers_ratio'][0],
                'max': ref['powers_ratio'][1],
                'unit': 'ratio'
            }}
            classify(powers, ref['powers_ratio'][0], ref['powers_ratio'][1], 'powers_ratio')

# --- CXA (Clivus-Canal Angle / Clivo-Axial Angle) ---
# Angle between clivus line (basion-dorsum sellae approx) and posterior C2 line
if all(k in lm for k in ['basion', 'c2_posteroinferior', 'opisthion']):
    clivus_vec = np.array(lm['basion']) - np.array(lm['opisthion'])
    canal_vec = np.array(lm['c2_posteroinferior']) - np.array(lm['basion'])
    cxa = vec_angle_deg(clivus_vec, canal_vec)
    measurements['CXA_deg'] = round(cxa, 1)
    # CXA normal: 150-165 degrees
    reference_ranges['CXA'] = {{'min': 150.0, 'max': 165.0, 'unit': 'degrees'}}
    classify(cxa, 150.0, 165.0, 'CXA')

# --- Ranawat Criterion ---
# Distance from C1 ring center to C2 pedicle axis line
if 'c1_centroid' in lm and 'c2_centroid' in lm and 'dens_tip' in lm:
    c1_c2_dist = point_dist(lm['c1_centroid'], lm['c2_centroid'])
    measurements['ranawat_value_mm'] = round(c1_c2_dist, 2)
    # Normal Ranawat: 15 mm (males), 13 mm (females); abnormal if <13
    reference_ranges['ranawat'] = {{'min': 13.0, 'max': 20.0, 'unit': 'mm'}}
    classify(c1_c2_dist, 13.0, 20.0, 'ranawat')

# --- McGregor Line / Chamberlain Line ---
# Both measure dens tip position relative to a skull base reference line
# McGregor: hard palate posterior to most inferior point of occipital curve
# Chamberlain: hard palate posterior to opisthion
if 'dens_tip' in lm and 'opisthion' in lm and 'basion' in lm:
    # Chamberlain line: basion to opisthion (approximation)
    dens_above_chamberlain = signed_dist_above_line(
        lm['dens_tip'], lm['basion'], lm['opisthion']
    )
    measurements['chamberlain_mm'] = round(dens_above_chamberlain, 2)
    # Normal: dens tip should be at or below the line (<=0 or up to 2-3mm above)
    reference_ranges['chamberlain'] = {{'min': -5.0, 'max': 3.0, 'unit': 'mm'}}
    classify(dens_above_chamberlain, -5.0, 3.0, 'chamberlain')

    # McGregor line approximation (using same landmarks)
    measurements['mcgregor_mm'] = round(dens_above_chamberlain, 2)
    reference_ranges['mcgregor'] = {{'min': -5.0, 'max': 4.5, 'unit': 'mm'}}
    classify(dens_above_chamberlain, -5.0, 4.5, 'mcgregor')

# --- Wackenheim Line ---
# Clivus line extended caudally; dens tip should be posterior to it
if 'basion' in lm and 'opisthion' in lm and 'dens_tip' in lm:
    clivus_dir = np.array(lm['basion']) - np.array(lm['opisthion'])
    dens_pt = np.array(lm['dens_tip'])
    basion_pt = np.array(lm['basion'])
    # Project dens onto line perpendicular in sagittal plane
    wack_dist = point_to_line_dist(lm['dens_tip'], lm['opisthion'], lm['basion'])
    measurements['wackenheim_mm'] = round(wack_dist, 2)
    reference_ranges['wackenheim'] = {{'min': 0.0, 'max': 10.0, 'unit': 'mm'}}
    classify(wack_dist, 0.0, 10.0, 'wackenheim')

# Compose final result
result['measurements'] = measurements
result['reference_ranges'] = reference_ranges
result['statuses'] = statuses
result['population'] = population

__execResult = result
"""
    # Replace the final __execResult in landmark_code with the measurement code
    # landmark_code ends with __execResult = result
    # We remove that final assignment and append measurement code
    code = landmark_code.rsplit("__execResult = result", 1)[0]
    return code + measurement_code


# =============================================================================
# Public Tool Functions — CCJ Measurements
# =============================================================================


def measure_ccj_angles(
    segmentation_node_id: str,
    population: str = "adult",
) -> dict[str, Any]:
    """Measure craniocervical junction (CCJ) angles and distances.

    Computes CXA, ADI, Powers ratio, BDI, BAI, Ranawat, McGregor,
    Chamberlain, and Wackenheim measurements from a spine segmentation.
    All calculations use 3D vectors in RAS coordinate space.

    Requires a segmentation containing at least C1 and C2 vertebral
    segments (e.g., from TotalSegmentator).

    Args:
        segmentation_node_id: MRML node ID of the segmentation containing
            vertebral segments (e.g., "vtkMRMLSegmentationNode1")
        population: Patient population for reference ranges -
            "adult" (ADI <= 3mm) or "child" (ADI <= 5mm)

    Returns:
        Dict with landmarks, measurements, reference_ranges, statuses,
        coordinate_system, and segment detection info

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable or computation fails
    """
    segmentation_node_id = validate_mrml_node_id(segmentation_node_id)

    if population not in VALID_POPULATIONS:
        raise ValidationError(
            f"Invalid population '{population}'. "
            f"Must be one of: {', '.join(sorted(VALID_POPULATIONS))}",
            "population",
            population,
        )

    client = get_client()

    safe_node_id = json.dumps(segmentation_node_id)
    safe_population = json.dumps(population)

    python_code = _build_ccj_angles_code(safe_node_id, safe_population)

    try:
        exec_result = client.exec_python(python_code, timeout=LONG_OPERATION_TIMEOUT)

        result_data = _parse_json_result(exec_result.get("result", ""), "CCJ angle measurement")

        logger.info(
            f"CCJ measurements completed for node {segmentation_node_id}: "
            f"{len(result_data.get('measurements', {}))} measurements"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"CCJ measurement failed: {e.message}")
        raise
