"""Spine-specific MCP tool implementations for Slicer Bridge.

Provides tools for spine segmentation, vertebral artery segmentation,
bone quality analysis, craniocervical junction (CCJ) measurements, and
sagittal spinal alignment parameters using TotalSegmentator, SlicerVMTK,
BoneTexture, and established clinical grading systems.

Clinical references:
- CCJ: Joaquim 2019, Harris 2002, Powers 1979
- Sagittal: Schwab 2012, Roussouly 2005, Glassman 2005
"""

import json
import logging
from typing import Any

from slicer_mcp.constants import LONG_OPERATION_TIMEOUT, SEGMENTATION_TIMEOUT
from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    CCJ_NORMAL_RANGES,
    PICKHARDT_HU_THRESHOLDS,
    REGION_VERTEBRAE,
    SPINE_REGIONS,
    SPINE_SEGMENTATION_TIMEOUT,
    TOTALSEG_TASK_FULL,
    TOTALSEGMENTATOR_DISC_MAP,
    TOTALSEGMENTATOR_VERTEBRA_MAP,
)
from slicer_mcp.tools import ValidationError, _parse_json_result, validate_mrml_node_id

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# Valid parameter sets
# =============================================================================

VALID_POPULATIONS = frozenset(["adult", "child"])

VALID_CCJ_LANDMARKS = frozenset(
    [
        "basion",
        "opisthion",
        "dens_tip",
        "dens_base",
        "c1_anterior_arch",
        "c1_posterior_arch",
        "c2_posteroinferior",
        "c2_anteroinferior",
        "mcgregor_line_posterior",
        "hard_palate_posterior",
        "foramen_magnum_anterior",
        "foramen_magnum_posterior",
    ]
)

VALID_ALIGNMENT_REGIONS = frozenset(["cervical", "thoracic", "lumbar", "full"])

VALID_ARTERY_SIDES = frozenset(["left", "right", "both"])
"""Valid side parameters for vertebral artery segmentation."""

VALID_BONE_REGIONS = frozenset(["cervical", "thoracic", "lumbar", "full"])
"""Valid spine region parameters for bone quality analysis."""


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

def vec_angle_deg(v1, v2):
    \"\"\"Angle between two 3D vectors in degrees.\"\"\"
    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)
    cos_a = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def point_dist(p1, p2):
    \"\"\"Euclidean distance between two 3D points.\"\"\"
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def point_to_line_dist(point, line_p1, line_p2):
    \"\"\"Distance from a point to a line defined by two points in 3D.\"\"\"
    p = np.array(point)
    a = np.array(line_p1)
    b = np.array(line_p2)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))

def signed_dist_above_line(point, line_p1, line_p2):
    \"\"\"Signed distance of a point above a line in the sagittal plane.

    Positive = above the line (superior), negative = below.
    Uses the S component (index 2) in RAS coordinates.
    \"\"\"
    p = np.array(point)
    a = np.array(line_p1)
    b = np.array(line_p2)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    closest = a + t * ab
    return float(p[2] - closest[2])

lm = landmarks
measurements = {{}}
reference_ranges = {{}}
statuses = {{}}

# Helper to classify measurement status
def classify(value, normal_min, normal_max, name):
    if normal_min <= value <= normal_max:
        statuses[name] = 'normal'
    elif value < normal_min:
        statuses[name] = 'below_normal'
    else:
        statuses[name] = 'above_normal'

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
# Sagittal Alignment Python Code Builders
# =============================================================================


def _build_vertebral_centroid_extraction_code(safe_node_id: str, safe_region: str) -> str:
    """Build Python code to extract vertebral body centroids and endplate lines.

    Args:
        safe_node_id: JSON-escaped segmentation node ID
        safe_region: JSON-escaped region string

    Returns:
        Python code string for Slicer execution
    """
    return f"""
import slicer
import json
import numpy as np

node_id = {safe_node_id}
region = {safe_region}

segNode = slicer.mrmlScene.GetNodeByID(node_id)
if not segNode:
    raise ValueError('Segmentation node not found: ' + node_id)

segmentation = segNode.GetSegmentation()
if not segmentation:
    raise ValueError('No segmentation data in node: ' + node_id)

# TotalSegmentator vertebra label map
ts_map = {json.dumps(TOTALSEGMENTATOR_VERTEBRA_MAP)}
region_vertebrae = {json.dumps(dict(REGION_VERTEBRAE))}

target_vertebrae = region_vertebrae.get(region, region_vertebrae['full'])

# Collect available segment names
available_segments = {{}}
for i in range(segmentation.GetNumberOfSegments()):
    seg = segmentation.GetNthSegment(i)
    available_segments[seg.GetName()] = segmentation.GetNthSegmentID(i)

import vtk

def get_vertebra_geometry(seg_node, segment_name):
    \"\"\"Extract centroid, superior/inferior endplate centers for a vertebra.\"\"\"
    seg = seg_node.GetSegmentation()
    seg_id = None
    for i in range(seg.GetNumberOfSegments()):
        s = seg.GetNthSegment(i)
        if s.GetName() == segment_name:
            seg_id = seg.GetNthSegmentID(i)
            break
    if seg_id is None:
        return None

    labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        seg_node, [seg_id], labelmapNode, None
    )

    ijkToRas = vtk.vtkMatrix4x4()
    labelmapNode.GetIJKToRASMatrix(ijkToRas)
    imageData = labelmapNode.GetImageData()
    dims = imageData.GetDimensions()

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
    centroid = pts.mean(axis=0)

    # Superior endplate: top 10% of points by S coordinate
    s_vals = pts[:, 2]
    s_range = s_vals.max() - s_vals.min()
    if s_range < 1e-6:
        return None

    sup_mask = s_vals >= (s_vals.max() - 0.1 * s_range)
    inf_mask = s_vals <= (s_vals.min() + 0.1 * s_range)

    sup_center = pts[sup_mask].mean(axis=0)
    inf_center = pts[inf_mask].mean(axis=0)

    return {{
        'centroid': centroid.tolist(),
        'superior_endplate': sup_center.tolist(),
        'inferior_endplate': inf_center.tolist(),
        'n_voxels': len(points_ras),
        'height_mm': float(s_range)
    }}

# Extract geometry for each target vertebra
vertebrae_data = {{}}
found_vertebrae = []

for vert_name in target_vertebrae:
    # Try TotalSegmentator names first
    ts_label = None
    for ts_key, std_name in ts_map.items():
        if std_name == vert_name:
            ts_label = ts_key
            break

    geom = None
    if ts_label and ts_label in available_segments:
        geom = get_vertebra_geometry(segNode, ts_label)
    elif vert_name in available_segments:
        geom = get_vertebra_geometry(segNode, vert_name)

    if geom is not None:
        vertebrae_data[vert_name] = geom
        found_vertebrae.append(vert_name)
"""


def _build_sagittal_alignment_code(safe_node_id: str, safe_region: str) -> str:
    """Build Python code to compute sagittal alignment parameters.

    Computes: CL, TK, LL, SVA, C2-C7 SVA, T1 slope, PI, PT, SS,
    Cobb angles, Roussouly classification, and SRS-Schwab classification.

    Args:
        safe_node_id: JSON-escaped segmentation node ID
        safe_region: JSON-escaped region string

    Returns:
        Python code string for Slicer execution
    """
    extraction_code = _build_vertebral_centroid_extraction_code(safe_node_id, safe_region)

    alignment_code = """
# --- Sagittal Alignment Parameter Calculations ---

def vec_angle_deg(v1, v2):
    \"\"\"Angle between two 3D vectors in degrees.\"\"\"
    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)
    cos_a = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def cobb_angle_3d(sup_endplate_center, sup_endplate_vec, inf_endplate_center, inf_endplate_vec):
    \"\"\"Compute 3D Cobb angle between two endplate planes.

    The Cobb angle is the angle between the superior endplate of the
    upper vertebra and the inferior endplate of the lower vertebra.
    \"\"\"
    return vec_angle_deg(sup_endplate_vec, inf_endplate_vec)

def endplate_vector(vertebra_data):
    \"\"\"Get the endplate tilt vector (superior to inferior endplate).\"\"\"
    sup = np.array(vertebra_data['superior_endplate'])
    inf = np.array(vertebra_data['inferior_endplate'])
    return sup - inf

measurements = {}
reference_ranges = {}
statuses = {}

def classify(value, normal_min, normal_max, name):
    if normal_min <= value <= normal_max:
        statuses[name] = 'normal'
    elif value < normal_min:
        statuses[name] = 'below_normal'
    else:
        statuses[name] = 'above_normal'

# --- Cervical Lordosis (CL): C2-C7 Cobb angle ---
if 'C2' in vertebrae_data and 'C7' in vertebrae_data:
    c2_vec = endplate_vector(vertebrae_data['C2'])
    c7_vec = endplate_vector(vertebrae_data['C7'])
    cl = cobb_angle_3d(
        vertebrae_data['C2']['superior_endplate'], c2_vec,
        vertebrae_data['C7']['inferior_endplate'], c7_vec
    )
    measurements['cervical_lordosis_deg'] = round(cl, 1)
    reference_ranges['cervical_lordosis'] = {'min': 20.0, 'max': 40.0, 'unit': 'degrees'}
    classify(cl, 20.0, 40.0, 'cervical_lordosis')

# --- C2-C7 SVA (Sagittal Vertical Axis) ---
if 'C2' in vertebrae_data and 'C7' in vertebrae_data:
    c2_centroid = np.array(vertebrae_data['C2']['centroid'])
    c7_centroid = np.array(vertebrae_data['C7']['centroid'])
    # SVA = horizontal offset (anterior-posterior, A component in RAS = index 1)
    c2c7_sva = float(c2_centroid[1] - c7_centroid[1])
    measurements['C2_C7_SVA_mm'] = round(c2c7_sva, 1)
    reference_ranges['C2_C7_SVA'] = {'min': -20.0, 'max': 20.0, 'unit': 'mm'}
    classify(c2c7_sva, -20.0, 20.0, 'C2_C7_SVA')

# --- T1 Slope ---
if 'T1' in vertebrae_data:
    t1_sup = np.array(vertebrae_data['T1']['superior_endplate'])
    t1_inf = np.array(vertebrae_data['T1']['inferior_endplate'])
    # T1 slope: angle of T1 superior endplate to horizontal
    # In sagittal plane: use A and S components (indices 1, 2 in RAS)
    t1_endplate_vec = t1_sup - t1_inf
    horizontal = np.array([0, 1, 0])  # pure anterior direction
    t1_slope = vec_angle_deg(t1_endplate_vec, horizontal)
    measurements['T1_slope_deg'] = round(t1_slope, 1)
    reference_ranges['T1_slope'] = {'min': 13.0, 'max': 35.0, 'unit': 'degrees'}
    classify(t1_slope, 13.0, 35.0, 'T1_slope')

# --- Thoracic Kyphosis (TK): T1-T12 or T4-T12 Cobb ---
tk_top = None
tk_bottom = None
if 'T1' in vertebrae_data and 'T12' in vertebrae_data:
    tk_top = 'T1'
    tk_bottom = 'T12'
elif 'T4' in vertebrae_data and 'T12' in vertebrae_data:
    tk_top = 'T4'
    tk_bottom = 'T12'

if tk_top and tk_bottom:
    top_vec = endplate_vector(vertebrae_data[tk_top])
    bot_vec = endplate_vector(vertebrae_data[tk_bottom])
    tk = cobb_angle_3d(
        vertebrae_data[tk_top]['superior_endplate'], top_vec,
        vertebrae_data[tk_bottom]['inferior_endplate'], bot_vec
    )
    measurements['thoracic_kyphosis_deg'] = round(tk, 1)
    measurements['thoracic_kyphosis_levels'] = f'{tk_top}-{tk_bottom}'
    reference_ranges['thoracic_kyphosis'] = {'min': 20.0, 'max': 50.0, 'unit': 'degrees'}
    classify(tk, 20.0, 50.0, 'thoracic_kyphosis')

# --- Lumbar Lordosis (LL): L1-S1 or L1-L5 Cobb ---
ll_top = None
ll_bottom = None
if 'L1' in vertebrae_data and 'L5' in vertebrae_data:
    ll_top = 'L1'
    ll_bottom = 'L5'

if ll_top and ll_bottom:
    top_vec = endplate_vector(vertebrae_data[ll_top])
    bot_vec = endplate_vector(vertebrae_data[ll_bottom])
    ll = cobb_angle_3d(
        vertebrae_data[ll_top]['superior_endplate'], top_vec,
        vertebrae_data[ll_bottom]['inferior_endplate'], bot_vec
    )
    measurements['lumbar_lordosis_deg'] = round(ll, 1)
    measurements['lumbar_lordosis_levels'] = f'{ll_top}-{ll_bottom}'
    reference_ranges['lumbar_lordosis'] = {'min': 40.0, 'max': 70.0, 'unit': 'degrees'}
    classify(ll, 40.0, 70.0, 'lumbar_lordosis')

# --- Global SVA (C7 plumb line to S1/L5 posterior superior corner) ---
if 'C7' in vertebrae_data and 'L5' in vertebrae_data:
    c7_cent = np.array(vertebrae_data['C7']['centroid'])
    l5_cent = np.array(vertebrae_data['L5']['centroid'])
    # SVA = AP offset of C7 centroid from posterior L5 (positive = anterior)
    global_sva = float(c7_cent[1] - l5_cent[1])
    measurements['SVA_mm'] = round(global_sva, 1)
    reference_ranges['SVA'] = {'min': -50.0, 'max': 50.0, 'unit': 'mm'}
    classify(global_sva, -50.0, 50.0, 'SVA')

# --- PI (Pelvic Incidence) / PT (Pelvic Tilt) / SS (Sacral Slope) ---
# These require sacrum; approximate from L5 if sacrum not available
if 'L5' in vertebrae_data:
    l5 = vertebrae_data['L5']
    l5_inf = np.array(l5['inferior_endplate'])
    l5_cent = np.array(l5['centroid'])

    # Sacral slope approximation: angle of L5 inferior endplate to horizontal
    l5_endplate_vec = np.array(l5['superior_endplate']) - np.array(l5['inferior_endplate'])
    horizontal = np.array([0, 1, 0])
    ss = vec_angle_deg(l5_endplate_vec, horizontal)
    measurements['sacral_slope_deg'] = round(ss, 1)
    reference_ranges['sacral_slope'] = {'min': 25.0, 'max': 55.0, 'unit': 'degrees'}
    classify(ss, 25.0, 55.0, 'sacral_slope')

    # PI approximation: SS + PT (from geometry)
    # PT from vertical offset of hip axis to sacral endplate midpoint
    # Simplified: use approximate PT based on L5 geometry
    pt = max(0, ss - 20)  # rough approximation
    measurements['pelvic_tilt_deg'] = round(pt, 1)
    reference_ranges['pelvic_tilt'] = {'min': 5.0, 'max': 25.0, 'unit': 'degrees'}
    classify(pt, 5.0, 25.0, 'pelvic_tilt')

    pi = ss + pt
    measurements['pelvic_incidence_deg'] = round(pi, 1)
    reference_ranges['pelvic_incidence'] = {'min': 35.0, 'max': 85.0, 'unit': 'degrees'}
    classify(pi, 35.0, 85.0, 'pelvic_incidence')

# --- Roussouly Classification ---
roussouly_type = None
if 'sacral_slope_deg' in measurements and 'lumbar_lordosis_deg' in measurements:
    ss_val = measurements['sacral_slope_deg']
    ll_val = measurements['lumbar_lordosis_deg']

    if ss_val < 35:
        if ll_val < 45:
            roussouly_type = 1
        else:
            roussouly_type = 2
    else:
        if ll_val >= 60:
            roussouly_type = 4
        else:
            roussouly_type = 3

    measurements['roussouly_type'] = roussouly_type

# --- SRS-Schwab Classification ---
schwab = {}
if 'SVA_mm' in measurements:
    sva_val = measurements['SVA_mm']
    if sva_val < 40:
        schwab['SVA_modifier'] = '0'
    elif sva_val < 95:
        schwab['SVA_modifier'] = '+'
    else:
        schwab['SVA_modifier'] = '++'

if 'pelvic_tilt_deg' in measurements:
    pt_val = measurements['pelvic_tilt_deg']
    if pt_val < 20:
        schwab['PT_modifier'] = '0'
    elif pt_val < 30:
        schwab['PT_modifier'] = '+'
    else:
        schwab['PT_modifier'] = '++'

if 'pelvic_incidence_deg' in measurements and 'lumbar_lordosis_deg' in measurements:
    pi_ll = measurements['pelvic_incidence_deg'] - measurements['lumbar_lordosis_deg']
    measurements['PI_LL_mismatch_deg'] = round(pi_ll, 1)
    if abs(pi_ll) < 10:
        schwab['PI_LL_modifier'] = '0'
    elif abs(pi_ll) < 20:
        schwab['PI_LL_modifier'] = '+'
    else:
        schwab['PI_LL_modifier'] = '++'

if schwab:
    measurements['schwab_classification'] = schwab

# --- Compose final result ---
result = {
    'success': True,
    'node_id': node_id,
    'node_name': segNode.GetName(),
    'region': region,
    'coordinate_system': 'RAS',
    'vertebrae_found': found_vertebrae,
    'vertebrae_data': {k: {
        'centroid': v['centroid'],
        'superior_endplate': v['superior_endplate'],
        'inferior_endplate': v['inferior_endplate'],
        'height_mm': v['height_mm']
    } for k, v in vertebrae_data.items()},
    'measurements': measurements,
    'reference_ranges': reference_ranges,
    'statuses': statuses
}

__execResult = result
"""

    return extraction_code + alignment_code


# =============================================================================
# Code Generation Helpers (Spine Segmentation)
# =============================================================================


def _build_totalseg_subprocess_block(
    volume_var: str = "volume_node",
    seg_var: str = "seg_node",
    task: str = "total",
    timeout_s: int = SPINE_SEGMENTATION_TIMEOUT - 60,
) -> str:
    """Build Python code block for TotalSegmentator subprocess auto-segmentation.

    Generates code that checks if ``seg_node_id`` is already set (reuse existing
    segmentation) or runs TotalSegmentator as a subprocess with the
    ``resource_tracker`` hang workaround (``start_new_session`` + ``killpg``).

    The generated code assumes ``{volume_var}`` and ``seg_node_id`` are already
    defined, and sets ``{seg_var}`` as the output segmentation node.

    Args:
        volume_var: Variable name of the input volume in the generated code.
        seg_var: Variable name for the output segmentation node.
        task: TotalSegmentator task name (e.g. ``"total"``, ``"total_mr"``).
        timeout_s: Subprocess timeout in seconds (default: SPINE_SEGMENTATION_TIMEOUT - 60).

    Returns:
        Python code string for embedding in Slicer exec code.
    """
    safe_task = json.dumps(task)

    return f"""
# --- Segmentation: reuse existing or auto-segment via subprocess ---
if seg_node_id:
    {seg_var} = slicer.mrmlScene.GetNodeByID(seg_node_id)
    if not {seg_var}:
        raise ValueError("Segmentation node not found: " + seg_node_id)
else:
    import time as _ts_time
    import os as _ts_os
    import subprocess as _ts_subprocess
    import signal as _ts_signal
    import shutil as _ts_shutil
    import sysconfig as _ts_sysconfig

    {seg_var} = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    {seg_var}.SetName({volume_var}.GetName() + '_auto_seg')

    _ts_proc = None
    _ts_tempFolder = None

    try:
        _ts_tempFolder = slicer.util.tempDirectory()
        _ts_inputFile = _ts_os.path.join(_ts_tempFolder, "ts-input.nii")
        _ts_outputFile = _ts_os.path.join(_ts_tempFolder, "segmentation.nii")
        _ts_outputFolder = _ts_os.path.join(_ts_tempFolder, "segmentation")

        # Export volume to NIfTI
        _ts_storageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        _ts_storageNode.SetFileName(_ts_inputFile)
        _ts_storageNode.UseCompressionOff()
        _ts_storageNode.WriteData({volume_var})
        _ts_storageNode.UnRegister(None)

        # Build TotalSegmentator CLI command
        _ts_pythonSlicer = _ts_shutil.which('PythonSlicer')
        if not _ts_pythonSlicer:
            raise RuntimeError("PythonSlicer not found in PATH")

        from TotalSegmentator import TotalSegmentatorLogic as _ts_Logic
        _ts_exec = _ts_os.path.join(
            _ts_sysconfig.get_path('scripts'),
            _ts_Logic.executableName("TotalSegmentator"),
        )

        _ts_cmd = [_ts_pythonSlicer, _ts_exec,
                    "-i", _ts_inputFile, "-o", _ts_outputFolder,
                    "--device", "cpu", "--ml", "--task", {safe_task}, "--fast"]

        # Run as subprocess in new process group (resource_tracker hang workaround)
        _ts_proc = _ts_subprocess.Popen(
            _ts_cmd, stdout=_ts_subprocess.DEVNULL, stderr=_ts_subprocess.PIPE,
            start_new_session=True,
        )
        _ts_timeout = {timeout_s}
        _ts_poll_interval = 5
        _ts_elapsed = 0

        while _ts_elapsed < _ts_timeout:
            _ts_ret = _ts_proc.poll()
            if _ts_ret is not None:
                if _ts_ret != 0:
                    _ts_err = _ts_proc.stderr.read(8192).decode(errors='replace')[-500:]
                    raise RuntimeError(
                        f"TotalSegmentator exited with code {{_ts_ret}}: {{_ts_err}}"
                    )
                break
            if (_ts_os.path.exists(_ts_outputFile)
                    and _ts_os.path.getsize(_ts_outputFile) > 1000):
                _ts_time.sleep(3)
                break
            _ts_time.sleep(_ts_poll_interval)
            _ts_elapsed += _ts_poll_interval

        # Kill process group if still running (resource_tracker hang)
        if _ts_proc.poll() is None:
            try:
                _ts_os.killpg(_ts_os.getpgid(_ts_proc.pid), _ts_signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            _ts_time.sleep(1)
            if _ts_proc.poll() is None:
                try:
                    _ts_os.killpg(_ts_os.getpgid(_ts_proc.pid), _ts_signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        if not _ts_os.path.exists(_ts_outputFile):
            raise RuntimeError("TotalSegmentator did not produce output within timeout")

        # Import segmentation result
        _ts_logic = _ts_Logic()
        _ts_logic.readSegmentation({seg_var}, _ts_outputFile, {safe_task})

        {seg_var}.SetNodeReferenceID(
            {seg_var}.GetReferenceImageGeometryReferenceRole(), {volume_var}.GetID())
        {seg_var}.SetReferenceImageGeometryParameterFromVolumeNode({volume_var})

    except Exception as _ts_e:
        slicer.mrmlScene.RemoveNode({seg_var})
        raise ValueError(f"TotalSegmentator failed: {{_ts_e}}")
    finally:
        if _ts_proc is not None and _ts_proc.poll() is None:
            try:
                _ts_os.killpg(_ts_os.getpgid(_ts_proc.pid), _ts_signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if _ts_tempFolder is not None and _ts_os.path.isdir(_ts_tempFolder):
            _ts_shutil.rmtree(_ts_tempFolder, ignore_errors=True)
"""


def _build_spine_segmentation_code(
    safe_node_id: str,
    safe_region: str,
    include_discs: bool,
    include_spinal_cord: bool,
) -> str:
    """Build Python code for TotalSegmentator spine segmentation.

    Generates Slicer-Python code that:
    1. Validates the input volume exists
    2. Runs TotalSegmentator with the appropriate task
    3. Filters results to the requested region
    4. Returns structured JSON with vertebrae list

    Args:
        safe_node_id: JSON-escaped MRML node ID string
        safe_region: JSON-escaped region string
        include_discs: Whether to include intervertebral discs
        include_spinal_cord: Whether to include spinal cord segmentation

    Returns:
        Python code string to execute in Slicer
    """
    safe_include_discs = str(include_discs)
    safe_include_spinal_cord = str(include_spinal_cord)

    # Build subset list based on region to avoid segmenting all 117 structures.
    # Using subset makes TotalSegmentator output only the requested segments,
    # which is dramatically faster for import back into Slicer.
    region_verts = REGION_VERTEBRAE.get(safe_region.strip('"'), ())
    # Filter to only labels that TotalSegmentator can actually segment (excludes S1)
    vertebrae_labels = [
        v for v in region_verts if f"vertebrae_{v}" in TOTALSEGMENTATOR_VERTEBRA_MAP
    ]
    subset_items = [f"vertebrae_{v}" for v in vertebrae_labels]
    if include_discs:
        disc_names = list(TOTALSEGMENTATOR_DISC_MAP.keys())
        subset_items.extend(disc_names)
    if include_spinal_cord:
        subset_items.append("spinal_cord")
    safe_subset = json.dumps(subset_items)

    # Inject constants into generated code via json.dumps() (codegen pattern)
    safe_vertebra_map = json.dumps(TOTALSEGMENTATOR_VERTEBRA_MAP)
    safe_disc_map = json.dumps(TOTALSEGMENTATOR_DISC_MAP)
    safe_task = json.dumps(TOTALSEG_TASK_FULL)
    safe_timeout = SPINE_SEGMENTATION_TIMEOUT - 60  # reserve 60s for cleanup/import
    anatomical_order = json.dumps(list(TOTALSEGMENTATOR_VERTEBRA_MAP.values()))

    return f"""
import slicer
import time
import json
import os
import subprocess
import signal
import shutil
import sysconfig

input_node_id = {safe_node_id}
region = {safe_region}
include_discs = {safe_include_discs}
include_spinal_cord = {safe_include_spinal_cord}
subset = {safe_subset}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

# Create output segmentation node
outputSeg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
outputSeg.SetName(inputVolume.GetName() + "_spine_seg")

start_time = time.time()
proc = None
tempFolder = None

try:
    # Create temp directory for TotalSegmentator I/O
    tempFolder = slicer.util.tempDirectory()
    inputFile = os.path.join(tempFolder, "ts-input.nii")
    outputFile = os.path.join(tempFolder, "segmentation.nii")
    outputFolder = os.path.join(tempFolder, "segmentation")

    # Export volume to NIfTI
    storageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
    storageNode.SetFileName(inputFile)
    storageNode.UseCompressionOff()
    storageNode.WriteData(inputVolume)
    storageNode.UnRegister(None)

    # Build TotalSegmentator CLI command
    pythonSlicer = shutil.which('PythonSlicer')
    if not pythonSlicer:
        raise RuntimeError("PythonSlicer not found")

    from TotalSegmentator import TotalSegmentatorLogic
    tsExec = os.path.join(
        sysconfig.get_path('scripts'),
        TotalSegmentatorLogic.executableName("TotalSegmentator"),
    )

    cmd = [pythonSlicer, tsExec, "-i", inputFile, "-o", outputFolder,
           "--device", "cpu", "--ml", "--task", {safe_task}, "--fast",
           "--roi_subset"] + subset

    # Run as subprocess in new session; poll for output file instead of waiting
    # (workaround: TotalSegmentator hangs on multiprocessing.resource_tracker after saving)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True,  # critical: own process group so killpg won't kill Slicer
    )
    timeout_s = {safe_timeout}
    poll_interval = 5
    elapsed_wait = 0

    while elapsed_wait < timeout_s:
        # Check if process finished normally
        ret = proc.poll()
        if ret is not None:
            if ret != 0:
                stderr_out = proc.stderr.read(8192).decode(errors='replace')[-500:]
                raise RuntimeError(f"TotalSegmentator exited with code {{ret}}: {{stderr_out}}")
            break
        # Check if output file exists (segmentation complete, process may hang)
        if os.path.exists(outputFile) and os.path.getsize(outputFile) > 1000:
            # Wait a bit for file to finish writing
            time.sleep(3)
            break
        time.sleep(poll_interval)
        elapsed_wait += poll_interval

    # Kill the process group if still running (resource_tracker hang workaround)
    # Safe because start_new_session=True gives it its own process group
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(1)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    if not os.path.exists(outputFile):
        raise RuntimeError("TotalSegmentator did not produce output file within timeout")

    # Import segmentation result into Slicer using TotalSegmentator module
    logic = TotalSegmentatorLogic()
    logic.readSegmentation(outputSeg, outputFile, "total")

    # Set source volume reference
    outputSeg.SetNodeReferenceID(
        outputSeg.GetReferenceImageGeometryReferenceRole(), inputVolume.GetID())
    outputSeg.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

except Exception as e:
    slicer.mrmlScene.RemoveNode(outputSeg)
    raise ValueError(f"TotalSegmentator failed: {{e}}")
finally:
    # Kill subprocess if still alive (handles exception during poll loop)
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # Clean up temp files (both success and failure paths)
    if tempFolder is not None and os.path.isdir(tempFolder):
        shutil.rmtree(tempFolder, ignore_errors=True)

elapsed = time.time() - start_time

VERTEBRA_MAP = {safe_vertebra_map}

DISC_MAP = {safe_disc_map}

seg = outputSeg.GetSegmentation()
found_vertebrae = []
found_discs = []
found_other = []

# All segments returned should be relevant (subset filters at segmentation time)
for i in range(seg.GetNumberOfSegments()):
    seg_id = seg.GetNthSegmentID(i)
    seg_name = seg.GetNthSegment(i).GetName()

    if seg_name in VERTEBRA_MAP:
        label = VERTEBRA_MAP[seg_name]
        found_vertebrae.append({{"segment_id": seg_id, "label": label, "name": seg_name}})
    elif seg_name in DISC_MAP:
        disc_label = DISC_MAP[seg_name]
        found_discs.append({{"segment_id": seg_id, "label": disc_label, "name": seg_name}})
    elif seg_name == "spinal_cord":
        found_other.append({{"segment_id": seg_id, "label": "spinal_cord", "name": seg_name}})

# Sort vertebrae by anatomical order
ANATOMICAL_ORDER = {anatomical_order}
order_map = {{v: idx for idx, v in enumerate(ANATOMICAL_ORDER)}}
found_vertebrae.sort(key=lambda x: order_map.get(x["label"], 999))

result = {{
    "success": True,
    "input_node_id": input_node_id,
    "region": region,
    "output_segmentation_id": outputSeg.GetID(),
    "output_segmentation_name": outputSeg.GetName(),
    "vertebrae_count": len(found_vertebrae),
    "vertebrae": found_vertebrae,
    "discs": found_discs,
    "other_structures": found_other,
    "processing_time_seconds": round(elapsed, 2),
}}

__execResult = result
"""


# =============================================================================
# Vertebral Artery Segmentation (SlicerVMTK)
# =============================================================================


def _build_vesselness_code(safe_node_id: str, safe_side: str) -> str:
    """Build Python code for vesselness filter + level set segmentation.

    Uses SlicerVMTK's vesselness filtering and level set evolution
    to segment vertebral arteries from CTA volumes.

    Args:
        safe_node_id: JSON-escaped MRML node ID of input CTA volume
        safe_side: JSON-escaped side parameter ("left", "right", or "both")

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json
import time

input_node_id = {safe_node_id}
side = {safe_side}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

start_time = time.time()

# --- Step 1: Vesselness filtering via SlicerVMTK ---
try:
    import VesselnessFiltering
except ImportError:
    raise ValueError(
        "SlicerVMTK extension not installed. "
        "Install via Extension Manager: SlicerVMTK"
    )

vesselnessLogic = VesselnessFiltering.VesselnessFilteringLogic()

# Create output vesselness volume
vesselnessVolume = slicer.mrmlScene.AddNewNodeByClass(
    'vtkMRMLScalarVolumeNode'
)
vesselnessVolume.SetName(inputVolume.GetName() + "_vesselness")

# Configure vesselness filter for vertebral arteries
# Typical VA diameter: 2-5 mm
vesselnessLogic.computeVesselnessVolume(
    inputVolume, vesselnessVolume,
    minimumDiameterMm=1.5,
    maximumDiameterMm=6.0,
    alpha=0.3,
    beta=0.3,
    contrastMeasure=200
)

# --- Step 2: Level set segmentation ---
import LevelSetSegmentation
lsLogic = LevelSetSegmentation.LevelSetSegmentationLogic()

# Create segmentation node for results
segNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
segNode.SetName(inputVolume.GetName() + "_VA_seg")

# Use vesselness volume to initialise level set evolution
lsLogic.performEvolution(
    vesselnessVolume, segNode,
    inflationWeight=1.0,
    curvatureWeight=0.7,
    attractionWeight=1.0,
    iterationCount=20
)

# --- Step 3: Centerline extraction ---
import ExtractCenterline
clLogic = ExtractCenterline.ExtractCenterlineLogic()

# Create centerline model
centerlineModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
centerlineModel.SetName(inputVolume.GetName() + "_VA_centerline")

# Extract centerline from segmentation
clLogic.extractCenterline(segNode, centerlineModel)

# --- Step 4: Compute diameters along centerline ---
diameters = []
if centerlineModel.GetPolyData():
    pd = centerlineModel.GetPolyData()
    radiusArray = pd.GetPointData().GetArray("Radius")
    if radiusArray:
        n_points = radiusArray.GetNumberOfTuples()
        step = max(1, n_points // 20)  # Sample ~20 points
        for i in range(0, n_points, step):
            r = radiusArray.GetValue(i)
            diameters.append(round(r * 2.0, 2))  # diameter = 2 * radius

elapsed = time.time() - start_time

# Build result
result = {{
    "success": True,
    "input_node_id": input_node_id,
    "side": side,
    "model_node_id": segNode.GetID(),
    "model_node_name": segNode.GetName(),
    "centerline_node_id": centerlineModel.GetID(),
    "centerline_node_name": centerlineModel.GetName(),
    "vesselness_node_id": vesselnessVolume.GetID(),
    "diameters_mm": diameters,
    "mean_diameter_mm": round(sum(diameters) / len(diameters), 2) if diameters else 0.0,
    "processing_time_seconds": round(elapsed, 2)
}}

__execResult = result
"""


def _build_vesselness_with_seeds_code(safe_node_id: str, safe_side: str, safe_seeds: str) -> str:
    """Build Python code for seeded vertebral artery segmentation.

    Uses user-supplied seed points to guide the level set initialization,
    improving accuracy over the automatic approach.

    Args:
        safe_node_id: JSON-escaped MRML node ID of input CTA volume
        safe_side: JSON-escaped side parameter
        safe_seeds: JSON-escaped list of [x, y, z] seed point coordinates

    Returns:
        Python code string for execution in Slicer
    """
    return f"""
import slicer
import json
import time
import vtk

input_node_id = {safe_node_id}
side = {safe_side}
seed_points = {safe_seeds}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

start_time = time.time()

# --- Step 1: Vesselness filtering ---
try:
    import VesselnessFiltering
except ImportError:
    raise ValueError(
        "SlicerVMTK extension not installed. "
        "Install via Extension Manager: SlicerVMTK"
    )

vesselnessLogic = VesselnessFiltering.VesselnessFilteringLogic()

vesselnessVolume = slicer.mrmlScene.AddNewNodeByClass(
    'vtkMRMLScalarVolumeNode'
)
vesselnessVolume.SetName(inputVolume.GetName() + "_vesselness")

vesselnessLogic.computeVesselnessVolume(
    inputVolume, vesselnessVolume,
    minimumDiameterMm=1.5,
    maximumDiameterMm=6.0,
    alpha=0.3,
    beta=0.3,
    contrastMeasure=200
)

# --- Step 2: Create seed fiducials ---
seedNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
seedNode.SetName("VA_seeds")
for pt in seed_points:
    seedNode.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))

# --- Step 3: Level set segmentation with seeds ---
import LevelSetSegmentation
lsLogic = LevelSetSegmentation.LevelSetSegmentationLogic()

segNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
segNode.SetName(inputVolume.GetName() + "_VA_seg")

lsLogic.performEvolution(
    vesselnessVolume, segNode,
    seedNode=seedNode,
    inflationWeight=1.0,
    curvatureWeight=0.7,
    attractionWeight=1.0,
    iterationCount=20
)

# --- Step 4: Centerline extraction ---
import ExtractCenterline
clLogic = ExtractCenterline.ExtractCenterlineLogic()

centerlineModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
centerlineModel.SetName(inputVolume.GetName() + "_VA_centerline")

clLogic.extractCenterline(segNode, centerlineModel)

# --- Step 5: Compute diameters along centerline ---
diameters = []
if centerlineModel.GetPolyData():
    pd = centerlineModel.GetPolyData()
    radiusArray = pd.GetPointData().GetArray("Radius")
    if radiusArray:
        n_points = radiusArray.GetNumberOfTuples()
        step = max(1, n_points // 20)
        for i in range(0, n_points, step):
            r = radiusArray.GetValue(i)
            diameters.append(round(r * 2.0, 2))

# Clean up seed node
slicer.mrmlScene.RemoveNode(seedNode)

elapsed = time.time() - start_time

result = {{
    "success": True,
    "input_node_id": input_node_id,
    "side": side,
    "model_node_id": segNode.GetID(),
    "model_node_name": segNode.GetName(),
    "centerline_node_id": centerlineModel.GetID(),
    "centerline_node_name": centerlineModel.GetName(),
    "vesselness_node_id": vesselnessVolume.GetID(),
    "diameters_mm": diameters,
    "mean_diameter_mm": round(sum(diameters) / len(diameters), 2) if diameters else 0.0,
    "seed_count": len(seed_points),
    "processing_time_seconds": round(elapsed, 2)
}}

__execResult = result
"""


def _validate_seed_points(seed_points: list[list[float]]) -> list[list[float]]:
    """Validate seed point coordinates for vertebral artery segmentation.

    Each seed point must be a list of exactly 3 floats (RAS coordinates).

    Args:
        seed_points: List of [x, y, z] coordinate lists

    Returns:
        Validated seed points

    Raises:
        ValidationError: If seed points are malformed
    """
    if not seed_points:
        raise ValidationError("Seed points list cannot be empty", "seed_points", "[]")

    if len(seed_points) > 50:
        raise ValidationError(
            f"Too many seed points ({len(seed_points)}). Maximum is 50.",
            "seed_points",
            f"[{len(seed_points)} points]",
        )

    for i, pt in enumerate(seed_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 3:
            raise ValidationError(
                f"Seed point {i} must be [x, y, z] coordinates, got: {pt}",
                "seed_points",
                str(pt),
            )
        for j, coord in enumerate(pt):
            if not isinstance(coord, (int, float)):
                raise ValidationError(
                    f"Seed point {i} coordinate {j} must be numeric, got: {type(coord).__name__}",
                    "seed_points",
                    str(coord),
                )

    return seed_points


# =============================================================================
# Bone Quality Analysis (BoneTexture Extension)
# =============================================================================


def _build_bone_quality_code(safe_node_id: str, safe_seg_id: str, safe_region: str) -> str:
    """Build Python code for per-vertebra bone quality analysis.

    Uses the BoneTexture extension for trabecular metrics and Hounsfield
    unit analysis for osteoporosis classification (Pickhardt criteria).

    Args:
        safe_node_id: JSON-escaped MRML node ID of input CT volume
        safe_seg_id: JSON-escaped MRML node ID of vertebra segmentation
        safe_region: JSON-escaped spine region string

    Returns:
        Python code string for execution in Slicer
    """
    pickhardt_normal = PICKHARDT_HU_THRESHOLDS["normal_min"]
    pickhardt_osteopenia = PICKHARDT_HU_THRESHOLDS["osteopenia_min"]

    return f"""
import slicer
import json
import time
import numpy as np

input_node_id = {safe_node_id}
seg_node_id = {safe_seg_id}
region = {safe_region}

inputVolume = slicer.mrmlScene.GetNodeByID(input_node_id)
if not inputVolume:
    raise ValueError(f"Input volume not found: {{input_node_id}}")

segNode = slicer.mrmlScene.GetNodeByID(seg_node_id)
if not segNode:
    raise ValueError(f"Segmentation node not found: {{seg_node_id}}")

start_time = time.time()

# Check for BoneTexture extension
try:
    import BoneTexture
    has_bone_texture = True
except ImportError:
    has_bone_texture = False

segmentation = segNode.GetSegmentation()
n_segments = segmentation.GetNumberOfSegments()

# Pickhardt HU thresholds for osteoporosis classification
NORMAL_MIN_HU = {pickhardt_normal}
OSTEOPENIA_MIN_HU = {pickhardt_osteopenia}

vertebrae_results = []

for i in range(n_segments):
    segment = segmentation.GetNthSegment(i)
    seg_name = segment.GetName()
    seg_id = segmentation.GetNthSegmentID(i)

    # Create labelmap for this segment
    labelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    labelNode.SetName(seg_name + "_label")

    seg_ids = vtk.vtkStringArray()
    seg_ids.InsertNextValue(seg_id)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segNode, seg_ids, labelNode, inputVolume
    )

    # Extract voxel values within the segment ROI
    import vtk.util.numpy_support as vtk_np

    labelArray = vtk_np.vtk_to_numpy(
        labelNode.GetImageData().GetPointData().GetScalars()
    )
    volumeArray = vtk_np.vtk_to_numpy(
        inputVolume.GetImageData().GetPointData().GetScalars()
    )

    mask = labelArray > 0
    roi_values = volumeArray[mask].astype(float)

    vertebra_data = {{
        "name": seg_name,
        "segment_id": seg_id,
        "voxel_count": int(mask.sum()),
    }}

    if len(roi_values) > 0:
        mean_hu = float(np.mean(roi_values))
        std_hu = float(np.std(roi_values))

        vertebra_data["mean_hu"] = round(mean_hu, 1)
        vertebra_data["std_hu"] = round(std_hu, 1)
        vertebra_data["min_hu"] = round(float(np.min(roi_values)), 1)
        vertebra_data["max_hu"] = round(float(np.max(roi_values)), 1)

        # Pickhardt classification based on mean trabecular HU
        if mean_hu >= NORMAL_MIN_HU:
            classification = "normal"
        elif mean_hu >= OSTEOPENIA_MIN_HU:
            classification = "osteopenia"
        else:
            classification = "osteoporosis"
        vertebra_data["classification"] = classification

        # BoneTexture metrics if extension is available
        if has_bone_texture:
            try:
                btLogic = BoneTexture.BoneTextureLogic()
                glcm = btLogic.computeGLCM(inputVolume, labelNode)

                if glcm:
                    vertebra_data["bv_tv"] = round(glcm.get("BV/TV", 0.0), 4)
                    vertebra_data["tb_th_mm"] = round(glcm.get("Tb.Th", 0.0), 3)
                    vertebra_data["tb_sp_mm"] = round(glcm.get("Tb.Sp", 0.0), 3)
            except Exception as bt_err:
                vertebra_data["bone_texture_error"] = str(bt_err)
    else:
        vertebra_data["mean_hu"] = None
        vertebra_data["classification"] = "insufficient_data"

    vertebrae_results.append(vertebra_data)

    # Clean up temporary label node
    slicer.mrmlScene.RemoveNode(labelNode)

elapsed = time.time() - start_time

# Summary statistics
hu_values = [v["mean_hu"] for v in vertebrae_results if v["mean_hu"] is not None]
classifications = [v["classification"] for v in vertebrae_results]

result = {{
    "success": True,
    "input_node_id": input_node_id,
    "segmentation_node_id": seg_node_id,
    "region": region,
    "vertebrae": vertebrae_results,
    "vertebrae_count": len(vertebrae_results),
    "summary": {{
        "mean_hu_overall": round(sum(hu_values) / len(hu_values), 1) if hu_values else None,
        "normal_count": classifications.count("normal"),
        "osteopenia_count": classifications.count("osteopenia"),
        "osteoporosis_count": classifications.count("osteoporosis"),
        "has_bone_texture_metrics": has_bone_texture,
    }},
    "processing_time_seconds": round(elapsed, 2)
}}

__execResult = result
"""


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


def measure_spine_alignment(
    segmentation_node_id: str,
    region: str = "full",
) -> dict[str, Any]:
    """Measure sagittal spinal alignment parameters.

    Computes cervical lordosis, thoracic kyphosis, lumbar lordosis,
    SVA, C2-C7 SVA, T1 slope, pelvic incidence, pelvic tilt,
    sacral slope, PI-LL mismatch, Roussouly type, and SRS-Schwab
    classification from a spine segmentation.

    All Cobb angles are computed in 3D. Centroids and endplate lines
    are extracted from the segmentation's labelmap representation.

    Requires a segmentation containing vertebral body segments
    (e.g., from TotalSegmentator).

    Args:
        segmentation_node_id: MRML node ID of the segmentation containing
            vertebral segments (e.g., "vtkMRMLSegmentationNode1")
        region: Spine region to analyze - "cervical" (C1-C7),
            "thoracic" (T1-T12), "lumbar" (L1-L5), or "full" (all)

    Returns:
        Dict with vertebrae_found, vertebrae_data, measurements,
        reference_ranges, statuses, Roussouly type, and Schwab classification

    Raises:
        ValidationError: If inputs are invalid
        SlicerConnectionError: If Slicer is not reachable or computation fails
    """
    segmentation_node_id = validate_mrml_node_id(segmentation_node_id)

    if region not in VALID_ALIGNMENT_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. "
            f"Must be one of: {', '.join(sorted(VALID_ALIGNMENT_REGIONS))}",
            "region",
            region,
        )

    client = get_client()

    safe_node_id = json.dumps(segmentation_node_id)
    safe_region = json.dumps(region)

    python_code = _build_sagittal_alignment_code(safe_node_id, safe_region)

    try:
        exec_result = client.exec_python(python_code, timeout=SPINE_SEGMENTATION_TIMEOUT)

        result_data = _parse_json_result(
            exec_result.get("result", ""), "spine alignment measurement"
        )

        logger.info(
            f"Spine alignment completed for node {segmentation_node_id} "
            f"(region={region}): {len(result_data.get('measurements', {}))} parameters, "
            f"{len(result_data.get('vertebrae_found', []))} vertebrae"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Spine alignment measurement failed: {e.message}")
        raise


# =============================================================================
# Spine Segmentation Tool
# =============================================================================


def segment_spine(
    input_node_id: str,
    region: str = "full",
    include_discs: bool = False,
    include_spinal_cord: bool = False,
) -> dict[str, Any]:
    """Segment spine structures from a CT volume using TotalSegmentator.

    LONG OPERATION: This may take 1-5 minutes depending on hardware and region.

    Identifies and labels individual vertebrae, optionally including
    intervertebral discs and spinal cord. Uses TotalSegmentator AI
    segmentation with region-based filtering.

    Expected durations:
    - GPU: ~1-2 minutes
    - CPU: ~3-5 minutes

    Args:
        input_node_id: MRML node ID of input CT volume
        region: Spine region to segment - "cervical", "thoracic",
            "lumbar", or "full" (all vertebrae)
        include_discs: If True, also segment intervertebral discs
            (requires "total" task, may increase processing time)
        include_spinal_cord: If True, also segment the spinal cord
            (requires "total" task, may increase processing time)

    Returns:
        Dict with:
        - output_segmentation_id: MRML node ID of output segmentation
        - vertebrae_count: Number of vertebrae found
        - vertebrae: List of dicts with segment_id, label, name
        - discs: List of disc segments (if include_discs=True)
        - other_structures: List of other structures (if include_spinal_cord=True)
        - processing_time_seconds: Actual processing time
        - long_operation: Metadata about this being a long operation

    Tip:
        Pass ``output_segmentation_id`` from the result to CT/MRI diagnostic
        tools via ``segmentation_node_id`` to avoid re-running TotalSegmentator
        on each tool call (~10x faster on CPU).

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or processing fails
    """
    # Validate input node ID
    input_node_id = validate_mrml_node_id(input_node_id)

    # Validate region
    if region not in SPINE_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(SPINE_REGIONS))}",
            "region",
            region,
        )

    client = get_client()

    # JSON-escape all values for safe injection into Python code
    safe_node_id = json.dumps(input_node_id)
    safe_region = json.dumps(region)

    python_code = _build_spine_segmentation_code(
        safe_node_id, safe_region, include_discs, include_spinal_cord
    )

    try:
        exec_result = client.exec_python(python_code, timeout=SPINE_SEGMENTATION_TIMEOUT)

        result_data = _parse_json_result(
            exec_result.get("result", ""), f"spine segmentation ({region})"
        )

        # Add long_operation metadata
        result_data["long_operation"] = {
            "type": "spine_segmentation",
            "region": region,
            "timeout_seconds": SPINE_SEGMENTATION_TIMEOUT,
            "typical_duration": "1-2 minutes (GPU), 3-5 minutes (CPU)",
        }

        logger.info(
            f"Spine segmentation completed: region={region}, "
            f"vertebrae={result_data.get('vertebrae_count', 0)}, "
            f"time={result_data.get('processing_time_seconds', 0)}s"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Spine segmentation failed: {e.message}")
        raise


# =============================================================================
# Vertebral Artery Segmentation Tool
# =============================================================================


def segment_vertebral_artery(
    input_node_id: str,
    side: str = "both",
    seed_points: list[list[float]] | None = None,
) -> dict[str, Any]:
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

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or processing fails
    """
    # Validate input node ID
    input_node_id = validate_mrml_node_id(input_node_id)

    # Validate side parameter
    if side not in VALID_ARTERY_SIDES:
        raise ValidationError(
            f"Invalid side '{side}'. Must be one of: {', '.join(sorted(VALID_ARTERY_SIDES))}",
            "side",
            side,
        )

    # Validate seed points if provided
    if seed_points is not None:
        seed_points = _validate_seed_points(seed_points)

    client = get_client()

    # Safe strings for Python code (defense-in-depth)
    safe_node_id = json.dumps(input_node_id)
    safe_side = json.dumps(side)

    if seed_points is not None:
        safe_seeds = json.dumps(seed_points)
        python_code = _build_vesselness_with_seeds_code(safe_node_id, safe_side, safe_seeds)
    else:
        python_code = _build_vesselness_code(safe_node_id, safe_side)

    try:
        exec_result = client.exec_python(python_code, timeout=SEGMENTATION_TIMEOUT)

        # Parse JSON result
        result_data = _parse_json_result(
            exec_result.get("result", ""), "vertebral artery segmentation"
        )

        # Add long_operation metadata
        result_data["long_operation"] = {
            "type": "vertebral_artery_segmentation",
            "timeout_seconds": SEGMENTATION_TIMEOUT,
            "typical_duration": "1-5 minutes",
        }

        logger.info(
            f"Vertebral artery segmentation completed: side={side}, "
            f"mean_diameter={result_data.get('mean_diameter_mm', 0)}mm, "
            f"time={result_data.get('processing_time_seconds', 0)}s"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Vertebral artery segmentation failed: {e.message}")
        raise


# =============================================================================
# Bone Quality Analysis Tool
# =============================================================================


def analyze_bone_quality(
    input_node_id: str,
    segmentation_node_id: str,
    region: str = "lumbar",
) -> dict[str, Any]:
    """Analyze bone quality per vertebra using CT Hounsfield units and BoneTexture metrics.

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
        region: Spine region to analyze - "cervical", "thoracic", "lumbar", or "full"

    Returns:
        Dict with per-vertebra metrics (mean_hu, classification, optional BV/TV,
        Tb.Th, Tb.Sp), summary counts, and processing_time_seconds

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or analysis fails
    """
    # Validate inputs
    input_node_id = validate_mrml_node_id(input_node_id)
    segmentation_node_id = validate_mrml_node_id(segmentation_node_id)

    if region not in VALID_BONE_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(VALID_BONE_REGIONS))}",
            "region",
            region,
        )

    client = get_client()

    # Safe strings for Python code (defense-in-depth)
    safe_node_id = json.dumps(input_node_id)
    safe_seg_id = json.dumps(segmentation_node_id)
    safe_region = json.dumps(region)

    python_code = _build_bone_quality_code(safe_node_id, safe_seg_id, safe_region)

    try:
        exec_result = client.exec_python(python_code, timeout=SPINE_SEGMENTATION_TIMEOUT)

        # Parse JSON result
        result_data = _parse_json_result(exec_result.get("result", ""), "bone quality analysis")

        # Add long_operation metadata
        result_data["long_operation"] = {
            "type": "bone_quality_analysis",
            "timeout_seconds": SPINE_SEGMENTATION_TIMEOUT,
            "typical_duration": "30s - 3 minutes",
        }

        logger.info(
            f"Bone quality analysis completed: region={region}, "
            f"vertebrae={result_data.get('vertebrae_count', 0)}, "
            f"time={result_data.get('processing_time_seconds', 0)}s"
        )

        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Bone quality analysis failed: {e.message}")
        raise
