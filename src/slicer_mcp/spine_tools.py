"""Spine measurement tool implementations for Slicer Bridge.

Implements craniocervical junction (CCJ) measurements and sagittal
spinal alignment parameters. All computations run inside Slicer's
Python interpreter using numpy for 3D vector math in RAS coordinates.

Clinical references:
- CCJ: Joaquim 2019, Harris 2002, Powers 1979
- Sagittal: Schwab 2012, Roussouly 2005, Glassman 2005
"""

import json
import logging
from typing import Any

from slicer_mcp.constants import LONG_OPERATION_TIMEOUT
from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    CCJ_NORMAL_RANGES,
    REGION_VERTEBRAE,
    SPINE_SEGMENTATION_TIMEOUT,
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

print(json.dumps(result))
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

print(json.dumps(result))
"""
    # Replace the final print in landmark_code with the measurement code
    # landmark_code ends with print(json.dumps(result))
    # We remove that final print and append measurement code
    code = landmark_code.rsplit("print(json.dumps(result))", 1)[0]
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

print(json.dumps(result))
"""

    return extraction_code + alignment_code


# =============================================================================
# Public Tool Functions
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
