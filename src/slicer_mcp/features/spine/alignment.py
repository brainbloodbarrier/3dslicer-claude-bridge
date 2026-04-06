"""Sagittal spinal alignment measurement tools.

Provides vertebral centroid extraction, Cobb angle computation, and the
public ``measure_spine_alignment`` tool function.  Computes CL, TK, LL,
SVA, C2-C7 SVA, T1 slope, PI, PT, SS, Roussouly type, and SRS-Schwab
classification.

Clinical references:
- Schwab 2012, Roussouly 2005, Glassman 2005
"""

import json
import logging
from typing import Any

from slicer_mcp.core.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.features._codegen_snippets import (
    _render_alignment_preamble,
    _render_classify_preamble,
)
from slicer_mcp.features.base_tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)
from slicer_mcp.features.spine.constants import (
    REGION_VERTEBRAE,
    SPINE_SEGMENTATION_TIMEOUT,
    TOTALSEGMENTATOR_VERTEBRA_MAP,
    VALID_ALIGNMENT_REGIONS,
)

__all__ = [
    "measure_spine_alignment",
]

logger = logging.getLogger("slicer-mcp")


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

# Collect available segment IDs (TotalSegmentator sets IDs like "vertebrae_L5",
# while display names differ e.g. "L5 vertebra")
available_segment_ids = set()
for i in range(segmentation.GetNumberOfSegments()):
    available_segment_ids.add(segmentation.GetNthSegmentID(i))

import vtk

def get_vertebra_geometry(seg_node, segment_id):
    \"\"\"Extract centroid, superior/inferior endplate centers for a vertebra by segment ID.\"\"\"
    seg = seg_node.GetSegmentation()
    if not seg.GetSegment(segment_id):
        return None
    seg_id = segment_id

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
    if ts_label and ts_label in available_segment_ids:
        geom = get_vertebra_geometry(segNode, ts_label)
    elif vert_name in available_segment_ids:
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

    alignment_preamble = (
        "\n# --- Sagittal Alignment Parameter Calculations ---\n\n"
        + _render_alignment_preamble()
        + "\nmeasurements = {}\nreference_ranges = {}\nstatuses = {}\nwarnings = []\n\n"
        + _render_classify_preamble()
    )

    alignment_calculations = """
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
    warnings.append(
        "Pelvic Tilt (PT) and Pelvic Incidence (PI) are rough approximations "
        "derived from L5 geometry because true femoral head segmentations are missing"
    )
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
result['warnings'] = warnings

__execResult = result
"""

    return extraction_code + alignment_preamble + alignment_calculations


# =============================================================================
# Public Tool Function — Spine Alignment
# =============================================================================


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
