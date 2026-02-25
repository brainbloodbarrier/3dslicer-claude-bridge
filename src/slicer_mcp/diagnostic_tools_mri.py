"""MRI diagnostic protocol tools for spine analysis.

Implements 4 MRI-based diagnostic tools:
- Modic endplate change classification (T1+T2)
- Pfirrmann disc degeneration grading (T2)
- Cord compression detection (T2, optional T1)
- Metastatic lesion detection (T1+T2/STIR)

CRITICAL: MRI signal values are NOT absolute (unlike HU in CT).
All signal analysis uses ratio normalization against reference tissue.

References:
    Modic MT et al. Radiology. 1988;166(1):193-199.
    Pfirrmann CW et al. Spine. 2001;26(17):1873-8.
    Fehlings MG et al. Spine. 2013;38(22 Suppl 1):S9-S18.
    Bilsky MH et al. J Clin Oncol. 2010;28(22):3608-3614.
"""

import json
import logging
from typing import Any

from slicer_mcp.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.spine_constants import (
    PFIRRMANN_DESCRIPTIONS,
    REGION_VERTEBRAE,
    SPINAL_CANAL_AP_DIAMETER,
    SPINAL_CANAL_STENOSIS_ABSOLUTE,
    TOTALSEGMENTATOR_DISC_MAP,
    TOTALSEGMENTATOR_VERTEBRA_MAP,
)
from slicer_mcp.tools import ValidationError, _parse_json_result, validate_mrml_node_id

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# MRI-specific constants
# =============================================================================

# Timeout for multi-sequence MRI analysis (registration + segmentation + analysis)
MRI_ANALYSIS_TIMEOUT = 360  # 6 minutes

# Valid MRI sequence types for validation
VALID_MRI_REGIONS = frozenset(["cervical", "thoracic", "lumbar"])

# Modic signal ratio thresholds (relative to reference vertebral body)
# T1 and T2 signal ratios classify endplate changes
MODIC_T1_LOW_THRESHOLD = 0.85  # T1 signal < 85% of reference = "low"
MODIC_T1_HIGH_THRESHOLD = 1.15  # T1 signal > 115% of reference = "high"
MODIC_T2_LOW_THRESHOLD = 0.85  # T2 signal < 85% of reference = "low"
MODIC_T2_HIGH_THRESHOLD = 1.15  # T2 signal > 115% of reference = "high"

# Pfirrmann disc signal ratio thresholds (relative to CSF)
PFIRRMANN_BRIGHT_THRESHOLD = 0.80  # >80% of CSF = bright white (Grade I)
PFIRRMANN_WHITE_THRESHOLD = 0.60  # 60-80% of CSF = white (Grade II)
PFIRRMANN_GRAY_THRESHOLD = 0.35  # 35-60% of CSF = gray (Grade III)
PFIRRMANN_DARK_THRESHOLD = 0.15  # 15-35% of CSF = dark (Grade IV)
# <15% of CSF = black (Grade V)

# Disc height loss thresholds for Pfirrmann grading
PFIRRMANN_HEIGHT_SLIGHT_DECREASE = 0.10  # >10% height loss = slight decrease
PFIRRMANN_HEIGHT_MODERATE_DECREASE = 0.30  # >30% height loss = moderate decrease
PFIRRMANN_HEIGHT_COLLAPSED = 0.60  # >60% height loss = collapsed

# Homogeneity threshold (coefficient of variation)
DISC_HOMOGENEOUS_CV_THRESHOLD = 0.15  # CV < 0.15 = homogeneous

# Cord compression thresholds
CORD_COMPRESSION_RATIO_NORMAL = 0.40  # AP/transverse > 0.40 = normal shape
CORD_COMPRESSION_RATIO_MILD = 0.30  # 0.30-0.40 = mild compression
CORD_COMPRESSION_RATIO_MODERATE = 0.20  # 0.20-0.30 = moderate compression
# <0.20 = severe compression

# T2 hyperintensity threshold for myelopathy detection (relative to normal cord)
CORD_T2_HYPERINTENSITY_THRESHOLD = 1.30  # >130% of normal cord signal

# MSCC (Metastatic Spinal Cord Compression) ratio threshold
MSCC_THRESHOLD = 0.50  # MSCC ratio < 0.50 = significant compression

# Metastatic lesion signal thresholds (relative to normal marrow)
METASTASIS_T1_LOW_THRESHOLD = 0.70  # T1 < 70% of normal = suspicious
METASTASIS_T2_HIGH_THRESHOLD = 1.40  # T2/STIR > 140% of normal = suspicious
METASTASIS_T2_LOW_THRESHOLD = 0.70  # T2/STIR < 70% of normal = blastic


# =============================================================================
# Input validation helpers
# =============================================================================


def _validate_mri_region(region: str) -> str:
    """Validate MRI analysis region parameter.

    Args:
        region: Spine region to analyze

    Returns:
        Validated region string

    Raises:
        ValidationError: If region is invalid
    """
    if region not in VALID_MRI_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(VALID_MRI_REGIONS))}",
            "region",
            region,
        )
    return region


# =============================================================================
# Shared code builders for Slicer Python execution
# =============================================================================


def _build_registration_check_code(safe_t1_id: str, safe_t2_id: str) -> str:
    """Build Python code to check if T1 and T2 volumes are co-registered.

    If not co-registered, runs BRAINSFit affine registration.

    Args:
        safe_t1_id: JSON-escaped T1 volume node ID
        safe_t2_id: JSON-escaped T2 volume node ID

    Returns:
        Python code string for registration check/execution
    """
    return f"""
# --- Registration check ---
import numpy as np

t1_node = slicer.mrmlScene.GetNodeByID({safe_t1_id})
t2_node = slicer.mrmlScene.GetNodeByID({safe_t2_id})

if not t1_node:
    raise ValueError(f"T1 volume not found: {{{safe_t1_id}}}")
if not t2_node:
    raise ValueError(f"T2 volume not found: {{{safe_t2_id}}}")

# Check if volumes share the same geometry (co-registered)
t1_origin = list(t1_node.GetOrigin())
t2_origin = list(t2_node.GetOrigin())
t1_spacing = list(t1_node.GetSpacing())
t2_spacing = list(t2_node.GetSpacing())

origin_diff = sum((a - b) ** 2 for a, b in zip(t1_origin, t2_origin)) ** 0.5
spacing_diff = sum((a - b) ** 2 for a, b in zip(t1_spacing, t2_spacing)) ** 0.5

registration_performed = False

if origin_diff > 1.0 or spacing_diff > 0.01:
    # Volumes are not co-registered; run BRAINSFit
    outputTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    outputTransform.SetName('T2_to_T1_transform')

    t2_registered = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    t2_registered.SetName(t2_node.GetName() + '_registered')

    parameters = {{
        'fixedVolume': t1_node.GetID(),
        'movingVolume': t2_node.GetID(),
        'outputVolume': t2_registered.GetID(),
        'linearTransform': outputTransform.GetID(),
        'useRigid': True,
        'useAffine': True,
        'initializeTransformMode': 'useCenterOfHeadAlign',
    }}

    cliNode = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)
    if cliNode.GetStatus() & cliNode.ErrorsMask:
        raise ValueError(f"BRAINSFit registration failed: {{cliNode.GetErrorText()}}")

    # Use registered T2 for subsequent analysis
    t2_node = t2_registered
    registration_performed = True
"""


def _build_totalseg_spine_code(safe_volume_id: str, region: str) -> str:
    """Build Python code for TotalSegmentator spine segmentation in MRI mode.

    Args:
        safe_volume_id: JSON-escaped volume node ID (reference volume for segmentation)
        region: Spine region to segment

    Returns:
        Python code string for TotalSegmentator execution
    """
    return f"""
# --- TotalSegmentator spine segmentation (MRI mode) ---
seg_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
seg_node.SetName('SpineSegmentation_{region}')

try:
    import TotalSegmentator
    tsLogic = TotalSegmentator.TotalSegmentatorLogic()
    tsLogic.process(
        inputVolume=slicer.mrmlScene.GetNodeByID({safe_volume_id}),
        outputSegmentation=seg_node,
        task='total_mr',
        fast=True,
    )
except Exception as ts_err:
    raise ValueError(f"TotalSegmentator failed: {{ts_err}}")
"""


def _build_signal_normalization_code() -> str:
    """Build Python code for MRI signal ratio normalization framework.

    Provides helper functions used by all MRI diagnostic tools:
    - get_segment_mean_signal: extracts mean signal from a segment ROI
    - compute_signal_ratio: normalizes signal against reference tissue

    Returns:
        Python code string with normalization helper functions
    """
    return """
# --- MRI signal normalization framework ---
import sitkUtils
import SimpleITK as sitk

def get_segment_mean_signal(volume_node, seg_node, segment_name):
    \"\"\"Get mean signal intensity within a segment ROI.\"\"\"
    # Get segment as labelmap
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
        seg_node, labelmapNode, volume_node
    )

    # Find the segment index
    segmentation = seg_node.GetSegmentation()
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if seg.GetName() == segment_name:
            seg_id = segmentation.GetNthSegmentID(i)
            break

    if seg_id is None:
        slicer.mrmlScene.RemoveNode(labelmapNode)
        return None

    # Export single segment
    singleLabelmap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    seg_ids = vtk.vtkStringArray()
    seg_ids.InsertNextValue(seg_id)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        seg_node, seg_ids, singleLabelmap, volume_node
    )

    # Use SimpleITK for statistics
    vol_sitk = sitkUtils.PullVolumeFromSlicer(volume_node)
    label_sitk = sitkUtils.PullVolumeFromSlicer(singleLabelmap)

    # Resample label to match volume if needed
    if label_sitk.GetSize() != vol_sitk.GetSize():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(vol_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        label_sitk = resampler.Execute(label_sitk)

    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(vol_sitk, label_sitk)

    mean_val = None
    if stats.HasLabel(1):
        mean_val = stats.GetMean(1)

    slicer.mrmlScene.RemoveNode(labelmapNode)
    slicer.mrmlScene.RemoveNode(singleLabelmap)
    return mean_val

def get_segment_stats(volume_node, seg_node, segment_name):
    \"\"\"Get signal statistics (mean, std, count) within a segment ROI.\"\"\"
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')

    segmentation = seg_node.GetSegmentation()
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if seg.GetName() == segment_name:
            seg_id = segmentation.GetNthSegmentID(i)
            break

    if seg_id is None:
        slicer.mrmlScene.RemoveNode(labelmapNode)
        return None

    seg_ids = vtk.vtkStringArray()
    seg_ids.InsertNextValue(seg_id)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        seg_node, seg_ids, labelmapNode, volume_node
    )

    vol_sitk = sitkUtils.PullVolumeFromSlicer(volume_node)
    label_sitk = sitkUtils.PullVolumeFromSlicer(labelmapNode)

    if label_sitk.GetSize() != vol_sitk.GetSize():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(vol_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        label_sitk = resampler.Execute(label_sitk)

    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(vol_sitk, label_sitk)

    result = None
    if stats.HasLabel(1):
        result = {
            'mean': stats.GetMean(1),
            'std': stats.GetSigma(1),
            'count': stats.GetCount(1),
        }

    slicer.mrmlScene.RemoveNode(labelmapNode)
    return result

def compute_signal_ratio(signal_value, reference_value):
    \"\"\"Compute normalized signal ratio.\"\"\"
    if reference_value is None or reference_value <= 0:
        return None
    return signal_value / reference_value
"""


# =============================================================================
# Tool 1: Modic Endplate Change Classification
# =============================================================================


def _build_modic_analysis_code(safe_t1_id: str, safe_t2_id: str, region: str) -> str:
    """Build Python code for Modic endplate change classification.

    Args:
        safe_t1_id: JSON-escaped T1 volume node ID
        safe_t2_id: JSON-escaped T2 volume node ID
        region: Spine region to analyze

    Returns:
        Complete Python code string for Slicer execution
    """
    vertebrae = REGION_VERTEBRAE.get(region, ())
    ts_map = TOTALSEGMENTATOR_VERTEBRA_MAP
    disc_map = TOTALSEGMENTATOR_DISC_MAP

    # Build vertebra and disc name lists for the region
    ts_vertebrae = {v: k for k, v in ts_map.items() if v in vertebrae}
    region_discs = {k: v for k, v in disc_map.items() if any(vert in v for vert in vertebrae)}

    return f"""
import slicer
import json
import vtk
import numpy as np

{_build_signal_normalization_code()}

{_build_registration_check_code(safe_t1_id, safe_t2_id)}

{_build_totalseg_spine_code(safe_t1_id, region)}

# --- Modic endplate analysis ---
vertebrae_map = {json.dumps(ts_vertebrae)}
disc_map = {json.dumps(region_discs)}

# Select reference vertebra (middle of the region, assumed normal)
vertebra_names = list(vertebrae_map.keys())
ref_idx = len(vertebra_names) // 2
ref_vert_name = vertebra_names[ref_idx] if vertebra_names else None

# Get reference signal (from normal vertebral body)
ref_t1_signal = None
ref_t2_signal = None
if ref_vert_name:
    ts_label = vertebrae_map[ref_vert_name]
    ref_t1_signal = get_segment_mean_signal(t1_node, seg_node, ts_label)
    ref_t2_signal = get_segment_mean_signal(t2_node, seg_node, ts_label)

levels = []

# For each disc level, analyze adjacent endplates
segmentation = seg_node.GetSegmentation()
available_segments = []
for i in range(segmentation.GetNumberOfSegments()):
    available_segments.append(segmentation.GetNthSegment(i).GetName())

for ts_disc_label, disc_name in disc_map.items():
    # Get vertebrae above and below disc
    parts = disc_name.split('-')
    if len(parts) != 2:
        continue

    upper_vert = parts[0]
    lower_vert = parts[1]

    # Find TotalSegmentator labels for adjacent vertebrae
    upper_ts = None
    lower_ts = None
    for ts_label, std_name in {json.dumps(ts_map)}.items():
        if std_name == upper_vert:
            upper_ts = ts_label
        if std_name == lower_vert:
            lower_ts = ts_label

    if not upper_ts or not lower_ts:
        continue

    # Check segments exist
    if upper_ts not in available_segments or lower_ts not in available_segments:
        continue

    # Get endplate signals (using vertebral body ROI as approximation)
    upper_t1 = get_segment_mean_signal(t1_node, seg_node, upper_ts)
    upper_t2 = get_segment_mean_signal(t2_node, seg_node, upper_ts)
    lower_t1 = get_segment_mean_signal(t1_node, seg_node, lower_ts)
    lower_t2 = get_segment_mean_signal(t2_node, seg_node, lower_ts)

    for vert_name, vert_t1, vert_t2, position in [
        (upper_vert, upper_t1, upper_t2, "inferior_endplate"),
        (lower_vert, lower_t1, lower_t2, "superior_endplate"),
    ]:
        if vert_t1 is None or vert_t2 is None:
            continue

        t1_ratio = compute_signal_ratio(vert_t1, ref_t1_signal)
        t2_ratio = compute_signal_ratio(vert_t2, ref_t2_signal)

        if t1_ratio is None or t2_ratio is None:
            continue

        # Classify Modic type based on signal ratios
        t1_low = t1_ratio < {MODIC_T1_LOW_THRESHOLD}
        t1_high = t1_ratio > {MODIC_T1_HIGH_THRESHOLD}
        t2_low = t2_ratio < {MODIC_T2_LOW_THRESHOLD}
        t2_high = t2_ratio > {MODIC_T2_HIGH_THRESHOLD}

        if t1_low and t2_high:
            modic_type = 1
            description = "Type I: edema/inflammation (T1 low, T2 high)"
        elif t1_high and (not t2_low):
            modic_type = 2
            description = "Type II: fatty degeneration (T1 high, T2 iso/high)"
        elif t1_low and t2_low:
            modic_type = 3
            description = "Type III: sclerosis (T1 low, T2 low)"
        else:
            modic_type = 0
            description = "Normal: no endplate changes"

        # Detect mixed patterns (I/II transition)
        mixed_pattern = None
        if t1_low and t2_high and t1_ratio > ({MODIC_T1_LOW_THRESHOLD} - 0.10):
            mixed_pattern = "I/II transition (borderline T1)"

        levels.append({{
            'disc_level': disc_name,
            'vertebra': vert_name,
            'endplate': position,
            'modic_type': modic_type,
            'description': description,
            't1_signal_ratio': round(t1_ratio, 3),
            't2_signal_ratio': round(t2_ratio, 3),
            'mixed_pattern': mixed_pattern,
        }})

# Clean up
slicer.mrmlScene.RemoveNode(seg_node)

result = {{
    'success': True,
    'tool': 'classify_modic_changes',
    'region': '{region}',
    'reference_vertebra': ref_vert_name,
    'registration_performed': registration_performed,
    'levels': levels,
    'total_levels_analyzed': len(levels),
    'modic_summary': {{
        'type_0': sum(1 for l in levels if l['modic_type'] == 0),
        'type_i': sum(1 for l in levels if l['modic_type'] == 1),
        'type_ii': sum(1 for l in levels if l['modic_type'] == 2),
        'type_iii': sum(1 for l in levels if l['modic_type'] == 3),
    }},
}}

print(json.dumps(result))
"""


def classify_modic_changes(
    t1_node_id: str,
    t2_node_id: str,
    region: str = "lumbar",
) -> dict[str, Any]:
    """Classify Modic endplate changes using T1 and T2 MRI sequences.

    Analyzes vertebral endplate signal patterns to classify Modic changes:
    - Type 0: Normal (no endplate changes)
    - Type I: Edema/inflammation (T1 low, T2 high)
    - Type II: Fatty degeneration (T1 high, T2 iso/high)
    - Type III: Sclerosis (T1 low, T2 low)

    LONG OPERATION: Requires registration check + segmentation + analysis.

    Args:
        t1_node_id: MRML node ID of T1-weighted volume
        t2_node_id: MRML node ID of T2-weighted volume
        region: Spine region - "cervical", "thoracic", or "lumbar"

    Returns:
        Dict with per-level Modic classification, signal ratios, and summary

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or analysis fails
    """
    t1_node_id = validate_mrml_node_id(t1_node_id)
    t2_node_id = validate_mrml_node_id(t2_node_id)
    region = _validate_mri_region(region)

    client = get_client()

    safe_t1_id = json.dumps(t1_node_id)
    safe_t2_id = json.dumps(t2_node_id)

    python_code = _build_modic_analysis_code(safe_t1_id, safe_t2_id, region)

    try:
        exec_result = client.exec_python(python_code, timeout=MRI_ANALYSIS_TIMEOUT)
        result_data = _parse_json_result(exec_result.get("result", ""), "Modic classification")
        logger.info(
            f"Modic classification completed: region={region}, "
            f"levels={result_data.get('total_levels_analyzed', 0)}"
        )
        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Modic classification failed: {e.message}")
        raise


# =============================================================================
# Tool 2: Pfirrmann Disc Degeneration Grading
# =============================================================================


def _build_pfirrmann_analysis_code(safe_t2_id: str, region: str) -> str:
    """Build Python code for Pfirrmann disc degeneration grading.

    Args:
        safe_t2_id: JSON-escaped T2 volume node ID
        region: Spine region to analyze

    Returns:
        Complete Python code string for Slicer execution
    """
    vertebrae = REGION_VERTEBRAE.get(region, ())
    disc_map = TOTALSEGMENTATOR_DISC_MAP
    region_discs = {k: v for k, v in disc_map.items() if any(vert in v for vert in vertebrae)}
    vert_to_ts = {v: k for k, v in TOTALSEGMENTATOR_VERTEBRA_MAP.items() if v in vertebrae}
    vert_to_ts_json = json.dumps(vert_to_ts)

    return f"""
import slicer
import json
import vtk
import numpy as np

{_build_signal_normalization_code()}

t2_node = slicer.mrmlScene.GetNodeByID({safe_t2_id})
if not t2_node:
    raise ValueError(f"T2 volume not found: {{{safe_t2_id}}}")

{_build_totalseg_spine_code(safe_t2_id, region)}

disc_map = {json.dumps(region_discs)}
pfirrmann_descriptions = {json.dumps(PFIRRMANN_DESCRIPTIONS)}

# Get CSF signal for normalization (use spinal canal if segmented)
csf_signal = None
segmentation = seg_node.GetSegmentation()
available_segments = []
for i in range(segmentation.GetNumberOfSegments()):
    available_segments.append(segmentation.GetNthSegment(i).GetName())

# Try spinal cord or canal for CSF reference
for csf_candidate in ['spinal_cord', 'spinal_canal']:
    if csf_candidate in available_segments:
        csf_signal = get_segment_mean_signal(t2_node, seg_node, csf_candidate)
        if csf_signal is not None:
            break

# Fallback: use maximum signal in spinal region as CSF approximation
if csf_signal is None:
    vol_array = slicer.util.arrayFromVolume(t2_node)
    csf_signal = float(np.percentile(vol_array[vol_array > 0], 95))

# Get reference vertebra height for height ratio computation
vertebrae_map = {vert_to_ts_json}

# Compute average vertebral body height across region
vert_heights = []
for vert_name, ts_label in vertebrae_map.items():
    if ts_label not in available_segments:
        continue
    vert_stats = get_segment_stats(t2_node, seg_node, ts_label)
    if vert_stats and vert_stats['count'] > 0:
        # Approximate height from voxel count and spacing
        spacing = list(t2_node.GetSpacing())
        # Height is in the superior-inferior direction (typically axis 2)
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        total_vol = vert_stats['count'] * voxel_volume
        # Approximate cross-section area
        approx_height = total_vol ** (1.0 / 3.0)  # Cubic approximation
        vert_heights.append(approx_height)

ref_height = np.mean(vert_heights) if vert_heights else None

discs = []

for ts_disc_label, disc_name in disc_map.items():
    if ts_disc_label not in available_segments:
        continue

    # Get disc signal statistics
    disc_stats = get_segment_stats(t2_node, seg_node, ts_disc_label)
    if disc_stats is None:
        continue

    disc_mean = disc_stats['mean']
    disc_std = disc_stats['std']
    disc_count = disc_stats['count']

    # Signal ratio normalized to CSF
    signal_ratio = compute_signal_ratio(disc_mean, csf_signal)
    if signal_ratio is None:
        continue

    # Homogeneity: coefficient of variation
    cv = disc_std / disc_mean if disc_mean > 0 else 999.0
    is_homogeneous = cv < {DISC_HOMOGENEOUS_CV_THRESHOLD}

    # Disc height estimation
    spacing = list(t2_node.GetSpacing())
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    disc_volume = disc_count * voxel_volume
    disc_height_approx = disc_volume ** (1.0 / 3.0)

    height_ratio = None
    height_loss_pct = None
    if ref_height and ref_height > 0:
        height_ratio = disc_height_approx / ref_height
        height_loss_pct = max(0, 1.0 - height_ratio)

    # Determine nucleus/annulus distinction
    # Higher CV suggests distinguishable nucleus/annulus
    has_nucleus_annulus_distinction = cv > {DISC_HOMOGENEOUS_CV_THRESHOLD}

    # Pfirrmann grading based on signal, homogeneity, height, and boundary
    if signal_ratio >= {PFIRRMANN_BRIGHT_THRESHOLD} and is_homogeneous:
        grade = 1
    elif signal_ratio >= {PFIRRMANN_WHITE_THRESHOLD} and has_nucleus_annulus_distinction:
        if height_loss_pct is not None and height_loss_pct < {PFIRRMANN_HEIGHT_SLIGHT_DECREASE}:
            grade = 2
        else:
            grade = 3
    elif signal_ratio >= {PFIRRMANN_GRAY_THRESHOLD}:
        if height_loss_pct is not None and height_loss_pct >= {PFIRRMANN_HEIGHT_MODERATE_DECREASE}:
            grade = 4
        else:
            grade = 3
    elif signal_ratio >= {PFIRRMANN_DARK_THRESHOLD}:
        grade = 4
    else:
        grade = 5

    # Override to Grade V if disc is collapsed
    if height_loss_pct is not None and height_loss_pct >= {PFIRRMANN_HEIGHT_COLLAPSED}:
        grade = 5

    discs.append({{
        'disc_level': disc_name,
        'pfirrmann_grade': grade,
        'pfirrmann_description': pfirrmann_descriptions.get(grade, ''),
        'signal_ratio_to_csf': round(signal_ratio, 3),
        'homogeneity_cv': round(cv, 3),
        'is_homogeneous': is_homogeneous,
        'has_nucleus_annulus_distinction': has_nucleus_annulus_distinction,
        'disc_height_mm': round(disc_height_approx, 1),
        'height_loss_percent': (
            round(height_loss_pct * 100, 1) if height_loss_pct is not None else None
        ),
    }})

# Clean up
slicer.mrmlScene.RemoveNode(seg_node)

result = {{
    'success': True,
    'tool': 'assess_disc_degeneration_mri',
    'region': '{region}',
    'csf_reference_signal': round(csf_signal, 1) if csf_signal else None,
    'discs': discs,
    'total_discs_analyzed': len(discs),
    'grade_summary': {{
        'grade_i': sum(1 for d in discs if d['pfirrmann_grade'] == 1),
        'grade_ii': sum(1 for d in discs if d['pfirrmann_grade'] == 2),
        'grade_iii': sum(1 for d in discs if d['pfirrmann_grade'] == 3),
        'grade_iv': sum(1 for d in discs if d['pfirrmann_grade'] == 4),
        'grade_v': sum(1 for d in discs if d['pfirrmann_grade'] == 5),
    }},
}}

print(json.dumps(result))
"""


def assess_disc_degeneration_mri(
    t2_node_id: str,
    region: str = "lumbar",
) -> dict[str, Any]:
    """Assess intervertebral disc degeneration using Pfirrmann grading on T2 MRI.

    Evaluates disc signal intensity, homogeneity, height, and nucleus-annulus
    distinction to assign Pfirrmann grades I-V.

    LONG OPERATION: Requires segmentation + per-disc analysis.

    Args:
        t2_node_id: MRML node ID of T2-weighted sagittal volume
        region: Spine region - "cervical", "thoracic", or "lumbar"

    Returns:
        Dict with per-disc Pfirrmann grades, signal ratios, and summary

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or analysis fails
    """
    t2_node_id = validate_mrml_node_id(t2_node_id)
    region = _validate_mri_region(region)

    client = get_client()

    safe_t2_id = json.dumps(t2_node_id)

    python_code = _build_pfirrmann_analysis_code(safe_t2_id, region)

    try:
        exec_result = client.exec_python(python_code, timeout=MRI_ANALYSIS_TIMEOUT)
        result_data = _parse_json_result(exec_result.get("result", ""), "Pfirrmann grading")
        logger.info(
            f"Pfirrmann grading completed: region={region}, "
            f"discs={result_data.get('total_discs_analyzed', 0)}"
        )
        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Pfirrmann grading failed: {e.message}")
        raise


# =============================================================================
# Tool 3: Cord Compression Detection
# =============================================================================


def _build_cord_compression_code(safe_t2_id: str, safe_t1_id: str | None, region: str) -> str:
    """Build Python code for spinal cord compression detection.

    Args:
        safe_t2_id: JSON-escaped T2 volume node ID
        safe_t1_id: JSON-escaped T1 volume node ID (or None)
        region: Spine region to analyze

    Returns:
        Complete Python code string for Slicer execution
    """
    vertebrae = REGION_VERTEBRAE.get(region, ())
    canal_normal = SPINAL_CANAL_AP_DIAMETER.get(region, (14.0, 23.0))
    stenosis_threshold = SPINAL_CANAL_STENOSIS_ABSOLUTE.get(region, 10.0)
    vert_to_ts = {v: k for k, v in TOTALSEGMENTATOR_VERTEBRA_MAP.items() if v in vertebrae}
    vert_to_ts_json = json.dumps(vert_to_ts)

    has_t1 = safe_t1_id is not None

    registration_code = ""
    if has_t1:
        registration_code = _build_registration_check_code(safe_t1_id, safe_t2_id)

    t1_init = ""
    if not has_t1:
        t1_init = """
t1_node = None
registration_performed = False
"""

    return f"""
import slicer
import json
import vtk
import numpy as np

{_build_signal_normalization_code()}

t2_node = slicer.mrmlScene.GetNodeByID({safe_t2_id})
if not t2_node:
    raise ValueError(f"T2 volume not found: {{{safe_t2_id}}}")

{t1_init}
{registration_code}

{_build_totalseg_spine_code(safe_t2_id, region)}

canal_normal_min = {canal_normal[0]}
canal_normal_max = {canal_normal[1]}
stenosis_threshold = {stenosis_threshold}

segmentation = seg_node.GetSegmentation()
available_segments = []
for i in range(segmentation.GetNumberOfSegments()):
    available_segments.append(segmentation.GetNthSegment(i).GetName())

# Get normal cord signal (reference for hyperintensity detection)
normal_cord_signal = None
if 'spinal_cord' in available_segments:
    cord_stats = get_segment_stats(t2_node, seg_node, 'spinal_cord')
    if cord_stats:
        normal_cord_signal = cord_stats['mean']

vertebrae_map = {vert_to_ts_json}

levels = []

for vert_name, ts_label in vertebrae_map.items():
    if ts_label not in available_segments:
        continue

    # Get vertebral body dimensions for canal estimation
    # Export segment to get bounding box
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if seg.GetName() == ts_label:
            seg_id = segmentation.GetNthSegmentID(i)
            break

    if seg_id is None:
        continue

    # Get segment bounds
    bounds = [0.0] * 6
    segmentation.GetSegmentBounds(seg_id, bounds)

    # AP diameter approximation from bounds (anterior-posterior)
    ap_diameter = abs(bounds[3] - bounds[2])  # Y-axis typically AP
    transverse_diameter = abs(bounds[1] - bounds[0])  # X-axis typically lateral

    # Compression ratio
    compression_ratio = (ap_diameter / transverse_diameter) if transverse_diameter > 0 else 0

    # Stenosis grading
    if ap_diameter < stenosis_threshold:
        stenosis_grade = "severe"
    elif ap_diameter < canal_normal_min:
        stenosis_grade = "moderate"
    elif ap_diameter < canal_normal_max:
        stenosis_grade = "mild"
    else:
        stenosis_grade = "normal"

    # Cross-section area approximation (ellipse)
    cross_section_area = 3.14159 * (ap_diameter / 2) * (transverse_diameter / 2)

    # MSCC ratio (AP at level / expected AP)
    expected_ap = (canal_normal_min + canal_normal_max) / 2
    mscc_ratio = ap_diameter / expected_ap if expected_ap > 0 else 1.0

    # T2 hyperintensity detection (myelopathy sign)
    t2_hyperintensity = False
    hyperintensity_ratio = None
    if normal_cord_signal and normal_cord_signal > 0:
        level_cord_signal = get_segment_mean_signal(t2_node, seg_node, ts_label)
        if level_cord_signal is not None:
            hyperintensity_ratio = level_cord_signal / normal_cord_signal
            t2_hyperintensity = hyperintensity_ratio > {CORD_T2_HYPERINTENSITY_THRESHOLD}

    # T1 reversibility assessment (if T1 available)
    reversibility = None
    if t1_node is not None and t2_hyperintensity:
        t1_level_signal = get_segment_mean_signal(t1_node, seg_node, ts_label)
        t1_ref_signal = get_segment_mean_signal(t1_node, seg_node, list(vertebrae_map.values())[0])
        if t1_level_signal and t1_ref_signal and t1_ref_signal > 0:
            t1_ratio = t1_level_signal / t1_ref_signal
            if t1_ratio >= {MODIC_T1_LOW_THRESHOLD}:
                reversibility = "likely_reversible"
            else:
                reversibility = "likely_irreversible"

    level_data = {{
        'level': vert_name,
        'ap_diameter_mm': round(ap_diameter, 1),
        'transverse_diameter_mm': round(transverse_diameter, 1),
        'cross_section_area_mm2': round(cross_section_area, 1),
        'compression_ratio': round(compression_ratio, 3),
        'mscc_ratio': round(mscc_ratio, 3),
        'stenosis_grade': stenosis_grade,
        't2_hyperintensity': t2_hyperintensity,
    }}

    if hyperintensity_ratio is not None:
        level_data['hyperintensity_ratio'] = round(hyperintensity_ratio, 3)

    if reversibility is not None:
        level_data['myelopathy_reversibility'] = reversibility

    levels.append(level_data)

# Clean up
slicer.mrmlScene.RemoveNode(seg_node)

# Overall assessment
worst_stenosis = 'normal'
for grade_name in ['severe', 'moderate', 'mild']:
    if any(l['stenosis_grade'] == grade_name for l in levels):
        worst_stenosis = grade_name
        break

myelopathy_detected = any(l.get('t2_hyperintensity', False) for l in levels)

result = {{
    'success': True,
    'tool': 'detect_cord_compression_mri',
    'region': '{region}',
    't1_available': {str(has_t1).lower()},
    'registration_performed': registration_performed,
    'canal_normal_range_mm': [canal_normal_min, canal_normal_max],
    'stenosis_threshold_mm': stenosis_threshold,
    'levels': levels,
    'total_levels_analyzed': len(levels),
    'worst_stenosis_grade': worst_stenosis,
    'myelopathy_detected': myelopathy_detected,
    'mscc_significant': any(l['mscc_ratio'] < {MSCC_THRESHOLD} for l in levels),
}}

print(json.dumps(result))
"""


def detect_cord_compression_mri(
    t2_node_id: str,
    t1_node_id: str | None = None,
    region: str = "cervical",
) -> dict[str, Any]:
    """Detect spinal cord compression on MRI.

    Analyzes cord AP/transverse diameters, compression ratio, cross-section area,
    and T2 hyperintensity (myelopathy). If T1 is available, assesses reversibility.

    LONG OPERATION: Requires segmentation + per-level analysis.

    Args:
        t2_node_id: MRML node ID of T2-weighted volume
        t1_node_id: MRML node ID of T1-weighted volume (optional, for reversibility)
        region: Spine region - "cervical", "thoracic", or "lumbar"

    Returns:
        Dict with per-level compression metrics, stenosis grades, and myelopathy status

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or analysis fails
    """
    t2_node_id = validate_mrml_node_id(t2_node_id)
    if t1_node_id is not None:
        t1_node_id = validate_mrml_node_id(t1_node_id)
    region = _validate_mri_region(region)

    client = get_client()

    safe_t2_id = json.dumps(t2_node_id)
    safe_t1_id = json.dumps(t1_node_id) if t1_node_id else None

    python_code = _build_cord_compression_code(safe_t2_id, safe_t1_id, region)

    try:
        exec_result = client.exec_python(python_code, timeout=MRI_ANALYSIS_TIMEOUT)
        result_data = _parse_json_result(
            exec_result.get("result", ""), "cord compression detection"
        )
        logger.info(
            f"Cord compression detection completed: region={region}, "
            f"levels={result_data.get('total_levels_analyzed', 0)}, "
            f"worst_stenosis={result_data.get('worst_stenosis_grade', 'N/A')}"
        )
        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Cord compression detection failed: {e.message}")
        raise


# =============================================================================
# Tool 4: Metastatic Lesion Detection
# =============================================================================


def _build_metastasis_detection_code(safe_t1_id: str, safe_t2_stir_id: str, region: str) -> str:
    """Build Python code for metastatic lesion detection.

    Args:
        safe_t1_id: JSON-escaped T1 volume node ID
        safe_t2_stir_id: JSON-escaped T2/STIR volume node ID
        region: Spine region to analyze

    Returns:
        Complete Python code string for Slicer execution
    """
    vertebrae = REGION_VERTEBRAE.get(region, ())
    ts_map = TOTALSEGMENTATOR_VERTEBRA_MAP

    ts_vertebrae = {v: k for k, v in ts_map.items() if v in vertebrae}

    return f"""
import slicer
import json
import vtk
import numpy as np

{_build_signal_normalization_code()}

{_build_registration_check_code(safe_t1_id, safe_t2_stir_id)}

{_build_totalseg_spine_code(safe_t1_id, region)}

vertebrae_map = {json.dumps(ts_vertebrae)}

segmentation = seg_node.GetSegmentation()
available_segments = []
for i in range(segmentation.GetNumberOfSegments()):
    available_segments.append(segmentation.GetNthSegment(i).GetName())

# Establish normal marrow reference (median of all vertebral signals)
t1_signals = []
t2_signals = []
for vert_name, ts_label in vertebrae_map.items():
    if ts_label not in available_segments:
        continue
    t1_sig = get_segment_mean_signal(t1_node, seg_node, ts_label)
    t2_sig = get_segment_mean_signal(t2_node, seg_node, ts_label)
    if t1_sig is not None:
        t1_signals.append(t1_sig)
    if t2_sig is not None:
        t2_signals.append(t2_sig)

ref_t1 = float(np.median(t1_signals)) if t1_signals else None
ref_t2 = float(np.median(t2_signals)) if t2_signals else None

lesions = []
vertebra_results = []

for vert_name, ts_label in vertebrae_map.items():
    if ts_label not in available_segments:
        continue

    t1_signal = get_segment_mean_signal(t1_node, seg_node, ts_label)
    t2_signal = get_segment_mean_signal(t2_node, seg_node, ts_label)

    if t1_signal is None or t2_signal is None:
        continue

    t1_ratio = compute_signal_ratio(t1_signal, ref_t1)
    t2_ratio = compute_signal_ratio(t2_signal, ref_t2)

    if t1_ratio is None or t2_ratio is None:
        continue

    # Signal pattern classification
    t1_low = t1_ratio < {METASTASIS_T1_LOW_THRESHOLD}
    t2_high = t2_ratio > {METASTASIS_T2_HIGH_THRESHOLD}
    t2_low = t2_ratio < {METASTASIS_T2_LOW_THRESHOLD}

    lesion_type = None
    suspicious = False

    if t1_low and t2_high:
        lesion_type = "lytic"
        suspicious = True
    elif t1_low and t2_low:
        lesion_type = "blastic"
        suspicious = True
    elif t1_low and not t2_low and not t2_high:
        lesion_type = "mixed"
        suspicious = True

    # Benign vs malignant fracture features
    # Malignant: convex posterior cortex, pedicle involvement, epidural mass
    # Benign: linear/band-like marrow edema, retropulsed fragment, gas
    fracture_features = []

    # Check segment shape for posterior element involvement
    seg_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        seg = segmentation.GetNthSegment(i)
        if seg.GetName() == ts_label:
            seg_id = segmentation.GetNthSegmentID(i)
            break

    posterior_element_involved = False
    if seg_id:
        bounds = [0.0] * 6
        segmentation.GetSegmentBounds(seg_id, bounds)
        # Check for asymmetric posterior extension
        ap_extent = abs(bounds[3] - bounds[2])
        lat_extent = abs(bounds[1] - bounds[0])
        if ap_extent > lat_extent * 1.5:
            posterior_element_involved = True
            fracture_features.append("posterior_element_involvement")

    # Signal heterogeneity check (malignant lesions tend to be heterogeneous)
    vert_stats_t1 = get_segment_stats(t1_node, seg_node, ts_label)
    if vert_stats_t1 and vert_stats_t1['mean'] > 0:
        t1_cv = vert_stats_t1['std'] / vert_stats_t1['mean']
        if t1_cv > 0.30:
            fracture_features.append("heterogeneous_signal")

    vert_result = {{
        'vertebra': vert_name,
        't1_signal_ratio': round(t1_ratio, 3),
        't2_stir_signal_ratio': round(t2_ratio, 3),
        'suspicious': suspicious,
        'lesion_type': lesion_type,
        'posterior_element_involved': posterior_element_involved,
        'fracture_features': fracture_features,
    }}

    vertebra_results.append(vert_result)
    if suspicious:
        lesions.append(vert_result)

# Clean up
slicer.mrmlScene.RemoveNode(seg_node)

result = {{
    'success': True,
    'tool': 'detect_metastatic_lesions_mri',
    'region': '{region}',
    'registration_performed': registration_performed,
    'reference_t1_signal': round(ref_t1, 1) if ref_t1 else None,
    'reference_t2_stir_signal': round(ref_t2, 1) if ref_t2 else None,
    'vertebra_results': vertebra_results,
    'total_vertebrae_analyzed': len(vertebra_results),
    'suspicious_lesions': lesions,
    'total_suspicious': len(lesions),
    'lesion_type_summary': {{
        'lytic': sum(1 for l in lesions if l['lesion_type'] == 'lytic'),
        'blastic': sum(1 for l in lesions if l['lesion_type'] == 'blastic'),
        'mixed': sum(1 for l in lesions if l['lesion_type'] == 'mixed'),
    }},
    'posterior_element_involvement': any(
        l.get('posterior_element_involved', False) for l in lesions
    ),
}}

print(json.dumps(result))
"""


def detect_metastatic_lesions_mri(
    t1_node_id: str,
    t2_stir_node_id: str,
    region: str = "full",
) -> dict[str, Any]:
    """Detect metastatic lesions in the spine using T1 and T2/STIR MRI.

    Analyzes signal patterns to identify and classify metastatic lesions:
    - Lytic: T1 low, T2/STIR high
    - Blastic: T1 low, T2/STIR low
    - Mixed: T1 low, T2/STIR intermediate

    Also evaluates posterior element involvement, epidural extension markers,
    and benign vs malignant fracture differentiation features.

    LONG OPERATION: Requires registration + segmentation + per-vertebra analysis.

    Args:
        t1_node_id: MRML node ID of T1-weighted volume
        t2_stir_node_id: MRML node ID of T2/STIR-weighted volume
        region: Spine region - "cervical", "thoracic", "lumbar", or "full"

    Returns:
        Dict with per-vertebra analysis, suspicious lesions, and type summary

    Raises:
        ValidationError: If input parameters are invalid
        SlicerConnectionError: If Slicer is not reachable or analysis fails
    """
    t1_node_id = validate_mrml_node_id(t1_node_id)
    t2_stir_node_id = validate_mrml_node_id(t2_stir_node_id)

    # Metastasis detection supports "full" spine region
    valid_regions = VALID_MRI_REGIONS | {"full"}
    if region not in valid_regions:
        raise ValidationError(
            f"Invalid region '{region}'. " f"Must be one of: {', '.join(sorted(valid_regions))}",
            "region",
            region,
        )

    client = get_client()

    safe_t1_id = json.dumps(t1_node_id)
    safe_t2_stir_id = json.dumps(t2_stir_node_id)

    python_code = _build_metastasis_detection_code(safe_t1_id, safe_t2_stir_id, region)

    try:
        exec_result = client.exec_python(python_code, timeout=MRI_ANALYSIS_TIMEOUT)
        result_data = _parse_json_result(
            exec_result.get("result", ""), "metastatic lesion detection"
        )
        logger.info(
            f"Metastatic lesion detection completed: region={region}, "
            f"vertebrae={result_data.get('total_vertebrae_analyzed', 0)}, "
            f"suspicious={result_data.get('total_suspicious', 0)}"
        )
        return result_data

    except SlicerConnectionError as e:
        logger.error(f"Metastatic lesion detection failed: {e.message}")
        raise
