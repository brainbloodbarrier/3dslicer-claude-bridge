"""Spine and vertebral artery segmentation tools.

Provides TotalSegmentator-based spine segmentation, SlicerVMTK-based
vertebral artery segmentation (vesselness + level set), and the public
``segment_spine`` and ``segment_vertebral_artery`` tool functions.
"""

import json
import logging
from typing import Any

from slicer_mcp.core.constants import SEGMENTATION_TIMEOUT
from slicer_mcp.core.slicer_client import SlicerConnectionError, get_client
from slicer_mcp.features._subprocess import (
    _kill_process_group_code,
)
from slicer_mcp.features.base_tools import (
    ValidationError,
    _parse_json_result,
    validate_mrml_node_id,
)
from slicer_mcp.features.spine.constants import (
    REGION_VERTEBRAE,
    SPINE_REGIONS,
    SPINE_SEGMENTATION_TIMEOUT,
    TOTALSEG_TASK_FULL,
    TOTALSEGMENTATOR_DISC_MAP,
    TOTALSEGMENTATOR_VERTEBRA_MAP,
    VA_CENTERLINE_SAMPLE_POINTS,
    VA_LEVEL_SET_ITERATION_COUNT,
    VA_VESSELNESS_CONTRAST_MEASURE,
    VALID_ARTERY_SIDES,
)

__all__ = [
    "segment_spine",
    "segment_vertebral_artery",
]

logger = logging.getLogger("slicer-mcp")


# =============================================================================
# Code Generation Helpers (Spine Segmentation)
# =============================================================================


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
    _writeOk = storageNode.WriteData(inputVolume)
    storageNode.UnRegister(None)
    if not _writeOk:
        raise RuntimeError(f"Failed to export volume to NIfTI: {{inputFile}}")

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
    _prev_size = 0

    while elapsed_wait < timeout_s:
        # Check if process finished normally
        ret = proc.poll()
        if ret is not None:
            if ret != 0:
                stderr_out = proc.stderr.read(8192).decode(errors='replace')[-500:]
                raise RuntimeError(f"TotalSegmentator exited with code {{ret}}: {{stderr_out}}")
            break
        # Check if output file exists and size is stable (fully written)
        if os.path.exists(outputFile):
            _curr_size = os.path.getsize(outputFile)
            if _curr_size > 1000 and _curr_size == _prev_size:
                time.sleep(3)
                break
            _prev_size = _curr_size
        time.sleep(poll_interval)
        elapsed_wait += poll_interval

    # Kill the process group if still running (resource_tracker hang workaround)
    # Safe because start_new_session=True gives it its own process group
{
        _kill_process_group_code(
            proc_var="proc",
            os_mod="os",
            signal_mod="signal",
            time_mod="time",
            logging_alias="_seg_logging",
            indent="    ",
        )
    }

    if not os.path.exists(outputFile):
        raise RuntimeError("TotalSegmentator did not produce output file within timeout")

    # Import segmentation result into Slicer using TotalSegmentator module
    logic = TotalSegmentatorLogic()
    logic.readSegmentation(outputSeg, outputFile, {safe_task})

    # Set source volume reference
    outputSeg.SetNodeReferenceID(
        outputSeg.GetReferenceImageGeometryReferenceRole(), inputVolume.GetID())
    outputSeg.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

except Exception as e:
    slicer.mrmlScene.RemoveNode(outputSeg)
    raise ValueError(f"TotalSegmentator failed ({{type(e).__name__}}): {{e}}") from e
finally:
    # Kill subprocess if still alive (handles exception during poll loop)
{
        _kill_process_group_code(
            proc_var="proc",
            os_mod="os",
            signal_mod="signal",
            time_mod="time",
            logging_alias="_seg_logging",
            indent="    ",
        )
    }
    # Clean up temp files (both success and failure paths)
    def _seg_rmtree_onerror(func, path, exc_info):
        import logging as _seg_log
        _seg_log.getLogger("slicer-mcp").warning(
            f"Failed to clean up temp file {{path}}: {{exc_info[1]}}"
        )

    if tempFolder is not None and os.path.isdir(tempFolder):
        shutil.rmtree(tempFolder, onerror=_seg_rmtree_onerror)

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

    # Match by segment ID (e.g. "vertebrae_L5") — TotalSegmentator sets IDs
    # matching VERTEBRA_MAP keys, while display names differ (e.g. "L5 vertebra")
    if seg_id in VERTEBRA_MAP:
        label = VERTEBRA_MAP[seg_id]
        found_vertebrae.append({{"segment_id": seg_id, "label": label, "name": seg_name}})
    elif seg_id in DISC_MAP:
        disc_label = DISC_MAP[seg_id]
        found_discs.append({{"segment_id": seg_id, "label": disc_label, "name": seg_name}})
    elif seg_id == "spinal_cord" or seg_name == "spinal_cord":
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
    contrastMeasure={VA_VESSELNESS_CONTRAST_MEASURE}
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
    iterationCount={VA_LEVEL_SET_ITERATION_COUNT}
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
        step = max(1, n_points // {VA_CENTERLINE_SAMPLE_POINTS})  # Sample ~N points
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
    contrastMeasure={VA_VESSELNESS_CONTRAST_MEASURE}
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
    iterationCount={VA_LEVEL_SET_ITERATION_COUNT}
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
        if not isinstance(pt, list | tuple) or len(pt) != 3:
            raise ValidationError(
                f"Seed point {i} must be [x, y, z] coordinates, got: {pt}",
                "seed_points",
                str(pt),
            )
        for j, coord in enumerate(pt):
            if not isinstance(coord, int | float):
                raise ValidationError(
                    f"Seed point {i} coordinate {j} must be numeric, got: {type(coord).__name__}",
                    "seed_points",
                    str(coord),
                )

    return seed_points


# =============================================================================
# Public Tool Functions — Segmentation
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
