"""Spine surgery tool registrations — measurements, segmentation, instrumentation."""

from typing import Any

from slicer_mcp._registry._common import register_tool
from slicer_mcp.features.spine import instrumentation as instrumentation_tools
from slicer_mcp.features.spine import tools as spine_tools


def register_spine_tools(mcp: Any) -> dict[str, Any]:
    """Register spine surgery and instrumentation tools.

    Returns:
        Dict mapping tool name → wrapper function.
    """
    wrappers: dict[str, Any] = {}

    def _reg(module: Any, fn_name: str, doc: str) -> None:
        wrappers[fn_name] = register_tool(mcp, module, fn_name, doc)

    _reg(
        instrumentation_tools,
        "plan_cervical_screws",
        """Plan cervical screw placement using one of 6 instrumentation techniques.

    Generates patient-specific screw entry points, trajectory vectors, safety
    assessments (VA, canal, nerve root), and 3D visualization in Slicer.

    Techniques:
        - pedicle: Cervical pedicle screws (C2-C7)
        - lateral_mass: Lateral mass screws (C3-C7) with 4 variants
            (roy_camille, magerl, an, anderson)
        - transarticular: C1-C2 Magerl transarticular screws (VA REQUIRED)
        - c1_lateral_mass: C1 lateral mass Harms/Goel screws
        - c2_pars: C2 pars interarticularis screws
        - occipital: Occipital screws with thickness mapping
        - auto: Analyze anatomy and recommend best technique

    Args:
        technique: Instrumentation technique name
        level: Vertebral level (e.g. "C5", "C1C2", "Occiput")
        segmentation_node_id: MRML node ID of vertebral segmentation
        side: "left", "right", or "bilateral"
        va_node_id: MRML node ID of vertebral artery segmentation
            (REQUIRED for transarticular, recommended for C1-C2)
        variant: Lateral mass variant (only for lateral_mass technique)
        screw_diameter_mm: Override default screw diameter (mm)
        screw_length_mm: Override default screw length (mm)

    Returns:
        Dict with technique details, screw parameters, safety assessment,
        visualization node IDs, warnings, and recommendations
    """,
    )

    _reg(
        spine_tools,
        "measure_ccj_angles",
        """Measure craniocervical junction (CCJ) angles and distances.

    Computes CXA, ADI, Powers ratio, BDI, BAI, Ranawat, McGregor,
    Chamberlain, and Wackenheim measurements from a spine segmentation.
    Requires C1/C2 vertebral segments (e.g., from TotalSegmentator).

    LONG OPERATION: May take 30-60 seconds depending on segmentation size.

    Args:
        segmentation_node_id: MRML node ID of segmentation containing
            vertebral segments (e.g., "vtkMRMLSegmentationNode1")
        population: Patient population - "adult" (ADI <= 3mm)
            or "child" (ADI <= 5mm)

    Returns:
        Dict with landmarks, measurements (CXA, ADI, Powers, BDI, BAI,
        Ranawat, McGregor, Chamberlain, Wackenheim), reference_ranges,
        statuses, and coordinate_system (RAS)
    """,
    )

    _reg(
        spine_tools,
        "measure_spine_alignment",
        """Measure sagittal spinal alignment parameters.

    Computes cervical lordosis, thoracic kyphosis, lumbar lordosis,
    SVA, C2-C7 SVA, T1 slope, pelvic incidence, pelvic tilt,
    sacral slope, PI-LL mismatch, Roussouly type, and SRS-Schwab
    classification from vertebral body segmentations.

    LONG OPERATION: May take 1-3 minutes depending on region and segmentation size.

    Args:
        segmentation_node_id: MRML node ID of segmentation containing
            vertebral segments (e.g., "vtkMRMLSegmentationNode1")
        region: Spine region - "cervical" (C1-C7), "thoracic" (T1-T12),
            "lumbar" (L1-L5), or "full" (all vertebrae)

    Returns:
        Dict with vertebrae_found, vertebrae_data (centroids, endplates),
        measurements (CL, TK, LL, SVA, T1 slope, PI, PT, SS, etc.),
        reference_ranges, statuses, Roussouly type, and Schwab classification
    """,
    )

    _reg(
        spine_tools,
        "segment_spine",
        """Segment spine structures from a CT volume using TotalSegmentator AI.

    LONG OPERATION: This tool may take 1-5 minutes depending on hardware and region.

    Identifies and labels individual vertebrae from C1-L5, optionally
    including intervertebral discs and spinal cord.

    Expected durations:
    - GPU: ~1-2 minutes
    - CPU: ~3-5 minutes

    Args:
        input_node_id: MRML node ID of input CT volume
        region: Spine region - "cervical" (C1-C7), "thoracic" (T1-T12),
            "lumbar" (L1-L5), or "full" (all vertebrae)
        include_discs: If True, also segment intervertebral discs
        include_spinal_cord: If True, also segment the spinal cord

    Returns:
        Dict with output_segmentation_id, vertebrae_count, vertebrae list,
        discs list, processing_time_seconds, and long_operation metadata
    """,
    )

    _reg(
        spine_tools,
        "visualize_spine_segmentation",
        """Create clinical visualization of spine segmentation.

    Generates a high-quality sagittal screenshot with distinct colors
    per vertebra (yellow-to-red gradient), outline-mode segmentation
    overlay, bone-window CT (W:2000 L:400), and anatomical labels
    at each vertebra centroid.

    Args:
        segmentation_node_id: MRML node ID of segmentation with vertebral
            segments (e.g., from segment_spine or TotalSegmentator)
        volume_node_id: MRML node ID of background CT volume
        output_path: File path for the output PNG screenshot
        region: Spine region - "cervical", "thoracic", "lumbar", or "full"

    Returns:
        Dict with output_path, file_size_bytes, vertebrae_colored,
        centroids (RAS), view parameters, and fiducial_node_ids
    """,
    )

    _reg(
        spine_tools,
        "segment_vertebral_artery",
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
    """,
    )

    _reg(
        spine_tools,
        "analyze_bone_quality",
        """Analyze bone quality per vertebra using CT HU and BoneTexture metrics.

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
        region: Spine region - "cervical", "thoracic", "lumbar", or "full"

    Returns:
        Dict with per-vertebra metrics (mean_hu, classification, optional BV/TV,
        Tb.Th, Tb.Sp), summary counts, and processing_time_seconds
    """,
    )

    return wrappers
