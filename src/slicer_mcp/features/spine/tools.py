"""Spine tools -- re-export hub for backward compatibility.

All tool implementations have been split into domain-specific modules:
- ccj: Craniocervical junction measurements
- alignment: Sagittal spinal alignment parameters
- segmentation: TotalSegmentator spine/VA segmentation
- analysis: Bone quality analysis and visualization
"""

from slicer_mcp.features._subprocess import (  # noqa: F401 -- re-export for tests
    _build_totalseg_subprocess_block,
)
from slicer_mcp.features.spine.alignment import (  # noqa: F401
    _build_sagittal_alignment_code,
    _build_vertebral_centroid_extraction_code,
    measure_spine_alignment,
)
from slicer_mcp.features.spine.analysis import (  # noqa: F401
    _build_bone_quality_code,
    _build_clinical_spine_visualization_code,
    analyze_bone_quality,
    visualize_spine_segmentation,
)
from slicer_mcp.features.spine.ccj import (  # noqa: F401
    _build_ccj_angles_code,
    _build_ccj_landmark_extraction_code,
    measure_ccj_angles,
)
from slicer_mcp.features.spine.constants import (  # noqa: F401
    VALID_ALIGNMENT_REGIONS,
    VALID_ARTERY_SIDES,
    VALID_BONE_REGIONS,
    VALID_CCJ_LANDMARKS,
    VALID_POPULATIONS,
)
from slicer_mcp.features.spine.segmentation import (  # noqa: F401
    _build_spine_segmentation_code,
    _validate_seed_points,
    segment_spine,
    segment_vertebral_artery,
)

__all__ = [
    "VALID_ALIGNMENT_REGIONS",
    "VALID_ARTERY_SIDES",
    "VALID_BONE_REGIONS",
    "VALID_CCJ_LANDMARKS",
    "VALID_POPULATIONS",
    "analyze_bone_quality",
    "measure_ccj_angles",
    "measure_spine_alignment",
    "segment_spine",
    "segment_vertebral_artery",
    "visualize_spine_segmentation",
]
