"""CT diagnostic protocol tools for spine analysis.

Re-export hub that unifies the split CT sub-modules into a single
public namespace. All downstream imports (``_registry/diagnostics.py``,
``features/workflows/onco_spine.py``, tests) continue to work unchanged.

Sub-modules:
- ``ct_fracture`` -- fracture detection, listhesis, canal morphometry
- ``ct_metabolic`` -- osteoporosis screening, metastatic lesion detection
- ``ct_scoring``   -- SINS score calculation
- ``_common``      -- shared validation helpers
"""

# Re-export public tools --------------------------------------------------
# Re-export get_client so @patch("...ct.get_client") keeps working ---------
from slicer_mcp.core.slicer_client import get_client  # noqa: F401

# Re-export _validate_region (tests import it from ct.py) ------------------
from slicer_mcp.features._validation import _validate_region  # noqa: F401

# Re-export private helpers (used by tests) --------------------------------
from slicer_mcp.features.diagnostics._common import (  # noqa: F401
    _validate_levels,
)
from slicer_mcp.features.diagnostics.ct_fracture import (  # noqa: F401  # noqa: F401
    _build_canal_measurement_code,
    _build_fracture_detection_code,
    _build_listhesis_code,
    _classify_genant,
    _classify_meyerding,
    _classify_stenosis,
    _validate_classification_system,
    detect_vertebral_fractures_ct,
    measure_listhesis_ct,
    measure_spinal_canal_ct,
)
from slicer_mcp.features.diagnostics.ct_metabolic import (  # noqa: F401  # noqa: F401
    _build_metastatic_detection_code,
    _build_osteoporosis_code,
    _classify_pickhardt,
    _validate_osteo_method,
    assess_osteoporosis_ct,
    detect_metastatic_lesions_ct,
)
from slicer_mcp.features.diagnostics.ct_scoring import (  # noqa: F401  # noqa: F401
    _build_sins_code,
    _classify_sins_total,
    _sins_alignment_score,
    _sins_collapse_score,
    _sins_lesion_type_score,
    _sins_location_score,
    _sins_posterolateral_score,
    calculate_sins_score,
)

__all__ = [
    "assess_osteoporosis_ct",
    "calculate_sins_score",
    "detect_metastatic_lesions_ct",
    "detect_vertebral_fractures_ct",
    "measure_listhesis_ct",
    "measure_spinal_canal_ct",
]
