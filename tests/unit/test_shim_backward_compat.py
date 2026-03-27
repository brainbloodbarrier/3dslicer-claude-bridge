"""Tests that the 14 root-level shim modules correctly re-export canonical symbols.

Each shim exists for backward compatibility and should re-export at least one
key symbol from its canonical module.
"""

import importlib
import warnings

import pytest

SHIM_TO_CANONICAL = [
    ("slicer_mcp.circuit_breaker", "slicer_mcp.core.circuit_breaker", "CircuitBreaker"),
    ("slicer_mcp.constants", "slicer_mcp.core.constants", "DEFAULT_SLICER_URL"),
    ("slicer_mcp.metrics", "slicer_mcp.core.metrics", "track_request"),
    ("slicer_mcp.resources", "slicer_mcp.core.resources", "get_status_resource"),
    ("slicer_mcp.slicer_client", "slicer_mcp.core.slicer_client", "SlicerClient"),
    ("slicer_mcp.tools", "slicer_mcp.features.base_tools", "capture_screenshot"),
    (
        "slicer_mcp.diagnostic_tools_ct",
        "slicer_mcp.features.diagnostics.ct",
        "detect_vertebral_fractures_ct",
    ),
    (
        "slicer_mcp.diagnostic_tools_mri",
        "slicer_mcp.features.diagnostics.mri",
        "classify_modic_changes",
    ),
    (
        "slicer_mcp.diagnostic_tools_xray",
        "slicer_mcp.features.diagnostics.xray",
        "measure_cobb_angle_xray",
    ),
    (
        "slicer_mcp.instrumentation_tools",
        "slicer_mcp.features.spine.instrumentation",
        "plan_cervical_screws",
    ),
    ("slicer_mcp.registration_tools", "slicer_mcp.features.registration", "place_landmarks"),
    ("slicer_mcp.rendering_tools", "slicer_mcp.features.rendering", "enable_volume_rendering"),
    ("slicer_mcp.spine_constants", "slicer_mcp.features.spine.constants", "SPINE_REGIONS"),
    ("slicer_mcp.spine_tools", "slicer_mcp.features.spine.tools", "segment_spine"),
]


@pytest.mark.parametrize(
    "shim_path, canonical_path, symbol",
    SHIM_TO_CANONICAL,
    ids=[s[0].split(".")[-1] for s in SHIM_TO_CANONICAL],
)
def test_shim_re_exports_canonical_symbol(shim_path, canonical_path, symbol):
    """Shim module re-exports the expected symbol from canonical path."""
    canonical_mod = importlib.import_module(canonical_path)
    canonical_obj = getattr(canonical_mod, symbol)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shim_mod = importlib.import_module(shim_path)

    shim_obj = getattr(shim_mod, symbol, None)
    assert shim_obj is not None, f"{shim_path} does not export '{symbol}'"
    assert (
        shim_obj is canonical_obj
    ), f"{shim_path}.{symbol} is not the same object as {canonical_path}.{symbol}"


@pytest.mark.parametrize(
    "shim_path",
    [s[0] for s in SHIM_TO_CANONICAL],
    ids=[s[0].split(".")[-1] for s in SHIM_TO_CANONICAL],
)
def test_shim_emits_deprecation_warning(shim_path):
    """Importing a shim module emits a DeprecationWarning."""
    # Force re-import by removing from cache
    import sys

    sys.modules.pop(shim_path, None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module(shim_path)

    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1, f"No DeprecationWarning from {shim_path}"
