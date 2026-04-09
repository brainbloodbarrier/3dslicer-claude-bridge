"""Tests that the remaining shim module correctly re-exports canonical symbols.

Only ``spine_constants`` remains as a backward-compat shim (active consumer
in ``test_diagnostic_tools_ct.py``).  The other 13 shims were removed in the
2026-04 aggressive refactoring.
"""

import importlib
import warnings

import pytest

SHIM_TO_CANONICAL = [
    ("slicer_mcp.spine_constants", "slicer_mcp.features.spine.constants", "SPINE_REGIONS"),
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
    assert shim_obj is canonical_obj, (
        f"{shim_path}.{symbol} is not the same object as {canonical_path}.{symbol}"
    )


@pytest.mark.parametrize(
    "shim_path",
    [s[0] for s in SHIM_TO_CANONICAL],
    ids=[s[0].split(".")[-1] for s in SHIM_TO_CANONICAL],
)
def test_shim_emits_deprecation_warning(shim_path):
    """Importing a shim module emits a DeprecationWarning."""
    import sys

    sys.modules.pop(shim_path, None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module(shim_path)

    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1, f"No DeprecationWarning from {shim_path}"
