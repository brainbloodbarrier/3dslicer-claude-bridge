"""Compatibility shim for relocated module."""

import warnings
from importlib import import_module as _import_module

_module = _import_module("slicer_mcp.features.diagnostics.mri")

warnings.warn(
    "Importing from 'slicer_mcp.diagnostic_tools_mri' is deprecated. "
    "Use 'slicer_mcp.features.diagnostics.mri' instead.",
    DeprecationWarning,
    stacklevel=2,
)

globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")}
)
__all__ = getattr(_module, "__all__", [name for name in globals() if not name.startswith("_")])

del _module
del _import_module
