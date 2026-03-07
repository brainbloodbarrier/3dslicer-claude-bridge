"""MCP server bridging Claude Code to 3D Slicer."""

__version__ = "0.9.0"

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import SlicerConnectionError, SlicerTimeoutError
from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.server import main

__all__ = [
    "__version__",
    "main",
    "SlicerConnectionError",
    "SlicerTimeoutError",
    "CircuitOpenError",
    "ValidationError",
]
