"""MCP server bridging Claude Code to 3D Slicer."""

__version__ = "1.0.0"

from slicer_mcp.circuit_breaker import CircuitOpenError
from slicer_mcp.server import main
from slicer_mcp.slicer_client import SlicerConnectionError, SlicerTimeoutError
from slicer_mcp.tools import ValidationError

__all__ = [
    "__version__",
    "main",
    "SlicerConnectionError",
    "SlicerTimeoutError",
    "CircuitOpenError",
    "ValidationError",
]
