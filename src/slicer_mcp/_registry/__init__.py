"""Tool registration modules organized by clinical domain.

Each module exports a ``register_*_tools(mcp, handle_error)`` function
that wires feature functions into the MCP server.  The main ``server.py``
calls them in sequence at import time.
"""

from slicer_mcp._registry._common import handle_tool_error, register_tool
from slicer_mcp._registry.base import register_base_tools
from slicer_mcp._registry.diagnostics import register_diagnostic_tools
from slicer_mcp._registry.registration import register_registration_tools
from slicer_mcp._registry.rendering import register_rendering_tools
from slicer_mcp._registry.resources import register_resources
from slicer_mcp._registry.spine import register_spine_tools
from slicer_mcp._registry.workflows import register_workflow_tools

__all__ = [
    "handle_tool_error",
    "register_tool",
    "register_base_tools",
    "register_diagnostic_tools",
    "register_registration_tools",
    "register_rendering_tools",
    "register_resources",
    "register_spine_tools",
    "register_workflow_tools",
]
