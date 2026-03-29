"""Shared error handling and tool registration helper.

Extracted from ``server.py`` to be reused across domain-specific
registration modules without duplication.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import SlicerConnectionError, SlicerTimeoutError
from slicer_mcp.features.base_tools import ValidationError

logger = logging.getLogger("slicer-mcp")


def handle_tool_error(error: Exception, tool_name: str) -> dict:
    """Handle tool errors and return standardized error response.

    Args:
        error: The caught exception
        tool_name: Name of the tool that failed

    Returns:
        Dict with error information
    """
    if isinstance(error, ValidationError):
        logger.warning(f"Tool {tool_name}: Validation error - {error.message}")
        return {
            "success": False,
            "error": error.message,
            "error_type": "validation",
            "field": error.field,
            "value": error.value,
        }
    elif isinstance(error, CircuitOpenError):
        logger.warning(f"Tool {tool_name}: Circuit breaker open - {error}")
        return {"success": False, "error": str(error), "error_type": "circuit_open"}
    elif isinstance(error, SlicerTimeoutError):
        logger.error(f"Tool {tool_name}: Timeout - {error.message}")
        return {
            "success": False,
            "error": error.message,
            "error_type": "timeout",
            "details": error.details,
        }
    elif isinstance(error, SlicerConnectionError):
        logger.error(f"Tool {tool_name}: Connection error - {error.message}")
        return {
            "success": False,
            "error": error.message,
            "error_type": "connection",
            "details": error.details,
        }
    else:
        logger.error(f"Tool {tool_name}: Unexpected error - {error}", exc_info=True)
        return {"success": False, "error": str(error), "error_type": "unexpected"}


def register_tool(
    mcp_instance: Any,
    module: Any,
    fn_name: str,
    doc: str,
) -> Callable[..., dict]:
    """Register a feature function as an MCP tool with error handling.

    Creates a wrapper that:
    1. Preserves the original function's signature (for MCP schema generation)
    2. Delegates to the feature function via late-bound getattr() lookup
       (so unittest.mock.patch on the module attribute works correctly)
    3. Catches all exceptions and routes them through handle_tool_error

    Args:
        mcp_instance: The FastMCP server instance
        module: The feature module object (as imported, e.g. ``tools``)
        fn_name: Attribute name of the function on *module*
        doc: Docstring for the MCP tool (used by MCP for tool descriptions)

    Returns:
        The registered MCP tool wrapper function
    """
    # Resolve once at import time only to copy the signature for MCP schema.
    tool_fn = getattr(module, fn_name)

    @mcp_instance.tool()
    @functools.wraps(tool_fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            # Late-bound lookup: picks up mocks installed by unittest.mock.patch.
            fn = getattr(module, fn_name)
            return fn(*args, **kwargs)
        except Exception as e:
            return handle_tool_error(e, fn_name)

    wrapper.__doc__ = doc
    return wrapper
