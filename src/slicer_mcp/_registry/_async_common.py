"""Async tool registration helper.

Drop-in replacement for ``register_tool`` that creates ``async def``
wrappers.  FastMCP v1.26+ detects async functions and handles them
natively, enabling progress reporting and concurrent tool execution.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

from slicer_mcp._registry._common import handle_tool_error

logger = logging.getLogger("slicer-mcp")


def register_async_tool(
    mcp_instance: Any,
    module: Any,
    fn_name: str,
    doc: str,
) -> Callable[..., Any]:
    """Register an async feature function as an MCP tool.

    Same contract as ``register_tool`` but the wrapper is ``async def``.
    The underlying feature function MUST be async.

    Args:
        mcp_instance: The FastMCP server instance
        module: The feature module containing the async function
        fn_name: Attribute name of the async function on *module*
        doc: Docstring for the MCP tool

    Returns:
        The registered async MCP tool wrapper function
    """
    tool_fn = getattr(module, fn_name)

    @mcp_instance.tool()
    @functools.wraps(tool_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            fn = getattr(module, fn_name)
            return await fn(*args, **kwargs)
        except Exception as e:
            return handle_tool_error(e, fn_name)

    wrapper.__doc__ = doc
    return wrapper
