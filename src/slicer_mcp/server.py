"""MCP Slicer Bridge server entry point.

Initializes the FastMCP server and delegates tool registration to
domain-specific modules in ``_registry/``.  Feature module imports
are kept at module level for backward compatibility with tests that
patch ``slicer_mcp.server.<module>.<function>``.
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP

# ── Registry imports ─────────────────────────────────────────────────
from slicer_mcp._registry import (
    register_base_tools,
    register_diagnostic_tools,
    register_registration_tools,
    register_rendering_tools,
    register_resources,
    register_spine_tools,
    register_workflow_tools,
)
from slicer_mcp._registry._common import handle_tool_error as _handle_tool_error  # noqa: F401

# ── Feature module imports (kept for backward-compatible patch targets) ──
from slicer_mcp.core import resources  # noqa: F401
from slicer_mcp.features import base_tools as tools  # noqa: F401
from slicer_mcp.features import registration as registration_tools  # noqa: F401
from slicer_mcp.features import rendering as rendering_tools  # noqa: F401
from slicer_mcp.features.diagnostics import ct as diagnostic_tools_ct  # noqa: F401
from slicer_mcp.features.diagnostics import mri as diagnostic_tools_mri  # noqa: F401
from slicer_mcp.features.diagnostics import xray as diagnostic_tools_xray  # noqa: F401
from slicer_mcp.features.spine import instrumentation as instrumentation_tools  # noqa: F401
from slicer_mcp.features.spine import tools as spine_tools  # noqa: F401
from slicer_mcp.features.workflows import ccj as workflow_ccj  # noqa: F401
from slicer_mcp.features.workflows import modic as workflow_modic  # noqa: F401
from slicer_mcp.features.workflows import onco_spine as workflow_onco  # noqa: F401

# ── Logging (stderr — stdout reserved for MCP protocol JSON) ─────────
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    stream=sys.stderr,
)
logger = logging.getLogger("slicer-mcp")

# ── FastMCP instance ─────────────────────────────────────────────────
mcp = FastMCP("slicer-bridge")

logger.info("Initializing MCP Slicer Bridge server")

# ── Register tools by domain ─────────────────────────────────────────
# Each register function returns {name: wrapper_fn} — we merge them
# into this module's globals so ``server.tool_name`` still works.
_all_wrappers: dict = {}
_all_wrappers.update(register_base_tools(mcp))
_all_wrappers.update(register_diagnostic_tools(mcp))
_all_wrappers.update(register_spine_tools(mcp))
_all_wrappers.update(register_workflow_tools(mcp))
_all_wrappers.update(register_registration_tools(mcp))
_all_wrappers.update(register_rendering_tools(mcp))

# Expose wrapper functions as module-level attributes (backward compat)
globals().update(_all_wrappers)

# ── Register resources ───────────────────────────────────────────────
_resource_uris, _resource_handlers = register_resources(mcp)
globals().update(_resource_handlers)

# ── Tool name list (for startup log) ─────────────────────────────────
_TOOL_NAMES = sorted(_all_wrappers.keys())


# ── Entry point ──────────────────────────────────────────────────────
def main():
    """Run the MCP Slicer Bridge server with stdio transport."""
    logger.info("Starting MCP Slicer Bridge server")
    logger.info("Registered %d tools: %s", len(_TOOL_NAMES), ", ".join(_TOOL_NAMES))
    logger.info(
        "Registered %d resources: %s",
        len(_resource_uris),
        ", ".join(_resource_uris),
    )

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
