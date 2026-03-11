"""Shared JSON parsing utilities for MCP results."""

import json
from typing import Any

from slicer_mcp.core.slicer_client import SlicerConnectionError


def _parse_json_result(result: str, context: str) -> Any:
    """Parse JSON result with null/empty handling.

    Args:
        result: JSON string to parse
        context: Description for error messages

    Returns:
        Parsed JSON data

    Raises:
        SlicerConnectionError: If result is empty, null, or malformed
    """
    if not result or result.strip() in ("", "null", "None"):
        raise SlicerConnectionError(
            f"Empty result from {context}", details={"result": result[:100] if result else "None"}
        )

    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        raise SlicerConnectionError(
            f"Failed to parse {context} result: {str(e)}",
            details={"result_preview": result[:100] if result else "None"},
        )
