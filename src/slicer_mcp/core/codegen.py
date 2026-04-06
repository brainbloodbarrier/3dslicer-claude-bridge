"""Code generation safety helpers for Slicer Python code templates.

Thin wrappers around ``json.dumps()`` providing type-specific escaping
for parameters embedded into code strings executed via ``exec_python()``.
"""

import json
from typing import Any

__all__ = [
    "safe_json",
    "safe_optional",
    "safe_string",
]


def safe_string(value: str) -> str:
    """Escape a string for safe embedding in generated Python code.

    Args:
        value: String to escape

    Returns:
        JSON-escaped string (with quotes), safe for f-string interpolation
    """
    return json.dumps(value)


def safe_json(value: Any) -> str:
    """Serialize any JSON-compatible value for code embedding.

    Args:
        value: Value to serialize (list, dict, number, etc.)

    Returns:
        JSON representation string
    """
    return json.dumps(value)


def safe_optional(value: Any | None) -> str:
    """Serialize a value that may be None.

    Args:
        value: Value to serialize, or None

    Returns:
        ``"None"`` if value is None, otherwise ``json.dumps(value)``
    """
    if value is None:
        return "None"
    return json.dumps(value)
