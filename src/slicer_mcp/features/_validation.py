"""Shared validation helpers for feature modules.

Provides reusable input validation functions used across multiple
feature domains (diagnostics, workflows) to avoid duplication.
"""

from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.features.spine.constants import SPINE_REGIONS

__all__ = [
    "_validate_region",
]


def _validate_region(
    region: str,
    valid_regions: frozenset[str] = SPINE_REGIONS,
    field: str = "region",
) -> str:
    """Validate spine region parameter.

    Args:
        region: Spine region string to validate
        valid_regions: Set of valid region values (default: SPINE_REGIONS)
        field: Field name for error messages

    Returns:
        Validated region string

    Raises:
        ValidationError: If region is not valid
    """
    if region not in valid_regions:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(valid_regions))}",
            field,
            region,
        )
    return region
