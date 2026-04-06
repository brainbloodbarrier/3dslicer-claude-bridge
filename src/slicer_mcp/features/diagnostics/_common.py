"""Shared validation helpers for CT diagnostic sub-modules."""

from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.features.spine.constants import REGION_VERTEBRAE

__all__ = [
    "_validate_levels",
]


def _validate_levels(levels: list[str] | None, default_levels: list[str]) -> list[str]:
    """Validate vertebral level list.

    Args:
        levels: User-provided level list or None for defaults
        default_levels: Default levels when None provided

    Returns:
        Validated list of vertebral levels

    Raises:
        ValidationError: If any level is invalid
    """
    if levels is None:
        return default_levels

    all_vertebrae = set(REGION_VERTEBRAE["full"])
    for level in levels:
        if level not in all_vertebrae:
            raise ValidationError(
                f"Invalid vertebral level '{level}'. Must be one of: C1-C7, T1-T12, L1-L5",
                field="levels",
                value=level,
            )
    return levels
