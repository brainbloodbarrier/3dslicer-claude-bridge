"""Registration & landmark tool registrations."""

from typing import Any

from slicer_mcp._registry._common import register_tool
from slicer_mcp.features import registration as registration_tools


def register_registration_tools(mcp: Any) -> dict[str, Any]:
    """Register landmark placement and volume registration tools.

    Returns:
        Dict mapping tool name → wrapper function.
    """
    wrappers: dict[str, Any] = {}

    def _reg(fn_name: str, doc: str) -> None:
        wrappers[fn_name] = register_tool(mcp, registration_tools, fn_name, doc)

    _reg(
        "place_landmarks",
        """Create a markup fiducial node with named control points in 3D Slicer.

    Places a set of RAS-coordinate landmarks as a vtkMRMLMarkupsFiducialNode.
    Useful for defining anatomical landmarks for registration or measurement.

    Args:
        name: Display name for the markup node (non-empty, max 64 chars)
        points: List of [x, y, z] RAS coordinates for each control point
        labels: Optional list of labels for each point (must match points length).
            Labels must match the pattern [A-Za-z0-9_-]+.

    Returns:
        Dict with success status, node_id, node_name, and point_count
    """,
    )

    _reg(
        "get_landmarks",
        """Retrieve all control points from a markup fiducial node.

    Args:
        node_id: MRML node ID of the markups fiducial node
            (e.g., "vtkMRMLMarkupsFiducialNode1")

    Returns:
        Dict with success status, node_id, node_name, point_count,
            and points list (index, label, position_ras)
    """,
    )

    _reg(
        "register_volumes",
        """Perform intensity-based volume registration using BRAINSFit.

    LONG OPERATION: May take up to 5 minutes depending on volume size and transform type.

    Aligns a moving volume to a fixed volume using intensity-based optimization.
    Supports rigid, affine, and deformable (BSpline) transforms.

    Args:
        fixed_node_id: MRML node ID of the fixed (reference) volume
        moving_node_id: MRML node ID of the moving volume to be registered
        transform_type: Registration transform - "Rigid", "ScaleVersor3D",
            "ScaleSkewVersor3D", "Affine", or "BSpline"
        init_mode: Initialization method - "useMomentsAlign",
            "useCenterOfHeadAlign", "useGeometryAlign", or "Off"
        sampling_percentage: Fraction of voxels to sample (0.0-1.0, default 0.01)
        histogram_match: Whether to match histograms before registration
        create_resampled: Whether to create a resampled output volume

    Returns:
        Dict with success status, transform_node_id, transform_node_name,
            transform_type, and optional resampled_node_id
    """,
    )

    _reg(
        "register_landmarks",
        """Perform landmark-based registration using paired fiducial points.

    Computes a spatial transform that aligns moving landmarks to fixed landmarks.
    Requires matching control points in both markup nodes.

    Args:
        fixed_landmarks_id: MRML node ID of the fixed (reference) landmarks
        moving_landmarks_id: MRML node ID of the moving landmarks
        transform_type: Registration transform - "Rigid", "Similarity", or "Affine"

    Returns:
        Dict with success status, transform_node_id, transform_node_name,
            and transform_type
    """,
    )

    _reg(
        "apply_transform",
        """Apply a spatial transform to any transformable node.

    Sets the transform on the node. If harden is True, bakes the transform
    into the node's data so the node moves to its transformed position permanently.

    Args:
        node_id: MRML node ID of the node to transform
        transform_node_id: MRML node ID of the transform to apply
        harden: If True, permanently bake the transform into the node data

    Returns:
        Dict with success status, node_id, transform_node_id, and hardened flag
    """,
    )

    return wrappers
