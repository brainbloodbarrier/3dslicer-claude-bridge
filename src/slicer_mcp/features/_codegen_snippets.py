"""Reusable code-string snippets for Slicer Python code generation.

These functions return Python code fragments (as strings) that define
helper functions for use inside code templates executed in 3D Slicer
via ``exec_python()``.  They are **not** callable Python functions —
they are text templates embedded into larger code strings.
"""

__all__ = [
    "_render_alignment_preamble",
    "_render_classify_preamble",
    "_render_geometry_preamble",
]


def _render_geometry_preamble() -> str:
    """Return code-string defining geometry helpers for CCJ measurements.

    Defines: ``vec_angle_deg``, ``point_dist``, ``point_to_line_dist``,
    ``signed_dist_above_line``.

    Requires ``numpy as np`` in the execution context.
    """
    return '''\
def vec_angle_deg(v1, v2):
    """Angle between two 3D vectors in degrees."""
    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)
    cos_a = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def point_dist(p1, p2):
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def point_to_line_dist(point, line_p1, line_p2):
    """Distance from a point to a line defined by two points in 3D."""
    p = np.array(point)
    a = np.array(line_p1)
    b = np.array(line_p2)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))

def signed_dist_above_line(point, line_p1, line_p2):
    """Signed distance of a point above a line in the sagittal plane.

    Positive = above the line (superior), negative = below.
    Uses the S component (index 2) in RAS coordinates.
    """
    p = np.array(point)
    a = np.array(line_p1)
    b = np.array(line_p2)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    closest = a + t * ab
    return float(p[2] - closest[2])
'''


def _render_alignment_preamble() -> str:
    """Return code-string defining alignment helpers for sagittal measurements.

    Defines: ``vec_angle_deg``, ``cobb_angle_3d``, ``endplate_vector``.

    Requires ``numpy as np`` in the execution context.
    """
    return '''\
def vec_angle_deg(v1, v2):
    """Angle between two 3D vectors in degrees."""
    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)
    cos_a = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def cobb_angle_3d(sup_endplate_center, sup_endplate_vec, inf_endplate_center, inf_endplate_vec):
    """Compute 3D Cobb angle between two endplate planes.

    The Cobb angle is the angle between the superior endplate of the
    upper vertebra and the inferior endplate of the lower vertebra.
    """
    return vec_angle_deg(sup_endplate_vec, inf_endplate_vec)

def endplate_vector(vertebra_data):
    """Get the endplate tilt vector (superior to inferior endplate)."""
    sup = np.array(vertebra_data['superior_endplate'])
    inf = np.array(vertebra_data['inferior_endplate'])
    return sup - inf
'''


def _render_classify_preamble() -> str:
    """Return code-string defining the classify helper for measurement status.

    Defines: ``classify(value, normal_min, normal_max, name)``.

    Requires a ``statuses`` dict in the execution context.
    """
    return '''\
def classify(value, normal_min, normal_max, name):
    if normal_min <= value <= normal_max:
        statuses[name] = 'normal'
    elif value < normal_min:
        statuses[name] = 'below_normal'
    else:
        statuses[name] = 'above_normal'
'''
