"""Centralized constants for Slicer MCP Bridge."""

# =============================================================================
# Connection Settings
# =============================================================================
DEFAULT_SLICER_URL = "http://localhost:2016"
DEFAULT_TIMEOUT_SECONDS = 30

# =============================================================================
# Retry Configuration
# =============================================================================
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.0  # seconds

# =============================================================================
# View Names (Slicer terminology)
# =============================================================================
VIEW_AXIAL = "Red"
VIEW_SAGITTAL = "Yellow"
VIEW_CORONAL = "Green"
VIEW_3D = "3d"
VIEW_FULL = "full"

VIEW_MAP = {
    "axial": VIEW_AXIAL,
    "sagittal": VIEW_SAGITTAL,
    "coronal": VIEW_CORONAL,
    "3d": VIEW_3D,
    "full": VIEW_FULL,
}

# =============================================================================
# Valid Layouts and GUI Modes
# =============================================================================
VALID_LAYOUTS = frozenset(["FourUp", "OneUp3D", "OneUpRedSlice", "Conventional", "SideBySide"])
VALID_GUI_MODES = frozenset(["full", "viewers"])

# =============================================================================
# Validation Limits
# =============================================================================
MAX_NODE_ID_LENGTH = 256
MAX_SEGMENT_NAME_LENGTH = 256

# =============================================================================
# Audit Logging
# =============================================================================
AUDIT_CODE_MAX_LENGTH = 500
AUDIT_RESULT_MAX_LENGTH = 200
