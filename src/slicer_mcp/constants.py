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
MAX_PYTHON_CODE_LENGTH = 100000  # 100 KB max for execute_python code

# =============================================================================
# Audit Logging
# =============================================================================
AUDIT_CODE_MAX_LENGTH = 2000  # Increased from 500 to capture meaningful code snippets
AUDIT_RESULT_MAX_LENGTH = 500  # Increased from 200 for better debugging context

# =============================================================================
# Slicer Version Compatibility
# =============================================================================
SLICER_MIN_VERSION = "5.0.0"
SLICER_TESTED_VERSIONS = frozenset(
    [
        "5.0.0",
        "5.2.0",
        "5.2.1",
        "5.4.0",
        "5.6.0",
        "5.6.1",
        "5.6.2",
    ]
)

# =============================================================================
# Circuit Breaker Configuration
# =============================================================================
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Open circuit after 5 consecutive failures
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 30.0  # Wait 30 seconds before testing recovery

# =============================================================================
# DICOM Configuration
# =============================================================================
# DICOM UID format: digits and dots only (e.g., "1.2.840.113619.2.55.3.604688")
DICOM_UID_PATTERN = r"^[0-9]+(\.[0-9]+)*$"
MAX_DICOM_UID_LENGTH = 64
MAX_FOLDER_PATH_LENGTH = 4096

# Forbidden path components for security (path traversal prevention)
FORBIDDEN_PATH_COMPONENTS = frozenset([".."])

# =============================================================================
# Brain Extraction Configuration
# =============================================================================
# Valid brain extraction methods
VALID_BRAIN_EXTRACTION_METHODS = frozenset(["hd-bet", "swiss"])

# Valid HD-BET device options (GPU indices as strings, plus special values)
VALID_HDBET_DEVICES = frozenset(["auto", "cpu", "0", "1", "2", "3"])

# Extended timeout for brain extraction (5 minutes for CPU processing)
BRAIN_EXTRACTION_TIMEOUT = 300

# =============================================================================
# Long Operation Timeouts
# =============================================================================
# Timeout values for operations that may take longer than the default
LONG_OPERATION_TIMEOUT = 120  # 2 minutes for general long operations
SEGMENTATION_TIMEOUT = 180  # 3 minutes for segmentation operations
REGISTRATION_TIMEOUT = 300  # 5 minutes for registration operations

# =============================================================================
# Sample Data Configuration
# =============================================================================
# Fallback list of common sample datasets (used when dynamic discovery fails)
FALLBACK_SAMPLE_DATASETS = (
    "MRHead",
    "CTChest",
    "CTACardio",
    "DTIBrain",
    "MRBrainTumor1",
    "MRBrainTumor2",
)

# =============================================================================
# Segment Statistics Keys
# =============================================================================
SEGMENT_STATISTICS_VOLUME_KEY = "SegmentStatistics.volume_cc"
