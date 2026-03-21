"""Centralized constants for Slicer MCP Bridge."""

__all__ = [
    "AUDIT_CODE_MAX_LENGTH",
    "AUDIT_RESULT_MAX_LENGTH",
    "BRAIN_EXTRACTION_TIMEOUT",
    "CAPTURE_3D_VIEW_TIMEOUT",
    "CAPTURE_MAX_DIMENSION",
    "CAPTURE_MIN_DIMENSION",
    "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
    "CORD_SCREENING_REGIONS",
    "DEFAULT_SAMPLING_PERCENTAGE",
    "DEFAULT_SLICER_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "DICOM_UID_PATTERN",
    "EXPORT_FILENAME_PATTERN",
    "EXPORT_FORMAT_EXTENSIONS",
    "FALLBACK_SAMPLE_DATASETS",
    "FORBIDDEN_PATH_COMPONENTS",
    "LANDMARK_LABEL_PATTERN",
    "LONG_OPERATION_TIMEOUT",
    "MAX_DICOM_UID_LENGTH",
    "MAX_EXPORT_FILENAME_LENGTH",
    "MAX_FOLDER_PATH_LENGTH",
    "MAX_LANDMARK_LABEL_LENGTH",
    "MAX_LANDMARKS",
    "MAX_NODE_ID_LENGTH",
    "MAX_PYTHON_CODE_LENGTH",
    "MAX_SEGMENT_NAME_LENGTH",
    "MAX_XRAY_LANDMARKS",
    "MIN_LANDMARK_PAIRS",
    "MODEL_EXPORT_TIMEOUT",
    "REGISTRATION_TIMEOUT",
    "RETRY_BACKOFF_BASE",
    "RETRY_MAX_ATTEMPTS",
    "SEGMENT_STATISTICS_VOLUME_KEY",
    "SEGMENTATION_TIMEOUT",
    "SLICER_MIN_VERSION",
    "SLICER_TESTED_VERSIONS",
    "VALID_3D_AXES",
    "VALID_BRAIN_EXTRACTION_METHODS",
    "VALID_EXPORT_FORMATS",
    "VALID_GUI_MODES",
    "VALID_HDBET_DEVICES",
    "VALID_INIT_MODES",
    "VALID_INTERPOLATION_MODES",
    "VALID_LANDMARK_TRANSFORM_TYPES",
    "VALID_LAYOUTS",
    "VALID_TRANSFORM_TYPES",
    "VALID_VR_PRESETS",
    "VIEW_3D",
    "VIEW_AXIAL",
    "VIEW_CORONAL",
    "VIEW_FULL",
    "VIEW_MAP",
    "VIEW_SAGITTAL",
    "VOLUME_RENDERING_TIMEOUT",
    "VR_OPACITY_SCALE_MAX",
    "VR_OPACITY_SCALE_MIN",
]

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

# Valid 3D view camera axes (for look_from_axis parameter)
VALID_3D_AXES = frozenset(["left", "right", "anterior", "posterior", "superior", "inferior"])

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
# Registration & Landmark Constants
# =============================================================================
VALID_TRANSFORM_TYPES: frozenset[str] = frozenset(
    {
        "Rigid",
        "ScaleVersor3D",
        "ScaleSkewVersor3D",
        "Affine",
        "BSpline",
    }
)
VALID_INIT_MODES: frozenset[str] = frozenset(
    {
        "useMomentsAlign",
        "useCenterOfHeadAlign",
        "useGeometryAlign",
        "Off",
    }
)
VALID_INTERPOLATION_MODES: frozenset[str] = frozenset(
    {
        "Linear",
        "BSpline",
        "WindowedSinc",
        "NearestNeighbor",
    }
)
VALID_LANDMARK_TRANSFORM_TYPES: frozenset[str] = frozenset(
    {
        "Rigid",
        "Similarity",
        "Affine",
    }
)
MIN_LANDMARK_PAIRS = 3
MAX_LANDMARK_LABEL_LENGTH = 64
LANDMARK_LABEL_PATTERN = r"^[a-zA-Z0-9_\-. ]+$"
DEFAULT_SAMPLING_PERCENTAGE = 0.01
MAX_LANDMARKS = 500
MAX_XRAY_LANDMARKS = 100  # per-tool safety limit for X-ray landmark measurements

# =============================================================================
# Workflow Configuration
# =============================================================================
# Regions where cord compression screening is clinically relevant
CORD_SCREENING_REGIONS = frozenset(["cervical", "thoracic"])

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


# =============================================================================
# Volume Rendering & 3D Model Export Constants
# =============================================================================
VOLUME_RENDERING_TIMEOUT = 60  # seconds for volume rendering setup
MODEL_EXPORT_TIMEOUT = 120  # seconds for model export (large meshes)
CAPTURE_3D_VIEW_TIMEOUT = 120  # seconds for high-res 3D view captures

# Volume rendering property ranges
VR_OPACITY_SCALE_MIN = 0.0
VR_OPACITY_SCALE_MAX = 10.0

# 3D view capture dimension limits
CAPTURE_MIN_DIMENSION = 1
CAPTURE_MAX_DIMENSION = 8192

VALID_VR_PRESETS: frozenset[str] = frozenset(
    {
        "CT-AAA",
        "CT-AAA2",
        "CT-Bone",
        "CT-Bones",
        "CT-Cardiac",
        "CT-Cardiac2",
        "CT-Cardiac3",
        "CT-Chest-Contrast-Enhanced",
        "CT-Chest-Vasculature",
        "CT-Coronary-Arteries",
        "CT-Coronary-Arteries-2",
        "CT-Coronary-Arteries-3",
        "CT-Cropped-Volume-Bone",
        "CT-Fat",
        "CT-Liver-Vasculature",
        "CT-Lung",
        "CT-MIP",
        "CT-Muscle",
        "CT-Pulmonary-Arteries",
        "CT-Soft-Tissue",
        "CT-Air",
        "MR-Angio",
        "MR-Default",
        "MR-MIP",
        "MR-T2-Brain",
    }
)

VALID_EXPORT_FORMATS: frozenset[str] = frozenset(
    {
        "STL",
        "OBJ",
        "PLY",
        "VTK",
    }
)

EXPORT_FORMAT_EXTENSIONS: dict[str, str] = {
    "STL": ".stl",
    "OBJ": ".obj",
    "PLY": ".ply",
    "VTK": ".vtk",
}

MAX_EXPORT_FILENAME_LENGTH = 255
EXPORT_FILENAME_PATTERN = r"^[a-zA-Z0-9_\-. ()]+$"
