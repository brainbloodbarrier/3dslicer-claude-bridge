"""Anatomical constants for spine analysis tools.

All values are derived from peer-reviewed literature and established
classification systems used in clinical spine assessment.
"""

from enum import Enum

# =============================================================================
# Spine Regions
# =============================================================================

SPINE_REGIONS = frozenset(["cervical", "thoracic", "lumbar", "full"])
"""Valid spine region parameters for segmentation tools."""

# Vertebral labels per region as used by TotalSegmentator
REGION_VERTEBRAE: dict[str, tuple[str, ...]] = {
    "cervical": ("C1", "C2", "C3", "C4", "C5", "C6", "C7"),
    "thoracic": (
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
    ),
    "lumbar": ("L1", "L2", "L3", "L4", "L5", "S1"),
    "full": (
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "S1",
    ),
}

# TotalSegmentator label names mapping to standard anatomical names
TOTALSEGMENTATOR_VERTEBRA_MAP: dict[str, str] = {
    "vertebrae_C1": "C1",
    "vertebrae_C2": "C2",
    "vertebrae_C3": "C3",
    "vertebrae_C4": "C4",
    "vertebrae_C5": "C5",
    "vertebrae_C6": "C6",
    "vertebrae_C7": "C7",
    "vertebrae_T1": "T1",
    "vertebrae_T2": "T2",
    "vertebrae_T3": "T3",
    "vertebrae_T4": "T4",
    "vertebrae_T5": "T5",
    "vertebrae_T6": "T6",
    "vertebrae_T7": "T7",
    "vertebrae_T8": "T8",
    "vertebrae_T9": "T9",
    "vertebrae_T10": "T10",
    "vertebrae_T11": "T11",
    "vertebrae_T12": "T12",
    "vertebrae_L1": "L1",
    "vertebrae_L2": "L2",
    "vertebrae_L3": "L3",
    "vertebrae_L4": "L4",
    "vertebrae_L5": "L5",
}

# TotalSegmentator disc labels
TOTALSEGMENTATOR_DISC_MAP: dict[str, str] = {
    "disc_L5_S1": "L5-S1",
    "disc_L4_L5": "L4-L5",
    "disc_L3_L4": "L3-L4",
    "disc_L2_L3": "L2-L3",
    "disc_L1_L2": "L1-L2",
    "disc_T12_L1": "T12-L1",
}

# TotalSegmentator task names
# Note: "vertebrae_body" task requires a license; "total" is open and includes vertebrae
TOTALSEG_TASK_VERTEBRAE = "total"
TOTALSEG_TASK_FULL = "total"

# =============================================================================
# Segmentation Timeout
# =============================================================================

SPINE_SEGMENTATION_TIMEOUT = 300  # 5 minutes for TotalSegmentator spine tasks


# =============================================================================
# Craniovertebral Junction (CCJ) Normal Ranges
# =============================================================================
# Source: Joaquim AF, Patel AA. "Craniovertebral Junction Anatomy and
# Biomechanics." Neurosurg Clin N Am. 2018;29(2):137-144.
# Harris MB et al. "Injuries to the cervico-cranium." Spine 2002.


class CCJParameter(Enum):
    """Craniovertebral junction measurement parameters."""

    BASION_DENS_INTERVAL = "BDI"
    BASION_AXIS_INTERVAL = "BAI"
    ATLANTODENTAL_INTERVAL = "ADI"
    POWERS_RATIO = "powers_ratio"
    DENS_ANGULATION = "dens_angulation"


# CCJ normal ranges: (min, max) in mm unless otherwise noted
CCJ_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "BDI": (0.0, 12.0),  # Basion-Dens Interval (mm)
    "BAI": (-4.0, 12.0),  # Basion-Axis Interval (mm)
    "ADI_adult": (0.0, 3.0),  # Atlantodental Interval adult (mm)
    "ADI_child": (0.0, 5.0),  # Atlantodental Interval pediatric (mm)
    "powers_ratio": (0.0, 1.0),  # Powers ratio (dimensionless; >1.0 = anterior dislocation)
}


# =============================================================================
# Vertebral Body Height Ratios
# =============================================================================
# Source: Genant HK et al. "Vertebral fracture assessment using a
# semiquantitative technique." J Bone Miner Res. 1993;8(9):1137-48.


class GenantGrade(Enum):
    """Genant semiquantitative vertebral fracture grading.

    Source: Genant HK et al. JBMR 1993;8(9):1137-48.
    Based on percentage reduction in vertebral body height compared to
    the expected (adjacent) height.
    """

    NORMAL = 0  # <20% height reduction
    MILD = 1  # 20-25% height reduction (Grade 1)
    MODERATE = 2  # 25-40% height reduction (Grade 2)
    SEVERE = 3  # >40% height reduction (Grade 3)


# Genant grade thresholds as height-reduction fractions
GENANT_THRESHOLDS: dict[str, float] = {
    "mild_min": 0.20,  # >=20% loss = Grade 1
    "moderate_min": 0.25,  # >=25% loss = Grade 2
    "severe_min": 0.40,  # >=40% loss = Grade 3
}


# =============================================================================
# Pickhardt CT Bone Density (HU Thresholds)
# =============================================================================
# Source: Pickhardt PJ et al. "Opportunistic Screening for Osteoporosis
# Using Abdominal CT." Ann Intern Med. 2013;158(8):588-595.
# L1 trabecular HU on non-contrast CT.

PICKHARDT_HU_THRESHOLDS: dict[str, float] = {
    "normal_min": 135.0,  # >=135 HU = normal bone density
    "osteopenia_min": 90.0,  # 90-134 HU = osteopenia
    "osteoporosis_max": 90.0,  # <90 HU = osteoporosis
}


# =============================================================================
# Spinal Instability Neoplastic Score (SINS)
# =============================================================================
# Source: Fisher CG et al. "A Novel Classification System for Spinal
# Instability in Neoplastic Disease." Spine 2010;35(22):E1221-9.


class SINSCategory(Enum):
    """SINS stability classification.

    Source: Fisher CG et al. Spine 2010;35(22):E1221-9.
    """

    STABLE = "stable"
    INDETERMINATE = "indeterminate"
    UNSTABLE = "unstable"


# SINS score ranges for classification
SINS_RANGES: dict[str, tuple[int, int]] = {
    "stable": (0, 6),  # 0-6: stable, no surgical consult needed
    "indeterminate": (7, 12),  # 7-12: possibly unstable, consider consult
    "unstable": (13, 18),  # 13-18: unstable, surgical consult recommended
}

# SINS component scores: location
SINS_LOCATION_SCORES: dict[str, int] = {
    "junctional": 3,  # Occiput-C2, C7-T2, T11-L1, L5-S1
    "mobile": 2,  # C3-C6, L2-L4
    "semi_rigid": 1,  # T3-T10
    "rigid": 0,  # S2-S5
}

# SINS component scores: pain
SINS_PAIN_SCORES: dict[str, int] = {
    "mechanical": 3,
    "occasional_non_mechanical": 1,
    "pain_free": 0,
}

# SINS component scores: bone lesion type
SINS_LESION_SCORES: dict[str, int] = {
    "lytic": 2,
    "mixed": 1,
    "blastic": 0,
}

# SINS component scores: spinal alignment
SINS_ALIGNMENT_SCORES: dict[str, int] = {
    "subluxation_translation": 4,
    "de_novo_deformity": 2,
    "normal": 0,
}

# SINS component scores: vertebral body collapse
SINS_COLLAPSE_SCORES: dict[str, int] = {
    "greater_50_percent": 3,
    "less_50_percent": 2,
    "no_collapse_gt_50_involved": 1,
    "none": 0,
}

# SINS component scores: posterolateral involvement
SINS_POSTEROLATERAL_SCORES: dict[str, int] = {
    "bilateral": 3,
    "unilateral": 1,
    "none": 0,
}


# =============================================================================
# Dynamic Instability Criteria
# =============================================================================
# Source: White AA, Panjabi MM. "Clinical Biomechanics of the Spine."
# 2nd ed. Philadelphia: JB Lippincott; 1990.

DYNAMIC_INSTABILITY_THRESHOLDS: dict[str, float] = {
    "cervical_translation_mm": 3.5,  # >3.5 mm sagittal translation
    "cervical_angulation_deg": 11.0,  # >11° sagittal angulation
    "lumbar_translation_mm": 4.5,  # >4.5 mm sagittal translation
    "lumbar_angulation_deg": 15.0,  # >15° sagittal angulation at single level
}


# =============================================================================
# Meyerding Spondylolisthesis Grading
# =============================================================================
# Source: Meyerding HW. "Spondylolisthesis." Surg Gynecol Obstet.
# 1932;54:371-377.


class MeyerdingGrade(Enum):
    """Meyerding spondylolisthesis grade.

    Source: Meyerding HW. Surg Gynecol Obstet. 1932;54:371-377.
    Based on percentage slip of superior vertebral body over inferior.
    """

    GRADE_I = 1  # 0-25% slip
    GRADE_II = 2  # 25-50% slip
    GRADE_III = 3  # 50-75% slip
    GRADE_IV = 4  # 75-100% slip
    GRADE_V = 5  # Spondyloptosis (>100%)


MEYERDING_THRESHOLDS: dict[str, float] = {
    "grade_i_max": 0.25,
    "grade_ii_max": 0.50,
    "grade_iii_max": 0.75,
    "grade_iv_max": 1.00,
    # >1.00 = spondyloptosis (Grade V)
}


# =============================================================================
# Modic Classification (Endplate Changes)
# =============================================================================
# Source: Modic MT et al. "Degenerative disk disease: assessment of
# changes in vertebral body marrow with MR imaging." Radiology.
# 1988;166(1):193-199.


class ModicType(Enum):
    """Modic endplate change classification.

    Source: Modic MT et al. Radiology. 1988;166(1):193-199.
    MRI signal patterns of vertebral endplate and marrow changes.
    """

    NORMAL = 0  # No endplate changes
    TYPE_I = 1  # Edema/inflammation (T1 low, T2 high)
    TYPE_II = 2  # Fatty degeneration (T1 high, T2 iso/high)
    TYPE_III = 3  # Sclerosis (T1 low, T2 low)


# Modic MRI signal patterns: (T1_signal, T2_signal)
MODIC_SIGNAL_PATTERNS: dict[str, tuple[str, str]] = {
    "type_i": ("low", "high"),
    "type_ii": ("high", "isointense_or_high"),
    "type_iii": ("low", "low"),
}


# =============================================================================
# Pfirrmann Disc Degeneration Grading
# =============================================================================
# Source: Pfirrmann CW et al. "Magnetic resonance classification of
# lumbar intervertebral disc degeneration." Spine. 2001;26(17):1873-8.


class PfirrmannGrade(Enum):
    """Pfirrmann intervertebral disc degeneration grade (MRI T2).

    Source: Pfirrmann CW et al. Spine. 2001;26(17):1873-8.
    """

    GRADE_I = 1  # Homogeneous, bright white; normal height; clear nucleus-annulus boundary
    GRADE_II = 2  # Inhomogeneous, white; normal height; clear boundary
    GRADE_III = 3  # Inhomogeneous, gray; normal-to-slight decrease; unclear boundary
    GRADE_IV = 4  # Inhomogeneous, dark gray/black; moderate decrease; lost boundary
    GRADE_V = 5  # Inhomogeneous, black; collapsed space; lost boundary


PFIRRMANN_DESCRIPTIONS: dict[int, str] = {
    1: "Homogeneous bright white signal, normal disc height, clear nucleus-annulus boundary",
    2: "Inhomogeneous white signal, normal disc height, clear boundary",
    3: "Inhomogeneous gray signal, normal to slight height decrease, unclear boundary",
    4: "Inhomogeneous dark signal, moderate height decrease, lost boundary",
    5: "Inhomogeneous black signal, collapsed disc space, lost boundary",
}


# =============================================================================
# Spinal Canal Diameters and Torg-Pavlov Ratio
# =============================================================================
# Source: Torg JS, Pavlov H. "Cervical stenosis." Clin Orthop Relat Res.
# 1987;(221):77-86.
# Pavlov H et al. "Cervical spinal stenosis: determination with
# vertebral body ratio method." Radiology. 1987;164:771-775.

# Normal anteroposterior spinal canal diameter (mm) by region
SPINAL_CANAL_AP_DIAMETER: dict[str, tuple[float, float]] = {
    "cervical": (14.0, 23.0),  # Normal range C3-C7
    "thoracic": (12.0, 17.0),  # Normal range T1-T12
    "lumbar": (15.0, 25.0),  # Normal range L1-L5
}

# Absolute stenosis threshold (mm)
SPINAL_CANAL_STENOSIS_ABSOLUTE: dict[str, float] = {
    "cervical": 10.0,  # <10 mm = absolute stenosis
    "thoracic": 10.0,
    "lumbar": 10.0,
}

# Torg-Pavlov ratio: sagittal canal diameter / sagittal vertebral body diameter
TORG_PAVLOV_THRESHOLD = 0.80
"""Torg-Pavlov ratio <0.80 suggests cervical spinal stenosis.

Source: Pavlov H et al. Radiology 1987;164:771-775.
"""


# =============================================================================
# Cervical Lateral Mass and Screw Dimensions
# =============================================================================
# Source: An HS et al. "Anatomic considerations for plate-screw fixation
# of the cervical spine." Spine. 1991;16(10 Suppl):S548-51.
# Jeanneret B et al. "Posterior stabilization of the cervical spine
# with hook plates." Spine. 1991;16(3 Suppl):S56-63.

# Default screw dimensions (mm) for cervical lateral mass fixation
CERVICAL_LATERAL_MASS_SCREW_DEFAULTS: dict[str, float] = {
    "diameter_mm": 3.5,
    "length_mm": 14.0,  # Typical length range 10-16mm
    "min_length_mm": 10.0,
    "max_length_mm": 18.0,
}

# Lateral mass height by level: (min_mm, max_mm)
# Source: Pait TG et al. "Surgical anatomy of the lateral mass of the
# cervical spine." Neurosurgery. 1995;37(1):10-14.
CERVICAL_LATERAL_MASS_HEIGHT: dict[str, tuple[float, float]] = {
    "C3": (10.5, 14.0),
    "C4": (10.5, 14.5),
    "C5": (10.0, 14.0),
    "C6": (10.5, 14.5),
    "C7": (10.0, 13.0),
}

# Valid levels for Magerl technique (lateral mass screws)
VALID_LATERAL_MASS_LEVELS = frozenset(["C3", "C4", "C5", "C6", "C7"])

# Valid levels for pedicle screw placement
VALID_PEDICLE_SCREW_LEVELS = frozenset(
    [
        "C2",
        "C7",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "S1",
    ]
)

# Safety margin for screw trajectory (mm from cortex/foramen)
SCREW_SAFETY_MARGIN_MM = 2.0
"""Minimum safety margin from cortical breach or neural foramen (mm)."""


# =============================================================================
# Vertebral Body Height Reference (mm)
# =============================================================================
# Source: Gilsanz V et al. "Vertebral bone density in children: effect
# of puberty." Radiology. 1988;166(3):847-850.
# Zhou SH et al. "Geometrical dimensions of the lower lumbar vertebrae."
# Eur Spine J. 2000;9(3):242-248.

# Typical anterior vertebral body heights (mm) for adults
VERTEBRAL_BODY_HEIGHTS_MM: dict[str, tuple[float, float]] = {
    "C3": (12.0, 16.0),
    "C4": (12.0, 16.0),
    "C5": (12.0, 16.0),
    "C6": (13.0, 17.0),
    "C7": (14.0, 18.0),
    "T1": (16.0, 20.0),
    "T6": (19.0, 24.0),
    "T12": (22.0, 28.0),
    "L1": (23.0, 29.0),
    "L3": (25.0, 31.0),
    "L5": (26.0, 32.0),
}


# =============================================================================
# Cervical Instrumentation -- Technique Definitions
# =============================================================================
# Sources:
# Roy-Camille R et al. "Internal fixation of the unstable cervical spine
#   by posterior osteosynthesis with plates and screws." 1979.
# Magerl F et al. "Stable posterior fusion of the atlas and axis by
#   transarticular screw fixation." 1987.
# An HS et al. "Anatomic considerations for plate-screw fixation of the
#   cervical spine." Spine. 1991;16(10 Suppl):S548-51.
# Harms J, Melcher RP. "Posterior C1-C2 fusion with polyaxial screw
#   and rod fixation." Spine. 2001;26(22):2467-71.
# Goel A, Laheri VK. "Plate and screw fixation for atlanto-axial
#   subluxation." Acta Neurochir (Wien). 1994;129:47-53.

VALID_INSTRUMENTATION_TECHNIQUES = frozenset(
    ["pedicle", "lateral_mass", "transarticular", "c1_lateral_mass", "c2_pars", "occipital", "auto"]
)
"""Valid technique names for plan_cervical_screws."""

VALID_SIDES = frozenset(["left", "right", "bilateral"])
"""Valid side parameters for screw placement."""

VALID_LATERAL_MASS_VARIANTS = frozenset(["roy_camille", "magerl", "an", "anderson"])
"""Lateral mass screw technique variants."""

# Valid vertebral levels per technique
TECHNIQUE_VALID_LEVELS: dict[str, frozenset[str]] = {
    "pedicle": frozenset(["C2", "C3", "C4", "C5", "C6", "C7"]),
    "lateral_mass": frozenset(["C3", "C4", "C5", "C6", "C7"]),
    "transarticular": frozenset(["C1C2"]),
    "c1_lateral_mass": frozenset(["C1"]),
    "c2_pars": frozenset(["C2"]),
    "occipital": frozenset(["Occiput"]),
}

# =============================================================================
# Per-Technique Screw Defaults (mm)
# =============================================================================

CERVICAL_PEDICLE_SCREW_DEFAULTS: dict[str, float] = {
    "diameter_mm": 3.5,
    "length_mm": 22.0,
    "min_length_mm": 16.0,
    "max_length_mm": 30.0,
}

TRANSARTICULAR_SCREW_DEFAULTS: dict[str, float] = {
    "diameter_mm": 3.5,
    "length_mm": 38.0,
    "min_length_mm": 32.0,
    "max_length_mm": 44.0,
}

C1_LATERAL_MASS_SCREW_DEFAULTS: dict[str, float] = {
    "diameter_mm": 3.5,
    "length_mm": 28.0,
    "min_length_mm": 24.0,
    "max_length_mm": 32.0,
}

C2_PARS_SCREW_DEFAULTS: dict[str, float] = {
    "diameter_mm": 3.5,
    "length_mm": 24.0,
    "min_length_mm": 18.0,
    "max_length_mm": 28.0,
}

OCCIPITAL_SCREW_DEFAULTS: dict[str, float] = {
    "diameter_mm": 4.0,
    "length_mm": 10.0,
    "min_length_mm": 6.0,
    "max_length_mm": 14.0,
}

# =============================================================================
# Per-Technique Default Angulation (degrees)
# =============================================================================
# Angles follow anatomical convention:
#   medial_deg: positive = toward midline
#   lateral_deg: positive = away from midline
#   cephalad_deg: positive = toward head
#   caudal_deg: positive = toward feet

TECHNIQUE_ANGULATION: dict[str, dict[str, float]] = {
    "pedicle": {"medial_deg": 25.0, "caudal_deg": 0.0},
    "lateral_mass_roy_camille": {"lateral_deg": 10.0, "cephalad_deg": 0.0},
    "lateral_mass_magerl": {"lateral_deg": 25.0, "cephalad_deg": 45.0},
    "lateral_mass_an": {"lateral_deg": 30.0, "cephalad_deg": 15.0},
    "lateral_mass_anderson": {"lateral_deg": 10.0, "cephalad_deg": 30.0},
    "transarticular": {"medial_deg": 10.0, "cephalad_deg": 50.0},
    "c1_lateral_mass": {"medial_deg": 10.0, "caudal_deg": 5.0},
    "c2_pars": {"medial_deg": 15.0, "cephalad_deg": 25.0},
    "occipital": {"perpendicular_deg": 0.0},
}

# =============================================================================
# Instrumentation Safety Thresholds (mm)
# =============================================================================

VA_SAFETY_DISTANCE_MM = 2.0
"""Minimum distance from screw trajectory to vertebral artery (mm)."""

CANAL_SAFETY_DISTANCE_MM = 2.0
"""Minimum distance from screw trajectory to spinal canal (mm)."""

NERVE_ROOT_SAFETY_DISTANCE_MM = 2.0
"""Minimum distance from screw trajectory to nerve root (mm)."""

ISTHMUS_MIN_HEIGHT_MM = 5.0
"""Minimum C2 isthmus height for transarticular screw (mm).
Source: Paramore CG et al. Spine 1996;21(13):1501-7."""

ISTHMUS_MIN_WIDTH_MM = 4.0
"""Minimum C2 isthmus width for transarticular screw (mm)."""

OCCIPITAL_MIN_THICKNESS_MM = 6.0
"""Minimum occipital bone thickness for bicortical purchase (mm)."""

CERVICAL_PEDICLE_MIN_WIDTH_MM = 4.5
"""Minimum pedicle width to safely accept 3.5mm screw (mm).
Source: Abumi K et al. Spine 1994;19(17):1983-6."""

# =============================================================================
# Instrumentation Timeout
# =============================================================================

INSTRUMENTATION_TIMEOUT = 120
"""Timeout for instrumentation planning code execution (seconds)."""

# =============================================================================
# Safety Color Codes
# =============================================================================

SAFETY_GREEN = "green"
"""Safe -- all clearances adequate."""

SAFETY_YELLOW = "yellow"
"""Warning -- marginal clearances, proceed with caution."""

SAFETY_RED = "red"
"""Blocked -- insufficient clearance, technique contraindicated."""


# =============================================================================
# X-ray Diagnostic Constants
# =============================================================================

# Coronal balance thresholds
CORONAL_C7_CSVL_THRESHOLD_MM = 20.0
CORONAL_COBB_MILD_THRESHOLD_DEG = 25.0
CORONAL_COBB_MODERATE_THRESHOLD_DEG = 45.0

# Vertebra label validation patterns
VERTEBRA_LABEL_PATTERN = r"^[CTLS][0-9]{1,2}$"
VERTEBRA_LEVEL_PATTERN = r"^[CTLS][0-9]{1,2}-[CTLS][0-9]{1,2}$"

# SRS-Schwab classification thresholds (Schwab F et al. Spine 2012)
SCHWAB_PI_LL_THRESHOLDS = {
    "matched": 10.0,
    "moderate": 20.0,
}

SCHWAB_SVA_THRESHOLDS = {
    "grade_0": 40.0,
    "grade_1": 95.0,
}

SCHWAB_PT_THRESHOLDS = {
    "grade_0": 20.0,
    "grade_1": 30.0,
}

# Roussouly lordosis type criteria (Roussouly P et al. Spine 2005)
ROUSSOULY_SS_THRESHOLDS = {
    "type_1_max": 35.0,
    "type_2_max": 35.0,
    "type_3_min": 35.0,
    "type_3_max": 45.0,
}

# =============================================================================
# MRI Diagnostic Constants
# =============================================================================

# Timeout for multi-sequence MRI analysis (registration + segmentation + analysis)
MRI_ANALYSIS_TIMEOUT = 360  # 6 minutes

# Modic signal ratio thresholds (relative to reference vertebral body)
MODIC_T1_LOW_THRESHOLD = 0.85
MODIC_T1_HIGH_THRESHOLD = 1.15
MODIC_T2_LOW_THRESHOLD = 0.85
MODIC_T2_HIGH_THRESHOLD = 1.15

# Pfirrmann disc signal ratio thresholds (relative to CSF)
PFIRRMANN_BRIGHT_THRESHOLD = 0.80
PFIRRMANN_WHITE_THRESHOLD = 0.60
PFIRRMANN_GRAY_THRESHOLD = 0.35
PFIRRMANN_DARK_THRESHOLD = 0.15

# Disc height loss thresholds for Pfirrmann grading
PFIRRMANN_HEIGHT_SLIGHT_DECREASE = 0.10
PFIRRMANN_HEIGHT_MODERATE_DECREASE = 0.30
PFIRRMANN_HEIGHT_COLLAPSED = 0.60

# Homogeneity threshold (coefficient of variation)
DISC_HOMOGENEOUS_CV_THRESHOLD = 0.15

# Cord compression thresholds
CORD_COMPRESSION_RATIO_NORMAL = 0.40
CORD_COMPRESSION_RATIO_MILD = 0.30
CORD_COMPRESSION_RATIO_MODERATE = 0.20

# T2 hyperintensity threshold for myelopathy detection
CORD_T2_HYPERINTENSITY_THRESHOLD = 1.30

# MSCC ratio threshold
MSCC_THRESHOLD = 0.50

# Metastatic lesion signal thresholds (relative to normal marrow)
METASTASIS_T1_LOW_THRESHOLD = 0.70
METASTASIS_T2_HIGH_THRESHOLD = 1.40
METASTASIS_T2_LOW_THRESHOLD = 0.70

# Supported regions for disc analysis (Pfirrmann)
DISC_SUPPORTED_REGIONS = frozenset(["lumbar"])
