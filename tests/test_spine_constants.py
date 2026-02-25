"""Unit tests for spine anatomical constants."""

from slicer_mcp.spine_constants import (
    CCJ_NORMAL_RANGES,
    CERVICAL_LATERAL_MASS_HEIGHT,
    CERVICAL_LATERAL_MASS_SCREW_DEFAULTS,
    DYNAMIC_INSTABILITY_THRESHOLDS,
    GENANT_THRESHOLDS,
    MEYERDING_THRESHOLDS,
    MODIC_SIGNAL_PATTERNS,
    PFIRRMANN_DESCRIPTIONS,
    PICKHARDT_HU_THRESHOLDS,
    REGION_VERTEBRAE,
    SCREW_SAFETY_MARGIN_MM,
    SINS_RANGES,
    SPINAL_CANAL_AP_DIAMETER,
    SPINAL_CANAL_STENOSIS_ABSOLUTE,
    SPINE_REGIONS,
    SPINE_SEGMENTATION_TIMEOUT,
    TORG_PAVLOV_THRESHOLD,
    TOTALSEG_TASK_FULL,
    TOTALSEG_TASK_VERTEBRAE,
    TOTALSEGMENTATOR_DISC_MAP,
    TOTALSEGMENTATOR_VERTEBRA_MAP,
    VALID_LATERAL_MASS_LEVELS,
    VALID_PEDICLE_SCREW_LEVELS,
    VERTEBRAL_BODY_HEIGHTS_MM,
    CCJParameter,
    GenantGrade,
    MeyerdingGrade,
    ModicType,
    PfirrmannGrade,
    SINSCategory,
)

# =============================================================================
# Spine Regions and Vertebrae
# =============================================================================


class TestSpineRegions:
    """Tests for spine region definitions."""

    def test_all_regions_present(self):
        """All expected regions must be defined."""
        assert SPINE_REGIONS == frozenset(["cervical", "thoracic", "lumbar", "full"])

    def test_region_vertebrae_cervical_count(self):
        """Cervical region must have 7 vertebrae (C1-C7)."""
        assert len(REGION_VERTEBRAE["cervical"]) == 7
        assert REGION_VERTEBRAE["cervical"][0] == "C1"
        assert REGION_VERTEBRAE["cervical"][-1] == "C7"

    def test_region_vertebrae_thoracic_count(self):
        """Thoracic region must have 12 vertebrae (T1-T12)."""
        assert len(REGION_VERTEBRAE["thoracic"]) == 12
        assert REGION_VERTEBRAE["thoracic"][0] == "T1"
        assert REGION_VERTEBRAE["thoracic"][-1] == "T12"

    def test_region_vertebrae_lumbar_count(self):
        """Lumbar region must have 5 vertebrae (L1-L5)."""
        assert len(REGION_VERTEBRAE["lumbar"]) == 5
        assert REGION_VERTEBRAE["lumbar"][0] == "L1"
        assert REGION_VERTEBRAE["lumbar"][-1] == "L5"

    def test_region_vertebrae_full_is_union(self):
        """Full region must be the union of cervical + thoracic + lumbar."""
        expected = (
            REGION_VERTEBRAE["cervical"] + REGION_VERTEBRAE["thoracic"] + REGION_VERTEBRAE["lumbar"]
        )
        assert REGION_VERTEBRAE["full"] == expected

    def test_full_region_count(self):
        """Full region must have 24 vertebrae (C1-L5)."""
        assert len(REGION_VERTEBRAE["full"]) == 24


# =============================================================================
# TotalSegmentator Mappings
# =============================================================================


class TestTotalSegmentatorMappings:
    """Tests for TotalSegmentator label mappings."""

    def test_vertebra_map_covers_all_vertebrae(self):
        """Map must cover all 24 vertebrae from C1-L5."""
        assert len(TOTALSEGMENTATOR_VERTEBRA_MAP) == 24
        assert "vertebrae_C1" in TOTALSEGMENTATOR_VERTEBRA_MAP
        assert "vertebrae_L5" in TOTALSEGMENTATOR_VERTEBRA_MAP

    def test_vertebra_map_values_match_labels(self):
        """Map values must match expected anatomical labels."""
        assert TOTALSEGMENTATOR_VERTEBRA_MAP["vertebrae_T12"] == "T12"
        assert TOTALSEGMENTATOR_VERTEBRA_MAP["vertebrae_C7"] == "C7"

    def test_disc_map_has_entries(self):
        """Disc map must have at least the lumbar discs."""
        assert len(TOTALSEGMENTATOR_DISC_MAP) >= 5
        assert TOTALSEGMENTATOR_DISC_MAP["disc_L4_L5"] == "L4-L5"

    def test_task_names_are_strings(self):
        """TotalSegmentator task names must be non-empty strings."""
        assert isinstance(TOTALSEG_TASK_VERTEBRAE, str) and TOTALSEG_TASK_VERTEBRAE
        assert isinstance(TOTALSEG_TASK_FULL, str) and TOTALSEG_TASK_FULL


# =============================================================================
# Segmentation Timeout
# =============================================================================


class TestSegmentationTimeout:
    """Tests for spine segmentation timeout constant."""

    def test_timeout_is_positive(self):
        """Timeout must be a positive integer."""
        assert SPINE_SEGMENTATION_TIMEOUT > 0

    def test_timeout_is_reasonable(self):
        """Timeout must be between 60 and 600 seconds."""
        assert 60 <= SPINE_SEGMENTATION_TIMEOUT <= 600


# =============================================================================
# CCJ Normal Ranges
# =============================================================================


class TestCCJNormalRanges:
    """Tests for craniovertebral junction normal ranges."""

    def test_ccj_parameter_enum_members(self):
        """CCJParameter enum must have expected members."""
        assert CCJParameter.BASION_DENS_INTERVAL.value == "BDI"
        assert CCJParameter.ATLANTODENTAL_INTERVAL.value == "ADI"
        assert CCJParameter.POWERS_RATIO.value == "powers_ratio"

    def test_bdi_range(self):
        """BDI normal range must be 0-12 mm."""
        assert CCJ_NORMAL_RANGES["BDI"] == (0.0, 12.0)

    def test_bai_range(self):
        """BAI normal range must include negative values."""
        low, high = CCJ_NORMAL_RANGES["BAI"]
        assert low < 0  # BAI can be negative
        assert high > 0

    def test_adi_adult_stricter_than_child(self):
        """Adult ADI range must be stricter (smaller max) than pediatric."""
        adult_max = CCJ_NORMAL_RANGES["ADI_adult"][1]
        child_max = CCJ_NORMAL_RANGES["ADI_child"][1]
        assert adult_max < child_max

    def test_powers_ratio_upper_bound(self):
        """Powers ratio >1.0 indicates anterior dislocation."""
        assert CCJ_NORMAL_RANGES["powers_ratio"][1] == 1.0

    def test_all_ranges_have_min_less_than_max(self):
        """All CCJ ranges must have min < max."""
        for key, (low, high) in CCJ_NORMAL_RANGES.items():
            assert low < high, f"CCJ range {key}: min {low} >= max {high}"


# =============================================================================
# Genant Fracture Grading
# =============================================================================


class TestGenantGrading:
    """Tests for Genant vertebral fracture grading."""

    def test_genant_grades(self):
        """GenantGrade enum must have 4 levels."""
        assert GenantGrade.NORMAL.value == 0
        assert GenantGrade.MILD.value == 1
        assert GenantGrade.MODERATE.value == 2
        assert GenantGrade.SEVERE.value == 3

    def test_thresholds_are_increasing(self):
        """Thresholds must increase: mild < moderate < severe."""
        assert GENANT_THRESHOLDS["mild_min"] < GENANT_THRESHOLDS["moderate_min"]
        assert GENANT_THRESHOLDS["moderate_min"] < GENANT_THRESHOLDS["severe_min"]

    def test_thresholds_are_fractions(self):
        """All thresholds must be between 0 and 1 (fractions)."""
        for key, val in GENANT_THRESHOLDS.items():
            assert 0.0 < val < 1.0, f"Genant threshold {key}={val} not a valid fraction"


# =============================================================================
# Pickhardt HU Thresholds
# =============================================================================


class TestPickhardtThresholds:
    """Tests for Pickhardt CT bone density thresholds."""

    def test_normal_above_osteopenia(self):
        """Normal HU threshold must be above osteopenia threshold."""
        assert PICKHARDT_HU_THRESHOLDS["normal_min"] > PICKHARDT_HU_THRESHOLDS["osteopenia_min"]

    def test_osteoporosis_boundary(self):
        """Osteoporosis cutoff must equal osteopenia lower bound."""
        assert (
            PICKHARDT_HU_THRESHOLDS["osteoporosis_max"] == PICKHARDT_HU_THRESHOLDS["osteopenia_min"]
        )

    def test_thresholds_positive(self):
        """All HU thresholds must be positive."""
        for key, val in PICKHARDT_HU_THRESHOLDS.items():
            assert val > 0, f"Pickhardt threshold {key}={val} is not positive"


# =============================================================================
# SINS Classification
# =============================================================================


class TestSINSClassification:
    """Tests for Spinal Instability Neoplastic Score."""

    def test_sins_category_enum(self):
        """SINSCategory enum must have 3 categories."""
        assert len(SINSCategory) == 3

    def test_sins_ranges_non_overlapping(self):
        """SINS score ranges must be contiguous and non-overlapping."""
        stable_max = SINS_RANGES["stable"][1]
        indet_min = SINS_RANGES["indeterminate"][0]
        indet_max = SINS_RANGES["indeterminate"][1]
        unstable_min = SINS_RANGES["unstable"][0]

        assert indet_min == stable_max + 1
        assert unstable_min == indet_max + 1

    def test_sins_total_range(self):
        """SINS must span 0-18."""
        assert SINS_RANGES["stable"][0] == 0
        assert SINS_RANGES["unstable"][1] == 18

    def test_sins_component_scores_non_negative(self):
        """All SINS component scores must be non-negative integers."""
        from slicer_mcp.spine_constants import (
            SINS_ALIGNMENT_SCORES,
            SINS_COLLAPSE_SCORES,
            SINS_LESION_SCORES,
            SINS_LOCATION_SCORES,
            SINS_PAIN_SCORES,
            SINS_POSTEROLATERAL_SCORES,
        )

        for mapping in [
            SINS_LOCATION_SCORES,
            SINS_PAIN_SCORES,
            SINS_LESION_SCORES,
            SINS_ALIGNMENT_SCORES,
            SINS_COLLAPSE_SCORES,
            SINS_POSTEROLATERAL_SCORES,
        ]:
            for key, val in mapping.items():
                assert isinstance(val, int) and val >= 0, f"SINS score {key}={val} invalid"


# =============================================================================
# Dynamic Instability Criteria
# =============================================================================


class TestDynamicInstability:
    """Tests for White-Panjabi dynamic instability thresholds."""

    def test_cervical_thresholds_defined(self):
        """Cervical translation and angulation thresholds must exist."""
        assert "cervical_translation_mm" in DYNAMIC_INSTABILITY_THRESHOLDS
        assert "cervical_angulation_deg" in DYNAMIC_INSTABILITY_THRESHOLDS

    def test_lumbar_thresholds_defined(self):
        """Lumbar translation and angulation thresholds must exist."""
        assert "lumbar_translation_mm" in DYNAMIC_INSTABILITY_THRESHOLDS
        assert "lumbar_angulation_deg" in DYNAMIC_INSTABILITY_THRESHOLDS

    def test_all_thresholds_positive(self):
        """All instability thresholds must be positive."""
        for key, val in DYNAMIC_INSTABILITY_THRESHOLDS.items():
            assert val > 0, f"Instability threshold {key}={val} is not positive"


# =============================================================================
# Meyerding Spondylolisthesis
# =============================================================================


class TestMeyerdingGrading:
    """Tests for Meyerding spondylolisthesis grading."""

    def test_meyerding_has_five_grades(self):
        """MeyerdingGrade enum must have 5 grades (I-V)."""
        assert len(MeyerdingGrade) == 5

    def test_thresholds_are_increasing(self):
        """Meyerding thresholds must increase by grade."""
        assert MEYERDING_THRESHOLDS["grade_i_max"] < MEYERDING_THRESHOLDS["grade_ii_max"]
        assert MEYERDING_THRESHOLDS["grade_ii_max"] < MEYERDING_THRESHOLDS["grade_iii_max"]
        assert MEYERDING_THRESHOLDS["grade_iii_max"] < MEYERDING_THRESHOLDS["grade_iv_max"]

    def test_grade_iv_max_is_100_percent(self):
        """Grade IV maximum is 100% slip (1.0)."""
        assert MEYERDING_THRESHOLDS["grade_iv_max"] == 1.0


# =============================================================================
# Modic Classification
# =============================================================================


class TestModicClassification:
    """Tests for Modic endplate change classification."""

    def test_modic_has_four_types(self):
        """ModicType enum must have 4 types (normal + I-III)."""
        assert len(ModicType) == 4
        assert ModicType.NORMAL.value == 0
        assert ModicType.TYPE_III.value == 3

    def test_signal_patterns_defined_for_all_types(self):
        """Signal patterns must be defined for types I-III."""
        assert "type_i" in MODIC_SIGNAL_PATTERNS
        assert "type_ii" in MODIC_SIGNAL_PATTERNS
        assert "type_iii" in MODIC_SIGNAL_PATTERNS

    def test_type_i_t2_high(self):
        """Type I must have high T2 signal (edema)."""
        _, t2 = MODIC_SIGNAL_PATTERNS["type_i"]
        assert t2 == "high"

    def test_type_iii_both_low(self):
        """Type III must have low T1 and low T2 (sclerosis)."""
        t1, t2 = MODIC_SIGNAL_PATTERNS["type_iii"]
        assert t1 == "low"
        assert t2 == "low"


# =============================================================================
# Pfirrmann Disc Degeneration
# =============================================================================


class TestPfirrmannGrading:
    """Tests for Pfirrmann disc degeneration grading."""

    def test_pfirrmann_has_five_grades(self):
        """PfirrmannGrade enum must have 5 grades (I-V)."""
        assert len(PfirrmannGrade) == 5

    def test_descriptions_match_grades(self):
        """A description must exist for each grade."""
        for grade in PfirrmannGrade:
            assert grade.value in PFIRRMANN_DESCRIPTIONS

    def test_descriptions_are_non_empty(self):
        """All descriptions must be non-empty strings."""
        for grade_val, desc in PFIRRMANN_DESCRIPTIONS.items():
            assert (
                isinstance(desc, str) and len(desc) > 10
            ), f"Pfirrmann description for grade {grade_val} is too short"


# =============================================================================
# Spinal Canal and Torg-Pavlov
# =============================================================================


class TestSpinalCanalDimensions:
    """Tests for spinal canal diameter and Torg-Pavlov ratio."""

    def test_canal_diameters_defined_for_all_regions(self):
        """AP diameters must be defined for cervical, thoracic, lumbar."""
        assert "cervical" in SPINAL_CANAL_AP_DIAMETER
        assert "thoracic" in SPINAL_CANAL_AP_DIAMETER
        assert "lumbar" in SPINAL_CANAL_AP_DIAMETER

    def test_canal_ranges_valid(self):
        """All canal diameter ranges must have min < max and positive values."""
        for region, (low, high) in SPINAL_CANAL_AP_DIAMETER.items():
            assert low > 0 and high > low, f"Canal diameter {region}: ({low}, {high}) invalid"

    def test_stenosis_thresholds_positive(self):
        """Stenosis thresholds must be positive."""
        for region, val in SPINAL_CANAL_STENOSIS_ABSOLUTE.items():
            assert val > 0, f"Stenosis threshold for {region} is not positive"

    def test_torg_pavlov_threshold(self):
        """Torg-Pavlov threshold must be 0.80."""
        assert TORG_PAVLOV_THRESHOLD == 0.80


# =============================================================================
# Screw and Lateral Mass Dimensions
# =============================================================================


class TestScrewDimensions:
    """Tests for cervical lateral mass and screw constants."""

    def test_screw_defaults_complete(self):
        """Screw defaults must include diameter and length."""
        assert "diameter_mm" in CERVICAL_LATERAL_MASS_SCREW_DEFAULTS
        assert "length_mm" in CERVICAL_LATERAL_MASS_SCREW_DEFAULTS
        assert "min_length_mm" in CERVICAL_LATERAL_MASS_SCREW_DEFAULTS
        assert "max_length_mm" in CERVICAL_LATERAL_MASS_SCREW_DEFAULTS

    def test_screw_length_range_valid(self):
        """Min length must be less than max length."""
        defaults = CERVICAL_LATERAL_MASS_SCREW_DEFAULTS
        assert defaults["min_length_mm"] < defaults["length_mm"] < defaults["max_length_mm"]

    def test_lateral_mass_height_levels(self):
        """Lateral mass heights must be defined for C3-C7."""
        for level in ["C3", "C4", "C5", "C6", "C7"]:
            assert level in CERVICAL_LATERAL_MASS_HEIGHT

    def test_valid_lateral_mass_levels_is_frozenset(self):
        """Lateral mass levels must be a frozenset."""
        assert isinstance(VALID_LATERAL_MASS_LEVELS, frozenset)
        assert "C3" in VALID_LATERAL_MASS_LEVELS
        assert "C7" in VALID_LATERAL_MASS_LEVELS

    def test_pedicle_screw_levels(self):
        """Pedicle screw levels must include thoracic and lumbar."""
        assert "T1" in VALID_PEDICLE_SCREW_LEVELS
        assert "L5" in VALID_PEDICLE_SCREW_LEVELS
        assert "S1" in VALID_PEDICLE_SCREW_LEVELS

    def test_safety_margin_positive(self):
        """Safety margin must be positive."""
        assert SCREW_SAFETY_MARGIN_MM > 0


# =============================================================================
# Vertebral Body Heights
# =============================================================================


class TestVertebralBodyHeights:
    """Tests for vertebral body height reference values."""

    def test_heights_have_valid_ranges(self):
        """All height ranges must have min < max and positive values."""
        for level, (low, high) in VERTEBRAL_BODY_HEIGHTS_MM.items():
            assert low > 0 and high > low, f"Height for {level}: ({low}, {high}) invalid"

    def test_height_increases_craniocaudal(self):
        """Vertebral heights generally increase from cervical to lumbar."""
        c3_max = VERTEBRAL_BODY_HEIGHTS_MM["C3"][1]
        l5_min = VERTEBRAL_BODY_HEIGHTS_MM["L5"][0]
        assert l5_min > c3_max
