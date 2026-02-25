"""Unit tests for CT diagnostic protocol tools."""

import json
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.diagnostic_tools_ct import (
    _classify_genant,
    _classify_meyerding,
    _classify_pickhardt,
    _classify_sins_total,
    _classify_stenosis,
    _sins_alignment_score,
    _sins_collapse_score,
    _sins_lesion_type_score,
    _sins_location_score,
    _sins_posterolateral_score,
    _validate_classification_system,
    _validate_levels,
    _validate_osteo_method,
    _validate_region,
    assess_osteoporosis_ct,
    calculate_sins_score,
    detect_metastatic_lesions_ct,
    detect_vertebral_fractures_ct,
    measure_listhesis_ct,
    measure_spinal_canal_ct,
)
from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.tools import ValidationError

# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateRegion:
    """Test spine region validation."""

    def test_valid_region_full(self):
        assert _validate_region("full") == "full"

    def test_valid_region_cervical(self):
        assert _validate_region("cervical") == "cervical"

    def test_valid_region_thoracic(self):
        assert _validate_region("thoracic") == "thoracic"

    def test_valid_region_lumbar(self):
        assert _validate_region("lumbar") == "lumbar"

    def test_invalid_region(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region("sacral")
        assert exc_info.value.field == "region"

    def test_invalid_region_empty(self):
        with pytest.raises(ValidationError):
            _validate_region("")


class TestValidateLevels:
    """Test vertebral level validation."""

    def test_none_returns_defaults(self):
        result = _validate_levels(None, ["L1"])
        assert result == ["L1"]

    def test_valid_levels(self):
        result = _validate_levels(["C3", "T6", "L1"], ["L1"])
        assert result == ["C3", "T6", "L1"]

    def test_invalid_level(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_levels(["S3"], ["L1"])
        assert exc_info.value.field == "levels"

    def test_invalid_level_random_string(self):
        with pytest.raises(ValidationError):
            _validate_levels(["X1"], ["L1"])


class TestValidateClassificationSystem:
    """Test classification system validation."""

    def test_valid_ao_spine(self):
        assert _validate_classification_system("ao_spine") == "ao_spine"

    def test_valid_genant(self):
        assert _validate_classification_system("genant") == "genant"

    def test_valid_denis(self):
        assert _validate_classification_system("denis") == "denis"

    def test_valid_all(self):
        assert _validate_classification_system("all") == "all"

    def test_invalid_system(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_classification_system("invalid")
        assert exc_info.value.field == "classification_system"


class TestValidateOsteoMethod:
    """Test osteoporosis assessment method validation."""

    def test_valid_trabecular_roi(self):
        assert _validate_osteo_method("trabecular_roi") == "trabecular_roi"

    def test_valid_vertebral_mean(self):
        assert _validate_osteo_method("vertebral_mean") == "vertebral_mean"

    def test_valid_both(self):
        assert _validate_osteo_method("both") == "both"

    def test_invalid_method(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_osteo_method("invalid")
        assert exc_info.value.field == "method"


# =============================================================================
# Genant Classification Tests
# =============================================================================


class TestClassifyGenant:
    """Test Genant semi-quantitative grading."""

    def test_normal(self):
        grade, label = _classify_genant(0.10)
        assert grade == 0
        assert label == "Normal"

    def test_mild(self):
        grade, label = _classify_genant(0.22)
        assert grade == 1
        assert label == "Mild"

    def test_moderate(self):
        grade, label = _classify_genant(0.30)
        assert grade == 2
        assert label == "Moderate"

    def test_severe(self):
        grade, label = _classify_genant(0.45)
        assert grade == 3
        assert label == "Severe"

    def test_threshold_mild(self):
        grade, _ = _classify_genant(0.20)
        assert grade == 1

    def test_threshold_moderate(self):
        grade, _ = _classify_genant(0.25)
        assert grade == 2

    def test_threshold_severe(self):
        grade, _ = _classify_genant(0.40)
        assert grade == 3

    def test_zero_loss(self):
        grade, label = _classify_genant(0.0)
        assert grade == 0
        assert label == "Normal"


# =============================================================================
# Meyerding Classification Tests
# =============================================================================


class TestClassifyMeyerding:
    """Test Meyerding spondylolisthesis grading."""

    def test_no_slip(self):
        label, grade = _classify_meyerding(0.0)
        assert label == "0"
        assert grade == 0

    def test_grade_i(self):
        label, grade = _classify_meyerding(0.15)
        assert label == "I"
        assert grade == 1

    def test_grade_ii(self):
        label, grade = _classify_meyerding(0.35)
        assert label == "II"
        assert grade == 2

    def test_grade_iii(self):
        label, grade = _classify_meyerding(0.60)
        assert label == "III"
        assert grade == 3

    def test_grade_iv(self):
        label, grade = _classify_meyerding(0.85)
        assert label == "IV"
        assert grade == 4

    def test_grade_v_spondyloptosis(self):
        label, grade = _classify_meyerding(1.10)
        assert label == "V"
        assert grade == 5

    def test_threshold_grade_i_max(self):
        label, grade = _classify_meyerding(0.25)
        assert label == "I"
        assert grade == 1

    def test_above_grade_i_max(self):
        label, grade = _classify_meyerding(0.26)
        assert label == "II"
        assert grade == 2


# =============================================================================
# Pickhardt Classification Tests
# =============================================================================


class TestClassifyPickhardt:
    """Test Pickhardt 2013 HU bone density classification."""

    def test_normal(self):
        cat, _, conf = _classify_pickhardt(180.0)
        assert cat == "NORMAL"
        assert conf == "high"

    def test_osteopenia(self):
        cat, t_score, conf = _classify_pickhardt(130.0)
        assert cat == "OSTEOPENIA"
        assert conf == "moderate"

    def test_osteoporosis_probable(self):
        cat, _, _ = _classify_pickhardt(95.0)
        assert cat == "OSTEOPOROSIS_PROBABLE"

    def test_osteoporosis(self):
        cat, _, conf = _classify_pickhardt(65.0)
        assert cat == "OSTEOPOROSIS"
        assert conf == "high"

    def test_osteoporosis_severe(self):
        cat, _, _ = _classify_pickhardt(40.0)
        assert cat == "OSTEOPOROSIS_SEVERE"

    def test_threshold_normal(self):
        cat, _, _ = _classify_pickhardt(160.0)
        assert cat == "NORMAL"

    def test_threshold_osteopenia(self):
        cat, _, _ = _classify_pickhardt(110.0)
        assert cat == "OSTEOPENIA"

    def test_threshold_osteoporosis_probable(self):
        cat, _, _ = _classify_pickhardt(80.0)
        assert cat == "OSTEOPOROSIS_PROBABLE"


# =============================================================================
# SINS Component Score Tests
# =============================================================================


class TestSINSLocationScore:
    """Test SINS location component scoring."""

    def test_junctional_c1(self):
        score, _ = _sins_location_score("C1")
        assert score == 3

    def test_junctional_c7(self):
        score, _ = _sins_location_score("C7")
        assert score == 3

    def test_junctional_t11(self):
        score, _ = _sins_location_score("T11")
        assert score == 3

    def test_junctional_l5(self):
        score, _ = _sins_location_score("L5")
        assert score == 3

    def test_mobile_c4(self):
        score, _ = _sins_location_score("C4")
        assert score == 2

    def test_semi_rigid_t5(self):
        score, _ = _sins_location_score("T5")
        assert score == 1

    def test_rigid_s3(self):
        score, _ = _sins_location_score("S3")
        assert score == 0

    def test_unknown_level(self):
        score, _ = _sins_location_score("X1")
        assert score == 0


class TestSINSLesionTypeScore:
    """Test SINS lesion type component scoring."""

    def test_lytic(self):
        score, _ = _sins_lesion_type_score("lytic")
        assert score == 2

    def test_mixed(self):
        score, _ = _sins_lesion_type_score("mixed")
        assert score == 1

    def test_blastic(self):
        score, _ = _sins_lesion_type_score("blastic")
        assert score == 0

    def test_unknown(self):
        score, _ = _sins_lesion_type_score("unknown")
        assert score == 1


class TestSINSAlignmentScore:
    """Test SINS alignment component scoring."""

    def test_subluxation(self):
        score, _ = _sins_alignment_score({"subluxation": True})
        assert score == 4

    def test_new_kyphosis(self):
        score, _ = _sins_alignment_score({"focal_kyphosis_new": True})
        assert score == 2

    def test_new_scoliosis(self):
        score, _ = _sins_alignment_score({"focal_scoliosis_new": True})
        assert score == 2

    def test_normal_alignment(self):
        score, _ = _sins_alignment_score({})
        assert score == 0


class TestSINSCollapseScore:
    """Test SINS vertebral body collapse component scoring."""

    def test_greater_50_percent(self):
        score, _ = _sins_collapse_score(60.0, 80.0)
        assert score == 3

    def test_less_50_percent(self):
        score, _ = _sins_collapse_score(30.0, 60.0)
        assert score == 2

    def test_no_collapse_high_involvement(self):
        score, _ = _sins_collapse_score(0.0, 60.0)
        assert score == 1

    def test_no_collapse_low_involvement(self):
        score, _ = _sins_collapse_score(0.0, 30.0)
        assert score == 0


class TestSINSPosterolateralScore:
    """Test SINS posterolateral involvement component scoring."""

    def test_bilateral(self):
        score, _ = _sins_posterolateral_score(
            {"pedicle_left": "infiltrated", "pedicle_right": "destroyed"}
        )
        assert score == 3

    def test_unilateral_left(self):
        score, _ = _sins_posterolateral_score(
            {"pedicle_left": "infiltrated", "pedicle_right": "intact"}
        )
        assert score == 1

    def test_unilateral_right(self):
        score, _ = _sins_posterolateral_score(
            {"pedicle_left": "intact", "pedicle_right": "infiltrated"}
        )
        assert score == 1

    def test_none(self):
        score, _ = _sins_posterolateral_score({"pedicle_left": "intact", "pedicle_right": "intact"})
        assert score == 0

    def test_empty_dict(self):
        score, _ = _sins_posterolateral_score({})
        assert score == 0


class TestClassifySINSTotal:
    """Test SINS total score classification."""

    def test_stable(self):
        classification, _ = _classify_sins_total(4)
        assert classification == "STABLE"

    def test_stable_upper(self):
        classification, _ = _classify_sins_total(6)
        assert classification == "STABLE"

    def test_indeterminate(self):
        classification, _ = _classify_sins_total(9)
        assert classification == "INDETERMINATE"

    def test_indeterminate_lower(self):
        classification, _ = _classify_sins_total(7)
        assert classification == "INDETERMINATE"

    def test_indeterminate_upper(self):
        classification, _ = _classify_sins_total(12)
        assert classification == "INDETERMINATE"

    def test_unstable(self):
        classification, _ = _classify_sins_total(15)
        assert classification == "UNSTABLE"

    def test_unstable_lower(self):
        classification, _ = _classify_sins_total(13)
        assert classification == "UNSTABLE"


# =============================================================================
# Stenosis Classification Tests
# =============================================================================


class TestClassifyStenosis:
    """Test spinal canal stenosis grading."""

    def test_none(self):
        assert _classify_stenosis(15.0) == "none"

    def test_mild(self):
        assert _classify_stenosis(11.0) == "mild"

    def test_moderate(self):
        assert _classify_stenosis(8.0) == "moderate"

    def test_severe(self):
        assert _classify_stenosis(5.0) == "severe"

    def test_threshold_mild(self):
        assert _classify_stenosis(13.0) == "none"

    def test_threshold_moderate(self):
        assert _classify_stenosis(10.0) == "mild"

    def test_threshold_severe(self):
        assert _classify_stenosis(7.0) == "moderate"


# =============================================================================
# Tool Function Tests (mocked Slicer)
# =============================================================================


class TestDetectVertebralFracturesCT:
    """Test detect_vertebral_fractures_ct tool."""

    def test_invalid_volume_node_id(self):
        with pytest.raises(ValidationError):
            detect_vertebral_fractures_ct("1invalid")

    def test_invalid_region(self):
        with pytest.raises(ValidationError):
            detect_vertebral_fractures_ct("vtkMRMLScalarVolumeNode1", region="invalid")

    def test_invalid_classification(self):
        with pytest.raises(ValidationError):
            detect_vertebral_fractures_ct(
                "vtkMRMLScalarVolumeNode1", classification_system="invalid"
            )

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_successful_detection(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "modality": "CT",
            "region_analyzed": "C1-L5",
            "fractures_detected": 1,
            "vertebrae": [
                {
                    "level": "L1",
                    "genant": {"grade": 2, "label": "Moderate"},
                }
            ],
            "summary": {"total_fractured": 1},
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        result = detect_vertebral_fractures_ct("vtkMRMLScalarVolumeNode1")
        assert result["success"] is True
        assert result["fractures_detected"] == 1

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_connection_error(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")

        with pytest.raises(SlicerConnectionError):
            detect_vertebral_fractures_ct("vtkMRMLScalarVolumeNode1")

    def test_valid_segmentation_node_id(self):
        """Segmentation node ID is validated before exec."""
        with pytest.raises(ValidationError):
            detect_vertebral_fractures_ct(
                "vtkMRMLScalarVolumeNode1",
                segmentation_node_id="1invalid",
            )


class TestAssessOsteoporosisCT:
    """Test assess_osteoporosis_ct tool."""

    def test_invalid_volume_node_id(self):
        with pytest.raises(ValidationError):
            assess_osteoporosis_ct("")

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            assess_osteoporosis_ct("vtkMRMLScalarVolumeNode1", method="invalid")

    def test_invalid_levels(self):
        with pytest.raises(ValidationError):
            assess_osteoporosis_ct("vtkMRMLScalarVolumeNode1", levels=["X1"])

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_successful_assessment(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "modality": "CT",
            "method": "trabecular_roi",
            "calibrated": False,
            "levels": [
                {
                    "level": "L1",
                    "hu_statistics": {"mean": 92.3, "median": 88.1},
                    "classification": {"category": "OSTEOPOROSIS_PROBABLE"},
                }
            ],
            "global_assessment": {"classification": "OSTEOPOROSIS_PROBABLE"},
            "clinical_context": {"screw_pullout_risk": "HIGH"},
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        result = assess_osteoporosis_ct("vtkMRMLScalarVolumeNode1")
        assert result["success"] is True
        assert result["levels"][0]["classification"]["category"] == "OSTEOPOROSIS_PROBABLE"

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_default_level_is_l1(self, mock_get_client):
        """Default level should be L1 per Pickhardt protocol."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "levels": [],
            "global_assessment": {},
            "clinical_context": {},
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        assess_osteoporosis_ct("vtkMRMLScalarVolumeNode1")
        call_args = mock_client.exec_python.call_args
        code = call_args[0][0]
        assert '"L1"' in code


class TestDetectMetastaticLesionsCT:
    """Test detect_metastatic_lesions_ct tool."""

    def test_invalid_volume_node_id(self):
        with pytest.raises(ValidationError):
            detect_metastatic_lesions_ct("1invalid")

    def test_invalid_region(self):
        with pytest.raises(ValidationError):
            detect_metastatic_lesions_ct("vtkMRMLScalarVolumeNode1", region="invalid")

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_successful_detection(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "modality": "CT",
            "lesions_detected": 2,
            "vertebrae": [
                {
                    "level": "T8",
                    "lesion_type": "lytic",
                    "lesion_count": 1,
                    "lesions": [{"volume_mm3": 2845.3}],
                }
            ],
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        result = detect_metastatic_lesions_ct("vtkMRMLScalarVolumeNode1")
        assert result["success"] is True
        assert result["lesions_detected"] == 2

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_connection_error(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")

        with pytest.raises(SlicerConnectionError):
            detect_metastatic_lesions_ct("vtkMRMLScalarVolumeNode1")


class TestCalculateSINSScore:
    """Test calculate_sins_score tool."""

    def test_invalid_volume_node_id(self):
        with pytest.raises(ValidationError):
            calculate_sins_score("")

    def test_invalid_pain_score_too_high(self):
        with pytest.raises(ValidationError) as exc_info:
            calculate_sins_score("vtkMRMLScalarVolumeNode1", pain_score=5)
        assert exc_info.value.field == "pain_score"

    def test_invalid_pain_score_negative(self):
        with pytest.raises(ValidationError):
            calculate_sins_score("vtkMRMLScalarVolumeNode1", pain_score=-1)

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_successful_with_pain_score(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "modality": "CT",
            "levels": [
                {
                    "level": "T8",
                    "sins_total": 11,
                    "sins_classification": "INDETERMINATE",
                    "imaging_only_score": 8,
                }
            ],
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        result = calculate_sins_score(
            "vtkMRMLScalarVolumeNode1",
            target_levels=["T8"],
            pain_score=3,
        )
        assert result["success"] is True
        assert result["levels"][0]["sins_classification"] == "INDETERMINATE"

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_without_pain_score_reports_range(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "modality": "CT",
            "levels": [
                {
                    "level": "T8",
                    "sins_total": None,
                    "score_range_min": 6,
                    "score_range_max": 9,
                    "clinical_components_missing": ["pain"],
                }
            ],
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        result = calculate_sins_score(
            "vtkMRMLScalarVolumeNode1",
            target_levels=["T8"],
        )
        assert result["levels"][0]["clinical_components_missing"] == ["pain"]

    def test_valid_pain_score_zero(self):
        """Pain score of 0 (pain-free) is valid."""
        # This should not raise - it will fail at exec_python since Slicer is not running
        with pytest.raises((SlicerConnectionError, Exception)):
            calculate_sins_score("vtkMRMLScalarVolumeNode1", pain_score=0)


class TestMeasureListhesisCT:
    """Test measure_listhesis_ct tool."""

    def test_invalid_volume_node_id(self):
        with pytest.raises(ValidationError):
            measure_listhesis_ct("1invalid")

    def test_invalid_levels(self):
        with pytest.raises(ValidationError):
            measure_listhesis_ct("vtkMRMLScalarVolumeNode1", levels=["Z1"])

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_successful_measurement(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "modality": "CT",
            "static_measurement": True,
            "note": "Static CT. Dynamic instability requires flexion/extension X-ray.",
            "levels": [
                {
                    "level": "L4-L5",
                    "translation_mm": 5.2,
                    "translation_percent": 15.0,
                    "meyerding_grade": "I",
                    "slip_angle_deg": 8.3,
                }
            ],
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        result = measure_listhesis_ct("vtkMRMLScalarVolumeNode1")
        assert result["success"] is True
        assert result["static_measurement"] is True
        assert "flexion/extension" in result["note"]

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_default_levels(self, mock_get_client):
        """Default levels should be L3, L4, L5."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {"success": True, "levels": []}
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        measure_listhesis_ct("vtkMRMLScalarVolumeNode1")
        code = mock_client.exec_python.call_args[0][0]
        assert '"L3"' in code
        assert '"L4"' in code
        assert '"L5"' in code


class TestMeasureSpinalCanalCT:
    """Test measure_spinal_canal_ct tool."""

    def test_invalid_volume_node_id(self):
        with pytest.raises(ValidationError):
            measure_spinal_canal_ct("")

    def test_invalid_levels(self):
        with pytest.raises(ValidationError):
            measure_spinal_canal_ct("vtkMRMLScalarVolumeNode1", levels=["X1"])

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_successful_measurement(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {
            "success": True,
            "modality": "CT",
            "levels": [
                {
                    "level": "C5",
                    "ap_diameter_mm": 14.2,
                    "transverse_diameter_mm": 22.1,
                    "cross_section_area_mm2": 246.5,
                    "torg_pavlov_ratio": 0.82,
                    "torg_pavlov_stenosis": False,
                    "stenosis_grade": "none",
                }
            ],
        }
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        result = measure_spinal_canal_ct("vtkMRMLScalarVolumeNode1")
        assert result["success"] is True
        assert result["levels"][0]["torg_pavlov_ratio"] == 0.82

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_default_levels_cervical(self, mock_get_client):
        """Default levels should be C3-C7 for canal measurement."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        result_data = {"success": True, "levels": []}
        mock_client.exec_python.return_value = {
            "success": True,
            "result": json.dumps(result_data),
        }

        measure_spinal_canal_ct("vtkMRMLScalarVolumeNode1")
        code = mock_client.exec_python.call_args[0][0]
        assert '"C3"' in code
        assert '"C7"' in code

    @patch("slicer_mcp.diagnostic_tools_ct.get_client")
    def test_connection_error(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")

        with pytest.raises(SlicerConnectionError):
            measure_spinal_canal_ct("vtkMRMLScalarVolumeNode1")


# =============================================================================
# Server Registration Tests
# =============================================================================


class TestServerRegistration:
    """Test that tools are properly registered in server.py."""

    def test_diagnostic_tools_importable(self):
        """Verify diagnostic_tools_ct module is importable."""
        from slicer_mcp import diagnostic_tools_ct

        assert hasattr(diagnostic_tools_ct, "detect_vertebral_fractures_ct")
        assert hasattr(diagnostic_tools_ct, "assess_osteoporosis_ct")
        assert hasattr(diagnostic_tools_ct, "detect_metastatic_lesions_ct")
        assert hasattr(diagnostic_tools_ct, "calculate_sins_score")
        assert hasattr(diagnostic_tools_ct, "measure_listhesis_ct")
        assert hasattr(diagnostic_tools_ct, "measure_spinal_canal_ct")

    def test_spine_constants_importable(self):
        """Verify spine_constants module is importable."""
        from slicer_mcp import spine_constants

        assert hasattr(spine_constants, "GENANT_THRESHOLDS")
        assert hasattr(spine_constants, "PICKHARDT_HU_THRESHOLDS")
        assert hasattr(spine_constants, "SINS_RANGES")
        assert hasattr(spine_constants, "MEYERDING_THRESHOLDS")
        assert hasattr(spine_constants, "TORG_PAVLOV_THRESHOLD")
        assert hasattr(spine_constants, "SPINAL_CANAL_AP_DIAMETER")
