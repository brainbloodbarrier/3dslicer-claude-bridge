"""Unit tests for X-ray diagnostic protocol tools."""

from unittest.mock import Mock, patch

import pytest

from slicer_mcp.diagnostic_tools_xray import (
    MAGNIFICATION_DISCLAIMER,
    _angle_between_lines_2d,
    _classify_genant,
    _classify_meyerding,
    _classify_roussouly,
    _classify_schwab_pi_ll,
    _classify_schwab_pt,
    _classify_schwab_sva,
    _cobb_angle_2d,
    _distance_2d,
    _horizontal_distance_2d,
    _is_dynamic_unstable,
    _validate_landmarks,
    _validate_view_type,
    detect_vertebral_fractures_xray,
    measure_cobb_angle_xray,
    measure_coronal_balance_xray,
    measure_listhesis_dynamic_xray,
    measure_sagittal_balance_xray,
)
from slicer_mcp.tools import ValidationError

# =============================================================================
# Geometry Helper Tests
# =============================================================================


class TestAngleBetweenLines2D:
    """Test _angle_between_lines_2d helper."""

    def test_parallel_lines(self):
        """Parallel lines have 0 degree angle."""
        angle = _angle_between_lines_2d((0, 0), (1, 0), (0, 1), (1, 1))
        assert abs(angle) < 1e-6

    def test_perpendicular_lines(self):
        """Perpendicular lines have 90 degree angle."""
        angle = _angle_between_lines_2d((0, 0), (1, 0), (0, 0), (0, 1))
        assert abs(angle - 90.0) < 1e-6

    def test_45_degree_angle(self):
        """45 degree angle between lines."""
        angle = _angle_between_lines_2d((0, 0), (1, 0), (0, 0), (1, 1))
        assert abs(angle - 45.0) < 1e-6

    def test_180_degree_antiparallel(self):
        """Anti-parallel lines have 180 degree angle."""
        angle = _angle_between_lines_2d((0, 0), (1, 0), (1, 0), (0, 0))
        assert abs(angle - 180.0) < 1e-6

    def test_zero_length_line_returns_zero(self):
        """Zero-length line returns 0 angle."""
        angle = _angle_between_lines_2d((0, 0), (0, 0), (0, 0), (1, 1))
        assert angle == 0.0


class TestDistance2D:
    """Test _distance_2d helper."""

    def test_same_point(self):
        """Distance between same point is 0."""
        assert _distance_2d((0, 0), (0, 0)) == 0.0

    def test_unit_distance(self):
        """Unit distance along axis."""
        assert abs(_distance_2d((0, 0), (1, 0)) - 1.0) < 1e-9

    def test_diagonal(self):
        """Diagonal distance (3-4-5 triangle)."""
        assert abs(_distance_2d((0, 0), (3, 4)) - 5.0) < 1e-9


class TestHorizontalDistance2D:
    """Test _horizontal_distance_2d helper."""

    def test_horizontal_only(self):
        """Horizontal distance ignores vertical component."""
        assert abs(_horizontal_distance_2d((0, 0), (5, 10)) - 5.0) < 1e-9

    def test_same_x(self):
        """Same X position has 0 horizontal distance."""
        assert _horizontal_distance_2d((3, 0), (3, 100)) == 0.0


class TestCobbAngle2D:
    """Test _cobb_angle_2d helper."""

    def test_parallel_endplates(self):
        """Parallel endplates give 0 degree Cobb angle."""
        angle = _cobb_angle_2d((0, 0), (10, 0), (0, 100), (10, 100))
        assert abs(angle) < 1e-6

    def test_converging_endplates(self):
        """Converging endplates give positive Cobb angle."""
        angle = _cobb_angle_2d((0, 0), (10, 1), (0, 100), (10, 99))
        assert angle > 0


# =============================================================================
# Classification Tests
# =============================================================================


class TestClassifySchwabPiLl:
    """Test SRS-Schwab PI-LL classification."""

    def test_matched(self):
        """PI-LL < 10 is matched."""
        assert _classify_schwab_pi_ll(5.0) == "matched"

    def test_matched_negative(self):
        """Negative PI-LL < 10 is also matched."""
        assert _classify_schwab_pi_ll(-8.0) == "matched"

    def test_moderate(self):
        """10 <= PI-LL < 20 is moderate."""
        assert _classify_schwab_pi_ll(15.0) == "moderate"

    def test_marked(self):
        """PI-LL >= 20 is marked."""
        assert _classify_schwab_pi_ll(25.0) == "marked"

    def test_boundary_10(self):
        """PI-LL exactly 10 is moderate."""
        assert _classify_schwab_pi_ll(10.0) == "moderate"

    def test_boundary_20(self):
        """PI-LL exactly 20 is marked."""
        assert _classify_schwab_pi_ll(20.0) == "marked"


class TestClassifySchwabSva:
    """Test SRS-Schwab SVA classification."""

    def test_grade_0(self):
        """SVA < 40 is grade 0."""
        assert _classify_schwab_sva(30.0) == "0"

    def test_grade_1(self):
        """40 <= SVA < 95 is grade +."""
        assert _classify_schwab_sva(60.0) == "+"

    def test_grade_2(self):
        """SVA >= 95 is grade ++."""
        assert _classify_schwab_sva(100.0) == "++"


class TestClassifySchwabPt:
    """Test SRS-Schwab PT classification."""

    def test_grade_0(self):
        """PT < 20 is grade 0."""
        assert _classify_schwab_pt(15.0) == "0"

    def test_grade_1(self):
        """20 <= PT < 30 is grade +."""
        assert _classify_schwab_pt(25.0) == "+"

    def test_grade_2(self):
        """PT >= 30 is grade ++."""
        assert _classify_schwab_pt(35.0) == "++"


class TestClassifyRoussoulyType:
    """Test Roussouly lordosis type classification."""

    def test_type_1(self):
        """Low SS with short lordosis at L5."""
        assert _classify_roussouly(30.0, 40.0, "L5") == "Type 1"

    def test_type_2(self):
        """Low SS with flat back."""
        assert _classify_roussouly(30.0, 50.0, "L3") == "Type 2"

    def test_type_3(self):
        """Medium SS."""
        assert _classify_roussouly(40.0, 55.0, "L4") == "Type 3"

    def test_type_4(self):
        """High SS."""
        assert _classify_roussouly(50.0, 70.0, "L4") == "Type 4"


class TestClassifyGenant:
    """Test Genant semi-quantitative fracture grading."""

    def test_normal(self):
        """<20% reduction is normal (grade 0)."""
        result = _classify_genant(0.15)
        assert result["grade"] == 0
        assert result["label"] == "normal"

    def test_mild(self):
        """20-25% reduction is mild (grade 1)."""
        result = _classify_genant(0.22)
        assert result["grade"] == 1
        assert result["label"] == "mild"

    def test_moderate(self):
        """25-40% reduction is moderate (grade 2)."""
        result = _classify_genant(0.30)
        assert result["grade"] == 2
        assert result["label"] == "moderate"

    def test_severe(self):
        """>40% reduction is severe (grade 3)."""
        result = _classify_genant(0.50)
        assert result["grade"] == 3
        assert result["label"] == "severe"

    def test_boundary_20(self):
        """Exactly 20% is mild."""
        result = _classify_genant(0.20)
        assert result["grade"] == 1

    def test_boundary_25(self):
        """Exactly 25% is moderate."""
        result = _classify_genant(0.25)
        assert result["grade"] == 2

    def test_boundary_40(self):
        """Exactly 40% is severe."""
        result = _classify_genant(0.40)
        assert result["grade"] == 3

    def test_no_reduction(self):
        """0% reduction is normal."""
        result = _classify_genant(0.0)
        assert result["grade"] == 0


class TestClassifyMeyerding:
    """Test Meyerding spondylolisthesis grading."""

    def test_grade_i(self):
        """0-25% slip is grade I."""
        result = _classify_meyerding(0.15)
        assert result["grade"] == 1

    def test_grade_ii(self):
        """25-50% slip is grade II."""
        result = _classify_meyerding(0.35)
        assert result["grade"] == 2

    def test_grade_iii(self):
        """50-75% slip is grade III."""
        result = _classify_meyerding(0.60)
        assert result["grade"] == 3

    def test_grade_iv(self):
        """75-100% slip is grade IV."""
        result = _classify_meyerding(0.85)
        assert result["grade"] == 4

    def test_spondyloptosis(self):
        """>100% slip is spondyloptosis (grade V)."""
        result = _classify_meyerding(1.2)
        assert result["grade"] == 5
        assert result["label"] == "spondyloptosis"


class TestIsDynamicUnstable:
    """Test White & Panjabi dynamic instability criteria."""

    def test_stable_lumbar(self):
        """Within thresholds is stable."""
        result = _is_dynamic_unstable(3.0, 10.0, "lumbar")
        assert result["unstable"] is False

    def test_unstable_lumbar_translation(self):
        """Lumbar translation > 4.5mm is unstable."""
        result = _is_dynamic_unstable(5.0, 10.0, "lumbar")
        assert result["unstable"] is True
        assert result["translation_exceeds"] is True

    def test_unstable_lumbar_angulation(self):
        """Lumbar angulation > 15 deg is unstable."""
        result = _is_dynamic_unstable(3.0, 20.0, "lumbar")
        assert result["unstable"] is True
        assert result["angulation_exceeds"] is True

    def test_unstable_cervical_translation(self):
        """Cervical translation > 3.5mm is unstable."""
        result = _is_dynamic_unstable(4.0, 5.0, "cervical")
        assert result["unstable"] is True

    def test_unstable_cervical_angulation(self):
        """Cervical angulation > 11 deg is unstable."""
        result = _is_dynamic_unstable(2.0, 12.0, "cervical")
        assert result["unstable"] is True

    def test_criteria_reference(self):
        """Result includes criteria reference."""
        result = _is_dynamic_unstable(1.0, 1.0, "lumbar")
        assert result["criteria"] == "White & Panjabi 1990"


# =============================================================================
# Landmark Validation Tests
# =============================================================================


class TestValidateLandmarks:
    """Test _validate_landmarks helper."""

    def test_valid_landmarks(self):
        """Valid landmarks pass validation."""
        landmarks = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
        result = _validate_landmarks(landmarks, ["A", "B"], "test_tool")
        assert result["A"] == (1.0, 2.0)
        assert result["B"] == (3.0, 4.0)

    def test_empty_landmarks_raises(self):
        """Empty landmarks dict raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            _validate_landmarks({}, ["A"], "test_tool")

    def test_missing_landmark_raises(self):
        """Missing required landmark raises ValidationError."""
        with pytest.raises(ValidationError, match="missing required landmarks"):
            _validate_landmarks({"A": [1.0, 2.0]}, ["A", "B"], "test_tool")

    def test_invalid_coordinate_format(self):
        """Non-list coordinate raises ValidationError."""
        with pytest.raises(ValidationError, match="must be"):
            _validate_landmarks({"A": [1.0]}, ["A"], "test_tool")

    def test_non_numeric_coordinate(self):
        """Non-numeric coordinate raises ValidationError."""
        with pytest.raises(ValidationError, match="must be numeric"):
            _validate_landmarks({"A": ["x", "y"]}, ["A"], "test_tool")

    def test_too_many_landmarks(self):
        """Exceeding MAX_LANDMARKS raises ValidationError."""
        big = {f"L{i}": [float(i), float(i)] for i in range(101)}
        with pytest.raises(ValidationError, match="too many landmarks"):
            _validate_landmarks(big, list(big.keys()), "test_tool")

    def test_tuple_coordinates_accepted(self):
        """Tuple coordinates are accepted."""
        landmarks = {"A": (1.0, 2.0)}
        result = _validate_landmarks(landmarks, ["A"], "test_tool")
        assert result["A"] == (1.0, 2.0)

    def test_integer_coordinates_converted(self):
        """Integer coordinates are converted to float."""
        landmarks = {"A": [1, 2]}
        result = _validate_landmarks(landmarks, ["A"], "test_tool")
        assert result["A"] == (1.0, 2.0)


class TestValidateViewType:
    """Test _validate_view_type helper."""

    def test_valid_lateral(self):
        """Lateral view type passes validation."""
        _validate_view_type("lateral", "lateral", "test_tool")

    def test_valid_ap(self):
        """AP view type passes validation."""
        _validate_view_type("ap", "ap", "test_tool")

    def test_invalid_view(self):
        """Invalid view type raises ValidationError."""
        with pytest.raises(ValidationError, match="invalid view_type"):
            _validate_view_type("oblique", "lateral", "test_tool")

    def test_wrong_view(self):
        """Correct view type but wrong expected raises ValidationError."""
        with pytest.raises(ValidationError, match="requires 'lateral'"):
            _validate_view_type("ap", "lateral", "test_tool")


# =============================================================================
# Helper: Mock fixtures for Slicer exec
# =============================================================================


def _mock_exec_success(code):
    """Return a successful Slicer exec result with landmark placement JSON."""
    return {
        "success": True,
        "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
        ' "markups_node_name": "Test", "num_landmarks": 5}',
    }


def _make_sagittal_landmarks():
    """Create a valid set of sagittal balance landmarks for testing."""
    return {
        "C2_centroid": [100.0, 50.0],
        "C7_centroid": [105.0, 150.0],
        "C2_sup_ant": [95.0, 45.0],
        "C2_sup_post": [105.0, 45.0],
        "C7_inf_ant": [100.0, 155.0],
        "C7_inf_post": [110.0, 155.0],
        "T1_sup_ant": [102.0, 160.0],
        "T1_sup_post": [112.0, 162.0],
        "T4_sup_ant": [100.0, 220.0],
        "T4_sup_post": [112.0, 222.0],
        "T12_inf_ant": [98.0, 380.0],
        "T12_inf_post": [112.0, 378.0],
        "L1_sup_ant": [97.0, 385.0],
        "L1_sup_post": [113.0, 383.0],
        "S1_sup_ant": [90.0, 500.0],
        "S1_sup_post": [120.0, 502.0],
        "S1_endplate_mid": [105.0, 501.0],
        "femoral_head_center_L": [80.0, 520.0],
        "femoral_head_center_R": [130.0, 520.0],
        "S1_post_sup": [120.0, 498.0],
    }


def _make_coronal_landmarks():
    """Create a valid set of coronal balance landmarks for testing."""
    return {
        "C7_centroid": [250.0, 100.0],
        "sacrum_center": [252.0, 500.0],
        "T1_centroid": [248.0, 120.0],
        "shoulder_L": [150.0, 80.0],
        "shoulder_R": [350.0, 82.0],
        "iliac_crest_L": [180.0, 450.0],
        "iliac_crest_R": [320.0, 452.0],
        "upper_end_vertebra_L": [230.0, 200.0],
        "upper_end_vertebra_R": [270.0, 205.0],
        "lower_end_vertebra_L": [235.0, 400.0],
        "lower_end_vertebra_R": [265.0, 395.0],
    }


def _make_cobb_landmarks():
    """Create a valid set of Cobb angle landmarks for testing."""
    return {
        "upper_end_sup_L": [230.0, 200.0],
        "upper_end_sup_R": [270.0, 205.0],
        "lower_end_inf_L": [235.0, 400.0],
        "lower_end_inf_R": [265.0, 395.0],
        "apex_centroid": [240.0, 300.0],
    }


def _make_fracture_landmarks():
    """Create fracture landmarks for two vertebrae."""
    return {
        "T12": {
            "ant_sup": [90.0, 300.0],
            "ant_inf": [90.0, 320.0],
            "mid_sup": [100.0, 300.0],
            "mid_inf": [100.0, 320.0],
            "post_sup": [110.0, 300.0],
            "post_inf": [110.0, 320.0],
        },
        "L1": {
            "ant_sup": [90.0, 325.0],
            "ant_inf": [90.0, 345.0],
            "mid_sup": [100.0, 325.0],
            "mid_inf": [100.0, 345.0],
            "post_sup": [110.0, 325.0],
            "post_inf": [110.0, 345.0],
        },
    }


def _make_listhesis_level_landmarks():
    """Create landmarks for a single level in one position."""
    return {
        "sup_post_inf": [110.0, 400.0],
        "inf_post_sup": [110.0, 405.0],
        "sup_ant_inf": [90.0, 400.0],
        "inf_ant_sup": [90.0, 405.0],
        "sup_inf_endplate_ant": [90.0, 400.0],
        "sup_inf_endplate_post": [110.0, 400.0],
        "inf_sup_endplate_ant": [90.0, 405.0],
        "inf_sup_endplate_post": [110.0, 405.0],
    }


# =============================================================================
# Tool 1: Sagittal Balance Tests
# =============================================================================


class TestMeasureSagittalBalanceXray:
    """Test measure_sagittal_balance_xray tool."""

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_successful_measurement(self, mock_get_client):
        """Full sagittal balance measurement returns expected structure."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "SagittalBalance_Landmarks", "num_landmarks": 20}',
        }
        mock_get_client.return_value = mock_client

        result = measure_sagittal_balance_xray(
            "vtkMRMLScalarVolumeNode1",
            _make_sagittal_landmarks(),
        )

        assert result["success"] is True
        assert result["tool"] == "measure_sagittal_balance_xray"
        assert "parameters" in result
        params = result["parameters"]
        assert "SVA_mm" in params
        assert "C2_C7_SVA_mm" in params
        assert "T1_slope_deg" in params
        assert "TPA_deg" in params
        assert "cervical_lordosis_deg" in params
        assert "thoracic_kyphosis_deg" in params
        assert "lumbar_lordosis_deg" in params
        assert "pelvic_incidence_deg" in params
        assert "pelvic_tilt_deg" in params
        assert "sacral_slope_deg" in params
        assert "PI_LL_mismatch_deg" in params
        assert "classifications" in result
        assert "SRS_Schwab" in result["classifications"]
        assert "Roussouly_type" in result["classifications"]
        assert result["disclaimer"] == MAGNIFICATION_DISCLAIMER

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_pi_equals_pt_plus_ss(self, mock_get_client):
        """PI = PT + SS relationship holds."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "SagittalBalance_Landmarks", "num_landmarks": 20}',
        }
        mock_get_client.return_value = mock_client

        result = measure_sagittal_balance_xray(
            "vtkMRMLScalarVolumeNode1",
            _make_sagittal_landmarks(),
        )

        params = result["parameters"]
        pi = params["pelvic_incidence_deg"]
        pt = params["pelvic_tilt_deg"]
        ss = params["sacral_slope_deg"]
        assert abs(pi - (pt + ss)) < 0.01

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_magnification_factor_applied(self, mock_get_client):
        """Magnification factor scales distance measurements."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "SagittalBalance_Landmarks", "num_landmarks": 20}',
        }
        mock_get_client.return_value = mock_client

        landmarks = _make_sagittal_landmarks()

        result_1x = measure_sagittal_balance_xray(
            "vtkMRMLScalarVolumeNode1", landmarks, magnification_factor=1.0
        )
        result_2x = measure_sagittal_balance_xray(
            "vtkMRMLScalarVolumeNode1", landmarks, magnification_factor=2.0
        )

        # SVA should be halved with 2x magnification
        sva_1x = result_1x["parameters"]["SVA_mm"]
        sva_2x = result_2x["parameters"]["SVA_mm"]
        assert abs(sva_2x - sva_1x / 2.0) < 0.01

    def test_invalid_node_id_raises(self):
        """Invalid node ID raises ValidationError."""
        with pytest.raises(ValidationError):
            measure_sagittal_balance_xray("1invalid", _make_sagittal_landmarks())

    def test_missing_landmark_raises(self):
        """Missing landmark raises ValidationError."""
        landmarks = _make_sagittal_landmarks()
        del landmarks["C2_centroid"]
        with pytest.raises(ValidationError, match="missing required"):
            measure_sagittal_balance_xray("vtkMRMLScalarVolumeNode1", landmarks)

    def test_negative_magnification_raises(self):
        """Negative magnification factor raises ValidationError."""
        with pytest.raises(ValidationError, match="magnification_factor must be positive"):
            measure_sagittal_balance_xray(
                "vtkMRMLScalarVolumeNode1",
                _make_sagittal_landmarks(),
                magnification_factor=-1.0,
            )

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_schwab_classification_present(self, mock_get_client):
        """Result includes SRS-Schwab classification."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "SagittalBalance_Landmarks", "num_landmarks": 20}',
        }
        mock_get_client.return_value = mock_client

        result = measure_sagittal_balance_xray(
            "vtkMRMLScalarVolumeNode1", _make_sagittal_landmarks()
        )

        schwab = result["classifications"]["SRS_Schwab"]
        assert "PI_LL" in schwab
        assert "SVA" in schwab
        assert "PT" in schwab
        assert schwab["PI_LL"] in ("matched", "moderate", "marked")
        assert schwab["SVA"] in ("0", "+", "++")
        assert schwab["PT"] in ("0", "+", "++")


# =============================================================================
# Tool 2: Coronal Balance Tests
# =============================================================================


class TestMeasureCoronalBalanceXray:
    """Test measure_coronal_balance_xray tool."""

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_successful_measurement(self, mock_get_client):
        """Full coronal balance measurement returns expected structure."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CoronalBalance_Landmarks", "num_landmarks": 11}',
        }
        mock_get_client.return_value = mock_client

        result = measure_coronal_balance_xray(
            "vtkMRMLScalarVolumeNode1",
            _make_coronal_landmarks(),
        )

        assert result["success"] is True
        assert result["tool"] == "measure_coronal_balance_xray"
        params = result["parameters"]
        assert "C7_CSVL_offset_mm" in params
        assert "trunk_shift_mm" in params
        assert "pelvic_obliquity_deg" in params
        assert "shoulder_balance_mm" in params
        assert "coronal_cobb_angle_deg" in params
        assert "interpretation" in result

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_balanced_coronal(self, mock_get_client):
        """Well-aligned spine detected as balanced."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CoronalBalance_Landmarks", "num_landmarks": 11}',
        }
        mock_get_client.return_value = mock_client

        # C7 nearly aligned with sacrum
        landmarks = _make_coronal_landmarks()
        landmarks["C7_centroid"] = [252.0, 100.0]

        result = measure_coronal_balance_xray("vtkMRMLScalarVolumeNode1", landmarks)

        assert result["interpretation"]["C7_CSVL"] == "balanced"

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_imbalanced_coronal(self, mock_get_client):
        """Large C7-CSVL offset detected as imbalanced."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CoronalBalance_Landmarks", "num_landmarks": 11}',
        }
        mock_get_client.return_value = mock_client

        landmarks = _make_coronal_landmarks()
        landmarks["C7_centroid"] = [300.0, 100.0]  # far from sacrum

        result = measure_coronal_balance_xray("vtkMRMLScalarVolumeNode1", landmarks)

        assert result["interpretation"]["C7_CSVL"] == "imbalanced"

    def test_missing_landmark_raises(self):
        """Missing landmark raises ValidationError."""
        landmarks = _make_coronal_landmarks()
        del landmarks["sacrum_center"]
        with pytest.raises(ValidationError, match="missing required"):
            measure_coronal_balance_xray("vtkMRMLScalarVolumeNode1", landmarks)


# =============================================================================
# Tool 3: Dynamic Listhesis Tests
# =============================================================================


class TestMeasureListhesisDynamicXray:
    """Test measure_listhesis_dynamic_xray tool."""

    def _make_listhesis_inputs(self):
        """Create a full set of valid listhesis inputs."""
        volume_ids = {
            "neutral": "vtkMRMLScalarVolumeNode1",
            "flexion": "vtkMRMLScalarVolumeNode2",
            "extension": "vtkMRMLScalarVolumeNode3",
        }
        landmarks = {
            "neutral": {"L4-L5": _make_listhesis_level_landmarks()},
            "flexion": {"L4-L5": _make_listhesis_level_landmarks()},
            "extension": {"L4-L5": _make_listhesis_level_landmarks()},
        }
        return volume_ids, landmarks

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_successful_measurement(self, mock_get_client):
        """Full listhesis measurement returns expected structure."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "Listhesis_neutral_Landmarks", "num_landmarks": 8}',
        }
        mock_get_client.return_value = mock_client

        vol_ids, landmarks = self._make_listhesis_inputs()
        result = measure_listhesis_dynamic_xray(vol_ids, landmarks, ["L4-L5"])

        assert result["success"] is True
        assert result["tool"] == "measure_listhesis_dynamic_xray"
        assert len(result["levels"]) == 1
        level = result["levels"][0]
        assert level["level"] == "L4-L5"
        assert "neutral" in level["positions"]
        assert "flexion" in level["positions"]
        assert "extension" in level["positions"]
        assert "dynamic_instability" in level
        assert "worst_meyerding" in result

    def test_missing_position_raises(self):
        """Missing volume_node_id for a position raises ValidationError."""
        vol_ids = {
            "neutral": "vtkMRMLScalarVolumeNode1",
            "flexion": "vtkMRMLScalarVolumeNode2",
            # missing "extension"
        }
        with pytest.raises(ValidationError, match="missing volume_node_id"):
            measure_listhesis_dynamic_xray(vol_ids, {}, ["L4-L5"])

    def test_invalid_region_raises(self):
        """Invalid region raises ValidationError."""
        vol_ids, landmarks = self._make_listhesis_inputs()
        with pytest.raises(ValidationError, match="region must be"):
            measure_listhesis_dynamic_xray(vol_ids, landmarks, ["L4-L5"], region="thoracic")

    def test_empty_levels_raises(self):
        """Empty levels list raises ValidationError."""
        vol_ids, landmarks = self._make_listhesis_inputs()
        with pytest.raises(ValidationError, match="levels list cannot be empty"):
            measure_listhesis_dynamic_xray(vol_ids, landmarks, [])

    def test_missing_landmarks_for_position_raises(self):
        """Missing landmarks for a position raises ValidationError."""
        vol_ids = {
            "neutral": "vtkMRMLScalarVolumeNode1",
            "flexion": "vtkMRMLScalarVolumeNode2",
            "extension": "vtkMRMLScalarVolumeNode3",
        }
        landmarks = {
            "neutral": {"L4-L5": _make_listhesis_level_landmarks()},
            # missing "flexion" and "extension"
        }
        with pytest.raises(ValidationError, match="missing landmarks"):
            measure_listhesis_dynamic_xray(vol_ids, landmarks, ["L4-L5"])

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_instability_pattern_reported(self, mock_get_client):
        """Instability pattern is included in result."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "Listhesis_neutral_Landmarks", "num_landmarks": 8}',
        }
        mock_get_client.return_value = mock_client

        vol_ids, landmarks = self._make_listhesis_inputs()
        result = measure_listhesis_dynamic_xray(vol_ids, landmarks, ["L4-L5"])

        assert result["instability_pattern"] in ("stable", "unstable")


# =============================================================================
# Tool 4: Vertebral Fracture Detection Tests
# =============================================================================


class TestDetectVertebralFracturesXray:
    """Test detect_vertebral_fractures_xray tool."""

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_successful_detection(self, mock_get_client):
        """Full fracture detection returns expected structure."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "VertebralFractures_Landmarks", "num_landmarks": 12}',
        }
        mock_get_client.return_value = mock_client

        result = detect_vertebral_fractures_xray(
            "vtkMRMLScalarVolumeNode1",
            _make_fracture_landmarks(),
        )

        assert result["success"] is True
        assert result["tool"] == "detect_vertebral_fractures_xray"
        assert len(result["vertebrae"]) == 2
        assert "summary" in result
        assert "fracture_count" in result["summary"]

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_normal_vertebrae_no_fractures(self, mock_get_client):
        """Equal-height vertebrae produce grade 0 (no fracture)."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "VertebralFractures_Landmarks", "num_landmarks": 12}',
        }
        mock_get_client.return_value = mock_client

        result = detect_vertebral_fractures_xray(
            "vtkMRMLScalarVolumeNode1",
            _make_fracture_landmarks(),
        )

        # With our test landmarks, heights are uniform -> grade 0
        for v in result["vertebrae"]:
            assert v["genant_grade"] == 0

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_wedge_fracture_detected(self, mock_get_client):
        """Reduced anterior height produces wedge fracture grade."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "VertebralFractures_Landmarks", "num_landmarks": 12}',
        }
        mock_get_client.return_value = mock_client

        landmarks = _make_fracture_landmarks()
        # Collapse anterior height of L1 to simulate wedge fracture
        landmarks["L1"]["ant_inf"] = [90.0, 332.0]  # Reduced from 340 -> height 7 vs 20

        result = detect_vertebral_fractures_xray("vtkMRMLScalarVolumeNode1", landmarks)

        l1_result = [v for v in result["vertebrae"] if v["vertebra"] == "L1"][0]
        assert l1_result["genant_grade"] >= 1
        assert l1_result["morphology"] == "wedge"
        assert result["summary"]["fracture_count"] >= 1

    def test_empty_landmarks_raises(self):
        """Empty landmarks dict raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            detect_vertebral_fractures_xray("vtkMRMLScalarVolumeNode1", {})

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_morphology_types(self, mock_get_client):
        """Different reduction patterns produce correct morphology labels."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "VertebralFractures_Landmarks", "num_landmarks": 6}',
        }
        mock_get_client.return_value = mock_client

        # Biconcave: middle height reduced
        landmarks = {
            "T12": {
                "ant_sup": [90.0, 300.0],
                "ant_inf": [90.0, 320.0],
                "mid_sup": [100.0, 300.0],
                "mid_inf": [100.0, 310.0],  # Collapsed middle
                "post_sup": [110.0, 300.0],
                "post_inf": [110.0, 320.0],
            },
        }

        result = detect_vertebral_fractures_xray("vtkMRMLScalarVolumeNode1", landmarks)

        vert = result["vertebrae"][0]
        assert vert["morphology"] == "biconcave"

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_confidence_field_present(self, mock_get_client):
        """Each vertebra result includes confidence score."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "VertebralFractures_Landmarks", "num_landmarks": 12}',
        }
        mock_get_client.return_value = mock_client

        result = detect_vertebral_fractures_xray(
            "vtkMRMLScalarVolumeNode1",
            _make_fracture_landmarks(),
        )

        for v in result["vertebrae"]:
            assert "confidence" in v
            assert 0.0 <= v["confidence"] <= 1.0


# =============================================================================
# Tool 5: Cobb Angle Tests
# =============================================================================


class TestMeasureCobbAngleXray:
    """Test measure_cobb_angle_xray tool."""

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_successful_measurement(self, mock_get_client):
        """Full Cobb angle measurement returns expected structure."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CobbAngle_Landmarks", "num_landmarks": 5}',
        }
        mock_get_client.return_value = mock_client

        result = measure_cobb_angle_xray(
            "vtkMRMLScalarVolumeNode1",
            _make_cobb_landmarks(),
            upper_end_vertebra="T6",
            lower_end_vertebra="L1",
        )

        assert result["success"] is True
        assert result["tool"] == "measure_cobb_angle_xray"
        assert "cobb_angle_deg" in result
        assert "curve_direction" in result
        assert "severity" in result
        assert result["curve_direction"] in ("left", "right")
        assert result["endvertebrae"]["upper"] == "T6"
        assert result["endvertebrae"]["lower"] == "L1"

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_parallel_endplates_zero_angle(self, mock_get_client):
        """Parallel endplates produce near-zero Cobb angle."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CobbAngle_Landmarks", "num_landmarks": 5}',
        }
        mock_get_client.return_value = mock_client

        landmarks = {
            "upper_end_sup_L": [200.0, 200.0],
            "upper_end_sup_R": [300.0, 200.0],
            "lower_end_inf_L": [200.0, 400.0],
            "lower_end_inf_R": [300.0, 400.0],
            "apex_centroid": [250.0, 300.0],
        }

        result = measure_cobb_angle_xray("vtkMRMLScalarVolumeNode1", landmarks)

        assert result["cobb_angle_deg"] < 1.0
        assert result["severity"] == "within normal limits"

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_severity_classification(self, mock_get_client):
        """Different angles produce correct severity classification."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CobbAngle_Landmarks", "num_landmarks": 5}',
        }
        mock_get_client.return_value = mock_client

        # Create landmarks that produce a moderate angle (~30°)
        landmarks = {
            "upper_end_sup_L": [200.0, 200.0],
            "upper_end_sup_R": [300.0, 230.0],  # Tilted endplate
            "lower_end_inf_L": [200.0, 400.0],
            "lower_end_inf_R": [300.0, 370.0],  # Tilted opposite way
            "apex_centroid": [240.0, 300.0],
        }

        result = measure_cobb_angle_xray("vtkMRMLScalarVolumeNode1", landmarks)

        assert result["cobb_angle_deg"] > 10.0

    def test_invalid_curve_type_raises(self):
        """Invalid curve_type raises ValidationError."""
        with pytest.raises(ValidationError, match="invalid curve_type"):
            measure_cobb_angle_xray(
                "vtkMRMLScalarVolumeNode1",
                _make_cobb_landmarks(),
                curve_type="invalid",
            )

    def test_missing_landmark_raises(self):
        """Missing landmark raises ValidationError."""
        landmarks = _make_cobb_landmarks()
        del landmarks["apex_centroid"]
        with pytest.raises(ValidationError, match="missing required"):
            measure_cobb_angle_xray("vtkMRMLScalarVolumeNode1", landmarks)

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_curve_direction_left(self, mock_get_client):
        """Apex left of midline is left curve."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CobbAngle_Landmarks", "num_landmarks": 5}',
        }
        mock_get_client.return_value = mock_client

        landmarks = _make_cobb_landmarks()
        landmarks["apex_centroid"] = [200.0, 300.0]  # Left of midline

        result = measure_cobb_angle_xray("vtkMRMLScalarVolumeNode1", landmarks)

        assert result["curve_direction"] == "left"

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_curve_direction_right(self, mock_get_client):
        """Apex right of midline is right curve."""
        mock_client = Mock()
        mock_client.exec_python.return_value = {
            "success": True,
            "result": '{"success": true, "markups_node_id": "vtkMRMLMarkupsFiducialNode1",'
            ' "markups_node_name": "CobbAngle_Landmarks", "num_landmarks": 5}',
        }
        mock_get_client.return_value = mock_client

        landmarks = _make_cobb_landmarks()
        landmarks["apex_centroid"] = [300.0, 300.0]  # Right of midline

        result = measure_cobb_angle_xray("vtkMRMLScalarVolumeNode1", landmarks)

        assert result["curve_direction"] == "right"


# =============================================================================
# Slicer Connection Error Tests
# =============================================================================


class TestSlicerConnectionErrors:
    """Test error handling when Slicer connection fails."""

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_sagittal_balance_connection_error(self, mock_get_client):
        """Sagittal balance raises on Slicer connection error."""
        from slicer_mcp.slicer_client import SlicerConnectionError

        mock_client = Mock()
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")
        mock_get_client.return_value = mock_client

        with pytest.raises(SlicerConnectionError):
            measure_sagittal_balance_xray("vtkMRMLScalarVolumeNode1", _make_sagittal_landmarks())

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_coronal_balance_connection_error(self, mock_get_client):
        """Coronal balance raises on Slicer connection error."""
        from slicer_mcp.slicer_client import SlicerConnectionError

        mock_client = Mock()
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")
        mock_get_client.return_value = mock_client

        with pytest.raises(SlicerConnectionError):
            measure_coronal_balance_xray("vtkMRMLScalarVolumeNode1", _make_coronal_landmarks())

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_cobb_angle_connection_error(self, mock_get_client):
        """Cobb angle raises on Slicer connection error."""
        from slicer_mcp.slicer_client import SlicerConnectionError

        mock_client = Mock()
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")
        mock_get_client.return_value = mock_client

        with pytest.raises(SlicerConnectionError):
            measure_cobb_angle_xray("vtkMRMLScalarVolumeNode1", _make_cobb_landmarks())

    @patch("slicer_mcp.diagnostic_tools_xray.get_client")
    def test_fractures_connection_error(self, mock_get_client):
        """Fracture detection raises on Slicer connection error."""
        from slicer_mcp.slicer_client import SlicerConnectionError

        mock_client = Mock()
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection failed")
        mock_get_client.return_value = mock_client

        with pytest.raises(SlicerConnectionError):
            detect_vertebral_fractures_xray("vtkMRMLScalarVolumeNode1", _make_fracture_landmarks())
