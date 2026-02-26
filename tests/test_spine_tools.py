"""Unit tests for spine-specific MCP tool implementations."""

import json
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.spine_tools import (
    VALID_ALIGNMENT_REGIONS,
    VALID_ARTERY_SIDES,
    VALID_BONE_REGIONS,
    VALID_POPULATIONS,
    _build_ccj_angles_code,
    _build_ccj_landmark_extraction_code,
    _build_sagittal_alignment_code,
    _build_spine_segmentation_code,
    _build_vertebral_centroid_extraction_code,
    _validate_seed_points,
    analyze_bone_quality,
    measure_ccj_angles,
    measure_spine_alignment,
    segment_spine,
    segment_vertebral_artery,
)
from slicer_mcp.tools import ValidationError

# =============================================================================
# Input Validation Tests -- measure_ccj_angles
# =============================================================================


class TestMeasureCCJAnglesValidation:
    """Test measure_ccj_angles input validation."""

    def test_rejects_empty_node_id(self):
        """Test empty node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            measure_ccj_angles("")
        assert exc_info.value.field == "node_id"

    def test_rejects_invalid_node_id_injection(self):
        """Test code injection in node ID is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            measure_ccj_angles("node'; import os; os.system('rm -rf /');")
        assert exc_info.value.field == "node_id"

    def test_rejects_invalid_population(self):
        """Test invalid population value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            measure_ccj_angles("vtkMRMLSegmentationNode1", population="elderly")
        assert exc_info.value.field == "population"
        assert "elderly" in str(exc_info.value)

    def test_accepts_adult_population(self):
        """Test adult population is accepted and proceeds to Slicer call."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": True,
                        "landmarks": {},
                        "measurements": {"ADI_mm": 2.5},
                        "reference_ranges": {},
                        "statuses": {},
                        "segments_found": {"C1": True, "C2": True, "skull_base": False},
                        "coordinate_system": "RAS",
                        "node_id": "vtkMRMLSegmentationNode1",
                        "node_name": "Spine",
                        "population": "adult",
                    }
                ),
            }
            mock_get_client.return_value = mock_client

            result = measure_ccj_angles("vtkMRMLSegmentationNode1", population="adult")

            assert result["success"] is True
            assert result["population"] == "adult"
            mock_client.exec_python.assert_called_once()

    def test_accepts_child_population(self):
        """Test child population is accepted and proceeds to Slicer call."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": True,
                        "landmarks": {},
                        "measurements": {"ADI_mm": 4.0},
                        "reference_ranges": {},
                        "statuses": {},
                        "segments_found": {"C1": True, "C2": True, "skull_base": False},
                        "coordinate_system": "RAS",
                        "node_id": "vtkMRMLSegmentationNode1",
                        "node_name": "Spine",
                        "population": "child",
                    }
                ),
            }
            mock_get_client.return_value = mock_client

            result = measure_ccj_angles("vtkMRMLSegmentationNode1", population="child")

            assert result["success"] is True
            assert result["population"] == "child"

    def test_defaults_to_adult_population(self):
        """Test population defaults to adult when not specified."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": True,
                        "landmarks": {},
                        "measurements": {},
                        "reference_ranges": {},
                        "statuses": {},
                        "segments_found": {"C1": False, "C2": False, "skull_base": False},
                        "coordinate_system": "RAS",
                        "node_id": "vtkMRMLSegmentationNode1",
                        "node_name": "Spine",
                        "population": "adult",
                    }
                ),
            }
            mock_get_client.return_value = mock_client

            result = measure_ccj_angles("vtkMRMLSegmentationNode1")

            assert result["population"] == "adult"


# =============================================================================
# Input Validation Tests -- measure_spine_alignment
# =============================================================================


class TestMeasureSpineAlignmentValidation:
    """Test measure_spine_alignment input validation."""

    def test_rejects_empty_node_id(self):
        """Test empty node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            measure_spine_alignment("")
        assert exc_info.value.field == "node_id"

    def test_rejects_invalid_node_id_injection(self):
        """Test code injection in node ID is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            measure_spine_alignment("'); DROP TABLE; ('")
        assert exc_info.value.field == "node_id"

    def test_rejects_invalid_region(self):
        """Test invalid region value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            measure_spine_alignment("vtkMRMLSegmentationNode1", region="sacral")
        assert exc_info.value.field == "region"
        assert "sacral" in str(exc_info.value)

    def test_accepts_cervical_region(self):
        """Test cervical region is accepted."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": True,
                        "node_id": "vtkMRMLSegmentationNode1",
                        "node_name": "Spine",
                        "region": "cervical",
                        "coordinate_system": "RAS",
                        "vertebrae_found": ["C2", "C7"],
                        "vertebrae_data": {},
                        "measurements": {"cervical_lordosis_deg": 25.0},
                        "reference_ranges": {},
                        "statuses": {},
                    }
                ),
            }
            mock_get_client.return_value = mock_client

            result = measure_spine_alignment("vtkMRMLSegmentationNode1", region="cervical")

            assert result["region"] == "cervical"
            mock_client.exec_python.assert_called_once()

    def test_accepts_all_valid_regions(self):
        """Test all valid regions are accepted."""
        for region in VALID_ALIGNMENT_REGIONS:
            with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
                mock_client = Mock()
                mock_client.exec_python.return_value = {
                    "success": True,
                    "result": json.dumps(
                        {
                            "success": True,
                            "node_id": "vtkMRMLSegmentationNode1",
                            "node_name": "Spine",
                            "region": region,
                            "coordinate_system": "RAS",
                            "vertebrae_found": [],
                            "vertebrae_data": {},
                            "measurements": {},
                            "reference_ranges": {},
                            "statuses": {},
                        }
                    ),
                }
                mock_get_client.return_value = mock_client

                result = measure_spine_alignment("vtkMRMLSegmentationNode1", region=region)
                assert result["region"] == region

    def test_defaults_to_full_region(self):
        """Test region defaults to full when not specified."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": True,
                        "node_id": "vtkMRMLSegmentationNode1",
                        "node_name": "Spine",
                        "region": "full",
                        "coordinate_system": "RAS",
                        "vertebrae_found": [],
                        "vertebrae_data": {},
                        "measurements": {},
                        "reference_ranges": {},
                        "statuses": {},
                    }
                ),
            }
            mock_get_client.return_value = mock_client

            result = measure_spine_alignment("vtkMRMLSegmentationNode1")

            assert result["region"] == "full"


# =============================================================================
# Code Generation Tests -- CCJ
# =============================================================================


class TestCCJCodeGeneration:
    """Test CCJ measurement code generation."""

    def test_landmark_code_uses_json_escaped_node_id(self):
        """Test landmark extraction code uses JSON-escaped node ID."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        code = _build_ccj_landmark_extraction_code(safe_id)

        assert 'node_id = "vtkMRMLSegmentationNode1"' in code
        assert "GetNodeByID(node_id)" in code

    def test_landmark_code_contains_ras_coordinate_system(self):
        """Test generated code reports RAS coordinate system."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        code = _build_ccj_landmark_extraction_code(safe_id)

        assert "'coordinate_system': 'RAS'" in code

    def test_landmark_code_extracts_c1_c2(self):
        """Test generated code looks for C1 and C2 segments."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        code = _build_ccj_landmark_extraction_code(safe_id)

        assert "'C1'" in code
        assert "'C2'" in code

    def test_angles_code_computes_adi(self):
        """Test CCJ angles code computes ADI measurement."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "ADI_mm" in code
        assert "c1_anterior_arch" in code
        assert "dens_tip" in code

    def test_angles_code_computes_powers_ratio(self):
        """Test CCJ angles code computes Powers ratio."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "powers_ratio" in code
        assert "basion" in code
        assert "opisthion" in code

    def test_angles_code_computes_cxa(self):
        """Test CCJ angles code computes CXA (clivo-axial angle)."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "CXA_deg" in code

    def test_angles_code_computes_bdi_bai(self):
        """Test CCJ angles code computes BDI and BAI."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "BDI_mm" in code
        assert "BAI_mm" in code

    def test_angles_code_computes_chamberlain_mcgregor(self):
        """Test CCJ angles code computes Chamberlain and McGregor lines."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "chamberlain_mm" in code
        assert "mcgregor_mm" in code

    def test_angles_code_computes_wackenheim(self):
        """Test CCJ angles code computes Wackenheim line."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "wackenheim_mm" in code

    def test_angles_code_computes_ranawat(self):
        """Test CCJ angles code computes Ranawat criterion."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "ranawat_value_mm" in code

    def test_angles_code_uses_numpy_for_3d(self):
        """Test generated code uses numpy for 3D vector calculations."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "import numpy as np" in code
        assert "np.linalg.norm" in code
        assert "np.degrees" in code

    def test_angles_code_includes_reference_ranges(self):
        """Test generated code includes reference ranges and status classification."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("adult")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "reference_ranges" in code
        assert "classify(" in code
        assert "'normal'" in code
        assert "'above_normal'" in code
        assert "'below_normal'" in code

    def test_child_population_uses_child_adi_range(self):
        """Test child population references ADI_child range."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_pop = json.dumps("child")
        code = _build_ccj_angles_code(safe_id, safe_pop)

        assert "ADI_child" in code


# =============================================================================
# Code Generation Tests -- Sagittal Alignment
# =============================================================================


class TestSagittalAlignmentCodeGeneration:
    """Test sagittal alignment code generation."""

    def test_centroid_code_uses_json_escaped_node_id(self):
        """Test centroid extraction code uses JSON-escaped node ID."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_vertebral_centroid_extraction_code(safe_id, safe_region)

        assert 'node_id = "vtkMRMLSegmentationNode1"' in code
        assert "GetNodeByID(node_id)" in code

    def test_centroid_code_uses_json_escaped_region(self):
        """Test centroid extraction code uses JSON-escaped region."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("cervical")
        code = _build_vertebral_centroid_extraction_code(safe_id, safe_region)

        assert 'region = "cervical"' in code

    def test_centroid_code_extracts_endplates(self):
        """Test centroid code extracts superior and inferior endplates."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_vertebral_centroid_extraction_code(safe_id, safe_region)

        assert "superior_endplate" in code
        assert "inferior_endplate" in code

    def test_alignment_code_computes_cervical_lordosis(self):
        """Test alignment code computes cervical lordosis (C2-C7)."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "cervical_lordosis_deg" in code
        assert "'C2'" in code
        assert "'C7'" in code

    def test_alignment_code_computes_thoracic_kyphosis(self):
        """Test alignment code computes thoracic kyphosis."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "thoracic_kyphosis_deg" in code

    def test_alignment_code_computes_lumbar_lordosis(self):
        """Test alignment code computes lumbar lordosis."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "lumbar_lordosis_deg" in code

    def test_alignment_code_computes_sva(self):
        """Test alignment code computes SVA and C2-C7 SVA."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "SVA_mm" in code
        assert "C2_C7_SVA_mm" in code

    def test_alignment_code_computes_t1_slope(self):
        """Test alignment code computes T1 slope."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "T1_slope_deg" in code

    def test_alignment_code_computes_pelvic_parameters(self):
        """Test alignment code computes PI, PT, and SS."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "pelvic_incidence_deg" in code
        assert "pelvic_tilt_deg" in code
        assert "sacral_slope_deg" in code

    def test_alignment_code_computes_pi_ll_mismatch(self):
        """Test alignment code computes PI-LL mismatch."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "PI_LL_mismatch_deg" in code

    def test_alignment_code_includes_roussouly(self):
        """Test alignment code includes Roussouly classification."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "roussouly_type" in code

    def test_alignment_code_includes_schwab(self):
        """Test alignment code includes SRS-Schwab classification."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "schwab_classification" in code
        assert "SVA_modifier" in code
        assert "PT_modifier" in code
        assert "PI_LL_modifier" in code

    def test_alignment_code_uses_numpy_for_3d(self):
        """Test alignment code uses numpy for 3D Cobb angle calculations."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "import numpy as np" in code
        assert "np.linalg.norm" in code
        assert "np.degrees" in code
        assert "cobb_angle_3d" in code

    def test_alignment_code_uses_totalsegmentator_map(self):
        """Test alignment code uses TotalSegmentator label mapping."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "ts_map" in code
        assert "vertebrae_C1" in code

    def test_alignment_code_includes_reference_ranges(self):
        """Test alignment code includes reference ranges for all parameters."""
        safe_id = json.dumps("vtkMRMLSegmentationNode1")
        safe_region = json.dumps("full")
        code = _build_sagittal_alignment_code(safe_id, safe_region)

        assert "reference_ranges" in code
        assert "statuses" in code
        assert "classify(" in code


# =============================================================================
# Full Tool Execution Tests with Mocked Slicer
# =============================================================================


class TestMeasureCCJAnglesExecution:
    """Test measure_ccj_angles with mocked Slicer client."""

    @pytest.fixture
    def ccj_result(self):
        """Sample CCJ measurement result from Slicer."""
        return {
            "success": True,
            "landmarks": {
                "basion": [0.0, 5.0, 40.0],
                "opisthion": [0.0, -15.0, 38.0],
                "dens_tip": [0.0, 3.0, 35.0],
                "c1_anterior_arch": [0.0, 12.0, 30.0],
                "c1_posterior_arch": [0.0, -8.0, 30.0],
                "c2_posteroinferior": [0.0, -5.0, 20.0],
                "c1_centroid": [0.0, 2.0, 30.0],
                "c2_centroid": [0.0, 1.0, 22.0],
            },
            "measurements": {
                "ADI_mm": 2.5,
                "BDI_mm": 7.1,
                "BAI_mm": 8.3,
                "powers_ratio": 0.85,
                "CXA_deg": 155.0,
                "ranawat_value_mm": 15.2,
                "chamberlain_mm": 1.5,
                "mcgregor_mm": 1.5,
                "wackenheim_mm": 3.2,
            },
            "reference_ranges": {
                "ADI": {"min": 0.0, "max": 3.0, "unit": "mm"},
                "BDI": {"min": 0.0, "max": 12.0, "unit": "mm"},
                "powers_ratio": {"min": 0.0, "max": 1.0, "unit": "ratio"},
                "CXA": {"min": 150.0, "max": 165.0, "unit": "degrees"},
            },
            "statuses": {
                "ADI": "normal",
                "BDI": "normal",
                "powers_ratio": "normal",
                "CXA": "normal",
            },
            "segments_found": {"C1": True, "C2": True, "skull_base": False},
            "coordinate_system": "RAS",
            "node_id": "vtkMRMLSegmentationNode1",
            "node_name": "SpineSegmentation",
            "population": "adult",
        }

    def test_successful_ccj_measurement(self, ccj_result):
        """Test successful CCJ measurement returns all expected fields."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(ccj_result),
            }
            mock_get_client.return_value = mock_client

            result = measure_ccj_angles("vtkMRMLSegmentationNode1")

            assert result["success"] is True
            assert "landmarks" in result
            assert "measurements" in result
            assert "reference_ranges" in result
            assert "statuses" in result
            assert result["coordinate_system"] == "RAS"

    def test_ccj_returns_all_measurements(self, ccj_result):
        """Test CCJ returns all expected measurement keys."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(ccj_result),
            }
            mock_get_client.return_value = mock_client

            result = measure_ccj_angles("vtkMRMLSegmentationNode1")

            measurements = result["measurements"]
            assert "ADI_mm" in measurements
            assert "BDI_mm" in measurements
            assert "BAI_mm" in measurements
            assert "powers_ratio" in measurements
            assert "CXA_deg" in measurements
            assert "ranawat_value_mm" in measurements
            assert "chamberlain_mm" in measurements
            assert "wackenheim_mm" in measurements

    def test_ccj_uses_extended_timeout(self, ccj_result):
        """Test CCJ measurement uses extended timeout."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(ccj_result),
            }
            mock_get_client.return_value = mock_client

            measure_ccj_angles("vtkMRMLSegmentationNode1")

            call_kwargs = mock_client.exec_python.call_args
            assert call_kwargs[1]["timeout"] > 30

    def test_ccj_connection_error_propagates(self):
        """Test SlicerConnectionError is re-raised."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                measure_ccj_angles("vtkMRMLSegmentationNode1")

    def test_ccj_empty_result_raises_error(self):
        """Test empty Slicer result raises SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                measure_ccj_angles("vtkMRMLSegmentationNode1")

    def test_ccj_null_result_raises_error(self):
        """Test null Slicer result raises SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "null",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                measure_ccj_angles("vtkMRMLSegmentationNode1")

    def test_ccj_malformed_json_raises_error(self):
        """Test malformed JSON result raises SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "{invalid json",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                measure_ccj_angles("vtkMRMLSegmentationNode1")


class TestMeasureSpineAlignmentExecution:
    """Test measure_spine_alignment with mocked Slicer client."""

    @pytest.fixture
    def alignment_result(self):
        """Sample sagittal alignment result from Slicer."""
        return {
            "success": True,
            "node_id": "vtkMRMLSegmentationNode1",
            "node_name": "SpineSegmentation",
            "region": "full",
            "coordinate_system": "RAS",
            "vertebrae_found": ["C2", "C7", "T1", "T4", "T12", "L1", "L5"],
            "vertebrae_data": {
                "C2": {
                    "centroid": [0.0, 5.0, 100.0],
                    "superior_endplate": [0.0, 5.5, 102.0],
                    "inferior_endplate": [0.0, 4.5, 98.0],
                    "height_mm": 15.0,
                },
                "C7": {
                    "centroid": [0.0, 3.0, 70.0],
                    "superior_endplate": [0.0, 3.2, 72.0],
                    "inferior_endplate": [0.0, 2.8, 68.0],
                    "height_mm": 16.0,
                },
                "T1": {
                    "centroid": [0.0, 2.0, 60.0],
                    "superior_endplate": [0.0, 2.5, 63.0],
                    "inferior_endplate": [0.0, 1.5, 57.0],
                    "height_mm": 18.0,
                },
                "L5": {
                    "centroid": [0.0, -5.0, -30.0],
                    "superior_endplate": [0.0, -4.0, -27.0],
                    "inferior_endplate": [0.0, -6.0, -33.0],
                    "height_mm": 28.0,
                },
            },
            "measurements": {
                "cervical_lordosis_deg": 25.0,
                "C2_C7_SVA_mm": 15.0,
                "T1_slope_deg": 25.0,
                "thoracic_kyphosis_deg": 35.0,
                "thoracic_kyphosis_levels": "T1-T12",
                "lumbar_lordosis_deg": 50.0,
                "lumbar_lordosis_levels": "L1-L5",
                "SVA_mm": 30.0,
                "sacral_slope_deg": 35.0,
                "pelvic_tilt_deg": 15.0,
                "pelvic_incidence_deg": 50.0,
                "PI_LL_mismatch_deg": 0.0,
                "roussouly_type": 3,
                "schwab_classification": {
                    "SVA_modifier": "0",
                    "PT_modifier": "0",
                    "PI_LL_modifier": "0",
                },
            },
            "reference_ranges": {
                "cervical_lordosis": {"min": 20.0, "max": 40.0, "unit": "degrees"},
                "thoracic_kyphosis": {"min": 20.0, "max": 50.0, "unit": "degrees"},
                "lumbar_lordosis": {"min": 40.0, "max": 70.0, "unit": "degrees"},
                "SVA": {"min": -50.0, "max": 50.0, "unit": "mm"},
            },
            "statuses": {
                "cervical_lordosis": "normal",
                "thoracic_kyphosis": "normal",
                "lumbar_lordosis": "normal",
                "SVA": "normal",
            },
        }

    def test_successful_alignment_measurement(self, alignment_result):
        """Test successful alignment measurement returns all expected fields."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(alignment_result),
            }
            mock_get_client.return_value = mock_client

            result = measure_spine_alignment("vtkMRMLSegmentationNode1")

            assert result["success"] is True
            assert "vertebrae_found" in result
            assert "vertebrae_data" in result
            assert "measurements" in result
            assert "reference_ranges" in result
            assert "statuses" in result
            assert result["coordinate_system"] == "RAS"

    def test_alignment_returns_all_sagittal_parameters(self, alignment_result):
        """Test alignment returns all expected sagittal parameters."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(alignment_result),
            }
            mock_get_client.return_value = mock_client

            result = measure_spine_alignment("vtkMRMLSegmentationNode1")

            measurements = result["measurements"]
            assert "cervical_lordosis_deg" in measurements
            assert "thoracic_kyphosis_deg" in measurements
            assert "lumbar_lordosis_deg" in measurements
            assert "SVA_mm" in measurements
            assert "C2_C7_SVA_mm" in measurements
            assert "T1_slope_deg" in measurements
            assert "sacral_slope_deg" in measurements
            assert "pelvic_tilt_deg" in measurements
            assert "pelvic_incidence_deg" in measurements
            assert "PI_LL_mismatch_deg" in measurements
            assert "roussouly_type" in measurements
            assert "schwab_classification" in measurements

    def test_alignment_returns_vertebrae_data(self, alignment_result):
        """Test alignment returns per-vertebra geometry data."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(alignment_result),
            }
            mock_get_client.return_value = mock_client

            result = measure_spine_alignment("vtkMRMLSegmentationNode1")

            vd = result["vertebrae_data"]
            assert "C2" in vd
            assert "centroid" in vd["C2"]
            assert "superior_endplate" in vd["C2"]
            assert "inferior_endplate" in vd["C2"]
            assert "height_mm" in vd["C2"]

    def test_alignment_uses_extended_timeout(self, alignment_result):
        """Test alignment measurement uses extended timeout."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(alignment_result),
            }
            mock_get_client.return_value = mock_client

            measure_spine_alignment("vtkMRMLSegmentationNode1")

            call_kwargs = mock_client.exec_python.call_args
            assert call_kwargs[1]["timeout"] > 30

    def test_alignment_connection_error_propagates(self):
        """Test SlicerConnectionError is re-raised."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                measure_spine_alignment("vtkMRMLSegmentationNode1")

    def test_alignment_empty_result_raises_error(self):
        """Test empty Slicer result raises SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                measure_spine_alignment("vtkMRMLSegmentationNode1")

    def test_alignment_null_result_raises_error(self):
        """Test null Slicer result raises SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "null",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                measure_spine_alignment("vtkMRMLSegmentationNode1")

    def test_alignment_cervical_region_passes_region(self):
        """Test cervical region is correctly passed to Slicer code."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(
                    {
                        "success": True,
                        "node_id": "vtkMRMLSegmentationNode1",
                        "node_name": "Spine",
                        "region": "cervical",
                        "coordinate_system": "RAS",
                        "vertebrae_found": ["C2", "C7"],
                        "vertebrae_data": {},
                        "measurements": {},
                        "reference_ranges": {},
                        "statuses": {},
                    }
                ),
            }
            mock_get_client.return_value = mock_client

            measure_spine_alignment("vtkMRMLSegmentationNode1", region="cervical")

            python_code = mock_client.exec_python.call_args[0][0]
            assert '"cervical"' in python_code


# =============================================================================
# Spine Segmentation -- Input Validation Tests
# =============================================================================


class TestSegmentSpineValidation:
    """Tests for segment_spine input validation."""

    def test_empty_node_id_rejected(self):
        """Empty node ID must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            segment_spine("")
        assert exc_info.value.field == "node_id"

    def test_invalid_node_id_rejected(self):
        """Node ID with invalid characters must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            segment_spine("1invalidNode")
        assert exc_info.value.field == "node_id"

    def test_invalid_region_rejected(self):
        """Invalid region must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            segment_spine("vtkMRMLScalarVolumeNode1", region="sacral")
        assert exc_info.value.field == "region"
        assert "sacral" in str(exc_info.value)

    def test_valid_regions_accepted(self):
        """All valid regions must not raise ValidationError on the region check."""
        for region in ["cervical", "thoracic", "lumbar", "full"]:
            # Will fail at client.exec_python, but should not fail at region validation
            with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
                mock_client = Mock()
                mock_client.exec_python.return_value = {
                    "success": True,
                    "result": (
                        '{"success": true, "input_node_id": "v1", "region": "' + region + '",'
                        ' "output_segmentation_id": "seg1",'
                        ' "output_segmentation_name": "test_spine_seg",'
                        ' "vertebrae_count": 0, "vertebrae": [],'
                        ' "discs": [], "other_structures": [],'
                        ' "processing_time_seconds": 1.0}'
                    ),
                }
                mock_get_client.return_value = mock_client

                result = segment_spine("vtkMRMLScalarVolumeNode1", region=region)
                assert result["success"] is True

    def test_node_id_with_injection_rejected(self):
        """Node ID with shell metacharacters must raise ValidationError."""
        with pytest.raises(ValidationError):
            segment_spine("node; rm -rf /")


# =============================================================================
# Spine Segmentation -- Code Generation Tests
# =============================================================================


class TestBuildSpineSegmentationCode:
    """Tests for spine segmentation code generation."""

    def test_code_contains_input_node_id(self):
        """Generated code must reference the input node ID."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"full"', False, False)
        assert '"vtkNode1"' in code

    def test_code_contains_region(self):
        """Generated code must reference the region."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"cervical"', False, False)
        assert '"cervical"' in code

    def test_code_uses_totalsegmentator(self):
        """Generated code must import TotalSegmentator."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"full"', False, False)
        assert "TotalSegmentator" in code

    def test_code_outputs_json(self):
        """Generated code must end with print(json.dumps(result))."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"full"', False, False)
        assert "print(json.dumps(result))" in code

    def test_code_uses_vertebral_body_task_when_no_extras(self):
        """Without discs/cord, code should use vertebral_body task."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"full"', False, False)
        assert '"vertebral_body"' in code

    def test_code_uses_total_task_with_discs(self):
        """With include_discs=True, code should use total task."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"full"', True, False)
        assert '"total"' in code

    def test_code_uses_total_task_with_spinal_cord(self):
        """With include_spinal_cord=True, code should use total task."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"full"', False, True)
        assert '"total"' in code

    def test_code_includes_disc_filtering(self):
        """Generated code must handle disc filtering logic."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"lumbar"', True, False)
        assert "include_discs" in code
        assert "DISC_MAP" in code

    def test_code_includes_spinal_cord_filtering(self):
        """Generated code must handle spinal cord filtering logic."""
        code = _build_spine_segmentation_code('"vtkNode1"', '"full"', False, True)
        assert "include_spinal_cord" in code
        assert "spinal_cord" in code


# =============================================================================
# Spine Segmentation -- Execution Tests (Mocked Client)
# =============================================================================


class TestSegmentSpineExecution:
    """Tests for segment_spine with mocked Slicer client."""

    def _mock_exec_result(self, region="full", vertebrae_count=24):
        """Create a standard mock exec_python result."""
        return {
            "success": True,
            "result": (
                '{"success": true,'
                f' "input_node_id": "vtkMRMLScalarVolumeNode1",'
                f' "region": "{region}",'
                ' "output_segmentation_id": "vtkMRMLSegmentationNode1",'
                ' "output_segmentation_name": "CT_spine_seg",'
                f' "vertebrae_count": {vertebrae_count},'
                ' "vertebrae": [],'
                ' "discs": [],'
                ' "other_structures": [],'
                ' "processing_time_seconds": 45.2}'
            ),
        }

    def test_successful_segmentation(self):
        """Successful segmentation must return structured result."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = self._mock_exec_result()
            mock_get_client.return_value = mock_client

            result = segment_spine("vtkMRMLScalarVolumeNode1")

            assert result["success"] is True
            assert result["region"] == "full"
            assert "output_segmentation_id" in result
            assert "vertebrae_count" in result

    def test_uses_extended_timeout(self):
        """segment_spine must use SPINE_SEGMENTATION_TIMEOUT."""
        from slicer_mcp.spine_constants import SPINE_SEGMENTATION_TIMEOUT

        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = self._mock_exec_result()
            mock_get_client.return_value = mock_client

            segment_spine("vtkMRMLScalarVolumeNode1")

            mock_client.exec_python.assert_called_once()
            _, call_kwargs = mock_client.exec_python.call_args
            assert call_kwargs["timeout"] == SPINE_SEGMENTATION_TIMEOUT

    def test_adds_long_operation_metadata(self):
        """Result must include long_operation metadata."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = self._mock_exec_result(region="cervical")
            mock_get_client.return_value = mock_client

            result = segment_spine("vtkMRMLScalarVolumeNode1", region="cervical")

            assert "long_operation" in result
            assert result["long_operation"]["type"] == "spine_segmentation"
            assert result["long_operation"]["region"] == "cervical"
            assert "timeout_seconds" in result["long_operation"]

    def test_connection_error_propagated(self):
        """SlicerConnectionError must propagate from exec_python."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                segment_spine("vtkMRMLScalarVolumeNode1")

    def test_empty_result_raises_error(self):
        """Empty result from Slicer must raise SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {"success": True, "result": ""}
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                segment_spine("vtkMRMLScalarVolumeNode1")

    def test_malformed_json_raises_error(self):
        """Malformed JSON result must raise SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "not valid json{{{",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                segment_spine("vtkMRMLScalarVolumeNode1")

    def test_cervical_region_passed_to_code(self):
        """Region parameter must be passed to generated Python code."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = self._mock_exec_result(
                region="cervical", vertebrae_count=7
            )
            mock_get_client.return_value = mock_client

            segment_spine("vtkMRMLScalarVolumeNode1", region="cervical")

            # Verify the code sent to Slicer contains the cervical region
            call_args = mock_client.exec_python.call_args
            python_code = call_args[0][0]
            assert '"cervical"' in python_code


# =============================================================================
# Spine Segmentation -- Server Registration Tests
# =============================================================================


class TestSegmentSpineRegistration:
    """Tests for segment_spine registration in server.py."""

    def test_segment_spine_registered_as_tool(self):
        """segment_spine must be registered as an MCP tool in server.py."""
        from slicer_mcp import server

        assert hasattr(server, "segment_spine")

    def test_server_wrapper_catches_exceptions(self):
        """Server wrapper must catch exceptions via _handle_tool_error."""
        from slicer_mcp.server import segment_spine as server_segment_spine

        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError(
                "Connection failed", details={"url": "http://localhost:2016"}
            )
            mock_get_client.return_value = mock_client

            result = server_segment_spine("vtkMRMLScalarVolumeNode1")

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_server_wrapper_handles_validation_error(self):
        """Server wrapper must catch ValidationError."""
        from slicer_mcp.server import segment_spine as server_segment_spine

        result = server_segment_spine("", region="full")

        assert result["success"] is False
        assert result["error_type"] == "unexpected"


# =============================================================================
# Vertebral Artery Segmentation -- Validation Tests
# =============================================================================


class TestSegmentVertebralArteryValidation:
    """Tests for segment_vertebral_artery input validation."""

    def test_valid_sides_defined(self):
        """Valid side parameters include left, right, and both."""
        assert "left" in VALID_ARTERY_SIDES
        assert "right" in VALID_ARTERY_SIDES
        assert "both" in VALID_ARTERY_SIDES
        assert len(VALID_ARTERY_SIDES) == 3

    def test_invalid_side_rejected(self):
        """Invalid side parameter raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            segment_vertebral_artery("vtkMRMLScalarVolumeNode1", side="anterior")
        assert exc_info.value.field == "side"

    def test_empty_node_id_rejected(self):
        """Empty input_node_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            segment_vertebral_artery("")
        assert exc_info.value.field == "node_id"

    def test_invalid_node_id_format_rejected(self):
        """Node ID starting with a number raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            segment_vertebral_artery("1invalidNode")
        assert exc_info.value.field == "node_id"

    def test_node_id_with_special_chars_rejected(self):
        """Node ID with special chars (code injection attempt) is rejected."""
        with pytest.raises(ValidationError):
            segment_vertebral_artery("node; rm -rf /")


class TestSeedPointValidation:
    """Tests for seed point validation in vertebral artery segmentation."""

    def test_valid_seed_points(self):
        """Valid seed points pass validation."""
        seeds = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = _validate_seed_points(seeds)
        assert result == seeds

    def test_single_seed_point(self):
        """Single seed point is valid."""
        seeds = [[10.5, -20.3, 15.0]]
        result = _validate_seed_points(seeds)
        assert len(result) == 1

    def test_integer_coordinates_accepted(self):
        """Integer coordinates are accepted alongside floats."""
        seeds = [[1, 2, 3]]
        result = _validate_seed_points(seeds)
        assert result == [[1, 2, 3]]

    def test_empty_seed_list_rejected(self):
        """Empty seed points list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_seed_points([])
        assert exc_info.value.field == "seed_points"

    def test_too_many_seeds_rejected(self):
        """More than 50 seed points raises ValidationError."""
        seeds = [[float(i), 0.0, 0.0] for i in range(51)]
        with pytest.raises(ValidationError) as exc_info:
            _validate_seed_points(seeds)
        assert "50" in str(exc_info.value)

    def test_wrong_dimension_rejected(self):
        """Seed point with != 3 coordinates raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_seed_points([[1.0, 2.0]])
        assert exc_info.value.field == "seed_points"

    def test_non_numeric_coordinate_rejected(self):
        """Non-numeric coordinate raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_seed_points([[1.0, "bad", 3.0]])
        assert exc_info.value.field == "seed_points"

    def test_string_in_list_rejected(self):
        """String instead of coordinate list raises ValidationError."""
        with pytest.raises(ValidationError):
            _validate_seed_points(["not_a_point"])


# =============================================================================
# Vertebral Artery Segmentation -- Execution Tests
# =============================================================================


class TestSegmentVertebralArteryExecution:
    """Tests for segment_vertebral_artery execution with mocked Slicer client."""

    def test_successful_segmentation_without_seeds(self):
        """Successful segmentation returns expected result structure."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": (
                    '{"success": true,'
                    ' "input_node_id": "vtkMRMLScalarVolumeNode1",'
                    ' "side": "both",'
                    ' "model_node_id": "vtkMRMLSegmentationNode1",'
                    ' "model_node_name": "CTA_VA_seg",'
                    ' "centerline_node_id": "vtkMRMLModelNode1",'
                    ' "centerline_node_name": "CTA_VA_centerline",'
                    ' "vesselness_node_id": "vtkMRMLScalarVolumeNode2",'
                    ' "diameters_mm": [3.2, 3.5, 3.1, 2.9],'
                    ' "mean_diameter_mm": 3.18,'
                    ' "processing_time_seconds": 45.2}'
                ),
            }
            mock_get_client.return_value = mock_client

            result = segment_vertebral_artery("vtkMRMLScalarVolumeNode1", side="both")

            assert result["success"] is True
            assert result["model_node_id"] == "vtkMRMLSegmentationNode1"
            assert result["centerline_node_id"] == "vtkMRMLModelNode1"
            assert len(result["diameters_mm"]) == 4
            assert result["mean_diameter_mm"] == 3.18
            assert "long_operation" in result
            assert result["long_operation"]["type"] == "vertebral_artery_segmentation"

    def test_successful_segmentation_with_seeds(self):
        """Segmentation with seed points uses seeded code path."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": (
                    '{"success": true,'
                    ' "input_node_id": "vtkMRMLScalarVolumeNode1",'
                    ' "side": "left",'
                    ' "model_node_id": "vtkMRMLSegmentationNode1",'
                    ' "model_node_name": "CTA_VA_seg",'
                    ' "centerline_node_id": "vtkMRMLModelNode1",'
                    ' "centerline_node_name": "CTA_VA_centerline",'
                    ' "vesselness_node_id": "vtkMRMLScalarVolumeNode2",'
                    ' "diameters_mm": [3.0, 3.2],'
                    ' "mean_diameter_mm": 3.1,'
                    ' "seed_count": 2,'
                    ' "processing_time_seconds": 38.5}'
                ),
            }
            mock_get_client.return_value = mock_client

            result = segment_vertebral_artery(
                "vtkMRMLScalarVolumeNode1",
                side="left",
                seed_points=[[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]],
            )

            assert result["success"] is True
            assert result["seed_count"] == 2
            # Verify exec_python was called with code containing seed_points
            call_args = mock_client.exec_python.call_args
            python_code = call_args[0][0]
            assert "seed_points" in python_code

    def test_uses_segmentation_timeout(self):
        """Tool uses SEGMENTATION_TIMEOUT for exec_python call."""
        from slicer_mcp.constants import SEGMENTATION_TIMEOUT

        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": (
                    '{"success": true,'
                    ' "mean_diameter_mm": 3.0,'
                    ' "processing_time_seconds": 10.0}'
                ),
            }
            mock_get_client.return_value = mock_client

            segment_vertebral_artery("vtkMRMLScalarVolumeNode1")

            mock_client.exec_python.assert_called_once()
            _, call_kwargs = mock_client.exec_python.call_args
            assert call_kwargs["timeout"] == SEGMENTATION_TIMEOUT

    def test_connection_error_propagated(self):
        """SlicerConnectionError from client is propagated."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                segment_vertebral_artery("vtkMRMLScalarVolumeNode1")

    def test_empty_result_raises_error(self):
        """Empty result from Slicer raises SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                segment_vertebral_artery("vtkMRMLScalarVolumeNode1")

    def test_left_side_parameter_passed(self):
        """Side parameter 'left' is included in generated Python code."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"success": true, "processing_time_seconds": 1.0}',
            }
            mock_get_client.return_value = mock_client

            segment_vertebral_artery("vtkMRMLScalarVolumeNode1", side="left")

            call_args = mock_client.exec_python.call_args
            python_code = call_args[0][0]
            assert '"left"' in python_code


# =============================================================================
# Bone Quality Analysis -- Validation Tests
# =============================================================================


class TestAnalyzeBoneQualityValidation:
    """Tests for analyze_bone_quality input validation."""

    def test_valid_regions_defined(self):
        """Valid bone regions match spine_constants SPINE_REGIONS."""
        assert "cervical" in VALID_BONE_REGIONS
        assert "thoracic" in VALID_BONE_REGIONS
        assert "lumbar" in VALID_BONE_REGIONS
        assert "full" in VALID_BONE_REGIONS
        assert len(VALID_BONE_REGIONS) == 4

    def test_invalid_region_rejected(self):
        """Invalid region raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            analyze_bone_quality(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLSegmentationNode1",
                region="sacral",
            )
        assert exc_info.value.field == "region"

    def test_empty_input_node_id_rejected(self):
        """Empty input_node_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            analyze_bone_quality("", "vtkMRMLSegmentationNode1")
        assert exc_info.value.field == "node_id"

    def test_empty_segmentation_node_id_rejected(self):
        """Empty segmentation_node_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            analyze_bone_quality("vtkMRMLScalarVolumeNode1", "")
        assert exc_info.value.field == "node_id"

    def test_invalid_input_node_format_rejected(self):
        """Input node ID with invalid format raises ValidationError."""
        with pytest.raises(ValidationError):
            analyze_bone_quality("123bad", "vtkMRMLSegmentationNode1")

    def test_invalid_segmentation_node_format_rejected(self):
        """Segmentation node ID with invalid format raises ValidationError."""
        with pytest.raises(ValidationError):
            analyze_bone_quality("vtkMRMLScalarVolumeNode1", "bad node!")


# =============================================================================
# Bone Quality Analysis -- Execution Tests
# =============================================================================


class TestAnalyzeBoneQualityExecution:
    """Tests for analyze_bone_quality execution with mocked Slicer client."""

    def _mock_bone_result(self):
        """Return a realistic bone quality analysis result JSON."""
        return (
            '{"success": true,'
            ' "input_node_id": "vtkMRMLScalarVolumeNode1",'
            ' "segmentation_node_id": "vtkMRMLSegmentationNode1",'
            ' "region": "lumbar",'
            ' "vertebrae": ['
            '   {"name": "L1", "mean_hu": 142.3, "classification": "normal",'
            '    "voxel_count": 15000},'
            '   {"name": "L2", "mean_hu": 118.5, "classification": "osteopenia",'
            '    "voxel_count": 14500},'
            '   {"name": "L3", "mean_hu": 85.2, "classification": "osteoporosis",'
            '    "voxel_count": 16000}'
            " ],"
            ' "vertebrae_count": 3,'
            ' "summary": {'
            '   "mean_hu_overall": 115.3,'
            '   "normal_count": 1,'
            '   "osteopenia_count": 1,'
            '   "osteoporosis_count": 1,'
            '   "has_bone_texture_metrics": false'
            " },"
            ' "processing_time_seconds": 25.4}'
        )

    def test_successful_analysis(self):
        """Successful bone quality analysis returns expected structure."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": self._mock_bone_result(),
            }
            mock_get_client.return_value = mock_client

            result = analyze_bone_quality(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLSegmentationNode1",
                region="lumbar",
            )

            assert result["success"] is True
            assert result["vertebrae_count"] == 3
            assert len(result["vertebrae"]) == 3
            assert result["summary"]["normal_count"] == 1
            assert result["summary"]["osteopenia_count"] == 1
            assert result["summary"]["osteoporosis_count"] == 1
            assert "long_operation" in result
            assert result["long_operation"]["type"] == "bone_quality_analysis"

    def test_vertebra_classification_values(self):
        """Per-vertebra classification values match Pickhardt thresholds."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": self._mock_bone_result(),
            }
            mock_get_client.return_value = mock_client

            result = analyze_bone_quality(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLSegmentationNode1",
            )

            vertebrae = result["vertebrae"]
            # L1: 142.3 HU >= 135 -> normal
            assert vertebrae[0]["classification"] == "normal"
            # L2: 118.5 HU >= 90, < 135 -> osteopenia
            assert vertebrae[1]["classification"] == "osteopenia"
            # L3: 85.2 HU < 90 -> osteoporosis
            assert vertebrae[2]["classification"] == "osteoporosis"

    def test_uses_spine_segmentation_timeout(self):
        """Tool uses SPINE_SEGMENTATION_TIMEOUT for exec_python call."""
        from slicer_mcp.spine_constants import SPINE_SEGMENTATION_TIMEOUT

        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": self._mock_bone_result(),
            }
            mock_get_client.return_value = mock_client

            analyze_bone_quality(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLSegmentationNode1",
            )

            mock_client.exec_python.assert_called_once()
            _, call_kwargs = mock_client.exec_python.call_args
            assert call_kwargs["timeout"] == SPINE_SEGMENTATION_TIMEOUT

    def test_region_parameter_passed_to_code(self):
        """Region parameter is embedded in generated Python code."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": self._mock_bone_result(),
            }
            mock_get_client.return_value = mock_client

            analyze_bone_quality(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLSegmentationNode1",
                region="thoracic",
            )

            call_args = mock_client.exec_python.call_args
            python_code = call_args[0][0]
            assert '"thoracic"' in python_code

    def test_connection_error_propagated(self):
        """SlicerConnectionError from client is propagated."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                analyze_bone_quality(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLSegmentationNode1",
                )

    def test_empty_result_raises_error(self):
        """Empty result from Slicer raises SlicerConnectionError."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                analyze_bone_quality(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLSegmentationNode1",
                )

    def test_default_region_is_lumbar(self):
        """Default region parameter is lumbar."""
        with patch("slicer_mcp.spine_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": self._mock_bone_result(),
            }
            mock_get_client.return_value = mock_client

            analyze_bone_quality(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLSegmentationNode1",
            )

            call_args = mock_client.exec_python.call_args
            python_code = call_args[0][0]
            assert '"lumbar"' in python_code


# =============================================================================
# Constants Validation Tests
# =============================================================================


class TestSpineConstants:
    """Test spine constants are properly imported and valid."""

    def test_valid_populations_contains_adult_and_child(self):
        """Test VALID_POPULATIONS has expected values."""
        assert "adult" in VALID_POPULATIONS
        assert "child" in VALID_POPULATIONS
        assert len(VALID_POPULATIONS) == 2

    def test_valid_alignment_regions(self):
        """Test VALID_ALIGNMENT_REGIONS has expected values."""
        assert "cervical" in VALID_ALIGNMENT_REGIONS
        assert "thoracic" in VALID_ALIGNMENT_REGIONS
        assert "lumbar" in VALID_ALIGNMENT_REGIONS
        assert "full" in VALID_ALIGNMENT_REGIONS

    def test_spine_constants_import(self):
        """Test spine_constants module imports correctly."""
        from slicer_mcp.spine_constants import (
            CCJ_NORMAL_RANGES,
            REGION_VERTEBRAE,
            SPINE_REGIONS,
            TOTALSEGMENTATOR_VERTEBRA_MAP,
        )

        assert "BDI" in CCJ_NORMAL_RANGES
        assert "ADI_adult" in CCJ_NORMAL_RANGES
        assert "ADI_child" in CCJ_NORMAL_RANGES
        assert "cervical" in REGION_VERTEBRAE
        assert "cervical" in SPINE_REGIONS
        assert "vertebrae_C1" in TOTALSEGMENTATOR_VERTEBRA_MAP


# =============================================================================
# Spine Constants Integration Tests
# =============================================================================


class TestSpineConstantsIntegration:
    """Tests that spine_tools correctly uses spine_constants values."""

    def test_pickhardt_thresholds_used_in_code(self):
        """Generated bone quality code embeds Pickhardt HU thresholds."""
        from slicer_mcp.spine_constants import PICKHARDT_HU_THRESHOLDS
        from slicer_mcp.spine_tools import _build_bone_quality_code

        code = _build_bone_quality_code('"node1"', '"seg1"', '"lumbar"')

        # Verify the thresholds are embedded in the code
        assert str(PICKHARDT_HU_THRESHOLDS["normal_min"]) in code
        assert str(PICKHARDT_HU_THRESHOLDS["osteopenia_min"]) in code

    def test_valid_bone_regions_match_spine_regions(self):
        """VALID_BONE_REGIONS matches SPINE_REGIONS from constants."""
        from slicer_mcp.spine_constants import SPINE_REGIONS

        assert VALID_BONE_REGIONS == SPINE_REGIONS


# =============================================================================
# Server Registration Tests (All Spine Tools)
# =============================================================================


class TestServerRegistration:
    """Tests that spine tools are properly registered in the MCP server."""

    def test_measure_ccj_angles_registered(self):
        """Test measure_ccj_angles is registered as MCP tool."""
        from slicer_mcp import server

        assert hasattr(server, "measure_ccj_angles")

    def test_measure_spine_alignment_registered(self):
        """Test measure_spine_alignment is registered as MCP tool."""
        from slicer_mcp import server

        assert hasattr(server, "measure_spine_alignment")

    def test_server_imports_spine_tools(self):
        """Test server.py imports spine_tools module."""
        from slicer_mcp import server

        assert hasattr(server, "spine_tools")

    def test_ccj_tool_handles_error(self):
        """Test CCJ tool wrapper catches exceptions via _handle_tool_error."""
        from slicer_mcp.server import measure_ccj_angles as server_ccj

        with patch("slicer_mcp.server.spine_tools") as mock_spine_tools:
            mock_spine_tools.measure_ccj_angles.side_effect = RuntimeError("test error")

            result = server_ccj("vtkMRMLSegmentationNode1")

            assert result["success"] is False
            assert result["error_type"] == "unexpected"

    def test_alignment_tool_handles_connection_error(self):
        """Test alignment tool wrapper handles SlicerConnectionError."""
        from slicer_mcp.server import measure_spine_alignment as server_align

        with patch("slicer_mcp.server.spine_tools") as mock_spine_tools:
            mock_spine_tools.measure_spine_alignment.side_effect = SlicerConnectionError(
                "Connection refused"
            )

            result = server_align("vtkMRMLSegmentationNode1")

            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_segment_spine_registered(self):
        """segment_spine is registered as an MCP tool."""
        from slicer_mcp.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "segment_spine" in tool_names

    def test_segment_vertebral_artery_registered(self):
        """segment_vertebral_artery is registered as an MCP tool."""
        from slicer_mcp.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "segment_vertebral_artery" in tool_names

    def test_analyze_bone_quality_registered(self):
        """analyze_bone_quality is registered as an MCP tool."""
        from slicer_mcp.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "analyze_bone_quality" in tool_names
