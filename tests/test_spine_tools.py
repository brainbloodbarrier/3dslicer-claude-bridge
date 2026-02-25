"""Unit tests for spine measurement tool implementations."""

import json
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.spine_tools import (
    VALID_ALIGNMENT_REGIONS,
    VALID_POPULATIONS,
    _build_ccj_angles_code,
    _build_ccj_landmark_extraction_code,
    _build_sagittal_alignment_code,
    _build_vertebral_centroid_extraction_code,
    measure_ccj_angles,
    measure_spine_alignment,
)
from slicer_mcp.tools import ValidationError

# =============================================================================
# Input Validation Tests — measure_ccj_angles
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
# Input Validation Tests — measure_spine_alignment
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
# Code Generation Tests — CCJ
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
# Code Generation Tests — Sagittal Alignment
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
# Server Registration Tests
# =============================================================================


class TestServerRegistration:
    """Test spine tools are properly registered in server.py."""

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
