"""Unit tests for MRI diagnostic protocol tools."""

import json
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.diagnostic_tools_mri import (
    CORD_COMPRESSION_RATIO_NORMAL,
    CORD_T2_HYPERINTENSITY_THRESHOLD,
    DISC_HOMOGENEOUS_CV_THRESHOLD,
    METASTASIS_T1_LOW_THRESHOLD,
    METASTASIS_T2_HIGH_THRESHOLD,
    METASTASIS_T2_LOW_THRESHOLD,
    MODIC_T1_HIGH_THRESHOLD,
    MODIC_T1_LOW_THRESHOLD,
    MODIC_T2_HIGH_THRESHOLD,
    MODIC_T2_LOW_THRESHOLD,
    MRI_ANALYSIS_TIMEOUT,
    MSCC_THRESHOLD,
    PFIRRMANN_BRIGHT_THRESHOLD,
    VALID_MRI_REGIONS,
    _validate_mri_region,
    assess_disc_degeneration_mri,
    classify_modic_changes,
    detect_cord_compression_mri,
    detect_metastatic_lesions_mri,
)
from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.tools import ValidationError

# =============================================================================
# MRI Region Validation Tests
# =============================================================================


class TestValidateMriRegion:
    """Test MRI region parameter validation."""

    def test_valid_cervical(self):
        assert _validate_mri_region("cervical") == "cervical"

    def test_valid_thoracic(self):
        assert _validate_mri_region("thoracic") == "thoracic"

    def test_valid_lumbar(self):
        assert _validate_mri_region("lumbar") == "lumbar"

    def test_invalid_region_full(self):
        """Full is not valid for _validate_mri_region (only for metastasis tool)."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_mri_region("full")
        assert exc_info.value.field == "region"

    def test_invalid_region_empty(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_mri_region("")
        assert exc_info.value.field == "region"

    def test_invalid_region_arbitrary(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_mri_region("sacral")
        assert exc_info.value.field == "region"


# =============================================================================
# Constants Sanity Tests
# =============================================================================


class TestMRIConstants:
    """Test that MRI-specific constants have sensible values."""

    def test_modic_thresholds_ordered(self):
        assert MODIC_T1_LOW_THRESHOLD < 1.0 < MODIC_T1_HIGH_THRESHOLD
        assert MODIC_T2_LOW_THRESHOLD < 1.0 < MODIC_T2_HIGH_THRESHOLD

    def test_pfirrmann_thresholds_ordered(self):
        assert PFIRRMANN_BRIGHT_THRESHOLD > 0

    def test_cord_compression_thresholds(self):
        assert CORD_COMPRESSION_RATIO_NORMAL > 0
        assert CORD_T2_HYPERINTENSITY_THRESHOLD > 1.0

    def test_mscc_threshold(self):
        assert 0 < MSCC_THRESHOLD < 1.0

    def test_metastasis_thresholds(self):
        assert METASTASIS_T1_LOW_THRESHOLD < 1.0
        assert METASTASIS_T2_HIGH_THRESHOLD > 1.0
        assert METASTASIS_T2_LOW_THRESHOLD < 1.0

    def test_disc_homogeneity_threshold(self):
        assert 0 < DISC_HOMOGENEOUS_CV_THRESHOLD < 1.0

    def test_mri_analysis_timeout(self):
        assert MRI_ANALYSIS_TIMEOUT > 0

    def test_valid_mri_regions(self):
        assert "cervical" in VALID_MRI_REGIONS
        assert "thoracic" in VALID_MRI_REGIONS
        assert "lumbar" in VALID_MRI_REGIONS
        assert "full" not in VALID_MRI_REGIONS


# =============================================================================
# Tool 1: Modic Classification Tests
# =============================================================================


class TestClassifyModicChanges:
    """Test classify_modic_changes tool."""

    def test_invalid_t1_node_id(self):
        """Test invalid T1 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            classify_modic_changes("1invalid", "vtkMRMLScalarVolumeNode2")
        assert exc_info.value.field == "node_id"

    def test_invalid_t2_node_id(self):
        """Test invalid T2 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            classify_modic_changes("vtkMRMLScalarVolumeNode1", "")
        assert exc_info.value.field == "node_id"

    def test_invalid_region(self):
        """Test invalid region raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            classify_modic_changes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                region="invalid",
            )
        assert exc_info.value.field == "region"

    def test_successful_modic_classification(self):
        """Test successful Modic classification returns expected structure."""
        mock_result = {
            "success": True,
            "tool": "classify_modic_changes",
            "region": "lumbar",
            "reference_vertebra": "L3",
            "registration_performed": False,
            "levels": [
                {
                    "disc_level": "L4-L5",
                    "vertebra": "L4",
                    "endplate": "inferior_endplate",
                    "modic_type": 1,
                    "description": "Type I: edema/inflammation (T1 low, T2 high)",
                    "t1_signal_ratio": 0.75,
                    "t2_signal_ratio": 1.30,
                    "mixed_pattern": None,
                }
            ],
            "total_levels_analyzed": 1,
            "modic_summary": {"type_0": 0, "type_i": 1, "type_ii": 0, "type_iii": 0},
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = classify_modic_changes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                region="lumbar",
            )

            assert result["success"] is True
            assert result["tool"] == "classify_modic_changes"
            assert result["region"] == "lumbar"
            assert len(result["levels"]) == 1
            assert result["levels"][0]["modic_type"] == 1
            assert result["modic_summary"]["type_i"] == 1

            # Verify exec_python was called with extended timeout
            mock_client.exec_python.assert_called_once()
            call_kwargs = mock_client.exec_python.call_args
            assert call_kwargs[1]["timeout"] == MRI_ANALYSIS_TIMEOUT

    def test_modic_type_ii_classification(self):
        """Test Type II (fatty degeneration) classification."""
        mock_result = {
            "success": True,
            "tool": "classify_modic_changes",
            "region": "lumbar",
            "reference_vertebra": "L3",
            "registration_performed": False,
            "levels": [
                {
                    "disc_level": "L5-S1",
                    "vertebra": "L5",
                    "endplate": "inferior_endplate",
                    "modic_type": 2,
                    "description": "Type II: fatty degeneration (T1 high, T2 iso/high)",
                    "t1_signal_ratio": 1.25,
                    "t2_signal_ratio": 1.05,
                    "mixed_pattern": None,
                }
            ],
            "total_levels_analyzed": 1,
            "modic_summary": {"type_0": 0, "type_i": 0, "type_ii": 1, "type_iii": 0},
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = classify_modic_changes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
            )

            assert result["levels"][0]["modic_type"] == 2
            assert result["modic_summary"]["type_ii"] == 1

    def test_connection_error_propagates(self):
        """Test SlicerConnectionError propagates from exec_python."""
        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                classify_modic_changes(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLScalarVolumeNode2",
                )

    def test_malformed_json_raises_error(self):
        """Test malformed JSON result raises SlicerConnectionError."""
        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "not valid json {",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError) as exc_info:
                classify_modic_changes(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLScalarVolumeNode2",
                )
            assert "Failed to parse" in str(exc_info.value)

    def test_empty_result_raises_error(self):
        """Test empty result raises SlicerConnectionError."""
        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {"success": True, "result": ""}
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError) as exc_info:
                classify_modic_changes(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLScalarVolumeNode2",
                )
            assert "Empty result" in str(exc_info.value)


# =============================================================================
# Tool 2: Pfirrmann Disc Degeneration Tests
# =============================================================================


class TestAssessDiscDegenerationMri:
    """Test assess_disc_degeneration_mri tool."""

    def test_invalid_t2_node_id(self):
        """Test invalid T2 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            assess_disc_degeneration_mri("")
        assert exc_info.value.field == "node_id"

    def test_invalid_region(self):
        """Test invalid region raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            assess_disc_degeneration_mri("vtkMRMLScalarVolumeNode1", region="invalid")
        assert exc_info.value.field == "region"

    def test_successful_pfirrmann_grading(self):
        """Test successful Pfirrmann grading returns expected structure."""
        mock_result = {
            "success": True,
            "tool": "assess_disc_degeneration_mri",
            "region": "lumbar",
            "csf_reference_signal": 850.5,
            "discs": [
                {
                    "disc_level": "L4-L5",
                    "pfirrmann_grade": 3,
                    "pfirrmann_description": (
                        "Inhomogeneous gray signal, normal to slight height decrease, "
                        "unclear boundary"
                    ),
                    "signal_ratio_to_csf": 0.45,
                    "homogeneity_cv": 0.22,
                    "is_homogeneous": False,
                    "has_nucleus_annulus_distinction": True,
                    "disc_height_mm": 8.5,
                    "height_loss_percent": 15.2,
                },
                {
                    "disc_level": "L5-S1",
                    "pfirrmann_grade": 4,
                    "pfirrmann_description": (
                        "Inhomogeneous dark signal, moderate height decrease, lost boundary"
                    ),
                    "signal_ratio_to_csf": 0.25,
                    "homogeneity_cv": 0.35,
                    "is_homogeneous": False,
                    "has_nucleus_annulus_distinction": True,
                    "disc_height_mm": 5.2,
                    "height_loss_percent": 42.0,
                },
            ],
            "total_discs_analyzed": 2,
            "grade_summary": {
                "grade_i": 0,
                "grade_ii": 0,
                "grade_iii": 1,
                "grade_iv": 1,
                "grade_v": 0,
            },
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = assess_disc_degeneration_mri("vtkMRMLScalarVolumeNode1")

            assert result["success"] is True
            assert result["tool"] == "assess_disc_degeneration_mri"
            assert result["region"] == "lumbar"
            assert len(result["discs"]) == 2
            assert result["discs"][0]["pfirrmann_grade"] == 3
            assert result["discs"][1]["pfirrmann_grade"] == 4
            assert result["grade_summary"]["grade_iii"] == 1
            assert result["grade_summary"]["grade_iv"] == 1
            assert result["csf_reference_signal"] == 850.5

    def test_grade_v_collapsed_disc(self):
        """Test Grade V detection for collapsed disc."""
        mock_result = {
            "success": True,
            "tool": "assess_disc_degeneration_mri",
            "region": "lumbar",
            "csf_reference_signal": 900.0,
            "discs": [
                {
                    "disc_level": "L5-S1",
                    "pfirrmann_grade": 5,
                    "pfirrmann_description": (
                        "Inhomogeneous black signal, collapsed disc space, lost boundary"
                    ),
                    "signal_ratio_to_csf": 0.08,
                    "homogeneity_cv": 0.45,
                    "is_homogeneous": False,
                    "has_nucleus_annulus_distinction": False,
                    "disc_height_mm": 2.1,
                    "height_loss_percent": 72.0,
                },
            ],
            "total_discs_analyzed": 1,
            "grade_summary": {
                "grade_i": 0,
                "grade_ii": 0,
                "grade_iii": 0,
                "grade_iv": 0,
                "grade_v": 1,
            },
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = assess_disc_degeneration_mri("vtkMRMLScalarVolumeNode1")

            assert result["discs"][0]["pfirrmann_grade"] == 5
            assert result["discs"][0]["height_loss_percent"] >= 60

    def test_connection_error_propagates(self):
        """Test SlicerConnectionError propagates."""
        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                assess_disc_degeneration_mri("vtkMRMLScalarVolumeNode1")

    def test_cervical_region(self):
        """Test cervical region is accepted."""
        mock_result = {
            "success": True,
            "tool": "assess_disc_degeneration_mri",
            "region": "cervical",
            "csf_reference_signal": 800.0,
            "discs": [],
            "total_discs_analyzed": 0,
            "grade_summary": {
                "grade_i": 0,
                "grade_ii": 0,
                "grade_iii": 0,
                "grade_iv": 0,
                "grade_v": 0,
            },
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = assess_disc_degeneration_mri("vtkMRMLScalarVolumeNode1", region="cervical")
            assert result["region"] == "cervical"


# =============================================================================
# Tool 3: Cord Compression Detection Tests
# =============================================================================


class TestDetectCordCompressionMri:
    """Test detect_cord_compression_mri tool."""

    def test_invalid_t2_node_id(self):
        """Test invalid T2 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            detect_cord_compression_mri("")
        assert exc_info.value.field == "node_id"

    def test_invalid_t1_node_id(self):
        """Test invalid optional T1 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            detect_cord_compression_mri("vtkMRMLScalarVolumeNode1", t1_node_id="1invalid")
        assert exc_info.value.field == "node_id"

    def test_invalid_region(self):
        """Test invalid region raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            detect_cord_compression_mri("vtkMRMLScalarVolumeNode1", region="invalid")
        assert exc_info.value.field == "region"

    def test_none_t1_is_accepted(self):
        """Test None T1 node ID is accepted (optional parameter)."""
        mock_result = {
            "success": True,
            "tool": "detect_cord_compression_mri",
            "region": "cervical",
            "t1_available": False,
            "registration_performed": False,
            "canal_normal_range_mm": [14.0, 23.0],
            "stenosis_threshold_mm": 10.0,
            "levels": [],
            "total_levels_analyzed": 0,
            "worst_stenosis_grade": "normal",
            "myelopathy_detected": False,
            "mscc_significant": False,
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = detect_cord_compression_mri("vtkMRMLScalarVolumeNode1", t1_node_id=None)
            assert result["success"] is True
            assert result["t1_available"] is False

    def test_successful_cord_compression_detection(self):
        """Test successful cord compression detection returns expected structure."""
        mock_result = {
            "success": True,
            "tool": "detect_cord_compression_mri",
            "region": "cervical",
            "t1_available": True,
            "registration_performed": False,
            "canal_normal_range_mm": [14.0, 23.0],
            "stenosis_threshold_mm": 10.0,
            "levels": [
                {
                    "level": "C5",
                    "ap_diameter_mm": 8.5,
                    "transverse_diameter_mm": 14.2,
                    "cross_section_area_mm2": 95.0,
                    "compression_ratio": 0.599,
                    "mscc_ratio": 0.459,
                    "stenosis_grade": "severe",
                    "t2_hyperintensity": True,
                    "hyperintensity_ratio": 1.45,
                    "myelopathy_reversibility": "likely_irreversible",
                },
                {
                    "level": "C6",
                    "ap_diameter_mm": 12.0,
                    "transverse_diameter_mm": 15.0,
                    "cross_section_area_mm2": 141.4,
                    "compression_ratio": 0.800,
                    "mscc_ratio": 0.649,
                    "stenosis_grade": "moderate",
                    "t2_hyperintensity": False,
                },
            ],
            "total_levels_analyzed": 2,
            "worst_stenosis_grade": "severe",
            "myelopathy_detected": True,
            "mscc_significant": True,
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = detect_cord_compression_mri(
                "vtkMRMLScalarVolumeNode1",
                t1_node_id="vtkMRMLScalarVolumeNode2",
                region="cervical",
            )

            assert result["success"] is True
            assert result["t1_available"] is True
            assert result["worst_stenosis_grade"] == "severe"
            assert result["myelopathy_detected"] is True
            assert result["mscc_significant"] is True
            assert len(result["levels"]) == 2
            assert result["levels"][0]["stenosis_grade"] == "severe"
            assert result["levels"][0]["t2_hyperintensity"] is True
            assert result["levels"][0]["myelopathy_reversibility"] == "likely_irreversible"

    def test_myelopathy_with_t1_reversibility(self):
        """Test T1-based myelopathy reversibility assessment."""
        mock_result = {
            "success": True,
            "tool": "detect_cord_compression_mri",
            "region": "cervical",
            "t1_available": True,
            "registration_performed": True,
            "canal_normal_range_mm": [14.0, 23.0],
            "stenosis_threshold_mm": 10.0,
            "levels": [
                {
                    "level": "C4",
                    "ap_diameter_mm": 9.0,
                    "transverse_diameter_mm": 13.5,
                    "cross_section_area_mm2": 95.4,
                    "compression_ratio": 0.667,
                    "mscc_ratio": 0.486,
                    "stenosis_grade": "severe",
                    "t2_hyperintensity": True,
                    "hyperintensity_ratio": 1.35,
                    "myelopathy_reversibility": "likely_reversible",
                }
            ],
            "total_levels_analyzed": 1,
            "worst_stenosis_grade": "severe",
            "myelopathy_detected": True,
            "mscc_significant": True,
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = detect_cord_compression_mri(
                "vtkMRMLScalarVolumeNode1",
                t1_node_id="vtkMRMLScalarVolumeNode2",
            )

            assert result["levels"][0]["myelopathy_reversibility"] == "likely_reversible"

    def test_connection_error_propagates(self):
        """Test SlicerConnectionError propagates."""
        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                detect_cord_compression_mri("vtkMRMLScalarVolumeNode1")


# =============================================================================
# Tool 4: Metastatic Lesion Detection Tests
# =============================================================================


class TestDetectMetastaticLesionsMri:
    """Test detect_metastatic_lesions_mri tool."""

    def test_invalid_t1_node_id(self):
        """Test invalid T1 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            detect_metastatic_lesions_mri("", "vtkMRMLScalarVolumeNode2")
        assert exc_info.value.field == "node_id"

    def test_invalid_t2_stir_node_id(self):
        """Test invalid T2/STIR node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            detect_metastatic_lesions_mri("vtkMRMLScalarVolumeNode1", "1invalid")
        assert exc_info.value.field == "node_id"

    def test_invalid_region(self):
        """Test invalid region raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            detect_metastatic_lesions_mri(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                region="invalid",
            )
        assert exc_info.value.field == "region"

    def test_full_region_accepted(self):
        """Test 'full' region is accepted for metastasis detection."""
        mock_result = {
            "success": True,
            "tool": "detect_metastatic_lesions_mri",
            "region": "full",
            "registration_performed": False,
            "reference_t1_signal": 500.0,
            "reference_t2_stir_signal": 300.0,
            "vertebra_results": [],
            "total_vertebrae_analyzed": 0,
            "suspicious_lesions": [],
            "total_suspicious": 0,
            "lesion_type_summary": {"lytic": 0, "blastic": 0, "mixed": 0},
            "posterior_element_involvement": False,
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = detect_metastatic_lesions_mri(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                region="full",
            )
            assert result["region"] == "full"

    def test_successful_lytic_lesion_detection(self):
        """Test detection of lytic metastatic lesion (T1 low, T2/STIR high)."""
        mock_result = {
            "success": True,
            "tool": "detect_metastatic_lesions_mri",
            "region": "thoracic",
            "registration_performed": True,
            "reference_t1_signal": 520.0,
            "reference_t2_stir_signal": 310.0,
            "vertebra_results": [
                {
                    "vertebra": "T7",
                    "t1_signal_ratio": 0.55,
                    "t2_stir_signal_ratio": 1.65,
                    "suspicious": True,
                    "lesion_type": "lytic",
                    "posterior_element_involved": False,
                    "fracture_features": ["heterogeneous_signal"],
                },
                {
                    "vertebra": "T8",
                    "t1_signal_ratio": 0.98,
                    "t2_stir_signal_ratio": 1.02,
                    "suspicious": False,
                    "lesion_type": None,
                    "posterior_element_involved": False,
                    "fracture_features": [],
                },
            ],
            "total_vertebrae_analyzed": 2,
            "suspicious_lesions": [
                {
                    "vertebra": "T7",
                    "t1_signal_ratio": 0.55,
                    "t2_stir_signal_ratio": 1.65,
                    "suspicious": True,
                    "lesion_type": "lytic",
                    "posterior_element_involved": False,
                    "fracture_features": ["heterogeneous_signal"],
                }
            ],
            "total_suspicious": 1,
            "lesion_type_summary": {"lytic": 1, "blastic": 0, "mixed": 0},
            "posterior_element_involvement": False,
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = detect_metastatic_lesions_mri(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                region="thoracic",
            )

            assert result["success"] is True
            assert result["total_suspicious"] == 1
            assert result["suspicious_lesions"][0]["lesion_type"] == "lytic"
            assert result["lesion_type_summary"]["lytic"] == 1
            assert result["vertebra_results"][1]["suspicious"] is False

    def test_blastic_lesion_detection(self):
        """Test detection of blastic metastatic lesion (T1 low, T2/STIR low)."""
        mock_result = {
            "success": True,
            "tool": "detect_metastatic_lesions_mri",
            "region": "lumbar",
            "registration_performed": False,
            "reference_t1_signal": 500.0,
            "reference_t2_stir_signal": 300.0,
            "vertebra_results": [
                {
                    "vertebra": "L2",
                    "t1_signal_ratio": 0.60,
                    "t2_stir_signal_ratio": 0.55,
                    "suspicious": True,
                    "lesion_type": "blastic",
                    "posterior_element_involved": True,
                    "fracture_features": [
                        "posterior_element_involvement",
                        "heterogeneous_signal",
                    ],
                }
            ],
            "total_vertebrae_analyzed": 1,
            "suspicious_lesions": [
                {
                    "vertebra": "L2",
                    "t1_signal_ratio": 0.60,
                    "t2_stir_signal_ratio": 0.55,
                    "suspicious": True,
                    "lesion_type": "blastic",
                    "posterior_element_involved": True,
                    "fracture_features": [
                        "posterior_element_involvement",
                        "heterogeneous_signal",
                    ],
                }
            ],
            "total_suspicious": 1,
            "lesion_type_summary": {"lytic": 0, "blastic": 1, "mixed": 0},
            "posterior_element_involvement": True,
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = detect_metastatic_lesions_mri(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                region="lumbar",
            )

            assert result["suspicious_lesions"][0]["lesion_type"] == "blastic"
            assert result["posterior_element_involvement"] is True
            assert "posterior_element_involvement" in (
                result["suspicious_lesions"][0]["fracture_features"]
            )

    def test_mixed_lesion_detection(self):
        """Test detection of mixed metastatic lesion."""
        mock_result = {
            "success": True,
            "tool": "detect_metastatic_lesions_mri",
            "region": "lumbar",
            "registration_performed": False,
            "reference_t1_signal": 500.0,
            "reference_t2_stir_signal": 300.0,
            "vertebra_results": [
                {
                    "vertebra": "L3",
                    "t1_signal_ratio": 0.65,
                    "t2_stir_signal_ratio": 1.10,
                    "suspicious": True,
                    "lesion_type": "mixed",
                    "posterior_element_involved": False,
                    "fracture_features": [],
                }
            ],
            "total_vertebrae_analyzed": 1,
            "suspicious_lesions": [
                {
                    "vertebra": "L3",
                    "t1_signal_ratio": 0.65,
                    "t2_stir_signal_ratio": 1.10,
                    "suspicious": True,
                    "lesion_type": "mixed",
                    "posterior_element_involved": False,
                    "fracture_features": [],
                }
            ],
            "total_suspicious": 1,
            "lesion_type_summary": {"lytic": 0, "blastic": 0, "mixed": 1},
            "posterior_element_involvement": False,
        }

        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": json.dumps(mock_result),
            }
            mock_get_client.return_value = mock_client

            result = detect_metastatic_lesions_mri(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                region="lumbar",
            )

            assert result["suspicious_lesions"][0]["lesion_type"] == "mixed"
            assert result["lesion_type_summary"]["mixed"] == 1

    def test_connection_error_propagates(self):
        """Test SlicerConnectionError propagates."""
        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                detect_metastatic_lesions_mri(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLScalarVolumeNode2",
                )

    def test_malformed_json_raises_error(self):
        """Test malformed JSON result raises SlicerConnectionError."""
        with patch("slicer_mcp.diagnostic_tools_mri.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "invalid json",
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError) as exc_info:
                detect_metastatic_lesions_mri(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLScalarVolumeNode2",
                )
            assert "Failed to parse" in str(exc_info.value)


# =============================================================================
# Server Registration Tests
# =============================================================================


class TestServerRegistration:
    """Test that MRI diagnostic tools are registered in server.py."""

    def test_classify_modic_changes_registered(self):
        """Test classify_modic_changes is registered as MCP tool."""
        from slicer_mcp import server

        assert hasattr(server, "classify_modic_changes")

    def test_assess_disc_degeneration_mri_registered(self):
        """Test assess_disc_degeneration_mri is registered as MCP tool."""
        from slicer_mcp import server

        assert hasattr(server, "assess_disc_degeneration_mri")

    def test_detect_cord_compression_mri_registered(self):
        """Test detect_cord_compression_mri is registered as MCP tool."""
        from slicer_mcp import server

        assert hasattr(server, "detect_cord_compression_mri")

    def test_detect_metastatic_lesions_mri_registered(self):
        """Test detect_metastatic_lesions_mri is registered as MCP tool."""
        from slicer_mcp import server

        assert hasattr(server, "detect_metastatic_lesions_mri")

    def test_server_error_handling_modic(self):
        """Test server wraps classify_modic_changes errors."""
        from slicer_mcp.server import classify_modic_changes as server_classify

        with patch("slicer_mcp.diagnostic_tools_mri.classify_modic_changes") as mock_tool:
            mock_tool.side_effect = RuntimeError("unexpected")
            result = server_classify("vtkMRMLScalarVolumeNode1", "vtkMRMLScalarVolumeNode2")
            assert result["success"] is False
            assert result["error_type"] == "unexpected"

    def test_server_error_handling_pfirrmann(self):
        """Test server wraps assess_disc_degeneration_mri errors."""
        from slicer_mcp.server import (
            assess_disc_degeneration_mri as server_assess,
        )

        with patch("slicer_mcp.diagnostic_tools_mri.assess_disc_degeneration_mri") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Connection refused")
            result = server_assess("vtkMRMLScalarVolumeNode1")
            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_server_error_handling_cord(self):
        """Test server wraps detect_cord_compression_mri errors."""
        from slicer_mcp.server import (
            detect_cord_compression_mri as server_detect,
        )

        with patch("slicer_mcp.diagnostic_tools_mri.detect_cord_compression_mri") as mock_tool:
            mock_tool.side_effect = SlicerConnectionError("Timeout")
            result = server_detect("vtkMRMLScalarVolumeNode1")
            assert result["success"] is False
            assert result["error_type"] == "connection"

    def test_server_error_handling_metastasis(self):
        """Test server wraps detect_metastatic_lesions_mri errors."""
        from slicer_mcp.server import (
            detect_metastatic_lesions_mri as server_detect,
        )

        with patch("slicer_mcp.diagnostic_tools_mri.detect_metastatic_lesions_mri") as mock_tool:
            mock_tool.side_effect = RuntimeError("test error")
            result = server_detect("vtkMRMLScalarVolumeNode1", "vtkMRMLScalarVolumeNode2")
            assert result["success"] is False
            assert result["error_type"] == "unexpected"
