"""Unit tests for cervical screw instrumentation planning tools."""

import json
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.instrumentation_tools import (
    _TECHNIQUE_SCREW_DEFAULTS,
    _validate_level,
    _validate_screw_dimensions,
    _validate_side,
    _validate_technique,
    _validate_variant,
    plan_cervical_screws,
)
from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.spine_constants import (
    C1_LATERAL_MASS_SCREW_DEFAULTS,
    C2_PARS_SCREW_DEFAULTS,
    CERVICAL_LATERAL_MASS_SCREW_DEFAULTS,
    CERVICAL_PEDICLE_SCREW_DEFAULTS,
    INSTRUMENTATION_TIMEOUT,
    OCCIPITAL_SCREW_DEFAULTS,
    TECHNIQUE_ANGULATION,
    TECHNIQUE_VALID_LEVELS,
    TRANSARTICULAR_SCREW_DEFAULTS,
    VALID_INSTRUMENTATION_TECHNIQUES,
    VALID_LATERAL_MASS_VARIANTS,
    VALID_SIDES,
)
from slicer_mcp.tools import ValidationError

# =============================================================================
# Technique Validation
# =============================================================================


class TestValidateTechnique:
    """Test _validate_technique function."""

    def test_valid_pedicle(self):
        assert _validate_technique("pedicle") == "pedicle"

    def test_valid_lateral_mass(self):
        assert _validate_technique("lateral_mass") == "lateral_mass"

    def test_valid_transarticular(self):
        assert _validate_technique("transarticular") == "transarticular"

    def test_valid_c1_lateral_mass(self):
        assert _validate_technique("c1_lateral_mass") == "c1_lateral_mass"

    def test_valid_c2_pars(self):
        assert _validate_technique("c2_pars") == "c2_pars"

    def test_valid_occipital(self):
        assert _validate_technique("occipital") == "occipital"

    def test_valid_auto(self):
        assert _validate_technique("auto") == "auto"

    def test_case_insensitive(self):
        assert _validate_technique("PEDICLE") == "pedicle"

    def test_strips_whitespace(self):
        assert _validate_technique("  pedicle  ") == "pedicle"

    def test_invalid_technique(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_technique("invalid_technique")
        assert exc_info.value.field == "technique"

    def test_empty_technique(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_technique("")
        assert exc_info.value.field == "technique"
        assert "cannot be empty" in str(exc_info.value)

    def test_all_valid_techniques_accepted(self):
        for technique in VALID_INSTRUMENTATION_TECHNIQUES:
            assert _validate_technique(technique) == technique


# =============================================================================
# Level Validation
# =============================================================================


class TestValidateLevel:
    """Test _validate_level function."""

    def test_pedicle_valid_c2(self):
        assert _validate_level("pedicle", "C2") == "C2"

    def test_pedicle_valid_c7(self):
        assert _validate_level("pedicle", "C7") == "C7"

    def test_pedicle_invalid_c1(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_level("pedicle", "C1")
        assert exc_info.value.field == "level"

    def test_lateral_mass_valid_c4(self):
        assert _validate_level("lateral_mass", "C4") == "C4"

    def test_lateral_mass_invalid_c2(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_level("lateral_mass", "C2")
        assert exc_info.value.field == "level"

    def test_transarticular_valid_c1c2(self):
        assert _validate_level("transarticular", "C1C2") == "C1C2"

    def test_transarticular_invalid_c3(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_level("transarticular", "C3")
        assert exc_info.value.field == "level"

    def test_c1_lateral_mass_valid_c1(self):
        assert _validate_level("c1_lateral_mass", "C1") == "C1"

    def test_c2_pars_valid_c2(self):
        assert _validate_level("c2_pars", "C2") == "C2"

    def test_occipital_valid(self):
        assert _validate_level("occipital", "Occiput") == "Occiput"

    def test_auto_accepts_any_level(self):
        assert _validate_level("auto", "C5") == "C5"
        assert _validate_level("auto", "Occiput") == "Occiput"

    def test_empty_level(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_level("pedicle", "")
        assert exc_info.value.field == "level"

    def test_strips_whitespace(self):
        assert _validate_level("pedicle", " C5 ") == "C5"

    def test_all_technique_levels_valid(self):
        for technique, levels in TECHNIQUE_VALID_LEVELS.items():
            for level in levels:
                assert _validate_level(technique, level) == level


# =============================================================================
# Side Validation
# =============================================================================


class TestValidateSide:
    """Test _validate_side function."""

    def test_valid_left(self):
        assert _validate_side("left") == "left"

    def test_valid_right(self):
        assert _validate_side("right") == "right"

    def test_valid_bilateral(self):
        assert _validate_side("bilateral") == "bilateral"

    def test_case_insensitive(self):
        assert _validate_side("LEFT") == "left"

    def test_strips_whitespace(self):
        assert _validate_side("  bilateral  ") == "bilateral"

    def test_invalid_side(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_side("both")
        assert exc_info.value.field == "side"

    def test_empty_side(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_side("")
        assert exc_info.value.field == "side"


# =============================================================================
# Variant Validation
# =============================================================================


class TestValidateVariant:
    """Test _validate_variant function."""

    def test_none_for_non_lateral_mass(self):
        assert _validate_variant(None, "pedicle") is None

    def test_none_defaults_to_magerl_for_lateral_mass(self):
        assert _validate_variant(None, "lateral_mass") == "magerl"

    def test_valid_roy_camille(self):
        assert _validate_variant("roy_camille", "lateral_mass") == "roy_camille"

    def test_valid_magerl(self):
        assert _validate_variant("magerl", "lateral_mass") == "magerl"

    def test_valid_an(self):
        assert _validate_variant("an", "lateral_mass") == "an"

    def test_valid_anderson(self):
        assert _validate_variant("anderson", "lateral_mass") == "anderson"

    def test_case_insensitive(self):
        assert _validate_variant("MAGERL", "lateral_mass") == "magerl"

    def test_variant_on_non_lateral_mass_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_variant("magerl", "pedicle")
        assert exc_info.value.field == "variant"

    def test_invalid_variant(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_variant("unknown", "lateral_mass")
        assert exc_info.value.field == "variant"

    def test_all_valid_variants_accepted(self):
        for variant in VALID_LATERAL_MASS_VARIANTS:
            assert _validate_variant(variant, "lateral_mass") == variant


# =============================================================================
# Screw Dimension Validation
# =============================================================================


class TestValidateScrewDimensions:
    """Test _validate_screw_dimensions function."""

    def test_defaults_for_pedicle(self):
        diam, length = _validate_screw_dimensions("pedicle", None, None)
        assert diam == CERVICAL_PEDICLE_SCREW_DEFAULTS["diameter_mm"]
        assert length == CERVICAL_PEDICLE_SCREW_DEFAULTS["length_mm"]

    def test_defaults_for_lateral_mass(self):
        diam, length = _validate_screw_dimensions("lateral_mass", None, None)
        assert diam == CERVICAL_LATERAL_MASS_SCREW_DEFAULTS["diameter_mm"]
        assert length == CERVICAL_LATERAL_MASS_SCREW_DEFAULTS["length_mm"]

    def test_defaults_for_transarticular(self):
        diam, length = _validate_screw_dimensions("transarticular", None, None)
        assert diam == TRANSARTICULAR_SCREW_DEFAULTS["diameter_mm"]
        assert length == TRANSARTICULAR_SCREW_DEFAULTS["length_mm"]

    def test_defaults_for_c1_lateral_mass(self):
        diam, length = _validate_screw_dimensions("c1_lateral_mass", None, None)
        assert diam == C1_LATERAL_MASS_SCREW_DEFAULTS["diameter_mm"]
        assert length == C1_LATERAL_MASS_SCREW_DEFAULTS["length_mm"]

    def test_defaults_for_c2_pars(self):
        diam, length = _validate_screw_dimensions("c2_pars", None, None)
        assert diam == C2_PARS_SCREW_DEFAULTS["diameter_mm"]
        assert length == C2_PARS_SCREW_DEFAULTS["length_mm"]

    def test_defaults_for_occipital(self):
        diam, length = _validate_screw_dimensions("occipital", None, None)
        assert diam == OCCIPITAL_SCREW_DEFAULTS["diameter_mm"]
        assert length == OCCIPITAL_SCREW_DEFAULTS["length_mm"]

    def test_override_diameter(self):
        diam, length = _validate_screw_dimensions("pedicle", 4.0, None)
        assert diam == 4.0
        assert length == CERVICAL_PEDICLE_SCREW_DEFAULTS["length_mm"]

    def test_override_length(self):
        diam, length = _validate_screw_dimensions("pedicle", None, 20.0)
        assert diam == CERVICAL_PEDICLE_SCREW_DEFAULTS["diameter_mm"]
        assert length == 20.0

    def test_invalid_diameter_negative(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_screw_dimensions("pedicle", -1.0, None)
        assert exc_info.value.field == "screw_diameter_mm"

    def test_invalid_diameter_too_large(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_screw_dimensions("pedicle", 10.0, None)
        assert exc_info.value.field == "screw_diameter_mm"

    def test_invalid_length_negative(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_screw_dimensions("pedicle", None, -5.0)
        assert exc_info.value.field == "screw_length_mm"

    def test_invalid_length_too_short_for_technique(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_screw_dimensions("pedicle", None, 5.0)
        assert exc_info.value.field == "screw_length_mm"
        assert "typical range" in str(exc_info.value)

    def test_invalid_length_too_long_for_technique(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_screw_dimensions("pedicle", None, 50.0)
        assert exc_info.value.field == "screw_length_mm"


# =============================================================================
# plan_cervical_screws — Input Validation
# =============================================================================


class TestPlanCervicalScrewsValidation:
    """Test plan_cervical_screws input validation (no Slicer needed)."""

    def test_invalid_technique(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws("bogus", "C5", "vtkMRMLSegmentationNode1")
        assert exc_info.value.field == "technique"

    def test_invalid_level_for_technique(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws("pedicle", "C1", "vtkMRMLSegmentationNode1")
        assert exc_info.value.field == "level"

    def test_invalid_segmentation_node_id(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws("pedicle", "C5", "'; DROP TABLE;")
        assert exc_info.value.field == "node_id"

    def test_invalid_side(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1", side="middle")
        assert exc_info.value.field == "side"

    def test_transarticular_requires_va(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws("transarticular", "C1C2", "vtkMRMLSegmentationNode1")
        assert exc_info.value.field == "va_node_id"
        assert "REQUIRED" in str(exc_info.value)

    def test_variant_only_for_lateral_mass(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1", variant="magerl")
        assert exc_info.value.field == "variant"

    def test_invalid_va_node_id(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws(
                "pedicle", "C5", "vtkMRMLSegmentationNode1", va_node_id="'; malicious"
            )
        assert exc_info.value.field == "node_id"


# =============================================================================
# plan_cervical_screws — Code Generation and Execution
# =============================================================================


def _mock_exec_result(result_dict: dict) -> dict:
    """Create a mock exec_python return value."""
    return {
        "success": True,
        "result": json.dumps(result_dict),
    }


class TestPlanCervicalScrewsPedicle:
    """Test pedicle screw planning code generation."""

    def test_pedicle_c5_bilateral(self):
        mock_result = {
            "success": True,
            "technique": "pedicle",
            "technique_name": "Cervical Pedicle Screw",
            "reference": "Abumi K et al. Spine 1994",
            "level": "C5",
            "segment_name": "vertebrae_C5",
            "segment_geometry": {
                "centroid_ras": [0.0, 0.0, 100.0],
                "dimensions_mm": [30.0, 20.0, 15.0],
            },
            "screws": [
                {
                    "side": "left",
                    "entry_point_ras": [-4.5, -4.0, 100.0],
                    "target_point_ras": [5.0, 14.0, 100.0],
                    "screw_diameter_mm": 3.5,
                    "screw_length_mm": 22.0,
                    "angulation": {"medial_deg": 25.0, "caudal_deg": 0.0},
                    "va_assessment": {"status": "not_assessed", "min_distance_mm": None},
                    "visualization": {
                        "markup_node_id": "vtkMRMLMarkupsLineNode1",
                        "model_node_id": "vtkMRMLModelNode1",
                    },
                },
                {
                    "side": "right",
                    "entry_point_ras": [4.5, -4.0, 100.0],
                    "target_point_ras": [-5.0, 14.0, 100.0],
                    "screw_diameter_mm": 3.5,
                    "screw_length_mm": 22.0,
                    "angulation": {"medial_deg": 25.0, "caudal_deg": 0.0},
                    "va_assessment": {"status": "not_assessed", "min_distance_mm": None},
                    "visualization": {
                        "markup_node_id": "vtkMRMLMarkupsLineNode2",
                        "model_node_id": "vtkMRMLModelNode2",
                    },
                },
            ],
            "warnings": [],
            "recommendations": [],
        }

        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")

            assert result["success"] is True
            assert result["technique"] == "pedicle"
            assert result["level"] == "C5"
            assert len(result["screws"]) == 2
            mock_client.exec_python.assert_called_once()

    def test_pedicle_code_uses_json_escaped_ids(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "pedicle",
                    "level": "C5",
                    "screws": [],
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")

            python_code = mock_client.exec_python.call_args[0][0]
            assert '"vtkMRMLSegmentationNode1"' in python_code
            assert '"C5"' in python_code

    def test_pedicle_uses_instrumentation_timeout(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "pedicle",
                    "level": "C5",
                    "screws": [],
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")

            _, kwargs = mock_client.exec_python.call_args
            assert kwargs.get("timeout") == INSTRUMENTATION_TIMEOUT

    def test_pedicle_left_side_only(self):
        mock_result = {
            "success": True,
            "technique": "pedicle",
            "level": "C5",
            "screws": [{"side": "left"}],
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1", side="left")
            python_code = mock_client.exec_python.call_args[0][0]
            assert '"left"' in python_code


class TestPlanCervicalScrewsLateralMass:
    """Test lateral mass screw planning."""

    def test_lateral_mass_magerl_default(self):
        mock_result = {
            "success": True,
            "technique": "lateral_mass",
            "technique_name": "Lateral Mass Screw (Magerl)",
            "level": "C4",
            "screws": [],
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("lateral_mass", "C4", "vtkMRMLSegmentationNode1")
            assert result["technique"] == "lateral_mass"

    def test_lateral_mass_roy_camille_variant(self):
        mock_result = {
            "success": True,
            "technique": "lateral_mass",
            "level": "C5",
            "screws": [{"variant": "roy_camille"}],
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            plan_cervical_screws(
                "lateral_mass",
                "C5",
                "vtkMRMLSegmentationNode1",
                variant="roy_camille",
            )
            python_code = mock_client.exec_python.call_args[0][0]
            assert "Roy-Camille" in python_code

    def test_lateral_mass_an_variant(self):
        mock_result = {
            "success": True,
            "technique": "lateral_mass",
            "level": "C6",
            "screws": [],
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            plan_cervical_screws("lateral_mass", "C6", "vtkMRMLSegmentationNode1", variant="an")
            python_code = mock_client.exec_python.call_args[0][0]
            assert "An" in python_code

    def test_lateral_mass_all_valid_levels(self):
        for level in TECHNIQUE_VALID_LEVELS["lateral_mass"]:
            mock_result = {
                "success": True,
                "technique": "lateral_mass",
                "level": level,
                "screws": [],
                "warnings": [],
                "recommendations": [],
            }
            with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
                mock_client = Mock()
                mock_client.exec_python.return_value = _mock_exec_result(mock_result)
                mock_get_client.return_value = mock_client

                result = plan_cervical_screws("lateral_mass", level, "vtkMRMLSegmentationNode1")
                assert result["technique"] == "lateral_mass"


class TestPlanCervicalScrewsTransarticular:
    """Test transarticular screw planning."""

    def test_transarticular_with_va(self):
        mock_result = {
            "success": True,
            "technique": "transarticular",
            "technique_name": "C1-C2 Transarticular Screw (Magerl)",
            "level": "C1C2",
            "screws": [],
            "isthmus_analysis": {"height_mm": 7.0, "width_mm": 5.0, "adequate": True},
            "blocked": False,
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws(
                "transarticular",
                "C1C2",
                "vtkMRMLSegmentationNode1",
                va_node_id="vtkMRMLSegmentationNode2",
            )
            assert result["technique"] == "transarticular"
            assert result["blocked"] is False

    def test_transarticular_without_va_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            plan_cervical_screws("transarticular", "C1C2", "vtkMRMLSegmentationNode1")
        assert "REQUIRED" in str(exc_info.value)
        assert exc_info.value.field == "va_node_id"

    def test_transarticular_code_includes_isthmus_check(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "transarticular",
                    "level": "C1C2",
                    "screws": [],
                    "isthmus_analysis": {},
                    "blocked": False,
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws(
                "transarticular",
                "C1C2",
                "vtkMRMLSegmentationNode1",
                va_node_id="vtkMRMLSegmentationNode2",
            )
            python_code = mock_client.exec_python.call_args[0][0]
            assert "isthmus" in python_code.lower()
            assert "Magerl" in python_code


class TestPlanCervicalScrewsC1LateralMass:
    """Test C1 lateral mass (Harms/Goel) screw planning."""

    def test_c1_lateral_mass_basic(self):
        mock_result = {
            "success": True,
            "technique": "c1_lateral_mass",
            "technique_name": "C1 Lateral Mass Screw (Harms/Goel)",
            "level": "C1",
            "screws": [],
            "warnings": [],
            "recommendations": ["Consider C2 pars screw as complement"],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("c1_lateral_mass", "C1", "vtkMRMLSegmentationNode1")
            assert result["technique"] == "c1_lateral_mass"

    def test_c1_lateral_mass_code_suggests_c2_complement(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "c1_lateral_mass",
                    "level": "C1",
                    "screws": [],
                    "warnings": [],
                    "recommendations": ["Consider C2 pars"],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws("c1_lateral_mass", "C1", "vtkMRMLSegmentationNode1")
            python_code = mock_client.exec_python.call_args[0][0]
            assert "C2 pars" in python_code

    def test_c1_lateral_mass_with_va(self):
        mock_result = {
            "success": True,
            "technique": "c1_lateral_mass",
            "level": "C1",
            "screws": [],
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            plan_cervical_screws(
                "c1_lateral_mass",
                "C1",
                "vtkMRMLSegmentationNode1",
                va_node_id="vtkMRMLSegmentationNode2",
            )
            python_code = mock_client.exec_python.call_args[0][0]
            assert '"vtkMRMLSegmentationNode2"' in python_code


class TestPlanCervicalScrewsC2Pars:
    """Test C2 pars interarticularis screw planning."""

    def test_c2_pars_basic(self):
        mock_result = {
            "success": True,
            "technique": "c2_pars",
            "technique_name": "C2 Pars Interarticularis Screw",
            "level": "C2",
            "screws": [],
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("c2_pars", "C2", "vtkMRMLSegmentationNode1")
            assert result["technique"] == "c2_pars"

    def test_c2_pars_code_includes_c2_level(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "c2_pars",
                    "level": "C2",
                    "screws": [],
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws("c2_pars", "C2", "vtkMRMLSegmentationNode1")
            python_code = mock_client.exec_python.call_args[0][0]
            assert "C2" in python_code
            assert "Pars" in python_code


class TestPlanCervicalScrewsOccipital:
    """Test occipital screw planning."""

    def test_occipital_basic(self):
        mock_result = {
            "success": True,
            "technique": "occipital",
            "technique_name": "Occipital Screw",
            "level": "Occiput",
            "thickness_map": {},
            "screws": [],
            "warnings": [],
            "recommendations": [],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("occipital", "Occiput", "vtkMRMLSegmentationNode1")
            assert result["technique"] == "occipital"

    def test_occipital_code_includes_thickness_mapping(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "occipital",
                    "level": "Occiput",
                    "thickness_map": {},
                    "screws": [],
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws("occipital", "Occiput", "vtkMRMLSegmentationNode1")
            python_code = mock_client.exec_python.call_args[0][0]
            assert "thickness" in python_code.lower()
            assert "keel" in python_code.lower()

    def test_occipital_no_va_needed(self):
        """Occipital technique should not require VA node."""
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "occipital",
                    "level": "Occiput",
                    "thickness_map": {},
                    "screws": [],
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("occipital", "Occiput", "vtkMRMLSegmentationNode1")
            assert result["success"] is True


class TestPlanCervicalScrewsAutoMode:
    """Test auto technique analysis mode."""

    def test_auto_c5(self):
        mock_result = {
            "success": True,
            "technique": "auto",
            "level": "C5",
            "analysis": {"pedicle_width_mm": 5.0},
            "recommendations": [
                {"technique": "lateral_mass", "confidence": "high"},
                {"technique": "pedicle", "confidence": "medium"},
            ],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("auto", "C5", "vtkMRMLSegmentationNode1")
            assert result["technique"] == "auto"
            assert len(result["recommendations"]) > 0

    def test_auto_c1c2(self):
        mock_result = {
            "success": True,
            "technique": "auto",
            "level": "C1C2",
            "analysis": {"isthmus_height_mm": 6.0, "isthmus_width_mm": 5.0},
            "recommendations": [
                {"technique": "transarticular", "confidence": "medium"},
                {"technique": "c1_lateral_mass", "confidence": "high"},
            ],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("auto", "C1C2", "vtkMRMLSegmentationNode1")
            assert result["technique"] == "auto"

    def test_auto_occiput(self):
        mock_result = {
            "success": True,
            "technique": "auto",
            "level": "Occiput",
            "analysis": {},
            "recommendations": [
                {"technique": "occipital", "confidence": "high"},
            ],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("auto", "Occiput", "vtkMRMLSegmentationNode1")
            assert result["recommendations"][0]["technique"] == "occipital"

    def test_auto_c7_prefers_pedicle(self):
        mock_result = {
            "success": True,
            "technique": "auto",
            "level": "C7",
            "analysis": {},
            "recommendations": [
                {"technique": "pedicle", "confidence": "high"},
                {"technique": "lateral_mass", "confidence": "medium"},
            ],
        }
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(mock_result)
            mock_get_client.return_value = mock_client

            result = plan_cervical_screws("auto", "C7", "vtkMRMLSegmentationNode1")
            assert result["recommendations"][0]["technique"] == "pedicle"


# =============================================================================
# Error Handling
# =============================================================================


class TestPlanCervicalScrewsErrorHandling:
    """Test error handling in plan_cervical_screws."""

    def test_slicer_connection_error_propagates(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")

    def test_empty_result_raises(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {"success": True, "result": ""}
            mock_get_client.return_value = mock_client

            from slicer_mcp.slicer_client import SlicerConnectionError

            with pytest.raises(SlicerConnectionError):
                plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")

    def test_malformed_json_raises(self):
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "not valid json{{{",
            }
            mock_get_client.return_value = mock_client

            from slicer_mcp.slicer_client import SlicerConnectionError

            with pytest.raises(SlicerConnectionError):
                plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")


# =============================================================================
# Constants Integrity
# =============================================================================


class TestInstrumentationConstants:
    """Test that instrumentation constants are properly defined."""

    def test_all_techniques_have_valid_levels(self):
        for technique in VALID_INSTRUMENTATION_TECHNIQUES:
            if technique == "auto":
                continue
            assert technique in TECHNIQUE_VALID_LEVELS

    def test_all_technique_angulations_defined(self):
        expected_keys = [
            "pedicle",
            "lateral_mass_roy_camille",
            "lateral_mass_magerl",
            "lateral_mass_an",
            "lateral_mass_anderson",
            "transarticular",
            "c1_lateral_mass",
            "c2_pars",
            "occipital",
        ]
        for key in expected_keys:
            assert key in TECHNIQUE_ANGULATION

    def test_all_techniques_have_screw_defaults(self):
        for technique in VALID_INSTRUMENTATION_TECHNIQUES:
            if technique == "auto":
                continue
            assert technique in _TECHNIQUE_SCREW_DEFAULTS

    def test_screw_defaults_have_required_keys(self):
        required_keys = {"diameter_mm", "length_mm", "min_length_mm", "max_length_mm"}
        for technique, defaults in _TECHNIQUE_SCREW_DEFAULTS.items():
            assert required_keys.issubset(
                defaults.keys()
            ), f"Missing keys in {technique}: {required_keys - defaults.keys()}"

    def test_valid_sides_contains_expected(self):
        assert "left" in VALID_SIDES
        assert "right" in VALID_SIDES
        assert "bilateral" in VALID_SIDES

    def test_valid_variants_contains_expected(self):
        assert "roy_camille" in VALID_LATERAL_MASS_VARIANTS
        assert "magerl" in VALID_LATERAL_MASS_VARIANTS
        assert "an" in VALID_LATERAL_MASS_VARIANTS
        assert "anderson" in VALID_LATERAL_MASS_VARIANTS


# =============================================================================
# Code Generation Safety
# =============================================================================


class TestCodeGenerationSafety:
    """Test that generated Slicer code uses safe patterns."""

    def test_node_ids_are_json_escaped(self):
        """Verify all node IDs in generated code are JSON-escaped."""
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "pedicle",
                    "level": "C5",
                    "screws": [],
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")
            python_code = mock_client.exec_python.call_args[0][0]

            # Node ID should be JSON-escaped (in double quotes)
            assert 'seg_node_id = "vtkMRMLSegmentationNode1"' in python_code

    def test_code_uses_variable_for_get_node(self):
        """Verify generated code uses variable, not direct interpolation."""
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "pedicle",
                    "level": "C5",
                    "screws": [],
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws("pedicle", "C5", "vtkMRMLSegmentationNode1")
            python_code = mock_client.exec_python.call_args[0][0]

            assert "GetNodeByID(seg_node_id)" in python_code
            assert "GetNodeByID('vtkMRMLSegmentationNode1')" not in python_code

    def test_code_includes_slicer_imports(self):
        """All generated code should import necessary Slicer modules."""
        techniques = [
            ("pedicle", "C5", None),
            ("lateral_mass", "C4", None),
            ("c1_lateral_mass", "C1", None),
            ("c2_pars", "C2", None),
            ("occipital", "Occiput", None),
            ("auto", "C5", None),
        ]
        for technique, level, _ in techniques:
            with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
                mock_client = Mock()
                mock_client.exec_python.return_value = _mock_exec_result(
                    {
                        "success": True,
                        "technique": technique,
                        "level": level,
                        "screws": [],
                        "warnings": [],
                        "recommendations": [],
                        "analysis": {},
                        "thickness_map": {},
                    }
                )
                mock_get_client.return_value = mock_client

                plan_cervical_screws(technique, level, "vtkMRMLSegmentationNode1")
                python_code = mock_client.exec_python.call_args[0][0]
                assert "import slicer" in python_code
                assert "import json" in python_code

    def test_transarticular_code_includes_va_check(self):
        """Transarticular code must include VA safety check."""
        with patch("slicer_mcp.instrumentation_tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = _mock_exec_result(
                {
                    "success": True,
                    "technique": "transarticular",
                    "level": "C1C2",
                    "screws": [],
                    "isthmus_analysis": {},
                    "blocked": False,
                    "warnings": [],
                    "recommendations": [],
                }
            )
            mock_get_client.return_value = mock_client

            plan_cervical_screws(
                "transarticular",
                "C1C2",
                "vtkMRMLSegmentationNode1",
                va_node_id="vtkMRMLSegmentationNode2",
            )
            python_code = mock_client.exec_python.call_args[0][0]
            assert "va_node_id" in python_code
            assert "HARD BLOCK" in python_code
