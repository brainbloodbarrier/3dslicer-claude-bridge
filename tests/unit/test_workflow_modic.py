"""Unit tests for the workflow_modic_eval workflow tool."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.constants import CORD_SCREENING_REGIONS
from slicer_mcp.core.slicer_client import SlicerConnectionError, SlicerTimeoutError
from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.features.workflows.modic import (
    _validate_region,
    workflow_modic_eval,
)

# =============================================================================
# Region Validation Tests
# =============================================================================


class TestValidateRegion:
    """Test region parameter validation for the Modic workflow."""

    @pytest.mark.parametrize("region", ["lumbar", "cervical", "thoracic"])
    def test_valid_regions(self, region):
        assert _validate_region(region) == region

    @pytest.mark.parametrize("region", ["full", "", "sacral", "LUMBAR"])
    def test_invalid_regions(self, region):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region(region)
        assert exc_info.value.field == "region"


class TestCordScreeningRegions:
    """Test cord screening region constant."""

    @pytest.mark.parametrize(
        "region, expected",
        [("cervical", True), ("thoracic", True), ("lumbar", False)],
    )
    def test_cord_screening_membership(self, region, expected):
        assert (region in CORD_SCREENING_REGIONS) is expected


# =============================================================================
# Fixtures: mock return values from underlying tools
# =============================================================================

MOCK_SEGMENT_RESULT = {
    "output_segmentation_id": "vtkMRMLSegmentationNode1",
    "vertebrae_count": 5,
    "vertebrae": ["L1", "L2", "L3", "L4", "L5"],
    "discs": ["L1-L2", "L2-L3", "L3-L4", "L4-L5"],
    "processing_time_seconds": 45.2,
}

MOCK_MODIC_RESULT = {
    "levels": {
        "L4-L5": {"modic_type": "I", "t1_ratio": 0.75, "t2_ratio": 1.45},
        "L5-S1": {"modic_type": "II", "t1_ratio": 1.35, "t2_ratio": 1.10},
    },
    "summary": {"type_0": 3, "type_I": 1, "type_II": 1},
}

MOCK_PFIRRMANN_RESULT = {
    "discs": {
        "L3-L4": {"grade": "II", "signal_ratio": 0.85},
        "L4-L5": {"grade": "IV", "signal_ratio": 0.45},
        "L5-S1": {"grade": "V", "signal_ratio": 0.20},
    },
    "summary": {"grade_I": 0, "grade_II": 1, "grade_III": 0, "grade_IV": 1, "grade_V": 1},
}

MOCK_CORD_RESULT = {
    "levels": {"C5-C6": {"compression_ratio": 0.35, "stenosis_grade": "moderate"}},
    "myelopathy_detected": False,
}

MOCK_SCREENSHOT_RESULT = {
    "success": True,
    "image": "base64encodedpng...",
    "view_type": "sagittal",
}

_PATCH_PREFIX = "slicer_mcp.features.workflows.modic"


@pytest.fixture()
def modic_mocks():
    """Patch all Modic pipeline tools and return mocks in a namespace."""
    with (
        patch(f"{_PATCH_PREFIX}.segment_spine") as m_segment,
        patch(f"{_PATCH_PREFIX}.classify_modic_changes") as m_modic,
        patch(f"{_PATCH_PREFIX}.assess_disc_degeneration_mri") as m_pfirrmann,
        patch(f"{_PATCH_PREFIX}.capture_screenshot") as m_screenshot,
    ):
        m_segment.return_value = MOCK_SEGMENT_RESULT
        m_modic.return_value = MOCK_MODIC_RESULT
        m_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        m_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        yield SimpleNamespace(
            segment=m_segment,
            modic=m_modic,
            pfirrmann=m_pfirrmann,
            screenshot=m_screenshot,
        )


# =============================================================================
# Input Validation Tests
# =============================================================================

_VALID_T1 = "vtkMRMLScalarVolumeNode1"
_VALID_T2 = "vtkMRMLScalarVolumeNode2"


class TestWorkflowModicEvalValidation:
    """Test input validation for workflow_modic_eval."""

    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"t1_volume_id": "1invalid", "t2_volume_id": _VALID_T2}, "node_id"),
            ({"t1_volume_id": _VALID_T1, "t2_volume_id": "DROP TABLE"}, "node_id"),
            ({"t1_volume_id": _VALID_T1, "t2_volume_id": _VALID_T2, "region": "sacral"}, "region"),
            (
                {
                    "t1_volume_id": _VALID_T1,
                    "t2_volume_id": _VALID_T2,
                    "segmentation_node_id": "123bad",
                },
                "node_id",
            ),
            ({"t1_volume_id": _VALID_T1, "t2_volume_id": _VALID_T2, "region": "full"}, "region"),
        ],
    )
    def test_invalid_inputs(self, kwargs, field):
        with pytest.raises(ValidationError) as exc_info:
            workflow_modic_eval(**kwargs)
        assert exc_info.value.field == field

    def test_empty_t1_node_id(self):
        with pytest.raises(ValidationError):
            workflow_modic_eval(t1_volume_id="", t2_volume_id=_VALID_T2)


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestWorkflowModicEvalHappyPath:
    """Test happy-path execution of workflow_modic_eval."""

    def test_full_pipeline_lumbar(self, modic_mocks):
        """Full pipeline with segmentation for lumbar region."""
        result = workflow_modic_eval(
            t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2, region="lumbar"
        )

        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode1"
        assert result["modic_changes"] == MOCK_MODIC_RESULT
        assert result["pfirrmann_grades"] == MOCK_PFIRRMANN_RESULT
        assert result["cord_compression"] is None
        assert result["region"] == "lumbar"
        assert len(result["screenshots"]) == 1

        expected_steps = {
            "segment_spine",
            "classify_modic_changes",
            "assess_disc_degeneration_mri",
            "capture_screenshot",
        }
        assert expected_steps.issubset(set(result["steps_completed"]))
        assert "detect_cord_compression_mri" not in result["steps_completed"]

        modic_mocks.segment.assert_called_once_with(
            input_node_id=_VALID_T2,
            region="lumbar",
            include_discs=True,
            include_spinal_cord=True,
        )
        modic_mocks.modic.assert_called_once_with(
            t1_node_id=_VALID_T1,
            t2_node_id=_VALID_T2,
            region="lumbar",
            segmentation_node_id="vtkMRMLSegmentationNode1",
        )
        modic_mocks.pfirrmann.assert_called_once_with(
            t2_node_id=_VALID_T2,
            region="lumbar",
            segmentation_node_id="vtkMRMLSegmentationNode1",
        )

    def test_existing_segmentation_skips_segment_spine(self, modic_mocks):
        """When segmentation_node_id is provided, segment_spine is skipped."""
        result = workflow_modic_eval(
            t1_volume_id=_VALID_T1,
            t2_volume_id=_VALID_T2,
            region="lumbar",
            segmentation_node_id="vtkMRMLSegmentationNode5",
        )

        modic_mocks.segment.assert_not_called()
        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode5"
        assert "segment_spine_skipped" in result["steps_completed"]
        assert "segment_spine" not in result["steps_completed"]

    @pytest.mark.parametrize("region", ["cervical", "thoracic"])
    def test_cord_screening_regions_include_cord(self, modic_mocks, region):
        """Cervical and thoracic regions include cord compression screening."""
        with patch(f"{_PATCH_PREFIX}.detect_cord_compression_mri") as m_cord:
            m_cord.return_value = MOCK_CORD_RESULT

            result = workflow_modic_eval(
                t1_volume_id=_VALID_T1,
                t2_volume_id=_VALID_T2,
                region=region,
            )

            assert result["cord_compression"] == MOCK_CORD_RESULT
            assert "detect_cord_compression_mri" in result["steps_completed"]
            m_cord.assert_called_once_with(
                t2_node_id=_VALID_T2,
                t1_node_id=_VALID_T1,
                region=region,
                segmentation_node_id="vtkMRMLSegmentationNode1",
            )

    def test_cord_screening_disabled(self, modic_mocks):
        """Cord screening can be explicitly disabled even for cervical."""
        with patch(f"{_PATCH_PREFIX}.detect_cord_compression_mri") as m_cord:
            result = workflow_modic_eval(
                t1_volume_id=_VALID_T1,
                t2_volume_id=_VALID_T2,
                region="cervical",
                include_cord_screening=False,
            )

            m_cord.assert_not_called()
            assert result["cord_compression"] is None
            assert "detect_cord_compression_mri" not in result["steps_completed"]

    def test_lumbar_cord_screening_true_no_cord_run(self, modic_mocks):
        """Even with include_cord_screening=True, lumbar does NOT run cord detection."""
        result = workflow_modic_eval(
            t1_volume_id=_VALID_T1,
            t2_volume_id=_VALID_T2,
            region="lumbar",
            include_cord_screening=True,
        )

        assert result["cord_compression"] is None
        assert "detect_cord_compression_mri" not in result["steps_completed"]


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestWorkflowModicEvalErrors:
    """Test error propagation from underlying tools."""

    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_segment_spine_error_propagates(self, mock_segment):
        mock_segment.side_effect = SlicerConnectionError("Slicer not responding")
        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2)

    @patch(f"{_PATCH_PREFIX}.classify_modic_changes")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_modic_error_propagates(self, mock_segment, mock_modic):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.side_effect = SlicerConnectionError("Modic analysis failed")
        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2)

    @patch(f"{_PATCH_PREFIX}.assess_disc_degeneration_mri")
    @patch(f"{_PATCH_PREFIX}.classify_modic_changes")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_pfirrmann_error_propagates(self, mock_segment, mock_modic, mock_pfirrmann):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.side_effect = SlicerConnectionError("Pfirrmann analysis failed")
        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2)

    @patch(f"{_PATCH_PREFIX}.detect_cord_compression_mri")
    @patch(f"{_PATCH_PREFIX}.assess_disc_degeneration_mri")
    @patch(f"{_PATCH_PREFIX}.classify_modic_changes")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_cord_compression_error_propagates(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_cord
    ):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_cord.side_effect = SlicerConnectionError("Cord analysis failed")
        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2, region="cervical")

    @pytest.mark.parametrize(
        "error",
        [
            SlicerConnectionError("Screenshot failed"),
            SlicerTimeoutError("Screenshot timed out"),
            CircuitOpenError("Circuit breaker is open", "slicer", 30.0),
            ValueError("Invalid view type"),
        ],
        ids=["connection", "timeout", "circuit_open", "value_error"],
    )
    def test_screenshot_failure_is_nonfatal(self, modic_mocks, error):
        """Screenshot failures of any type do not abort the workflow."""
        modic_mocks.screenshot.side_effect = error

        result = workflow_modic_eval(t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2)

        assert result["modic_changes"] == MOCK_MODIC_RESULT
        assert result["pfirrmann_grades"] == MOCK_PFIRRMANN_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.classify_modic_changes")
    def test_validation_error_from_underlying_tool_propagates(self, mock_modic):
        mock_modic.side_effect = ValidationError("Bad input", "t1_node_id", "bad")
        with pytest.raises(ValidationError):
            workflow_modic_eval(
                t1_volume_id=_VALID_T1,
                t2_volume_id=_VALID_T2,
                segmentation_node_id="vtkMRMLSegmentationNode1",
            )


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestWorkflowModicEvalResultStructure:
    """Test that the result dict has the expected keys."""

    def test_result_has_all_expected_keys(self, modic_mocks):
        result = workflow_modic_eval(t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2)
        expected_keys = {
            "segmentation_node_id",
            "modic_changes",
            "pfirrmann_grades",
            "cord_compression",
            "screenshots",
            "region",
            "steps_completed",
        }
        assert set(result.keys()) == expected_keys

    def test_default_region_is_lumbar(self, modic_mocks):
        result = workflow_modic_eval(t1_volume_id=_VALID_T1, t2_volume_id=_VALID_T2)
        assert result["region"] == "lumbar"
