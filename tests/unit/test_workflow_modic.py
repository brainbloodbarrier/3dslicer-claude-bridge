"""Unit tests for the workflow_modic_eval workflow tool."""

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

    def test_valid_lumbar(self):
        assert _validate_region("lumbar") == "lumbar"

    def test_valid_cervical(self):
        assert _validate_region("cervical") == "cervical"

    def test_valid_thoracic(self):
        assert _validate_region("thoracic") == "thoracic"

    def test_invalid_full(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region("full")
        assert exc_info.value.field == "region"

    def test_invalid_empty(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region("")
        assert exc_info.value.field == "region"

    def test_invalid_arbitrary(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region("sacral")
        assert exc_info.value.field == "region"


class TestCordScreeningRegions:
    """Test cord screening region constant."""

    def test_cervical_included(self):
        assert "cervical" in CORD_SCREENING_REGIONS

    def test_thoracic_included(self):
        assert "thoracic" in CORD_SCREENING_REGIONS

    def test_lumbar_excluded(self):
        assert "lumbar" not in CORD_SCREENING_REGIONS


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
    "levels": {
        "C5-C6": {"compression_ratio": 0.35, "stenosis_grade": "moderate"},
    },
    "myelopathy_detected": False,
}

MOCK_SCREENSHOT_RESULT = {
    "success": True,
    "image": "base64encodedpng...",
    "view_type": "sagittal",
}


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestWorkflowModicEvalValidation:
    """Test input validation for workflow_modic_eval."""

    def test_invalid_t1_node_id(self):
        """Invalid T1 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_modic_eval(
                t1_volume_id="1invalid",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
            )
        assert exc_info.value.field == "node_id"

    def test_invalid_t2_node_id(self):
        """Invalid T2 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="DROP TABLE",
            )
        assert exc_info.value.field == "node_id"

    def test_invalid_region(self):
        """Invalid region raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
                region="sacral",
            )
        assert exc_info.value.field == "region"

    def test_invalid_segmentation_node_id(self):
        """Invalid segmentation node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
                segmentation_node_id="123bad",
            )
        assert exc_info.value.field == "node_id"

    def test_empty_t1_node_id(self):
        """Empty T1 node ID raises ValidationError."""
        with pytest.raises(ValidationError):
            workflow_modic_eval(
                t1_volume_id="",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
            )

    def test_region_full_rejected(self):
        """Region 'full' is not valid for MRI workflows."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
                region="full",
            )
        assert exc_info.value.field == "region"


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestWorkflowModicEvalHappyPath:
    """Test happy-path execution of workflow_modic_eval."""

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_full_pipeline_lumbar(self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot):
        """Full pipeline with segmentation for lumbar region."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
            region="lumbar",
        )

        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode1"
        assert result["modic_changes"] == MOCK_MODIC_RESULT
        assert result["pfirrmann_grades"] == MOCK_PFIRRMANN_RESULT
        assert result["cord_compression"] is None
        assert result["region"] == "lumbar"
        assert len(result["screenshots"]) == 1
        assert "segment_spine" in result["steps_completed"]
        assert "classify_modic_changes" in result["steps_completed"]
        assert "assess_disc_degeneration_mri" in result["steps_completed"]
        assert "capture_screenshot" in result["steps_completed"]
        # Cord compression should NOT be run for lumbar
        assert "detect_cord_compression_mri" not in result["steps_completed"]

        # Verify segment_spine was called with T2 volume and correct params
        mock_segment.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode2",
            region="lumbar",
            include_discs=True,
            include_spinal_cord=True,
        )

        # Verify modic was called with correct params
        mock_modic.assert_called_once_with(
            t1_node_id="vtkMRMLScalarVolumeNode1",
            t2_node_id="vtkMRMLScalarVolumeNode2",
            region="lumbar",
            segmentation_node_id="vtkMRMLSegmentationNode1",
        )

        # Verify pfirrmann was called with correct params
        mock_pfirrmann.assert_called_once_with(
            t2_node_id="vtkMRMLScalarVolumeNode2",
            region="lumbar",
            segmentation_node_id="vtkMRMLSegmentationNode1",
        )

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_existing_segmentation_skips_segment_spine(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """When segmentation_node_id is provided, segment_spine is skipped."""
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
            region="lumbar",
            segmentation_node_id="vtkMRMLSegmentationNode5",
        )

        # segment_spine should NOT have been called
        mock_segment.assert_not_called()
        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode5"
        assert "segment_spine_skipped" in result["steps_completed"]
        assert "segment_spine" not in result["steps_completed"]

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.detect_cord_compression_mri")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_cervical_includes_cord_screening(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_cord, mock_screenshot
    ):
        """Cervical region includes cord compression screening by default."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_cord.return_value = MOCK_CORD_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
            region="cervical",
        )

        assert result["cord_compression"] == MOCK_CORD_RESULT
        assert "detect_cord_compression_mri" in result["steps_completed"]

        mock_cord.assert_called_once_with(
            t2_node_id="vtkMRMLScalarVolumeNode2",
            t1_node_id="vtkMRMLScalarVolumeNode1",
            region="cervical",
            segmentation_node_id="vtkMRMLSegmentationNode1",
        )

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.detect_cord_compression_mri")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_thoracic_includes_cord_screening(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_cord, mock_screenshot
    ):
        """Thoracic region includes cord compression screening by default."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_cord.return_value = MOCK_CORD_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
            region="thoracic",
        )

        assert result["cord_compression"] == MOCK_CORD_RESULT
        assert "detect_cord_compression_mri" in result["steps_completed"]

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.detect_cord_compression_mri")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_cord_screening_disabled(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_cord, mock_screenshot
    ):
        """Cord screening can be explicitly disabled even for cervical."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
            region="cervical",
            include_cord_screening=False,
        )

        mock_cord.assert_not_called()
        assert result["cord_compression"] is None
        assert "detect_cord_compression_mri" not in result["steps_completed"]

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_lumbar_cord_screening_true_no_cord_run(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """Even with include_cord_screening=True, lumbar does NOT run cord detection."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
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

    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_segment_spine_error_propagates(self, mock_segment):
        """SlicerConnectionError from segment_spine propagates to caller."""
        mock_segment.side_effect = SlicerConnectionError("Slicer not responding")

        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
            )

    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_modic_error_propagates(self, mock_segment, mock_modic):
        """SlicerConnectionError from classify_modic_changes propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.side_effect = SlicerConnectionError("Modic analysis failed")

        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
            )

    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_pfirrmann_error_propagates(self, mock_segment, mock_modic, mock_pfirrmann):
        """SlicerConnectionError from assess_disc_degeneration_mri propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.side_effect = SlicerConnectionError("Pfirrmann analysis failed")

        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
            )

    @patch("slicer_mcp.features.workflows.modic.detect_cord_compression_mri")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_cord_compression_error_propagates(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_cord
    ):
        """SlicerConnectionError from detect_cord_compression_mri propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_cord.side_effect = SlicerConnectionError("Cord analysis failed")

        with pytest.raises(SlicerConnectionError):
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
                region="cervical",
            )

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_screenshot_failure_is_nonfatal(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """Screenshot failure does not abort the workflow."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.side_effect = SlicerConnectionError("Screenshot failed")

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
        )

        # Workflow should still succeed
        assert result["modic_changes"] == MOCK_MODIC_RESULT
        assert result["pfirrmann_grades"] == MOCK_PFIRRMANN_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_screenshot_value_error_is_nonfatal(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """ValueError from screenshot does not abort the workflow."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.side_effect = ValueError("Invalid view type")

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
        )

        assert result["modic_changes"] == MOCK_MODIC_RESULT
        assert len(result["screenshots"]) == 0

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_screenshot_timeout_is_nonfatal(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """SlicerTimeoutError from screenshot does not abort the workflow."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.side_effect = SlicerTimeoutError("Screenshot timed out")

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
        )

        assert result["modic_changes"] == MOCK_MODIC_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_screenshot_circuit_open_is_nonfatal(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """CircuitOpenError from screenshot does not abort the workflow."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.side_effect = CircuitOpenError("Circuit breaker is open", "slicer", 30.0)

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
        )

        assert result["modic_changes"] == MOCK_MODIC_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    def test_validation_error_from_underlying_tool_propagates(self, mock_modic):
        """ValidationError raised inside a tool propagates unchanged."""
        mock_modic.side_effect = ValidationError("Bad input", "t1_node_id", "bad")

        with pytest.raises(ValidationError):
            workflow_modic_eval(
                t1_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="vtkMRMLScalarVolumeNode2",
                segmentation_node_id="vtkMRMLSegmentationNode1",
            )


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestWorkflowModicEvalResultStructure:
    """Test that the result dict has the expected keys."""

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_result_has_all_expected_keys(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """Result dict contains all documented keys."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
        )

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

    @patch("slicer_mcp.features.workflows.modic.capture_screenshot")
    @patch("slicer_mcp.features.workflows.modic.assess_disc_degeneration_mri")
    @patch("slicer_mcp.features.workflows.modic.classify_modic_changes")
    @patch("slicer_mcp.features.workflows.modic.segment_spine")
    def test_default_region_is_lumbar(
        self, mock_segment, mock_modic, mock_pfirrmann, mock_screenshot
    ):
        """Default region should be lumbar when not specified."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_modic.return_value = MOCK_MODIC_RESULT
        mock_pfirrmann.return_value = MOCK_PFIRRMANN_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_modic_eval(
            t1_volume_id="vtkMRMLScalarVolumeNode1",
            t2_volume_id="vtkMRMLScalarVolumeNode2",
        )

        assert result["region"] == "lumbar"
