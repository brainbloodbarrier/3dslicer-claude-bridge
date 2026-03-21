"""Unit tests for the workflow_ccj_protocol workflow tool."""

from unittest.mock import patch

import pytest

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import (
    SlicerConnectionError,
    SlicerTimeoutError,
)
from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.features.workflows.ccj import (
    _validate_population,
    workflow_ccj_protocol,
)

# =============================================================================
# Population Validation Tests
# =============================================================================


class TestValidatePopulation:
    """Test population parameter validation for CCJ workflow."""

    def test_valid_adult(self):
        assert _validate_population("adult") == "adult"

    def test_valid_child(self):
        assert _validate_population("child") == "child"

    def test_invalid_infant(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_population("infant")
        assert exc_info.value.field == "population"

    def test_invalid_empty(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_population("")
        assert exc_info.value.field == "population"

    def test_invalid_arbitrary(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_population("elderly")
        assert exc_info.value.field == "population"


# =============================================================================
# Fixtures: mock return values from underlying tools
# =============================================================================

MOCK_SEGMENT_RESULT = {
    "output_segmentation_id": "vtkMRMLSegmentationNode1",
    "vertebrae_count": 3,
    "vertebrae": ["C0", "C1", "C2"],
    "discs": [],
    "processing_time_seconds": 38.5,
}

MOCK_CCJ_RESULT = {
    "measurements": {
        "CXA": {"value": 148.2, "unit": "degrees"},
        "ADI": {"value": 2.1, "unit": "mm"},
        "Powers_ratio": {"value": 0.78, "unit": "ratio"},
        "BDI": {"value": 6.5, "unit": "mm"},
        "BAI": {"value": 8.2, "unit": "mm"},
        "Ranawat": {"value": 15.0, "unit": "mm"},
        "McGregor": {"value": 3.1, "unit": "mm"},
        "Chamberlain": {"value": 2.5, "unit": "mm"},
        "Wackenheim": {"value": "normal", "unit": ""},
    },
    "population": "adult",
}

MOCK_VA_RESULT = {
    "output_segmentation_id": "vtkMRMLSegmentationNode2",
    "left_va": {"diameter_mm": 3.2, "dominance": True},
    "right_va": {"diameter_mm": 2.8, "dominance": False},
}

MOCK_BONE_RESULT = {
    "vertebrae": {
        "C1": {"mean_hu": 285.3, "quality": "normal"},
        "C2": {"mean_hu": 310.1, "quality": "normal"},
    },
    "summary": {"osteoporotic_count": 0},
}

MOCK_SCREENSHOT_RESULT = {
    "success": True,
    "image": "base64encodedpng...",
    "view_type": "sagittal",
}


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestWorkflowCcjProtocolValidation:
    """Test input validation for workflow_ccj_protocol."""

    def test_invalid_ct_node_id(self):
        """Invalid CT node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_ccj_protocol(
                ct_volume_id="1invalid",
            )
        assert exc_info.value.field == "node_id"

    def test_invalid_segmentation_node_id(self):
        """Invalid segmentation node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                segmentation_node_id="123bad",
            )
        assert exc_info.value.field == "node_id"

    def test_invalid_cta_node_id(self):
        """Invalid CTA node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                cta_volume_id="DROP TABLE",
            )
        assert exc_info.value.field == "node_id"

    def test_invalid_population(self):
        """Invalid population raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                population="infant",
            )
        assert exc_info.value.field == "population"

    def test_empty_ct_node_id(self):
        """Empty CT node ID raises ValidationError."""
        with pytest.raises(ValidationError):
            workflow_ccj_protocol(
                ct_volume_id="",
            )

    def test_population_elderly_rejected(self):
        """Population 'elderly' is not valid."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                population="elderly",
            )
        assert exc_info.value.field == "population"


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestWorkflowCcjProtocolHappyPath:
    """Test happy-path execution of workflow_ccj_protocol."""

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_full_pipeline(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """Full pipeline with segmentation."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode1"
        assert result["ccj_angles"] == MOCK_CCJ_RESULT
        assert result["vertebral_artery"] is None
        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert result["population"] == "adult"
        assert len(result["screenshots"]) == 1
        assert "segment_spine" in result["steps_completed"]
        assert "measure_ccj_angles" in result["steps_completed"]
        assert "analyze_bone_quality" in result["steps_completed"]
        assert "capture_screenshot" in result["steps_completed"]
        # VA should NOT be run without CTA
        assert "segment_vertebral_artery" not in result["steps_completed"]

        # Verify segment_spine called correctly
        mock_segment.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode1",
            region="cervical",
            include_discs=False,
            include_spinal_cord=False,
        )

        # Verify CCJ angles called correctly
        mock_ccj.assert_called_once_with(
            segmentation_node_id=("vtkMRMLSegmentationNode1"),
            population="adult",
        )

        # Verify bone quality called correctly
        mock_bone.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id=("vtkMRMLSegmentationNode1"),
            region="cervical",
        )

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_existing_segmentation_skips_segment_spine(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """Existing segmentation skips segment_spine."""
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id=("vtkMRMLSegmentationNode5"),
        )

        # segment_spine should NOT have been called
        mock_segment.assert_not_called()
        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode5"
        assert "segment_spine_skipped" in result["steps_completed"]
        assert "segment_spine" not in result["steps_completed"]

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_vertebral_artery",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_with_cta_includes_va_segmentation(
        self,
        mock_segment,
        mock_ccj,
        mock_va,
        mock_bone,
        mock_screenshot,
    ):
        """CTA volume triggers vertebral artery segmentation."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_va.return_value = MOCK_VA_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            cta_volume_id="vtkMRMLScalarVolumeNode2",
        )

        assert result["vertebral_artery"] == MOCK_VA_RESULT
        assert "segment_vertebral_artery" in result["steps_completed"]

        mock_va.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode2",
        )

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_without_cta_no_va_segmentation(
        self,
        mock_segment,
        mock_ccj,
        mock_screenshot,
    ):
        """No CTA volume means no VA segmentation."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            include_bone_quality=False,
        )

        assert result["vertebral_artery"] is None
        assert "segment_vertebral_artery" not in result["steps_completed"]

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_bone_quality_enabled_by_default(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """Bone quality analysis runs by default."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert "analyze_bone_quality" in result["steps_completed"]
        mock_bone.assert_called_once()

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_bone_quality_disabled(
        self,
        mock_segment,
        mock_ccj,
        mock_screenshot,
    ):
        """Bone quality analysis can be disabled."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            include_bone_quality=False,
        )

        assert result["bone_quality"] is None
        assert "analyze_bone_quality" not in result["steps_completed"]

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_child_population(
        self,
        mock_segment,
        mock_ccj,
        mock_screenshot,
    ):
        """Child population is passed through."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            population="child",
            include_bone_quality=False,
        )

        assert result["population"] == "child"
        mock_ccj.assert_called_once_with(
            segmentation_node_id=("vtkMRMLSegmentationNode1"),
            population="child",
        )


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestWorkflowCcjProtocolErrors:
    """Test error propagation from underlying tools."""

    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_segment_spine_error_propagates(self, mock_segment):
        """SlicerConnectionError from segment_spine propagates."""
        mock_segment.side_effect = SlicerConnectionError("Slicer not responding")

        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_ccj_angles_error_propagates(self, mock_segment, mock_ccj):
        """SlicerConnectionError from measure_ccj_angles propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.side_effect = SlicerConnectionError(
            "CCJ analysis failed",
        )

        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_vertebral_artery",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_va_segmentation_error_propagates(self, mock_segment, mock_ccj, mock_va):
        """SlicerConnectionError from VA segmentation propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_va.side_effect = SlicerConnectionError(
            "VA segmentation failed",
        )

        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                cta_volume_id=("vtkMRMLScalarVolumeNode2"),
            )

    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_bone_quality_error_propagates(self, mock_segment, mock_ccj, mock_bone):
        """SlicerConnectionError from bone quality propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.side_effect = SlicerConnectionError(
            "Bone quality analysis failed",
        )

        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_screenshot_failure_is_nonfatal(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """Screenshot failure does not abort the workflow."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = SlicerConnectionError("Screenshot failed")

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        # Workflow should still succeed
        assert result["ccj_angles"] == MOCK_CCJ_RESULT
        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_screenshot_value_error_is_nonfatal(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """ValueError from screenshot does not abort."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = ValueError(
            "Invalid view type",
        )

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["ccj_angles"] == MOCK_CCJ_RESULT
        assert len(result["screenshots"]) == 0

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_screenshot_timeout_is_nonfatal(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """SlicerTimeoutError from screenshot is non-fatal."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = SlicerTimeoutError("Screenshot timed out")

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["ccj_angles"] == MOCK_CCJ_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_screenshot_circuit_open_is_nonfatal(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """CircuitOpenError from screenshot is non-fatal."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = CircuitOpenError(
            "Circuit breaker is open",
            "slicer",
            30.0,
        )

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["ccj_angles"] == MOCK_CCJ_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    def test_validation_error_from_tool_propagates(self, mock_ccj):
        """ValidationError raised inside a tool propagates."""
        mock_ccj.side_effect = ValidationError(
            "Bad input",
            "segmentation_node_id",
            "bad",
        )

        with pytest.raises(ValidationError):
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                segmentation_node_id=("vtkMRMLSegmentationNode1"),
            )


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestWorkflowCcjProtocolResultStructure:
    """Test that the result dict has expected keys."""

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_result_has_all_expected_keys(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """Result dict contains all documented keys."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        expected_keys = {
            "segmentation_node_id",
            "ccj_angles",
            "vertebral_artery",
            "bone_quality",
            "screenshots",
            "population",
            "steps_completed",
        }
        assert set(result.keys()) == expected_keys

    @patch(
        "slicer_mcp.features.workflows.ccj" ".capture_screenshot",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".analyze_bone_quality",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".measure_ccj_angles",
    )
    @patch(
        "slicer_mcp.features.workflows.ccj" ".segment_spine",
    )
    def test_default_population_is_adult(
        self,
        mock_segment,
        mock_ccj,
        mock_bone,
        mock_screenshot,
    ):
        """Default population should be adult."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["population"] == "adult"
