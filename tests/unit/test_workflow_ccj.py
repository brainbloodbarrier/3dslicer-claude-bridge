"""Unit tests for the workflow_ccj_protocol workflow tool."""

from types import SimpleNamespace
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

    @pytest.mark.parametrize("population", ["adult", "child"])
    def test_valid_populations(self, population):
        assert _validate_population(population) == population

    @pytest.mark.parametrize("population", ["infant", "", "elderly", "ADULT"])
    def test_invalid_populations(self, population):
        with pytest.raises(ValidationError) as exc_info:
            _validate_population(population)
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

_PATCH_PREFIX = "slicer_mcp.features.workflows.ccj"


@pytest.fixture()
def ccj_mocks():
    """Patch all CCJ pipeline tools and return mocks in a namespace."""
    with (
        patch(f"{_PATCH_PREFIX}.segment_spine") as m_segment,
        patch(f"{_PATCH_PREFIX}.measure_ccj_angles") as m_ccj,
        patch(f"{_PATCH_PREFIX}.analyze_bone_quality") as m_bone,
        patch(f"{_PATCH_PREFIX}.capture_screenshot") as m_screenshot,
    ):
        m_segment.return_value = MOCK_SEGMENT_RESULT
        m_ccj.return_value = MOCK_CCJ_RESULT
        m_bone.return_value = MOCK_BONE_RESULT
        m_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        yield SimpleNamespace(
            segment=m_segment,
            ccj=m_ccj,
            bone=m_bone,
            screenshot=m_screenshot,
        )


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestWorkflowCcjProtocolValidation:
    """Test input validation for workflow_ccj_protocol."""

    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"ct_volume_id": "1invalid"}, "node_id"),
            (
                {
                    "ct_volume_id": "vtkMRMLScalarVolumeNode1",
                    "segmentation_node_id": "123bad",
                },
                "node_id",
            ),
            (
                {
                    "ct_volume_id": "vtkMRMLScalarVolumeNode1",
                    "cta_volume_id": "DROP TABLE",
                },
                "node_id",
            ),
            (
                {
                    "ct_volume_id": "vtkMRMLScalarVolumeNode1",
                    "population": "infant",
                },
                "population",
            ),
            (
                {
                    "ct_volume_id": "vtkMRMLScalarVolumeNode1",
                    "population": "elderly",
                },
                "population",
            ),
        ],
    )
    def test_invalid_inputs(self, kwargs, field):
        with pytest.raises(ValidationError) as exc_info:
            workflow_ccj_protocol(**kwargs)
        assert exc_info.value.field == field

    def test_empty_ct_node_id(self):
        with pytest.raises(ValidationError):
            workflow_ccj_protocol(ct_volume_id="")


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestWorkflowCcjProtocolHappyPath:
    """Test happy-path execution of workflow_ccj_protocol."""

    def test_full_pipeline(self, ccj_mocks):
        """Full pipeline with segmentation."""
        result = workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")

        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode1"
        assert result["ccj_angles"] == MOCK_CCJ_RESULT
        assert result["vertebral_artery"] is None
        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert result["population"] == "adult"
        assert len(result["screenshots"]) == 1

        expected_steps = {
            "segment_spine",
            "measure_ccj_angles",
            "analyze_bone_quality",
            "capture_screenshot",
        }
        assert expected_steps.issubset(set(result["steps_completed"]))
        assert "segment_vertebral_artery" not in result["steps_completed"]

        ccj_mocks.segment.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode1",
            region="cervical",
            include_discs=False,
            include_spinal_cord=False,
        )
        ccj_mocks.ccj.assert_called_once_with(
            segmentation_node_id="vtkMRMLSegmentationNode1",
            population="adult",
        )
        ccj_mocks.bone.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id="vtkMRMLSegmentationNode1",
            region="cervical",
        )

    def test_existing_segmentation_skips_segment_spine(self, ccj_mocks):
        """Existing segmentation skips segment_spine."""
        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id="vtkMRMLSegmentationNode5",
        )

        ccj_mocks.segment.assert_not_called()
        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode5"
        assert "segment_spine_skipped" in result["steps_completed"]
        assert "segment_spine" not in result["steps_completed"]

    def test_with_cta_includes_va_segmentation(self, ccj_mocks):
        """CTA volume triggers vertebral artery segmentation."""
        with patch(f"{_PATCH_PREFIX}.segment_vertebral_artery") as m_va:
            m_va.return_value = MOCK_VA_RESULT

            result = workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                cta_volume_id="vtkMRMLScalarVolumeNode2",
            )

            assert result["vertebral_artery"] == MOCK_VA_RESULT
            assert "segment_vertebral_artery" in result["steps_completed"]
            m_va.assert_called_once_with(input_node_id="vtkMRMLScalarVolumeNode2")

    def test_without_cta_no_va_segmentation(self, ccj_mocks):
        """No CTA volume means no VA segmentation."""
        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            include_bone_quality=False,
        )

        assert result["vertebral_artery"] is None
        assert "segment_vertebral_artery" not in result["steps_completed"]

    def test_bone_quality_enabled_by_default(self, ccj_mocks):
        """Bone quality analysis runs by default."""
        result = workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")

        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert "analyze_bone_quality" in result["steps_completed"]
        ccj_mocks.bone.assert_called_once()

    def test_bone_quality_disabled(self, ccj_mocks):
        """Bone quality analysis can be disabled."""
        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            include_bone_quality=False,
        )

        assert result["bone_quality"] is None
        assert "analyze_bone_quality" not in result["steps_completed"]

    def test_child_population(self, ccj_mocks):
        """Child population is passed through."""
        result = workflow_ccj_protocol(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            population="child",
            include_bone_quality=False,
        )

        assert result["population"] == "child"
        ccj_mocks.ccj.assert_called_once_with(
            segmentation_node_id="vtkMRMLSegmentationNode1",
            population="child",
        )


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestWorkflowCcjProtocolErrors:
    """Test error propagation from underlying tools."""

    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_segment_spine_error_propagates(self, mock_segment):
        mock_segment.side_effect = SlicerConnectionError("Slicer not responding")
        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.measure_ccj_angles")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_ccj_angles_error_propagates(self, mock_segment, mock_ccj):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.side_effect = SlicerConnectionError("CCJ analysis failed")
        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.segment_vertebral_artery")
    @patch(f"{_PATCH_PREFIX}.measure_ccj_angles")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_va_segmentation_error_propagates(self, mock_segment, mock_ccj, mock_va):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_va.side_effect = SlicerConnectionError("VA segmentation failed")
        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                cta_volume_id="vtkMRMLScalarVolumeNode2",
            )

    @patch(f"{_PATCH_PREFIX}.analyze_bone_quality")
    @patch(f"{_PATCH_PREFIX}.measure_ccj_angles")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_bone_quality_error_propagates(self, mock_segment, mock_ccj, mock_bone):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_ccj.return_value = MOCK_CCJ_RESULT
        mock_bone.side_effect = SlicerConnectionError("Bone quality analysis failed")
        with pytest.raises(SlicerConnectionError):
            workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")

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
    def test_screenshot_failure_is_nonfatal(self, ccj_mocks, error):
        """Screenshot failures of any type do not abort the workflow."""
        ccj_mocks.screenshot.side_effect = error

        result = workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")

        assert result["ccj_angles"] == MOCK_CCJ_RESULT
        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.measure_ccj_angles")
    def test_validation_error_from_tool_propagates(self, mock_ccj):
        mock_ccj.side_effect = ValidationError("Bad input", "segmentation_node_id", "bad")
        with pytest.raises(ValidationError):
            workflow_ccj_protocol(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                segmentation_node_id="vtkMRMLSegmentationNode1",
            )


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestWorkflowCcjProtocolResultStructure:
    """Test that the result dict has expected keys."""

    def test_result_has_all_expected_keys(self, ccj_mocks):
        result = workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")
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

    def test_default_population_is_adult(self, ccj_mocks):
        result = workflow_ccj_protocol(ct_volume_id="vtkMRMLScalarVolumeNode1")
        assert result["population"] == "adult"
