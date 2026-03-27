"""Unit tests for the workflow_onco_spine workflow tool."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import (
    SlicerConnectionError,
    SlicerTimeoutError,
)
from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.features.spine.constants import SINS_PAIN_SCORES, SPINE_REGIONS
from slicer_mcp.features.workflows.onco_spine import (
    _validate_pain_type,
    _validate_region,
    workflow_onco_spine,
)

# ================================================================
# Region Validation Tests
# ================================================================


class TestValidateRegion:
    """Test region parameter validation for onco-spine."""

    @pytest.mark.parametrize("region", ["cervical", "thoracic", "lumbar", "full"])
    def test_valid_regions(self, region):
        assert _validate_region(region) == region

    @pytest.mark.parametrize("region", ["", "sacral", "neck", "CERVICAL"])
    def test_invalid_regions(self, region):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region(region)
        assert exc_info.value.field == "region"


# ================================================================
# Pain Type Validation Tests
# ================================================================


class TestValidatePainType:
    """Test pain_type parameter validation."""

    @pytest.mark.parametrize(
        "pain_type",
        ["mechanical", "occasional_non_mechanical", "pain_free"],
    )
    def test_valid_pain_types(self, pain_type):
        assert _validate_pain_type(pain_type) == pain_type

    def test_valid_none(self):
        assert _validate_pain_type(None) is None

    @pytest.mark.parametrize("pain_type", ["sharp", "", "chronic"])
    def test_invalid_pain_types(self, pain_type):
        with pytest.raises(ValidationError) as exc_info:
            _validate_pain_type(pain_type)
        assert exc_info.value.field == "pain_type"


# ================================================================
# Constants Tests
# ================================================================


class TestConstants:
    """Test onco-spine workflow constants."""

    @pytest.mark.parametrize("region", ["full", "cervical", "thoracic", "lumbar"])
    def test_valid_regions(self, region):
        assert region in SPINE_REGIONS

    def test_valid_pain_types(self):
        assert SINS_PAIN_SCORES == {
            "mechanical": 3,
            "occasional_non_mechanical": 1,
            "pain_free": 0,
        }


# ================================================================
# Fixtures: mock return values from underlying tools
# ================================================================

MOCK_SEGMENT_RESULT = {
    "output_segmentation_id": "vtkMRMLSegmentationNode1",
    "vertebrae_count": 24,
    "vertebrae": [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
    ],
    "discs": ["L4-L5", "L5-S1"],
    "processing_time_seconds": 120.5,
}

MOCK_METASTATIC_CT_RESULT = {
    "lesions": [
        {"level": "T10", "type": "lytic", "volume_mm3": 450.2, "body_involvement_pct": 35},
        {"level": "L2", "type": "blastic", "volume_mm3": 280.1, "body_involvement_pct": 20},
    ],
    "total_lesions": 2,
}

MOCK_SINS_RESULT = {
    "levels": {"T10": {"total_score": 10, "classification": "potentially_unstable"}},
    "summary": {"max_score": 10},
}

MOCK_LISTHESIS_RESULT = {
    "levels": {"L4": {"translation_mm": 2.1, "meyerding_grade": "I"}},
}

MOCK_CANAL_RESULT = {
    "levels": {"T10": {"ap_diameter_mm": 11.5, "stenosis_grade": "moderate"}},
}

MOCK_BONE_RESULT = {
    "levels": {
        "L1": {"mean_hu": 95.3, "classification": "osteopenic"},
        "L2": {"mean_hu": 120.5, "classification": "normal"},
        "L3": {"mean_hu": 88.1, "classification": "osteopenic"},
    },
    "summary": {"osteoporotic_count": 0, "osteopenic_count": 2, "normal_count": 1},
}

MOCK_METASTATIC_MRI_RESULT = {
    "lesions": [
        {"level": "T10", "type": "lytic", "t1_signal": "low", "t2_signal": "high"},
    ],
    "total_lesions": 1,
}

MOCK_SCREENSHOT_RESULT = {
    "success": True,
    "image": "base64encodedpng...",
    "view_type": "sagittal",
}

_PATCH_PREFIX = "slicer_mcp.features.workflows.onco_spine"


# ================================================================
# Shared fixture for CT-only pipeline mocks
# ================================================================


@pytest.fixture()
def onco_ct_mocks():
    """Patch all CT-pipeline tools and return mocks in a namespace."""
    with (
        patch(f"{_PATCH_PREFIX}.segment_spine") as m_segment,
        patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct") as m_met_ct,
        patch(f"{_PATCH_PREFIX}.calculate_sins_score") as m_sins,
        patch(f"{_PATCH_PREFIX}.measure_listhesis_ct") as m_listhesis,
        patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct") as m_canal,
        patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct") as m_bone,
        patch(f"{_PATCH_PREFIX}.capture_screenshot") as m_screenshot,
    ):
        m_segment.return_value = MOCK_SEGMENT_RESULT
        m_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        m_sins.return_value = MOCK_SINS_RESULT
        m_listhesis.return_value = MOCK_LISTHESIS_RESULT
        m_canal.return_value = MOCK_CANAL_RESULT
        m_bone.return_value = MOCK_BONE_RESULT
        m_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        yield SimpleNamespace(
            segment=m_segment,
            met_ct=m_met_ct,
            sins=m_sins,
            listhesis=m_listhesis,
            canal=m_canal,
            bone=m_bone,
            screenshot=m_screenshot,
        )


# ================================================================
# Input Validation Tests
# ================================================================


class TestWorkflowOncoSpineValidation:
    """Test input validation for workflow_onco_spine."""

    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"ct_volume_id": "1invalid"}, "node_id"),
            ({"ct_volume_id": "vtkMRMLScalarVolumeNode1", "region": "sacral"}, "region"),
            ({"ct_volume_id": "vtkMRMLScalarVolumeNode1", "pain_type": "sharp"}, "pain_type"),
            ({"ct_volume_id": "vtkMRMLScalarVolumeNode1", "t1_volume_id": "DROP TABLE"}, "node_id"),
            ({"ct_volume_id": "vtkMRMLScalarVolumeNode1", "t2_volume_id": "123bad"}, "node_id"),
            (
                {
                    "ct_volume_id": "vtkMRMLScalarVolumeNode1",
                    "segmentation_node_id": "123bad",
                },
                "node_id",
            ),
        ],
    )
    def test_invalid_inputs(self, kwargs, field):
        with pytest.raises(ValidationError) as exc_info:
            workflow_onco_spine(**kwargs)
        assert exc_info.value.field == field

    def test_empty_ct_node_id(self):
        with pytest.raises(ValidationError):
            workflow_onco_spine(ct_volume_id="")


# ================================================================
# Happy Path Tests
# ================================================================


class TestWorkflowOncoSpineHappyPath:
    """Test happy-path execution of workflow_onco_spine."""

    def test_full_pipeline(self, onco_ct_mocks):
        """Full pipeline with segmentation, no MRI."""
        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            region="full",
        )

        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode1"
        assert result["metastatic_lesions_ct"] == MOCK_METASTATIC_CT_RESULT
        assert result["sins_scores"] == MOCK_SINS_RESULT
        assert result["listhesis"] == MOCK_LISTHESIS_RESULT
        assert result["canal_stenosis"] == MOCK_CANAL_RESULT
        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert result["metastatic_lesions_mri"] is None
        assert result["region"] == "full"
        assert result["pain_type"] is None
        assert len(result["screenshots"]) == 1

        expected_steps = {
            "segment_spine",
            "detect_metastatic_lesions_ct",
            "calculate_sins_score",
            "measure_listhesis_ct",
            "measure_spinal_canal_ct",
            "assess_osteoporosis_ct",
            "capture_screenshot",
        }
        assert expected_steps.issubset(set(result["steps_completed"]))
        assert "detect_metastatic_lesions_mri" not in result["steps_completed"]

        onco_ct_mocks.segment.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode1",
            region="full",
            include_discs=True,
            include_spinal_cord=False,
        )

    def test_existing_segmentation_skips_segment(self, onco_ct_mocks):
        """Pre-existing segmentation skips segment_spine."""
        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id="vtkMRMLSegmentationNode5",
        )

        onco_ct_mocks.segment.assert_not_called()
        assert result["segmentation_node_id"] == "vtkMRMLSegmentationNode5"
        assert "segment_spine_skipped" in result["steps_completed"]
        assert "segment_spine" not in result["steps_completed"]

    def test_with_mri(self, onco_ct_mocks):
        """MRI analysis runs when both T1 and T2 are provided."""
        with patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_mri") as m_met_mri:
            m_met_mri.return_value = MOCK_METASTATIC_MRI_RESULT

            result = workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                t1_volume_id="vtkMRMLScalarVolumeNode2",
                t2_volume_id="vtkMRMLScalarVolumeNode3",
            )

            assert result["metastatic_lesions_mri"] == MOCK_METASTATIC_MRI_RESULT
            assert "detect_metastatic_lesions_mri" in result["steps_completed"]

            m_met_mri.assert_called_once_with(
                t1_node_id="vtkMRMLScalarVolumeNode2",
                t2_stir_node_id="vtkMRMLScalarVolumeNode3",
                region="full",
                segmentation_node_id="vtkMRMLSegmentationNode1",
            )

    def test_without_mri(self, onco_ct_mocks):
        """MRI analysis skipped when T1/T2 not provided."""
        result = workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

        assert result["metastatic_lesions_mri"] is None
        assert "detect_metastatic_lesions_mri" not in result["steps_completed"]

    def test_only_t1_no_mri_analysis(self, onco_ct_mocks):
        """MRI analysis skipped when only T1 provided."""
        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            t1_volume_id="vtkMRMLScalarVolumeNode2",
        )

        assert result["metastatic_lesions_mri"] is None
        assert "detect_metastatic_lesions_mri" not in result["steps_completed"]

    def test_with_pain_type(self, onco_ct_mocks):
        """Pain type is passed to SINS calculation."""
        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            pain_type="mechanical",
        )

        assert result["pain_type"] == "mechanical"
        onco_ct_mocks.sins.assert_called_once_with(
            volume_node_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id="vtkMRMLSegmentationNode1",
            pain_score=3,
        )

    def test_without_pain_type(self, onco_ct_mocks):
        """SINS called without pain_score when pain_type is None."""
        workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

        onco_ct_mocks.sins.assert_called_once_with(
            volume_node_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id="vtkMRMLSegmentationNode1",
        )


# ================================================================
# Error Propagation Tests
# ================================================================


class TestWorkflowOncoSpineErrors:
    """Test error propagation from underlying tools."""

    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_segment_spine_error_propagates(self, mock_segment):
        mock_segment.side_effect = SlicerConnectionError("Slicer not responding")
        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_metastatic_ct_error_propagates(self, mock_segment, mock_met_ct):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.side_effect = SlicerConnectionError("Lesion detection failed")
        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_sins_error_propagates(self, mock_segment, mock_met_ct, mock_sins):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.side_effect = SlicerConnectionError("SINS calculation failed")
        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_listhesis_error_propagates(self, mock_segment, mock_met_ct, mock_sins, mock_listhesis):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.side_effect = SlicerConnectionError("Listhesis measurement failed")
        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_canal_error_propagates(
        self, mock_segment, mock_met_ct, mock_sins, mock_listhesis, mock_canal
    ):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.side_effect = SlicerConnectionError("Canal measurement failed")
        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_bone_quality_error_propagates(
        self, mock_segment, mock_met_ct, mock_sins, mock_listhesis, mock_canal, mock_bone
    ):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.side_effect = SlicerConnectionError("Osteoporosis assessment failed")
        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_mri")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_mri_lesion_error_propagates(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_met_mri,
    ):
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_met_mri.side_effect = SlicerConnectionError("MRI lesion detection failed")
        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                t1_volume_id="vtkMRMLScalarVolumeNode2",
                t2_volume_id="vtkMRMLScalarVolumeNode3",
            )

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
    def test_screenshot_failure_is_nonfatal(self, onco_ct_mocks, error):
        """Screenshot failures of any type do not abort the workflow."""
        onco_ct_mocks.screenshot.side_effect = error

        result = workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")

        assert result["metastatic_lesions_ct"] == MOCK_METASTATIC_CT_RESULT
        assert result["sins_scores"] == MOCK_SINS_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    def test_validation_error_from_tool_propagates(self, mock_met_ct):
        mock_met_ct.side_effect = ValidationError("Bad input", "volume_node_id", "bad")
        with pytest.raises(ValidationError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                segmentation_node_id="vtkMRMLSegmentationNode1",
            )


# ================================================================
# Result Structure Tests
# ================================================================


class TestWorkflowOncoSpineResultStructure:
    """Test that the result dict has the expected keys."""

    def test_result_has_all_expected_keys(self, onco_ct_mocks):
        result = workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")
        expected_keys = {
            "segmentation_node_id",
            "metastatic_lesions_ct",
            "sins_scores",
            "listhesis",
            "canal_stenosis",
            "bone_quality",
            "metastatic_lesions_mri",
            "screenshots",
            "region",
            "pain_type",
            "steps_completed",
        }
        assert set(result.keys()) == expected_keys

    def test_default_region_is_full(self, onco_ct_mocks):
        result = workflow_onco_spine(ct_volume_id="vtkMRMLScalarVolumeNode1")
        assert result["region"] == "full"
