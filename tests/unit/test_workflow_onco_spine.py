"""Unit tests for the workflow_onco_spine workflow tool."""

from unittest.mock import patch

import pytest

from slicer_mcp.core.circuit_breaker import CircuitOpenError
from slicer_mcp.core.slicer_client import (
    SlicerConnectionError,
    SlicerTimeoutError,
)
from slicer_mcp.features.base_tools import ValidationError
from slicer_mcp.features.workflows.onco_spine import (
    VALID_ONCO_REGIONS,
    VALID_PAIN_TYPES,
    _validate_pain_type,
    _validate_region,
    workflow_onco_spine,
)

# ================================================================
# Region Validation Tests
# ================================================================


class TestValidateRegion:
    """Test region parameter validation for onco-spine."""

    def test_valid_cervical(self):
        assert _validate_region("cervical") == "cervical"

    def test_valid_thoracic(self):
        assert _validate_region("thoracic") == "thoracic"

    def test_valid_lumbar(self):
        assert _validate_region("lumbar") == "lumbar"

    def test_valid_full(self):
        assert _validate_region("full") == "full"

    def test_invalid_empty(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region("")
        assert exc_info.value.field == "region"

    def test_invalid_arbitrary(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_region("sacral")
        assert exc_info.value.field == "region"


# ================================================================
# Pain Type Validation Tests
# ================================================================


class TestValidatePainType:
    """Test pain_type parameter validation."""

    def test_valid_mechanical(self):
        assert _validate_pain_type("mechanical") == "mechanical"

    def test_valid_non_mechanical(self):
        result = _validate_pain_type("non_mechanical")
        assert result == "non_mechanical"

    def test_valid_none(self):
        assert _validate_pain_type(None) is None

    def test_invalid_pain_type(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_pain_type("sharp")
        assert exc_info.value.field == "pain_type"

    def test_invalid_empty_string(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate_pain_type("")
        assert exc_info.value.field == "pain_type"


# ================================================================
# Constants Tests
# ================================================================


class TestConstants:
    """Test onco-spine workflow constants."""

    def test_valid_regions_contains_full(self):
        assert "full" in VALID_ONCO_REGIONS

    def test_valid_regions_contains_cervical(self):
        assert "cervical" in VALID_ONCO_REGIONS

    def test_valid_regions_contains_thoracic(self):
        assert "thoracic" in VALID_ONCO_REGIONS

    def test_valid_regions_contains_lumbar(self):
        assert "lumbar" in VALID_ONCO_REGIONS

    def test_valid_pain_types(self):
        assert VALID_PAIN_TYPES == {
            "mechanical",
            "non_mechanical",
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
        {
            "level": "T10",
            "type": "lytic",
            "volume_mm3": 450.2,
            "body_involvement_pct": 35,
        },
        {
            "level": "L2",
            "type": "blastic",
            "volume_mm3": 280.1,
            "body_involvement_pct": 20,
        },
    ],
    "total_lesions": 2,
}

MOCK_SINS_RESULT = {
    "levels": {
        "T10": {
            "total_score": 10,
            "classification": "potentially_unstable",
        },
    },
    "summary": {"max_score": 10},
}

MOCK_LISTHESIS_RESULT = {
    "levels": {
        "L4": {
            "translation_mm": 2.1,
            "meyerding_grade": "I",
        },
    },
}

MOCK_CANAL_RESULT = {
    "levels": {
        "T10": {
            "ap_diameter_mm": 11.5,
            "stenosis_grade": "moderate",
        },
    },
}

MOCK_BONE_RESULT = {
    "levels": {
        "L1": {
            "mean_hu": 95.3,
            "classification": "osteopenic",
        },
    },
}

MOCK_METASTATIC_MRI_RESULT = {
    "lesions": [
        {
            "level": "T10",
            "type": "lytic",
            "t1_signal": "low",
            "t2_signal": "high",
        },
    ],
    "total_lesions": 1,
}

MOCK_SCREENSHOT_RESULT = {
    "success": True,
    "image": "base64encodedpng...",
    "view_type": "sagittal",
}


# ================================================================
# Input Validation Tests
# ================================================================


class TestWorkflowOncoSpineValidation:
    """Test input validation for workflow_onco_spine."""

    def test_invalid_ct_node_id(self):
        """Invalid CT node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_onco_spine(ct_volume_id="1invalid")
        assert exc_info.value.field == "node_id"

    def test_invalid_region(self):
        """Invalid region raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                region="sacral",
            )
        assert exc_info.value.field == "region"

    def test_invalid_pain_type(self):
        """Invalid pain_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                pain_type="sharp",
            )
        assert exc_info.value.field == "pain_type"

    def test_invalid_t1_node_id(self):
        """Invalid T1 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                t1_volume_id="DROP TABLE",
            )
        assert exc_info.value.field == "node_id"

    def test_invalid_t2_node_id(self):
        """Invalid T2 node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                t2_volume_id="123bad",
            )
        assert exc_info.value.field == "node_id"

    def test_invalid_segmentation_node_id(self):
        """Invalid segmentation node ID raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                segmentation_node_id="123bad",
            )
        assert exc_info.value.field == "node_id"

    def test_empty_ct_node_id(self):
        """Empty CT node ID raises ValidationError."""
        with pytest.raises(ValidationError):
            workflow_onco_spine(ct_volume_id="")


# ================================================================
# Happy Path Tests
# ================================================================

_PATCH_PREFIX = "slicer_mcp.features.workflows.onco_spine"


class TestWorkflowOncoSpineHappyPath:
    """Test happy-path execution of workflow_onco_spine."""

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_full_pipeline(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """Full pipeline with segmentation, no MRI."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            region="full",
        )

        seg_id = "vtkMRMLSegmentationNode1"
        assert result["segmentation_node_id"] == seg_id
        assert result["metastatic_lesions_ct"] == MOCK_METASTATIC_CT_RESULT
        assert result["sins_scores"] == MOCK_SINS_RESULT
        assert result["listhesis"] == MOCK_LISTHESIS_RESULT
        assert result["canal_stenosis"] == MOCK_CANAL_RESULT
        assert result["bone_quality"] == MOCK_BONE_RESULT
        assert result["metastatic_lesions_mri"] is None
        assert result["region"] == "full"
        assert result["pain_type"] is None
        assert len(result["screenshots"]) == 1
        assert "segment_spine" in result["steps_completed"]
        assert "detect_metastatic_lesions_ct" in result["steps_completed"]
        assert "calculate_sins_score" in result["steps_completed"]
        assert "measure_listhesis_ct" in result["steps_completed"]
        assert "measure_spinal_canal_ct" in result["steps_completed"]
        assert "assess_osteoporosis_ct" in result["steps_completed"]
        assert "capture_screenshot" in result["steps_completed"]
        assert "detect_metastatic_lesions_mri" not in result["steps_completed"]

        mock_segment.assert_called_once_with(
            input_node_id="vtkMRMLScalarVolumeNode1",
            region="full",
            include_discs=True,
            include_spinal_cord=False,
        )

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_existing_segmentation_skips_segment(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """Pre-existing segmentation skips segment_spine."""
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id="vtkMRMLSegmentationNode5",
        )

        mock_segment.assert_not_called()
        seg_id = "vtkMRMLSegmentationNode5"
        assert result["segmentation_node_id"] == seg_id
        assert "segment_spine_skipped" in result["steps_completed"]
        assert "segment_spine" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_mri")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_with_mri(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_met_mri,
        mock_screenshot,
    ):
        """MRI analysis runs when both T1 and T2 are provided."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_met_mri.return_value = MOCK_METASTATIC_MRI_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            t1_volume_id="vtkMRMLScalarVolumeNode2",
            t2_volume_id="vtkMRMLScalarVolumeNode3",
        )

        assert result["metastatic_lesions_mri"] == MOCK_METASTATIC_MRI_RESULT
        assert "detect_metastatic_lesions_mri" in result["steps_completed"]

        mock_met_mri.assert_called_once_with(
            t1_node_id="vtkMRMLScalarVolumeNode2",
            t2_stir_node_id="vtkMRMLScalarVolumeNode3",
            region="full",
            segmentation_node_id=("vtkMRMLSegmentationNode1"),
        )

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_without_mri(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """MRI analysis skipped when T1/T2 not provided."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["metastatic_lesions_mri"] is None
        assert "detect_metastatic_lesions_mri" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_only_t1_no_mri_analysis(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """MRI analysis skipped when only T1 provided."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            t1_volume_id="vtkMRMLScalarVolumeNode2",
        )

        assert result["metastatic_lesions_mri"] is None
        assert "detect_metastatic_lesions_mri" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_with_pain_type(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """Pain type is passed to SINS calculation."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
            pain_type="mechanical",
        )

        assert result["pain_type"] == "mechanical"

        mock_sins.assert_called_once_with(
            volume_node_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id=("vtkMRMLSegmentationNode1"),
            pain_score="mechanical",
        )

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_without_pain_type(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """SINS called without pain_score when pain_type is None."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        mock_sins.assert_called_once_with(
            volume_node_id="vtkMRMLScalarVolumeNode1",
            segmentation_node_id=("vtkMRMLSegmentationNode1"),
        )


# ================================================================
# Error Propagation Tests
# ================================================================


class TestWorkflowOncoSpineErrors:
    """Test error propagation from underlying tools."""

    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_segment_spine_error_propagates(self, mock_segment):
        """SlicerConnectionError from segment_spine propagates."""
        mock_segment.side_effect = SlicerConnectionError("Slicer not responding")

        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_metastatic_ct_error_propagates(self, mock_segment, mock_met_ct):
        """SlicerConnectionError from detect_metastatic_lesions_ct propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.side_effect = SlicerConnectionError("Lesion detection failed")

        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_sins_error_propagates(self, mock_segment, mock_met_ct, mock_sins):
        """SlicerConnectionError from calculate_sins_score propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.side_effect = SlicerConnectionError("SINS calculation failed")

        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_listhesis_error_propagates(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
    ):
        """SlicerConnectionError from measure_listhesis_ct propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.side_effect = SlicerConnectionError("Listhesis measurement failed")

        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_canal_error_propagates(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
    ):
        """SlicerConnectionError from measure_spinal_canal_ct propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.side_effect = SlicerConnectionError("Canal measurement failed")

        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_bone_quality_error_propagates(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
    ):
        """SlicerConnectionError from assess_osteoporosis_ct propagates."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.side_effect = SlicerConnectionError("Osteoporosis assessment failed")

        with pytest.raises(SlicerConnectionError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
            )

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
        """SlicerConnectionError from detect_metastatic_lesions_mri propagates."""
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

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_screenshot_failure_is_nonfatal(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """Screenshot failure does not abort the workflow."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = SlicerConnectionError("Screenshot failed")

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["metastatic_lesions_ct"] == MOCK_METASTATIC_CT_RESULT
        assert result["sins_scores"] == MOCK_SINS_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_screenshot_timeout_is_nonfatal(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """SlicerTimeoutError from screenshot is non-fatal."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = SlicerTimeoutError("Screenshot timed out")

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["metastatic_lesions_ct"] == MOCK_METASTATIC_CT_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_screenshot_circuit_open_is_nonfatal(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """CircuitOpenError from screenshot is non-fatal."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = CircuitOpenError("Circuit breaker is open", "slicer", 30.0)

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["metastatic_lesions_ct"] == MOCK_METASTATIC_CT_RESULT
        assert len(result["screenshots"]) == 0
        assert "capture_screenshot" not in result["steps_completed"]

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_screenshot_value_error_is_nonfatal(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """ValueError from screenshot is non-fatal."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.side_effect = ValueError("Invalid view type")

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["metastatic_lesions_ct"] == MOCK_METASTATIC_CT_RESULT
        assert len(result["screenshots"]) == 0

    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    def test_validation_error_from_tool_propagates(self, mock_met_ct):
        """ValidationError raised inside a tool propagates."""
        mock_met_ct.side_effect = ValidationError("Bad input", "volume_node_id", "bad")

        with pytest.raises(ValidationError):
            workflow_onco_spine(
                ct_volume_id="vtkMRMLScalarVolumeNode1",
                segmentation_node_id=("vtkMRMLSegmentationNode1"),
            )


# ================================================================
# Result Structure Tests
# ================================================================


class TestWorkflowOncoSpineResultStructure:
    """Test that the result dict has the expected keys."""

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_result_has_all_expected_keys(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """Result dict contains all documented keys."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

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

    @patch(f"{_PATCH_PREFIX}.capture_screenshot")
    @patch(f"{_PATCH_PREFIX}.assess_osteoporosis_ct")
    @patch(f"{_PATCH_PREFIX}.measure_spinal_canal_ct")
    @patch(f"{_PATCH_PREFIX}.measure_listhesis_ct")
    @patch(f"{_PATCH_PREFIX}.calculate_sins_score")
    @patch(f"{_PATCH_PREFIX}.detect_metastatic_lesions_ct")
    @patch(f"{_PATCH_PREFIX}.segment_spine")
    def test_default_region_is_full(
        self,
        mock_segment,
        mock_met_ct,
        mock_sins,
        mock_listhesis,
        mock_canal,
        mock_bone,
        mock_screenshot,
    ):
        """Default region should be full when not specified."""
        mock_segment.return_value = MOCK_SEGMENT_RESULT
        mock_met_ct.return_value = MOCK_METASTATIC_CT_RESULT
        mock_sins.return_value = MOCK_SINS_RESULT
        mock_listhesis.return_value = MOCK_LISTHESIS_RESULT
        mock_canal.return_value = MOCK_CANAL_RESULT
        mock_bone.return_value = MOCK_BONE_RESULT
        mock_screenshot.return_value = MOCK_SCREENSHOT_RESULT

        result = workflow_onco_spine(
            ct_volume_id="vtkMRMLScalarVolumeNode1",
        )

        assert result["region"] == "full"
