"""Unit tests for spine-specific MCP tool implementations."""

from unittest.mock import Mock, patch

import pytest

from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.spine_tools import (
    VALID_ARTERY_SIDES,
    VALID_BONE_REGIONS,
    _build_spine_segmentation_code,
    _validate_seed_points,
    analyze_bone_quality,
    segment_spine,
    segment_vertebral_artery,
)
from slicer_mcp.tools import ValidationError

# =============================================================================
# Spine Segmentation – Input Validation Tests
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
# Spine Segmentation – Code Generation Tests
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
# Spine Segmentation – Execution Tests (Mocked Client)
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
# Spine Segmentation – Server Registration Tests
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
# Vertebral Artery Segmentation – Validation Tests
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
# Vertebral Artery Segmentation – Execution Tests
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
# Bone Quality Analysis – Validation Tests
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
# Bone Quality Analysis – Execution Tests
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
