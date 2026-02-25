"""Unit tests for spine tool implementations."""

from unittest.mock import Mock, patch

import pytest

from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.spine_tools import _build_spine_segmentation_code, segment_spine
from slicer_mcp.tools import ValidationError

# =============================================================================
# Input Validation Tests
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
# Code Generation Tests
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
# segment_spine Execution Tests (Mocked Client)
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
# Server Registration Tests
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
