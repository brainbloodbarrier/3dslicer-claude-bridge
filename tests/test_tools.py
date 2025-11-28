"""Unit tests for MCP tool implementations."""

import pytest
from unittest.mock import Mock, patch

from slicer_mcp.tools import (
    validate_mrml_node_id,
    validate_segment_name,
    ValidationError,
    measure_volume,
)


class TestValidateMrmlNodeId:
    """Test MRML node ID validation."""

    def test_valid_node_id_simple(self):
        """Test valid simple node ID."""
        result = validate_mrml_node_id("vtkMRMLScalarVolumeNode1")
        assert result == "vtkMRMLScalarVolumeNode1"

    def test_valid_node_id_with_underscore(self):
        """Test valid node ID with underscore."""
        result = validate_mrml_node_id("vtkMRMLSegmentationNode_1")
        assert result == "vtkMRMLSegmentationNode_1"

    def test_valid_node_id_letters_only(self):
        """Test valid node ID with letters only."""
        result = validate_mrml_node_id("MyCustomNode")
        assert result == "MyCustomNode"

    def test_invalid_node_id_empty(self):
        """Test empty node ID raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_mrml_node_id("")
        assert exc_info.value.field == "node_id"
        assert "cannot be empty" in str(exc_info.value)

    def test_invalid_node_id_starts_with_number(self):
        """Test node ID starting with number raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_mrml_node_id("1vtkNode")
        assert exc_info.value.field == "node_id"

    def test_invalid_node_id_special_chars(self):
        """Test node ID with special characters raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_mrml_node_id("node'; DROP TABLE;")
        assert exc_info.value.field == "node_id"

    def test_invalid_node_id_injection_attempt_quotes(self):
        """Test code injection attempt with quotes is blocked."""
        injection = "'); import subprocess; ('"
        with pytest.raises(ValidationError) as exc_info:
            validate_mrml_node_id(injection)
        assert exc_info.value.field == "node_id"

    def test_invalid_node_id_injection_attempt_semicolon(self):
        """Test code injection attempt with semicolon is blocked."""
        injection = "node; malicious_code()"
        with pytest.raises(ValidationError) as exc_info:
            validate_mrml_node_id(injection)
        assert exc_info.value.field == "node_id"

    def test_invalid_node_id_too_long(self):
        """Test node ID exceeding max length raises error."""
        long_id = "v" * 300
        with pytest.raises(ValidationError) as exc_info:
            validate_mrml_node_id(long_id)
        assert "maximum length" in str(exc_info.value)


class TestValidateSegmentName:
    """Test segment name validation."""

    def test_valid_segment_name_simple(self):
        """Test valid simple segment name."""
        result = validate_segment_name("Tumor")
        assert result == "Tumor"

    def test_valid_segment_name_with_space(self):
        """Test valid segment name with space."""
        result = validate_segment_name("Left Lung")
        assert result == "Left Lung"

    def test_valid_segment_name_with_hyphen(self):
        """Test valid segment name with hyphen."""
        result = validate_segment_name("Brain-Stem")
        assert result == "Brain-Stem"

    def test_valid_segment_name_with_underscore(self):
        """Test valid segment name with underscore."""
        result = validate_segment_name("Segment_1")
        assert result == "Segment_1"

    def test_valid_segment_name_numbers(self):
        """Test valid segment name starting with number."""
        result = validate_segment_name("1st Vertebra")
        assert result == "1st Vertebra"

    def test_invalid_segment_name_empty(self):
        """Test empty segment name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_segment_name("")
        assert exc_info.value.field == "segment_name"

    def test_invalid_segment_name_special_chars(self):
        """Test segment name with special characters raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_segment_name("Tumor'; DROP TABLE;")
        assert exc_info.value.field == "segment_name"

    def test_invalid_segment_name_injection_attempt(self):
        """Test code injection attempt is blocked."""
        injection = "'); import subprocess; ('"
        with pytest.raises(ValidationError) as exc_info:
            validate_segment_name(injection)
        assert exc_info.value.field == "segment_name"

    def test_invalid_segment_name_too_long(self):
        """Test segment name exceeding max length raises error."""
        long_name = "A" * 300
        with pytest.raises(ValidationError) as exc_info:
            validate_segment_name(long_name)
        assert "maximum length" in str(exc_info.value)


class TestMeasureVolumeValidation:
    """Test measure_volume input validation integration."""

    def test_measure_volume_invalid_node_id(self):
        """Test measure_volume rejects invalid node_id."""
        with pytest.raises(ValidationError) as exc_info:
            measure_volume("invalid'; DROP TABLE;")
        assert exc_info.value.field == "node_id"

    def test_measure_volume_invalid_segment_name(self):
        """Test measure_volume rejects invalid segment_name."""
        with pytest.raises(ValidationError) as exc_info:
            measure_volume("vtkMRMLSegmentationNode1", "bad'; injection")
        assert exc_info.value.field == "segment_name"

    def test_measure_volume_valid_inputs_proceeds(self):
        """Test measure_volume with valid inputs proceeds to Slicer call."""
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"node_id": "vtkMRMLSegmentationNode1", "node_name": "Test", "total_volume_mm3": 1000, "total_volume_ml": 1.0, "segments": []}'
            }
            mock_get_client.return_value = mock_client

            result = measure_volume("vtkMRMLSegmentationNode1")

            assert result["node_id"] == "vtkMRMLSegmentationNode1"
            mock_client.exec_python.assert_called_once()
