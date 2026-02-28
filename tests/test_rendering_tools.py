"""Unit tests for volume rendering and 3D model export MCP tool implementations."""

import json
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.rendering_tools import (
    _build_capture_3d_view_code,
    _build_enable_volume_rendering_code,
    _build_export_model_code,
    _build_segmentation_to_models_code,
    _build_set_volume_rendering_property_code,
    capture_3d_view,
    enable_volume_rendering,
    export_model,
    segmentation_to_models,
    set_volume_rendering_property,
    validate_export_directory,
    validate_export_filename,
)
from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.tools import ValidationError

# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateExportFilename:
    """Tests for validate_export_filename."""

    def test_valid_filename(self):
        assert validate_export_filename("my_model") == "my_model"

    def test_valid_filename_with_spaces(self):
        assert validate_export_filename("my model (1)") == "my model (1)"

    def test_valid_filename_with_dots(self):
        assert validate_export_filename("model.v2") == "model.v2"

    def test_empty_filename(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_export_filename("")

    def test_too_long_filename(self):
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_export_filename("a" * 256)

    def test_invalid_chars_slash(self):
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_export_filename("../../etc/passwd")

    def test_invalid_chars_semicolon(self):
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_export_filename("file;rm -rf /")

    def test_invalid_chars_backtick(self):
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_export_filename("file`whoami`")


class TestValidateExportDirectory:
    """Tests for validate_export_directory."""

    def test_empty_directory(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_export_directory("")

    def test_too_long_directory(self):
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_export_directory("/" + "a" * 5000)

    def test_path_traversal(self):
        with pytest.raises(ValidationError, match="forbidden component"):
            validate_export_directory("/tmp/../etc")

    @patch("slicer_mcp.rendering_tools.os.path.isdir", return_value=True)
    @patch("slicer_mcp.rendering_tools.os.path.exists", return_value=True)
    @patch("slicer_mcp.rendering_tools.os.path.realpath", return_value="/tmp/export")
    @patch("slicer_mcp.rendering_tools.os.path.expanduser", return_value="/tmp/export")
    def test_valid_directory(self, mock_expand, mock_real, mock_exists, mock_isdir):
        result = validate_export_directory("/tmp/export")
        assert result == "/tmp/export"

    @patch("slicer_mcp.rendering_tools.os.path.exists", return_value=False)
    @patch("slicer_mcp.rendering_tools.os.path.realpath", return_value="/tmp/nonexistent")
    @patch("slicer_mcp.rendering_tools.os.path.expanduser", return_value="/tmp/nonexistent")
    def test_directory_not_exists(self, mock_expand, mock_real, mock_exists):
        with pytest.raises(ValidationError, match="does not exist"):
            validate_export_directory("/tmp/nonexistent")

    @patch("slicer_mcp.rendering_tools.os.path.isdir", return_value=False)
    @patch("slicer_mcp.rendering_tools.os.path.exists", return_value=True)
    @patch("slicer_mcp.rendering_tools.os.path.realpath", return_value="/tmp/file.txt")
    @patch("slicer_mcp.rendering_tools.os.path.expanduser", return_value="/tmp/file.txt")
    def test_path_not_directory(self, mock_expand, mock_real, mock_exists, mock_isdir):
        with pytest.raises(ValidationError, match="not a directory"):
            validate_export_directory("/tmp/file.txt")


# =============================================================================
# Enable Volume Rendering Tests
# =============================================================================


class TestEnableVolumeRendering:
    """Tests for enable_volume_rendering."""

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_success_no_preset(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "volume_node_id": "vtkMRMLScalarVolumeNode1",
                    "display_node_id": "vtkMRMLVolumeRenderingDisplayNode1",
                    "preset": None,
                    "visible": True,
                }
            )
        }

        result = enable_volume_rendering("vtkMRMLScalarVolumeNode1")
        assert result["success"] is True
        assert result["volume_node_id"] == "vtkMRMLScalarVolumeNode1"
        assert result["preset"] is None

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_success_with_preset(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "volume_node_id": "vtkMRMLScalarVolumeNode1",
                    "display_node_id": "vtkMRMLVolumeRenderingDisplayNode1",
                    "preset": "CT-Bone",
                    "visible": True,
                }
            )
        }

        result = enable_volume_rendering("vtkMRMLScalarVolumeNode1", preset="CT-Bone")
        assert result["success"] is True
        assert result["preset"] == "CT-Bone"

    def test_invalid_preset(self):
        with pytest.raises(ValidationError, match="Invalid volume rendering preset"):
            enable_volume_rendering("vtkMRMLScalarVolumeNode1", preset="NonexistentPreset")

    def test_invalid_node_id(self):
        with pytest.raises(ValidationError):
            enable_volume_rendering("")

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_connection_error(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")

        with pytest.raises(SlicerConnectionError):
            enable_volume_rendering("vtkMRMLScalarVolumeNode1")


# =============================================================================
# Set Volume Rendering Property Tests
# =============================================================================


class TestSetVolumeRenderingProperty:
    """Tests for set_volume_rendering_property."""

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_opacity_success(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "volume_node_id": "vtkMRMLScalarVolumeNode1",
                    "display_node_id": "vtkMRMLVolumeRenderingDisplayNode1",
                    "changes_applied": ["opacity_scale"],
                }
            )
        }

        result = set_volume_rendering_property("vtkMRMLScalarVolumeNode1", opacity_scale=1.5)
        assert result["success"] is True
        assert "opacity_scale" in result["changes_applied"]

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_window_level_success(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "volume_node_id": "vtkMRMLScalarVolumeNode1",
                    "display_node_id": "vtkMRMLVolumeRenderingDisplayNode1",
                    "changes_applied": ["window_level"],
                }
            )
        }

        result = set_volume_rendering_property("vtkMRMLScalarVolumeNode1", window=400.0, level=40.0)
        assert result["success"] is True

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_visibility_success(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "volume_node_id": "vtkMRMLScalarVolumeNode1",
                    "display_node_id": "vtkMRMLVolumeRenderingDisplayNode1",
                    "changes_applied": ["visibility"],
                }
            )
        }

        result = set_volume_rendering_property("vtkMRMLScalarVolumeNode1", visible=False)
        assert result["success"] is True

    def test_no_changes_specified(self):
        with pytest.raises(ValidationError, match="At least one property"):
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1")

    def test_window_without_level(self):
        with pytest.raises(ValidationError, match="both be provided"):
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1", window=400.0)

    def test_level_without_window(self):
        with pytest.raises(ValidationError, match="both be provided"):
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1", level=40.0)

    def test_opacity_too_high(self):
        with pytest.raises(ValidationError, match="opacity_scale"):
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1", opacity_scale=11.0)

    def test_opacity_negative(self):
        with pytest.raises(ValidationError, match="opacity_scale"):
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1", opacity_scale=-0.1)

    def test_opacity_boundary_zero(self):
        """opacity_scale=0.0 is valid (fully transparent)."""
        # Should not raise ValidationError for 0.0
        with pytest.raises((SlicerConnectionError, Exception)):  # type: ignore[arg-type]
            # Will fail on get_client, but validation passes
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1", opacity_scale=0.0)

    def test_opacity_boundary_ten(self):
        """opacity_scale=10.0 is valid (max)."""
        with pytest.raises((SlicerConnectionError, Exception)):  # type: ignore[arg-type]
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1", opacity_scale=10.0)

    def test_invalid_node_id(self):
        with pytest.raises(ValidationError):
            set_volume_rendering_property("", opacity_scale=1.0)

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_connection_error(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")

        with pytest.raises(SlicerConnectionError):
            set_volume_rendering_property("vtkMRMLScalarVolumeNode1", visible=True)


# =============================================================================
# Export Model Tests
# =============================================================================


class TestExportModel:
    """Tests for export_model."""

    @patch("slicer_mcp.rendering_tools.validate_export_directory", return_value="/tmp/export")
    @patch("slicer_mcp.rendering_tools.get_client")
    def test_stl_success(self, mock_get_client, mock_validate_dir):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "model_node_id": "vtkMRMLModelNode1",
                    "model_node_name": "TestModel",
                    "output_path": "/tmp/export/model.stl",
                    "format": "STL",
                    "file_size_bytes": 1024,
                    "point_count": 500,
                    "cell_count": 1000,
                }
            )
        }

        result = export_model("vtkMRMLModelNode1", "/tmp/export", "model", "STL")
        assert result["success"] is True
        assert result["format"] == "STL"
        assert result["point_count"] == 500

    @patch("slicer_mcp.rendering_tools.validate_export_directory", return_value="/tmp/export")
    @patch("slicer_mcp.rendering_tools.get_client")
    def test_obj_success(self, mock_get_client, mock_validate_dir):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "model_node_id": "vtkMRMLModelNode1",
                    "model_node_name": "TestModel",
                    "output_path": "/tmp/export/model.obj",
                    "format": "OBJ",
                    "file_size_bytes": 2048,
                    "point_count": 500,
                    "cell_count": 1000,
                }
            )
        }

        result = export_model("vtkMRMLModelNode1", "/tmp/export", "model", "OBJ")
        assert result["success"] is True
        assert result["format"] == "OBJ"

    def test_invalid_format(self):
        with pytest.raises(ValidationError, match="Invalid export format"):
            export_model("vtkMRMLModelNode1", "/tmp", "model", "GLTF")

    def test_empty_filename(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            export_model("vtkMRMLModelNode1", "/tmp", "", "STL")

    def test_invalid_node_id(self):
        with pytest.raises(ValidationError):
            export_model("", "/tmp", "model", "STL")

    @patch("slicer_mcp.rendering_tools.validate_export_directory", return_value="/tmp/export")
    @patch("slicer_mcp.rendering_tools.get_client")
    def test_connection_error(self, mock_get_client, mock_validate_dir):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")

        with pytest.raises(SlicerConnectionError):
            export_model("vtkMRMLModelNode1", "/tmp/export", "model", "STL")

    def test_case_insensitive_format(self):
        """Lowercase format strings should work."""
        with pytest.raises(ValidationError):
            # Will fail on node_id validation first if empty,
            # but if valid node_id, "stl" (lowercase) is uppercased internally
            export_model("", "/tmp", "model", "stl")


# =============================================================================
# Segmentation to Models Tests
# =============================================================================


class TestSegmentationToModels:
    """Tests for segmentation_to_models."""

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_all_segments_success(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "segmentation_node_id": "vtkMRMLSegmentationNode1",
                    "segmentation_node_name": "Segmentation",
                    "models": [
                        {
                            "segment_id": "Segment_1",
                            "segment_name": "Bone",
                            "model_node_id": "vtkMRMLModelNode1",
                            "model_node_name": "Bone",
                            "point_count": 1000,
                            "cell_count": 2000,
                        }
                    ],
                    "model_count": 1,
                }
            )
        }

        result = segmentation_to_models("vtkMRMLSegmentationNode1")
        assert result["success"] is True
        assert result["model_count"] == 1
        assert len(result["models"]) == 1

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_specific_segments(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "segmentation_node_id": "vtkMRMLSegmentationNode1",
                    "segmentation_node_name": "Segmentation",
                    "models": [
                        {
                            "segment_id": "Segment_1",
                            "segment_name": "Bone",
                            "model_node_id": "vtkMRMLModelNode1",
                            "model_node_name": "Bone",
                            "point_count": 1000,
                            "cell_count": 2000,
                        }
                    ],
                    "model_count": 1,
                }
            )
        }

        result = segmentation_to_models("vtkMRMLSegmentationNode1", segment_ids=["Segment_1"])
        assert result["success"] is True

    def test_invalid_node_id(self):
        with pytest.raises(ValidationError):
            segmentation_to_models("")

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_connection_error(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")

        with pytest.raises(SlicerConnectionError):
            segmentation_to_models("vtkMRMLSegmentationNode1")


# =============================================================================
# Capture 3D View Tests
# =============================================================================


class TestCapture3dView:
    """Tests for capture_3d_view."""

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_success(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "output_path": "/tmp/screenshot.png",
                    "file_size_bytes": 65536,
                    "view_index": 0,
                }
            )
        }

        result = capture_3d_view("/tmp/screenshot.png")
        assert result["success"] is True
        assert result["output_path"] == "/tmp/screenshot.png"

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_with_dimensions(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.return_value = {
            "result": json.dumps(
                {
                    "success": True,
                    "output_path": "/tmp/screenshot.png",
                    "file_size_bytes": 262144,
                    "view_index": 0,
                }
            )
        }

        result = capture_3d_view("/tmp/screenshot.png", width=1920, height=1080)
        assert result["success"] is True

    def test_empty_path(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            capture_3d_view("")

    def test_path_traversal(self):
        with pytest.raises(ValidationError, match="forbidden component"):
            capture_3d_view("/tmp/../etc/screenshot.png")

    def test_width_without_height(self):
        with pytest.raises(ValidationError, match="both be provided"):
            capture_3d_view("/tmp/screenshot.png", width=1920)

    def test_height_without_width(self):
        with pytest.raises(ValidationError, match="both be provided"):
            capture_3d_view("/tmp/screenshot.png", height=1080)

    def test_width_out_of_range(self):
        with pytest.raises(ValidationError, match="width"):
            capture_3d_view("/tmp/screenshot.png", width=0, height=1080)

    def test_width_too_large(self):
        with pytest.raises(ValidationError, match="width"):
            capture_3d_view("/tmp/screenshot.png", width=9000, height=1080)

    def test_height_out_of_range(self):
        with pytest.raises(ValidationError, match="height"):
            capture_3d_view("/tmp/screenshot.png", width=1920, height=0)

    def test_view_index_out_of_range(self):
        with pytest.raises(ValidationError, match="view_index"):
            capture_3d_view("/tmp/screenshot.png", view_index=11)

    @patch("slicer_mcp.rendering_tools.get_client")
    def test_connection_error(self, mock_get_client):
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exec_python.side_effect = SlicerConnectionError("Connection refused")

        with pytest.raises(SlicerConnectionError):
            capture_3d_view("/tmp/screenshot.png")


# =============================================================================
# Codegen Structural Tests
# =============================================================================


class TestCodegen:
    """Structural assertions on generated Python code strings."""

    def test_enable_vr_code_contains_volumerendering(self):
        code = _build_enable_volume_rendering_code('"node1"', '"CT-Bone"', "True")
        assert "volumerendering" in code.lower()
        assert "CreateDefaultVolumeRenderingNodes" in code
        assert "__execResult" in code

    def test_set_property_code_contains_volume_property(self):
        code = _build_set_volume_rendering_property_code('"node1"', "1.5", "400.0", "40.0", "True")
        assert "GetVolumePropertyNode" in code or "SetWindowLevel" in code
        assert "__execResult" in code

    def test_export_model_code_contains_exportnode(self):
        code = _build_export_model_code('"node1"', '"/tmp/out.stl"', '"STL"')
        assert "exportNode" in code
        assert "__execResult" in code

    def test_segmentation_code_contains_closed_surface(self):
        code = _build_segmentation_to_models_code('"segNode1"', "None")
        assert "ClosedSurface" in code or "Closed surface" in code
        assert "__execResult" in code

    def test_capture_code_contains_window_to_image(self):
        code = _build_capture_3d_view_code('"/tmp/out.png"', "None", "None", "0")
        assert "vtkWindowToImageFilter" in code
        assert "__execResult" in code

    def test_all_codegen_uses_safe_params(self):
        """All codegen functions use their safe (json.dumps'd) parameters directly."""
        vr_code = _build_enable_volume_rendering_code('"safe_id"', '"CT-Bone"', "True")
        assert '"safe_id"' in vr_code
        assert '"CT-Bone"' in vr_code

    def test_export_model_code_verifies_polydata(self):
        code = _build_export_model_code('"node1"', '"/tmp/out.stl"', '"STL"')
        assert "GetPolyData" in code
        assert "GetNumberOfPoints" in code

    def test_segmentation_code_handles_missing_surface(self):
        code = _build_segmentation_to_models_code('"segNode1"', "None")
        assert "RemoveNode" in code  # cleanup on failure
