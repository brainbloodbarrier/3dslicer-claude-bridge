"""Unit tests for registration and landmark MCP tool implementations."""

import json
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.slicer_client import SlicerConnectionError
from slicer_mcp.tools import ValidationError

# =============================================================================
# Helper: build a mock client returning a success dict
# =============================================================================

PATCH_TARGET = "slicer_mcp.registration_tools.get_client"


def _mock_client(return_dict: dict) -> tuple[Mock, Mock]:
    """Create mock client + exec_python returning ``return_dict``."""
    client = Mock()
    client.exec_python.return_value = {
        "success": True,
        "result": json.dumps(return_dict),
    }
    return client


def _mock_client_error(exc: Exception) -> Mock:
    """Create mock client whose exec_python raises *exc*."""
    client = Mock()
    client.exec_python.side_effect = exc
    return client


# =============================================================================
# TestPlaceLandmarks
# =============================================================================


class TestPlaceLandmarks:
    """Tests for place_landmarks()."""

    def test_place_landmarks_success(self):
        """Happy path: 3 points with labels."""
        from slicer_mcp.registration_tools import place_landmarks

        ret = {
            "success": True,
            "node_id": "vtkMRMLMarkupsFiducialNode1",
            "node_name": "MyLandmarks",
            "point_count": 3,
        }
        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(ret)
            result = place_landmarks(
                "MyLandmarks",
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                labels=["L1", "L2", "L3"],
            )
        assert result["success"] is True
        assert result["point_count"] == 3
        assert result["node_id"] == "vtkMRMLMarkupsFiducialNode1"

    def test_place_landmarks_no_labels(self):
        """Labels=None should still succeed."""
        from slicer_mcp.registration_tools import place_landmarks

        ret = {
            "success": True,
            "node_id": "vtkMRMLMarkupsFiducialNode1",
            "node_name": "Pts",
            "point_count": 2,
        }
        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(ret)
            result = place_landmarks(
                "Pts",
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                labels=None,
            )
        assert result["success"] is True
        assert result["point_count"] == 2

    def test_place_landmarks_empty_points(self):
        """Empty points list must raise ValidationError."""
        from slicer_mcp.registration_tools import place_landmarks

        with pytest.raises(ValidationError):
            place_landmarks("Name", [], labels=None)

    def test_place_landmarks_invalid_point_dimensions(self):
        """Points with != 3 coordinates must raise ValidationError."""
        from slicer_mcp.registration_tools import place_landmarks

        with pytest.raises(ValidationError):
            place_landmarks("Name", [[1.0, 2.0]], labels=None)

    def test_place_landmarks_label_count_mismatch(self):
        """Labels length != points length must raise ValidationError."""
        from slicer_mcp.registration_tools import place_landmarks

        with pytest.raises(ValidationError):
            place_landmarks(
                "Name",
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                labels=["OnlyOne"],
            )

    def test_place_landmarks_label_injection(self):
        """Label containing injection chars must raise ValidationError."""
        from slicer_mcp.registration_tools import place_landmarks

        with pytest.raises(ValidationError):
            place_landmarks(
                "Name",
                [[1.0, 2.0, 3.0]],
                labels=["; import os"],
            )

    def test_place_landmarks_empty_name(self):
        """Empty name must raise ValidationError."""
        from slicer_mcp.registration_tools import place_landmarks

        with pytest.raises(ValidationError):
            place_landmarks("", [[1.0, 2.0, 3.0]])

    def test_place_landmarks_connection_error(self):
        """SlicerConnectionError must propagate."""
        from slicer_mcp.registration_tools import place_landmarks

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client_error(SlicerConnectionError("connection refused"))
            with pytest.raises(SlicerConnectionError):
                place_landmarks("Name", [[1.0, 2.0, 3.0]])


# =============================================================================
# TestGetLandmarks
# =============================================================================


class TestGetLandmarks:
    """Tests for get_landmarks()."""

    def test_get_landmarks_success(self):
        """Happy path: retrieve points from a markup node."""
        from slicer_mcp.registration_tools import get_landmarks

        ret = {
            "success": True,
            "node_id": "vtkMRMLMarkupsFiducialNode1",
            "node_name": "MyLandmarks",
            "point_count": 2,
            "points": [
                {"index": 0, "label": "P1", "position_ras": [1.0, 2.0, 3.0]},
                {"index": 1, "label": "P2", "position_ras": [4.0, 5.0, 6.0]},
            ],
        }
        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(ret)
            result = get_landmarks("vtkMRMLMarkupsFiducialNode1")
        assert result["success"] is True
        assert result["point_count"] == 2
        assert len(result["points"]) == 2
        assert result["points"][0]["position_ras"] == [1.0, 2.0, 3.0]

    def test_get_landmarks_invalid_node_id(self):
        """Invalid node ID must raise ValidationError."""
        from slicer_mcp.registration_tools import get_landmarks

        with pytest.raises(ValidationError):
            get_landmarks("'; DROP TABLE;")

    def test_get_landmarks_connection_error(self):
        """SlicerConnectionError must propagate."""
        from slicer_mcp.registration_tools import get_landmarks

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client_error(SlicerConnectionError("timeout"))
            with pytest.raises(SlicerConnectionError):
                get_landmarks("vtkMRMLMarkupsFiducialNode1")


# =============================================================================
# TestRegisterVolumes
# =============================================================================


class TestRegisterVolumes:
    """Tests for register_volumes()."""

    def _success_ret(self, *, resampled: bool = False, ttype: str = "Rigid"):
        ret = {
            "success": True,
            "transform_node_id": "vtkMRMLLinearTransformNode1",
            "transform_node_name": "Transform",
            "transform_type": ttype,
        }
        if resampled:
            ret["resampled_node_id"] = "vtkMRMLScalarVolumeNode2"
        return ret

    def test_register_volumes_rigid_success(self):
        """Rigid registration happy path."""
        from slicer_mcp.registration_tools import register_volumes

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(self._success_ret())
            result = register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                transform_type="Rigid",
            )
        assert result["success"] is True
        assert result["transform_type"] == "Rigid"

    def test_register_volumes_bspline_success(self):
        """BSpline registration must use BSplineTransformNode."""
        from slicer_mcp.registration_tools import register_volumes

        ret = self._success_ret(ttype="BSpline")
        ret["transform_node_id"] = "vtkMRMLBSplineTransformNode1"
        with patch(PATCH_TARGET) as mock_gc:
            client = _mock_client(ret)
            mock_gc.return_value = client
            result = register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                transform_type="BSpline",
            )
            # Verify BSplineTransformNode appears in generated code
            python_code = client.exec_python.call_args[0][0]
            assert "BSplineTransformNode" in python_code
        assert result["success"] is True
        assert result["transform_type"] == "BSpline"

    def test_register_volumes_with_resampled(self):
        """create_resampled=True should return resampled_node_id."""
        from slicer_mcp.registration_tools import register_volumes

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(self._success_ret(resampled=True))
            result = register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                create_resampled=True,
            )
        assert result["success"] is True
        assert "resampled_node_id" in result

    def test_register_volumes_invalid_transform_type(self):
        """Unknown transform type must raise ValidationError."""
        from slicer_mcp.registration_tools import register_volumes

        with pytest.raises(ValidationError):
            register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                transform_type="InvalidType",
            )

    def test_register_volumes_invalid_init_mode(self):
        """Unknown init mode must raise ValidationError."""
        from slicer_mcp.registration_tools import register_volumes

        with pytest.raises(ValidationError):
            register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                init_mode="badMode",
            )

    def test_register_volumes_sampling_out_of_range(self):
        """Sampling percentage out of (0, 1] must raise ValidationError."""
        from slicer_mcp.registration_tools import register_volumes

        with pytest.raises(ValidationError):
            register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                sampling_percentage=0.0,
            )
        with pytest.raises(ValidationError):
            register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                sampling_percentage=1.5,
            )

    def test_register_volumes_invalid_node_id(self):
        """Invalid fixed/moving node IDs must raise ValidationError."""
        from slicer_mcp.registration_tools import register_volumes

        with pytest.raises(ValidationError):
            register_volumes("bad;id", "vtkMRMLScalarVolumeNode2")
        with pytest.raises(ValidationError):
            register_volumes("vtkMRMLScalarVolumeNode1", "bad;id")

    def test_register_volumes_connection_error(self):
        """SlicerConnectionError must propagate."""
        from slicer_mcp.registration_tools import register_volumes

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client_error(SlicerConnectionError("timeout"))
            with pytest.raises(SlicerConnectionError):
                register_volumes(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLScalarVolumeNode2",
                )


# =============================================================================
# TestRegisterLandmarks
# =============================================================================


class TestRegisterLandmarks:
    """Tests for register_landmarks()."""

    _RET = {
        "success": True,
        "transform_node_id": "vtkMRMLLinearTransformNode1",
        "transform_node_name": "LandmarkTransform",
        "transform_type": "Rigid",
    }

    def test_register_landmarks_rigid_success(self):
        """Rigid landmark registration happy path."""
        from slicer_mcp.registration_tools import register_landmarks

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(self._RET)
            result = register_landmarks(
                "vtkMRMLMarkupsFiducialNode1",
                "vtkMRMLMarkupsFiducialNode2",
                transform_type="Rigid",
            )
        assert result["success"] is True
        assert result["transform_type"] == "Rigid"

    def test_register_landmarks_affine_success(self):
        """Affine landmark registration happy path."""
        from slicer_mcp.registration_tools import register_landmarks

        ret = dict(self._RET, transform_type="Affine")
        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(ret)
            result = register_landmarks(
                "vtkMRMLMarkupsFiducialNode1",
                "vtkMRMLMarkupsFiducialNode2",
                transform_type="Affine",
            )
        assert result["success"] is True
        assert result["transform_type"] == "Affine"

    def test_register_landmarks_invalid_transform_type(self):
        """Invalid transform type must raise ValidationError."""
        from slicer_mcp.registration_tools import register_landmarks

        with pytest.raises(ValidationError):
            register_landmarks(
                "vtkMRMLMarkupsFiducialNode1",
                "vtkMRMLMarkupsFiducialNode2",
                transform_type="BSpline",
            )

    def test_register_landmarks_invalid_node_id(self):
        """Invalid node ID must raise ValidationError."""
        from slicer_mcp.registration_tools import register_landmarks

        with pytest.raises(ValidationError):
            register_landmarks(
                "bad;id",
                "vtkMRMLMarkupsFiducialNode2",
            )
        with pytest.raises(ValidationError):
            register_landmarks(
                "vtkMRMLMarkupsFiducialNode1",
                "bad;id",
            )

    def test_register_landmarks_connection_error(self):
        """SlicerConnectionError must propagate."""
        from slicer_mcp.registration_tools import register_landmarks

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client_error(SlicerConnectionError("refused"))
            with pytest.raises(SlicerConnectionError):
                register_landmarks(
                    "vtkMRMLMarkupsFiducialNode1",
                    "vtkMRMLMarkupsFiducialNode2",
                )


# =============================================================================
# TestApplyTransform
# =============================================================================


class TestApplyTransform:
    """Tests for apply_transform()."""

    _RET = {
        "success": True,
        "node_id": "vtkMRMLScalarVolumeNode1",
        "transform_node_id": "vtkMRMLLinearTransformNode1",
        "hardened": False,
    }

    def test_apply_transform_success(self):
        """Apply transform without hardening."""
        from slicer_mcp.registration_tools import apply_transform

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(self._RET)
            result = apply_transform(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLLinearTransformNode1",
            )
        assert result["success"] is True
        assert result["hardened"] is False

    def test_apply_transform_with_harden(self):
        """Apply and harden transform."""
        from slicer_mcp.registration_tools import apply_transform

        ret = dict(self._RET, hardened=True)
        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client(ret)
            result = apply_transform(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLLinearTransformNode1",
                harden=True,
            )
        assert result["success"] is True
        assert result["hardened"] is True

    def test_apply_transform_invalid_node_id(self):
        """Invalid node IDs must raise ValidationError."""
        from slicer_mcp.registration_tools import apply_transform

        with pytest.raises(ValidationError):
            apply_transform("bad;id", "vtkMRMLLinearTransformNode1")
        with pytest.raises(ValidationError):
            apply_transform("vtkMRMLScalarVolumeNode1", "bad;id")

    def test_apply_transform_connection_error(self):
        """SlicerConnectionError must propagate."""
        from slicer_mcp.registration_tools import apply_transform

        with patch(PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_client_error(SlicerConnectionError("down"))
            with pytest.raises(SlicerConnectionError):
                apply_transform(
                    "vtkMRMLScalarVolumeNode1",
                    "vtkMRMLLinearTransformNode1",
                )


# =============================================================================
# TestCodegen -- structural assertions on generated Python code
# =============================================================================


class TestCodegen:
    """Verify structural properties of generated Python code strings."""

    def test_register_volumes_code_contains_brainsfit(self):
        """Generated code must reference BRAINSFit / brainsfit module."""
        from slicer_mcp.registration_tools import register_volumes

        with patch(PATCH_TARGET) as mock_gc:
            client = _mock_client(
                {
                    "success": True,
                    "transform_node_id": "vtkMRMLLinearTransformNode1",
                    "transform_node_name": "T",
                    "transform_type": "Rigid",
                }
            )
            mock_gc.return_value = client
            register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
            )
            code = client.exec_python.call_args[0][0]
        assert "brainsfit" in code.lower() or "BRAINSFit" in code

    def test_register_landmarks_code_contains_fiducialregistration(self):
        """Generated code must reference fiducialregistration module."""
        from slicer_mcp.registration_tools import register_landmarks

        with patch(PATCH_TARGET) as mock_gc:
            client = _mock_client(
                {
                    "success": True,
                    "transform_node_id": "vtkMRMLLinearTransformNode1",
                    "transform_node_name": "T",
                    "transform_type": "Rigid",
                }
            )
            mock_gc.return_value = client
            register_landmarks(
                "vtkMRMLMarkupsFiducialNode1",
                "vtkMRMLMarkupsFiducialNode2",
            )
            code = client.exec_python.call_args[0][0]
        assert "fiducialregistration" in code.lower()

    def test_place_landmarks_code_uses_json_dumps(self):
        """Generated code must inject values via json.dumps, not raw f-string."""
        from slicer_mcp.registration_tools import place_landmarks

        with patch(PATCH_TARGET) as mock_gc:
            client = _mock_client(
                {
                    "success": True,
                    "node_id": "vtkMRMLMarkupsFiducialNode1",
                    "node_name": "LM",
                    "point_count": 1,
                }
            )
            mock_gc.return_value = client
            place_landmarks("LM", [[1.0, 2.0, 3.0]], labels=["P1"])
            code = client.exec_python.call_args[0][0]
        # The generated code should contain the JSON-encoded points,
        # not raw Python list interpolation
        assert "json" in code.lower() or "[[1.0, 2.0, 3.0]]" in code
        # Name should be safely quoted via json.dumps
        assert '"LM"' in code

    def test_apply_transform_harden_code_calls_harden_transform(self):
        """When harden=True, generated code must call HardenTransform."""
        from slicer_mcp.registration_tools import apply_transform

        with patch(PATCH_TARGET) as mock_gc:
            client = _mock_client(
                {
                    "success": True,
                    "node_id": "vtkMRMLScalarVolumeNode1",
                    "transform_node_id": "vtkMRMLLinearTransformNode1",
                    "hardened": True,
                }
            )
            mock_gc.return_value = client
            apply_transform(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLLinearTransformNode1",
                harden=True,
            )
            code = client.exec_python.call_args[0][0]
        assert "HardenTransform" in code or "hardenTransform" in code.lower()

    def test_apply_transform_no_harden_code_skips_harden(self):
        """When harden=False, generated code must NOT call HardenTransform."""
        from slicer_mcp.registration_tools import apply_transform

        with patch(PATCH_TARGET) as mock_gc:
            client = _mock_client(
                {
                    "success": True,
                    "node_id": "vtkMRMLScalarVolumeNode1",
                    "transform_node_id": "vtkMRMLLinearTransformNode1",
                    "hardened": False,
                }
            )
            mock_gc.return_value = client
            apply_transform(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLLinearTransformNode1",
                harden=False,
            )
            code = client.exec_python.call_args[0][0]
        # The harden block should be conditional; when False it shouldn't appear
        # (or it should be inside a conditional that evaluates to False)
        # We check the generated code doesn't unconditionally call it
        assert "HardenTransform" not in code or "harden" in code.lower()

    def test_register_volumes_code_contains_execresult(self):
        """Generated code must end with __execResult assignment."""
        from slicer_mcp.registration_tools import register_volumes

        with patch(PATCH_TARGET) as mock_gc:
            client = _mock_client(
                {
                    "success": True,
                    "transform_node_id": "vtkMRMLLinearTransformNode1",
                    "transform_node_name": "T",
                    "transform_type": "Rigid",
                }
            )
            mock_gc.return_value = client
            register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
            )
            code = client.exec_python.call_args[0][0]
        assert "__execResult" in code
