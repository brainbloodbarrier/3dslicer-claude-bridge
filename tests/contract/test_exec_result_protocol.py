"""Contract tests for the __execResult protocol.

The __execResult protocol is the mechanism by which feature tools communicate
results back from 3D Slicer:

1. Feature code builds a Python code string ending with ``__execResult = <value>``
2. This code is POSTed to Slicer's ``/slicer/exec`` endpoint
3. Slicer executes the code and returns the value of ``__execResult`` as text
4. The bridge parses the result back from JSON via ``_parse_json_result()``

These contract tests verify all three phases: code generation, response parsing,
and round-trip integration through the mocked pipeline.
"""

import json
import re
from unittest.mock import Mock, patch

import pytest

from slicer_mcp.core.parsing import _parse_json_result
from slicer_mcp.core.slicer_client import SlicerConnectionError

# =============================================================================
# Helpers
# =============================================================================

# Regex that matches ``__execResult = <something>`` on any line (possibly indented)
EXEC_RESULT_PATTERN = re.compile(r"^\s*__execResult\s*=\s*", re.MULTILINE)


def _mock_client(return_dict: dict | list | str | int | float) -> Mock:
    """Create a mock SlicerClient whose exec_python returns *return_dict* as JSON."""
    client = Mock()
    client.exec_python.return_value = {
        "success": True,
        "result": json.dumps(return_dict),
    }
    return client


def _mock_client_raw(raw_text: str) -> Mock:
    """Create a mock SlicerClient whose exec_python returns raw text."""
    client = Mock()
    client.exec_python.return_value = {
        "success": True,
        "result": raw_text,
    }
    return client


def _mock_client_error(exc: Exception) -> Mock:
    """Create a mock SlicerClient whose exec_python raises *exc*."""
    client = Mock()
    client.exec_python.side_effect = exc
    return client


# =============================================================================
# 1. Code Generation Contracts
# =============================================================================


class TestCodeGenerationContracts:
    """Verify that feature code-generation functions produce valid __execResult assignments."""

    # ---- base_tools code builders ----

    def test_single_segment_volume_code_has_exec_result(self):
        """_build_single_segment_volume_code must end with __execResult assignment."""
        from slicer_mcp.features.base_tools import _build_single_segment_volume_code

        code = _build_single_segment_volume_code(
            json.dumps("vtkMRMLSegmentationNode1"),
            json.dumps("Liver"),
        )
        assert EXEC_RESULT_PATTERN.search(code), (
            "Generated code must contain __execResult assignment"
        )

    def test_all_segments_volume_code_has_exec_result(self):
        """_build_all_segments_volume_code must end with __execResult assignment."""
        from slicer_mcp.features.base_tools import _build_all_segments_volume_code

        code = _build_all_segments_volume_code(json.dumps("vtkMRMLSegmentationNode1"))
        assert EXEC_RESULT_PATTERN.search(code)

    # ---- registration code builders ----

    def test_place_landmarks_code_has_exec_result(self):
        from slicer_mcp.features.registration import _build_place_landmarks_code

        code = _build_place_landmarks_code(
            json.dumps("MyNode"),
            json.dumps([[1.0, 2.0, 3.0]]),
            "None",
        )
        assert EXEC_RESULT_PATTERN.search(code)

    def test_get_landmarks_code_has_exec_result(self):
        from slicer_mcp.features.registration import _build_get_landmarks_code

        code = _build_get_landmarks_code(json.dumps("vtkMRMLMarkupsFiducialNode1"))
        assert EXEC_RESULT_PATTERN.search(code)

    def test_register_volumes_code_has_exec_result(self):
        from slicer_mcp.features.registration import _build_register_volumes_code

        code = _build_register_volumes_code(
            json.dumps("vtkMRMLScalarVolumeNode1"),
            json.dumps("vtkMRMLScalarVolumeNode2"),
            json.dumps("Rigid"),
            json.dumps("useMomentsAlign"),
            json.dumps(0.5),
            "False",
            "False",
        )
        assert EXEC_RESULT_PATTERN.search(code)

    def test_register_landmarks_code_has_exec_result(self):
        from slicer_mcp.features.registration import _build_register_landmarks_code

        code = _build_register_landmarks_code(
            json.dumps("vtkMRMLMarkupsFiducialNode1"),
            json.dumps("vtkMRMLMarkupsFiducialNode2"),
            json.dumps("Rigid"),
        )
        assert EXEC_RESULT_PATTERN.search(code)

    def test_apply_transform_code_has_exec_result(self):
        from slicer_mcp.features.registration import _build_apply_transform_code

        code = _build_apply_transform_code(
            json.dumps("vtkMRMLScalarVolumeNode1"),
            json.dumps("vtkMRMLLinearTransformNode1"),
            harden=True,
        )
        assert EXEC_RESULT_PATTERN.search(code)

    # ---- json.dumps safety: parameters must be JSON-encoded, not raw f-string ----

    def test_node_id_is_json_escaped_in_single_segment_code(self):
        """Node IDs with special characters must be safely escaped via json.dumps."""
        from slicer_mcp.features.base_tools import _build_single_segment_volume_code

        tricky_id = 'vtkMRMLNode"1'
        code = _build_single_segment_volume_code(
            json.dumps(tricky_id),
            json.dumps("Segment_1"),
        )
        # The JSON-escaped form should appear in the code
        assert json.dumps(tricky_id) in code
        # Raw unescaped form should NOT appear as a bare assignment
        assert f"node_id = {tricky_id}" not in code

    def test_segment_name_with_unicode_is_json_safe(self):
        """Segment names with unicode (e.g., Greek) must survive json.dumps encoding."""
        from slicer_mcp.features.base_tools import _build_single_segment_volume_code

        unicode_name = "Vertebra_\u03b1"
        code = _build_single_segment_volume_code(
            json.dumps("vtkMRMLSegmentationNode1"),
            json.dumps(unicode_name),
        )
        # json.dumps produces a valid Python string literal
        assert json.dumps(unicode_name) in code

    def test_landmark_points_are_json_serialized(self):
        """Point coordinates must be serialized via json.dumps, not repr() or str()."""
        from slicer_mcp.features.registration import _build_place_landmarks_code

        points = [[1.5, -2.3, 4.0], [0.0, 0.0, 0.0]]
        code = _build_place_landmarks_code(
            json.dumps("LM"),
            json.dumps(points),
            json.dumps(["A", "B"]),
        )
        # json.dumps of points should be embedded literally
        assert json.dumps(points) in code

    # ---- data type coverage: dict, list, string, number, nested ----

    def test_generated_code_assigns_dict_to_exec_result(self):
        """All _build_* functions assign a dict to __execResult."""
        from slicer_mcp.features.base_tools import _build_single_segment_volume_code

        code = _build_single_segment_volume_code(
            json.dumps("vtkMRMLSegmentationNode1"),
            json.dumps("Seg1"),
        )
        # The code should construct a dict result
        assert "result = {" in code or "result = {{" in code

    def test_nested_structures_in_generated_code(self):
        """Code that returns nested dicts (e.g., segments list) uses __execResult."""
        from slicer_mcp.features.base_tools import _build_all_segments_volume_code

        code = _build_all_segments_volume_code(json.dumps("vtkMRMLSegmentationNode1"))
        # Must have segments list building and __execResult
        assert "segments" in code
        assert EXEC_RESULT_PATTERN.search(code)


# =============================================================================
# 2. Response Parsing Contracts
# =============================================================================


class TestResponseParsingContracts:
    """Verify _parse_json_result() handles all expected response shapes."""

    # ---- valid JSON ----

    def test_parse_valid_dict(self):
        data = {"success": True, "node_id": "vtkMRMLNode1"}
        result = _parse_json_result(json.dumps(data), "test")
        assert result == data

    def test_parse_valid_list(self):
        data = [{"id": 1}, {"id": 2}]
        result = _parse_json_result(json.dumps(data), "test")
        assert result == data

    def test_parse_valid_string(self):
        result = _parse_json_result(json.dumps("hello"), "test")
        assert result == "hello"

    def test_parse_valid_number_int(self):
        result = _parse_json_result(json.dumps(42), "test")
        assert result == 42

    def test_parse_valid_number_float(self):
        result = _parse_json_result(json.dumps(3.14), "test")
        assert result == pytest.approx(3.14)

    def test_parse_nested_structure(self):
        data = {
            "success": True,
            "vertebrae": [
                {"label": "C1", "position": [1.0, 2.0, 3.0]},
                {"label": "C2", "position": [4.0, 5.0, 6.0]},
            ],
            "metadata": {"count": 2, "region": "cervical"},
        }
        result = _parse_json_result(json.dumps(data), "test")
        assert result == data
        assert len(result["vertebrae"]) == 2
        assert result["metadata"]["count"] == 2

    def test_parse_special_characters_in_values(self):
        data = {
            "name": 'Vertebra "C1" (atlas)',
            "path": "/tmp/slicer/out\\file.nrrd",
            "notes": "Line1\nLine2\tTabbed",
            "unicode": "\u00e9\u00f1\u00fc",
        }
        result = _parse_json_result(json.dumps(data), "test")
        assert result["name"] == 'Vertebra "C1" (atlas)'
        assert result["path"] == "/tmp/slicer/out\\file.nrrd"
        assert "\n" in result["notes"]
        assert result["unicode"] == "\u00e9\u00f1\u00fc"

    def test_parse_boolean_values(self):
        data = {"success": True, "hardened": False}
        result = _parse_json_result(json.dumps(data), "test")
        assert result["success"] is True
        assert result["hardened"] is False

    def test_parse_null_value_in_dict(self):
        """JSON null inside a valid dict should parse correctly."""
        data = {"resampled_node_id": None, "success": True}
        result = _parse_json_result(json.dumps(data), "test")
        assert result["resampled_node_id"] is None

    # ---- empty / invalid responses ----

    def test_parse_empty_string_raises(self):
        with pytest.raises(SlicerConnectionError, match="Empty result"):
            _parse_json_result("", "test op")

    def test_parse_whitespace_only_raises(self):
        with pytest.raises(SlicerConnectionError, match="Empty result"):
            _parse_json_result("   ", "test op")

    def test_parse_null_string_raises(self):
        with pytest.raises(SlicerConnectionError, match="Empty result"):
            _parse_json_result("null", "test op")

    def test_parse_none_string_raises(self):
        with pytest.raises(SlicerConnectionError, match="Empty result"):
            _parse_json_result("None", "test op")

    def test_parse_none_value_raises(self):
        with pytest.raises(SlicerConnectionError, match="Empty result"):
            _parse_json_result(None, "test op")

    def test_parse_non_json_string_raises(self):
        with pytest.raises(SlicerConnectionError, match="Failed to parse"):
            _parse_json_result("this is not json", "test op")

    def test_parse_malformed_json_raises(self):
        with pytest.raises(SlicerConnectionError, match="Failed to parse"):
            _parse_json_result("{key: value}", "test op")

    def test_parse_truncated_json_raises(self):
        with pytest.raises(SlicerConnectionError, match="Failed to parse"):
            _parse_json_result('{"success": true, "data":', "test op")

    def test_error_message_includes_context(self):
        """Error messages should include the operation context for debugging."""
        with pytest.raises(SlicerConnectionError, match="spine segmentation"):
            _parse_json_result("not json", "spine segmentation")


# =============================================================================
# 3. Round-Trip Contracts
# =============================================================================


class TestRoundTripContracts:
    """Verify end-to-end: tool call -> code generation -> mock exec -> result parsing."""

    # ---- Registration round trips ----

    def test_place_landmarks_round_trip(self):
        """place_landmarks: validation -> codegen -> exec -> parse -> dict."""
        from slicer_mcp.features.registration import place_landmarks

        expected = {
            "success": True,
            "node_id": "vtkMRMLMarkupsFiducialNode1",
            "node_name": "TestLandmarks",
            "point_count": 2,
        }
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(expected)
            result = place_landmarks(
                "TestLandmarks",
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            )

        assert result == expected
        # Verify exec_python was called with code containing __execResult
        call_args = mock_gc.return_value.exec_python.call_args
        code_sent = call_args[0][0]
        assert EXEC_RESULT_PATTERN.search(code_sent)

    def test_get_landmarks_round_trip(self):
        """get_landmarks: validate node_id -> codegen -> exec -> parse."""
        from slicer_mcp.features.registration import get_landmarks

        expected = {
            "success": True,
            "node_id": "vtkMRMLMarkupsFiducialNode1",
            "node_name": "Pts",
            "point_count": 1,
            "points": [{"index": 0, "label": "A", "position_ras": [1.0, 2.0, 3.0]}],
        }
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(expected)
            result = get_landmarks("vtkMRMLMarkupsFiducialNode1")

        assert result == expected
        assert result["points"][0]["position_ras"] == [1.0, 2.0, 3.0]

    def test_apply_transform_round_trip(self):
        """apply_transform: validate IDs -> codegen -> exec -> parse."""
        from slicer_mcp.features.registration import apply_transform

        expected = {
            "success": True,
            "node_id": "vtkMRMLScalarVolumeNode1",
            "transform_node_id": "vtkMRMLLinearTransformNode1",
            "hardened": True,
        }
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(expected)
            result = apply_transform(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLLinearTransformNode1",
                harden=True,
            )

        assert result == expected
        assert result["hardened"] is True

    def test_register_volumes_round_trip(self):
        """register_volumes: validate -> codegen -> exec -> parse with nested result."""
        from slicer_mcp.features.registration import register_volumes

        expected = {
            "success": True,
            "transform_node_id": "vtkMRMLLinearTransformNode1",
            "transform_node_name": "Moving_to_Fixed_Rigid",
            "transform_type": "Rigid",
            "resampled_node_id": None,
        }
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(expected)
            result = register_volumes(
                "vtkMRMLScalarVolumeNode1",
                "vtkMRMLScalarVolumeNode2",
                transform_type="Rigid",
            )

        assert result == expected
        assert result["resampled_node_id"] is None

    # ---- base_tools round trips ----

    def test_measure_volume_round_trip(self):
        """measure_volume: validate -> codegen -> exec -> parse with segments list."""
        from slicer_mcp.features.base_tools import measure_volume

        expected = {
            "node_id": "vtkMRMLSegmentationNode1",
            "node_name": "Segmentation",
            "total_volume_mm3": 15000.0,
            "total_volume_ml": 15.0,
            "segments": [
                {"name": "Liver", "volume_mm3": 15000.0, "volume_ml": 15.0},
            ],
        }
        with patch("slicer_mcp.features.base_tools.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(expected)
            result = measure_volume("vtkMRMLSegmentationNode1", segment_name="Liver")

        assert result == expected
        assert len(result["segments"]) == 1

    # ---- Error round trips ----

    def test_slicer_connection_error_propagates(self):
        """When Slicer is unreachable, SlicerConnectionError must propagate."""
        from slicer_mcp.features.registration import get_landmarks

        error = SlicerConnectionError("Connection refused")
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client_error(error)
            with pytest.raises(SlicerConnectionError, match="Connection refused"):
                get_landmarks("vtkMRMLMarkupsFiducialNode1")

    def test_empty_exec_result_raises_connection_error(self):
        """When Slicer returns empty result (no __execResult set), parsing should fail."""
        from slicer_mcp.features.registration import get_landmarks

        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client_raw("")
            with pytest.raises(SlicerConnectionError, match="Empty result"):
                get_landmarks("vtkMRMLMarkupsFiducialNode1")

    def test_non_json_exec_result_raises_connection_error(self):
        """When Slicer returns non-JSON text, parsing should fail gracefully."""
        from slicer_mcp.features.registration import get_landmarks

        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client_raw("Traceback (most recent call last):\n...")
            with pytest.raises(SlicerConnectionError, match="Failed to parse"):
                get_landmarks("vtkMRMLMarkupsFiducialNode1")

    def test_slicer_returns_error_dict(self):
        """When Slicer code itself returns an error dict, it should parse correctly."""
        from slicer_mcp.features.registration import register_landmarks

        error_result = {
            "success": False,
            "error": "Fixed landmarks: 2 pts, need >= 3",
        }
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(error_result)
            result = register_landmarks(
                "vtkMRMLMarkupsFiducialNode1",
                "vtkMRMLMarkupsFiducialNode2",
                transform_type="Rigid",
            )

        # The error dict parses successfully — it's the caller's job to check success
        assert result["success"] is False
        assert "need >= 3" in result["error"]

    # ---- Code fidelity: verify the code sent to Slicer is well-formed ----

    def test_code_sent_to_slicer_uses_json_dumps_values(self):
        """Verify that parameters in the code sent to exec_python are JSON-encoded.

        Uses _build_place_landmarks_code directly to bypass input validation
        (which correctly rejects special chars). This tests the codegen contract.
        """
        from slicer_mcp.features.registration import _build_place_landmarks_code

        tricky_name = 'Land"marks'
        code = _build_place_landmarks_code(
            json.dumps(tricky_name),
            json.dumps([[0.0, 0.0, 0.0]]),
            "None",
        )
        # The name must be JSON-encoded in the code, not raw
        assert json.dumps(tricky_name) in code

    def test_round_trip_uses_json_dumps_for_valid_name(self):
        """Verify a valid name with dots/hyphens is JSON-encoded in the code sent."""
        from slicer_mcp.features.registration import place_landmarks

        name = "Land-marks.v2"
        expected = {
            "success": True,
            "node_id": "vtkMRMLMarkupsFiducialNode1",
            "node_name": name,
            "point_count": 1,
        }
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(expected)
            place_landmarks(name, [[0.0, 0.0, 0.0]])

        code_sent = mock_gc.return_value.exec_python.call_args[0][0]
        assert json.dumps(name) in code_sent

    def test_code_sent_preserves_float_precision(self):
        """Floating-point coordinates must maintain precision through json.dumps."""
        from slicer_mcp.features.registration import place_landmarks

        points = [[1.123456789, -2.987654321, 0.000001]]
        expected = {
            "success": True,
            "node_id": "vtkMRMLMarkupsFiducialNode1",
            "node_name": "Precise",
            "point_count": 1,
        }
        with patch("slicer_mcp.features.registration.get_client") as mock_gc:
            mock_gc.return_value = _mock_client(expected)
            place_landmarks("Precise", points)

        code_sent = mock_gc.return_value.exec_python.call_args[0][0]
        assert json.dumps(points) in code_sent

    def test_version_detection_uses_exec_result_protocol(self):
        """get_slicer_version() uses __execResult via exec_python internally."""
        from slicer_mcp.core.slicer_client import SlicerClient

        client = SlicerClient(base_url="http://localhost:2016", timeout=5)

        # Simulate what Slicer returns: double-JSON-encoded version string
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.text = json.dumps("5.6.2")

        with patch("requests.post", return_value=mock_response):
            version = client.get_slicer_version()

        assert version == "5.6.2"
