"""Unit tests for MCP tool implementations."""

import os
import pytest
from unittest.mock import Mock, patch

from slicer_mcp.tools import (
    validate_mrml_node_id,
    validate_segment_name,
    ValidationError,
    measure_volume,
    _validate_audit_log_path,
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


class TestSegmentNameNormalization:
    """Test segment name whitespace normalization."""

    def test_segment_name_strips_leading_whitespace(self):
        """Test leading whitespace is stripped."""
        result = validate_segment_name("  Brain")
        assert result == "Brain"

    def test_segment_name_strips_trailing_whitespace(self):
        """Test trailing whitespace is stripped."""
        result = validate_segment_name("Brain  ")
        assert result == "Brain"

    def test_segment_name_collapses_multiple_spaces(self):
        """Test multiple spaces are collapsed to single space."""
        result = validate_segment_name("Left   Lung")
        assert result == "Left Lung"

    def test_segment_name_normalizes_complex_whitespace(self):
        """Test complex whitespace is fully normalized."""
        result = validate_segment_name("  Brain   Stem  ")
        assert result == "Brain Stem"

    def test_segment_name_only_whitespace_rejected(self):
        """Test segment name with only whitespace is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_segment_name("   ")
        assert "only whitespace" in str(exc_info.value) or "cannot be" in str(exc_info.value)

    def test_segment_name_tabs_normalized(self):
        """Test tabs are treated as whitespace and normalized."""
        result = validate_segment_name("Brain\tStem")
        assert result == "Brain Stem"


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


# =============================================================================
# Code Injection Defense-in-Depth Tests (Batch 1 Fix 1.3)
# =============================================================================

class TestMeasureVolumeCodeGeneration:
    """Test measure_volume generates safe Python code using json.dumps()."""

    def test_measure_volume_uses_json_escaped_node_id(self):
        """Test measure_volume uses json.dumps for node_id escaping."""
        import json
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"node_id": "vtkMRMLSegmentationNode1", "node_name": "Test", "total_volume_mm3": 1000, "total_volume_ml": 1.0, "segments": []}'
            }
            mock_get_client.return_value = mock_client

            measure_volume("vtkMRMLSegmentationNode1")

            # Get the Python code that was passed to exec_python
            python_code = mock_client.exec_python.call_args[0][0]

            # Verify json.dumps is used (node_id should be assigned via JSON)
            # The code should contain: node_id = "vtkMRMLSegmentationNode1"
            # (the value comes from json.dumps which produces quoted string)
            assert 'node_id = "vtkMRMLSegmentationNode1"' in python_code
            # Should NOT contain direct f-string interpolation like '{node_id}'
            assert "'{vtkMRMLSegmentationNode1}'" not in python_code

    def test_measure_volume_uses_json_escaped_segment_name(self):
        """Test measure_volume uses json.dumps for segment_name escaping."""
        import json
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"node_id": "vtkMRMLSegmentationNode1", "node_name": "Test", "total_volume_mm3": 1000, "total_volume_ml": 1.0, "segments": [{"name": "Tumor", "volume_mm3": 1000, "volume_ml": 1.0}]}'
            }
            mock_get_client.return_value = mock_client

            measure_volume("vtkMRMLSegmentationNode1", "Tumor")

            # Get the Python code that was passed to exec_python
            python_code = mock_client.exec_python.call_args[0][0]

            # Verify segment_name is JSON-escaped
            assert 'segment_name = "Tumor"' in python_code
            # Should NOT contain direct f-string interpolation
            assert "'{Tumor}'" not in python_code

    def test_measure_volume_code_uses_variable_not_interpolation(self):
        """Test generated code uses variables instead of direct interpolation."""
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"node_id": "vtkMRMLSegmentationNode1", "node_name": "Test", "total_volume_mm3": 1000, "total_volume_ml": 1.0, "segments": []}'
            }
            mock_get_client.return_value = mock_client

            measure_volume("vtkMRMLSegmentationNode1")

            python_code = mock_client.exec_python.call_args[0][0]

            # Verify the code uses the variable, not direct interpolation
            assert 'GetNodeByID(node_id)' in python_code
            # Should NOT have the old pattern with direct string interpolation
            assert "GetNodeByID('vtkMRMLSegmentationNode1')" not in python_code

    def test_measure_volume_segment_code_uses_variable(self):
        """Test generated code for segment measurement uses variables."""
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"node_id": "vtkMRMLSegmentationNode1", "node_name": "Test", "total_volume_mm3": 1000, "total_volume_ml": 1.0, "segments": [{"name": "Brain", "volume_mm3": 1000, "volume_ml": 1.0}]}'
            }
            mock_get_client.return_value = mock_client

            measure_volume("vtkMRMLSegmentationNode1", "Brain")

            python_code = mock_client.exec_python.call_args[0][0]

            # Verify GetSegment uses the variable
            assert 'GetSegment(segment_name)' in python_code
            # Should NOT have old pattern with direct interpolation
            assert "GetSegment('Brain')" not in python_code


# =============================================================================
# Audit Log Path Validation Tests (Batch 5 Fix 5.1)
# =============================================================================

class TestAuditLogPathValidation:
    """Test audit log path validation."""

    def test_valid_path_in_home_directory(self):
        """Test valid path in home directory is accepted."""
        result = _validate_audit_log_path("~/audit.log")
        assert result.endswith("audit.log")
        assert os.path.expanduser("~") in result

    def test_valid_path_relative(self):
        """Test valid relative path is accepted."""
        result = _validate_audit_log_path("./logs/audit.log")
        assert "audit.log" in result

    def test_forbidden_path_etc(self):
        """Test path in /etc is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/etc/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_forbidden_path_system(self):
        """Test path in /System is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/System/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_forbidden_path_usr(self):
        """Test path in /usr is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/usr/local/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_forbidden_path_root(self):
        """Test path in /root is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/root/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_forbidden_path_library_macos(self):
        """Test path in /Library (macOS) is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/Library/Logs/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_forbidden_path_applications_macos(self):
        """Test path in /Applications (macOS) is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/Applications/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_forbidden_path_windows_system32(self):
        """Test path in /Windows is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/Windows/System32/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_forbidden_path_program_files(self):
        """Test path in /Program Files is rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/Program Files/MyApp/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_case_insensitive_validation_etc(self):
        """Test forbidden path check is case-insensitive."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/ETC/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_case_insensitive_validation_system(self):
        """Test forbidden path check is case-insensitive for /System."""
        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path("/system/audit.log")
        assert "forbidden directory" in str(exc_info.value)

    def test_valid_path_absolute(self):
        """Test valid absolute path is accepted."""
        result = _validate_audit_log_path("/tmp/audit.log")
        assert result == "/tmp/audit.log"

    def test_path_expansion_tilde(self):
        """Test ~ is properly expanded to home directory."""
        result = _validate_audit_log_path("~/logs/audit.log")
        # Should not contain ~ in result
        assert "~" not in result
        # Should contain expanded home path
        assert result.startswith(os.path.expanduser("~"))

    def test_path_converted_to_absolute(self):
        """Test relative path is converted to absolute."""
        result = _validate_audit_log_path("audit.log")
        # Should be absolute path
        assert os.path.isabs(result)
        assert result.endswith("audit.log")


# =============================================================================
# Malformed JSON Response Tests (Batch 6 Fix 6.2)
# =============================================================================

class TestMalformedJsonHandling:
    """Test handling of malformed JSON responses from Slicer."""

    def test_measure_volume_malformed_json(self):
        """Test measure_volume handles malformed JSON gracefully."""
        from slicer_mcp.slicer_client import SlicerConnectionError
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "not valid json {"
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError) as exc_info:
                measure_volume("vtkMRMLSegmentationNode1")

            assert "Failed to parse" in str(exc_info.value)

    def test_measure_volume_empty_result(self):
        """Test measure_volume handles empty result."""
        from slicer_mcp.slicer_client import SlicerConnectionError
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": ""
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError) as exc_info:
                measure_volume("vtkMRMLSegmentationNode1")

            assert "Empty result" in str(exc_info.value)

    def test_measure_volume_null_result(self):
        """Test measure_volume handles null JSON result."""
        from slicer_mcp.slicer_client import SlicerConnectionError
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "null"
            }
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError) as exc_info:
                measure_volume("vtkMRMLSegmentationNode1")

            assert "Empty result" in str(exc_info.value) or "null" in str(exc_info.value).lower()

    def test_list_sample_data_malformed_json(self):
        """Test list_sample_data handles malformed JSON gracefully."""
        from slicer_mcp.tools import list_sample_data
        with patch('slicer_mcp.tools.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": "invalid json"
            }
            mock_get_client.return_value = mock_client

            # list_sample_data should handle errors gracefully and return fallback
            result = list_sample_data()
            assert result["source"] == "fallback"


# =============================================================================
# Unicode Segment Name Tests (Batch 7: Unicode Support for Medical Terminology)
# =============================================================================

class TestUnicodeSegmentNames:
    """Test Unicode handling in segment names.

    Medical terminology frequently uses Greek letters (α, β, μ), accented
    characters (é, ñ, ü), and international alphabets. These should be accepted
    while still blocking security-sensitive characters like emoji or shell
    metacharacters.
    """

    # --- Tests for ACCEPTED Unicode (medical terminology) ---

    def test_segment_name_accepts_greek_letters(self):
        """Test segment names with Greek letters are accepted (medical terms)."""
        # Greek letters common in medical/scientific terminology
        assert validate_segment_name("α-fetoprotein") == "α-fetoprotein"
        assert validate_segment_name("β-amyloid plaque") == "β-amyloid plaque"
        assert validate_segment_name("μm scale") == "μm scale"
        assert validate_segment_name("γ-aminobutyric") == "γ-aminobutyric"
        assert validate_segment_name("δ region") == "δ region"

    def test_segment_name_accepts_accented_characters(self):
        """Test segment names with accented characters are accepted."""
        # Common in international medical terminology
        assert validate_segment_name("Müller cells") == "Müller cells"
        assert validate_segment_name("señal region") == "señal region"
        assert validate_segment_name("naïve tissue") == "naïve tissue"
        assert validate_segment_name("Cérebro") == "Cérebro"
        assert validate_segment_name("Tumör") == "Tumör"

    def test_segment_name_accepts_cyrillic(self):
        """Test segment names with Cyrillic characters are accepted."""
        # For international collaboration
        assert validate_segment_name("Мозг region") == "Мозг region"
        assert validate_segment_name("Опухоль") == "Опухоль"

    def test_segment_name_accepts_chinese_characters(self):
        """Test segment names with Chinese characters are accepted."""
        # For international medical collaboration
        assert validate_segment_name("肿瘤") == "肿瘤"
        assert validate_segment_name("脑部 region") == "脑部 region"

    def test_segment_name_accepts_japanese_characters(self):
        """Test segment names with Japanese characters are accepted."""
        assert validate_segment_name("腫瘍") == "腫瘍"
        assert validate_segment_name("脳 region") == "脳 region"

    def test_segment_name_accepts_mixed_scripts(self):
        """Test segment names with mixed Unicode scripts are accepted."""
        assert validate_segment_name("Brain α-region") == "Brain α-region"
        assert validate_segment_name("Tumor 1 β-type") == "Tumor 1 β-type"

    # --- Tests for REJECTED characters (security) ---

    def test_segment_name_rejects_emoji(self):
        """Test segment names with emoji are rejected (not word characters)."""
        with pytest.raises(ValidationError):
            validate_segment_name("Heart ❤")
        with pytest.raises(ValidationError):
            validate_segment_name("Brain 🧠")
        with pytest.raises(ValidationError):
            validate_segment_name("😀 Happy Tumor")

    def test_segment_name_rejects_shell_metacharacters(self):
        """Test segment names with shell metacharacters are rejected."""
        with pytest.raises(ValidationError):
            validate_segment_name("test; rm -rf /")
        with pytest.raises(ValidationError):
            validate_segment_name("test`whoami`")
        with pytest.raises(ValidationError):
            validate_segment_name("test$(cmd)")
        with pytest.raises(ValidationError):
            validate_segment_name("test | cat")
        with pytest.raises(ValidationError):
            validate_segment_name("test & background")

    def test_segment_name_rejects_quotes(self):
        """Test segment names with quotes are rejected."""
        with pytest.raises(ValidationError):
            validate_segment_name("test'injection")
        with pytest.raises(ValidationError):
            validate_segment_name('test"injection')

    def test_segment_name_rejects_brackets(self):
        """Test segment names with brackets are rejected."""
        with pytest.raises(ValidationError):
            validate_segment_name("test[0]")
        with pytest.raises(ValidationError):
            validate_segment_name("test{}")
        with pytest.raises(ValidationError):
            validate_segment_name("test()")

    def test_segment_name_rejects_special_symbols(self):
        """Test segment names with special symbols are rejected."""
        with pytest.raises(ValidationError):
            validate_segment_name("test@domain")
        with pytest.raises(ValidationError):
            validate_segment_name("test#hash")
        with pytest.raises(ValidationError):
            validate_segment_name("test%percent")
        with pytest.raises(ValidationError):
            validate_segment_name("test^caret")
        with pytest.raises(ValidationError):
            validate_segment_name("test*glob")

    # --- Tests for Unicode normalization (NFKC) security ---

    def test_segment_name_normalizes_zero_width_characters(self):
        """Test that zero-width characters are removed via NFKC normalization."""
        # Zero-width space (U+200B) should be removed
        # "test\u200Binjection" should become "testinjection"
        result = validate_segment_name("test\u200Binjection")
        assert result == "testinjection"

        # Zero-width non-joiner (U+200C) should be removed
        result = validate_segment_name("test\u200Cvalue")
        assert result == "testvalue"

    def test_segment_name_normalizes_soft_hyphen(self):
        """Test that soft hyphens are removed via NFKC normalization."""
        # Soft hyphen (U+00AD) should be removed
        result = validate_segment_name("test\u00ADhidden")
        assert result == "testhidden"

    def test_segment_name_normalizes_compatibility_characters(self):
        """Test that compatibility characters are normalized."""
        # Fullwidth Latin letters should be normalized to ASCII
        # Ａ (U+FF21) -> A, ｂ (U+FF42) -> b
        result = validate_segment_name("Ｔｅｓｔ")
        assert result == "Test"

    def test_segment_name_handles_bom(self):
        """Test that byte order marks are removed."""
        # BOM (U+FEFF) should be removed
        result = validate_segment_name("\uFEFFTumor")
        assert result == "Tumor"

    # --- Node ID still rejects Unicode (stricter for MRML IDs) ---

    def test_node_id_rejects_unicode(self):
        """Test node IDs with Unicode are rejected (MRML IDs are ASCII-only)."""
        with pytest.raises(ValidationError):
            validate_mrml_node_id("vtkMRMLNödë1")
        with pytest.raises(ValidationError):
            validate_mrml_node_id("vtkMRMLαNode1")
