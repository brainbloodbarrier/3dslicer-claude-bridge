"""Unit tests for core/codegen.py safety helpers."""


from slicer_mcp.core.codegen import safe_json, safe_optional, safe_string


class TestSafeString:
    def test_simple_string(self):
        assert safe_string("hello") == '"hello"'

    def test_string_with_quotes(self):
        assert safe_string('say "hi"') == '"say \\"hi\\""'

    def test_string_with_backslash(self):
        assert safe_string("path\\to") == '"path\\\\to"'

    def test_empty_string(self):
        assert safe_string("") == '""'

    def test_unicode(self):
        result = safe_string("café")
        assert "caf" in result


class TestSafeJson:
    def test_list(self):
        assert safe_json([1, 2, 3]) == "[1, 2, 3]"

    def test_dict(self):
        result = safe_json({"a": 1})
        assert '"a"' in result
        assert "1" in result

    def test_number(self):
        assert safe_json(42) == "42"

    def test_float(self):
        assert safe_json(3.14) == "3.14"

    def test_bool_produces_json_not_python(self):
        # json.dumps(True) -> "true", NOT "True"
        # This is intentional — safe_json is for JSON values, not Python bools
        assert safe_json(True) == "true"

    def test_nested(self):
        result = safe_json({"levels": ["C1", "C2"]})
        assert '"levels"' in result
        assert '"C1"' in result


class TestSafeOptional:
    def test_none_returns_none_string(self):
        assert safe_optional(None) == "None"

    def test_string_value(self):
        assert safe_optional("hello") == '"hello"'

    def test_list_value(self):
        assert safe_optional([1, 2]) == "[1, 2]"

    def test_bool_value(self):
        assert safe_optional(True) == "true"

    def test_int_value(self):
        assert safe_optional(42) == "42"
