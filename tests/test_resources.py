"""Unit tests for MCP resource implementations."""

from datetime import datetime

from slicer_mcp.resources import _iso_timestamp


class TestIsoTimestamp:
    """Test _iso_timestamp helper function."""

    def test_iso_timestamp_returns_string(self):
        """_iso_timestamp should return a string."""
        ts = _iso_timestamp()
        assert isinstance(ts, str)

    def test_iso_timestamp_ends_with_z_suffix(self):
        """_iso_timestamp should end with 'Z' suffix for UTC."""
        ts = _iso_timestamp()
        assert ts.endswith("Z"), f"Expected timestamp to end with 'Z', got: {ts}"

    def test_iso_timestamp_contains_t_separator(self):
        """_iso_timestamp should contain 'T' separator between date and time."""
        ts = _iso_timestamp()
        assert "T" in ts, f"Expected 'T' separator in timestamp, got: {ts}"

    def test_iso_timestamp_is_parseable(self):
        """_iso_timestamp should return a parseable ISO 8601 timestamp."""
        ts = _iso_timestamp()

        # Replace 'Z' with '+00:00' for Python's fromisoformat
        parseable_ts = ts.replace("Z", "+00:00")

        # This should not raise an exception
        parsed = datetime.fromisoformat(parseable_ts)
        assert parsed is not None

    def test_iso_timestamp_format_structure(self):
        """_iso_timestamp should follow YYYY-MM-DDTHH:MM:SSZ format."""
        ts = _iso_timestamp()

        # Should be exactly 20 characters: 2024-01-15T14:30:00Z
        assert len(ts) == 20, f"Expected 20 characters, got {len(ts)}: {ts}"

        # Check structure
        assert ts[4] == "-", f"Expected '-' at position 4, got: {ts}"
        assert ts[7] == "-", f"Expected '-' at position 7, got: {ts}"
        assert ts[10] == "T", f"Expected 'T' at position 10, got: {ts}"
        assert ts[13] == ":", f"Expected ':' at position 13, got: {ts}"
        assert ts[16] == ":", f"Expected ':' at position 16, got: {ts}"
        assert ts[19] == "Z", f"Expected 'Z' at position 19, got: {ts}"

    def test_iso_timestamp_is_utc(self):
        """_iso_timestamp should return UTC time (verified by 'Z' suffix)."""
        ts = _iso_timestamp()

        # The 'Z' suffix indicates Zulu time (UTC)
        assert ts.endswith("Z")

        # Parse and verify timezone is UTC
        parseable_ts = ts.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(parseable_ts)

        # The parsed datetime should have tzinfo indicating UTC
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_iso_timestamp_seconds_precision(self):
        """_iso_timestamp should have seconds precision (not milliseconds)."""
        ts = _iso_timestamp()

        # Should NOT contain a decimal point (no sub-second precision)
        assert "." not in ts, f"Expected no decimal point in timestamp, got: {ts}"
