# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all Critical and Important issues identified in the full project code review, plus lint violations.

**Architecture:** These are targeted fixes across the existing codebase — no new modules or structural changes. Fixes span input validation (symlink resolution), test corrections (failing test + coverage), lint compliance, type safety, and defensive coding. All changes follow existing patterns.

**Tech Stack:** Python 3.10+, pytest, ruff, Black

---

### Task 1: Fix Failing Test (C1)

**Files:**
- Modify: `tests/test_slicer_client.py:543-554`

**Step 1: Read the current failing test**

Read `tests/test_slicer_client.py` lines 543-554 to confirm the test.

**Step 2: Fix the test assertion and docstring**

The test `test_exec_python_exhausts_all_retries` asserts `mock_post.call_count == 4`, expecting retries. But `exec_python()` is intentionally NOT decorated with `@with_retry` (non-idempotent). Fix the test to reflect actual behavior:

```python
    def test_exec_python_does_not_retry(self, slicer_client):
        """Test exec_python does NOT retry on connection error (non-idempotent)."""
        with (
            patch("slicer_mcp.slicer_client.requests.post") as mock_post,
            patch("slicer_mcp.slicer_client.time.sleep") as mock_sleep,
        ):
            mock_post.side_effect = ConnectionError("Connection refused")

            with pytest.raises(SlicerConnectionError):
                slicer_client.exec_python("print('test')")

            # exec_python is NOT retried - single attempt only
            assert mock_post.call_count == 1
            # No sleep should occur (no retries)
            assert mock_sleep.call_count == 0
```

**Step 3: Run the test to verify it passes**

Run: `uv run pytest tests/test_slicer_client.py::TestRetryExhaustion::test_exec_python_does_not_retry -v`
Expected: PASS

**Step 4: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: 0 failures

**Step 5: Commit**

```bash
git add tests/test_slicer_client.py
git commit -m "fix(test): correct exec_python retry test to match non-retry design"
```

---

### Task 2: Fix Symlink Bypass in Audit Log Path (I3)

**Files:**
- Modify: `src/slicer_mcp/tools.py:101`
- Modify: `tests/test_tools.py` (add symlink test)

**Step 1: Write the failing test for symlink bypass**

Add to `tests/test_tools.py` in class `TestAuditLogPathValidation`:

```python
    def test_symlink_to_forbidden_directory_rejected(self, tmp_path):
        """Test that symlink pointing to forbidden directory is rejected."""
        import os

        # Create a symlink pointing to /etc
        symlink_path = tmp_path / "innocent.log"
        symlink_path.symlink_to("/etc/audit.log")

        with pytest.raises(ValueError) as exc_info:
            _validate_audit_log_path(str(symlink_path))
        assert "forbidden directory" in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py::TestAuditLogPathValidation::test_symlink_to_forbidden_directory_rejected -v`
Expected: FAIL (current code uses `os.path.abspath` which doesn't resolve symlinks)

**Step 3: Fix `_validate_audit_log_path` to use `os.path.realpath`**

In `src/slicer_mcp/tools.py` line 101, change:

```python
# Before:
    abs_path = os.path.abspath(os.path.expanduser(path))

# After:
    abs_path = os.path.realpath(os.path.expanduser(path))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tools.py::TestAuditLogPathValidation -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: 0 failures

**Step 6: Commit**

```bash
git add src/slicer_mcp/tools.py tests/test_tools.py
git commit -m "fix(security): resolve symlinks in audit log path validation"
```

---

### Task 3: Fix Symlink Bypass in Folder Path Validation (I4)

**Files:**
- Modify: `src/slicer_mcp/tools.py:364`
- Modify: `tests/test_tools.py` (add symlink test)

**Step 1: Write the failing test for symlink bypass in folder validation**

Add to `tests/test_tools.py` in class `TestDICOMValidation`:

```python
    def test_validate_folder_path_symlink_to_parent_directory(self, tmp_path):
        """Test that symlink resolving outside expected location is handled."""
        import os
        from slicer_mcp.tools import validate_folder_path

        # Create a real directory and a symlink pointing outside
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        symlink_dir = tmp_path / "link_dir"
        symlink_dir.symlink_to(real_dir)

        # Should resolve the symlink and return the real path
        result = validate_folder_path(str(symlink_dir))
        assert result == str(real_dir)
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/test_tools.py::TestDICOMValidation::test_validate_folder_path_symlink_to_parent_directory -v`
Expected: FAIL (abspath doesn't resolve symlinks, so returned path would be the symlink not the real path)

**Step 3: Fix `validate_folder_path` to use `os.path.realpath`**

In `src/slicer_mcp/tools.py` line 364, change:

```python
# Before:
    abs_path = os.path.abspath(os.path.expanduser(path))

# After:
    abs_path = os.path.realpath(os.path.expanduser(path))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tools.py::TestDICOMValidation -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/slicer_mcp/tools.py tests/test_tools.py
git commit -m "fix(security): resolve symlinks in folder path validation"
```

---

### Task 4: Add `look_from_axis` Validation (I2)

**Files:**
- Modify: `src/slicer_mcp/constants.py` (add constant)
- Modify: `src/slicer_mcp/tools.py:411-469` (add validation)
- Modify: `tests/test_tools.py` (add tests)

**Step 1: Write failing tests for look_from_axis validation**

Add new test class to `tests/test_tools.py`:

```python
class TestCaptureScreenshotValidation:
    """Test capture_screenshot input validation."""

    def test_capture_screenshot_rejects_invalid_look_from_axis(self):
        """Invalid look_from_axis should raise ValueError."""
        from slicer_mcp.tools import capture_screenshot

        with pytest.raises(ValueError) as exc_info:
            capture_screenshot("3d", look_from_axis="malicious_value")
        assert "look_from_axis" in str(exc_info.value)

    def test_capture_screenshot_accepts_valid_look_from_axis(self):
        """Valid look_from_axis should pass validation."""
        from slicer_mcp.tools import capture_screenshot

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_3d_screenshot.return_value = b"\x89PNG\r\n\x1a\n"
            mock_get_client.return_value = mock_client

            result = capture_screenshot("3d", look_from_axis="anterior")
            assert result["success"] is True

    def test_capture_screenshot_allows_none_look_from_axis(self):
        """None look_from_axis should be allowed for 3d view."""
        from slicer_mcp.tools import capture_screenshot

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_3d_screenshot.return_value = b"\x89PNG\r\n\x1a\n"
            mock_get_client.return_value = mock_client

            result = capture_screenshot("3d", look_from_axis=None)
            assert result["success"] is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py::TestCaptureScreenshotValidation -v`
Expected: `test_capture_screenshot_rejects_invalid_look_from_axis` FAIL

**Step 3: Add the constant**

In `src/slicer_mcp/constants.py`, add after the `VIEW_MAP` definition (after line 30):

```python
# Valid 3D view camera axes (for look_from_axis parameter)
VALID_3D_AXES = frozenset(["left", "right", "anterior", "posterior", "superior", "inferior"])
```

**Step 4: Add validation in `capture_screenshot`**

In `src/slicer_mcp/tools.py`, import `VALID_3D_AXES` from constants (add to the imports at top), then add validation in `capture_screenshot` after the `view_type` validation (after line 432):

```python
    # Validate look_from_axis if provided
    if look_from_axis is not None:
        if look_from_axis not in VALID_3D_AXES:
            raise ValueError(
                f"Invalid look_from_axis '{look_from_axis}'. "
                f"Must be one of: {', '.join(sorted(VALID_3D_AXES))}"
            )
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py::TestCaptureScreenshotValidation -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/slicer_mcp/constants.py src/slicer_mcp/tools.py tests/test_tools.py
git commit -m "fix(security): validate look_from_axis parameter against allowed values"
```

---

### Task 5: Fix `with_retry` Implicit None Return (I6)

**Files:**
- Modify: `src/slicer_mcp/slicer_client.py:103-105`
- Modify: `tests/test_slicer_client.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_slicer_client.py`, in a new class or existing retry class:

```python
class TestWithRetryEdgeCases:
    """Test edge cases in the with_retry decorator."""

    def test_retry_decorator_never_returns_none_silently(self):
        """with_retry should never silently return None."""
        from slicer_mcp.slicer_client import with_retry, SlicerConnectionError

        @with_retry(max_retries=0, retryable_exceptions=(SlicerConnectionError,))
        def always_fails():
            raise SlicerConnectionError("fail")

        with pytest.raises(SlicerConnectionError):
            always_fails()
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/test_slicer_client.py::TestWithRetryEdgeCases -v`
Expected: PASS (current code does raise when `last_exception is not None`). But the defensive assertion is still good practice.

**Step 3: Add defensive assertion after the retry loop**

In `src/slicer_mcp/slicer_client.py`, after line 104 (`raise last_exception`), add:

```python
            if last_exception is not None:
                raise last_exception
            # Should never reach here - all paths should either return or raise
            raise RuntimeError(f"Retry logic error in {func.__name__}")
```

**Step 4: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: 0 failures

**Step 5: Commit**

```bash
git add src/slicer_mcp/slicer_client.py tests/test_slicer_client.py
git commit -m "fix: add defensive assertion to with_retry decorator"
```

---

### Task 6: Fix Type Mismatch in `_get_valid_datasets` (I7)

**Files:**
- Modify: `src/slicer_mcp/tools.py:819`

**Step 1: Read the current code**

Confirm `_get_valid_datasets()` at line 808-819 returns `FALLBACK_SAMPLE_DATASETS` (a tuple) but is annotated `list[str]`.

**Step 2: Fix by converting fallback to list**

In `src/slicer_mcp/tools.py` line 819, change:

```python
# Before:
        return FALLBACK_SAMPLE_DATASETS

# After:
        return list(FALLBACK_SAMPLE_DATASETS)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_tools.py -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/slicer_mcp/tools.py
git commit -m "fix: convert tuple to list in _get_valid_datasets return type"
```

---

### Task 7: Fix Node Type Extraction (M4)

**Files:**
- Modify: `src/slicer_mcp/slicer_client.py:650`
- Modify: `tests/test_slicer_client.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_slicer_client.py` in `TestGetSceneNodes` (or a new class):

```python
    def test_get_scene_nodes_preserves_digits_in_type_name(self, slicer_client):
        """Node type extraction should only strip trailing digits, not all digits.

        e.g., vtkMRML3DViewNode1 should produce type vtkMRML3DViewNode, not vtkMRMLDViewNode.
        """
        mock_names_response = Mock()
        mock_names_response.text = '["3D View"]'
        mock_names_response.raise_for_status = Mock()

        mock_ids_response = Mock()
        mock_ids_response.text = '["vtkMRML3DViewNode1"]'
        mock_ids_response.raise_for_status = Mock()

        with patch("slicer_mcp.slicer_client.requests.get") as mock_get:
            mock_get.side_effect = [mock_names_response, mock_ids_response]

            nodes = slicer_client.get_scene_nodes()

            assert nodes[0]["type"] == "vtkMRML3DViewNode"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slicer_client.py::TestGetSceneNodes::test_get_scene_nodes_preserves_digits_in_type_name -v`
Expected: FAIL — current code strips all digits, producing "vtkMRMLDViewNode"

**Step 3: Fix the node type extraction**

In `src/slicer_mcp/slicer_client.py` line 650, change:

```python
# Before:
                    node_type = "".join(c for c in node_id if not c.isdigit())

# After:
                    node_type = re.sub(r'\d+$', '', node_id)
```

Also add `import re` at the top of the file if not already present. Check first — it may already be imported.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slicer_client.py::TestGetSceneNodes -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/slicer_mcp/slicer_client.py tests/test_slicer_client.py
git commit -m "fix: only strip trailing digits from node IDs in type extraction"
```

---

### Task 8: Fix Ruff Lint Violations (I5)

**Files:**
- Modify: `src/slicer_mcp/server.py` (E501 — long docstring lines)
- Modify: `tests/test_slicer_client.py` (F841 — unused mock_sleep)
- Modify: `tests/test_tools.py` (E501 — long mock data strings)
- Modify: `pyproject.toml` (M1 — deprecated ruff config)

**Step 1: Fix deprecated ruff config in pyproject.toml**

In `pyproject.toml`, change lines 82-86 from:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "W", "I", "N", "UP"]
ignore = []
```

to:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP"]
ignore = []
```

**Step 2: Fix unused `mock_sleep` variables (F841)**

In `tests/test_slicer_client.py`, find all `as mock_sleep` bindings where `mock_sleep` is never used and change them to `as _mock_sleep`. These are at lines: 547, 1039, 1093, 1234.

Note: line 531 and 560 use `mock_sleep` (checking `call_count` or `call_args_list`), so do NOT change those.

**Step 3: Fix E501 in server.py docstrings**

Wrap long docstring lines in `src/slicer_mcp/server.py` to fit within 100 chars. These are all in `@mcp.tool()` decorator docstrings. Wrap the `Args:` and `Returns:` description lines by splitting them across multiple lines. For example, line 72-74:

```python
    """Capture a screenshot from a specific 3D Slicer viewport and return as base64 PNG.

    Args:
        view_type: Viewport type - "axial" (Red slice), "sagittal" (Yellow slice),
            "coronal" (Green slice), "3d" (3D view), "full" (complete window)
        scroll_position: Slice position from 0.0 to 1.0
            (only for axial/sagittal/coronal views)
        look_from_axis: Camera axis for 3D view - "left", "right", "anterior",
            "posterior", "superior", "inferior" (only for 3d view)
```

Apply similar wrapping to all E501 violations in server.py (lines 142, 173-174, 193, 206, 222, 261-262, 297, 307).

**Step 4: Fix E501 in test_tools.py**

Wrap long mock data strings in `tests/test_tools.py` by splitting across lines. At lines 189, 214, 237, 257, 276, 789, 808, use string concatenation or multi-line strings:

```python
                "result": (
                    '{"node_id": "vtkMRMLSegmentationNode1",'
                    ' "node_name": "Test",'
                    ' "total_volume_mm3": 1000,'
                    ' "total_volume_ml": 1.0,'
                    ' "segments": []}'
                ),
```

**Step 5: Run ruff to verify 0 violations**

Run: `uv run ruff check src tests`
Expected: No errors

**Step 6: Run black to verify formatting**

Run: `uv run black --check src tests`
Expected: No reformatting needed (or run `uv run black src tests` to fix)

**Step 7: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: 0 failures

**Step 8: Commit**

```bash
git add pyproject.toml src/slicer_mcp/server.py tests/test_slicer_client.py tests/test_tools.py
git commit -m "style: fix all ruff lint violations and deprecated config"
```

---

### Task 9: Fix Duplicate Circuit Breaker Fixture (M2)

**Files:**
- Modify: `tests/test_slicer_client.py:12-19`

**Step 1: Remove the duplicate fixture**

Remove the `reset_circuit_breaker_fixture` fixture from `tests/test_slicer_client.py` (lines 12-19). The global `conftest.py` already defines an autouse `reset_circuit_breaker` fixture that covers all tests.

**Step 2: Run tests to verify no breakage**

Run: `uv run pytest tests/test_slicer_client.py -v --tb=short`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_slicer_client.py
git commit -m "refactor(test): remove duplicate circuit breaker reset fixture"
```

---

### Task 10: Fix `Optional` Import Style (M5)

**Files:**
- Modify: `src/slicer_mcp/slicer_client.py:9,114`

**Step 1: Update the import and usage**

In `src/slicer_mcp/slicer_client.py`:

1. Line 9: Remove `Optional` from the typing import (keep `Any`, `NoReturn`, `TypeVar`)
2. Line 114: Change `Optional["SlicerClient"]` to `"SlicerClient" | None`

```python
# Before (line 9):
from typing import Any, NoReturn, Optional, TypeVar

# After:
from typing import Any, NoReturn, TypeVar
```

```python
# Before (line 114):
_client_instance: Optional["SlicerClient"] = None

# After:
_client_instance: "SlicerClient | None" = None
```

**Step 2: Run tests**

Run: `uv run pytest -v --tb=short`
Expected: 0 failures

**Step 3: Commit**

```bash
git add src/slicer_mcp/slicer_client.py
git commit -m "style: use Python 3.10+ union syntax instead of Optional"
```

---

### Task 11: Add SLICER_TIMEOUT Environment Variable Validation (M6)

**Files:**
- Modify: `src/slicer_mcp/slicer_client.py:199`
- Modify: `tests/test_slicer_client.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_slicer_client.py` in `TestSlicerClientInit`:

```python
    def test_invalid_timeout_env_falls_back_to_default(self):
        """Non-numeric SLICER_TIMEOUT should fall back to default."""
        with patch.dict("os.environ", {"SLICER_TIMEOUT": "not_a_number"}):
            client = SlicerClient()
            assert client.timeout == 30  # DEFAULT_TIMEOUT_SECONDS
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slicer_client.py::TestSlicerClientInit::test_invalid_timeout_env_falls_back_to_default -v`
Expected: FAIL with `ValueError: invalid literal for int()`

**Step 3: Add validation with fallback**

In `src/slicer_mcp/slicer_client.py` line 199, change:

```python
# Before:
            timeout = int(os.environ.get("SLICER_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))

# After:
            timeout_str = os.environ.get("SLICER_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
            try:
                timeout = int(timeout_str)
                if timeout <= 0:
                    logger.warning(
                        f"SLICER_TIMEOUT must be positive, got {timeout}. "
                        f"Using default: {DEFAULT_TIMEOUT_SECONDS}s"
                    )
                    timeout = DEFAULT_TIMEOUT_SECONDS
            except ValueError:
                logger.warning(
                    f"Invalid SLICER_TIMEOUT value: '{timeout_str}'. "
                    f"Using default: {DEFAULT_TIMEOUT_SECONDS}s"
                )
                timeout = DEFAULT_TIMEOUT_SECONDS
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slicer_client.py::TestSlicerClientInit -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/slicer_mcp/slicer_client.py tests/test_slicer_client.py
git commit -m "fix: validate SLICER_TIMEOUT env var with fallback to default"
```

---

### Task 12: Increase Test Coverage to 85% (C2)

**Files:**
- Modify: `tests/test_resources.py` (add resource tests)
- Modify: `tests/test_tools.py` (add missing tool tests)

This task fills the biggest coverage gaps: `resources.py` at 55%, `tools.py` at 71%.

**Step 1: Add tests for `get_volumes_resource`**

In `tests/test_resources.py`, add:

```python
from unittest.mock import Mock, patch

from slicer_mcp.resources import get_volumes_resource, get_status_resource, get_scene_resource
from slicer_mcp.slicer_client import SlicerConnectionError


class TestGetSceneResource:
    """Test get_scene_resource."""

    def test_get_scene_resource_returns_json(self):
        """get_scene_resource should return valid JSON with nodes."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_scene_nodes.return_value = [
                {"id": "vtkMRMLScalarVolumeNode1", "name": "MRHead", "type": "vtkMRMLScalarVolumeNode"}
            ]
            mock_get_client.return_value = mock_client

            result = get_scene_resource()
            import json
            data = json.loads(result)
            assert data["node_count"] == 1
            assert data["nodes"][0]["name"] == "MRHead"

    def test_get_scene_resource_connection_error(self):
        """get_scene_resource should raise on connection error."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_scene_nodes.side_effect = SlicerConnectionError("fail")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                get_scene_resource()


class TestGetVolumesResource:
    """Test get_volumes_resource."""

    def test_get_volumes_resource_returns_json(self):
        """get_volumes_resource should return valid JSON with volumes."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"volumes": [], "total_count": 0}',
            }
            mock_get_client.return_value = mock_client

            result = get_volumes_resource()
            import json
            data = json.loads(result)
            assert data["total_count"] == 0

    def test_get_volumes_resource_connection_error(self):
        """get_volumes_resource should raise on connection error."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.side_effect = SlicerConnectionError("fail")
            mock_get_client.return_value = mock_client

            with pytest.raises(SlicerConnectionError):
                get_volumes_resource()


class TestGetStatusResource:
    """Test get_status_resource."""

    def test_get_status_resource_connected(self):
        """get_status_resource should return connected status."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.health_check.return_value = {
                "connected": True,
                "webserver_url": "http://localhost:2016",
                "response_time_ms": 10,
            }
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"slicer_version": "5.6.2", "scene_loaded": true, "python_available": true}',
            }
            mock_get_client.return_value = mock_client

            result = get_status_resource()
            import json
            data = json.loads(result)
            assert data["connected"] is True
            assert data["slicer_version"] == "5.6.2"

    def test_get_status_resource_disconnected(self):
        """get_status_resource should return disconnected status on error."""
        with patch("slicer_mcp.resources.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.base_url = "http://localhost:2016"
            mock_client.health_check.side_effect = SlicerConnectionError("fail")
            mock_get_client.return_value = mock_client

            result = get_status_resource()
            import json
            data = json.loads(result)
            assert data["connected"] is False
            assert data["error"] == "fail"
```

**Step 2: Add tests for `capture_screenshot` in tools.py**

In `tests/test_tools.py`, add (these cover the untested success paths):

```python
class TestCaptureScreenshotTool:
    """Test capture_screenshot tool."""

    def test_capture_screenshot_axial(self):
        """capture_screenshot should return base64 image for axial view."""
        from slicer_mcp.tools import capture_screenshot

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_screenshot.return_value = b"\x89PNG\r\n\x1a\n"
            mock_get_client.return_value = mock_client

            result = capture_screenshot("axial")
            assert result["success"] is True
            assert result["view_type"] == "axial"
            assert "image_base64" in result

    def test_capture_screenshot_full(self):
        """capture_screenshot should work for full view."""
        from slicer_mcp.tools import capture_screenshot

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_full_screenshot.return_value = b"\x89PNG\r\n\x1a\n"
            mock_get_client.return_value = mock_client

            result = capture_screenshot("full")
            assert result["success"] is True
            assert result["view_type"] == "full"

    def test_capture_screenshot_invalid_view(self):
        """capture_screenshot should reject invalid view_type."""
        from slicer_mcp.tools import capture_screenshot

        with pytest.raises(ValueError) as exc_info:
            capture_screenshot("invalid_view")
        assert "Invalid view_type" in str(exc_info.value)
```

**Step 3: Add tests for `set_layout` in tools.py**

```python
class TestSetLayoutTool:
    """Test set_layout tool."""

    def test_set_layout_valid(self):
        """set_layout should succeed with valid layout."""
        from slicer_mcp.tools import set_layout

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"success": true}',
            }
            mock_get_client.return_value = mock_client

            result = set_layout("FourUp")
            assert result["success"] is True

    def test_set_layout_invalid_layout(self):
        """set_layout should reject invalid layout name."""
        from slicer_mcp.tools import set_layout

        with pytest.raises(ValueError) as exc_info:
            set_layout("InvalidLayout")
        assert "Invalid layout" in str(exc_info.value)

    def test_set_layout_invalid_gui_mode(self):
        """set_layout should reject invalid gui_mode."""
        from slicer_mcp.tools import set_layout

        with pytest.raises(ValueError) as exc_info:
            set_layout("FourUp", gui_mode="invalid")
        assert "Invalid gui_mode" in str(exc_info.value)
```

**Step 4: Add tests for DICOM tools (list_dicom_studies, list_dicom_series, load_dicom_series, import_dicom)**

```python
class TestDICOMTools:
    """Test DICOM tool functions."""

    def test_list_dicom_studies(self):
        """list_dicom_studies should return parsed results."""
        from slicer_mcp.tools import list_dicom_studies

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"studies": [], "total_count": 0}',
            }
            mock_get_client.return_value = mock_client

            result = list_dicom_studies()
            assert result["total_count"] == 0

    def test_list_dicom_series(self):
        """list_dicom_series should validate study_uid and return results."""
        from slicer_mcp.tools import list_dicom_series

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"series": [], "total_count": 0}',
            }
            mock_get_client.return_value = mock_client

            result = list_dicom_series("1.2.840.113619")
            assert result["total_count"] == 0

    def test_load_dicom_series(self):
        """load_dicom_series should validate series_uid and return results."""
        from slicer_mcp.tools import load_dicom_series

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"success": true, "node_id": "vtkMRMLScalarVolumeNode1"}',
            }
            mock_get_client.return_value = mock_client

            result = load_dicom_series("1.2.840.113619")
            assert result["success"] is True

    def test_import_dicom(self, tmp_path):
        """import_dicom should validate path and return results."""
        from slicer_mcp.tools import import_dicom

        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()

        with patch("slicer_mcp.tools.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.exec_python.return_value = {
                "success": True,
                "result": '{"success": true, "patients_count": 1}',
            }
            mock_get_client.return_value = mock_client

            result = import_dicom(str(dicom_dir))
            assert result["success"] is True
```

**Step 5: Run coverage report**

Run: `uv run pytest --cov=slicer_mcp --cov-report=term-missing -q`
Expected: Coverage >= 85%

**Step 6: If still below 85%, add more tests for the largest remaining gaps**

Check the report and add tests for any remaining uncovered functions. Focus on `tools.py` lines that are still uncovered.

**Step 7: Commit**

```bash
git add tests/test_resources.py tests/test_tools.py
git commit -m "test: add resource and tool tests to reach 85% coverage"
```

---

### Task 13: Final Verification

**Step 1: Run full lint check**

Run: `uv run ruff check src tests`
Expected: 0 errors

**Step 2: Run black formatting check**

Run: `uv run black --check src tests`
Expected: No reformatting needed

**Step 3: Run full test suite with coverage**

Run: `uv run pytest --cov=slicer_mcp -v`
Expected: 0 failures, coverage >= 85%

**Step 4: Verify no regressions**

Run: `uv run pytest -v --tb=short`
Expected: All pass, 0 failures, 0 errors
