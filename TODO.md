# PR #22 Review TODO List

> Generated from `silent-failure-hunter` and `pr-test-analyzer` agents
> Branch: `refactor/code-quality-improvements`
> Mobile-friendly for iteration on remote environment "Vibing"

---

## Quick Stats

| Category | Pending | Total |
|----------|---------|-------|
| Silent Failures | 6 | 7 |
| Test Gaps | 6 | 6 |
| **Total** | **12** | **13** |

---

## Silent Failure Fixes

### Critical (Must Fix)

- [ ] **`tools.py:1165-1167`** - Bare `except Exception` in `_get_valid_datasets()`
  ```python
  # Current: silently returns empty list on ANY error
  except Exception:
      return []

  # Fix: Add logging and specific exceptions
  except (SlicerConnectionError, json.JSONDecodeError) as e:
      logger.warning(f"Failed to get valid datasets: {e}")
      return []
  ```

### High Priority

- [ ] **`slicer_client.py:425-431`** - Version check logs at DEBUG, should be WARNING
  ```python
  # Current: DEBUG level hides version parse failures
  logger.debug(f"Could not parse Slicer version: {e}")

  # Fix: Elevate to WARNING
  logger.warning(f"Could not parse Slicer version: {e}")
  ```

- [ ] **`tools.py:1127-1130`** - Fallback list masks connection problems
  ```python
  # Current: Returns fallback silently
  except SlicerConnectionError:
      return FALLBACK_SAMPLE_DATASETS

  # Fix: Log before returning fallback
  except SlicerConnectionError as e:
      logger.warning(f"Slicer unavailable, returning fallback datasets: {e}")
      return FALLBACK_SAMPLE_DATASETS
  ```

### Medium Priority

- [ ] **`tools.py:892-897`** - Lost error context in brain extraction
  ```python
  # Fix: Chain exceptions
  except Exception as e:
      raise SlicerConnectionError(f"Segment lookup failed: {e}") from e
  ```

- [ ] **`slicer_client.py:356-360`** - Request exception handling too broad
  ```python
  # Consider distinguishing:
  # - ConnectionError: retry
  # - Timeout: no retry, log warning
  # - HTTPError: depends on status code
  ```

- [ ] **`resources.py:49-51`** - Scene resource error lacks context
  ```python
  # Fix: Add resource-specific context
  except SlicerConnectionError as e:
      logger.error(f"Scene resource retrieval failed: {e.message}")
      raise SlicerConnectionError(
          f"Failed to retrieve scene resource: {e.message}",
          error_code=e.error_code
      ) from e
  ```

### Acceptable (By Design)

- [x] **`circuit_breaker.py:163-165`** - Intentional fail-fast pattern
  - Documented in `ref/resilience-patterns.md`

---

## Test Coverage Gaps

### Critical (Blocks Merge)

- [ ] **Test `MAX_PYTHON_CODE_LENGTH` validation**
  - File: `tests/test_tools.py`
  - Function: `test_execute_python_exceeds_max_length`
  ```python
  def test_execute_python_exceeds_max_length():
      """execute_python should reject code exceeding MAX_PYTHON_CODE_LENGTH."""
      oversized_code = "x = 1\n" * 20000  # ~120KB
      with pytest.raises(ValidationError) as exc_info:
          execute_python(oversized_code)
      assert "maximum length" in str(exc_info.value)
  ```

### High Priority

- [ ] **Test `_build_segment_statistics_code()` helper**
  - File: `tests/test_tools.py`
  - Function: `test_build_segment_statistics_code`
  ```python
  def test_build_segment_statistics_code():
      """_build_segment_statistics_code should generate valid Python."""
      code = _build_segment_statistics_code("segNode")
      assert "import slicer" in code
      assert "SegmentStatisticsLogic" in code
      assert "segNode" in code
  ```

### Medium Priority

- [ ] **Test JSON error handling in `get_scene_nodes()`**
  - File: `tests/test_slicer_client.py`
  - Mock malformed JSON response
  ```python
  def test_get_scene_nodes_malformed_json(mock_client):
      """get_scene_nodes should raise on malformed JSON."""
      mock_client._mock_response.text = "not valid json"
      with pytest.raises(SlicerConnectionError) as exc_info:
          mock_client.get_scene_nodes()
      assert "parse" in str(exc_info.value).lower()
  ```

- [ ] **Test `_iso_timestamp()` format**
  - File: `tests/test_resources.py`
  ```python
  def test_iso_timestamp_format():
      """_iso_timestamp should return ISO 8601 UTC with Z suffix."""
      ts = _iso_timestamp()
      assert ts.endswith("Z")
      assert "T" in ts
      # Verify parseable
      datetime.fromisoformat(ts.replace("Z", "+00:00"))
  ```

- [ ] **Test `get_node_properties()` retry behavior**
  - File: `tests/test_slicer_client.py`
  - Verify `@with_retry` decorator works
  ```python
  def test_get_node_properties_retries_on_connection_error(mock_client):
      """get_node_properties should retry on SlicerConnectionError."""
      # Setup: fail twice, succeed third time
      mock_client._call_count = 0
      # Verify 3 calls made
  ```

### Low Priority (Observation)

- [ ] **Unused constant `SEGMENT_STATISTICS_VOLUME_KEY`**
  - Location: `src/slicer_mcp/constants.py`
  - Action: Import in `tools.py` or add docstring explaining reserved status

---

## Quick Commands

```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_tools.py -v

# Check test coverage
uv run pytest --cov=slicer_mcp --cov-report=term-missing

# Lint check
uv run ruff check src tests
```

---

## Progress Tracking

When completing items:
1. Mark checkbox with `[x]`
2. Add commit hash in comment if applicable
3. Run `grep -c "^\- \[ \]" TODO.md` to count remaining

**Last Updated:** 2026-01-10
