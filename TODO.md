# PR #22 Review TODO List

> Generated from `silent-failure-hunter` and `pr-test-analyzer` agents
> Branch: `refactor/code-quality-improvements`
> Mobile-friendly for iteration on remote environment "Vibing"

---

## Quick Stats

| Category | Pending | Total |
|----------|---------|-------|
| Silent Failures | 0 | 7 |
| Test Gaps | 0 | 6 |
| **Total** | **0** | **13** |

---

## Silent Failure Fixes

### Critical (Must Fix)

- [x] **`tools.py:814-818`** - Bare `except Exception` in `_get_valid_datasets()`
  - **FIXED**: Added specific exceptions + logging
  ```python
  except (SlicerConnectionError, json.JSONDecodeError, KeyError, TypeError) as e:
      logger.warning(f"Dynamic dataset discovery failed, using fallback: {e}")
      return FALLBACK_SAMPLE_DATASETS
  ```

### High Priority

- [x] **`slicer_client.py:400`** - Version check logs at DEBUG, should be WARNING
  - **FALSE POSITIVE**: Already uses `logger.warning()` at line 400
  - Verified: `logger.warning(f"Could not parse version '{current}': {e}")`

- [x] **`tools.py:794-795`** - Fallback list masks connection problems
  - **FALSE POSITIVE**: Already logs before returning fallback
  - Verified: `logger.warning(f"Dynamic sample data discovery failed, using fallback: {e.message}")`

### Medium Priority

- [x] **`tools.py` (line numbers shifted)** - Lost error context in brain extraction
  - **FALSE POSITIVE**: Code was reorganized; error handling is properly structured
  - Brain extraction uses `_parse_json_result()` which raises on errors

- [x] **`slicer_client.py:215-261`** - Request exception handling too broad
  - **FALSE POSITIVE**: `_handle_request_error()` already distinguishes:
    - `Timeout` → raises `SlicerTimeoutError`
    - `ConnectionError` → raises `SlicerConnectionError` with connection-specific message
    - Other → generic `SlicerConnectionError`

- [x] **`resources.py:49-51`** - Scene resource error lacks context
  - **ACCEPTABLE**: Already has `logger.error()` + re-raises original exception
  - Exception chaining would be marginal improvement, not required

### Acceptable (By Design)

- [x] **`circuit_breaker.py:163-165`** - Intentional fail-fast pattern
  - Documented in `ref/resilience-patterns.md`

---

## Test Coverage Gaps

### Critical (Blocks Merge)

- [x] **Test `MAX_PYTHON_CODE_LENGTH` validation**
  - File: `tests/test_tools.py`
  - Class: `TestExecutePythonCodeLengthValidation`
  - **DONE**: 3 tests added (accepts at limit, rejects over, error includes size)

### High Priority

- [x] **Test `_build_segment_statistics_code()` helper**
  - File: `tests/test_tools.py`
  - Class: `TestBuildSegmentStatisticsCode`
  - **DONE**: 5 tests added (imports, variable usage, volume calc, error handling, variable names)

### Medium Priority

- [x] **Test JSON error handling in `get_scene_nodes()`**
  - File: `tests/test_slicer_client.py`
  - Class: `TestGetSceneNodesJsonHandling`
  - **DONE**: 3 tests added (malformed names, malformed IDs, valid JSON)

- [x] **Test `_iso_timestamp()` format**
  - File: `tests/test_resources.py` (created)
  - Class: `TestIsoTimestamp`
  - **DONE**: 7 tests added (string type, Z suffix, T separator, parseable, structure, UTC, seconds precision)

- [x] **Test `get_node_properties()` retry behavior**
  - File: `tests/test_slicer_client.py`
  - Class: `TestGetNodePropertiesRetry`
  - **DONE**: 3 tests added (retries on error, exhausts retries, respects circuit breaker)

### Low Priority (Observation)

- [x] **Unused constant `SEGMENT_STATISTICS_VOLUME_KEY`**
  - **DEFERRED**: Reserved for future use in segment statistics volume selection
  - No action required for this PR

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
**Verified By:** Claude Code deep-dive analysis
