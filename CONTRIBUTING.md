# Contributing to slicer-mcp

Thank you for your interest in contributing to the MCP Slicer Bridge -- an MCP server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis.

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- 3D Slicer with WebServer extension (only for integration tests)

### Setup

```bash
# Clone the repository
git clone https://github.com/brainbloodbarrier/3dslicer-claude-bridge.git
cd 3dslicer-claude-bridge

# Install all dependencies (runtime + dev + metrics)
uv sync --extra dev --extra metrics

# Install pre-commit hooks
uv run pre-commit install
```

For Claude Code and Cursor MCP client configuration, see
[`docs/guides/setup-mcp-clients.md`](docs/guides/setup-mcp-clients.md).

## Architecture Overview

The codebase follows a strict three-layer architecture:

```
server.py  -->  features/  -->  core/
(MCP registration)  (domain logic)  (infrastructure)
```

- **`server.py`** -- FastMCP entry point. Delegates tool registration to `_registry/` modules.
- **`_registry/*.py`** -- Domain-specific registration modules (base, spine, diagnostics, workflows, rendering, registration, resources). Each exports a `register_*_tools(mcp)` function.
- **`features/`** -- Domain logic organized by clinical area: `base_tools`, `spine/`, `diagnostics/`, `workflows/`, `registration`, `rendering`.
- **`core/`** -- Infrastructure: HTTP client (`slicer_client.py`), circuit breaker, constants, metrics, resources.
- **`*.py` (14 shims)** -- Root-level backward-compatibility re-exports. New code must not import from these.

**Data flow**: Claude --> MCP stdio --> `server.py` --> `features/*.py` builds Python code string --> `core/slicer_client.py:exec_python()` sends HTTP POST to Slicer WebServer API --> Slicer executes Python --> response returned.

For full architectural details, see [`src/slicer_mcp/AGENTS.md`](src/slicer_mcp/AGENTS.md).

## Adding a New Tool

### 1. Implement the feature function

Create or extend a module in `features/`. Your function receives parameters, builds a Python code string, and executes it via the Slicer client:

```python
# src/slicer_mcp/features/my_domain.py
import json
from slicer_mcp.core.slicer_client import get_client
from slicer_mcp.features.base_tools import ValidationError

def my_new_tool(param: str, threshold: float = 0.5) -> dict:
    if not param:
        raise ValidationError("param is required", field="param", value=param)

    client = get_client()
    code = f"""
import slicer
result = do_something({json.dumps(param)}, {threshold})
print(json.dumps(result))
"""
    return client.exec_python(code)
```

Key rules:
- Use `json.dumps()` for safe parameter substitution inside f-strings.
- Raise `ValidationError` for bad input (before any network call).
- Never swallow exceptions -- let them propagate to the error handler.
- No `assert` for validation; use `ValidationError` (user input) or `RuntimeError` (internal invariants).

### 2. Register the tool in `_registry/`

Add a registration call in the appropriate `_registry/*.py` module (or create a new one for a new domain):

```python
# src/slicer_mcp/_registry/my_domain.py
from typing import Any
from slicer_mcp._registry._common import register_tool
from slicer_mcp.features import my_domain

def register_my_domain_tools(mcp: Any) -> dict[str, Any]:
    wrappers: dict[str, Any] = {}

    def _reg(fn_name: str, doc: str) -> None:
        wrappers[fn_name] = register_tool(mcp, my_domain, fn_name, doc)

    _reg(
        "my_new_tool",
        """Description for MCP tool schema.

    Args:
        param: What this parameter does
        threshold: Numeric threshold (default 0.5)

    Returns:
        Dict with result data
    """,
    )

    return wrappers
```

Then wire it into `_registry/__init__.py` and call it from `server.py`.

### 3. Write unit tests

Mock `get_client()` and `exec_python()` -- never require a running Slicer instance:

```python
# tests/unit/test_my_domain.py
from unittest.mock import MagicMock, patch

def test_my_new_tool_success():
    mock_client = MagicMock()
    mock_client.exec_python.return_value = {"success": True, "data": "result"}

    with patch("slicer_mcp.features.my_domain.get_client", return_value=mock_client):
        result = my_new_tool("test_param", threshold=0.8)

    assert result["success"] is True
    mock_client.exec_python.assert_called_once()

def test_my_new_tool_validation_error():
    with pytest.raises(ValidationError):
        my_new_tool("")
```

### 4. Add wrapper tests in `test_server.py`

Verify the `server.py` wrapper correctly routes calls and handles errors via `_handle_tool_error()`.

## Code Conventions

- **Imports**: Always use canonical paths (`slicer_mcp.core.*`, `slicer_mcp.features.*`). The 14 root shim files exist only for backward compatibility.
- **Error handling**: Feature code raises `ValidationError` or `SlicerConnectionError`. Never swallow exceptions -- let them propagate to `_handle_tool_error()`.
- **No `assert` for validation**: Use `ValidationError` for user input, `RuntimeError` for internal invariants.
- **Line length**: 100 characters (enforced by Black + Ruff).
- **Target Python**: 3.10+ (no walrus operator in hot paths, but f-strings and `|` union types are fine).
- **Logging**: Goes to stderr (stdout is reserved for MCP protocol JSON). Use JSON structured logging.
- **Constants**: Check `features/spine/constants.py` and `features/spine/tools.py` before defining new constant sets.
- **Generated code**: Features build Python code strings executed in Slicer via `exec_python()`. Always use `json.dumps()` for safe parameter interpolation.

## Testing

```bash
# Unit tests (no running Slicer needed)
uv run pytest -v -m "not integration and not benchmark"

# Single test file or specific test
uv run pytest tests/unit/test_spine_tools.py -v
uv run pytest tests/unit/test_spine_tools.py::test_function_name -v

# Coverage report (85% enforced in CI)
uv run pytest --cov=slicer_mcp --cov-report=term-missing --cov-fail-under=85 \
  -m "not integration and not benchmark"

# Integration tests (requires Slicer running on localhost:2016)
uv run pytest -v -m integration

# Lint + format + type check
uv run ruff check src tests
uv run black --check src tests
uv run mypy src/

# Run all checks at once
uv run pre-commit run --all-files
```

- **Unit tests** mock `get_client()` and `exec_python()` and never require a live Slicer instance.
- **Integration tests** are marked with `@pytest.mark.integration` and require a running 3D Slicer with the WebServer extension on `localhost:2016`.
- **Coverage target**: 85%+ (currently ~93%). CI enforces the 85% floor.
- `conftest.py` auto-resets the circuit breaker between tests to ensure isolation.

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix      | Use case                           |
|-------------|------------------------------------|
| `feat:`     | New feature or tool                |
| `fix:`      | Bug fix                            |
| `docs:`     | Documentation changes              |
| `test:`     | Test additions or changes          |
| `refactor:` | Code restructuring (no behavior change) |
| `chore:`    | Maintenance, dependency updates    |
| `perf:`     | Performance improvements           |
| `ci:`       | CI/CD pipeline changes             |

Example: `feat: add vertebral body segmentation tool`

## PR Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests first** -- follow TDD (RED -> GREEN -> REFACTOR).
3. **Implement** the feature or fix.
4. **Run the full check suite**: `uv run pre-commit run --all-files`
5. **Ensure coverage** stays at 85%+.
6. **Submit a PR** with a description that includes:
   - **Summary**: What changed and why (1-3 bullet points).
   - **Test plan**: How reviewers can verify the change.
   - **Breaking changes**: Flag any changes to existing tool signatures or behavior.
7. All CI checks must pass before merge.

## Questions?

- Open a [GitHub issue](https://github.com/brainbloodbarrier/3dslicer-claude-bridge/issues)
