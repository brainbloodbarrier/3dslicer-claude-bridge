# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

MCP server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis. Three layers: `server.py` (MCP registration) → `features/` (domain logic) → `core/` (infrastructure).

```
src/slicer_mcp/
├── server.py           # ALL @mcp.tool() + @mcp.resource() wrappers (FastMCP)
├── core/               # HTTP client, circuit breaker, constants, metrics, resources
├── features/           # Domain tools: base, diagnostics/, spine/, workflows/, registration, rendering
└── *.py (14 shims)     # Backward-compat re-exports → canonical core/ and features/ paths
```

**Data flow**: Claude → MCP stdio → `server.py` wrapper → `features/*.py` builds Python code string → `core/slicer_client.py:exec_python()` sends HTTP POST to Slicer WebServer API → Slicer executes Python internally → response returned.

**Error propagation** (innermost to outermost):
1. `core/slicer_client.py` — maps `requests` exceptions to `SlicerConnectionError`/`SlicerTimeoutError`
2. `features/*.py` — raises `ValidationError` (bad input) before any network call
3. `server.py:_handle_tool_error()` — catch-all, maps to standardized `error_type` field

Full architectural details: `src/slicer_mcp/AGENTS.md`

## Development Commands

```bash
# Install all deps (runtime + dev + metrics)
uv sync && uv pip install pytest-cov black mypy types-requests

# Run unit tests (no running Slicer needed)
uv run pytest -v -m "not integration and not benchmark"

# Run a single test file or specific test
uv run pytest tests/unit/test_spine_tools.py -v
uv run pytest tests/unit/test_spine_tools.py::test_function_name -v

# Coverage (85% enforced in CI; currently 93%)
uv run pytest --cov=slicer_mcp --cov-report=term-missing --cov-fail-under=85 -m "not integration and not benchmark"

# Lint + format + type check
uv run ruff check src tests
uv run black --check src tests
uv run mypy src/

# Pre-commit (runs all above)
uv run pre-commit run --all-files
```

## Key Conventions

- **New tools**: implement in `features/*.py`, register wrapper in `server.py` with try/except calling `_handle_tool_error()`. No auto-discovery — every tool must be manually registered.
- **Imports**: always use canonical paths `slicer_mcp.core.*` or `slicer_mcp.features.*`. The 14 root shim files exist only for backward compat.
- **Reuse constants**: check `features/spine/constants.py` and `features/spine/tools.py` before defining new sets (e.g., `SPINE_REGIONS`, `VALID_POPULATIONS`, `SINS_PAIN_SCORES`)
- **Error handling**: feature code raises `ValidationError` or `SlicerConnectionError`; never swallow exceptions — let them propagate to `_handle_tool_error()`.
- **No `assert` for validation**: use `ValidationError` (user input) or `RuntimeError` (internal invariants)
- **Generated Python code**: features build code strings executed in Slicer via `exec_python()`. Use `json.dumps()` for safe param substitution inside f-strings.
- **Tests**: mock `get_client()` and `exec_python()` — never require a running Slicer instance. `conftest.py` auto-resets the circuit breaker between tests.
- **Line length**: 100 chars (black + ruff). Target Python 3.10+.
- **Commit style**: conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)
- **Logging**: goes to stderr (stdout reserved for MCP protocol JSON). Format is JSON structured logging.

## Project Status

`docs/plans/PROJECT_STATUS.md` — single tracking file (pending, blocked, completed work)
