> Extends `~/.claude/CLAUDE.md` — see global config for universal conventions (security, testing, debugging, communication).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

MCP server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis. Three layers: `server.py` (MCP registration) → `features/` (domain logic) → `core/` (infrastructure).

```
src/slicer_mcp/
├── server.py           # Entry point (~90 lines): FastMCP init + delegates to _registry/
├── _registry/          # Tool registration by domain (base, diagnostics, spine, workflows, etc.)
│   ├── _common.py      # Shared: handle_tool_error(), register_tool()
│   ├── _async_common.py # Async variant: register_async_tool()
│   ├── base.py         # 12 base tools (screenshot, DICOM, scene)
│   ├── diagnostics.py  # 16 diagnostic tools (CT, MRI, X-ray)
│   ├── spine.py        # 7 spine surgery tools + instrumentation
│   ├── workflows.py    # 3 orchestrated clinical workflows
│   ├── registration.py # 5 landmark + volume registration tools
│   ├── rendering.py    # 5 volume rendering + export tools
│   └── resources.py    # 4 slicer:// MCP resources
├── core/               # HTTP clients, circuit breaker, config, constants, metrics, resources
│   ├── config.py       # Pydantic Settings runtime config validation
│   ├── async_client.py # Async HTTP client (httpx) for non-blocking I/O
│   └── ...
├── features/           # Domain tools: base, diagnostics/, spine/, workflows/, registration, rendering
└── *.py (14 shims)     # Backward-compat re-exports → canonical core/ and features/ paths
```

**Data flow**: Claude → MCP stdio → `server.py` wrapper → `features/*.py` builds Python code string → `core/slicer_client.py:exec_python()` sends HTTP POST to Slicer WebServer API → Slicer executes Python internally → response returned.

**Error propagation** (innermost to outermost):
1. `core/slicer_client.py` — maps `requests` exceptions to `SlicerConnectionError`/`SlicerTimeoutError`
2. `features/*.py` — raises `ValidationError` (bad input) before any network call
3. `_registry/_common.py:handle_tool_error()` — catch-all, maps to standardized `error_type` field

Full architectural details: `src/slicer_mcp/AGENTS.md`

## Development Commands

```bash
# Install all deps (runtime + dev + metrics)
uv sync --extra dev --extra metrics

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

- **New tools**: implement in `features/*.py`, register in the appropriate `_registry/*.py` domain module via `register_tool()`. No auto-discovery — every tool must be manually registered.
- **Imports**: always use canonical paths `slicer_mcp.core.*` or `slicer_mcp.features.*`. The 14 root shim files exist only for backward compat.
- **Reuse constants**: check `features/spine/constants.py` and `features/spine/tools.py` before defining new sets (e.g., `SPINE_REGIONS`, `VALID_POPULATIONS`, `SINS_PAIN_SCORES`)
- **Error handling**: feature code raises `ValidationError` or `SlicerConnectionError`; never swallow exceptions — let them propagate to `_handle_tool_error()`.
- **No `assert` for validation**: use `ValidationError` (user input) or `RuntimeError` (internal invariants)
- **Generated Python code**: features build code strings executed in Slicer via `exec_python()`. Use `json.dumps()` for safe param substitution inside f-strings.
- **Tests**: mock `get_client()` and `exec_python()` — never require a running Slicer instance. `conftest.py` auto-resets the circuit breaker between tests.
- **Line length**: 100 chars (black + ruff). Target Python 3.10+.
- **Commit style**: conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)
- **Logging**: goes to stderr (stdout reserved for MCP protocol JSON). Format is JSON structured logging.

## Production Surface

`slicer-prod/` — minimal directory for using the MCP server with Claude Code and OpenCode without loading the full development tree. See `slicer-prod/README.md` for usage.

- Claude Code config: `slicer-prod/.mcp.json`
- OpenCode config: `slicer-prod/opencode.json`
- Claude commands: symlinked from `.claude/commands/`
- OpenCode commands: generated via `python3 slicer-prod/scripts/sync_surface.py`
- CI validates sync: `slicer-prod/scripts/sync_surface.py --check` runs in the lint job

## Project Status

`docs/plans/PROJECT_STATUS.md` — single tracking file (pending, blocked, completed work)
