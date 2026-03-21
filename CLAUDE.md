# CLAUDE.md — slicer-mcp

## Architecture

Three layers: `server.py` (MCP registration) → `features/` (domain logic) → `core/` (infrastructure).

```
src/slicer_mcp/
├── server.py           # MCP tool + resource registration wrappers
├── core/               # HTTP client, circuit breaker, constants, metrics, resources
├── features/           # Domain tools: base, diagnostics/, spine/, workflows/, registration, rendering
└── *.py (14 shims)     # Backward-compat re-exports → canonical core/ and features/ paths
```

Full architectural details: `src/slicer_mcp/AGENTS.md`

## Development Commands

```bash
# Tests (no Slicer required)
uv run pytest -v -m "not integration and not benchmark"

# Coverage (85% enforced in CI on Python 3.12; currently 93%)
uv run pytest --cov=slicer_mcp --cov-report=term-missing --cov-fail-under=85 -m "not integration and not benchmark"

# Lint + format + type check
uv run ruff check src tests
uv run black --check src tests
uv run mypy src/

# Pre-commit (runs all above)
uv run pre-commit run --all-files
```

Note: `uv sync` installs runtime deps only. Dev tools need: `uv pip install pytest-cov black mypy types-requests`

## Key Conventions

- **New tools**: implement in `features/*.py`, register wrapper in `server.py`
- **New code must import from canonical paths**: `slicer_mcp.core.*` or `slicer_mcp.features.*`
- **Reuse canonical constants**: check `spine/constants.py` and `spine/tools.py` before defining new sets (e.g., `SPINE_REGIONS`, `VALID_POPULATIONS`, `SINS_PAIN_SCORES`)
- **Error handling**: feature code raises `ValidationError` or `SlicerConnectionError`; `server.py:_handle_tool_error()` catches all
- **No `assert` for validation/type narrowing**: use `ValidationError` (user input) or `RuntimeError` (internal invariants)
- **Validation**: all user inputs validated before network calls (node IDs, paths, DICOM UIDs)
- **Generated Python code**: features build code strings executed in Slicer via `exec_python()`. Use `json.dumps()` for param substitution. Edits happen inside f-strings.
- **Tests**: mock `get_client()` and `exec_python()` — never require a running Slicer instance for unit tests
- **Commit style**: conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)

## Project Status

`docs/plans/PROJECT_STATUS.md` — single tracking file (pending, blocked, completed work)
