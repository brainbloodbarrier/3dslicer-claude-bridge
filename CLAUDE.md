# CLAUDE.md — slicer-mcp

## Architecture

Three layers: `server.py` (MCP registration) → `features/` (domain logic) → `core/` (infrastructure).

```
src/slicer_mcp/
├── server.py           # 46 @mcp.tool() + 4 @mcp.resource() wrappers
├── core/               # HTTP client, circuit breaker, constants, metrics, resources
├── features/           # Domain tools: base, diagnostics/, spine/, workflows/, registration, rendering
└── *.py (14 shims)     # Backward-compat re-exports → canonical core/ and features/ paths
```

Full architectural details: `src/slicer_mcp/AGENTS.md`

## Development Commands

```bash
# Tests (no Slicer required)
uv run pytest -v -m "not integration and not benchmark"

# Coverage
uv run pytest --cov=slicer_mcp --cov-report=term-missing --cov-fail-under=85 -m "not integration and not benchmark"

# Lint + format + type check
uv run ruff check src tests
uv run black --check src tests
uv run mypy src/

# Pre-commit (runs all above)
uv run pre-commit run --all-files
```

## Key Conventions

- **New tools**: implement in `features/*.py`, register wrapper in `server.py`
- **New code must import from canonical paths**: `slicer_mcp.core.*` or `slicer_mcp.features.*`
- **Error handling**: feature code raises `ValidationError` or `SlicerConnectionError`; `server.py:_handle_tool_error()` catches all
- **Validation**: all user inputs validated before network calls (node IDs, paths, DICOM UIDs)
- **Generated Python code**: use `json.dumps()` for parameter substitution (prevents injection)
- **Tests**: mock `get_client()` and `exec_python()` — never require a running Slicer instance for unit tests
- **Commit style**: conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)

## Active Plans

- `docs/plans/2026-03-07-v2-roadmap.md` — v2 direction and phasing
- `docs/plans/v2-workflow-surface.md` — workflow tier assignment (Tier 1/2/3)
- `docs/plans/workflow-audit-results.md` — workflow doc vs tool signature audit
