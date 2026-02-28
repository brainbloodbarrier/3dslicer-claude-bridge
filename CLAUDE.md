# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis. Uses FastMCP with stdio transport. The server exposes 12 tools and 3 resources over MCP/stdio, communicating with Slicer's WebServer extension over HTTP.

Entry point: `slicer-mcp` → `slicer_mcp:main` (in `server.py`).

## Commands

```bash
# Dependencies
uv sync                                    # Install dependencies
uv sync --all-extras                       # Install with dev + metrics extras

# Testing
uv run pytest -v                           # All tests (unit + integration + benchmarks)
uv run pytest -v -m "not integration and not benchmark"  # Unit tests only (no Slicer needed)
uv run pytest -v -m integration            # Integration tests (requires running Slicer)
uv run pytest -v -m benchmark              # Performance benchmarks (requires running Slicer)
uv run pytest --cov=slicer_mcp             # Coverage (fail_under=85, configured in .coveragerc)
uv run pytest tests/test_tools.py -v       # Single test file
uv run pytest tests/test_tools.py::TestCaptureScreenshotTool::test_axial_view -v  # Single test

# Code quality
uv run black src tests                     # Format code
uv run ruff check src tests                # Lint
uv run ruff check src tests --fix          # Lint with auto-fix
uv run pre-commit run --all-files          # All pre-commit hooks (black, ruff, mypy)

# Run
uv run slicer-mcp                          # Start the MCP server
```

## Architecture

```
Claude Code ──(MCP/stdio)──▶ server.py ──(HTTP)──▶ Slicer WebServer (localhost:2016)
                               │
                               ├─ tools.py           Tool logic (validation + Slicer Python codegen)
                               ├─ resources.py       Resource handlers (scene, volumes, status)
                               ├─ slicer_client.py   HTTP client (singleton + retry + circuit breaker)
                               ├─ circuit_breaker.py Three-state circuit breaker (closed/open/half-open)
                               ├─ constants.py       All config values, validation limits, patterns
                               └─ metrics.py         Optional Prometheus metrics (NullMetric when disabled)
```

### Data Flow

1. **server.py** registers tools/resources with `@mcp.tool()` / `@mcp.resource()`. Each tool wrapper catches all exceptions via `_handle_tool_error()` and returns standardized error dicts with `error_type` field (`circuit_open`, `timeout`, `connection`, `unexpected`).
2. **tools.py** contains the actual logic: validates inputs, builds Python code strings, calls `slicer_client.get_client().exec_python(code)`, and parses JSON results via `_parse_json_result()`. Tool code uses `__execResult = result` in generated Python to return data from Slicer (dict assigned directly — Slicer serializes it to JSON in the HTTP response).
3. **slicer_client.py** manages HTTP communication:
   - Singleton via `get_client()` with thread-safe double-checked locking (`threading.Lock`).
   - HTTP methods decorated with `@with_retry` (3 attempts, exponential backoff, `ConnectionError` only — timeouts NOT retried).
   - **`exec_python()` is intentionally NOT retried** — Python execution is not idempotent (Slicer may execute code even if HTTP response is lost). Sends code via `POST /slicer/exec` with `Content-Type: text/plain`.
   - No `requests.Session` — Slicer's WebServer closes connections immediately after response.
4. **circuit_breaker.py** protects against cascading failures. Opens after 5 consecutive failures, auto-recovers after 30s via HALF_OPEN test request. Thread-safe via `threading.Lock`; state lazily transitions to HALF_OPEN on read after timeout.

### Slicer Exec Endpoint

The `/slicer/exec` endpoint must be explicitly enabled in Slicer (Modules > Developer Tools > Web Server > "Enable Slicer API" + "Enable exec"). Result patterns:
- **`__execResult = value`**: Value serialized to JSON in response body — this is what all tools use. Assign a dict directly (not `json.dumps()`; that would double-encode).
- **`print()`**: In Slicer ≥5.8 stdout is captured in response text, but Slicer 5.10.0 stable does NOT capture `print()` output (returns `{}`). Do not use `print()` for returning data.
- Bare expressions return `{}` with no output.
- VTK collections don't support Python `len()` — use `.GetNumberOfItems()` instead.

### Key Patterns

- **Two-layer error handling**: tools.py catches `ValidationError` and `SlicerConnectionError` specifically; server.py wraps every tool call in `_handle_tool_error()` catching all remaining exceptions.
- **Input validation**: `validate_mrml_node_id()` checks regex `^[a-zA-Z][a-zA-Z0-9_]*$`. `validate_segment_name()` applies NFKC Unicode normalization and strips invisible zero-width characters. `validate_folder_path()` resolves symlinks via `os.path.realpath()` and prevents path traversal. Values are JSON-escaped via `json.dumps()` when injected into Python code (defense-in-depth).
- **Audit logging**: `execute_python` logs all code execution when `SLICER_AUDIT_LOG` env var is set. Entries include code hash, truncated preview, result preview, and UUID request ID. Audit log paths are validated against forbidden system directories (also resolving symlinks).
- **Metrics null object**: `NullMetric` provides interface compatibility when Prometheus is disabled, so code can always call metrics without conditional checks. `prometheus_client` is an optional dependency (`[metrics]` extra).
- **Logging to stderr**: stdout is reserved for MCP protocol; all logging goes to stderr in JSON format.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICER_URL` | `http://localhost:2016` | Slicer WebServer URL |
| `SLICER_TIMEOUT` | `30` | HTTP timeout in seconds (validated; invalid/non-positive values fall back to default) |
| `SLICER_AUDIT_LOG` | *(none)* | Audit log file path for execute_python |
| `SLICER_METRICS_ENABLED` | `false` | Enable Prometheus metrics (requires `prometheus_client`) |

## Testing

- Unit tests mock `requests.get/post` and `slicer_client.get_client()` — **no running Slicer needed**.
- Integration tests (`-m integration`) and benchmarks (`-m benchmark`) require a running Slicer instance with WebServer enabled, exec API enabled, and sample data loaded (e.g., MRHead).
- `conftest.py` has an autouse `reset_circuit_breaker` fixture that resets state before/after each test.
- Use `slicer_client.reset_client()` to reset the singleton between tests when needed.
- Fixtures: `slicer_client`, `mock_response`, `mock_slicer_exec_result`.
- Coverage threshold: 85% (configured in `.coveragerc` with branch coverage).

## Code Style

- **Formatter**: Black (line-length 100)
- **Linter**: Ruff (rules: E, F, W, I, N, UP — configured in `[tool.ruff.lint]`)
- **Type checker**: mypy (pre-commit hook, `src/` only, `--ignore-missing-imports`)
- **Python**: 3.10+ (uses `X | Y` union syntax, not `Optional[X]`)
- **Type hints**: Required on all public functions
- **Docstrings**: Google-style
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)

## Adding Features

- **New tool**: Implement in `tools.py`, register with `@mcp.tool()` in `server.py`, test in `tests/test_tools.py`
- **New resource**: Implement in `resources.py`, register with `@mcp.resource()` in `server.py`, test in `tests/test_resources.py`
- **New constant**: Add to `constants.py` (never hardcode validation limits, patterns, or config values in other files)

## Quick Reference

| Topic | File |
|-------|------|
| Tool API (12 tools) | [ref/api-tools.md](ref/api-tools.md) |
| Resource API (3 resources) | [ref/api-resources.md](ref/api-resources.md) |
| Error codes | [ref/error-codes.md](ref/error-codes.md) |
| Slicer HTTP endpoints | [ref/slicer-webserver.md](ref/slicer-webserver.md) |
| Design patterns | [ref/project-patterns.md](ref/project-patterns.md) |
| Circuit breaker & retry | [ref/resilience-patterns.md](ref/resilience-patterns.md) |
| Security model | [ref/security.md](ref/security.md) |
| Performance benchmarks | [ref/benchmarks.md](ref/benchmarks.md) |
| FastMCP framework | [ref/fastmcp.md](ref/fastmcp.md) |
| Troubleshooting | [ref/troubleshooting.md](ref/troubleshooting.md) |

## Agent Orchestration Guidelines
- **Scope First:** Use `find`, `grep`, or `lsp` to pinpoint exactly where changes belong.
- **Parallelize:** Obsessively use the Task tool to run `explore` (for discovery) or `reviewer` / `code-simplifier` / `pr-test-analyzer` (for validation) subagents in parallel.
- **Isolate Code:** Subagents must be given precise context (files to change, exact rules) because they lack history.
