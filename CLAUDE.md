# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis. Uses FastMCP with stdio transport. The server exposes 12 tools and 3 resources over MCP/stdio, communicating with Slicer's WebServer extension over HTTP. Entry point: `slicer-mcp` → `slicer_mcp:main` (in `server.py`).

## Commands

```bash
uv sync                                    # Install dependencies
uv run pytest -v                           # Run all unit tests (no Slicer required)
uv run pytest tests/test_tools.py -v       # Run a single test file
uv run pytest tests/test_tools.py::test_capture_screenshot_axial -v  # Run a single test
uv run pytest -v -m integration            # Integration tests (requires running Slicer)
uv run pytest -v -m benchmark              # Performance benchmarks
uv run pytest --cov=slicer_mcp             # Coverage (uses .coveragerc, fail_under=85)
uv run black src tests                     # Format code
uv run ruff check src tests                # Lint
uv run ruff check src tests --fix          # Lint with auto-fix
uv run pre-commit run --all-files          # Run all pre-commit hooks
uv run slicer-mcp                          # Run the MCP server
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

### Key Data Flow

1. **server.py** registers tools/resources with `@mcp.tool()` / `@mcp.resource()`. Each tool wrapper catches all exceptions via `_handle_tool_error()` and returns standardized error dicts.
2. **tools.py** contains the actual logic: validates inputs, builds Python code strings, calls `slicer_client.get_client().exec_python(code)`, and parses JSON results via `_parse_json_result()`.
3. **slicer_client.py** manages HTTP communication. Uses a singleton pattern (`get_client()`) with thread-safe double-checked locking. Every HTTP method is decorated with `@with_retry` (3 attempts, exponential backoff for `ConnectionError` only; timeouts are NOT retried). **`exec_python()` is intentionally NOT retried** — code execution is not idempotent (Slicer may have received the code even if the HTTP response was lost).
4. **circuit_breaker.py** protects against cascading failures. Opens after 5 consecutive failures, auto-recovers after 30s via HALF_OPEN test request. Thread-safe via `threading.Lock`; state lazily transitions to HALF_OPEN on read after timeout.

### Important Patterns

- **Tool implementation**: server.py delegates to tools.py which generates Python code strings executed in Slicer's environment. Many tools work by building multi-line Python scripts, sending them via `exec_python()`, and parsing JSON output via `print(json.dumps(result))`. All JSON parsing goes through the shared `_parse_json_result()` helper for consistent error handling.
- **Two-layer error handling**: tools.py catches `ValidationError` and `SlicerConnectionError` specifically; server.py wraps every tool call in `_handle_tool_error()` which catches all remaining exceptions and returns standardized error dicts with `error_type` field (`circuit_open`, `timeout`, `connection`, `unexpected`).
- **Input validation**: All user inputs are validated before use. `validate_mrml_node_id()` checks regex `^[a-zA-Z][a-zA-Z0-9_]*$`. `validate_segment_name()` applies NFKC Unicode normalization and strips invisible zero-width characters (homoglyph attack prevention). `validate_folder_path()` prevents path traversal. Values are also JSON-escaped via `json.dumps()` when injected into Python code (defense-in-depth).
- **Audit logging**: `execute_python` logs all code execution to a dedicated audit logger when `SLICER_AUDIT_LOG` env var is set. Entries include code hash, truncated preview, result preview, and a UUID request ID. Audit log paths are validated to prevent writing to sensitive directories.
- **No `requests.Session`**: Direct `requests.get/post` calls are used intentionally — Slicer's WebServer closes connections immediately after response.
- **Metrics null object**: `NullMetric` provides interface compatibility when `SLICER_METRICS_ENABLED` is false, so tool code can always call metrics without conditional checks.
- **Logging to stderr**: stdout is reserved for MCP protocol; all logging goes to stderr in JSON format.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICER_URL` | `http://localhost:2016` | Slicer WebServer URL |
| `SLICER_TIMEOUT` | `30` | HTTP timeout in seconds |
| `SLICER_AUDIT_LOG` | *(none)* | Audit log file path for execute_python |
| `SLICER_METRICS_ENABLED` | `false` | Enable Prometheus metrics (requires `prometheus_client`) |

## Testing

- Unit tests mock `requests.get/post` and `slicer_client.get_client()` — **no running Slicer needed**.
- `conftest.py` has an autouse `reset_circuit_breaker` fixture that resets state before/after each test to prevent cross-test contamination.
- Use `slicer_client.reset_client()` to reset the singleton between tests when needed.
- Fixtures: `slicer_client`, `mock_response`, `mock_slicer_exec_result`.
- Pytest markers: `integration` (requires running Slicer), `benchmark` (performance tests). Unit tests run by default with no markers.
- Coverage threshold: 85% (configured in `.coveragerc`).

## Code Style

- **Formatter**: Black (line-length 100)
- **Linter**: Ruff (rules: E, F, W, I, N, UP)
- **Type checker**: mypy (pre-commit hook, `src/` only, `--ignore-missing-imports`)
- **Python**: 3.10+ (uses `X | Y` union syntax)
- **Type hints**: Required on all public functions
- **Docstrings**: Google-style
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)

## Adding Features

- **New tool**: Implement in `tools.py`, register with `@mcp.tool()` in `server.py`, test in `tests/test_tools.py`
- **New resource**: Implement in `resources.py`, register with `@mcp.resource()` in `server.py`, test in `tests/test_mcp_protocol.py`
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
