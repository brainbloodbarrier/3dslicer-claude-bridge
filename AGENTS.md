# Repository Guidelines

## Project Overview
This repository contains `slicer-mcp`, a Model Context Protocol (MCP) server that bridges Claude Code to 3D Slicer for AI-assisted medical image analysis. It exposes Slicer's Python environment and MRML scene data to Claude via an MCP stdio transport.

## Architecture & Data Flow
The system acts as a middleware between Claude and a running 3D Slicer instance.

**Data Flow:**
1. Claude sends an MCP tool call over `stdio`.
2. `server.py` routes it to the correct handler (`@mcp.tool()` or `@mcp.resource()`).
3. The implementation module (e.g., `tools.py`, `spine_tools.py`) validates the input to prevent injection attacks.
4. It dynamically generates a Slicer Python script and serializes inputs safely via `json.dumps()`.
5. The HTTP client (`slicer_client.py`) POSTs the script to the 3D Slicer WebServer's `/slicer/exec` endpoint.
6. The result is returned via the `__execResult = <dict>` assignment (not `print()`), parsed, and sent back to Claude.

**Resilience Patterns:**
- **Retry Mechanism**: `@with_retry` decorator handles transient `SlicerConnectionError` (3 attempts, exponential backoff) for non-mutating calls.
- **Circuit Breaker**: A global singleton thread-safe circuit breaker prevents cascading failures if Slicer is down (opens after 5 failures, recovers after 30s).
- **Error Handling**: Top-level catch in `server.py` converts exceptions into typed JSON error dictionaries (`circuit_open`, `timeout`, `connection`, `unexpected`).

## Key Directories
- `src/slicer_mcp/` - Core Python package
  - `server.py` - FastMCP server entry point and tool registrations.
  - `tools.py` - Core utilities, Python codegen helpers, input validators.
  - `diagnostic_tools_*.py` / `spine_tools.py` / `instrumentation_tools.py` - Domain-specific tool implementations.
  - `slicer_client.py` - Singleton HTTP client handling retries and errors.
  - `circuit_breaker.py` - State machine for HTTP resilience.
  - `constants.py` / `spine_constants.py` - Single source of truth for ALL configuration, limits, and medical constants.
- `tests/` - Unit, integration, and benchmark tests.
- `ref/` - Internal reference documentation detailing API, tools, FastMCP, and Slicer WebServer.
- `scripts/` - Utilities like `claude-setup.sh` (tooling helper, not part of the build).

## Code Conventions & Common Patterns
- **Language**: Python 3.10+ (Use `X | Y` for Unions, not `Optional[X]`).
- **Formatting & Linting**: Managed by Black (100 line length) and Ruff.
- **Type Hinting**: Required on all public functions. Checked via `mypy`.
- **Python Code Generation**: Tools construct Slicer Python code dynamically.
  - ALWAYS wrap injected variables with `json.dumps()` to prevent Python injection attacks in Slicer.
  - Slicer code MUST return results by assigning to `__execResult` (e.g., `__execResult = {"status": "ok"}`). DO NOT use `print()`.
- **No Hardcoded Constants**: ALL thresholds, validation limits, or clinical measurements must be placed in `constants.py` or `spine_constants.py`.
- **Logging**: All logs go to `stderr` (since `stdout` is used by MCP). Use `logger = logging.getLogger('slicer-mcp')`.
- **Metrics**: A `NullMetric` pattern is used. Instrument code assuming metrics exist; if disabled, `NullMetric` makes calls a no-op.

## Development Commands
The project uses `uv` for dependency management and execution.

- **Run all unit tests**: `uv run pytest -v`
- **Run tests with coverage**: `uv run pytest --cov=slicer_mcp --cov-report=html` (requires 85% branch coverage)
- **Run integration tests**: `uv run pytest -v -m integration` (requires 3D Slicer running on `localhost:2016`)
- **Run benchmarks**: `uv run pytest tests/benchmarks/ -v -s --durations=0`
- **Format code**: `uv run black src tests`
- **Lint code**: `uv run ruff check src tests`
- **Type check**: `uv run pre-commit run mypy --all-files`

## Important Files
- `src/slicer_mcp/server.py`: The root of the application, defining all MCP endpoints. Start here to understand the exposed API.
- `src/slicer_mcp/tools.py`: Contains critical validation functions (e.g., `validate_mrml_node_id`) that protect the Slicer instance.
- `src/slicer_mcp/constants.py`: Centralized configuration.
- `tests/conftest.py`: Contains global fixtures. Crucially, it automatically resets the circuit breaker between tests to prevent test pollution.
- `CLAUDE.md`: Highly detailed development documentation. Refer to this for complex Slicer-specific behaviors.

## Testing & QA Expectations
- **Framework**: `pytest`, `pytest-asyncio`, and `pytest-cov`.
- **Mocking**: Tests do NOT require Slicer to run. The `slicer_client.get_client()` method is patched to return mock responses.
- **Isolation**: The `reset_circuit_breaker` autouse fixture runs around every test to prevent shared singleton state from leaking.
- **Coverage**: A minimum of 85% branch coverage is strictly enforced by `.coveragerc`.
- **Security Testing**: Any new validation logic must include dedicated unit tests proving it blocks code injection attempts.

## Runtime/Tooling Preferences
- **Package Manager**: `uv` (Build backend: `hatchling`).
- **Commits**: Conventional Commits format (`feat:`, `fix:`, `chore:`, etc.).
- **Docker**: A multi-stage `python:3.11-slim` container using `stdio` transport, configurable via `docker-compose.yml` for dev profiles.