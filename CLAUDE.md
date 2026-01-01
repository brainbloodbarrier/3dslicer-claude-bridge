# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP (Model Context Protocol) server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis. Uses FastMCP framework with stdio transport.

## Commands

```bash
# Install dependencies
uv sync

# Run tests (no Slicer required)
uv run pytest -v

# Run integration tests (requires Slicer WebServer on localhost:2016)
uv run pytest -v -m integration

# Run with coverage
uv run pytest --cov=slicer_mcp --cov-report=html

# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Run the MCP server
uv run slicer-mcp
```

## Architecture

```
Claude Code ──(MCP/stdio)──▶ server.py ──(HTTP)──▶ Slicer WebServer (localhost:2016)
                               │
                               ├─ tools.py      (7 tools: capture_screenshot, execute_python, etc.)
                               ├─ resources.py  (3 resources: scene, volumes, status)
                               └─ slicer_client.py (HTTP client with retry + circuit breaker)
```

### Key Components

| File | Purpose |
|------|---------|
| `src/slicer_mcp/server.py` | FastMCP server entry point, registers tools/resources |
| `src/slicer_mcp/slicer_client.py` | Singleton HTTP client with retry (3x backoff) and circuit breaker |
| `src/slicer_mcp/tools.py` | 7 MCP tools (capture_screenshot, execute_python, measure_volume, etc.) |
| `src/slicer_mcp/resources.py` | 3 MCP resources (slicer://scene, slicer://volumes, slicer://status) |
| `src/slicer_mcp/circuit_breaker.py` | Circuit breaker pattern (CLOSED→OPEN→HALF_OPEN) |
| `src/slicer_mcp/constants.py` | Centralized configuration and validation limits |

### Data Flow

1. Claude Code sends MCP request via stdio (JSON-RPC 2.0)
2. FastMCP routes to appropriate tool/resource handler
3. Handler validates input, calls SlicerClient
4. SlicerClient makes HTTP request to Slicer WebServer with retry logic
5. Response flows back through the same path

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICER_URL` | `http://localhost:2016` | Slicer WebServer URL |
| `SLICER_TIMEOUT` | `30` | HTTP timeout in seconds |
| `SLICER_AUDIT_LOG` | *(none)* | Path to audit log for execute_python |
| `SLICER_METRICS_ENABLED` | `false` | Enable Prometheus metrics |

## Test Markers

- `@pytest.mark.integration` - Requires running Slicer instance
- `@pytest.mark.benchmark` - Performance benchmarks

## Code Style

- **Formatter**: Black (100 char line length)
- **Linter**: Ruff (rules: E, F, W, I, N, UP)
- **Python**: 3.10+
- **Type hints**: Used throughout

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed design decisions
- [SPECIFICATION.md](SPECIFICATION.md) - Complete API reference
- [.claude/CLAUDE.md](.claude/CLAUDE.md) - Canonical codebase standards
