# slicer-mcp

[![CI](https://github.com/brainbloodbarrier/3dslicer-claude-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/brainbloodbarrier/3dslicer-claude-bridge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io)

An MCP (Model Context Protocol) server that bridges Claude to
[3D Slicer](https://www.slicer.org), enabling AI-assisted medical image
analysis through natural language.

## Features

- **46 MCP tools** spanning DICOM management, diagnostics (X-ray, CT, MRI),
  spine surgery planning, volume rendering, and multi-step clinical workflows.
- **4 MCP resources** for real-time scene inspection (`slicer://scene`,
  `slicer://volumes`, `slicer://status`, `slicer://workflows`).
- **Spine-specific tooling** -- CCJ angles, sagittal/coronal balance,
  Cobb angle, Pfirrmann grading, Modic classification, SINS scoring,
  cervical screw planning, vertebral artery segmentation, and bone quality
  analysis.
- **Resilience** -- circuit breaker, retry with exponential backoff,
  configurable timeouts, and graceful degradation.
- **Security** -- input validation (MRML node IDs, DICOM UIDs, Unicode
  normalization), audit logging with code hashing, path traversal prevention.
- **Observability** -- JSON-structured logging, optional Prometheus metrics.

## Prerequisites

- Python 3.10+
- [3D Slicer](https://www.slicer.org) with the
  [WebServer extension](https://github.com/Slicer/SlicerWeb) enabled
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone https://github.com/brainbloodbarrier/3dslicer-claude-bridge.git
cd 3dslicer-claude-bridge
uv sync
```

## Quick Start

1. Open 3D Slicer and start the WebServer from
   **Modules > Developer Tools > WebServer** (default port `2016`).

2. Run the MCP server:

   ```bash
   uv run slicer-mcp
   ```

3. The server communicates over stdio using the MCP protocol. Point your MCP
   client (Claude Code, Cursor, etc.) at it -- see
   [MCP Client Setup](docs/guides/setup-mcp-clients.md) for details.

## MCP Client Configuration

Add the following to your MCP client config (e.g., `~/.claude.json` for
Claude Code or `~/.cursor/mcp.json` for Cursor):

```json
{
  "mcpServers": {
    "slicer-bridge": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/3dslicer-claude-bridge",
        "run",
        "slicer-mcp"
      ],
      "env": {
        "SLICER_URL": "http://localhost:2016"
      }
    }
  }
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICER_URL` | `http://localhost:2016` | 3D Slicer WebServer URL |
| `SLICER_TIMEOUT` | `30` | Request timeout in seconds |

## Architecture

```
src/slicer_mcp/
  core/           # Infrastructure: client, circuit breaker, constants, metrics, resources
  features/       # Tool implementations
    diagnostics/  # X-ray, CT, MRI diagnostic protocols
    spine/        # Spine surgery planning and instrumentation
    workflows/    # Multi-step clinical workflows (e.g., Modic evaluation)
  server.py       # MCP server entry point and tool registration
```

The server registers all tools via `@mcp.tool()` decorators and delegates
to feature modules. Each tool call is wrapped with centralized error handling
that maps domain exceptions (`ValidationError`, `CircuitOpenError`,
`SlicerTimeoutError`, `SlicerConnectionError`) to structured error responses.

Communication with 3D Slicer happens over HTTP to the WebServer extension.
Python code is executed in Slicer's interpreter via the `/exec` endpoint,
with input validation and audit logging for security.

## Docker

```bash
docker compose up
```

The container expects 3D Slicer to be running on the host. On Docker Desktop,
`SLICER_URL` defaults to `http://host.docker.internal:2016`.

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest -v

# Lint and format
uv run ruff check src tests
uv run ruff format --check src tests

# Type check
uv run mypy src/

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## License

[MIT](LICENSE)
