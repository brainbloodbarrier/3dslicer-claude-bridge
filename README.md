# slicer-mcp

[![CI](https://github.com/brainbloodbarrier/3dslicer-claude-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/brainbloodbarrier/3dslicer-claude-bridge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io)

An MCP (Model Context Protocol) server that bridges Claude Code and Cursor to
[3D Slicer](https://www.slicer.org), enabling AI-assisted medical image
analysis through natural language.

## Features

- **MCP tools** for diagnostics (X-ray, CT, MRI), spine surgery planning,
  DICOM management, volume rendering, registration, and clinical workflows.
- **MCP resources** for real-time scene inspection (`slicer://scene`,
  `slicer://volumes`, `slicer://status`, `slicer://workflows`).
- **Spine-specific tooling** -- CCJ craniometry, sagittal/coronal balance,
  Cobb angles, Pfirrmann grading, Modic classification, SINS scoring,
  cervical screw planning, vertebral artery segmentation, bone quality.
- **Resilience** -- circuit breaker, retry with exponential backoff,
  configurable timeouts, graceful degradation.
- **Security** -- input validation (MRML node IDs, DICOM UIDs, Unicode
  normalization), audit logging, path traversal prevention.

## Prerequisites

- Python 3.10+
- [3D Slicer](https://www.slicer.org) with the WebServer extension enabled
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone https://github.com/brainbloodbarrier/3dslicer-claude-bridge.git
cd 3dslicer-claude-bridge
uv sync
```

## Quick Start

1. In 3D Slicer, start **Modules > Developer Tools > WebServer**.
2. Run the MCP server:

```bash
uv run slicer-mcp
```

3. Configure your MCP client:

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

Detailed setup guide: [docs/guides/setup-mcp-clients.md](docs/guides/setup-mcp-clients.md).

## Structure

```text
src/slicer_mcp/
  core/       shared infrastructure (HTTP client, circuit breaker, constants)
  features/   domain tools (diagnostics, spine, rendering, registration, workflows)
  server.py   MCP tool and resource registration
tests/        unit tests and benchmarks
docs/         guides and plans
```

## Development

```bash
uv run pytest -v -m "not integration and not benchmark"
uv run pytest -v -m integration
uv run ruff check src tests
uv run black --check src tests
uv run mypy src/
uv run pre-commit run --all-files
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICER_URL` | `http://localhost:2016` | 3D Slicer WebServer endpoint |
| `SLICER_TIMEOUT` | `30` | HTTP request timeout (seconds) |

## Related Documentation

- [CLAUDE.md](CLAUDE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/guides/setup-mcp-clients.md](docs/guides/setup-mcp-clients.md)

## License

[MIT](LICENSE)
