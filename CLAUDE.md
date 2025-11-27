# CLAUDE.md - MCP Slicer Bridge

## Project Overview

MCP server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis. Enables Claude to capture screenshots, inspect scene data, execute Python scripts, and perform quantitative measurements in 3D Slicer.

## Quick Reference

```bash
# Install dependencies
uv sync

# Run the MCP server (for testing)
uv run slicer-mcp

# Run unit tests
uv run pytest -v

# Run integration tests (requires Slicer running on localhost:2016)
uv run pytest -v --integration

# Format code
uv run black src tests

# Lint code
uv run ruff check src tests
```

## Architecture

```
Claude Code <--MCP/stdio--> MCP Server (Python) <--HTTP REST--> 3D Slicer WebServer
```

- **Transport**: stdio (JSON-RPC 2.0)
- **Framework**: FastMCP for MCP server implementation
- **Slicer API**: HTTP REST via WebServer extension (default: localhost:2016)

## Project Structure

```
src/slicer_mcp/
├── __init__.py        # Package entry point, exports main()
├── server.py          # MCP server setup, tool/resource registration
├── slicer_client.py   # HTTP client for 3D Slicer WebServer
├── tools.py           # Tool implementations (screenshot, execute, measure, etc.)
└── resources.py       # Resource implementations (scene, volumes, status)

tests/
├── __init__.py
└── test_slicer_client.py  # Unit tests with mocked HTTP responses
```

## MCP Tools (6)

| Tool | Description |
|------|-------------|
| `capture_screenshot` | Capture PNG from Slicer viewport (axial, sagittal, coronal, 3d, full) |
| `list_scene_nodes` | List all MRML scene nodes with metadata |
| `execute_python` | Execute Python code in Slicer's environment |
| `measure_volume` | Calculate segmentation volumes in mm³/mL |
| `load_sample_data` | Load sample datasets (MRHead, CTChest, etc.) |
| `set_layout` | Set viewer layout (FourUp, OneUp3D, etc.) |

## MCP Resources (3)

| Resource | Description |
|----------|-------------|
| `slicer://scene` | Current MRML scene structure as JSON |
| `slicer://volumes` | Loaded imaging volumes with metadata |
| `slicer://status` | Slicer health status and connection info |

## Key Implementation Details

### SlicerClient (slicer_client.py)
- Uses `requests` library with session reuse for connection pooling
- Default timeout: 30 seconds
- Base URL configurable via `SLICER_URL` env var (default: http://localhost:2016)

### Slicer WebServer Endpoints Used
- `POST /slicer/exec` - Execute Python code
- `GET /slicer/slice?view=<view>&scrollTo=<pos>` - Capture slice screenshot
- `GET /slicer/threeD?lookFromAxis=<axis>` - Capture 3D screenshot
- `GET /slicer/mrml/names` - List node names
- `GET /slicer/mrml/ids` - List node IDs
- `GET /slicer/sampledata?name=<name>` - Load sample data
- `GET /slicer/gui?contents=<mode>&viewersLayout=<layout>` - Set GUI layout

### Error Handling
- `SlicerConnectionError` - Raised when Slicer is unreachable
- Errors are mapped to structured MCP errors with actionable codes

## Testing

Unit tests use mocked HTTP responses and don't require a running Slicer instance.

```bash
# Unit tests only (fast)
uv run pytest -v

# With coverage
uv run pytest --cov=slicer_mcp --cov-report=html
```

Integration tests require 3D Slicer running with WebServer enabled on localhost:2016:

```bash
uv run pytest -v --integration
```

## Configuration

### Claude Code MCP Configuration (~/.claude/mcp.json)

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

## Development Notes

- Logging goes to stderr (stdout reserved for MCP protocol)
- Log format: JSON structured logging
- Python 3.10+ required
- Package manager: uv (not pip)

## Security Considerations

- `execute_python` executes arbitrary code in Slicer (no sandboxing)
- Designed for local use only (localhost)
- Not suitable for clinical/production use without additional security controls
