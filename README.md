# MCP Slicer Bridge

> Model Context Protocol (MCP) server providing Claude Code with programmatic access to 3D Slicer for AI-assisted medical image analysis.

## Overview

The MCP Slicer Bridge enables Claude Code to interact with [3D Slicer](https://www.slicer.org), a powerful open-source platform for medical image analysis, surgical planning, and radiomics research.

**Use Cases:**
- AI-assisted surgical planning with real-time visualization
- Automated radiomics analysis workflows
- Interactive medical image segmentation
- Teaching medical imaging concepts with AI guidance

## Features

### Tools (12)

| Tool | Description |
|------|-------------|
| `capture_screenshot` | Capture PNG from any viewport (axial, sagittal, coronal, 3D, full) |
| `list_scene_nodes` | Inspect MRML scene structure and node metadata |
| `execute_python` | Execute Python code in Slicer's environment |
| `measure_volume` | Calculate segmentation volumes in mm³/mL |
| `list_sample_data` | Discover available sample datasets |
| `load_sample_data` | Load sample datasets (MRHead, CTChest, etc.) |
| `set_layout` | Configure viewer layout (FourUp, OneUp3D, etc.) |
| `import_dicom` | Import DICOM files from folder into database |
| `list_dicom_studies` | List all studies in DICOM database |
| `list_dicom_series` | List series within a DICOM study |
| `load_dicom_series` | Load DICOM series as volume |
| `run_brain_extraction` | Brain extraction/skull stripping (HD-BET or Swiss) |

### Resources (3)

| Resource | Description |
|----------|-------------|
| `slicer://scene` | Current MRML scene structure |
| `slicer://volumes` | Loaded volumes with metadata |
| `slicer://status` | Connection health status |

## Prerequisites

### 1. 3D Slicer

Download [3D Slicer 5.0+](https://download.slicer.org) and enable WebServer extension:
1. **View** → **Extension Manager** → Search "WebServer" → **Install**
2. Restart Slicer
3. **Modules** → **Developer Tools** → **WebServer** → **Start Server**

### 2. Python Environment

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd slicer-bridge
uv sync
```

## Installation

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "slicer-bridge": {
      "command": "uv",
      "args": ["--directory", "/path/to/slicer-bridge", "run", "slicer-mcp"],
      "env": {
        "SLICER_URL": "http://localhost:2016"
      }
    }
  }
}
```

Restart Claude Code to load the configuration.

## Usage Examples

**Scene inspection:**
> "What volumes are loaded in Slicer?"

**Screenshot capture:**
> "Show me an axial view of the current volume"

**Volume measurement:**
> "Calculate the volume of the tumor segmentation"

**Custom analysis:**
> "Apply a Gaussian blur with sigma=2.0 to the current volume"

## Troubleshooting

| Error | Solution |
|-------|----------|
| `SLICER_CONNECTION_ERROR` | Start Slicer, enable WebServer, check port 2016 |
| `SLICER_TIMEOUT` | Restart Slicer (may be frozen) |
| `PYTHON_EXECUTION_ERROR` | Add `import slicer` to code |
| MCP server not starting | Check `mcp.json` syntax, verify path, run `uv sync` |

## Development

```bash
uv run pytest -v                     # Run tests
uv run pytest --cov=slicer_mcp       # Coverage
uv run black src tests               # Format
uv run ruff check src tests          # Lint
```

## Security Notice

This is an **educational/research tool** for localhost use only. Not suitable for clinical data or production environments. See [ref/security.md](ref/security.md) for details.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Quick reference index
- [ref/](ref/) - Detailed reference documentation

## License

[MIT License](LICENSE)

## Acknowledgments

- [3D Slicer](https://www.slicer.org) - Open-source medical imaging platform
- [Anthropic MCP](https://modelcontextprotocol.io) - Model Context Protocol
- [FastMCP](https://gofastmcp.com) - Pythonic MCP server framework
