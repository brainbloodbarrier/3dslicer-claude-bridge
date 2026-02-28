# MCP Slicer Bridge

> Model Context Protocol (MCP) server providing Claude Code with programmatic access to 3D Slicer for AI-assisted medical image analysis.

## Overview

The MCP Slicer Bridge enables Claude Code to interact with [3D Slicer](https://www.slicer.org), a powerful open-source platform for medical image analysis, surgical planning, and radiomics research. 

It provides an extensive suite of tools out of the box including base scene interaction, Spine Segmentation (via TotalSegmentator), CT/MRI Diagnostic Tools, X-Ray measurements, and Cervical Screw instrumentation planning.

**Use Cases:**
- AI-assisted surgical planning with real-time visualization
- Automated radiomics analysis workflows
- Interactive medical image segmentation
- Teaching medical imaging concepts with AI guidance

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

**Spine Analysis:**
> "Run the TotalSegmentator pipeline on the loaded CT spine and compute the sagittal balance."

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

## License

[MIT License](LICENSE)

## Acknowledgments

- [3D Slicer](https://www.slicer.org) - Open-source medical imaging platform
- [Anthropic MCP](https://modelcontextprotocol.io) - Model Context Protocol
- [FastMCP](https://gofastmcp.com) - Pythonic MCP server framework
