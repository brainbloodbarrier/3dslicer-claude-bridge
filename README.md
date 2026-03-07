# MCP Slicer Bridge

> Model Context Protocol (MCP) server providing Claude Code and Cursor with
> programmatic access to 3D Slicer for AI-assisted medical image analysis.

## Overview

The MCP Slicer Bridge enables Claude Code and Cursor to interact with
[3D Slicer](https://www.slicer.org), a powerful open-source platform for medical
image analysis, surgical planning, and radiomics research.

It provides an extensive suite of tools out of the box including base scene interaction, Spine Segmentation (via TotalSegmentator), CT/MRI Diagnostic Tools, X-Ray measurements, and Cervical Screw instrumentation planning.

**Use Cases:**

- AI-assisted surgical planning with real-time visualization
- Automated radiomics analysis workflows
- Interactive medical image segmentation
- Teaching medical imaging concepts with AI guidance

## Current Direction

The project is moving toward a v2 workflow model for spine analysis and agentic
MCP usage, but the documentation process is intentionally lightweight.

- The current v2 direction is tracked in
  [`docs/plans/2026-03-07-v2-roadmap.md`](docs/plans/2026-03-07-v2-roadmap.md).
- The active task tracker is the repo-root [`TODO.md`](TODO.md).
- Existing docs and `.claude/commands/*` remain useful, but they should be kept
  aligned with the real MCP surface instead of growing into a separate planning
  system.

## Prerequisites

### 1. 3D Slicer

Download [3D Slicer 5.0+](https://download.slicer.org) and enable WebServer extension:

1. **View** → **Extension Manager** → Search "WebServer" → **Install**
2. Restart Slicer
3. **Modules** → **Developer Tools** → **WebServer** → **Start Server**

### 2. Project Dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/brainbloodbarrier/3dslicer-claude-bridge.git
cd 3dslicer-claude-bridge

# Install dependencies
uv sync
```

## Client Setup

Once Slicer is running and `uv sync` has completed, add this server definition to
your MCP client using the absolute path to this repository:

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

### Choose Your Client

- **Claude Code:** Add the server under `mcpServers` in `~/.claude.json`, then
  restart Claude Code.
- **Cursor:** Add the same server to `~/.cursor/mcp.json`, then restart Cursor.
- **Need detailed instructions?** See
  [`docs/guides/setup-mcp-clients.md`](docs/guides/setup-mcp-clients.md) for config file
  locations, full examples, verification steps, and troubleshooting.

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
| ----- | -------- |
| `SLICER_CONNECTION_ERROR` | Start Slicer, enable WebServer, check port 2016 |
| `SLICER_TIMEOUT` | Restart Slicer (may be frozen) |
| `PYTHON_EXECUTION_ERROR` | Add `import slicer` to code |
| MCP server not starting | Check the client config file, verify the repo path, run `uv sync`, then restart the client |

## License

[MIT License](LICENSE)

## Acknowledgments

- [3D Slicer](https://www.slicer.org) - Open-source medical imaging platform
- [Anthropic MCP](https://modelcontextprotocol.io) - Model Context Protocol
- [FastMCP](https://gofastmcp.com) - Pythonic MCP server framework
