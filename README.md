# MCP Slicer Bridge

> Model Context Protocol (MCP) server providing Claude Code with programmatic access to 3D Slicer for AI-assisted medical image analysis.

## Overview

The MCP Slicer Bridge enables Claude Code to interact with [3D Slicer](https://www.slicer.org), a powerful open-source platform for medical image analysis, surgical planning, and radiomics research. Through this bridge, Claude can capture screenshots, inspect scene data, execute Python scripts, and perform quantitative measurements directly in Slicer.

**Use Cases**:
- AI-assisted surgical planning with real-time visualization
- Automated radiomics analysis workflows
- Interactive medical image segmentation
- Teaching medical imaging concepts with AI guidance
- Reproducible research with documented Slicer operations

## Features

### Tools (7)

1. **capture_screenshot** - Capture PNG screenshots from any Slicer viewport
   - Support for axial, sagittal, coronal, 3D, and full window views
   - Optional slice position and camera axis configuration
   - Returns base64-encoded PNG with metadata

2. **list_scene_nodes** - Inspect MRML scene structure
   - List all volumes, segmentations, models, and other nodes
   - Retrieve node metadata (dimensions, spacing, properties)
   - Understand scene state before operations

3. **execute_python** - Execute Python code in Slicer's environment
   - Full access to Slicer, VTK, and Qt APIs
   - Create/modify nodes, perform calculations
   - All executions logged to audit log for security monitoring

4. **measure_volume** - Calculate segmentation volumes
   - Measure total or per-segment volumes
   - Returns volumes in mm³ and mL
   - Input validation prevents code injection

5. **list_sample_data** - Discover available sample datasets
   - Dynamically queries Slicer for registered samples
   - Falls back to known list when disconnected
   - Returns dataset names, categories, and descriptions

6. **load_sample_data** - Load sample datasets for testing
   - Load MRHead, CTChest, CTACardio, and other sample volumes
   - Great for demonstrations and testing
   - Returns loaded node information

7. **set_layout** - Configure viewer layout
   - Switch between FourUp, OneUp3D, Conventional, etc.
   - Control GUI mode (full or viewers-only)
   - Optimize display for specific workflows

### Resources (3)

1. **slicer://scene** - Current MRML scene structure as JSON
2. **slicer://volumes** - Loaded imaging volumes with metadata
3. **slicer://status** - Slicer health status and connection info

## Prerequisites

### 1. 3D Slicer Installation

Download and install [3D Slicer 5.0+](https://download.slicer.org) for your platform.

**Enable WebServer Extension**:
1. Open 3D Slicer
2. Go to **View** → **Extension Manager**
3. Search for "WebServer"
4. Click **Install**
5. Restart Slicer when prompted

**Configure WebServer**:
1. Go to **Modules** → **Developer Tools** → **WebServer**
2. Set **Port**: `2016`
3. Click **Start Server**
4. Verify status shows "Running on http://localhost:2016"

### 2. Python Environment

This project requires Python 3.10+ and [uv](https://github.com/astral-sh/uv) package manager.

**Install uv**:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Install Dependencies**:
```bash
cd mcp-servers/slicer-bridge
uv sync
```

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd escritor/mcp-servers/slicer-bridge
```

### 2. Install Package

```bash
uv sync
```

### 3. Configure Claude Code

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "slicer-bridge": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/slicer-bridge",
        "run",
        "slicer-mcp"
      ],
      "env": {
        "SLICER_URL": "http://localhost:2016",
        "SLICER_TIMEOUT": "30",
        "SLICER_AUDIT_LOG": "/var/log/slicer-mcp-audit.log"
      }
    }
  }
}
```

**Important**: Replace `/absolute/path/to/slicer-bridge` with the actual absolute path to your repository.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICER_URL` | `http://localhost:2016` | URL of Slicer WebServer |
| `SLICER_TIMEOUT` | `30` | HTTP request timeout in seconds |
| `SLICER_AUDIT_LOG` | *(none)* | Path to audit log file (optional) |

### 4. Restart Claude Code

Restart Claude Code to load the MCP server configuration.

### 5. Verify Installation

In Claude Code, ask:
> "Can you check if Slicer is connected?"

Claude should use the `slicer://status` resource to verify the connection.

## Usage Examples

### Example 1: Basic Scene Inspection

**User Prompt**:
> "What volumes are loaded in Slicer?"

**Claude's Workflow**:
1. Calls `list_scene_nodes()` tool
2. Filters for volume nodes
3. Reports volume names, types, and dimensions

### Example 2: Screenshot Capture

**User Prompt**:
> "Show me an axial view of the current volume"

**Claude's Workflow**:
1. Calls `capture_screenshot(view_type="Red")`
2. Receives base64 PNG
3. Displays image inline in chat

### Example 3: Volume Measurement

**User Prompt**:
> "Calculate the volume of the tumor segmentation"

**Claude's Workflow**:
1. Calls `list_scene_nodes()` to find segmentation node ID
2. Calls `measure_volume(node_id="vtkMRMLSegmentationNode1")`
3. Reports volume in mL with segment breakdown

### Example 4: Automated Analysis

**User Prompt**:
> "Create a segmentation for the cerebellum and measure its volume"

**Claude's Workflow**:
1. Calls `execute_python()` to create segmentation node
2. Guides user to segment cerebellum in Slicer UI
3. Calls `measure_volume()` to calculate result
4. Calls `capture_screenshot()` to document result

### Example 5: Custom Python Script

**User Prompt**:
> "Apply a Gaussian blur with sigma=2.0 to the current volume"

**Claude's Workflow**:
```python
code = """
import slicer, SimpleITK as sitk, sitkUtils

# Get input volume
inputNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
inputImage = sitkUtils.PullVolumeFromSlicer(inputNode)

# Apply Gaussian blur
blurFilter = sitk.SmoothingRecursiveGaussianImageFilter()
blurFilter.SetSigma(2.0)
outputImage = blurFilter.Execute(inputImage)

# Create output volume
outputNode = sitkUtils.PushVolumeToSlicer(outputImage, name='Blurred')
outputNode.GetID()
"""

result = execute_python(code)
# Returns new volume node ID
```

## Troubleshooting

### Connection Errors

**Error**: `SLICER_CONNECTION_ERROR: Could not connect to Slicer WebServer`

**Solutions**:
1. Verify Slicer is running
2. Check WebServer module is started (Modules → WebServer)
3. Confirm port is 2016 (default)
4. Try accessing http://localhost:2016 in browser

### Timeout Errors

**Error**: `SLICER_TIMEOUT: Request timeout after 30 seconds`

**Solutions**:
1. Restart Slicer (may be frozen)
2. Simplify Python code (if using `execute_python`)
3. Check system resources (Slicer may be out of memory)

### Screenshot Failures

**Error**: `SCREENSHOT_FAILED: Could not capture screenshot`

**Solutions**:
1. Verify view name is correct (Red, Yellow, Green, ThreeD)
2. Ensure view is visible in Slicer UI
3. Try resetting layouts (View → Layout)

### Python Execution Errors

**Error**: `PYTHON_EXECUTION_ERROR: NameError: name 'slicer' is not defined`

**Solutions**:
1. Import required modules in code: `import slicer`
2. Check Slicer Python API documentation
3. Test code in Slicer's Python console first

### MCP Server Not Starting

**Symptom**: Claude Code doesn't recognize Slicer tools

**Solutions**:
1. Check `~/.claude/mcp.json` syntax is valid JSON
2. Verify absolute path to `slicer-bridge` directory
3. Run `uv sync` to ensure dependencies installed
4. Check Claude Code MCP logs for errors
5. Restart Claude Code after config changes

## Development

### Running Tests

```bash
# Unit tests only (fast, no Slicer required)
uv run pytest -v

# Integration tests (requires Slicer running)
uv run pytest -v --integration

# With coverage
uv run pytest --cov=slicer_mcp --cov-report=html
```

### Code Formatting

```bash
# Format code
uv run black src tests

# Lint code
uv run ruff check src tests
```

### Manual Testing

Test the server manually via stdio:

```bash
cd mcp-servers/slicer-bridge
uv run slicer-mcp
```

Then send MCP commands via stdin:
```json
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
```

## Security Notice

**IMPORTANT**: This implementation is designed for:
- **Educational use**: Learning medical image analysis with AI assistance
- **Personal research**: Individual research projects
- **Local environments**: Server and Slicer run on same machine

**NOT suitable for**:
- Clinical use with patient data (lacks HIPAA/GDPR compliance)
- Multi-user environments (no authentication)
- Remote access (no encryption or access controls)

**Security Features**:
- **Input validation**: Node IDs and segment names are validated to prevent code injection
- **Audit logging**: All Python code executions are logged with timestamps, code hashes, and results
- **Configurable audit file**: Set `SLICER_AUDIT_LOG` for persistent audit trail
- **Retry with backoff**: Network errors are handled gracefully with exponential backoff

**Security Limitations**:
- `execute_python` tool executes arbitrary code in Slicer (no sandboxing)
- No authentication or authorization mechanisms
- Medical imaging data transmitted over unencrypted localhost HTTP

**Recommendations**: Use only with de-identified research data in controlled environments.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions and component documentation.

## Specification

See [SPECIFICATION.md](SPECIFICATION.md) for complete API reference and protocol details.

## License

[Add license information]

## Contributing

[Add contribution guidelines]

## Acknowledgments

- [3D Slicer](https://www.slicer.org) - Open-source medical imaging platform
- [Anthropic MCP](https://modelcontextprotocol.io) - Model Context Protocol
- [FastMCP](https://github.com/jlowin/fastmcp) - Pythonic MCP server framework

## Support

For issues and questions:
- GitHub Issues: [Link to issues]
- Documentation: [SPECIFICATION.md](SPECIFICATION.md)
- 3D Slicer Forum: https://discourse.slicer.org
