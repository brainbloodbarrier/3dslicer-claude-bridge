# FastMCP Framework Reference

Quick reference for the FastMCP framework used in this MCP server.

## Server Initialization

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slicer-bridge")
```

## Tool Definition

Tools are functions Claude can invoke. Use the `@mcp.tool()` decorator:

```python
@mcp.tool()
def capture_screenshot(view_type: str, scroll_position: float = 0.5) -> dict:
    """Capture screenshot from Slicer viewport.

    Args:
        view_type: Viewport type (axial, sagittal, coronal, 3d, full)
        scroll_position: Slice position 0.0-1.0 (default: 0.5)

    Returns:
        Dict with image_base64, view_type, and metadata
    """
    # Implementation - schema auto-generated from type hints
    ...
```

**Key points:**
- Type hints generate JSON schema automatically
- Docstrings become tool descriptions
- Return type should be JSON-serializable (dict, list, str, etc.)
- Tools in this project are **sync** - FastMCP handles async wrapping

## Resource Definition

Resources are read-only data endpoints:

```python
@mcp.resource("slicer://status")
def get_status() -> dict:
    """Slicer connection status."""
    return {"connected": True, "version": "5.6.2"}
```

**URI scheme:** This project uses `slicer://` prefix for all resources.

## Running the Server

**stdio transport** (used by Claude Code):
```python
if __name__ == "__main__":
    mcp.run()  # Defaults to stdio
```

**Entry point** (pyproject.toml):
```toml
[project.scripts]
slicer-mcp = "slicer_mcp:main"
```

## Error Handling

Raise exceptions with clear messages - FastMCP converts to JSON-RPC errors:

```python
@mcp.tool()
def measure_volume(node_id: str) -> dict:
    if not node_id:
        raise ValueError("node_id is required")
    # ...
```

## Pydantic Integration

For complex parameters, use Pydantic models:

```python
from pydantic import BaseModel

class ScreenshotParams(BaseModel):
    view_type: str
    scroll_position: float = 0.5

@mcp.tool()
def capture_screenshot(params: ScreenshotParams) -> dict:
    ...
```

## Links

- FastMCP docs: https://gofastmcp.com
- MCP specification: https://modelcontextprotocol.io
- Project server: `src/slicer_mcp/server.py`
