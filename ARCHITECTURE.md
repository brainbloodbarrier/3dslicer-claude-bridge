# MCP Slicer Bridge - Architecture Documentation

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Claude Code Desktop                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Claude Code AI Engine                          │  │
│  │  • Analyzes medical imaging requests                              │  │
│  │  • Generates Slicer commands                                      │  │
│  │  • Interprets results and screenshots                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                │                                         │
│                                │ MCP Protocol (stdio)                    │
│                                ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    MCP Client (Built-in)                          │  │
│  │  • Manages MCP server lifecycle                                   │  │
│  │  • Handles stdio communication                                    │  │
│  │  • Provides tool/resource discovery                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ stdio (JSON-RPC 2.0)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MCP Slicer Bridge Server (Python)                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      FastMCP Framework                            │  │
│  │  • stdio transport handler                                        │  │
│  │  • JSON-RPC message routing                                       │  │
│  │  • Schema validation (Pydantic)                                   │  │
│  │  • Tool/resource registration                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Tool Implementations                         │  │
│  │  • capture_screenshot()     • execute_python()                    │  │
│  │  • list_scene_nodes()       • measure_volume()                    │  │
│  │  • load_sample_data()       • set_layout()                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Resource Implementations                       │  │
│  │  • slicer://scene           • slicer://status                     │  │
│  │  • slicer://volumes                                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Slicer HTTP Client                           │  │
│  │  • Session management (connection pooling)                        │  │
│  │  • Request/response handling                                      │  │
│  │  • Error mapping and retry logic                                  │  │
│  │  • Base64 encoding/decoding                                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ HTTP REST API
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        3D Slicer Application                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      WebServer Extension                          │  │
│  │  • HTTP server (localhost:2016)                                   │  │
│  │  • REST API endpoints                                             │  │
│  │  • Python execution interface                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Slicer Core                                  │  │
│  │  • MRML scene management                                          │  │
│  │  • Volume rendering engine                                        │  │
│  │  • Segmentation tools                                             │  │
│  │  • Python interpreter                                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Communication Flow

```
┌─────────────┐      MCP/stdio      ┌─────────────┐      HTTP       ┌─────────────┐
│ Claude Code │◄───────────────────►│ MCP Server  │◄───────────────►│ 3D Slicer   │
│  (Client)   │                     │  (Python)   │                 │ WebServer   │
└─────────────┘                     └─────────────┘                 └─────────────┘
       │                                   │                               │
       │ 1. Request screenshot             │                               │
       │───────────────────────────────────►                               │
       │                                   │ 2. HTTP POST /exec            │
       │                                   │──────────────────────────────►
       │                                   │                               │
       │                                   │ 3. Execute Python, capture    │
       │                                   │    screenshot, encode base64  │
       │                                   │◄──────────────────────────────│
       │                                   │                               │
       │ 4. Return screenshot + metadata   │                               │
       │◄───────────────────────────────────                               │
       │                                   │                               │
```

---

## Design Decisions

### 1. Why stdio Transport?

**Decision**: Use stdio (standard input/output) as the MCP transport mechanism.

**Rationale**:
- **Native Claude Code Support**: Claude Code's MCP client natively supports stdio transport, requiring no additional network configuration
- **Process Isolation**: Each MCP server runs in its own process, providing clean isolation and lifecycle management
- **Simplicity**: No need for port management, firewall rules, or network authentication
- **Security**: Communication never leaves the local machine, reducing attack surface
- **Debugging**: Easy to test via command line and monitor with stderr logging

**Alternatives Considered**:
- **HTTP/SSE**: Would require port management and CORS configuration
- **WebSocket**: Added complexity for bidirectional communication not needed for request/response pattern

**Trade-offs**:
- **Limited Concurrency**: stdio is synchronous, only one request at a time
- **No Remote Access**: Cannot connect from remote machines (acceptable for MVP)

---

### 2. Why FastMCP Framework?

**Decision**: Build on the FastMCP framework for MCP server implementation.

**Rationale**:
- **Pythonic API**: Decorator-based tool/resource registration feels natural in Python
- **Automatic Schema Generation**: Pydantic models auto-generate JSON schemas for parameters
- **Built-in Validation**: Request/response validation with clear error messages
- **stdio Support**: First-class support for stdio transport
- **Maintained**: Active development by Anthropic ecosystem

**Example**:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slicer-bridge")

@mcp.tool()
async def capture_screenshot(view_type: str, slice_offset: float = 0) -> dict:
    """Capture screenshot from Slicer viewport."""
    # Implementation auto-gets schema from type hints
    ...
```

**Alternatives Considered**:
- **Raw MCP SDK**: More control but requires manual schema writing and validation
- **Custom Framework**: Unnecessary reinvention for MVP

**Trade-offs**:
- **Framework Dependency**: Tied to FastMCP's development and breaking changes
- **Learning Curve**: Requires understanding FastMCP conventions

---

### 3. HTTP Client Design

**Decision**: Implement a dedicated `SlicerClient` class as a **singleton** with per-request connections and configurable timeout.

**Architecture**:
```python
# Singleton pattern for client reuse
_client_instance = None

def get_client() -> SlicerClient:
    """Get the singleton SlicerClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = SlicerClient()
    return _client_instance

class SlicerClient:
    def __init__(self, base_url: str = None, timeout: int = None):
        # Read from environment variables with defaults
        self.base_url = base_url or os.environ.get('SLICER_URL', 'http://localhost:2016')
        self.timeout = timeout or int(os.environ.get('SLICER_TIMEOUT', '30'))
        # Note: Per-request connections used (not session pooling)
        # Slicer WebServer closes connections immediately, causing
        # "Connection reset by peer" errors with session reuse

    @with_retry(max_retries=3, backoff_base=1.0)
    def exec_python(self, code: str) -> dict:
        """Execute Python code in Slicer via POST /slicer/exec."""
        response = requests.post(
            f"{self.base_url}/slicer/exec",
            data=code,
            timeout=self.timeout
        )
        return {"success": True, "result": response.text}
```

**Why NOT Session Pooling**:
Slicer's WebServer extension closes HTTP connections immediately after each response.
Using `requests.Session()` for connection pooling causes "Connection reset by peer" errors.
Per-request connections are more reliable, though slightly less efficient.

**Rationale**:
- **Environment Configuration**: `SLICER_URL` and `SLICER_TIMEOUT` environment variables
- **Singleton Pattern**: Single client instance reused across all tool/resource calls
- **Retry Logic**: Exponential backoff (1s, 2s, 4s) for connection errors
- **Timeout Separation**: Timeouts not retried (Slicer may be frozen)
- **Input Validation**: Security validation for code injection prevention
- **Encapsulation**: Isolates HTTP communication details from MCP tool logic
- **Error Mapping**: Translates HTTP errors to domain-specific error codes

**Design Patterns**:
- **Singleton**: `get_client()` returns single instance per process
- **Decorator**: `@with_retry` for retry logic
- **Adapter**: Adapts HTTP API to clean Python interface

**Trade-offs**:
- **Blocking I/O**: requests library is synchronous (FastMCP handles async wrapping)
- **No Connection Pooling**: Per-request connections due to Slicer WebServer behavior
- **Retry Overhead**: Exponential backoff adds latency for transient failures

---

### 4. Error Propagation Strategy

**Decision**: Map Slicer errors to structured MCP errors with actionable error codes.

**Error Flow**:
```
Slicer Error (HTTP 500)
    ↓
SlicerClient catches exception
    ↓
Map to domain error code (e.g., PYTHON_EXECUTION_ERROR)
    ↓
Raise MCP error with structured details
    ↓
FastMCP serializes to JSON-RPC error
    ↓
Claude Code receives error message
```

**Error Structure**:
```python
class SlicerError(Exception):
    def __init__(self, code: str, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}

    def to_mcp_error(self):
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }
```

**Error Categories**:
1. **Connection Errors**: Slicer not reachable (retry possible)
2. **Validation Errors**: Invalid parameters (client must fix)
3. **Execution Errors**: Python code failed (code must be corrected)
4. **Resource Errors**: Node not found, volume calculation failed (check inputs)

**Rationale**:
- **Debuggability**: Error codes help users diagnose issues quickly
- **Actionability**: Suggestions in details guide resolution
- **Consistency**: All errors follow same structure

---

### 5. Data Format Decisions

**Decision**: Use Base64 for images, JSON for structured data.

**Image Handling** (Screenshots):
- **Format**: PNG (lossless, widely supported)
- **Encoding**: Base64 string (JSON-safe)
- **Size**: Typical 800x600 screenshot ~500KB base64 (~375KB PNG)

**Rationale**:
- **JSON Compatibility**: Base64 embeds binary data in JSON responses
- **Lossless**: PNG preserves medical image quality
- **Browser Compatible**: Claude Code can render base64 PNGs directly

**Trade-offs**:
- **Size Overhead**: Base64 adds ~33% size overhead vs raw bytes
- **Encoding Cost**: CPU time to encode/decode (negligible for screenshots)

**Structured Data** (Scene nodes, volumes):
- **Format**: JSON objects with typed fields
- **Validation**: Pydantic models ensure type safety

**Example**:
```python
from pydantic import BaseModel

class VolumeInfo(BaseModel):
    id: str
    name: str
    dimensions: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
```

**Rationale**:
- **Type Safety**: Catch errors at development time
- **Documentation**: Models serve as API documentation
- **Validation**: Automatic validation of Slicer data

---

### 6. Testing Strategy

**Decision**: Implement unit tests with mocked Slicer, integration tests with live Slicer.

**Test Pyramid**:
```
        ┌─────────────────┐
        │  Integration    │  ← 5 tests: End-to-end with live Slicer
        │     Tests       │
        ├─────────────────┤
        │   Unit Tests    │  ← 20 tests: Mocked Slicer responses
        │  (SlicerClient, │
        │   MCP Tools)    │
        └─────────────────┘
```

**Unit Tests** (Fast, isolated):
```python
import pytest
from unittest.mock import Mock, patch

def test_capture_screenshot():
    with patch('slicer_mcp.client.SlicerClient') as mock_client:
        mock_client.get_screenshot.return_value = b'fake_png_data'

        result = capture_screenshot(view_type="Red")

        assert result["view_type"] == "Red"
        assert "image_base64" in result
```

**Integration Tests** (Slow, requires Slicer):
```python
@pytest.mark.integration
def test_real_screenshot():
    # Requires: Slicer running with WebServer on localhost:2016
    client = SlicerClient()
    result = client.get_screenshot("Red")

    assert len(result) > 1000  # PNG should be >1KB
    assert result.startswith(b'\x89PNG')  # PNG magic bytes
```

**Test Fixtures**:
- Mock Slicer responses for common operations
- Sample medical images (synthetic or public datasets)
- Pre-configured scene states for testing

**CI/CD**:
- Unit tests run on every commit (fast, no dependencies)
- Integration tests run nightly or on-demand (require Slicer)

---

### 7. Logging Strategy

**Decision**: Log to stderr with structured JSON format, avoiding stdout pollution.

**Rationale**:
- **stdio Transport**: stdout is reserved for MCP JSON-RPC messages
- **Structured Logs**: JSON logs enable programmatic parsing
- **Debug-Friendly**: stderr appears in Claude Code's MCP server logs

**Log Levels**:
- **ERROR**: Request failures, connection errors
- **WARNING**: Timeouts, retries
- **INFO**: Request/response summaries
- **DEBUG**: Full request/response bodies (verbose)

**Example**:
```python
import logging
import sys

# Configure stderr logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    stream=sys.stderr
)

logger = logging.getLogger("slicer-mcp")

logger.info("Screenshot captured", extra={
    "view_type": "Red",
    "size_kb": 450,
    "duration_ms": 234
})
```

**Trade-offs**:
- **Performance**: Structured logging has minimal overhead
- **Readability**: JSON logs less human-readable than plain text (use jq for viewing)

---

## Component Details

### SlicerClient Class

**Responsibilities**:
- Manage HTTP session with Slicer WebServer
- Execute Python code in Slicer
- Retrieve screenshots, scene data, volumes
- Handle errors and timeouts

**HTTP Endpoints Used**:
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/slicer/exec` | Execute Python code |
| GET | `/slicer/slice?view=<view>&scrollTo=<pos>` | Capture slice screenshot |
| GET | `/slicer/threeD?lookFromAxis=<axis>` | Capture 3D screenshot |
| GET | `/slicer/screenshot` | Full window screenshot |
| GET | `/slicer/mrml/names` | List node names |
| GET | `/slicer/mrml/ids` | List node IDs |
| GET | `/slicer/mrml/properties?id=<id>` | Get node properties |
| GET | `/slicer/sampledata?name=<name>` | Load sample data |
| GET | `/slicer/gui?contents=<mode>&viewersLayout=<layout>` | Set GUI layout |

**Public Interface**:
```python
class SlicerClient:
    def exec_python(self, code: str) -> dict
    def get_screenshot(self, view: str = "Red", scroll_to: float = None) -> bytes
    def get_3d_screenshot(self, look_from_axis: str = None) -> bytes
    def get_scene_nodes(self) -> list[dict]
    def get_node_properties(self, node_id: str) -> dict
    def load_sample_data(self, name: str) -> dict
    def set_layout(self, layout: str, gui_mode: str = "full") -> dict
    def health_check(self) -> dict
```

### MCP Tool Handlers

**Responsibilities**:
- Validate parameters (via Pydantic)
- Call SlicerClient methods
- Transform responses to MCP format
- Handle errors gracefully

**Pattern**:
```python
@mcp.tool()
async def tool_name(param1: Type1, param2: Type2 = default) -> ReturnType:
    """Tool description for Claude Code."""
    try:
        # Validate business logic
        # Call SlicerClient
        # Transform response
        return result
    except SlicerError as e:
        # Convert to MCP error
        raise MCPError(code=e.code, message=e.message, data=e.details)
```

### MCP Resource Handlers

**Responsibilities**:
- Provide read-only views of Slicer state
- Cache data when appropriate (future optimization)
- Return fresh data on each request (MVP)

**Pattern**:
```python
@mcp.resource("slicer://resource-name")
async def get_resource() -> dict:
    """Resource description."""
    client = SlicerClient()
    data = client.get_resource_data()
    return format_resource(data)
```

---

## Security Architecture

### Threat Model (MVP Scope)

**Assumptions**:
- Attacker has no network access (localhost only)
- User runs trusted code (no malicious prompts)
- Environment is controlled (personal machine)

**Out of Scope** (for MVP):
- Network attacks (no remote access)
- Multi-user scenarios (single user)
- Clinical data protection (de-identified research data only)

**In Scope**:
- Accidental data corruption (provide undo guidance)
- Resource exhaustion (implement timeouts)
- Error handling (prevent information leakage)

### Security Controls (MVP)

1. **Network Isolation**: Slicer WebServer binds to localhost only
2. **Timeouts**: 30-second timeout prevents hanging requests
3. **Error Sanitization**: Don't expose file paths in error messages
4. **Logging**: Log operations for audit trail

### Production Roadmap

For clinical/production use:
- [ ] Authentication (API keys, OAuth)
- [ ] Authorization (role-based access control)
- [ ] Code execution sandboxing (whitelist allowed operations)
- [ ] Data encryption (TLS for localhost)
- [ ] Audit logging (tamper-proof logs)
- [ ] HIPAA compliance (BAA, PHI handling)

---

## Performance Optimization

### Bottlenecks Identified

1. **Screenshot Encoding**: Base64 encoding adds CPU overhead
   - **Mitigation**: Use PIL's efficient PNG encoder
   - **Future**: Support direct binary transfer (requires MCP binary support)

2. **Large Scenes**: Querying 1000+ nodes can take seconds
   - **Mitigation**: Implement pagination in future versions
   - **Current**: Acceptable for typical scenes (10-100 nodes)

3. **Volume Calculations**: Large segmentations (>500MB) take 1-5 seconds
   - **Mitigation**: Show progress indicator (future)
   - **Current**: Document expected latency in tool description

### Optimization Strategy

**Current** (MVP):
- Keep it simple, measure later
- Optimize for correctness and reliability
- Document expected performance characteristics

**Future** (if needed):
- Implement caching for scene data (invalidate on modifications)
- Use HTTP/2 for parallel requests (requires Slicer WebServer update)
- Stream large responses (screenshots, volume data)

---

## Deployment Architecture

### Development Environment

```
~/.claude/
├── mcp.json                      ← MCP server configuration
└── logs/
    └── slicer-bridge.log         ← MCP server logs (if configured)

~/Documents/escritor/
└── mcp-servers/
    └── slicer-bridge/
        ├── src/slicer_mcp/       ← Source code
        ├── tests/                ← Test suite
        └── pyproject.toml        ← Dependencies
```

### Claude Code Configuration

```json
{
  "mcpServers": {
    "slicer-bridge": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/fax/Documents/escritor/mcp-servers/slicer-bridge",
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

### Runtime Flow

1. User starts Claude Code
2. Claude Code reads `mcp.json`
3. Spawns `uv run slicer-mcp` subprocess
4. MCP server initializes, connects to Slicer
5. Tools/resources become available in Claude Code
6. User requests medical imaging analysis
7. Claude Code calls MCP tools
8. MCP server executes Slicer operations
9. Results returned to Claude Code
10. Claude Code presents analysis to user

---

## Future Architecture Considerations

### Scalability

**Multi-Instance Support**:
- Support multiple Slicer instances (e.g., different ports)
- Configure via environment variables: `SLICER_URL`, `SLICER_PORT`

**Async I/O**:
- Replace `requests` with `httpx` for async HTTP
- Enable concurrent tool calls (if MCP supports it)

### Extensibility

**Plugin Architecture**:
- Allow custom tools via plugin system
- Load domain-specific extensions (e.g., neuroimaging, radiomics)

**Event Streaming**:
- Stream progress updates for long operations
- Real-time scene change notifications

### Integration

**External Systems**:
- PACS integration (DICOM query/retrieve)
- Workflow engines (Nextflow, Snakemake)
- Data repositories (XNAT, TCIA)

---

## Version History

- **1.0.0** (2025-11-26): Initial MVP architecture
  - stdio transport with FastMCP
  - SlicerClient with session management
  - 6 tools: capture_screenshot, list_scene_nodes, execute_python, measure_volume, load_sample_data, set_layout
  - 3 resources: slicer://scene, slicer://volumes, slicer://status
  - Complete Slicer WebServer endpoint mapping
  - Unit + integration test strategy
  - Development deployment model
