# Project Patterns Reference

Canonical design principles and conventions for the MCP Slicer Bridge.

## Design Principles

### 1. Singleton HTTP Client

`SlicerClient` uses thread-safe double-checked locking. All HTTP communication goes through this single instance.

```python
_client_instance: SlicerClient | None = None
_client_lock = threading.Lock()

def get_client() -> SlicerClient:
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = SlicerClient()
    return _client_instance
```

**Why:** Prevents connection exhaustion, ensures consistent retry/circuit breaker behavior.

**Note:** Uses per-request connections (not session pooling) because Slicer WebServer closes connections immediately.

### 2. Retry with Exponential Backoff

Failed HTTP requests retry 3 times with delays: 1s → 2s → 4s. Only connection errors trigger retries; timeouts do not.

**Why:** Slicer may be temporarily busy. Exponential backoff prevents thundering herd.

### 3. Circuit Breaker Pattern

After 5 consecutive failures, circuit opens for 30 seconds. Requests fail fast without HTTP calls. After timeout, one test request allowed.

**Why:** Prevents resource exhaustion when Slicer is down. Allows recovery without being hammered.

### 4. Input Validation at Tool Boundary

Validate before sending to SlicerClient:

```python
def measure_volume(node_id: str, segment_name: str | None = None) -> dict:
    validate_node_id(node_id)  # ASCII only, max 256 chars
    if segment_name:
        segment_name = validate_segment_name(segment_name)  # NFKC, max 256 chars
    client = get_client()
    return client.measure_volume(node_id, segment_name)
```

**Why:** Defense-in-depth. Prevent injection even though we trust Slicer.

### 5. stdio Transport

MCP communication uses stdin/stdout. Logs go to stderr as JSON.

**Why:** MCP protocol requirement. Keeps protocol messages separate from diagnostics.

---

## Anti-Patterns to Avoid

### Over-Engineering
- Don't add authentication/authorization - localhost only
- Don't add database persistence - state lives in Slicer's MRML scene
- Don't create abstractions for single-use operations
- Don't add config options that aren't needed yet

### Breaking Simplicity
- Don't replace `requests` with async HTTP unless measured performance need
- Don't add middleware layers between tools and SlicerClient
- Don't create wrapper classes around Slicer responses

### Scope Creep
- Don't add features beyond MCP tool/resource interface
- Don't try to sandbox `execute_python` - Slicer's whole API must be accessible
- Don't add GUI components - this is a CLI bridge

---

## Architectural Invariants

These must always remain true:

1. **Single HTTP Client**: All Slicer communication through `SlicerClient` singleton
2. **Fail-Fast on Circuit Open**: Never queue or retry when circuit is open
3. **No State in Bridge**: All state lives in Slicer's MRML scene
4. **Logs to stderr**: stdout is reserved for MCP protocol only
5. **Validation Before HTTP**: All inputs validated before reaching SlicerClient
6. **Constants Centralized**: All magic numbers in `constants.py`

---

## Code Conventions

### Error Hierarchy

```python
SlicerError (base)
├── SlicerConnectionError  # Retryable network errors
├── SlicerTimeoutError     # Not retried (Slicer is busy)
└── SlicerExecutionError   # Python code failed
```

### Naming

- Tools: verb_noun (e.g., `capture_screenshot`, `measure_volume`)
- Resources: `slicer://` URI scheme
- Constants: UPPER_SNAKE_CASE

### Imports

Standard library first, then third-party, then local. Sorted by `ruff`.

### Constants

All magic numbers in `constants.py`:

```python
SLICER_DEFAULT_URL = "http://localhost:2016"
SLICER_DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 30
MAX_NODE_ID_LENGTH = 256
MAX_SEGMENT_NAME_LENGTH = 256
```

---

## Tool Pattern

Standard implementation:

```python
@mcp.tool()
def tool_name(required_param: str, optional_param: str = "default") -> dict:
    """Brief description for Claude."""
    # 1. Validate inputs
    validate_node_id(required_param)

    # 2. Get client
    client = get_client()

    # 3. Call client method
    result = client.some_method(required_param, optional_param)

    # 4. Transform response
    return {"success": True, "data": result}
```

---

## File Structure

```
src/slicer_mcp/
├── __init__.py      # Package entry, main()
├── server.py        # FastMCP server, tool/resource registration
├── tools.py         # Tool implementations
├── resources.py     # Resource implementations
├── slicer_client.py # HTTP client singleton
├── circuit_breaker.py
├── constants.py     # All configuration values
└── metrics.py       # Optional Prometheus metrics
```
