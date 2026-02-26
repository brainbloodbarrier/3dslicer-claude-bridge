# Troubleshooting Reference

Common issues and MCP configuration examples.

## Connection Issues

### Cannot connect to Slicer

**Symptoms:**
- `error_type: "connection"`
- `SLICER_CONNECTION_ERROR`

**Checks:**
1. Slicer is running
2. WebServer extension is started (Modules → WebServer → Start)
3. `SLICER_URL` points to correct host/port (default: `localhost:2016`)
4. If Docker: use `host.docker.internal` instead of `localhost`

### Timeouts

**Symptoms:**
- `error_type: "timeout"`
- `SlicerTimeoutError` in logs

**Checks:**
1. Slicer may be frozen → restart Slicer
2. Long operation in progress → increase `SLICER_TIMEOUT`
3. Timeouts are NOT retried (by design)

### Circuit Breaker Open

**Symptoms:**
- `error_type: "circuit_open"`
- Immediate failures without HTTP attempt

**Checks:**
1. Wait 30s for recovery timeout
2. Fix underlying connectivity issue
3. Circuit opens after 5 consecutive failures

## MCP Configuration

### Standard (uv)

```json
{
  "mcpServers": {
    "slicer-bridge": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/NC_Slicer-Claude-Bridge",
        "run",
        "slicer-mcp"
      ],
      "env": {
        "SLICER_URL": "http://localhost:2016",
        "SLICER_TIMEOUT": "30"
      }
    }
  }
}
```

### Docker

```json
{
  "mcpServers": {
    "slicer-bridge": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-e", "SLICER_URL=http://host.docker.internal:2016",
        "-e", "SLICER_TIMEOUT=30",
        "slicer-mcp:latest"
      ]
    }
  }
}
```

**Notes:**
- `-i` required for stdio MCP servers
- `host.docker.internal` reaches host from container

## Debugging Steps

### 1. Capture Error Details

```bash
# Test runs
uv run pytest -v --tb=long tests/test_tools.py -x

# MCP server stderr
uv run slicer-mcp 2>&1 | tee debug.log
```

### 2. Check Environment

```bash
echo $SLICER_URL      # Should be http://localhost:2016
echo $SLICER_TIMEOUT  # Should be 30
```

### 3. Manual MCP Test

```bash
uv run slicer-mcp
```

Then send on stdin:
```json
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
```

### 4. Localize Failure

| Error Type | Check |
|------------|-------|
| Connection/timeout | `slicer_client.py` retry logic |
| Validation failure | `tools.py` validation functions |
| JSON parsing | `tools._parse_json_result()` |
| Tool exception | `server.py` error wrapper |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: mcp` | Missing deps | `uv sync` |
| `NameError: slicer` | Missing import | Add `import slicer` to code |
| `Invalid node ID` | Stale reference | Re-query with `list_scene_nodes` |
| `Metrics enabled but no client` | Missing extra | `uv sync --all-extras` |
