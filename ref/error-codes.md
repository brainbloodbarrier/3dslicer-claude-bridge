# Error Codes Reference

Structured error format and resolution guide.

## Error Response Format

```json
{
  "error": {
    "code": "SLICER_CONNECTION_ERROR",
    "message": "Could not connect to Slicer WebServer at localhost:2016",
    "details": {
      "url": "http://localhost:2016",
      "timeout": 30,
      "suggestion": "Ensure Slicer is running with WebServer extension enabled"
    }
  }
}
```

## Error Codes

| Code | Description | Retried? | Resolution |
|------|-------------|----------|------------|
| `SLICER_CONNECTION_ERROR` | Cannot connect to WebServer | Yes (3x) | Start Slicer, enable WebServer extension |
| `SLICER_TIMEOUT` | Request timeout (>30s) | No | Restart Slicer (may be frozen) |
| `SLICER_UNAVAILABLE` | Circuit breaker open | No | Wait 30s for recovery timeout |
| `INVALID_NODE_ID` | Node ID not found | No | Verify node exists with `list_scene_nodes` |
| `INVALID_PARAMETER` | Parameter validation failed | No | Check parameter types and allowed values |
| `PYTHON_EXECUTION_ERROR` | Python code failed | No | Review code syntax and Slicer API usage |
| `SCREENSHOT_FAILED` | Screenshot capture failed | No | Verify view exists and is visible |
| `VOLUME_CALCULATION_ERROR` | Volume measurement failed | No | Check segmentation node has valid geometry |

## Retry Strategy

Connection errors retry with exponential backoff:

| Attempt | Delay |
|---------|-------|
| 1 | 0s (immediate) |
| 2 | 1s |
| 3 | 2s |
| 4 | 4s (then fail) |

**What gets retried:**
- Connection refused
- Connection reset
- HTTP 5xx errors

**What does NOT get retried:**
- Timeouts (Slicer likely frozen)
- HTTP 4xx errors (client must fix)
- Validation errors
- Python execution errors

## Exception Hierarchy

```python
SlicerError (base)
├── SlicerConnectionError  # Retryable network errors
├── SlicerTimeoutError     # Not retried
└── SlicerExecutionError   # Python code failed

CircuitBreakerOpen         # Fail fast, don't attempt connection
```

## Common Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `SLICER_CONNECTION_ERROR` repeatedly | Slicer not running | Start Slicer, go to WebServer module, click Start |
| `SLICER_TIMEOUT` after long operation | Slicer processing large data | Wait or restart Slicer |
| `INVALID_NODE_ID` | Stale node reference | Re-query scene with `list_scene_nodes` |
| `PYTHON_EXECUTION_ERROR: NameError` | Missing import | Add `import slicer` to code |
| `SCREENSHOT_FAILED` | View not visible | Reset layout with `set_layout("FourUp")` |
