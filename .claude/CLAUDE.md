# Codebase Standards

Canonical design principles for the MCP Slicer Bridge. This file should rarely change.

## Design Principles

### 1. Singleton HTTP Client
`SlicerClient` uses thread-safe double-checked locking. All HTTP communication goes through this single instance (note: per-request connections, not session pooling, due to Slicer WebServer closing connections immediately).

**Why**: Prevents connection exhaustion, ensures consistent retry/circuit breaker behavior across all requests.

### 2. Retry with Exponential Backoff
Failed HTTP requests retry 3 times with delays: 1s → 2s → 4s. Only connection errors trigger retries; timeouts do not.

**Why**: Slicer may be temporarily busy processing images. Exponential backoff prevents thundering herd.

### 3. Circuit Breaker Pattern
After 5 consecutive failures, circuit opens for 30 seconds. During this time, requests fail fast without attempting HTTP calls. After the timeout, circuit transitions to HALF_OPEN state and allows one test request - if it succeeds, circuit closes; if it fails, circuit reopens.

**Why**: Prevents resource exhaustion when Slicer is down. Allows Slicer to recover without being hammered.

### 4. Input Validation at Tool Boundary
Node IDs: ASCII-only, max 256 chars. Segment names: NFKC Unicode normalization, max 256 chars.

**Why**: Defense-in-depth. Even though we trust Slicer, we validate before sending to prevent injection.

### 5. stdio Transport
MCP communication uses stdin/stdout. Logs go to stderr as JSON.

**Why**: MCP protocol requirement. Keeps protocol messages separate from diagnostic output.

## Anti-Patterns to Avoid

### Over-Engineering
- Don't add authentication/authorization - this is designed for localhost only
- Don't add database persistence - state lives in Slicer's MRML scene
- Don't create abstractions for single-use operations
- Don't add configuration options that aren't needed yet

### Breaking Simplicity
- Don't replace `requests` with async HTTP unless there's a measured performance need
- Don't add middleware layers between tools and SlicerClient
- Don't create wrapper classes around Slicer responses

### Scope Creep
- Don't add features beyond MCP tool/resource interface
- Don't try to sandbox `execute_python` - Slicer's whole API must be accessible
- Don't add GUI components - this is a CLI bridge

## Security Boundaries

### What This Is
- Educational/research tool for **localhost only**
- No encryption, no authentication, no multi-user support
- `execute_python` runs arbitrary code by design

### What This Is NOT
- Clinical/production system
- HIPAA/GDPR compliant
- Suitable for patient data
- Safe for remote access

### Audit Logging
All `execute_python` calls are logged with timestamp, code hash, and result. This is for debugging and research reproducibility, not security enforcement.

## Architectural Invariants

These must always remain true:

1. **Single HTTP Client**: All Slicer communication through `SlicerClient` singleton
2. **Fail-Fast on Circuit Open**: Never queue or retry when circuit is open
3. **No State in Bridge**: All state lives in Slicer's MRML scene
4. **Logs to stderr**: stdout is reserved for MCP protocol only
5. **Validation Before HTTP**: All inputs validated before reaching SlicerClient
6. **Constants Centralized**: All magic numbers and config values in `constants.py`

## Code Conventions

### Imports
Standard library first, then third-party, then local. Sorted by `ruff`.

### Error Handling
- `SlicerConnectionError` - retryable network errors
- `SlicerTimeoutError` - not retried (Slicer is busy)
- `CircuitBreakerOpen` - fail fast, don't attempt connection

### Naming
- Tools: verb_noun (e.g., `capture_screenshot`, `measure_volume`)
- Resources: `slicer://` URI scheme
- Constants: UPPER_SNAKE_CASE
