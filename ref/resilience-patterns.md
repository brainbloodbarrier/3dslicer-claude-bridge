# Resilience Patterns Reference

Error handling and recovery patterns used in this MCP server.

## Circuit Breaker Pattern

Prevents cascading failures when Slicer is unresponsive.

### States

```
CLOSED ──(5 failures)──▶ OPEN ──(30s timeout)──▶ HALF_OPEN
   ▲                       │                         │
   │                       │ (fail fast)             │
   │                       ▼                         │
   └──────(success)────────────────────(test request)┘
```

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation, requests pass through |
| **OPEN** | All requests fail immediately (no HTTP calls) |
| **HALF_OPEN** | One test request allowed; success closes, failure reopens |

### Configuration

```python
# From constants.py
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5   # Failures before opening
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 30   # Seconds before half-open
```

### Usage

```python
from slicer_mcp.circuit_breaker import CircuitBreaker, CircuitOpenError, with_circuit_breaker

breaker = CircuitBreaker(name="slicer", failure_threshold=5, recovery_timeout=30)

# Decorator approach (recommended)
@with_circuit_breaker(breaker, failure_exceptions=(ConnectionError,))
def call_slicer():
    return requests.get(url)

# Only ConnectionError (or specified exceptions) trip the circuit
# Other exceptions pass through without affecting circuit state
```

### Exception Filtering

The circuit breaker only trips on **specified exception types** (defaults to `ConnectionError`).
This prevents validation errors or timeouts from incorrectly opening the circuit.

```python
@with_circuit_breaker(breaker, failure_exceptions=(ConnectionError, IOError))
def risky_operation():
    # ConnectionError/IOError -> trips circuit
    # ValueError/TimeoutError -> passes through, circuit unaffected
    pass
```

## Retry with Exponential Backoff

Recovers from transient connection failures.

### Configuration

```python
# From constants.py
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds
```

### Backoff Schedule

| Attempt | Delay |
|---------|-------|
| 1 | 0s (immediate) |
| 2 | 1s |
| 3 | 2s |
| 4 | 4s (max) |

### What Gets Retried

| Error Type | Retried? | Reason |
|------------|----------|--------|
| Connection refused | Yes | Slicer may be starting |
| Connection reset | Yes | Transient network issue |
| Timeout | **No** | Slicer likely frozen |
| HTTP 4xx | No | Client error, fix parameters |
| HTTP 5xx | Yes | Server may recover |

### Implementation

```python
from slicer_mcp.slicer_client import with_retry

@with_retry(max_retries=3, backoff_base=1.0)
def exec_python(self, code: str) -> dict:
    response = requests.post(f"{self.base_url}/slicer/exec", data=code)
    return {"success": True, "result": response.text}
```

## Error Categories

| Exception | Code | Action |
|-----------|------|--------|
| `SlicerConnectionError` | `SLICER_CONNECTION_ERROR` | Retry, check Slicer running |
| `SlicerTimeoutError` | `SLICER_TIMEOUT` | No retry, Slicer may be frozen |
| `CircuitBreakerOpen` | `SLICER_UNAVAILABLE` | Fail fast, wait for recovery |
| `ValidationError` | `INVALID_PARAMETER` | Fix input, no retry |

## Links

- Circuit breaker: `src/slicer_mcp/circuit_breaker.py`
- Retry decorator: `src/slicer_mcp/slicer_client.py`
- Constants: `src/slicer_mcp/constants.py`
