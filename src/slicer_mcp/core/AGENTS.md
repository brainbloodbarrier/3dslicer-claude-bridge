# core/ — Infrastructure Layer

## OVERVIEW

HTTP transport to Slicer WebServer, circuit breaker resilience, centralized constants, optional Prometheus metrics, and MCP resource implementations.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Change HTTP behavior | `slicer_client.py` | Singleton via `get_client()`, thread-safe |
| Change retry/backoff | `slicer_client.py::with_retry` | 3 attempts, 1s/2s/4s exponential |
| Change circuit breaker thresholds | `constants.py` | `CIRCUIT_BREAKER_FAILURE_THRESHOLD=5`, `RECOVERY_TIMEOUT=30` |
| Add validation constant | `constants.py` | Single source of truth — never hardcode elsewhere |
| Add a resource | `resources.py` + `server.py` | 4 existing: scene, volumes, status, workflows |
| Enable metrics | Set `SLICER_METRICS_ENABLED=true` env var | `metrics.py` uses NullMetric when disabled |

## SLICER CLIENT (`slicer_client.py`, 849 LOC)

**Singleton**: `get_client()` with double-checked locking (`threading.Lock`). Never construct `SlicerClient()` directly.

**HTTP**: Uses `requests.get/post` with fresh connections (no `Session`). Slicer WebServer closes connections immediately — session reuse causes "connection reset by peer".

**Retry**: `@with_retry` decorator on all methods EXCEPT `exec_python()`.
- `exec_python()` is intentionally unretried — non-idempotent (Slicer may execute code even if HTTP response lost)
- Retries: 3 attempts with 1s, 2s, 4s exponential backoff
- Only `SlicerConnectionError` is retryable; `SlicerTimeoutError` is never retried (Slicer may be frozen)

**Error mapping** in `_handle_request_error()`:
- `requests.Timeout` → `SlicerTimeoutError` (not retried, not circuit-tripping)
- `requests.ConnectionError` → `SlicerConnectionError` (retried if decorated, trips circuit)
- `requests.RequestException` → `SlicerConnectionError` (not retried, trips circuit)

## CIRCUIT BREAKER (`circuit_breaker.py`, 244 LOC)

Three-state machine: CLOSED → OPEN → HALF_OPEN → CLOSED.

- **CLOSED**: Normal. Opens after 5 consecutive failures.
- **OPEN**: Fails fast with `CircuitOpenError`. No HTTP calls made.
- **HALF_OPEN**: Lazy transition after 30s timeout. Allows one test request. Success → CLOSED, failure → OPEN.

Transition to HALF_OPEN is lazy — checked on `.state` property read, not via timer. Thread-safe via `threading.Lock`.

Only connection-type exceptions trip the breaker. `ValidationError`, `ValueError`, `SlicerTimeoutError` pass through without affecting circuit state.

Testing: `reset_circuit_breaker()` exposed for `conftest.py` autouse fixture.

## CONSTANTS (`constants.py`, 236 LOC)

All magic values centralized here. Key groups: connection settings, retry config, view names/maps, validation limits (node ID, segment name, code length), DICOM patterns, Slicer version compatibility, circuit breaker thresholds.

Spine-specific constants live in `features/spine/constants.py` instead.

## METRICS (`metrics.py`, 179 LOC)

NullMetric null-object pattern. When `SLICER_METRICS_ENABLED != "true"` or `prometheus_client` not installed, all metric objects are no-op `NullMetric` instances. Code calls metrics unconditionally — no `if METRICS_ENABLED:` guards needed.

`prometheus_client` is an optional `[metrics]` extra in pyproject.toml.

## RESOURCES (`resources.py`, 280 LOC)

4 MCP resources: `slicer://scene`, `slicer://volumes`, `slicer://status`, `slicer://workflows`.

Imports `_parse_json_result` from `features/base_tools.py` — known layer inversion, acceptable for shared JSON parsing utility.
