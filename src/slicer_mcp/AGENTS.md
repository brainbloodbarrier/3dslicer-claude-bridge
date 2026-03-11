# slicer_mcp — Package Architecture

## OVERVIEW

Package root with three layers: `server.py` (registration), `core/` (infrastructure), `features/` (domain logic). 14 shim files at root provide backward-compatible imports.

## STRUCTURE

```
slicer_mcp/
├── __init__.py           # Exports: main, 4 exception classes, __version__
├── __main__.py           # python -m slicer_mcp entry point
├── server.py             # 1515 LOC — ALL 46 @mcp.tool() + 4 @mcp.resource() wrappers
├── core/                 # Infrastructure: HTTP transport, circuit breaker, constants, metrics
├── features/             # Domain logic: tools, diagnostics, spine, rendering, registration
└── *.py (14 shims)       # importlib re-exports to canonical core/ and features/ paths
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add a tool | `features/*.py` + `server.py` | Implement in features, wrap in server.py with try/except |
| Change entry point | `server.py::main()` | Calls `mcp.run(transport="stdio")` |
| See public API | `__init__.py` | 5 symbols in `__all__` |
| Understand shim pattern | Any root `*.py` except `server.py`/`__init__.py`/`__main__.py` | All identical: `importlib.import_module` + `globals().update()` |

## TOOL REGISTRATION PATTERN

Every tool in `server.py` follows this exact structure (~15 lines each):

```python
@mcp.tool()
def tool_name(args) -> dict:
    """Docstring (FastMCP extracts this for tool schema)."""
    try:
        return feature_module.implementation(args)
    except Exception as e:
        return _handle_tool_error(e, "tool_name")
```

No auto-discovery. Adding a tool requires manually adding a wrapper here.

## ERROR HANDLING ARCHITECTURE

Three layers, outermost to innermost:

1. **server.py `_handle_tool_error()`** — catch-all, maps exception types to standardized `error_type` field: `validation`, `circuit_open`, `timeout`, `connection`, `unexpected`
2. **features/*.py** — domain catches + re-raises `SlicerConnectionError`; raises `ValidationError` before any network call
3. **core/slicer_client.py `_handle_request_error()`** — transport-level, maps `requests` exceptions to typed errors, records circuit breaker failures

Feature code must never swallow exceptions silently — let them propagate to layer 1.

## SHIM LAYER (14 files)

All shims are identical:
```python
from importlib import import_module as _import_module
_module = _import_module("slicer_mcp.<canonical_location>")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})
```

| Shim | Canonical Target |
|------|-----------------|
| `tools.py` | `features.base_tools` |
| `slicer_client.py` | `core.slicer_client` |
| `constants.py` | `core.constants` |
| `circuit_breaker.py` | `core.circuit_breaker` |
| `resources.py` | `core.resources` |
| `metrics.py` | `core.metrics` |
| `spine_tools.py` | `features.spine.tools` |
| `spine_constants.py` | `features.spine.constants` |
| `instrumentation_tools.py` | `features.spine.instrumentation` |
| `rendering_tools.py` | `features.rendering` |
| `registration_tools.py` | `features.registration` |
| `diagnostic_tools_{ct,mri,xray}.py` | `features.diagnostics.{ct,mri,xray}` |

Tests now import from canonical paths. Shims remain for backward compatibility. New code must use canonical paths.

## DEPENDENCY DIRECTION

```
server.py → features/* → core/*
                ↓
         features depend on each other only downward:
           workflows/modic → diagnostics/mri + spine/tools
           diagnostics/*   → spine/tools (_build_totalseg_subprocess_block)
           ALL features    → features/base_tools (validation, JSON parse)

Shared JSON parsing lives in `core/parsing.py`; `features/base_tools.py` re-exports `_parse_json_result` for feature modules.
```

No circular imports exist.
