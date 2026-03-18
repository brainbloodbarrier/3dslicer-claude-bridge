# features/ — Domain Logic Layer

## OVERVIEW

All 46 MCP tool implementations. Each builds Python code strings, sends them to Slicer via `exec_python()`, and parses the JSON result. Organized by medical imaging domain.

## STRUCTURE

```
features/
├── base_tools.py         # 1546 LOC — Shared: validation, audit, screenshots, DICOM, exec
├── registration.py       # 731 LOC  — Landmark/volume registration, transforms
├── rendering.py          # 825 LOC  — Volume rendering, model export, 3D capture
├── diagnostics/          # 4821 LOC — Modality-specific diagnostic tools
│   ├── ct.py             # Fractures, osteoporosis, metastasis, SINS, canal stenosis
│   ├── mri.py            # Modic, Pfirrmann disc grading, cord compression, MRI metastasis
│   └── xray.py           # Sagittal/coronal balance, Cobb angle, dynamic listhesis
├── spine/                # 4707 LOC — Spine domain (see spine/AGENTS.md)
└── workflows/
    └── modic.py          # 169 LOC  — Multi-step: segment → modic → pfirrmann → cord
```

## CODE-AS-STRING PATTERN (architectural constraint)

Every tool is a Python-code-string factory. Features do NOT call Slicer APIs through typed abstractions.

```python
def some_tool(node_id: str) -> dict:
    # 1. Validate
    validate_mrml_node_id(node_id)
    # 2. Escape (defense-in-depth, even after validation)
    safe_id = json.dumps(node_id)
    # 3. Build Python code string
    code = f"""
    import slicer
    node = slicer.mrmlScene.GetNodeByID({safe_id})
    __execResult = {{"success": True, "data": ...}}
    """
    # 4. Execute
    client = get_client()
    result = client.exec_python(code)
    # 5. Parse
    return _parse_json_result(result.get("result", ""), "context")
```

**Critical rules**: Always `json.dumps()` user values before interpolation. Always assign `__execResult = dict` (not `json.dumps(dict)`). Never `print()`.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add a base tool (scene/DICOM/screenshot) | `base_tools.py` | Also update `server.py` wrapper |
| Add input validation | `base_tools.py` | `validate_mrml_node_id`, `validate_segment_name`, `validate_folder_path`, `validate_dicom_uid` |
| Add CT/MRI/X-ray diagnostic | `diagnostics/{ct,mri,xray}.py` | Follow existing tool pattern in same file |
| Add registration/rendering tool | `registration.py` / `rendering.py` | |
| Add workflow (multi-step) | `workflows/` | See `modic.py` as template |
| Understand `_parse_json_result` | `../core/parsing.py` | Returns `SlicerConnectionError` on empty/null/malformed |

## INPUT VALIDATION (base_tools.py)

| Validator | Pattern | Raises |
|-----------|---------|--------|
| `validate_mrml_node_id` | `^[a-zA-Z][a-zA-Z0-9_]*$`, max 256 | `ValidationError(field="node_id")` |
| `validate_segment_name` | NFKC normalize, strip invisible chars, `^[\w\s\-]+$`, max 256 | `ValidationError(field="segment_name")` |
| `validate_folder_path` | Resolve symlinks, reject `..`, must exist + be dir | `ValidationError(field="folder_path")` |
| `validate_dicom_uid` | `^[0-9]+(\.[0-9]+)*$`, max 64 | `ValidationError(field=field_name)` |

Invisible character stripping covers: U+200B, U+200C, U+200D, U+FEFF, U+00AD, U+2060–U+2064.

## DIAGNOSTICS SUBDIRECTORY

Three modality files, each self-contained with 4-6 tools. All follow the same code-as-string pattern. Each file imports `get_client`, validators from `base_tools.py`, shared JSON parsing (via the `base_tools.py` re-export), and `_build_totalseg_subprocess_block` from `spine/tools.py` for segmentation-dependent diagnostics.

## WORKFLOWS SUBDIRECTORY

Currently one workflow (`modic.py`, 169 LOC). Orchestrates multiple tools: segment spine → classify Modic → grade Pfirrmann → detect cord compression. Template for future multi-step workflows.

## CROSS-MODULE DEPENDENCIES

```
base_tools.py    ← imported by ALL feature modules (validation, exec helpers, `_parse_json_result` re-export)
spine/tools.py   ← imported by diagnostics/ct.py, diagnostics/mri.py (_build_totalseg_subprocess_block)
diagnostics/mri.py + spine/tools.py ← imported by workflows/modic.py
```
