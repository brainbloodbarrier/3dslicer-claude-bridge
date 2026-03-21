# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 48 MCP tools (diagnostics, spine planning, rendering, workflows)
- 4 MCP resources (`slicer://scene`, `slicer://volumes`, `slicer://status`, `slicer://workflows`)
- X-ray diagnostic protocols (sagittal/coronal balance, Cobb angle, fracture detection)
- CT diagnostic protocols (fractures, osteoporosis, metastases, SINS, listhesis, canal)
- MRI diagnostic protocols (Modic, Pfirrmann, cord compression, metastases)
- Spine surgery planning (cervical screws, CCJ angles, alignment, bone quality)
- Workflow orchestration tools:
  - `workflow_modic_eval` — MRI-based Modic/Pfirrmann/cord assessment
  - `workflow_ccj_protocol` — craniocervical junction craniometry + VA + bone quality
  - `workflow_onco_spine` — oncologic spine assessment (SINS, metastases, stability)
- Volume rendering and 3D model export tools
- Registration and landmark tools
- CI/CD pipeline with GitHub Actions
- CLAUDE.md project guidance file
- .gitignore for Python, medical imaging, and IDE artifacts
- CI coverage enforcement (--cov-fail-under=85)

### Changed
- README.md restored to English
- `__all__` exports added to all canonical modules in `core/` and `features/`
- `slicer://workflows` resource updated: ccj_protocol and onco_spine now "available"
- Deferred imports in base_tools.py hoisted to top-level
- Duplicated process cleanup code in spine/tools.py extracted to helper
- CONTRIBUTING.md updated to reference v2 workflow surface doc

### Deprecated
- Root-level shim modules (e.g., `slicer_mcp.tools`) now emit DeprecationWarning;
  use `slicer_mcp.features.*` or `slicer_mcp.core.*` instead

### Fixed
- `assert` in rendering.py replaced with proper `ValidationError` raise
- Removed unused `asyncio_mode` config from pyproject.toml (no async tests)
- Plan docs updated with completion status

## [0.9.0] - 2024-12-15

### Added
- Initial release of MCP Slicer Bridge
- **7 MCP Tools**:
  - `capture_screenshot` - Capture 2D slice or 3D view screenshots
  - `list_scene_nodes` - List all nodes in the MRML scene
  - `execute_python` - Execute Python code in Slicer's environment
  - `measure_volume` - Measure volume of segmentation segments
  - `list_sample_data` - List available sample datasets
  - `load_sample_data` - Load sample datasets into Slicer
  - `set_layout` - Configure Slicer's view layout
- **3 MCP Resources**:
  - `slicer://scene` - Current scene structure and metadata
  - `slicer://volumes` - Loaded volume information
  - `slicer://status` - Connection health and server status
- **Security Features**:
  - Input validation for MRML node IDs (ASCII-only)
  - Unicode segment name support with NFKC normalization
  - Shell metacharacter blocking
  - Audit logging with code hashing
  - Defense-in-depth with JSON escaping
- **Resilience Patterns**:
  - Circuit breaker (5 failures -> open, 30s recovery)
  - Retry with exponential backoff (1s, 2s, 4s)
  - Graceful degradation for sample data
  - Configurable timeouts
- **Observability**:
  - Optional Prometheus metrics
  - JSON-structured logging to stderr
  - Request tracking and timing

### Security
- Input validation prevents code injection attacks
- Unicode normalization blocks homoglyph attacks
- Audit log path validation prevents writes to system directories
- Designed for localhost-only use (no authentication by design)

[Unreleased]: https://github.com/brainbloodbarrier/3dslicer-claude-bridge/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/brainbloodbarrier/3dslicer-claude-bridge/releases/tag/v0.9.0
