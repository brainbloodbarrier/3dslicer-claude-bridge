# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-15

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
  - Circuit breaker (5 failures → open, 30s recovery)
  - Retry with exponential backoff (1s, 2s, 4s)
  - Graceful degradation for sample data
  - Configurable timeouts
- **Observability**:
  - Optional Prometheus metrics
  - JSON-structured logging to stderr
  - Request tracking and timing
- **Documentation**:
  - Comprehensive README with usage examples
  - Architecture documentation (ARCHITECTURE.md)
  - API specification (SPECIFICATION.md)
  - Security documentation (SECURITY.md)
  - Testing guide (TESTING.md)
  - Performance benchmarks (BENCHMARKS.md)

### Security
- Input validation prevents code injection attacks
- Unicode normalization blocks homoglyph attacks
- Audit log path validation prevents writes to system directories
- Designed for localhost-only use (no authentication by design)

[1.0.0]: https://github.com/brainbloodbarrier/3dslicer-claude-bridge/releases/tag/v1.0.0
