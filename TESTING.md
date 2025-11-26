# Testing Instructions for MCP Slicer Bridge

## Prerequisites

1. Install `uv` package manager (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   cd mcp-servers/slicer-bridge
   uv sync
   ```

3. Ensure 3D Slicer is running with WebServer extension enabled on localhost:2016

## Running Unit Tests

Unit tests use mocked HTTP responses and don't require a running Slicer instance:

```bash
cd mcp-servers/slicer-bridge
uv run pytest tests/test_slicer_client.py -v
```

Expected output:
```
test_slicer_client.py::TestSlicerClientInit::test_default_initialization PASSED
test_slicer_client.py::TestSlicerClientInit::test_custom_initialization PASSED
test_slicer_client.py::TestHealthCheck::test_health_check_success PASSED
test_slicer_client.py::TestExecPython::test_exec_python_success PASSED
test_slicer_client.py::TestGetScreenshot::test_get_screenshot_default PASSED
...
```

## Running Integration Tests

Integration tests require a running Slicer instance:

```bash
# 1. Start 3D Slicer with WebServer extension
# 2. Run integration tests
cd mcp-servers/slicer-bridge
uv run pytest tests/test_slicer_client.py -v -m integration
```

## Testing MCP Server Manually

### Test 1: Start the server

```bash
cd mcp-servers/slicer-bridge
uv run slicer-mcp
```

The server should start and output JSON-RPC initialization messages.

### Test 2: Check Slicer connection

With Slicer running, the server should be able to connect to localhost:2016.

### Test 3: Verify tools are registered

Check the MCP server output for the tool registration messages:
```
Registered 6 tools: capture_screenshot, list_scene_nodes, execute_python, measure_volume, load_sample_data, set_layout
Registered 3 resources: slicer://scene, slicer://volumes, slicer://status
```

## Testing via Claude Code

1. Ensure `.claude/mcp.json` is configured with slicer-bridge server
2. Restart Claude Code to pick up the new MCP server
3. Try example queries from `examples/simple_query.md`:
   - "Is 3D Slicer connected?"
   - "Load the MRHead sample dataset"
   - "Show me an axial slice of the brain"

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'mcp'"

**Solution**: Install dependencies with `uv sync`

### Problem: "Could not connect to Slicer WebServer"

**Solution**:
1. Ensure 3D Slicer is running
2. In Slicer, go to Edit > Application Settings > Developer
3. Enable developer mode
4. Go to Modules > Developer Tools > WebServer
5. Start the WebServer (it should bind to localhost:2016)

Alternatively, install WebServer extension:
1. View > Extension Manager
2. Search for "WebServer"
3. Install and restart Slicer

### Problem: Tests fail with timeout errors

**Solution**: Increase timeout in test fixtures or check Slicer responsiveness

### Problem: "git commit" fails with author issues

**Solution**: Configure git user:
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Test Coverage

Current test coverage:
- SlicerClient initialization: ✓
- Health check: ✓
- Python execution: ✓
- Screenshot capture (2D): ✓
- Screenshot capture (3D): ✓
- Scene nodes listing: ✓
- Sample data loading: ✓
- Layout setting: ✓
- Connection error handling: ✓
- Timeout handling: ✓

To check coverage:
```bash
uv run pytest tests/ --cov=slicer_mcp --cov-report=html
open htmlcov/index.html
```

## Performance Benchmarks

Expected response times (with Slicer running):
- Health check: <50ms
- List scene nodes: <200ms
- Capture screenshot: <500ms
- Execute Python: <100ms (simple code)
- Load sample data: 2-10s (includes download)
- Set layout: <100ms

Run benchmarks:
```bash
cd mcp-servers/slicer-bridge
uv run python -m pytest tests/test_slicer_client.py -v -m integration --durations=10
```

## Continuous Integration

The project uses GitHub Actions for CI. Tests run automatically on:
- Push to main branch
- Pull requests
- Manual workflow dispatch

See `.github/workflows/test-slicer-mcp.yml` for CI configuration.

## Next Steps

1. Add integration tests for all 6 tools
2. Add integration tests for all 3 resources
3. Add performance benchmarks
4. Add coverage reporting
5. Set up pre-commit hooks
6. Add linting (black, ruff)
7. Add type checking (mypy)
