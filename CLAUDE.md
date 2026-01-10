# CLAUDE.md

MCP server bridging Claude Code to 3D Slicer for AI-assisted medical image analysis. Uses FastMCP with stdio transport.

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest -v                     # Run tests (no Slicer required)
uv run pytest -v -m integration      # Integration tests (requires Slicer)
uv run pytest --cov=slicer_mcp       # Test coverage (uses .coveragerc)
uv run pytest --cov-report=html      # Coverage with HTML report
uv run black src tests               # Format code
uv run ruff check src tests          # Lint code
uv run slicer-mcp                    # Run the MCP server
```

## Architecture

```
Claude Code ──(MCP/stdio)──▶ server.py ──(HTTP)──▶ Slicer WebServer (localhost:2016)
                               │
                               ├─ tools.py      (12 tools)
                               ├─ resources.py  (3 resources)
                               └─ slicer_client.py (singleton + retry + circuit breaker)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICER_URL` | `http://localhost:2016` | Slicer WebServer URL |
| `SLICER_TIMEOUT` | `30` | HTTP timeout in seconds |
| `SLICER_AUDIT_LOG` | *(none)* | Audit log path for execute_python |

## Quick Reference

| Topic | File |
|-------|------|
| Tool API (12 tools) | [ref/api-tools.md](ref/api-tools.md) |
| Resource API (3 resources) | [ref/api-resources.md](ref/api-resources.md) |
| Error codes | [ref/error-codes.md](ref/error-codes.md) |
| Slicer HTTP endpoints | [ref/slicer-webserver.md](ref/slicer-webserver.md) |
| Design patterns | [ref/project-patterns.md](ref/project-patterns.md) |
| Circuit breaker & retry | [ref/resilience-patterns.md](ref/resilience-patterns.md) |
| FastMCP framework | [ref/fastmcp.md](ref/fastmcp.md) |
| Security model | [ref/security.md](ref/security.md) |
| Performance benchmarks | [ref/benchmarks.md](ref/benchmarks.md) |
| Troubleshooting | [ref/troubleshooting.md](ref/troubleshooting.md) |

## Code Style

- **Formatter**: Black (100 char line length)
- **Linter**: Ruff (rules: E, F, W, I, N, UP)
- **Python**: 3.10+
- **Type hints**: Used throughout
