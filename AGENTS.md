# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-10
**Commit:** b551185
**Branch:** main

## OVERVIEW

MCP server (FastMCP/stdio) bridging Claude Code and Cursor to 3D Slicer's WebServer for AI-assisted medical image analysis. 46 tools + 4 resources. Python 3.10+, Hatchling build, uv package manager.

## STRUCTURE

```
./
├── src/slicer_mcp/           # Package root (see src/slicer_mcp/AGENTS.md)
│   ├── server.py             # Entry point: all 46 @mcp.tool() + 4 @mcp.resource() wrappers
│   ├── core/                 # Transport, resilience, config (see core/AGENTS.md)
│   ├── features/             # Domain logic: tools, diagnostics, spine (see features/AGENTS.md)
│   └── *.py (14 files)       # Backward-compat shims (importlib re-exports)
├── tests/
│   ├── unit/                 # 16 test files, mock-only (no Slicer needed)
│   └── benchmarks/           # Latency tests (requires live Slicer)
├── docs/plans/               # V2 roadmap, audit results
├── .claude/commands/         # 11 spine workflow command docs
└── TODO.md                   # Active task tracker
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add a new tool | `features/*.py` impl + `server.py` wrapper + `tests/unit/` | ~15 lines boilerplate per tool in server.py |
| Add a resource | `core/resources.py` impl + `server.py` wrapper | 4 existing: scene, volumes, status, workflows |
| Change validation limits | `core/constants.py` | Single source of truth for all limits/patterns |
| Change retry/timeout | `core/slicer_client.py` | `@with_retry` decorator; exec_python has none |
| Spine-specific constants | `features/spine/constants.py` | Vertebra maps, segment labels, reference ranges |
| Test a tool | `tests/unit/test_*.py` | Mock `get_client()`, check generated Python code |
| V2 planning | `docs/plans/2026-03-07-v2-roadmap.md` + `TODO.md` | Keep docs lightweight per CLAUDE.md policy |
| Workflow commands | `.claude/commands/*.md` | 11 Claude-only command docs (spine-eval, screw-plan, etc.) |

## DATA FLOW

```
MCP/stdio → server.py @mcp.tool() → features/*.py → slicer_client.exec_python(code_string)
         → HTTP POST /slicer/exec → Slicer executes Python → __execResult = dict → JSON response
         → _parse_json_result() → dict returned through MCP
```

## ANTI-PATTERNS (THIS PROJECT)

| Forbidden | Why |
|-----------|-----|
| `@with_retry` on `exec_python()` | Non-idempotent: Slicer may execute code even if HTTP response lost |
| `requests.Session()` | Slicer WebServer closes connections immediately (connection reset) |
| `print()` in Slicer code strings | Not captured in Slicer 5.10.0; use `__execResult = dict` |
| `json.dumps()` on `__execResult` value | Double-encodes; assign dict directly |
| `len()` on VTK collections | Use `.GetNumberOfItems()` |
| Log to stdout | Reserved for MCP stdio protocol; use `logging` (goes to stderr) |
| Hardcode limits outside `constants.py` | All validation limits, patterns, timeouts centralized there |
| Construct `SlicerClient()` directly | Use `get_client()` singleton |
| Suppress type errors (`as any`, `@ts-ignore`) | N/A for Python but same principle: no `# type: ignore` without comment |

## CONVENTIONS

- **Line length**: 100 (Black + Ruff)
- **Union syntax**: `X | Y` not `Optional[X]` (enforced by ruff UP rules)
- **Imports**: isort via ruff `I` rule
- **Docstrings**: Google-style
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`)
- **Coverage**: 85% minimum, branch coverage enabled
- **CI matrix**: Python 3.10, 3.11, 3.12 (coverage on 3.11 only)
- **Pre-commit**: Black → Ruff --fix → mypy (src/ only)

## COMMANDS

```bash
uv sync                                              # Install deps
uv sync --all-extras                                 # Install with dev + metrics
uv run pytest -v -m "not integration and not benchmark"  # Unit tests only
uv run pytest --cov=slicer_mcp                       # With coverage
uv run black src tests && uv run ruff check src tests    # Format + lint
uv run pre-commit run --all-files                    # All hooks
uv run slicer-mcp                                    # Start MCP server
```

## NOTES

- **14 compatibility shims** at package root re-export from `core/` and `features/` via `importlib.import_module`. Tests still import through shims. New code must use canonical paths.
- **`core/resources.py` imports from `features/base_tools.py`** — minor layer inversion for `_parse_json_result()`.
- **No `__main__.py`** — `python -m slicer_mcp` won't work. Use `uv run slicer-mcp`.
- **asyncio_mode = "auto"** in pytest despite synchronous codebase (MCP framework is async underneath).
- **TotalSegmentator subprocess** uses `start_new_session=True` + `os.killpg()` to prevent hanging. Never switch to `proc.wait()`.
