# Project Status

Single tracking file for slicer-mcp development. Updated continuously.
Replaces individual plan files (archived in `docs/plans/archive/`).

## Active Decisions

- **Architecture**: Three layers (server.py → features/ → core/). No rewrites.
- **Workflow tiers**: Tier 1 = MCP tools (server-side). Tier 2 = Claude commands (LLM-guided). Tier 3 = merge/deprecate.
- **Tier 1** (implemented): `workflow_onco_spine`, `workflow_ccj_protocol`, `workflow_modic_eval`
- **Tier 2** (implemented): spine-eval, fracture-eval, screw-plan, full-spine-workup
- **Tier 3** (done): construct-compare → merged into screw-plan; instability-protocol → deprecated
- **Discoverability**: `slicer://workflows` resource lists workflow tools for Cursor/MCP clients
- **Command docs**: `.claude/commands/*.md` — 9 workflow command files committed to repo

## Pending

No pending items at this time. All planned work has been completed.

## Completed Work

### Command Docs (2026-03-21)

Created all `.claude/commands/*.md` from scratch with correct tool signatures:
- **Tier 1 refs**: modic-eval, ccj-protocol, onco-spine (point to MCP workflow tools)
- **Tier 2 guides**: spine-eval, fracture-eval, screw-plan, full-spine-workup
- **Tier 3**: construct-compare (merged → screw-plan), instability-protocol (deprecated)
- All 6 audit signature mismatches resolved (correct params from source code)

### Code Review Fixes (2026-02 to 2026-03)

| ID | Summary | Category |
|----|---------|----------|
| C1 | Fixed failing exec_python retry test | Test |
| C2 | Achieved 93%+ test coverage (target 85%) | Test |
| I2 | Added look_from_axis validation | Security |
| I3 | Symlink resolution in audit log path | Security |
| I4 | Symlink resolution in folder path validation | Security |
| I5 | Fixed all ruff lint violations + deprecated config | Style |
| I6 | Defensive assertion in with_retry | Reliability |
| I7 | Fixed _get_valid_datasets return type | Type safety |
| M2 | Removed duplicate circuit breaker fixture | Test |
| M4 | Fixed node type extraction (trailing digits only) | Bug |
| M5 | Migrated Optional to Python 3.10+ union syntax | Style |
| M6 | SLICER_TIMEOUT env var validation with fallback | Reliability |

### Reliability Hardening (2026-03)

| ID | Summary |
|----|---------|
| B-1 | Merged Dependabot PRs #44, #45 |
| B-2 | Fixed rendering.py orphan node (Subject Hierarchy parenting) |
| B-3 | PR #42 closed — TotalSeg subprocess already in canonical files |
| B-4 | Command doc validation — completed via fresh creation with correct signatures |

### V2 Restructure + Full Sweep (2026-03)

- Three-layer architecture (server / features / core)
- Workflow audit of 9 commands vs 27 tools
- Tier 1 workflow tools implemented (onco-spine, ccj, modic) + 73 tests
- `slicer://workflows` resource
- `__all__` exports on 14 canonical modules
- DeprecationWarning on 14 shim files
- .gitignore, CLAUDE.md, README.md English, CI coverage enforcement
- Plans consolidated into single PROJECT_STATUS.md

## Archived Plans

Detailed historical plans in `docs/plans/archive/`:
- `2026-02-24-code-review-fixes.md` — 13 code review tasks (all completed)
- `2026-03-07-path-b-reliability.md` — 4 reliability tasks (all completed)
- `2026-03-07-v2-roadmap.md` — v2 strategic direction (achieved)
- `v2-workflow-surface.md` — workflow tier system and tool signatures
- `workflow-audit-results.md` — full audit with per-workflow tables
