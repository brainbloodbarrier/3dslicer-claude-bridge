# Project Status

Single tracking file for slicer-mcp development. Updated continuously.
Replaces individual plan files (archived in `docs/plans/archive/`).

## Active Decisions

- **Architecture**: Three layers (server.py → features/ → core/). No rewrites.
- **Workflow tiers**: Tier 1 = MCP tools (server-side). Tier 2 = Claude commands (LLM-guided). Tier 3 = merge/deprecate.
- **Tier 1** (implemented): `workflow_onco_spine`, `workflow_ccj_protocol`, `workflow_modic_eval`
- **Tier 2** (keep as commands, fix docs): spine-eval, fracture-eval, screw-plan, full-spine-workup
- **Tier 3** (merge/deprecate): construct-compare → merge into screw-plan; instability-protocol → deprecate
- **Discoverability**: `slicer://workflows` resource lists workflow tools for Cursor/MCP clients

## Pending

### Tier 3 Workflow Cleanup

- [ ] Merge `construct-compare` guidance into `screw-plan.md` command doc
- [ ] Deprecate `instability-protocol` command doc (landmark UX prerequisite not met)

### Low-Priority Audit Fix

- [ ] `screw-plan.md`: add `region="cervical"` to `analyze_bone_quality()` call

## Blocked

All items below require `.claude/commands/*.md` to be committed to this repo.

### Audit Signature Fixes

**Critical (blocks execution):**

- [ ] `onco-spine.md`: remove `include_sins=true` from `detect_metastatic_lesions_ct()`. SINS is a separate tool.
- [ ] `fracture-eval.md` + `full-spine-workup.md`: `detect_vertebral_fractures_xray()` missing required `landmarks_per_vertebra`. Add `place_landmarks()` + `get_landmarks()` steps.
- [ ] `instability-protocol.md` + `full-spine-workup.md`: `measure_listhesis_dynamic_xray()` wrong param structure. Needs `volume_node_ids` (dict), `landmarks_per_position` (dict), `levels` (list).

**High (param rename):**

- [ ] `ccj-protocol.md` + `screw-plan.md`: `segment_vertebral_artery()` uses `volume_node_id` → should be `input_node_id`

**Medium (param rename):**

- [ ] `spine-eval.md`: `measure_spine_alignment()` uses `segmentation_id` → should be `segmentation_node_id`

### Command Doc Validation (B-4)

- [ ] After fixes above: validate all tool calls in all command docs against `features/` signatures

## Completed Work

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

### V2 Restructure + Full Sweep (2026-03)

- Three-layer architecture (server / features / core)
- Workflow audit of 9 commands vs 27 tools
- Tier 1 workflow tools implemented (onco-spine, ccj, modic) + 73 tests
- `slicer://workflows` resource
- `__all__` exports on 14 canonical modules
- DeprecationWarning on 14 shim files
- .gitignore, CLAUDE.md, README.md English, CI coverage enforcement

## Reference

### Tool Signatures (for Blocked Fixes)

```python
detect_metastatic_lesions_ct(volume_node_id, segmentation_node_id=None, region=None, include_posterior_elements=False)
detect_vertebral_fractures_xray(volume_node_id, landmarks_per_vertebra, magnification_factor=1.0)
measure_listhesis_dynamic_xray(volume_node_ids, landmarks_per_position, levels, region="lumbar", magnification_factor=1.0)
segment_vertebral_artery(input_node_id, side=None, seed_points=None)
measure_spine_alignment(segmentation_node_id, region=None)
```

### Archived Plans

Detailed historical plans in `docs/plans/archive/`:
- `2026-02-24-code-review-fixes.md` — 13 code review tasks (all completed)
- `2026-03-07-path-b-reliability.md` — 4 reliability tasks (3 done, 1 blocked)
- `2026-03-07-v2-roadmap.md` — v2 strategic direction (achieved)
- `v2-workflow-surface.md` — workflow tier system and tool signatures
- `workflow-audit-results.md` — full audit with per-workflow tables
