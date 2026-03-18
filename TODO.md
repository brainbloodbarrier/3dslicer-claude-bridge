# TODO

## Critical

- [ ] Audit the current spine workflow docs in `.claude/commands/` against the
  real MCP tool signatures in `src/slicer_mcp/server.py`
- [ ] Identify the highest-risk workflow mismatches for `spine-eval`,
  `fracture-eval`, `modic-eval`, `ccj-protocol`, `screw-plan`, and
  `construct-compare`
- [ ] Decide which spine workflows should become first-class MCP workflow tools
  in v2 instead of remaining Claude-only command docs

## High

- [ ] Define the minimum v2 workflow surface needed for both Claude Code and
  Cursor
- [ ] Map the current primitive tools that should remain stable during v2
- [ ] Identify the first workflow slice to migrate and test end-to-end
- [ ] Tighten `CLAUDE.md`, `README.md`, and `.claude/commands/*` so they point
  to the same current direction

## Medium

- [ ] Improve capability discovery for Cursor users beyond setup-only docs
- [ ] Define a lightweight compatibility check for command docs vs tool
  signatures
- [ ] Decide how to expose workflow-level status and discovery through MCP
  resources

## Low

- [ ] Revisit whether the root `TODO.md` should later move under `docs/`
- [ ] Add a short “current v2 status” note to more user-facing docs if the
  roadmap starts to drift from reality
