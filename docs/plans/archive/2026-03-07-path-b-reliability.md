# Path B: Reliability Hardening — Implementation Plan

**STATUS:**
- B-1 (Dependabot PRs #44, #45): MERGED (2026-03)
- B-2 (rendering.py orphan node): FIXED (Subject Hierarchy parenting applied)
- B-3 (PR #42 TotalSeg port): CLOSED — TotalSeg subprocess code already in canonical files
- B-4 (command doc validation): PENDING — `.claude/commands/` not yet committed to repo

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize the existing codebase by porting PR #42's TotalSegmentator changes to the v2 structure, fixing the rendering.py orphan node bug, and merging Dependabot PRs.

**Architecture:** PR #42 was created pre-restructure and cannot merge as-is (it reverts all `core/` and `features/` packages). The 3 commits with useful changes must be extracted and applied to canonical `features/` paths. All other issues are independent and can run in parallel.

**Tech Stack:** Python 3.10+, FastMCP, pytest, GitHub CLI

---

## Issue Map & Labels

| Issue # | Title | Labels | Parallel Group | Depends On |
|---------|-------|--------|----------------|------------|
| B-1 | Merge Dependabot PRs (#44, #45) | `dependencies`, `chore` | A | — |
| B-2 | Fix rendering.py orphan folder node | `bug`, `rendering` | A | — |
| B-3 | Port PR #42 TotalSeg subprocess to v2 structure | `feat`, `spine`, `diagnostics` | B | B-1 |
| B-4 | Validate command docs post-merge | `docs`, `qa` | C | B-3 |

**Execution order:** Group A (B-1 + B-2) in parallel → Group B (B-3) → Group C (B-4)

---

## GitHub Labels to Create

```bash
gh label create "rendering" --color "D4C5F9" --description "Volume rendering & 3D export" 2>/dev/null
gh label create "qa" --color "FBCA04" --description "Quality assurance & validation" 2>/dev/null
```

Existing labels: `bug`, `feat`, `chore`, `dependencies`, `docs`, `spine`, `diagnostics`

---

### Task B-1: Merge Dependabot PRs (#44, #45)

**Assignee:** Team Lead (trivial, no agent needed)

**Files:** None (GitHub Actions config only)

**Step 1: Verify CI status**

```bash
gh pr checks 44
gh pr checks 45
```

Expected: All checks PASSED (already confirmed)

**Step 2: Merge both PRs**

```bash
gh pr merge 44 --squash --delete-branch
gh pr merge 45 --squash --delete-branch
```

**Step 3: Pull main**

```bash
git checkout main && git pull origin main
```

---

### Task B-2: Fix rendering.py orphan folder node

**Assignee:** Agent Alpha

**Files:**
- Modify: `src/slicer_mcp/features/rendering.py` (lines 328-374)
- Modify: `tests/unit/test_rendering_tools.py` (add test)

**Context:** `segmentation_to_models()` creates a `vtkMRMLFolderDisplayNode` for organization but never parents the exported model nodes under it. The models are orphaned in the MRML scene.

**Step 1: Write the failing test**

In `tests/unit/test_rendering_tools.py`, add to the `TestSegmentationToModels` class:

```python
def test_models_parented_under_folder(self):
    """Model nodes should be parented under the folder display node."""
    with patch("slicer_mcp.features.rendering.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.exec_python.return_value = {
            "result": json.dumps({
                "success": True,
                "folder_node_id": "vtkMRMLFolderDisplayNode1",
                "folder_name": "TestSeg_models",
                "models": [{
                    "segment_id": "Segment_1",
                    "segment_name": "L1",
                    "model_node_id": "vtkMRMLModelNode1",
                    "model_node_name": "L1",
                    "point_count": 100,
                    "cell_count": 50,
                }],
                "model_count": 1,
            })
        }
        mock_get_client.return_value = mock_client

        result = segmentation_to_models(segmentation_node_id="vtkMRMLSegmentationNode1")

        assert result["folder_node_id"] == "vtkMRMLFolderDisplayNode1"
        # Verify the generated code includes folder parenting
        code = mock_client.exec_python.call_args[0][0]
        assert "SetItemParent" in code or "shNode.SetItemParent" in code
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_rendering_tools.py::TestSegmentationToModels::test_models_parented_under_folder -v
```

Expected: FAIL (no folder parenting in generated code, no `folder_node_id` in result)

**Step 3: Fix the generated Python code in rendering.py**

In `src/slicer_mcp/features/rendering.py`, in the `_build_segmentation_to_models_code()` function, after the model creation loop, add Subject Hierarchy parenting:

Replace the current orphan block (lines ~328-331):

```python
# Create a folder node for organization
shFolderNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLFolderDisplayNode')
folderName = segNode.GetName() + '_models'
shFolderNode.SetName(folderName)
```

With proper Subject Hierarchy folder + parenting:

```python
# Create a Subject Hierarchy folder for organization
shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
sceneItemID = shNode.GetSceneItemID()
folderName = segNode.GetName() + '_models'
folderItemID = shNode.CreateFolderItem(sceneItemID, folderName)
```

Then inside the model creation loop, after `modelNode.CreateDefaultDisplayNodes()`, add:

```python
        # Parent model under folder in Subject Hierarchy
        modelItemID = shNode.GetItemByDataNode(modelNode)
        if modelItemID:
            shNode.SetItemParent(modelItemID, folderItemID)
```

And in the result dict at the end, add `folder_node_id`:

```python
'folder_item_id': folderItemID,
'folder_name': folderName,
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_rendering_tools.py -v -k "segmentation_to_models"
```

Expected: ALL PASS

**Step 5: Run full test suite**

```bash
uv run pytest -m "not integration and not benchmark" -q
```

Expected: 1033+ passed

**Step 6: Commit**

```bash
git add src/slicer_mcp/features/rendering.py tests/unit/test_rendering_tools.py
git commit -m "fix(rendering): parent model nodes under Subject Hierarchy folder

segmentation_to_models() created a folder display node but never
parented exported models under it, leaving orphan nodes in the scene.
Use Subject Hierarchy API to properly organize models in a folder."
```

---

### Task B-3: Port PR #42 TotalSeg Subprocess to V2 Structure

**Assignee:** Agent Bravo (primary) + Agent Charlie (tests)

**This is the most complex task.** PR #42 has 3 commits with changes to files at OLD paths. These must be ported to the canonical `features/` locations.

#### Sub-task B-3a: Extract the actual changes from PR #42

**Step 1: Create working branch**

```bash
git checkout main
git checkout -b feat/totalseg-subprocess-v2
```

**Step 2: Identify what changed in PR #42**

The 3 commits in PR #42 (`feat/totalseg-subprocess-cpu-optimization`):

```
f32b12e fix(ct-tools): use free 'total' task, subprocess segmentation, and CPU timeout
12d41f9 feat(mri-tools): add segmentation reuse + subprocess TotalSegmentator
12d4420 fix(review): address PR #42 review findings
```

Changes by file (OLD paths → NEW canonical paths):

| Old Path | New Canonical Path | Change Type |
|----------|-------------------|-------------|
| `src/slicer_mcp/diagnostic_tools_ct.py` | `src/slicer_mcp/features/diagnostics/ct.py` | TotalSeg subprocess, CPU timeout |
| `src/slicer_mcp/diagnostic_tools_mri.py` | `src/slicer_mcp/features/diagnostics/mri.py` | Segmentation reuse, subprocess TotalSeg |
| `src/slicer_mcp/spine_tools.py` | `src/slicer_mcp/features/spine/tools.py` | Subprocess TotalSegmentator migration |
| `src/slicer_mcp/spine_constants.py` | `src/slicer_mcp/features/spine/constants.py` | New constants (if any) |
| `src/slicer_mcp/server.py` | `src/slicer_mcp/server.py` | New tool registrations (if any) |
| `tests/test_diagnostic_tools_ct.py` | `tests/unit/test_diagnostic_tools_ct.py` | Updated tests |
| `tests/test_diagnostic_tools_mri.py` | `tests/unit/test_diagnostic_tools_mri.py` | Updated tests |
| `tests/test_spine_tools.py` | `tests/unit/test_spine_tools.py` | Updated tests |

**Step 3: Generate diffs per-file from PR #42 branch**

For each changed source file, extract the diff against its pre-v2 base:

```bash
# Get the merge-base of PR #42 with main before v2
MERGE_BASE=$(git merge-base origin/feat/totalseg-subprocess-cpu-optimization 64c235e)

# Diff each file against its pre-v2 ancestor
git diff $MERGE_BASE origin/feat/totalseg-subprocess-cpu-optimization -- src/slicer_mcp/diagnostic_tools_ct.py > /tmp/pr42-ct.diff
git diff $MERGE_BASE origin/feat/totalseg-subprocess-cpu-optimization -- src/slicer_mcp/diagnostic_tools_mri.py > /tmp/pr42-mri.diff
git diff $MERGE_BASE origin/feat/totalseg-subprocess-cpu-optimization -- src/slicer_mcp/spine_tools.py > /tmp/pr42-spine.diff
git diff $MERGE_BASE origin/feat/totalseg-subprocess-cpu-optimization -- src/slicer_mcp/spine_constants.py > /tmp/pr42-constants.diff
git diff $MERGE_BASE origin/feat/totalseg-subprocess-cpu-optimization -- src/slicer_mcp/server.py > /tmp/pr42-server.diff
```

**Step 4: Apply diffs to canonical files**

For each diff, manually apply the semantic changes to the canonical `features/` files. The code is identical — only the file paths differ. The agent should:

1. Read the diff to understand each change
2. Find the corresponding function in the canonical file
3. Apply the change

**Key changes to look for (from PR title and commits):**
- `subprocess.run()` calls replacing in-process TotalSegmentator
- CPU-specific timeout handling
- `--fast` or `--task total` flags for TotalSegmentator CLI
- Segmentation reuse logic (check if segmentation already exists)
- New constants for subprocess timeouts

**Step 5: Apply test changes**

Same process for test files — extract diffs and apply to `tests/unit/` paths.

**Step 6: Run lint**

```bash
uv run black src tests
uv run ruff check src tests --fix
uv run mypy src/
```

**Step 7: Run full tests**

```bash
uv run pytest -m "not integration and not benchmark" -v
```

Expected: ALL PASS

**Step 8: Commit**

```bash
git add -A
git commit -m "feat(totalseg): port subprocess migration from PR #42 to v2 structure

Port TotalSegmentator subprocess migration and CPU optimization from
feat/totalseg-subprocess-cpu-optimization (PR #42) to the canonical
core/features/ package structure. Original PR could not merge due to
v2 restructure.

Changes ported:
- CT tools: subprocess TotalSeg with 'total' task, CPU timeout
- MRI tools: segmentation reuse + subprocess TotalSeg
- Spine tools: subprocess-based segmentation
- Tests updated for all changes"
```

#### Sub-task B-3b: Push, create PR, close old PR

**Step 9: Push and create PR**

```bash
git push -u origin feat/totalseg-subprocess-v2
gh pr create --title "feat(totalseg): subprocess migration + CPU optimization (v2 port)" \
  --body "$(cat <<'EOF'
## Summary
- Port of #42 to v2 package structure (`core/` + `features/`)
- TotalSegmentator migrated from in-process to subprocess execution
- CPU-specific timeout handling for diagnostic tools
- Segmentation reuse to avoid redundant computation

Supersedes #42 (cannot merge due to v2 restructure conflict).

## Test plan
- [ ] All unit tests pass (1033+)
- [ ] Lint (black + ruff + mypy) clean
- [ ] CI passes on 3.10, 3.11, 3.12

Closes #42

Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 10: Close old PR #42 with explanation**

```bash
gh pr close 42 --comment "Superseded by #<new_pr_number>. Original changes ported to v2 package structure (core/ + features/)."
```

---

### Task B-4: Validate Command Docs Post-Merge

**Assignee:** Agent Delta

**Files:** `.claude/commands/*.md` (9 workflow commands)

**Context:** After all merges, validate that every tool call in every command doc matches the actual function signatures in `features/` modules. The v2 audit already fixed most issues, but the TotalSeg port may introduce new parameters.

**Step 1: For each command file, extract tool calls**

```bash
grep -h "^[a-z_]*(" .claude/commands/*.md | sort -u
```

**Step 2: For each tool call, verify signature match**

Cross-reference parameter names against actual function signatures in:
- `src/slicer_mcp/features/diagnostics/ct.py`
- `src/slicer_mcp/features/diagnostics/mri.py`
- `src/slicer_mcp/features/diagnostics/xray.py`
- `src/slicer_mcp/features/spine/tools.py`
- `src/slicer_mcp/features/base_tools.py`
- `src/slicer_mcp/features/registration.py`
- `src/slicer_mcp/features/rendering.py`

**Step 3: Document findings**

If mismatches found, fix them in the same commit:

```bash
git add .claude/commands/*.md
git commit -m "docs: fix command doc parameter mismatches after TotalSeg port"
```

If no mismatches:

```
echo "All 9 command docs validated — no mismatches found."
```

---

## Team Execution Plan

### Roles

| Agent | Role | Tasks | Tools |
|-------|------|-------|-------|
| **Lead** | Orchestrator | B-1, dispatch, review, merge | gh, git |
| **Alpha** | Bug fix | B-2 (rendering.py) | Edit, pytest |
| **Bravo** | Feature port | B-3 source files | Read, Edit, diff |
| **Charlie** | Test port | B-3 test files | Read, Edit, pytest |
| **Delta** | QA validation | B-4 command docs | Grep, Read |

### Timeline

```
Phase 1 (parallel):
  Lead ──── B-1: merge Dependabot PRs (2 min)
  Alpha ─── B-2: fix rendering.py orphan node (15 min)

Phase 2 (after B-1):
  Bravo ─── B-3a: port PR #42 source changes (30 min)
  Charlie ─ B-3a: port PR #42 test changes (20 min)
  [Bravo + Charlie sync, then commit together]

Phase 3 (after B-3):
  Delta ─── B-4: validate all command docs (10 min)
  Lead ──── Final review + merge
```

### CI/CD Integration

Each task creates a branch and PR:
- B-2: `fix/rendering-orphan-node` → PR to `main`
- B-3: `feat/totalseg-subprocess-v2` → PR to `main` (closes #42)
- B-4: `docs/command-validation` → direct commit or PR

Required CI checks before merge:
- `lint` (black + ruff + mypy)
- `test` (3.10, 3.11, 3.12)
- `auto-label`

### Success Criteria

- [ ] Dependabot PRs #44, #45 merged
- [ ] rendering.py folder parenting works (test proves it)
- [ ] TotalSeg subprocess changes live in `features/` (not old paths)
- [ ] PR #42 closed with reference to new PR
- [ ] All 9 command docs validated against tool signatures
- [ ] CI green on main after all merges
- [ ] 1033+ unit tests passing
