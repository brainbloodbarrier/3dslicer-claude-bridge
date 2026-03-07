# V2 Workflow Surface Decision

**Date**: 2026-03-07
**Status**: Proposed
**Branch**: v2/restructure-codebase
**Prerequisite**: workflow-audit-results.md (completed)

## 1. Current State Analysis

### 9 Claude Commands (excluding bootstrap-env and todo, which are dev-ops)

| Command | Tools Used | Orchestration Steps | Modality | Audit Issues |
|---------|-----------|---------------------|----------|-------------|
| spine-eval | 8 | 9 | CT + MRI + X-ray | 1 medium (param rename) |
| fracture-eval | 7 | ~8 | CT or X-ray | 1 critical (missing landmarks) |
| modic-eval | 6 | 7 | MRI (T1+T2) | 0 |
| ccj-protocol | 6 | 8 | CT + CTA | 1 high (param rename) |
| screw-plan | 6 | 7 | CT + CTA | 1 high (param rename) |
| construct-compare | 5 | 6 | CT + CTA | 0 (inherits from screw-plan) |
| instability-protocol | 6 | 5 | X-ray (3) + CT | 1 critical (param structure) |
| onco-spine | 9 | 9 | CT + MRI | 1 critical (extra param) |
| full-spine-workup | 18 | 8 | All modalities | 1 critical (inherits instability) |

### Highest-Value Workflows

By clinical utility: **onco-spine**, **ccj-protocol**, and **screw-plan** are the most
specialized. They encode domain knowledge (SINS scoring, CCJ craniometry, VA safety
checks) that a general user would not know to orchestrate.

By orchestration complexity: **full-spine-workup** (18 tools), **onco-spine** (9 tools),
and **spine-eval** (8 tools) combine the most primitives.

By correctness today: **modic-eval** and **construct-compare** have zero audit issues
and are closest to being promotable without fixes.

### Tool Dependency Clusters

Most workflows share a common preamble: `list_scene_nodes` then `segment_spine` then
`capture_screenshot`. The differentiating tools are:

- **CT diagnostics cluster**: `detect_vertebral_fractures_ct`, `assess_osteoporosis_ct`,
  `measure_listhesis_ct`, `measure_spinal_canal_ct`, `detect_metastatic_lesions_ct`,
  `calculate_sins_score`
- **MRI cluster**: `classify_modic_changes`, `assess_disc_degeneration_mri`,
  `detect_cord_compression_mri`, `detect_metastatic_lesions_mri`
- **X-ray cluster**: `measure_sagittal_balance_xray`, `measure_coronal_balance_xray`,
  `measure_cobb_angle_xray`, `detect_vertebral_fractures_xray`,
  `measure_listhesis_dynamic_xray`
- **Instrumentation cluster**: `plan_cervical_screws`, `segment_vertebral_artery`,
  `analyze_bone_quality`
- **CCJ cluster**: `measure_ccj_angles` (uses spine cluster for segmentation)

## 2. Recommendation: Tier Assignment

### Tier 1 -- Promote to MCP Workflow Tools

These orchestrate 5+ primitives, encode significant clinical domain knowledge, and
give Cursor users access to capabilities they cannot get from the Claude command files.

| Workflow | Justification |
|----------|--------------|
| **onco-spine** | 9 tools, SINS scoring logic, multi-modality correlation, clear clinical decision tree (stable/indeterminate/unstable). Highest single-workflow clinical value. |
| **ccj-protocol** | 6 tools, specialized craniometry + VA safety assessment + bone quality. CCJ evaluation requires domain knowledge that benefits from server-side orchestration. |
| **modic-eval** | 6 tools, zero audit issues, MRI-specific pipeline (T1+T2 required), Modic + Pfirrmann + cord compression. Clean and self-contained. |

### Tier 2 -- Keep as Claude Commands, Fix Docs

These are better as guided orchestration because they require significant user
interaction (modality identification, landmark placement, level selection) or are
supersets of Tier 1 workflows.

| Workflow | Justification |
|----------|--------------|
| **spine-eval** | Good general-purpose workflow but highly adaptive (branches on available modalities). Better as a guided command where the LLM adapts the pipeline. |
| **fracture-eval** | CT path is promotable, but the X-ray path requires manual landmark placement that cannot be automated in a single MCP call. Keep as command. |
| **screw-plan** | Requires interactive user confirmation for C1-C2 without CTA. The safety gate ("WARN and wait for confirmation") is inherently conversational. |
| **full-spine-workup** | Superset of everything. Too many branches and too long-running for a single MCP tool. Better as a guided orchestration that calls Tier 1 workflows as sub-steps. |

### Tier 3 -- Merge or Deprecate

| Workflow | Justification |
|----------|--------------|
| **construct-compare** | Thin wrapper around multiple `plan_cervical_screws` calls with a comparison table. The comparison logic is better done by the LLM. Merge guidance into `screw-plan.md`. |
| **instability-protocol** | Has the most severe audit issues (critical param structure mismatch, requires landmark placement on 3 X-ray images). Keep the tool primitive; deprecate the workflow command until landmark UX improves. |

## 3. Proposed Tool Signatures (Tier 1)

### workflow_onco_spine

```python
@mcp.tool()
def workflow_onco_spine(
    ct_volume_id: str,                          # MRML node ID of CT volume (REQUIRED)
    region: str = "full",                       # Spine region to evaluate
    t1_volume_id: str | None = None,            # Optional MRI T1 for enhanced detection
    t2_volume_id: str | None = None,            # Optional MRI T2/STIR
    segmentation_node_id: str | None = None,    # Existing segmentation (skips segment_spine)
    pain_type: str | None = None,               # "mechanical", "non_mechanical", or None
) -> dict:
    """Run oncologic spine assessment: lesion detection, SINS scoring, stability analysis.

    Orchestrates: segment_spine, detect_metastatic_lesions_ct, calculate_sins_score,
    measure_listhesis_ct, measure_spinal_canal_ct, assess_osteoporosis_ct,
    detect_metastatic_lesions_mri (if MRI provided), capture_screenshot.

    Returns dict with lesion_map, sins_scores, canal_status, bone_quality,
    stability_classification, and screenshots.
    """
```

### workflow_ccj_protocol

```python
@mcp.tool()
def workflow_ccj_protocol(
    ct_volume_id: str,                          # MRML node ID of CT volume (REQUIRED)
    segmentation_node_id: str | None = None,    # Existing segmentation (skips segment_spine)
    cta_volume_id: str | None = None,           # CTA for vertebral artery assessment
    population: str = "adult",                  # "adult" or "child" (ADI threshold)
    include_bone_quality: bool = True,          # Include C1-C2 HU analysis
) -> dict:
    """Run craniocervical junction protocol: craniometry, VA assessment, bone quality.

    Orchestrates: segment_spine, measure_ccj_angles, segment_vertebral_artery (if CTA),
    analyze_bone_quality, capture_screenshot.

    Returns dict with craniometry (CXA, ADI, Powers, Ranawat, McGregor, Chamberlain,
    Wackenheim), instability_flags, vascular_assessment, bone_quality, and screenshots.
    """
```

### workflow_modic_eval

```python
@mcp.tool()
def workflow_modic_eval(
    t1_volume_id: str,                          # MRML node ID of T1 MRI (REQUIRED)
    t2_volume_id: str,                          # MRML node ID of T2 MRI (REQUIRED)
    region: str = "lumbar",                     # Spine region
    segmentation_node_id: str | None = None,    # Existing segmentation (skips segment_spine)
    include_cord_screening: bool = True,        # Cord compression for cervical/thoracic
) -> dict:
    """Run Modic endplate and disc degeneration assessment from MRI.

    Orchestrates: segment_spine, classify_modic_changes, assess_disc_degeneration_mri,
    detect_cord_compression_mri (if cervical/thoracic), capture_screenshot.

    Returns dict with modic_changes (per level/endplate), pfirrmann_grades,
    cord_compression (if screened), pain_source_analysis, and screenshots.
    """
```

## 4. Discoverability Strategy

### New MCP Resource: `slicer://workflows`

Add a resource that lists available workflow tools with required inputs and clinical
use cases. Gives Cursor users a way to discover high-level operations.

```python
@mcp.resource("slicer://workflows")
def get_workflows() -> str:
    """List available spine workflow tools with required inputs and clinical use cases."""
```

Returns JSON with each workflow's name, required modalities, clinical indication,
and expected runtime.

### Tool Description Improvements

Each workflow tool's docstring should include:
1. **Clinical indication** in the first line
2. **Required modalities** clearly stated
3. **List of orchestrated primitives**
4. **Expected runtime** (2-10 minutes for workflow tools)

### No New Framework Needed

The existing `@mcp.tool()` pattern plus one `slicer://workflows` resource is
sufficient. Cursor reads tool descriptions from MCP; Claude Code reads
`.claude/commands/*.md`. Both surfaces are served.

## 5. Migration Plan

### Order of Implementation

1. **workflow_modic_eval** (simplest Tier 1, zero audit issues)
2. **workflow_ccj_protocol** (moderate complexity)
3. **workflow_onco_spine** (highest complexity, most tools orchestrated)
4. **slicer://workflows resource**
5. **Merge/deprecate Tier 3** command docs
6. **Update Tier 2 command docs** with cross-references

### Dependencies

```
audit fixes ─────┬──> workflow_modic_eval ──> slicer://workflows resource
                 ├──> workflow_ccj_protocol ──────────────┘
                 └──> workflow_onco_spine ────────────────┘
```

All three workflow implementations are independent and can be developed in parallel.

### Testing Strategy

- **Unit tests**: Each workflow tool gets a test file that mocks its constituent
  primitive tools. Verify correct call sequence, error propagation, and result
  aggregation.
- **Integration tests**: Mark with `@pytest.mark.integration`. Require Slicer with
  sample data. Test one happy path per workflow.
- **Regression**: Existing primitive tool tests must continue to pass.

### What Does Not Change

- All 45 existing primitive tools remain unchanged
- Existing 3 resources remain unchanged
- Tier 2 Claude commands continue to work (with fixed docs)
- `server.py` error handling pattern (`_handle_tool_error`) is reused by workflow tools
