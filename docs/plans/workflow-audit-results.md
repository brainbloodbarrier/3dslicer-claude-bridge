# Workflow Audit Results

**Date**: 2026-03-07  
**Branch**: v2/restructure-codebase  
**Scope**: Audit of 9 spine workflow command docs against actual MCP tool signatures in `server.py`

## Executive Summary

- **Total workflows audited**: 9
- **Total tools referenced**: 27 unique tools
- **Total issues found**: 6
  - **Critical mismatches**: 3
  - **High-priority mismatches**: 1
  - **Medium-priority mismatches**: 1
  - **Low-priority mismatches**: 1

All issues are parameter-related. No workflow references non-existent tools. All referenced tools exist and are properly registered in `server.py`.

## Critical Issues (Block Execution)

### 1. `onco-spine.md` → `detect_metastatic_lesions_ct`

**Issue**: Parameter `include_sins` does not exist in tool signature  
**Severity**: CRITICAL  
**Location**: Lines 40-45

```markdown
# Workflow (WRONG)
detect_metastatic_lesions_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    include_posterior_elements=true,
    include_sins=true
)
```

**Actual signature**:
```python
detect_metastatic_lesions_ct() -> dict
  - volume_node_id: str (REQUIRED)
  - segmentation_node_id: str | None (optional)
  - region: str (optional)
  - include_posterior_elements: bool (optional)
```

**Why this fails**: SINS (Spinal Instability Neoplastic Score) calculation is a separate tool (`calculate_sins_score`), not a parameter of the lesion detection tool.

**Fix recommendation**: 
1. Remove `include_sins=true` from the call
2. Call `calculate_sins_score()` separately after detecting lesions (already done in step 5 of the workflow, so this is redundant)

---

### 2. `fracture-eval.md` → `detect_vertebral_fractures_xray`

**Issue**: Missing required parameter `landmarks_per_vertebra`  
**Severity**: CRITICAL  
**Location**: Lines 106-110

```markdown
# Workflow (INCOMPLETE)
detect_vertebral_fractures_xray(
    volume_node_id=<xray_id>,
    region="thoracolumbar"
)
```

**Actual signature**:
```python
detect_vertebral_fractures_xray() -> dict
  - volume_node_id: str (REQUIRED)
  - landmarks_per_vertebra: dict[str, dict[str, list[float]]] (REQUIRED)
  - magnification_factor: float (optional)
```

**Why this fails**: The tool requires precise landmark coordinates (e.g., superior/inferior endplate corners) for each vertebra to measure heights and apply Genant grading. X-ray fracture detection is landmark-driven, not automated segmentation.

**Fix recommendation**: 
1. Either call `place_landmarks()` first to place anatomical landmarks on the X-ray
2. Then extract those landmarks via `get_landmarks()` 
3. Pass the structured landmark dict to this tool
4. **Alternative**: Document that this tool requires manual landmark placement first (may not be suitable for fully automated workflow)

---

### 3. `instability-protocol.md` & `full-spine-workup.md` → `measure_listhesis_dynamic_xray`

**Issue**: Parameter structure mismatch—workflow uses separate volume parameters, but signature requires dict structure + missing landmarks  
**Severity**: CRITICAL  
**Locations**: 
- instability-protocol.md, lines 37-43
- full-spine-workup.md, lines 96-101

```markdown
# Workflow (WRONG STRUCTURE)
measure_listhesis_dynamic_xray(
    neutral_volume_id=<neutral_id>,
    flexion_volume_id=<flexion_id>,
    extension_volume_id=<extension_id>,
    levels=<levels_if_specified>
)
```

**Actual signature**:
```python
measure_listhesis_dynamic_xray() -> dict
  - volume_node_ids: dict[str, str] (REQUIRED)         # e.g., {"neutral": id, "flexion": id, "extension": id}
  - landmarks_per_position: dict[str, dict[str, dict[str, list[float]]]] (REQUIRED)  # Landmarks for EACH position
  - levels: list[str] (REQUIRED)
  - region: str (optional)
  - magnification_factor: float (optional)
```

**Why this fails**: 
1. Tool expects volume IDs in a dict keyed by position name, not as separate parameters
2. Tool requires landmarks for ALL THREE positions (neutral, flexion, extension), not just volume IDs
3. Workflow doesn't provide landmarks at all

**Fix recommendation**: 
1. Restructure volume IDs as dict: `volume_node_ids={"neutral": id, "flexion": id, "extension": id}`
2. Place landmarks on each X-ray image separately: call `place_landmarks()` three times (once per position)
3. Call `get_landmarks()` three times to retrieve them
4. Assemble nested dict structure for `landmarks_per_position`:
   ```
   {
       "neutral": { "L4": {"superior": [x,y], "inferior": [x,y]}, "L5": {...}, ...},
       "flexion": { "L4": {...}, ...},
       "extension": { "L4": {...}, ...}
   }
   ```
5. Ensure `levels` is a list: `levels=["L4", "L5", "S1"]`

---

## High-Priority Issues (Parameter Renames)

### 4. `ccj-protocol.md` & `screw-plan.md` → `segment_vertebral_artery`

**Issue**: Parameter name mismatch: workflow uses `volume_node_id`, signature requires `input_node_id`  
**Severity**: HIGH  
**Locations**:
- ccj-protocol.md, lines 59-62
- screw-plan.md, lines 42-45

```markdown
# Workflow (WRONG)
segment_vertebral_artery(
    volume_node_id=<cta_volume_id>,
    segmentation_node_id=<seg_id>
)

# Correct
segment_vertebral_artery(
    input_node_id=<cta_volume_id>,
    segmentation_node_id=<seg_id>
)
```

**Actual signature**:
```python
segment_vertebral_artery() -> dict
  - input_node_id: str (REQUIRED)
  - side: str (optional)
  - seed_points: list[list[float]] | None (optional)
```

**Why this fails**: Parameter name does not match tool signature. Tool will reject unknown parameter `volume_node_id`.

**Fix recommendation**: Rename `volume_node_id=<cta_volume_id>` to `input_node_id=<cta_volume_id>` in both workflows.

---

## Medium-Priority Issues (Parameter Renames)

### 5. `spine-eval.md` → `measure_spine_alignment`

**Issue**: Parameter name mismatch: workflow uses `segmentation_id`, signature requires `segmentation_node_id`  
**Severity**: MEDIUM  
**Location**: Lines 58-59

```markdown
# Workflow (WRONG)
measure_spine_alignment(segmentation_id=<seg_id>, region=<region>)

# Correct
measure_spine_alignment(segmentation_node_id=<seg_id>, region=<region>)
```

**Actual signature**:
```python
measure_spine_alignment() -> dict
  - segmentation_node_id: str (REQUIRED)
  - region: str (optional)
```

**Why this fails**: Parameter name does not match. This is a simple rename—all other workflows use the correct `segmentation_node_id`.

**Fix recommendation**: Rename `segmentation_id` to `segmentation_node_id`.

---

## Low-Priority Issues (Missing Optional Parameters)

### 6. `screw-plan.md` → `analyze_bone_quality` (Missing optional `region`)

**Issue**: Workflow calls tool without optional `region` parameter that could improve specificity  
**Severity**: LOW  
**Location**: Lines 72-77

```markdown
# Workflow (WORKS, but incomplete)
analyze_bone_quality(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=<levels_list>
)

# Better
analyze_bone_quality(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    region="cervical",  # Specify region for context
    levels=<levels_list>
)
```

**Actual signature**:
```python
analyze_bone_quality() -> dict
  - input_node_id: str (REQUIRED)
  - segmentation_node_id: str (REQUIRED)
  - region: str (optional)
```

**Why this is suboptimal**: The optional `region` parameter allows the tool to apply region-specific bone quality thresholds (cervical vs. subaxial vs. lumbar). For C1-C2 fixation planning, specifying `region="cervical"` would be contextually useful.

**Fix recommendation**: Add `region="cervical"` to the call to provide anatomical context, though the tool will still execute without it.

---

## Per-Workflow Detailed Results

### spine-eval.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct: uses input_node_id, region (optional) |
| capture_screenshot | ✓ Yes | None | — | Correct |
| measure_spine_alignment | ✓ Yes | `segmentation_id` → `segmentation_node_id` | MEDIUM | Parameter rename required |
| measure_ccj_angles | ✓ Yes | None | — | Correct |
| measure_sagittal_balance_xray | ✓ Yes | Missing landmarks param | — | Can work with volume_node_id only; landmarks optional |
| measure_coronal_balance_xray | ✓ Yes | Missing landmarks param | — | Can work with volume_node_id only; landmarks optional |
| analyze_bone_quality | ✓ Yes | Missing optional region | LOW | Region parameter available but not used |

**Overall**: 1 medium issue (parameter rename)

---

### fracture-eval.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct |
| detect_vertebral_fractures_ct | ✓ Yes | None | — | Correct: has all required params |
| assess_osteoporosis_ct | ✓ Yes | None | — | Correct |
| measure_spinal_canal_ct | ✓ Yes | None | — | Correct |
| capture_screenshot | ✓ Yes | None | — | Correct |
| detect_vertebral_fractures_xray | ✓ Yes | **Missing `landmarks_per_vertebra`** | **CRITICAL** | X-ray fracture detection requires landmark-based measurement; region parameter alone insufficient |

**Overall**: 1 critical issue (missing required parameter)

---

### modic-eval.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct |
| classify_modic_changes | ✓ Yes | None | — | Correct: maps t1_volume_id → t1_node_id, t2_volume_id → t2_node_id (acceptable convention) |
| assess_disc_degeneration_mri | ✓ Yes | None | — | Correct: maps t2_volume_id → t2_node_id |
| detect_cord_compression_mri | ✓ Yes | None | — | Correct |
| capture_screenshot | ✓ Yes | None | — | Correct |

**Overall**: No issues. Uses `_volume_id` naming convention consistently, which aligns with parameter mapping (e.g., `t1_volume_id` → `t1_node_id`).

---

### ccj-protocol.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct |
| measure_ccj_angles | ✓ Yes | None | — | Correct |
| segment_vertebral_artery | ✓ Yes | `volume_node_id` → `input_node_id` | HIGH | Parameter rename required |
| analyze_bone_quality | ✓ Yes | Missing optional region | LOW | Region parameter optional |
| capture_screenshot | ✓ Yes | None | — | Correct |

**Overall**: 1 high issue (parameter rename)

---

### screw-plan.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct |
| segment_vertebral_artery | ✓ Yes | `volume_node_id` → `input_node_id` | HIGH | Parameter rename required |
| plan_cervical_screws | ✓ Yes | None | — | Correct: signature has level (singular), workflow uses levels (plural in prose but singular in param lists) |
| analyze_bone_quality | ✓ Yes | Missing optional region | LOW | Region parameter optional; should be "cervical" |
| capture_screenshot | ✓ Yes | None | — | Correct |

**Overall**: 1 high issue (parameter rename)

---

### construct-compare.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct |
| segment_vertebral_artery | ✓ Yes | None | — | No specific params in this workflow (covered elsewhere) |
| plan_cervical_screws | ✓ Yes | None | — | Correct |
| capture_screenshot | ✓ Yes | None | — | Correct |

**Overall**: No direct issues in this workflow (constructs concepts from other workflows)

---

### instability-protocol.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| measure_listhesis_dynamic_xray | ✓ Yes | **Wrong param structure; missing landmarks** | **CRITICAL** | Requires volume_node_ids (dict), landmarks_per_position (dict), levels (list) |
| segment_spine | ✓ Yes | None | — | Correct |
| measure_listhesis_ct | ✓ Yes | None | — | Correct |
| detect_vertebral_fractures_ct | ✓ Yes | None | — | Correct |
| capture_screenshot | ✓ Yes | None | — | Correct |

**Overall**: 1 critical issue (parameter structure/missing landmarks)

---

### onco-spine.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct |
| detect_metastatic_lesions_ct | ✓ Yes | **Extra param `include_sins`** | **CRITICAL** | Parameter does not exist; SINS is separate tool |
| detect_metastatic_lesions_mri | ✓ Yes | None | — | Correct |
| calculate_sins_score | ✓ Yes | None | — | Correct (step 5) |
| measure_listhesis_ct | ✓ Yes | None | — | Correct |
| measure_spinal_canal_ct | ✓ Yes | None | — | Correct |
| assess_osteoporosis_ct | ✓ Yes | None | — | Correct |
| capture_screenshot | ✓ Yes | None | — | Correct |

**Overall**: 1 critical issue (extra parameter that doesn't exist)

---

### full-spine-workup.md

| Tool Referenced | Exists? | Param Issues | Severity | Notes |
|-----------------|---------|--------------|----------|-------|
| list_scene_nodes | ✓ Yes | None | — | Correct |
| segment_spine | ✓ Yes | None | — | Correct |
| detect_vertebral_fractures_ct | ✓ Yes | None | — | Correct |
| assess_osteoporosis_ct | ✓ Yes | None | — | Correct |
| detect_metastatic_lesions_ct | ✓ Yes | None | — | Correct (doesn't use include_sins here) |
| calculate_sins_score | ✓ Yes | None | — | Correct |
| measure_listhesis_ct | ✓ Yes | None | — | Correct |
| measure_spinal_canal_ct | ✓ Yes | None | — | Correct |
| measure_sagittal_balance_xray | ✓ Yes | None | — | Correct |
| detect_vertebral_fractures_xray | ✓ Yes | Missing landmarks | — | Same as fracture-eval (could fail) |
| measure_coronal_balance_xray | ✓ Yes | None | — | Correct |
| measure_cobb_angle_xray | ✓ Yes | None | — | Correct |
| measure_listhesis_dynamic_xray | ✓ Yes | **Wrong param structure; missing landmarks** | **CRITICAL** | Same as instability-protocol |
| classify_modic_changes | ✓ Yes | None | — | Correct |
| assess_disc_degeneration_mri | ✓ Yes | None | — | Correct |
| detect_cord_compression_mri | ✓ Yes | None | — | Correct |
| detect_metastatic_lesions_mri | ✓ Yes | None | — | Correct |
| capture_screenshot | ✓ Yes | None | — | Correct |

**Overall**: 1 critical issue (parameter structure/missing landmarks—same as instability-protocol)

---

## Tools Referenced but Not Used in Workflows

The following tools exist in `server.py` but are NOT referenced in any workflow document:

- `apply_transform`
- `capture_3d_view`
- `classify_disc_degeneration_xray`
- `enable_volume_rendering`
- `execute_python`
- `export_model`
- `get_landmarks`
- `import_dicom`
- `list_dicom_series`
- `list_dicom_studies`
- `load_dicom_series`
- `load_sample_data`
- `measure_volume`
- `place_landmarks`
- `register_landmarks`
- `register_volumes`
- `run_brain_extraction`
- `segmentation_to_models`
- `set_layout`
- `set_volume_rendering_property`
- `visualize_spine_segmentation`

**Note**: Some of these (e.g., `place_landmarks`, `get_landmarks`) are REQUIRED for X-ray workflows but not explicitly documented in the workflow command docs. They should be included in the workflow descriptions for clarity.

---

## Recommendations for Remediation

### Immediate Actions (Critical Issues)

1. **onco-spine.md** (Line 40-45):
   - Remove `include_sins=true` parameter
   - Document that SINS scoring happens separately in step 5

2. **fracture-eval.md** (Line 106-110) & **full-spine-workup.md** (Line 83):
   - Document that X-ray fracture detection requires landmark placement
   - Add prerequisite steps for `place_landmarks()` and `get_landmarks()`
   - **OR** clarify that X-ray fracture detection in this pipeline is informational only and note limitations

3. **instability-protocol.md** & **full-spine-workup.md** (Lines 37-43 and 96-101):
   - Restructure to use dict-based volume parameter: `volume_node_ids={"neutral": id, "flexion": id, "extension": id}`
   - Document requirement for landmark placement on each X-ray position
   - Provide example landmark structure for `landmarks_per_position`

### Secondary Actions (High/Medium Priority)

4. **ccj-protocol.md** & **screw-plan.md**:
   - Rename `volume_node_id` → `input_node_id` in `segment_vertebral_artery()` calls

5. **spine-eval.md**:
   - Rename `segmentation_id` → `segmentation_node_id` in `measure_spine_alignment()` call

### Enhancement Opportunities (Low Priority)

6. **screw-plan.md**:
   - Add optional `region="cervical"` parameter to `analyze_bone_quality()` call for anatomical specificity

7. **All workflows**:
   - Document usage of `place_landmarks()` and `get_landmarks()` for X-ray-based workflows where they are prerequisites

---

## Appendix: Tool Signature Reference

### Critical Tools for X-ray Workflows

#### `detect_vertebral_fractures_xray`
```python
def detect_vertebral_fractures_xray(
    volume_node_id: str,                                    # REQUIRED
    landmarks_per_vertebra: dict[str, dict[str, list[float]]],  # REQUIRED
    magnification_factor: float = 1.0                       # optional
) -> dict
```

**Example `landmarks_per_vertebra` structure**:
```python
{
    "T12": {"superior": [x1, y1], "inferior": [x2, y2]},
    "L1": {"superior": [x3, y3], "inferior": [x4, y4]},
    ...
}
```

#### `measure_listhesis_dynamic_xray`
```python
def measure_listhesis_dynamic_xray(
    volume_node_ids: dict[str, str],                        # REQUIRED - {"neutral": id, "flexion": id, "extension": id}
    landmarks_per_position: dict[str, dict[str, dict[str, list[float]]]],  # REQUIRED
    levels: list[str],                                      # REQUIRED - ["L4", "L5", "S1"]
    region: str = "lumbar",                                 # optional
    magnification_factor: float = 1.0                       # optional
) -> dict
```

**Example `landmarks_per_position` structure**:
```python
{
    "neutral": {
        "L4": {"superior": [x1, y1], "inferior": [x2, y2]},
        "L5": {"superior": [x3, y3], "inferior": [x4, y4]},
    },
    "flexion": {
        "L4": {"superior": [x5, y5], "inferior": [x6, y6]},
        ...
    },
    "extension": {
        ...
    }
}
```

---

## Conclusion

The workflow documentation is largely accurate and well-structured. The 6 issues identified are all correctable and fall into two categories:

1. **Parameter renames** (4 issues): Simple name mismatches that need documentation updates
2. **Structural/missing parameters** (2 issues): More complex—require additional steps (landmark placement) that aren't documented in current workflows

The X-ray-based workflows (`fracture-eval`, `instability-protocol`, `full-spine-workup`) have the most issues due to their reliance on manual landmark placement, which should be explicitly documented as a prerequisite step.

**No workflows reference non-existent tools.** All 27 unique tools referenced are properly registered in `server.py`.
