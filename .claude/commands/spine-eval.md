# /spine-eval - Spine Pre-operative Evaluation

Automated pre-operative spine evaluation workflow.

## Description

Complete spine evaluation combining segmentation, alignment measurement, and clinical classification. Adapts pipeline based on available data (CT, MRI, or X-ray).

## Usage

```
/spine-eval                  # Full evaluation with all loaded data
/spine-eval cervical         # Focus on cervical region
/spine-eval lumbar           # Focus on lumbar region
```

## Instructions

When the user invokes this command:

### 1. Inventory loaded data

Call `list_scene_nodes()` to identify:
- Volume nodes (CT, MRI T1/T2, X-ray)
- Existing segmentations
- Existing markup/landmark nodes

Display what was found:
```
Spine Evaluation - Data Inventory

Volumes found:
- [node_id]: [description/name] ([modality if detectable])

Segmentations: [list or "none"]
```

### 2. Segment spine (if needed)

If no spine segmentation exists:
```
segment_spine(input_node_id=<volume_id>, region=<region_or_full>)
```

If `$ARGUMENTS` specifies a region ("cervical", "thoracic", "lumbar"), use it. Otherwise use "full".

### 3. Capture initial screenshot

```
capture_screenshot(view_type="sagittal")
```

Analyze the image to orient yourself anatomically.

### 4. Measure alignment

```
measure_spine_alignment(segmentation_id=<seg_id>, region=<region>)
```

### 5. CCJ assessment (if cervical)

If the region includes cervical spine (region is "cervical" or "full"):
```
measure_ccj_angles(segmentation_node_id=<seg_id>)
```

### 6. X-ray measurements (if X-ray loaded)

If a lateral X-ray is available:
```
measure_sagittal_balance_xray(volume_node_id=<xray_id>)
```

If an AP X-ray is available:
```
measure_coronal_balance_xray(volume_node_id=<xray_id>)
```

### 7. Bone quality (if CT)

If CT data is available:
```
analyze_bone_quality(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```

### 8. Documentation screenshots

```
capture_screenshot(view_type="3d")
capture_screenshot(view_type="axial")
```

### 9. Generate structured report

Compile all measurements into a structured report:

```
SPINE PRE-OPERATIVE EVALUATION REPORT

Patient data: [from loaded volumes if available]
Modalities analyzed: [CT / MRI / X-ray]
Region: [cervical / thoracic / lumbar / full]

ALIGNMENT:
  - Sagittal balance: [SVA, PI-LL mismatch, etc.]
  - Coronal balance: [if measured]
  - Classification: [Roussouly type, SRS-Schwab modifiers]

CCJ (if applicable):
  - CXA: [value] ([normal/abnormal])
  - ADI: [value] ([normal/abnormal])
  - Powers ratio: [value]
  - [other CCJ measurements]

BONE QUALITY:
  - HU values: [per level]
  - Osteoporosis screening: [result]
  - Implications for fixation: [cement augmentation flag]

SCREENSHOTS: [references to captured images]

CLINICAL SUMMARY:
  - Key findings: [prioritized list]
  - Suggested next steps: [based on findings]
```

## Notes

- This skill orchestrates multiple MCP tools in sequence
- Always start with `list_scene_nodes()` to understand what data is available
- Adapt the pipeline to available modalities (don't call X-ray tools if no X-ray loaded)
- If segmentation fails, report the error and suggest loading appropriate data
- All angle measurements follow signed conventions where clinically appropriate
