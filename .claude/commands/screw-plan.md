---
description: Cervical screw planning workflow. Requires CT, optionally CTA for vertebral artery safety.
---

# Cervical Screw Planning

Tier 2 workflow — adapt based on available modalities and target levels.

## Pipeline

### 1. Scene inventory
```
list_scene_nodes()
```
Identify CT and CTA volumes. Ask user which levels and techniques if not specified.

### 2. Segment cervical spine
```
segment_spine(input_node_id=<ct_id>, region="cervical")
```
Store `<seg_id>` from result.

### 3. Vertebral artery (if CTA available)
```
segment_vertebral_artery(input_node_id=<cta_id>)
```
Store `<va_id>` from result.

> **WARNING:** If planning C1-C2 transarticular screws without CTA, inform the user of vertebral artery injury risk and recommend obtaining CTA before proceeding.

### 4. Bone quality assessment
```
analyze_bone_quality(input_node_id=<ct_id>, segmentation_node_id=<seg_id>, region="cervical")
```
Flag osteoporotic levels — may need upsized screws or alternative fixation.

### 5. Plan screws — for each target level and side
```
plan_cervical_screws(
  technique=<technique>,
  level=<level>,
  segmentation_node_id=<seg_id>,
  side=<side>,
  va_node_id=<va_id_or_None>,
  variant=<variant_or_None>,
  screw_diameter_mm=<diameter>,
  screw_length_mm=<length>
)
```

**Techniques by level:**
| Technique | Levels | Default screw |
|-----------|--------|---------------|
| `"pedicle"` | C2-C7 | 3.5mm x 24mm |
| `"lateral_mass"` | C3-C7 | 3.5mm x 14mm |
| `"transarticular"` | C1-C2 | 3.5mm x 40mm |
| `"c1_lateral_mass"` | C1 | -- |
| `"c2_pars"` | C2 | -- |
| `"occipital"` | Occiput | -- |

**Lateral mass variants:** `"roy_camille"`, `"magerl"`, `"an"`, `"anderson"`

### 6. Document
```
capture_screenshot(view_type="3d")
capture_screenshot(view_type="axial")
```

## Decision rules

- **Comparing techniques** (e.g., pedicle vs lateral mass at C5): run `plan_cervical_screws` for each technique and compare safety margins.
- **VA clearance**: VA segmentation (`va_node_id`) is strongly recommended for transarticular and pedicle screws — verify >= 3mm clearance.
- **Parameter note**: `segment_vertebral_artery` uses `input_node_id` (NOT `volume_node_id`).
