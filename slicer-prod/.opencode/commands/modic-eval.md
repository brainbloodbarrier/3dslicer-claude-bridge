---
description: '# Modic Evaluation'
---

# Modic Evaluation

MRI-based Modic endplate and disc degeneration assessment.

**This workflow is available as an MCP tool.** Call it directly:

```
workflow_modic_eval(
    t1_volume_id=<t1_mrml_id>,
    t2_volume_id=<t2_mrml_id>,
    region="lumbar",                    # or "cervical", "thoracic"
    segmentation_node_id=<seg_id>,      # optional, auto-segments if omitted
    include_cord_screening=true         # cervical/thoracic only
)
```

Orchestrates: segment_spine → classify_modic_changes → assess_disc_degeneration_mri → detect_cord_compression_mri (if cervical/thoracic) → capture_screenshot.

Returns: modic_changes, pfirrmann_grades, cord_compression, screenshots.

**Prerequisite:** Both T1 and T2 MRI volumes must be loaded in the scene. Use `list_scene_nodes()` to identify them.
