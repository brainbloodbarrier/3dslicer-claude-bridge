# CCJ Protocol

Craniocervical junction assessment: craniometry, vertebral artery, bone quality.

**This workflow is available as an MCP tool.** Call it directly:

```
workflow_ccj_protocol(
    ct_volume_id=<ct_mrml_id>,
    segmentation_node_id=<seg_id>,      # optional, auto-segments if omitted
    cta_volume_id=<cta_id>,             # optional, for VA assessment
    population="adult",                 # or "child" (affects ADI threshold)
    include_bone_quality=true
)
```

Orchestrates: segment_spine → measure_ccj_angles → segment_vertebral_artery (if CTA) → analyze_bone_quality → capture_screenshot.

Returns: ccj_angles (CXA, ADI, Powers, BDI, BAI, Ranawat, McGregor, Chamberlain, Wackenheim), vertebral_artery, bone_quality, screenshots.

**Prerequisite:** CT volume loaded. CTA recommended for vertebral artery assessment.
