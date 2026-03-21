# Oncologic Spine Assessment

Lesion detection, SINS scoring, stability analysis for metastatic spine disease.

**This workflow is available as an MCP tool.** Call it directly:

```
workflow_onco_spine(
    ct_volume_id=<ct_mrml_id>,
    region="full",                      # or "cervical", "thoracic", "lumbar"
    t1_volume_id=<t1_id>,               # optional, for enhanced MRI detection
    t2_volume_id=<t2_id>,               # optional, both T1+T2 needed for MRI
    segmentation_node_id=<seg_id>,      # optional, auto-segments if omitted
    pain_type="mechanical"              # or "occasional_non_mechanical", "pain_free", null
)
```

Orchestrates: segment_spine → detect_metastatic_lesions_ct → calculate_sins_score → measure_listhesis_ct → measure_spinal_canal_ct → assess_osteoporosis_ct → detect_metastatic_lesions_mri (if MRI) → capture_screenshot.

Returns: metastatic_lesions_ct, sins_scores, listhesis, canal_stenosis, bone_quality, metastatic_lesions_mri, screenshots.

**Prerequisite:** CT volume loaded. MRI T1+T2 optional for enhanced lesion characterization.
