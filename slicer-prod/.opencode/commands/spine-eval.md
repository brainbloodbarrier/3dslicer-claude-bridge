---
description: 'General pre-operative spine evaluation. Adapts the pipeline based on available imaging modalities (CT, MRI, X-ray).'
---

General pre-operative spine evaluation. Adapts the pipeline based on available imaging modalities (CT, MRI, X-ray).

## Steps

1. `list_scene_nodes()` -- identify all loaded volumes and determine available modalities (CT, MRI T1, MRI T2, X-ray).

2. `segment_spine(input_node_id=<ct_or_mri_id>, region=<region>)` -- segment vertebrae from the primary volume. Reuse this segmentation across all subsequent tools.

3. `capture_screenshot(view_type="sagittal")` -- baseline sagittal view before analysis.

4. `measure_spine_alignment(segmentation_node_id=<seg_id>, region=<region>)` -- measure sagittal alignment parameters (lordosis, kyphosis, etc.).

5. If CT available:
   - `detect_vertebral_fractures_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, region=<region>)` -- screen for vertebral fractures.
   - `assess_osteoporosis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)` -- evaluate bone density from CT.

6. If MRI T1 + T2 available:
   - `classify_modic_changes(t1_node_id=<t1_id>, t2_node_id=<t2_id>, region=<region>, segmentation_node_id=<seg_id>)` -- classify endplate changes (Type I/II/III).

7. If MRI T2 available:
   - `assess_disc_degeneration_mri(t2_node_id=<t2_id>, region=<region>, segmentation_node_id=<seg_id>)` -- Pfirrmann grading of disc degeneration.

8. If cervical or thoracic MRI T2 available:
   - `detect_cord_compression_mri(t2_node_id=<t2_id>, region=<region>, segmentation_node_id=<seg_id>)` -- assess spinal cord compression.

9. If lateral X-ray available:
   - Guide user to place landmarks on vertebral endplates via `place_landmarks()`.
   - `get_landmarks(node_id=<landmark_node_id>)` -- retrieve placed landmarks.
   - `measure_sagittal_balance_xray(volume_node_id=<xray_id>, landmarks=<landmarks>)` -- sagittal vertical axis, pelvic incidence, etc.

10. If AP X-ray available:
    - Guide user to place landmarks, then retrieve via `get_landmarks(node_id=<landmark_node_id>)`.
    - `measure_coronal_balance_xray(volume_node_id=<xray_id>, landmarks=<landmarks>)` -- coronal balance and Cobb angle.

11. `analyze_bone_quality(input_node_id=<ct_id>, segmentation_node_id=<seg_id>, region=<region>)` -- cortical/trabecular bone quality assessment.

12. `capture_screenshot(view_type="sagittal")` -- final sagittal view documenting findings.

## Notes

- Start by identifying modalities from `list_scene_nodes()` and adapt the pipeline accordingly. Not all steps apply to every case.
- X-ray measurements require manual landmark placement first -- prompt the user to place landmarks before calling measurement tools.
- Reuse the segmentation node from step 2 across all CT and MRI tools to maintain consistency.
- For region, use anatomical labels: `"cervical"`, `"thoracic"`, `"lumbar"`, `"thoracolumbar"`, or `"full"`.
