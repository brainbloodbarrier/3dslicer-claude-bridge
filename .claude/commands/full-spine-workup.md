---
description: Comprehensive multi-modality spine assessment. Superset of all other workflows.
---

# Full Spine Workup

Tier 2 workflow — adapt the pipeline based on available modalities. Skip sections for unavailable data.

## Pipeline

### 1. Scene inventory
```
list_scene_nodes()
```
Identify ALL available volumes: CT, MRI (T1, T2, STIR), standing X-rays, dynamic X-rays.

---

## CT Assessment (if CT available)

### 2. Full spine segmentation
```
segment_spine(input_node_id=<ct_id>, region="full", include_discs=True)
```
Store `<seg_id>`. Reuse across ALL subsequent tools — do NOT segment twice.

### 3. Sagittal balance
```
measure_spine_alignment(segmentation_node_id=<seg_id>, region="full")
```

### 4. Fracture detection
```
detect_vertebral_fractures_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, region="full")
```

### 5. Osteoporosis assessment
```
assess_osteoporosis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```

### 6. Metastatic screening
```
detect_metastatic_lesions_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, region="full")
```

### 7. SINS score (if metastatic lesions found)
```
calculate_sins_score(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```

### 8. Listhesis measurement
```
measure_listhesis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```

### 9. Spinal canal measurement
```
measure_spinal_canal_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```

### 10. Bone quality
```
analyze_bone_quality(input_node_id=<ct_id>, segmentation_node_id=<seg_id>, region="full")
```

---

## MRI Assessment (if MRI T1 + T2 available)

### 11. Modic changes
```
classify_modic_changes(t1_node_id=<t1_id>, t2_node_id=<t2_id>, region=<region>, segmentation_node_id=<seg_id>)
```

### 12. Disc degeneration
```
assess_disc_degeneration_mri(t2_node_id=<t2_id>, region=<region>, segmentation_node_id=<seg_id>)
```

### 13. Cord compression (if cervical or thoracic)
```
detect_cord_compression_mri(t2_node_id=<t2_id>, region=<region>, segmentation_node_id=<seg_id>)
```

### 14. Metastatic lesions (if oncologic concern)
```
detect_metastatic_lesions_mri(t1_node_id=<t1_id>, t2_stir_node_id=<t2_id>, region=<region>, segmentation_node_id=<seg_id>)
```

---

## X-ray Assessment (if standing X-rays available)

X-ray tools require manual landmark placement. Guide user to place landmarks on lateral X-ray (C2-S1 vertebral corners, femoral heads).

### 15-16. Landmark placement
```
place_landmarks(name="sagittal_landmarks", points=<points>, labels=<labels>)
```

### 17. Retrieve landmarks
```
get_landmarks(node_id=<landmark_id>)
```

### 18. Sagittal balance
```
measure_sagittal_balance_xray(volume_node_id=<xray_lat_id>, landmarks=<landmarks>)
```

### 19. Coronal balance (if AP X-ray available)
```
measure_coronal_balance_xray(volume_node_id=<xray_ap_id>, landmarks=<ap_landmarks>)
```

### 20. Cobb angle (if scoliosis suspected)
```
measure_cobb_angle_xray(volume_node_id=<xray_ap_id>, landmarks=<landmarks>, upper_end_vertebra=<upper>, lower_end_vertebra=<lower>)
```

### 21. Dynamic listhesis (if neutral/flex/ext X-rays available)
```
measure_listhesis_dynamic_xray(
  volume_node_ids={"neutral": <id1>, "flexion": <id2>, "extension": <id3>},
  landmarks_per_position=<nested_landmarks>,
  levels=<levels>
)
```
Place landmarks on each position separately before calling. Parameters: `volume_node_ids` (dict), `landmarks_per_position` (nested dict), `levels` (list).

---

## Documentation

### 22-23. Screenshots
```
capture_screenshot(view_type="sagittal")
capture_screenshot(view_type="3d")
```

## Decision rules

- **Reuse segmentation**: pass the same `<seg_id>` to every tool. Never re-segment.
- **SINS score**: only run `calculate_sins_score` if `detect_metastatic_lesions_ct` found lesions.
- **Cord compression**: only relevant for cervical and thoracic regions.
- **X-ray landmarks**: these are manual — guide the user through placement before calling measurement tools.
