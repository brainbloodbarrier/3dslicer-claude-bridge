---
description: 'Vertebral fracture evaluation. CT path is fully automated via TotalSegmentator. X-ray path requires manual landmark placement.'
---

Vertebral fracture evaluation. CT path is fully automated via TotalSegmentator. X-ray path requires manual landmark placement.

## Steps

1. `list_scene_nodes()` -- identify loaded volumes and determine if CT or X-ray is available.

2. If CT available:
   - `segment_spine(input_node_id=<ct_id>, region=<region>)` -- segment vertebrae for fracture detection.

3. **CT path (automated):**
   - `detect_vertebral_fractures_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, region=<region>, classification_system="ao_spine")` -- detect and classify fractures using AO Spine system.
   - `assess_osteoporosis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)` -- evaluate bone density to contextualize fracture risk.
   - `measure_spinal_canal_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)` -- measure canal dimensions for stenosis assessment.

4. **X-ray path (requires landmark placement):**
   - Guide the user to identify vertebral endplate corners on the X-ray image. Each vertebra needs superior and inferior endplate corner coordinates.
   - `place_landmarks(name="fracture_landmarks", points=<points>, labels=<labels>)` -- place superior and inferior endplate corners for each vertebra of interest.
   - `get_landmarks(node_id=<landmark_node_id>)` -- retrieve the placed landmark coordinates.
   - `detect_vertebral_fractures_xray(volume_node_id=<xray_id>, landmarks_per_vertebra=<landmarks_dict>, magnification_factor=1.0)` -- detect fractures using vertebral height ratios.
   - The `landmarks_per_vertebra` structure is: `{"T12": {"superior": [x, y], "inferior": [x, y]}, "L1": {"superior": [x, y], "inferior": [x, y]}}`.

5. `capture_screenshot(view_type="sagittal")` -- document findings with a final screenshot.

## Notes

- CT path is fully automated -- segmentation via TotalSegmentator feeds directly into fracture detection and canal measurement.
- X-ray path requires the user to manually place anatomical landmarks before analysis can proceed. Prompt them clearly with which points are needed.
- Genant semi-quantitative grading scale: Grade 0 (normal), Grade 1 (mild, 20-25% height loss), Grade 2 (moderate, 25-40% height loss), Grade 3 (severe, >40% height loss).
- AO Spine classification is only available on the CT path.
- If both CT and X-ray are available, prefer the CT path for higher diagnostic accuracy.
