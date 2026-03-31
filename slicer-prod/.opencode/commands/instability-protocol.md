---
description: '# Instability Protocol'
---

# Instability Protocol

**DEPRECATED:** This workflow requires dynamic X-ray landmark placement which is not yet practical for automated orchestration.

The underlying tools exist but require manual landmark placement on 3 separate X-ray images (neutral, flexion, extension), which makes full automation impractical.

**For CT-based instability assessment**, use the individual tools directly:
- `measure_listhesis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)`
- `detect_vertebral_fractures_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)`

**For dynamic X-ray assessment** (advanced, requires manual landmarks):
1. Place landmarks on each X-ray (neutral, flexion, extension) via `place_landmarks()`
2. Retrieve via `get_landmarks()`
3. Call `measure_listhesis_dynamic_xray(volume_node_ids={"neutral": <id1>, "flexion": <id2>, "extension": <id3>}, landmarks_per_position=<nested_dict>, levels=<levels>)`
