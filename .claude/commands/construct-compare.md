# Construct Comparison

**DEPRECATED:** This workflow has been merged into `screw-plan`. Use `/screw-plan` instead.

To compare screw techniques, run `plan_cervical_screws()` with different `technique` parameters for the same level and compare the safety margins, entry points, and trajectory angles in the results.

Example: Compare pedicle vs lateral mass at C5:
1. `plan_cervical_screws(technique="pedicle", level="C5", segmentation_node_id=<seg_id>, side="bilateral")`
2. `plan_cervical_screws(technique="lateral_mass", level="C5", segmentation_node_id=<seg_id>, side="bilateral", variant="magerl")`
3. Compare: safety margins, VA clearance, screw dimensions, trajectory angles
