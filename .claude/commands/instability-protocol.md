# /instability-protocol - Segmental Instability Assessment

Dynamic instability evaluation using flexion-extension X-rays.

## Description

Assesses segmental spinal instability by analyzing dynamic X-rays (neutral + flexion + extension). Measures translation and angular motion per level, applying White & Panjabi criteria. Optionally correlates with CT for structural assessment.

## Usage

```
/instability-protocol                # Evaluate all loaded dynamic X-rays
/instability-protocol L4-L5 L5-S1   # Focus on specific levels
```

## Instructions

When the user invokes this command:

### 1. Inventory available data

Call `list_scene_nodes()`. This protocol requires **3 X-ray volumes** (neutral, flexion, extension). CT is optional but adds structural information.

Identify which volume is which. If unclear, ask the user:
```
Dynamic instability assessment requires 3 lateral X-rays:
  1. Neutral standing
  2. Maximum flexion
  3. Maximum extension

Please identify which loaded volume corresponds to each position:
[list loaded volumes]
```

### 2. Measure dynamic listhesis

```
measure_listhesis_dynamic_xray(
    neutral_volume_id=<neutral_id>,
    flexion_volume_id=<flexion_id>,
    extension_volume_id=<extension_id>,
    levels=<levels_if_specified>
)
```

### 3. CT correlation (if available)

If CT is loaded:
```
segment_spine(input_node_id=<ct_id>)

measure_listhesis_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=<levels>
)
```

If spondylolysis is detected on dynamic X-ray:
```
detect_vertebral_fractures_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    region="lumbar"
)
```

And capture axial screenshots at the pars level:
```
capture_screenshot(view_type="axial")
```

### 4. Documentation

```
capture_screenshot(view_type="sagittal")   # Each position if possible
```

### 5. Generate instability report

```
SEGMENTAL INSTABILITY ASSESSMENT

DYNAMIC MEASUREMENTS:
  Level   | Position  | Translation | Trans %  | Seg Angle
  L4-L5   | Neutral   | 4.2mm       | 12.0%    | 10.5 deg
  L4-L5   | Flexion   | 8.1mm       | 23.1%    | 18.2 deg
  L4-L5   | Extension | 2.8mm       | 8.0%     | 5.1 deg

INSTABILITY ANALYSIS:
  Level   | Trans Range | Angle Range | Trans Unstable | Angle Unstable | Pattern
  L4-L5   | 5.3mm       | 13.1 deg    | YES (>4.5mm)   | YES (>11 deg)  | Combined
  L5-S1   | 1.2mm       | 6.3 deg     | NO             | NO             | Stable

Criteria applied: White & Panjabi
  - Lumbar translation threshold: 4.5mm
  - Lumbar angular motion threshold: 11 deg
  - Cervical translation threshold: 3.5mm
  - Cervical angular motion threshold: 11 deg

MEYERDING GRADING (maximum translation position):
  L4-L5: Grade I (23.1% in flexion)

SPONDYLOLYSIS:
  [Detected/Not detected]
  [If detected: bilateral/unilateral, pars gap mm]

CT CORRELATION (if available):
  Static listhesis: [values]
  Pars defect confirmation: [if applicable]
  Canal status: [if measured]

CLINICAL SUMMARY:
  Unstable levels: [list]
  Stable levels: [list]
  Pattern: [translational / angular / combined]
  Recommendation:
    - [Physical therapy / Facet block / Fusion]
    - [If spondylolysis + instability: "Consider direct pars repair vs fusion"]
```

## Notes

- Dynamic instability CANNOT be assessed on CT alone (static imaging)
- White & Panjabi criteria are the standard for defining instability
- Translation > 4.5mm OR angular motion > 11 deg = unstable (lumbar)
- Translation > 3.5mm OR angular motion > 11 deg = unstable (cervical subaxial)
- Spondylolysis (pars defect) is best confirmed on CT axial cuts
- Some authors use 15 deg angular threshold for lumbar — report both if borderline
- Always note that X-ray measurements may be affected by patient effort during flexion/extension
