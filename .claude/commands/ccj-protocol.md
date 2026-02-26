# /ccj-protocol - Craniocervical Junction Protocol

Specialized protocol for craniocervical junction (CCJ) evaluation.

## Description

Comprehensive CCJ assessment including craniometry, stability evaluation, vertebral artery analysis, and bone quality for surgical planning of C1-C2 fixation.

## Usage

```
/ccj-protocol               # Full CCJ protocol
/ccj-protocol stability     # Focus on instability markers
/ccj-protocol vascular      # Include vertebral artery analysis
```

## Instructions

When the user invokes this command:

### 1. Verify appropriate data

Call `list_scene_nodes()`. This protocol requires CT of the skull base + cervical spine. CTA is optional but recommended for vascular assessment.

If no CT is found, inform the user:
```
CCJ Protocol requires CT of skull base + cervical spine.
Please load the appropriate DICOM series first.
```

### 2. Segment cervical spine

```
segment_spine(input_node_id=<ct_volume_id>, region="cervical")
```

### 3. Measure CCJ angles

```
measure_ccj_angles(
    segmentation_node_id=<seg_id>,
    population="adult"
)
```

This measures:
- **CXA** (Clivo-axial angle): normal > 150 deg, < 135 deg = significant
- **ADI** (Atlanto-dental interval): normal < 3mm adults, > 3mm = instability
- **Powers ratio**: normal < 1.0, > 1.0 = anterior atlanto-occipital dislocation
- **Ranawat index**: normal > 15mm, < 13mm = cranial settling
- **McGregor line**: dens tip < 4.5mm above = normal
- **Chamberlain line**: dens tip < 3mm above = normal
- **Wackenheim line**: should be tangent to posterior dens

### 4. Vertebral artery assessment (if CTA or `$ARGUMENTS` includes "vascular")

If CTA volume is available:
```
segment_vertebral_artery(
    volume_node_id=<cta_volume_id>,
    segmentation_node_id=<seg_id>
)
```

Evaluate:
- VA course anomalies (high-riding VA)
- VA dominance (hypoplastic side)
- Safe corridor for C1 lateral mass screws
- Safe corridor for C2 pars/pedicle screws

### 5. Bone quality at C1-C2

```
analyze_bone_quality(
    volume_node_id=<ct_volume_id>,
    segmentation_node_id=<seg_id>,
    levels=["C1", "C2"]
)
```

### 6. Documentation

```
capture_screenshot(view_type="sagittal")   # CCJ measurements
capture_screenshot(view_type="3d")         # 3D reconstruction
capture_screenshot(view_type="axial")      # Axial at C1-C2
```

### 7. Generate CCJ report

```
CRANIOCERVICAL JUNCTION PROTOCOL REPORT

CRANIOMETRY:
  CXA: [value] deg  [NORMAL / ABNORMAL (< 150)]
  ADI: [value] mm   [NORMAL / INSTABILITY FLAG (> 3mm)]
  Powers ratio: [value]  [NORMAL / ABNORMAL (> 1.0)]
  Ranawat index: [value] mm  [NORMAL / CRANIAL SETTLING (< 13mm)]
  McGregor: [value] mm  [NORMAL / BASILAR INVAGINATION]
  Chamberlain: [value] mm  [NORMAL / BASILAR INVAGINATION]
  Wackenheim: [NORMAL / ABNORMAL]

INSTABILITY FLAGS:
  - ADI > 3mm: [YES/NO]
  - CXA < 150 deg: [YES/NO]
  - Powers > 1.0: [YES/NO]
  - Overall: [STABLE / POTENTIALLY UNSTABLE / UNSTABLE]

VASCULAR (if CTA available):
  - VA course: [normal / anomalous]
  - High-riding VA: [LEFT: yes/no] [RIGHT: yes/no]
  - VA dominance: [LEFT / RIGHT / CODOMINANT]
  - Safe screw corridors: [description]

BONE QUALITY:
  - C1 lateral mass: [HU value] [adequate/poor]
  - C2 body/pedicle: [HU value] [adequate/poor]
  - Fixation implications: [description]

CLINICAL SUMMARY:
  - Stability assessment: [summary]
  - Recommended fixation strategy: [if applicable]
  - Warnings: [high-riding VA, poor bone quality, etc.]
```

### 8. If `$ARGUMENTS` is "stability"

Focus output on instability markers only — skip vascular and bone quality sections.

## Notes

- CCJ pathology includes: basilar invagination, atlantoaxial instability (RA, Down syndrome, os odontoideum), Chiari malformation
- Always flag ADI > 3mm, CXA < 150 deg, and Powers > 1.0 as instability indicators
- High-riding VA is a CRITICAL finding that changes surgical technique (C1 lateral mass screw may be contraindicated)
- If only MRI is available (no CT), inform the user that CT is preferred for bony measurements
