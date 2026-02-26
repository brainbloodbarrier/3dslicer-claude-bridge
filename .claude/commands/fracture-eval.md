# /fracture-eval - Vertebral Fracture Evaluation

Automated vertebral fracture detection and classification.

## Description

Detects and classifies vertebral fractures using CT and/or X-ray data. Applies AO Spine, Denis, and Genant classification systems. Includes osteoporosis screening and canal compromise assessment.

## Usage

```
/fracture-eval              # Auto-detect modality and evaluate
/fracture-eval ct           # CT-specific pipeline
/fracture-eval xray         # X-ray-specific pipeline
```

## Instructions

When the user invokes this command:

### 1. Inventory available data

Call `list_scene_nodes()` to identify CT volumes, X-ray volumes, and existing segmentations.

Determine modality:
- If `$ARGUMENTS` specifies "ct" or "xray", use that pipeline
- If not specified, auto-detect from loaded volumes
- If both CT and X-ray available, run CT pipeline (more comprehensive)

### 2. CT Pipeline

#### 2a. Segment spine
```
segment_spine(input_node_id=<ct_id>)
```

#### 2b. Detect fractures
```
detect_vertebral_fractures_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    classification_system="all"
)
```

#### 2c. Osteoporosis screening
```
assess_osteoporosis_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=["L1"]
)
```

#### 2d. Canal assessment (if burst fracture detected)

If any fracture has AO type A3, A4, B, or C (posterior wall or canal involvement):
```
measure_spinal_canal_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=<fractured_levels>
)
```

#### 2e. Screenshots
```
capture_screenshot(view_type="sagittal")   # Sagittal at fracture level
capture_screenshot(view_type="axial")      # Axial at fracture level
```

#### 2f. CT Report
```
VERTEBRAL FRACTURE EVALUATION (CT)

FRACTURES DETECTED: [count]

Per level:
  [LEVEL]:
    Genant: Grade [0-3] ([morphology]) - [height loss %]
    AO Spine: [Type] - [description]
    Denis: [stability status]
    Canal compromise: [% if applicable]
    Posterior elements: [status]

OSTEOPOROSIS SCREENING:
  L1 trabecular HU: [value]
  Classification: [normal/osteopenia/osteoporosis]
  Screw pullout risk: [LOW/MODERATE/HIGH]

CANAL STATUS (if burst):
  Retropulsion: [mm]
  Canal compromise: [%]
  Neurological risk: [assessment]

CLINICAL SUMMARY:
  Most severe: [level] [AO type]
  Stability: [STABLE/UNSTABLE]
  Recommendation: [conservative/surgical]
  If osteoporotic: [cement augmentation consideration]
```

### 3. X-ray Pipeline

#### 3a. Detect fractures
```
detect_vertebral_fractures_xray(
    volume_node_id=<xray_id>,
    region="thoracolumbar"
)
```

#### 3b. Screenshots
```
capture_screenshot(view_type="sagittal")
```

#### 3c. X-ray Report
```
VERTEBRAL FRACTURE EVALUATION (X-ray)

LIMITATIONS: X-ray provides 2D assessment only. Sensitivity is lower than CT,
especially for T1-T8 (rib overlap). CT recommended for Genant >= 2.

FRACTURES DETECTED: [count]

Per level:
  [LEVEL]:
    Genant: Grade [0-3] ([morphology]) - [height loss %]
    Confidence: [high/moderate/low based on visibility]

RECOMMENDATION:
  - CT recommended: [YES/NO]
  - Reason: [if yes, explain why]
```

## Notes

- CT is the gold standard for fracture classification (3D data, posterior elements, canal)
- X-ray is limited to Genant semi-quantitative grading (no AO Spine, no Denis)
- Always recommend CT if X-ray shows Genant grade >= 2 (moderate)
- Burst fractures (AO A3/A4) ALWAYS need canal assessment
- Osteoporosis screening is opportunistic (not equivalent to DXA)
- For pathological fractures (suspected tumor), suggest `/onco-spine` instead
