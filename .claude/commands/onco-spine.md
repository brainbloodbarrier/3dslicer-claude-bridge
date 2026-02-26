# /onco-spine - Oncologic Spine Assessment

Comprehensive assessment of metastatic spinal disease with SINS scoring.

## Description

Detects metastatic lesions in the spine, calculates SINS (Spinal Instability Neoplastic Score), and provides treatment recommendations. Combines CT (required) and MRI (optional, improves accuracy) data.

## Usage

```
/onco-spine                 # Full oncologic spine assessment
/onco-spine thoracic        # Focus on thoracic region
```

## Instructions

When the user invokes this command:

### 1. Inventory available data

Call `list_scene_nodes()`. CT is required. MRI (T1 + T2/STIR) is optional but improves lesion characterization and benign/malignant differentiation.

If no CT is found:
```
Oncologic spine assessment requires CT data.
MRI is optional but recommended for improved lesion characterization.
Please load CT DICOM series first.
```

### 2. Segment spine

```
segment_spine(input_node_id=<ct_id>, region=<region_or_full>)
```

### 3. Detect metastatic lesions on CT

```
detect_metastatic_lesions_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    include_posterior_elements=true,
    include_sins=true
)
```

### 4. MRI assessment (if available)

If MRI T1 and T2/STIR volumes are loaded:
```
detect_metastatic_lesions_mri(
    t1_volume_id=<t1_id>,
    t2_or_stir_volume_id=<t2_id>,
    segmentation_node_id=<seg_id>
)
```

MRI adds:
- Better soft tissue characterization
- Epidural disease assessment
- Cord compression evaluation
- Benign vs malignant fracture differentiation

### 5. SINS score calculation

For each level with detected lesions:
```
calculate_sins_score(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    target_levels=<affected_levels>
)
```

**Note**: SINS includes a clinical pain component (0-3 points). If not provided by the user, the tool reports the score range (minimum without pain, maximum with mechanical pain). Ask the user:
```
For complete SINS scoring, please provide:
- Does the patient have pain at the affected level(s)? (yes/no)
- If yes, is it mechanical (worse with movement) or non-mechanical?
```

### 6. Additional assessments for unstable disease (SINS >= 7)

If SINS >= 7 (indeterminate or unstable):
```
measure_listhesis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, levels=<affected>)
measure_spinal_canal_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, levels=<affected>)
```

### 7. Osteoporosis screening (uninvolved levels)

```
assess_osteoporosis_ct(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=<uninvolved_levels>
)
```

### 8. Documentation

```
capture_screenshot(view_type="sagittal")
capture_screenshot(view_type="axial")       # At most affected level
capture_screenshot(view_type="3d")
```

### 9. Generate oncologic spine report

```
ONCOLOGIC SPINE ASSESSMENT

LESION MAP:
  Level | Type     | Volume   | Body %  | Posterior | Canal  | Epidural
  T8    | Lytic    | 2845mm3  | 35%     | L pedicle | 12%    | Suspected
  L2    | Blastic  | 1691mm3  | 25%     | None      | 0%     | None
  [...]

SINS SCORING (per affected level):
  T8:
    Location: [score] - [rationale]
    Pain: [score or range] - [source: clinical/not provided]
    Lesion type: [score] - [rationale]
    Alignment: [score] - [rationale]
    Collapse: [score] - [rationale]
    Posterolateral: [score] - [rationale]
    TOTAL: [score] / 18
    Classification: [STABLE / INDETERMINATE / UNSTABLE]

  L2: [same format]

MRI FINDINGS (if available):
  - Cord compression: [present/absent, grade if present]
  - Epidural disease: [description]
  - Benign vs malignant differentiation: [for levels with collapse]

BONE QUALITY (uninvolved levels):
  [HU values and osteoporosis screening]

CLINICAL SUMMARY:
  - Total affected levels: [count]
  - Most critical level: [level] (SINS [score])
  - Stability: [STABLE / INDETERMINATE / UNSTABLE]
  - Canal compromise: [present at levels X, Y]
  - Recommendations:
    * [Observation / Radiation / Stabilization / Decompression]
    * [If SINS >= 13: "Surgical stabilization likely indicated"]
    * [If epidural disease: "Consider decompression"]
    * [If cord compression on MRI: "URGENT decompression evaluation"]
```

## Notes

- SINS 0-6 = Stable (conservative treatment)
- SINS 7-12 = Indeterminate (surgical consultation)
- SINS 13-18 = Unstable (surgical intervention likely)
- MRI is superior to CT for detecting epidural disease and cord compression
- Always separate imaging-derived SINS components from clinical components
- Pathological fractures need differentiation from osteoporotic fractures (MRI helps)
- Full-spine MRI analysis may take longer due to the number of levels
