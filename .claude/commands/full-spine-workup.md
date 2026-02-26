# /full-spine-workup - Comprehensive Spine Workup

Complete multi-modal spine evaluation using all available imaging data.

## Description

Runs the appropriate analysis pipeline for every loaded modality (CT, X-ray lateral, X-ray AP, dynamic X-rays, MRI). Generates an integrated report combining findings across all modalities, organized by vertebral level.

## Usage

```
/full-spine-workup                  # Analyze everything available
/full-spine-workup --oncologic      # Include oncologic screening
```

## Instructions

When the user invokes this command:

### 1. Complete data inventory

Call `list_scene_nodes()` and classify every loaded volume by modality:

```
FULL SPINE WORKUP - Data Inventory

CT volumes: [list or "none"]
MRI T1: [found/none]
MRI T2: [found/none]
X-ray lateral: [list or "none"]
X-ray AP: [list or "none"]
Dynamic X-rays: [neutral/flexion/extension available?]
Segmentations: [list or "none"]

Pipelines to run:
  [x] CT structural analysis
  [x] X-ray sagittal balance
  [ ] X-ray coronal balance (no AP X-ray)
  [x] MRI disc/endplate assessment
  [ ] Dynamic instability (no flex/ext X-rays)
```

### 2. CT Pipeline (if CT available)

#### 2a. Segmentation
```
segment_spine(input_node_id=<ct_id>)
```

#### 2b. Fracture detection
```
detect_vertebral_fractures_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, classification_system="all")
```

#### 2c. Osteoporosis screening
```
assess_osteoporosis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, levels=["L1"])
```

#### 2d. Oncologic screening (if `$ARGUMENTS` includes "--oncologic" or lesions detected)
```
detect_metastatic_lesions_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```
If lesions found:
```
calculate_sins_score(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>, target_levels=<affected>)
```

#### 2e. Listhesis
```
measure_listhesis_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```

#### 2f. Canal morphometry
```
measure_spinal_canal_ct(volume_node_id=<ct_id>, segmentation_node_id=<seg_id>)
```

### 3. X-ray Lateral Pipeline (if lateral X-ray available)

```
measure_sagittal_balance_xray(volume_node_id=<xray_lat_id>)
detect_vertebral_fractures_xray(volume_node_id=<xray_lat_id>)
```

### 4. X-ray AP Pipeline (if AP X-ray available)

```
measure_coronal_balance_xray(volume_node_id=<xray_ap_id>)
measure_cobb_angle_xray(volume_node_id=<xray_ap_id>)
```

### 5. Dynamic X-ray Pipeline (if flex/ext available)

```
measure_listhesis_dynamic_xray(
    neutral_volume_id=<neutral_id>,
    flexion_volume_id=<flexion_id>,
    extension_volume_id=<extension_id>
)
```

### 6. MRI Pipeline (if T1 + T2 available)

```
classify_modic_changes(t1_volume_id=<t1_id>, t2_volume_id=<t2_id>, segmentation_node_id=<seg_id>)
assess_disc_degeneration_mri(t2_volume_id=<t2_id>, segmentation_node_id=<seg_id>)
detect_cord_compression_mri(t2_volume_id=<t2_id>, t1_volume_id=<t1_id>, segmentation_node_id=<seg_id>)
```

If oncologic screening requested and MRI available:
```
detect_metastatic_lesions_mri(t1_volume_id=<t1_id>, t2_or_stir_volume_id=<t2_id>, segmentation_node_id=<seg_id>)
```

### 7. Documentation screenshots

```
capture_screenshot(view_type="sagittal")
capture_screenshot(view_type="coronal")
capture_screenshot(view_type="axial")
capture_screenshot(view_type="3d")
```

### 8. Integrated report

```
COMPREHENSIVE SPINE WORKUP

Modalities analyzed: [CT, X-ray lateral, X-ray AP, Dynamic X-ray, MRI T1+T2]
Region: [full spine / specific region]

LEVEL-BY-LEVEL SUMMARY:
  Level | Fracture    | Listhesis | Disc (Pf) | Modic | Stenosis | Lesion | Notes
  T12   | Genant 1    | -         | -         | -     | None     | -      |
  L1    | -           | -         | -         | -     | None     | -      | HU: 92 (osteoporosis)
  L2    | -           | -         | III       | 0     | Mild     | -      |
  L3    | -           | -         | III       | 0     | None     | -      |
  L4    | -           | Gr I 12%  | IV        | I     | Moderate | -      | Pain source?
  L5    | -           | -         | V         | II    | Severe   | -      |
  S1    | -           | -         | -         | 0     | -        | -      |

GLOBAL ALIGNMENT:
  Sagittal:
    SVA: [mm] [status]
    PI-LL mismatch: [deg] [status]
    SRS-Schwab: [modifiers]
    Roussouly type: [type]
  Coronal (if measured):
    C7-CSVL: [mm] [status]
    Cobb angle: [deg] [if scoliosis]

DYNAMIC INSTABILITY (if measured):
  [Per level: stable/unstable, criteria]

BONE QUALITY:
  L1 HU: [value] - [classification]
  Implications: [fixation considerations]

ONCOLOGIC (if screened):
  Lesions: [count, levels, types]
  SINS: [scores per level]

KEY FINDINGS (prioritized):
  1. [Most critical finding]
  2. [Second most critical]
  3. [...]

RECOMMENDATIONS:
  - [Prioritized clinical recommendations]
  - [Additional imaging if needed]
  - [Surgical considerations]
```

## Notes

- This is the most comprehensive skill — run time depends on number of modalities
- Adapt pipeline to available data: don't error if only one modality is loaded
- The level-by-level summary is the most valuable output — correlates all modalities
- For purely oncologic cases, `/onco-spine` is more focused
- For purely degenerative disc disease, `/modic-eval` is more appropriate
- Use this when the clinical picture is complex or multiple pathologies coexist
