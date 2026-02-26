# /modic-eval - Modic Changes & Disc Degeneration Assessment

MRI evaluation of Modic endplate changes and disc degeneration.

## Description

Classifies Modic endplate changes (Types 0-III) and Pfirrmann disc degeneration grades using T1 and T2-weighted MRI sequences. Includes cord compression screening for cervical and thoracic regions.

## Usage

```
/modic-eval                 # Full Modic + Pfirrmann assessment
/modic-eval lumbar          # Focus on lumbar region
```

## Instructions

When the user invokes this command:

### 1. Verify MRI data

Call `list_scene_nodes()`. This protocol requires **T1 and T2** weighted MRI volumes.

If missing:
```
Modic evaluation requires both T1 and T2 weighted MRI sequences.
Please load the appropriate DICOM series.

Currently loaded:
  T1: [found/missing]
  T2: [found/missing]
```

### 2. Segment spine (MRI mode)

```
segment_spine(input_node_id=<t2_volume_id>, region=<region>)
```

### 3. Classify Modic changes

```
classify_modic_changes(
    t1_volume_id=<t1_id>,
    t2_volume_id=<t2_id>,
    segmentation_node_id=<seg_id>
)
```

Modic classification:
- **Type 0**: Normal endplate signal
- **Type I**: T1 hypointense + T2 hyperintense (edema/inflammation — ACTIVE, symptomatic)
- **Type II**: T1 hyperintense + T2 iso/hyperintense (fatty replacement — stable)
- **Type III**: T1 hypointense + T2 hypointense (sclerosis)
- **Mixed I/II**: Transitional pattern

### 4. Assess disc degeneration (Pfirrmann)

```
assess_disc_degeneration_mri(
    t2_volume_id=<t2_id>,
    segmentation_node_id=<seg_id>
)
```

Pfirrmann grading (T2-weighted):
- **Grade I**: Homogeneous, hyperintense, normal height
- **Grade II**: Heterogeneous with horizontal band, hyperintense, normal height
- **Grade III**: Heterogeneous gray, intermediate signal, normal to slight height loss
- **Grade IV**: Heterogeneous dark gray, hypo to intermediate signal, moderate height loss
- **Grade V**: Heterogeneous black, hypointense, collapsed

### 5. Cord compression screening (cervical/thoracic)

If the region includes cervical or thoracic spine:
```
detect_cord_compression_mri(
    t2_volume_id=<t2_id>,
    t1_volume_id=<t1_id>,
    segmentation_node_id=<seg_id>
)
```

### 6. Documentation

```
capture_screenshot(view_type="sagittal")   # T2 sagittal (disc signal)
```

### 7. Generate Modic/disc report

```
MODIC & DISC DEGENERATION ASSESSMENT (MRI)

Sequences: T1 + T2
Region: [analyzed region]
Registration: [applied/not needed]

MODIC CHANGES:
  Level   | Sup EP    | Inf EP    | Type   | Depth  | Area % | Distribution
  L4-L5   | L5 sup: I | L4 inf: I | I      | 8.2mm  | 75%    | Diffuse
  L5-S1   | S1 sup: II| L5 inf: 0 | II(asym)| 5.1mm | 40%    | Anterior

  EP = endplate, sup = superior, inf = inferior

DISC DEGENERATION (PFIRRMANN):
  Level   | Grade | Signal    | Nucleus/Annulus | Height    | Notes
  L3-L4   | II    | Hyper     | Clear           | Normal    |
  L4-L5   | IV    | Hypo-Int  | Lost            | Moderate loss |
  L5-S1   | V     | Hypo      | Lost            | Collapsed |

CORD COMPRESSION (if cervical/thoracic):
  Level   | AP diam | Compression | T2 signal    | T1 signal   | Grade
  [...]   | [mm]    | [ratio]     | [normal/hyper]| [normal/hypo]| [mild/mod/severe]

  T2 hyperintensity + T1 normal = potentially reversible myelopathy
  T2 hyperintensity + T1 hypointensity = likely irreversible (gliosis)

CLINICAL CORRELATIONS:
  Pain source analysis:
  - Modic I + Pfirrmann IV-V at [level]: HIGH probability discogenic pain source
  - Modic II at [level]: MODERATE probability (stable degeneration)
  - Modic I may convert to Modic II over 1-3 years

  Most symptomatic level: [level] (Modic [type] + Pfirrmann [grade])

  If cord compression present:
  - mJOA assessment recommended
  - Surgical decompression may be indicated if myelopathy signs present
```

## Notes

- Modic I is the most clinically relevant type (active inflammation, correlates with pain)
- Modic I at same level as Pfirrmann IV-V is a strong predictor of discogenic pain
- Signal normalization is critical — absolute MRI signal values vary between scanners
- STIR can be used instead of T2 for better sensitivity to edema (Modic I)
- Reference signal uses median of all healthy vertebral body signals
- Disc assessment (Pfirrmann) is only validated for lumbar region
