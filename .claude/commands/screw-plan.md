# /screw-plan - Cervical Screw Planning

Instrumentation planning for cervical spine fixation.

## Description

Plans cervical pedicle/lateral mass/pars screw trajectories with safety assessment. Integrates vertebral artery evaluation for C1-C2 constructs.

## Usage

```
/screw-plan C1-C2                    # Plan C1-C2 fixation (auto technique)
/screw-plan C3-C6 lateral_mass       # Specific technique
/screw-plan C1-C2 harms              # Harms technique specifically
```

## Instructions

When the user invokes this command:

### 1. Parse arguments

Extract from `$ARGUMENTS`:
- **levels**: vertebral levels for instrumentation (e.g., "C1-C2", "C3-C6")
- **technique** (optional): "auto", "harms", "magerl", "lateral_mass", "pedicle"

If no levels specified, ask the user which levels to instrument.

### 2. Verify data and segmentation

Call `list_scene_nodes()`. Requires CT with segmentation.

If no segmentation:
```
segment_spine(input_node_id=<ct_id>, region="cervical")
```

### 3. Vertebral artery check (MANDATORY for C1-C2)

If levels include C1 or C2:
```
segment_vertebral_artery(
    volume_node_id=<cta_volume_id>,
    segmentation_node_id=<seg_id>
)
```

**CRITICAL**: If no CTA is available and C1-C2 is involved, WARN the user:
```
WARNING: C1-C2 instrumentation planned without CTA.
Vertebral artery assessment is MANDATORY before C1-C2 screw placement.
Please load CTA data or confirm proceeding without VA assessment.
```

Do NOT proceed with C1-C2 planning without user confirmation if no CTA.

### 4. Plan screws

```
plan_cervical_screws(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=<parsed_levels>,
    technique=<technique_or_auto>,
    va_segmentation_id=<va_seg_id_if_available>
)
```

### 5. Bone quality per level

```
analyze_bone_quality(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=<levels_list>
)
```

### 6. Documentation screenshots

```
capture_screenshot(view_type="3d")       # Full construct visualization
capture_screenshot(view_type="axial")    # Entry points per level
capture_screenshot(view_type="sagittal") # Lateral view of construct
```

### 7. Generate instrumentation report

```
CERVICAL INSTRUMENTATION PLAN

CONSTRUCT: [technique name] [levels]
  Example: "Harms C1-C2: C1 lateral mass + C2 pars bilateral"

SCREW TABLE:
  Level | Side  | Type         | Entry Point | Trajectory      | Length | Diameter | Angles
  C1    | Left  | Lateral mass | [desc]      | [desc]          | Xmm   | Xmm      | med X / ceph X
  C1    | Right | Lateral mass | [desc]      | [desc]          | Xmm   | Xmm      | med X / ceph X
  C2    | Left  | Pars/Pedicle | [desc]      | [desc]          | Xmm   | Xmm      | med X / ceph X
  C2    | Right | Pars/Pedicle | [desc]      | [desc]          | Xmm   | Xmm      | med X / ceph X

VERTEBRAL ARTERY ASSESSMENT:
  Left VA: [clearance mm] [SAFE / CAUTION / HIGH RISK]
  Right VA: [clearance mm] [SAFE / CAUTION / HIGH RISK]
  High-riding: [bilateral/unilateral/none]

BONE QUALITY:
  [per level HU and adequacy]

WARNINGS:
  - [Any contraindications or caution flags]

TECHNIQUE REFERENCES:
  - [Relevant AOSpine references if available]
```

## Notes

- C1-C2 screw planning WITHOUT vertebral artery assessment is dangerous — always enforce CTA requirement
- High-riding VA may contraindicate C1 lateral mass screws on the affected side
- Technique selection depends on anatomy: pedicle size, VA course, bone quality
- "auto" technique analyzes anatomy and recommends the safest option
- For subaxial cervical (C3-C7), lateral mass screws (Magerl or An technique) are standard
- Pedicle screws in subaxial cervical require CT confirmation of adequate pedicle diameter (> 4.5mm)
