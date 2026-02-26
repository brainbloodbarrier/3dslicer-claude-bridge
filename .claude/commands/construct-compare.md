# /construct-compare - Compare Instrumentation Options

Compare alternative screw constructs for the same case.

## Description

Runs multiple screw planning scenarios with different techniques and generates a comparative analysis to aid surgical decision-making.

## Usage

```
/construct-compare C1-C2          # Compare all viable C1-C2 techniques
/construct-compare C3-C6          # Compare subaxial options
```

## Instructions

When the user invokes this command:

### 1. Parse levels from `$ARGUMENTS`

If no levels specified, ask the user.

### 2. Verify data

Call `list_scene_nodes()`. Requires CT + segmentation + CTA (if C1-C2 involved).

If segmentation is missing, run `segment_spine()`.
If C1-C2 involved and no VA segmentation, run `segment_vertebral_artery()`.

### 3. Get auto recommendation first

```
plan_cervical_screws(
    volume_node_id=<ct_id>,
    segmentation_node_id=<seg_id>,
    levels=<levels>,
    technique="auto",
    va_segmentation_id=<va_seg_id>
)
```

Note the recommended technique and viable alternatives from the output.

### 4. Run each viable technique

For each viable technique returned by the auto analysis, run:
```
plan_cervical_screws(
    ...,
    technique=<specific_technique>
)
```

Common alternatives for C1-C2:
- Harms (C1 lateral mass + C2 pars/pedicle)
- Magerl (transarticular C1-C2)
- Wright (C1 lateral mass + C2 translaminar)

Common alternatives for subaxial:
- Lateral mass (Magerl technique)
- Lateral mass (An technique)
- Pedicle screws (if anatomy allows)

### 5. Generate comparative table

```
CONSTRUCT COMPARISON - [levels]

OPTION A: [technique name] (RECOMMENDED)
  - Screws: [summary]
  - VA clearance: [mm]
  - Bone quality match: [good/fair/poor]

OPTION B: [technique name]
  - Screws: [summary]
  - VA clearance: [mm]
  - Bone quality match: [good/fair/poor]

OPTION C: [technique name]
  - Screws: [summary]
  - VA clearance: [mm]
  - Bone quality match: [good/fair/poor]

COMPARISON TABLE:
  Criteria         | Option A | Option B | Option C
  Biomechanical    | [rating] | [rating] | [rating]
  VA Risk          | [rating] | [rating] | [rating]
  Technical Ease   | [rating] | [rating] | [rating]
  Bone Req.        | [rating] | [rating] | [rating]
  Viable           | YES/NO   | YES/NO   | YES/NO

RECOMMENDATION:
  [Technique name] because [anatomical justification]

CONTRAINDICATIONS:
  - [Any techniques ruled out and why]
```

### 6. Screenshots

Capture 3D screenshots showing each viable construct if possible.

## Notes

- Some techniques may not be viable due to anatomy (e.g., high-riding VA blocks Magerl transarticular)
- Always show WHY a technique is contraindicated, not just that it is
- The "auto" recommendation should be listed first as the primary option
- Bone quality affects technique choice: poor bone may favor longer constructs or cement augmentation
