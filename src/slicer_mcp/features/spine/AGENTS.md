# spine/ — Spine Analysis Domain

## OVERVIEW

Highest-complexity feature area: spine segmentation (TotalSegmentator), alignment measurement, CCJ angles, cervical screw planning, and vertebral artery segmentation. 4707 LOC across 3 files.

## STRUCTURE

| File | LOC | Contents |
|------|-----|----------|
| `tools.py` | 2384 | Segmentation, alignment, CCJ, vertebral artery, visualization, bone quality |
| `instrumentation.py` | 1612 | Cervical screw planning (6 techniques + auto-selection) |
| `constants.py` | 711 | Vertebra label maps, segment names, reference ranges, screw dimensions |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add spine measurement | `tools.py` | Follow `measure_spine_alignment` pattern |
| Add screw technique | `instrumentation.py` | Add to technique dict + update auto-selection |
| Add vertebra label mapping | `constants.py` | `VERTEBRA_LABEL_MAP`, `VERTEBRA_SEGMENTS` |
| Add reference range | `constants.py` | Clinical reference ranges for measurements |
| Fix TotalSegmentator hang | `tools.py` | `_build_totalseg_subprocess_block` — see gotchas below |

## TOTALSEGMENTATOR SUBPROCESS (critical gotcha)

TotalSegmentator hangs on `multiprocessing.resource_tracker` after saving output. Workaround:

1. **`start_new_session=True`** — creates own process group (so `killpg` won't kill Slicer itself)
2. **Poll for output** — check file existence + stable size instead of `proc.wait()`
3. **`os.killpg(os.getpgid(proc.pid), SIGTERM)`** — kill entire process group, escalate to `SIGKILL` after 1s
4. All kill calls wrapped in `try/except (ProcessLookupError, PermissionError, OSError)`

**Never switch to `proc.wait()` or `proc.communicate()`** — will hang indefinitely.

`_build_totalseg_subprocess_block()` is also imported by `diagnostics/ct.py` and `diagnostics/mri.py` for segmentation-dependent diagnostics.

## SCREW PLANNING (`instrumentation.py`)

6 cervical screw techniques: Magerl, lateral mass (Roy-Camille, Magerl, An, Anderson), pedicle. Auto-selection based on vertebral level and anatomy.

Each technique generates a Slicer Python code string that:
1. Validates vertebra segmentation exists
2. Computes entry point, trajectory, and screw dimensions from anatomy
3. Creates markup fiducials and line nodes for visualization
4. Returns measurement results as `__execResult`

## CONSTANTS (`constants.py`)

- `VERTEBRA_LABEL_MAP`: TotalSegmentator label → vertebra name mapping
- `VERTEBRA_SEGMENTS`: Ordered list of vertebral segments (C1–S5)
- `SCREW_DIMENSIONS`: Per-level screw diameter/length reference ranges
- Clinical reference ranges for sagittal balance, lordosis, kyphosis angles
