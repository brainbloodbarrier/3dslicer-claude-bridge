# MCP Tools API Reference

12 tools for interacting with 3D Slicer. All tools use sync functions (FastMCP handles async wrapping).

## 1. capture_screenshot

Capture PNG screenshot from a Slicer viewport.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `view_type` | string | Yes | `axial`, `sagittal`, `coronal`, `3d`, `full` |
| `scroll_position` | float | No | Slice position 0.0-1.0 (2D views only) |
| `look_from_axis` | string | No | Camera axis for 3D: `left`, `right`, `anterior`, `posterior`, `superior`, `inferior` |

**Returns:**
```json
{"success": true, "image_base64": "iVBORw0...", "view_type": "axial", "content_type": "image/png"}
```

**HTTP:** `GET /slicer/slice?view=Red&scrollTo=0.5` or `GET /slicer/threeD?lookFromAxis=left`

---

## 2. list_scene_nodes

List all MRML scene nodes with metadata.

**Parameters:** None

**Returns:**
```json
{
  "nodes": [
    {"id": "vtkMRMLScalarVolumeNode1", "name": "MRHead", "type": "vtkMRMLScalarVolumeNode"}
  ],
  "total_count": 1
}
```

**HTTP:** `GET /slicer/mrml/names` + `GET /slicer/mrml/ids`

---

## 3. execute_python

Execute Python code in Slicer's interpreter.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code` | string | Yes | Python 3 code with access to `slicer`, `vtk`, `qt` modules |

**Returns:**
```json
{"success": true, "result": "...", "stdout": "...", "stderr": "..."}
```

**HTTP:** `POST /slicer/exec` (body = code string)

**Security:** All executions logged to audit file if `SLICER_AUDIT_LOG` is set.

---

## 4. measure_volume

Calculate segmentation volume in mm³/mL.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `node_id` | string | Yes | Segmentation node ID (e.g., `vtkMRMLSegmentationNode1`) |
| `segment_name` | string | No | Specific segment to measure (omit for all) |

**Returns:**
```json
{
  "node_id": "vtkMRMLSegmentationNode1",
  "total_volume_mm3": 12543.7,
  "total_volume_ml": 12.54,
  "segments": [{"name": "Tumor", "volume_mm3": 8234.2, "volume_ml": 8.23}]
}
```

**HTTP:** Uses `POST /slicer/exec` with SegmentStatistics code internally.

**Validation:** Node ID must be ASCII-only, max 256 chars. Segment name NFKC normalized, max 256 chars.

---

## 5. list_sample_data

Discover available sample datasets.

**Parameters:** None

**Returns:**
```json
{
  "datasets": [
    {"name": "MRHead", "category": "MRI", "description": "T1-weighted brain MRI"}
  ]
}
```

**HTTP:** Queries Slicer's SampleData module via `POST /slicer/exec`.

---

## 6. load_sample_data

Load a sample dataset into Slicer.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `dataset_name` | string | Yes | `MRHead`, `CTChest`, `CTACardio`, `DTIBrain`, `MRBrainTumor1`, `MRBrainTumor2` |

**Returns:**
```json
{"success": true, "dataset_name": "MRHead", "loaded_node_id": "vtkMRMLScalarVolumeNode1"}
```

**HTTP:** `GET /slicer/sampledata?name=MRHead`

---

## 7. set_layout

Configure viewer layout and GUI mode.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `layout` | string | Yes | `FourUp`, `OneUp3D`, `OneUpRedSlice`, `Conventional`, `SideBySide` |
| `gui_mode` | string | No | `full` (default) or `viewers` (hide toolbars) |

**Returns:**
```json
{"success": true, "layout": "FourUp", "gui_mode": "full"}
```

**HTTP:** `GET /slicer/gui?viewersLayout=FourUp&contents=full`

---

## 8. import_dicom

Import DICOM files from a folder into Slicer's DICOM database.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `folder_path` | string | Yes | Path to folder containing DICOM files |

**Returns:**
```json
{
  "success": true,
  "folder_path": "/path/to/dicoms",
  "patients_count": 1,
  "studies_count": 2,
  "series_count": 5,
  "new_patients": 1
}
```

**HTTP:** Uses `POST /slicer/exec` with DICOMUtils.importDicom internally.

**Validation:** Path must exist, be a directory, no path traversal (`..`), max 4096 chars.

---

## 9. list_dicom_studies

List all studies in the DICOM database.

**Parameters:** None

**Returns:**
```json
{
  "success": true,
  "studies": [
    {
      "patient_id": "12345",
      "patient_name": "DOE^JOHN",
      "study_uid": "1.2.840.113619...",
      "study_date": "20240115",
      "study_description": "Brain MRI",
      "modalities": ["MR"],
      "series_count": 3
    }
  ],
  "total_count": 1
}
```

**HTTP:** Uses `POST /slicer/exec` with DICOM database queries.

---

## 10. list_dicom_series

List all series within a DICOM study.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `study_uid` | string | Yes | DICOM Study UID (digits and dots only) |

**Returns:**
```json
{
  "success": true,
  "study_uid": "1.2.840.113619...",
  "series": [
    {
      "series_uid": "1.2.840.113619...",
      "series_number": "1",
      "series_description": "T1 MPRAGE",
      "modality": "MR",
      "file_count": 176
    }
  ],
  "total_count": 3
}
```

**HTTP:** Uses `POST /slicer/exec` with DICOM database queries.

**Validation:** Study UID must be digits and dots only, max 64 chars.

---

## 11. load_dicom_series

Load a DICOM series as a volume into the scene.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `series_uid` | string | Yes | DICOM Series UID to load |

**Returns:**
```json
{
  "success": true,
  "series_uid": "1.2.840.113619...",
  "node_id": "vtkMRMLScalarVolumeNode1",
  "node_name": "T1_MPRAGE",
  "node_class": "vtkMRMLScalarVolumeNode",
  "dimensions": [256, 256, 176],
  "spacing": [1.0, 1.0, 1.0],
  "origin": [0.0, 0.0, 0.0],
  "scalar_range": [0, 4095]
}
```

**HTTP:** Uses `POST /slicer/exec` with DICOMUtils.loadSeriesByUID.

---

## 12. run_brain_extraction

Run brain extraction (skull stripping) on a brain MRI or CT scan.

**⚠️ LONG OPERATION:** May take 20 seconds to 5 minutes depending on method and hardware.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `input_node_id` | string | Yes | MRML node ID of input brain MRI/CT volume |
| `method` | string | No | `hd-bet` (AI, default) or `swiss` (atlas-based) |
| `device` | string | No | For HD-BET: `auto` (default), `cpu`, or GPU index (`0`, `1`) |

**Returns:**
```json
{
  "success": true,
  "input_node_id": "vtkMRMLScalarVolumeNode1",
  "method": "hd-bet",
  "output_volume_id": "vtkMRMLScalarVolumeNode2",
  "output_volume_name": "MRHead_brain",
  "output_segmentation_id": "vtkMRMLSegmentationNode1",
  "output_segmentation_name": "MRHead_brain_mask",
  "brain_volume_ml": 1423.5,
  "processing_time_seconds": 28.4,
  "long_operation": {
    "type": "brain_extraction",
    "method": "hd-bet",
    "timeout_seconds": 600
  }
}
```

**HTTP:** Uses `POST /slicer/exec` with HD-BET or SwissSkullStripper modules.

**Expected Duration:**
| Method | Hardware | Time |
|--------|----------|------|
| hd-bet | GPU | 20-30s |
| hd-bet | CPU | 3-5 min |
| swiss | Any | 2-3 min |

**Requires:** HD-BET extension (for `hd-bet`) or SwissSkullStripper extension (for `swiss`).

---

## Performance Notes

| Tool | Expected Time |
|------|---------------|
| `capture_screenshot` | 200-500ms |
| `list_scene_nodes` | 50-200ms |
| `execute_python` | 100ms-5s (depends on code) |
| `measure_volume` | 200-1000ms |
| `list_sample_data` | 50-100ms |
| `load_sample_data` | 2-10s (includes download) |
| `set_layout` | 50-100ms |
| `import_dicom` | 1-30s (depends on file count) |
| `list_dicom_studies` | 50-200ms |
| `list_dicom_series` | 50-200ms |
| `load_dicom_series` | 1-10s (depends on series size) |
| `run_brain_extraction` | 20s-5min (see tool docs) |
