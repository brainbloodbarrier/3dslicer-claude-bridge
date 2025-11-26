# MCP Slicer Bridge - Technical Specification

## Overview

The MCP Slicer Bridge is a Model Context Protocol (MCP) server that provides Claude Code with programmatic access to 3D Slicer, enabling AI-assisted medical image analysis, surgical planning, and radiomics workflows.

**Purpose**: Enable Claude Code to interact with 3D Slicer for medical imaging tasks including visualization, segmentation, measurement, and Python scripting.

**Scope**: MVP implementation focused on core functionality for educational and personal research use.

## Protocol Details

- **MCP Version**: 2025-06-18
- **Transport**: stdio (standard input/output)
- **Message Format**: JSON-RPC 2.0
- **Python Version**: 3.10+
- **Slicer Version**: 5.0+ with WebServer extension

## Prerequisites

1. **3D Slicer Installation**
   - 3D Slicer 5.0 or later installed and running
   - WebServer extension enabled (via Extension Manager)
   - WebServer configured on `localhost:2016`

2. **Python Environment**
   - Python 3.10 or later
   - `uv` package manager installed
   - Dependencies: `mcp[cli]>=1.1.0`, `requests>=2.31.0`, `pydantic>=2.0`

3. **Claude Code Configuration**
   - MCP server registered in `.claude/mcp.json`
   - Server configured with stdio transport

---

## Slicer WebServer HTTP Endpoints Reference

The MCP server communicates with Slicer via these REST endpoints:

### Data Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/slicer/sampledata?name=<name>` | GET | Load sample dataset (e.g., MRHead) |
| `/slicer/volume` | GET | Download volume in NRRD format |
| `/slicer/volume?id=<node_id>` | GET | Download specific volume |

### Viewer Control
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/slicer/gui?contents=<mode>&viewersLayout=<layout>` | GET | Set GUI mode and layout |
| `/slicer/screenshot` | GET | Screenshot of main window (PNG) |

### Slice Views
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/slicer/slice` | GET | Current slice view (Red/Axial) |
| `/slicer/slice?view=Red` | GET | Axial slice |
| `/slicer/slice?view=Yellow` | GET | Sagittal slice |
| `/slicer/slice?view=Green` | GET | Coronal slice |
| `/slicer/slice?orientation=<orient>&scrollTo=<0-1>` | GET | Slice with position |

### 3D Views
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/slicer/threeD` | GET | 3D view screenshot |
| `/slicer/threeD?lookFromAxis=<axis>` | GET | 3D view from axis (left/right/anterior/posterior/superior/inferior) |

### Python Execution
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/slicer/exec` | POST | Execute Python code, body = code string |

### Scene Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/slicer/mrml` | GET | List MRML node names |
| `/slicer/mrml/ids` | GET | List MRML node IDs |
| `/slicer/mrml/properties?id=<id>` | GET | Get node properties |

### DICOMweb (if enabled)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dicom/studies` | GET | List all DICOM studies |
| `/dicom/studies/<uid>/series` | GET | List series for study |

### GUI Layout Options
- `contents`: `full` (complete GUI) or `viewers` (viewers only)
- `viewersLayout`: `FourUp`, `OneUp3D`, `OneUpRedSlice`, `Conventional`, etc.

## Tools Specification

### 1. capture_screenshot

Captures a screenshot from a specific 3D Slicer viewport and returns it as a base64-encoded PNG.

**HTTP Endpoint**:
- Slice views: `GET /slicer/slice?view=<view>&scrollTo=<position>`
- 3D view: `GET /slicer/threeD?lookFromAxis=<axis>`
- Full window: `GET /slicer/screenshot`

**Parameters**:
- `view_type` (string, required): Viewport type to capture
  - Allowed values: `"axial"`, `"sagittal"`, `"coronal"`, `"3d"`, `"full"`
  - `axial`: Red slice view (GET /slicer/slice?view=Red)
  - `sagittal`: Yellow slice view (GET /slicer/slice?view=Yellow)
  - `coronal`: Green slice view (GET /slicer/slice?view=Green)
  - `3d`: 3D rendering view (GET /slicer/threeD)
  - `full`: Full application window (GET /slicer/screenshot)

- `scroll_position` (number, optional): Slice position for 2D views (0.0 to 1.0)
  - 0.0 = start of volume
  - 0.5 = middle of volume
  - 1.0 = end of volume
  - Only applicable to axial/sagittal/coronal views
  - Default: current position (omit parameter)

- `look_from_axis` (string, optional): Camera axis for 3D view
  - Allowed values: `"left"`, `"right"`, `"anterior"`, `"posterior"`, `"superior"`, `"inferior"`
  - Only applicable to 3d view_type
  - Default: current camera position

**Returns**:
```json
{
  "success": true,
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "view_type": "axial",
  "scroll_position": 0.5,
  "content_type": "image/png"
}
```

**Example**:
```python
# Capture axial view at middle of volume
result = await capture_screenshot(view_type="axial", scroll_position=0.5)

# Capture sagittal view at 30% position
result = await capture_screenshot(view_type="sagittal", scroll_position=0.3)

# Capture 3D rendering from left side
result = await capture_screenshot(view_type="3d", look_from_axis="left")

# Capture full application window
result = await capture_screenshot(view_type="full")
```

**Error Cases**:
- Invalid view_type: Returns error with valid options
- Slicer not responding: Connection timeout error
- Screenshot capture failure: Slicer-specific error message

---

### 2. list_scene_nodes

Lists all nodes in the current MRML scene with metadata including type, name, and properties.

**HTTP Endpoints**:
- Names: `GET /slicer/mrml` or `GET /slicer/mrml/names`
- IDs: `GET /slicer/mrml/ids`
- Properties: `GET /slicer/mrml/properties?id=<node_id>`

**Parameters**: None

**Returns**:
```json
{
  "nodes": [
    {
      "id": "vtkMRMLScalarVolumeNode1",
      "name": "Brain-MRI-T1",
      "type": "vtkMRMLScalarVolumeNode",
      "properties": {
        "dimensions": [256, 256, 180],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [-128.0, -128.0, -90.0]
      }
    },
    {
      "id": "vtkMRMLSegmentationNode1",
      "name": "TumorSegmentation",
      "type": "vtkMRMLSegmentationNode",
      "properties": {
        "segment_count": 3,
        "segments": ["Tumor", "Edema", "Necrosis"]
      }
    }
  ],
  "total_count": 2
}
```

**Example**:
```python
# List all scene nodes
result = await list_scene_nodes()
for node in result["nodes"]:
    print(f"{node['name']} ({node['type']})")
```

**Error Cases**:
- Slicer not responding: Connection timeout error
- Empty scene: Returns empty nodes array with total_count=0
- Scene query failure: Slicer-specific error message

---

### 3. execute_python

Executes arbitrary Python code in the 3D Slicer Python environment and returns the result.

**HTTP Endpoint**: `POST /slicer/exec` (body = Python code string)

**Security Note**: This tool executes code directly in Slicer's Python interpreter. Use only with trusted code in controlled environments.

**Parameters**:
- `code` (string, required): Python code to execute
  - Must be valid Python 3 syntax
  - Has access to Slicer's Python environment (slicer, vtk, qt modules)
  - Can modify scene, create nodes, perform calculations
  - Return value serialized as JSON

**Returns**:
```json
{
  "success": true,
  "result": "Any JSON-serializable return value",
  "stdout": "Captured print statements",
  "stderr": "Captured error output"
}
```

**Example**:
```python
# Get Slicer version
result = await execute_python(code="import slicer; slicer.app.applicationVersion")

# Count scene nodes
code = """
import slicer
scene = slicer.mrmlScene
node_count = scene.GetNumberOfNodes()
node_count
"""
result = await execute_python(code=code)

# Create segmentation
code = """
import slicer
segmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
segmentation.SetName('NewSegmentation')
segmentation.GetID()
"""
result = await execute_python(code=code)
```

**Error Cases**:
- Syntax error: Returns Python traceback in stderr
- Runtime error: Returns exception details in stderr
- Slicer not responding: Connection timeout error
- Non-serializable return: JSON serialization error

---

### 4. measure_volume

Calculates the volume of a segmentation node or specific segment in cubic millimeters.

**HTTP Endpoint**: Uses `POST /slicer/exec` with SegmentStatistics Python code internally.

**Parameters**:
- `node_id` (string, required): MRML node ID of segmentation
  - Must be a valid vtkMRMLSegmentationNode ID
  - Format: typically "vtkMRMLSegmentationNode1", "vtkMRMLSegmentationNode2", etc.

- `segment_name` (string, optional): Specific segment to measure
  - If provided, measures only this segment
  - If omitted, measures all segments and returns total + per-segment breakdown
  - Must match exact segment name in segmentation node

**Returns**:
```json
{
  "node_id": "vtkMRMLSegmentationNode1",
  "node_name": "TumorSegmentation",
  "total_volume_mm3": 12543.7,
  "total_volume_ml": 12.54,
  "segments": [
    {
      "name": "Tumor",
      "volume_mm3": 8234.2,
      "volume_ml": 8.23
    },
    {
      "name": "Edema",
      "volume_mm3": 4309.5,
      "volume_ml": 4.31
    }
  ]
}
```

**Example**:
```python
# Measure all segments
result = await measure_volume(node_id="vtkMRMLSegmentationNode1")
print(f"Total volume: {result['total_volume_ml']} ml")

# Measure specific segment
result = await measure_volume(
    node_id="vtkMRMLSegmentationNode1",
    segment_name="Tumor"
)
print(f"Tumor volume: {result['segments'][0]['volume_ml']} ml")
```

**Error Cases**:
- Invalid node_id: Node not found error
- Node is not segmentation: Type mismatch error
- Segment not found: Segment name not found in node
- Volume calculation failure: Slicer-specific error message
- Slicer not responding: Connection timeout error

---

### 5. load_sample_data

Loads a sample dataset into 3D Slicer for testing and demonstration purposes.

**HTTP Endpoint**: `GET /slicer/sampledata?name=<dataset_name>`

**Parameters**:
- `dataset_name` (string, required): Name of sample dataset to load
  - Available datasets: `"MRHead"`, `"CTChest"`, `"CTACardio"`, `"DTIBrain"`, `"MRBrainTumor1"`, `"MRBrainTumor2"`
  - `MRHead`: T1-weighted brain MRI (most common for testing)
  - `CTChest`: Chest CT scan
  - `CTACardio`: CT angiography of heart
  - `DTIBrain`: Diffusion tensor imaging brain
  - `MRBrainTumor1/2`: Brain MRI with tumor

**Returns**:
```json
{
  "success": true,
  "dataset_name": "MRHead",
  "loaded_node_id": "vtkMRMLScalarVolumeNode1",
  "loaded_node_name": "MRHead",
  "message": "Sample data 'MRHead' loaded successfully"
}
```

**Example**:
```python
# Load MRHead sample data for testing
result = await load_sample_data(dataset_name="MRHead")
print(f"Loaded: {result['loaded_node_name']}")

# Load CT chest scan
result = await load_sample_data(dataset_name="CTChest")
```

**Error Cases**:
- Invalid dataset_name: Returns error with available options
- Download failure: Network or server error
- Slicer not responding: Connection timeout error

---

### 6. set_layout

Sets the viewer layout and GUI mode in 3D Slicer.

**HTTP Endpoint**: `GET /slicer/gui?contents=<mode>&viewersLayout=<layout>`

**Parameters**:
- `layout` (string, required): Viewer layout configuration
  - Common layouts: `"FourUp"`, `"OneUp3D"`, `"OneUpRedSlice"`, `"Conventional"`, `"SideBySide"`
  - `FourUp`: Standard 4-panel view (axial, sagittal, coronal, 3D)
  - `OneUp3D`: Single 3D view maximized
  - `OneUpRedSlice`: Single axial slice view
  - `Conventional`: Traditional radiology layout
  - `SideBySide`: Side-by-side comparison view

- `gui_mode` (string, optional): GUI display mode
  - `"full"`: Complete application GUI (default)
  - `"viewers"`: Viewers only (minimal chrome)

**Returns**:
```json
{
  "success": true,
  "layout": "FourUp",
  "gui_mode": "full",
  "message": "Layout changed to FourUp"
}
```

**Example**:
```python
# Set standard 4-panel layout
result = await set_layout(layout="FourUp")

# Maximize 3D view for surgical planning
result = await set_layout(layout="OneUp3D", gui_mode="viewers")

# Set axial-only view for slice review
result = await set_layout(layout="OneUpRedSlice")
```

**Error Cases**:
- Invalid layout: Returns error with available options
- Invalid gui_mode: Returns error with valid options
- Slicer not responding: Connection timeout error

---

## Resources Specification

### 1. slicer://scene

Provides the current MRML scene structure as JSON.

**URI**: `slicer://scene`

**Returns**:
```json
{
  "scene_id": "vtkMRMLScene",
  "modified_time": "2025-11-26T10:30:00Z",
  "node_count": 15,
  "nodes": [
    {
      "id": "vtkMRMLScalarVolumeNode1",
      "name": "Brain-MRI",
      "type": "vtkMRMLScalarVolumeNode"
    }
  ],
  "connections": [
    {
      "source": "vtkMRMLScalarVolumeNode1",
      "target": "vtkMRMLSliceNode1",
      "type": "display"
    }
  ]
}
```

**Use Cases**:
- Understanding current scene state before operations
- Identifying available nodes for processing
- Tracking scene modifications over time

---

### 2. slicer://volumes

Lists all loaded imaging volumes with metadata.

**URI**: `slicer://volumes`

**Returns**:
```json
{
  "volumes": [
    {
      "id": "vtkMRMLScalarVolumeNode1",
      "name": "Brain-MRI-T1",
      "type": "scalar",
      "dimensions": [256, 256, 180],
      "spacing": [1.0, 1.0, 1.0],
      "origin": [-128.0, -128.0, -90.0],
      "scalar_range": [0, 4095],
      "file_path": "/path/to/brain.nrrd"
    },
    {
      "id": "vtkMRMLVectorVolumeNode1",
      "name": "DTI-FA",
      "type": "vector",
      "dimensions": [128, 128, 60],
      "spacing": [2.0, 2.0, 2.0]
    }
  ],
  "total_count": 2
}
```

**Use Cases**:
- Identifying available volumes for analysis
- Checking image properties before processing
- Validating data loaded correctly

---

### 3. slicer://status

Provides health status and connection information for 3D Slicer.

**URI**: `slicer://status`

**Returns**:
```json
{
  "connected": true,
  "slicer_version": "5.6.2",
  "webserver_url": "http://localhost:2016",
  "response_time_ms": 15,
  "scene_loaded": true,
  "python_available": true,
  "last_check": "2025-11-26T10:30:00Z"
}
```

**Use Cases**:
- Verifying Slicer connection before operations
- Debugging connection issues
- Monitoring system health

---

## Error Handling

### Error Response Format

All errors follow consistent JSON structure:

```json
{
  "error": {
    "code": "SLICER_CONNECTION_ERROR",
    "message": "Could not connect to Slicer WebServer at localhost:2016",
    "details": {
      "url": "http://localhost:2016",
      "timeout": 30,
      "suggestion": "Ensure Slicer is running with WebServer extension enabled"
    }
  }
}
```

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `SLICER_CONNECTION_ERROR` | Cannot connect to Slicer WebServer | Start Slicer, enable WebServer extension |
| `SLICER_TIMEOUT` | Request timeout (>30s) | Check Slicer responsiveness, restart if frozen |
| `INVALID_NODE_ID` | Node ID not found in scene | Verify node exists with `list_scene_nodes` |
| `INVALID_PARAMETER` | Parameter validation failed | Check parameter types and allowed values |
| `PYTHON_EXECUTION_ERROR` | Python code execution failed | Review code syntax and Slicer API usage |
| `SCREENSHOT_FAILED` | Screenshot capture failed | Verify view exists and is visible |
| `VOLUME_CALCULATION_ERROR` | Volume measurement failed | Check segmentation node has valid geometry |

### Retry Strategy

- Connection errors: Retry up to 3 times with exponential backoff (1s, 2s, 4s)
- Timeout errors: No retry (likely Slicer is frozen)
- Validation errors: No retry (client must fix parameters)
- Python execution errors: No retry (code must be corrected)

---

## Security Considerations

### MVP Scope

This MVP implementation is designed for:
- **Local use only**: Server and Slicer run on same machine
- **Educational purposes**: Learning medical image analysis with AI assistance
- **Personal research**: Individual research projects in controlled environments

### Security Limitations

1. **Code Execution**: `execute_python` tool executes arbitrary code in Slicer
   - No sandboxing or code validation
   - Full access to Slicer's Python environment
   - Can modify or delete scene data

2. **Authentication**: No authentication mechanism
   - WebServer connection is unauthenticated
   - Assumes trusted local environment

3. **Data Privacy**: No encryption or privacy controls
   - Medical imaging data transmitted over localhost HTTP
   - Suitable only for de-identified research data

### Recommendations for Production Use

For production/clinical environments, implement:
- Authentication and authorization (API keys, OAuth)
- Code execution sandboxing or whitelist
- Audit logging of all operations
- Data encryption in transit (HTTPS)
- HIPAA/GDPR compliance measures
- Network isolation and firewall rules

---

## Performance Characteristics

### Expected Response Times

- `list_scene_nodes`: 50-200ms (depends on scene size)
- `capture_screenshot`: 200-500ms (includes rendering)
- `execute_python`: 100ms-5s (depends on code complexity)
- `measure_volume`: 200-1000ms (depends on segmentation size)
- `load_sample_data`: 2-10s (includes download and loading)
- `set_layout`: 50-100ms (UI update only)

### Resource Usage

- Memory: ~50MB for MCP server process
- Network: localhost HTTP (no network overhead)
- CPU: Minimal (primary processing in Slicer)

### Scalability Limits

- Concurrent requests: 1 (stdio transport is synchronous)
- Scene size: Limited by Slicer's memory capacity
- Volume size: Recommended <2GB per volume for responsive performance

---

## Version History

- **1.0.0** (2025-11-26): Initial MVP specification
  - 6 core tools:
    - `capture_screenshot`: viewport capture with scroll/axis control
    - `list_scene_nodes`: MRML scene inspection
    - `execute_python`: arbitrary Python execution
    - `measure_volume`: segmentation volumetrics
    - `load_sample_data`: sample dataset loading
    - `set_layout`: viewer layout control
  - 3 resources (scene, volumes, status)
  - stdio transport with FastMCP framework
  - Local-only security model
  - Complete Slicer WebServer endpoint mapping
