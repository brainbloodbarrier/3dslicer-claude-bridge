# MCP Slicer Bridge - Example Usage

This document demonstrates example conversations with Claude Code using the MCP Slicer Bridge.

## Prerequisites

1. 3D Slicer 5.0+ installed and running
2. WebServer extension enabled in Slicer (Extensions Manager > Install Extensions > WebServer)
3. WebServer configured on localhost:2016 (default)
4. MCP Slicer Bridge server registered in `.claude/mcp.json`

## Example 1: Check Connection Status

**User**: "Is 3D Slicer connected?"

**Claude Code**: Uses the `slicer://status` resource to check connection.

```json
{
  "connected": true,
  "slicer_version": "5.6.2",
  "webserver_url": "http://localhost:2016",
  "response_time_ms": 15,
  "scene_loaded": false,
  "python_available": true,
  "last_check": "2025-11-26T10:30:00Z"
}
```

**Response**: "Yes, 3D Slicer is connected and running version 5.6.2. The WebServer is responding in 15ms. No scene is currently loaded."

---

## Example 2: Load Sample Data

**User**: "Load the MRHead sample dataset"

**Claude Code**: Uses `load_sample_data` tool with dataset_name="MRHead".

```json
{
  "success": true,
  "dataset_name": "MRHead",
  "loaded_node_id": "vtkMRMLScalarVolumeNode1",
  "loaded_node_name": "MRHead",
  "message": "Sample data 'MRHead' loaded successfully"
}
```

**Response**: "I've loaded the MRHead sample dataset. It's now available as a volume node (vtkMRMLScalarVolumeNode1) in the scene."

---

## Example 3: Capture and Analyze Screenshot

**User**: "Show me an axial slice of the brain at the middle of the volume"

**Claude Code**: Uses `capture_screenshot` tool with view_type="axial" and scroll_position=0.5.

```json
{
  "success": true,
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "view_type": "axial",
  "scroll_position": 0.5,
  "content_type": "image/png"
}
```

**Response**: "Here's an axial slice at the middle of the MR brain volume:"

[Displays base64-encoded PNG image of the axial brain slice]

---

## Example 4: List Scene Contents

**User**: "What's currently loaded in the Slicer scene?"

**Claude Code**: Uses `list_scene_nodes` tool.

```json
{
  "nodes": [
    {
      "id": "vtkMRMLScalarVolumeNode1",
      "name": "MRHead",
      "type": "vtkMRMLScalarVolumeNode"
    },
    {
      "id": "vtkMRMLSliceNode1",
      "name": "Red",
      "type": "vtkMRMLSliceNode"
    },
    {
      "id": "vtkMRMLSliceNode2",
      "name": "Yellow",
      "type": "vtkMRMLSliceNode"
    },
    {
      "id": "vtkMRMLSliceNode3",
      "name": "Green",
      "type": "vtkMRMLSliceNode"
    }
  ],
  "total_count": 4
}
```

**Response**: "The scene contains 4 nodes:
1. MRHead - A scalar volume (brain MRI)
2. Red, Yellow, Green - Slice view nodes (axial, sagittal, coronal)"

---

## Example 5: Set Layout for Surgical Planning

**User**: "Switch to a single 3D view for surgical planning"

**Claude Code**: Uses `set_layout` tool with layout="OneUp3D" and gui_mode="viewers".

```json
{
  "success": true,
  "layout": "OneUp3D",
  "gui_mode": "viewers",
  "message": "Layout changed to OneUp3D"
}
```

**Response**: "I've switched Slicer to a single 3D view with minimal GUI chrome, optimized for surgical planning visualization."

---

## Example 6: Execute Custom Python Code

**User**: "Get the dimensions and spacing of the MRHead volume"

**Claude Code**: Uses `execute_python` tool with custom code.

```python
import slicer
import json

volumeNode = slicer.util.getNode('MRHead')
imageData = volumeNode.GetImageData()

result = {
    'dimensions': list(imageData.GetDimensions()),
    'spacing': list(volumeNode.GetSpacing()),
    'origin': list(volumeNode.GetOrigin())
}

json.dumps(result)
```

```json
{
  "success": true,
  "result": "{\"dimensions\": [256, 256, 130], \"spacing\": [0.9375, 0.9375, 1.5], \"origin\": [-119.5312, -119.5312, -97.5]}",
  "stdout": "",
  "stderr": ""
}
```

**Response**: "The MRHead volume has:
- Dimensions: 256x256x130 voxels
- Spacing: 0.94mm x 0.94mm x 1.5mm
- Origin: (-119.5, -119.5, -97.5) mm"

---

## Example 7: Measure Segmentation Volume

**User**: "Create a simple segmentation and measure its volume"

**Claude Code**: First creates a segmentation using `execute_python`:

```python
import slicer

# Create segmentation node
segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
segmentationNode.SetName('TestSegmentation')

# Add a test segment (sphere)
segmentation = segmentationNode.GetSegmentation()
segment = segmentation.AddEmptySegment('TestSegment')

# Get the node ID
segmentationNode.GetID()
```

Then measures volume using `measure_volume` tool:

```json
{
  "node_id": "vtkMRMLSegmentationNode1",
  "node_name": "TestSegmentation",
  "total_volume_mm3": 5234.7,
  "total_volume_ml": 5.23,
  "segments": [
    {
      "name": "TestSegment",
      "volume_mm3": 5234.7,
      "volume_ml": 5.23
    }
  ]
}
```

**Response**: "I've created a test segmentation and measured its volume:
- Segmentation: TestSegmentation
- Segment: TestSegment
- Volume: 5.23 ml (5,234.7 mm³)"

---

## Example 8: Multi-View Screenshot Workflow

**User**: "Show me all three orthogonal views of the brain"

**Claude Code**: Calls `capture_screenshot` three times with different view_types.

1. `capture_screenshot(view_type="axial", scroll_position=0.5)`
2. `capture_screenshot(view_type="sagittal", scroll_position=0.5)`
3. `capture_screenshot(view_type="coronal", scroll_position=0.5)`

**Response**: "Here are the three orthogonal views of the brain at mid-volume:

**Axial (Red slice):**
[Displays axial image]

**Sagittal (Yellow slice):**
[Displays sagittal image]

**Coronal (Green slice):**
[Displays coronal image]"

---

## Example 9: 3D View from Different Angles

**User**: "Show me the 3D brain from the left and right sides"

**Claude Code**: Calls `capture_screenshot` twice with view_type="3d" and different look_from_axis values.

1. `capture_screenshot(view_type="3d", look_from_axis="left")`
2. `capture_screenshot(view_type="3d", look_from_axis="right")`

**Response**: "Here are 3D renderings of the brain from left and right perspectives:

**Left view:**
[Displays 3D left view]

**Right view:**
[Displays 3D right view]"

---

## Example 10: Inspect Volume Metadata

**User**: "Give me detailed information about all loaded volumes"

**Claude Code**: Uses `slicer://volumes` resource.

```json
{
  "volumes": [
    {
      "id": "vtkMRMLScalarVolumeNode1",
      "name": "MRHead",
      "type": "scalar",
      "dimensions": [256, 256, 130],
      "spacing": [0.9375, 0.9375, 1.5],
      "origin": [-119.5312, -119.5312, -97.5],
      "scalar_range": [0, 279],
      "file_path": null
    }
  ],
  "total_count": 1
}
```

**Response**: "Currently loaded volumes:

**MRHead** (vtkMRMLScalarVolumeNode1)
- Type: Scalar volume
- Dimensions: 256 x 256 x 130 voxels
- Spacing: 0.94 x 0.94 x 1.5 mm
- Origin: (-119.5, -119.5, -97.5) mm
- Intensity range: 0 to 279
- Source: Sample data (not saved to file)"

---

## Tips for Using MCP Slicer Bridge

1. **Always check connection first**: Use `slicer://status` resource to verify Slicer is running
2. **Load sample data for testing**: Use `load_sample_data` with "MRHead" for quick testing
3. **Set appropriate layout**: Use `set_layout` to optimize the view for your task
4. **Capture multiple views**: Combine different view_types to get comprehensive visualization
5. **Use Python execution for advanced tasks**: `execute_python` provides full access to Slicer's API
6. **Check scene contents**: Use `list_scene_nodes` or `slicer://scene` to understand current state

## Troubleshooting

**Problem**: "Could not connect to Slicer WebServer"

**Solution**:
1. Ensure 3D Slicer is running
2. Open Extension Manager in Slicer
3. Install "WebServer" extension if not already installed
4. Restart Slicer
5. Verify WebServer is running (check Slicer's Python console)

**Problem**: "Screenshot is black or empty"

**Solution**:
1. Ensure data is loaded in Slicer
2. Use `set_layout` to configure appropriate view
3. Check that the view type matches your data (e.g., don't use "3d" if no 3D renderable data exists)

**Problem**: "Python execution fails"

**Solution**:
1. Check Python code syntax
2. Verify you're using Slicer's Python API correctly
3. Check Slicer's Python console for detailed error messages
4. Use `import slicer` at the start of your code

---

## Advanced Workflows

### Workflow 1: Automated Brain Analysis

```
1. load_sample_data(dataset_name="MRHead")
2. set_layout(layout="FourUp", gui_mode="full")
3. capture_screenshot(view_type="full")
4. execute_python(code="<segmentation code>")
5. measure_volume(node_id="vtkMRMLSegmentationNode1")
6. capture_screenshot(view_type="3d", look_from_axis="left")
```

### Workflow 2: Multi-Dataset Comparison

```
1. load_sample_data(dataset_name="MRBrainTumor1")
2. list_scene_nodes() - note node IDs
3. capture_screenshot(view_type="axial", scroll_position=0.5)
4. load_sample_data(dataset_name="MRBrainTumor2")
5. set_layout(layout="SideBySide")
6. capture_screenshot(view_type="full")
```

### Workflow 3: Surgical Planning Visualization

```
1. <Load patient data via execute_python>
2. set_layout(layout="OneUp3D", gui_mode="viewers")
3. execute_python(code="<adjust 3D visualization settings>")
4. capture_screenshot(view_type="3d", look_from_axis="anterior")
5. capture_screenshot(view_type="3d", look_from_axis="left")
6. capture_screenshot(view_type="3d", look_from_axis="superior")
```

---

## References

- [3D Slicer Documentation](https://slicer.readthedocs.io/)
- [Slicer WebServer Extension](https://github.com/Slicer/SlicerWebServer)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)
