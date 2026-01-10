# 3D Slicer WebServer API Reference

HTTP REST API endpoints provided by Slicer's WebServer extension (default: `localhost:2016`).

## Python Execution

```
POST /slicer/exec
Body: Python code as plain text
Returns: Execution result (text/plain)
```

**Example:**
```bash
curl -X POST localhost:2016/slicer/exec \
  --data "import slicer; slicer.app.applicationVersion"
```

Use `__execResult` variable to return structured data:
```python
import slicer
scene = slicer.mrmlScene
__execResult = {"nodeCount": scene.GetNumberOfNodes()}
```

## Screenshot Capture

### Slice Views
```
GET /slicer/slice                           # Current Red/Axial view
GET /slicer/slice?view=Red                  # Axial
GET /slicer/slice?view=Yellow               # Sagittal
GET /slicer/slice?view=Green                # Coronal
GET /slicer/slice?view=Red&scrollTo=0.5     # Position 0.0-1.0
```

### 3D View
```
GET /slicer/threeD                          # Current camera
GET /slicer/threeD?lookFromAxis=left        # From left side
GET /slicer/threeD?lookFromAxis=anterior    # From front
```
Axes: `left`, `right`, `anterior`, `posterior`, `superior`, `inferior`

### Full Window
```
GET /slicer/screenshot                      # Entire application window
```

All screenshots return PNG image data.

## Scene Data

### Node Lists
```
GET /slicer/mrml              # Node names (JSON array)
GET /slicer/mrml/names        # Same as above
GET /slicer/mrml/ids          # Node IDs (JSON array)
```

### Node Properties
```
GET /slicer/mrml/properties?id=vtkMRMLScalarVolumeNode1
```
Returns key=value pairs (text/plain).

## Sample Data

```
GET /slicer/sampledata?name=MRHead          # Load MRHead dataset
GET /slicer/sampledata?name=CTChest         # Load CT Chest
```

Common datasets: `MRHead`, `CTChest`, `CTACardio`, `DTIBrain`, `MRBrainTumor1`, `MRBrainTumor2`

## GUI Control

```
GET /slicer/gui?viewersLayout=FourUp        # Set layout
GET /slicer/gui?contents=viewers            # Viewers only (hide toolbars)
GET /slicer/gui?contents=full               # Full GUI
```

Layouts: `FourUp`, `OneUp3D`, `OneUpRedSlice`, `Conventional`, `SideBySide`

## Volume Download

```
GET /slicer/volume                          # First volume as NRRD
GET /slicer/volume?id=vtkMRMLScalarVolumeNode1
```

## Connection Notes

- **Port:** Default 2016, configurable in Slicer WebServer module
- **Connection behavior:** Slicer closes connections immediately after response
- **Session pooling:** Not supported (causes "Connection reset by peer")
- **Timeout:** Recommend 30s for complex operations

## Links

- Official docs: https://slicer.readthedocs.io/en/latest/user_guide/modules/webserver.html
- Project client: `src/slicer_mcp/slicer_client.py`
