# MCP Resources API Reference

3 read-only resources providing Slicer state information.

## 1. slicer://scene

Current MRML scene structure.

**Returns:**
```json
{
  "scene_id": "vtkMRMLScene",
  "modified_time": "2025-11-26T10:30:00Z",
  "node_count": 15,
  "nodes": [
    {"id": "vtkMRMLScalarVolumeNode1", "name": "Brain-MRI", "type": "vtkMRMLScalarVolumeNode"}
  ],
  "connections": [
    {"source": "vtkMRMLScalarVolumeNode1", "target": "vtkMRMLSliceNode1", "type": "display"}
  ]
}
```

**Use cases:**
- Understanding scene state before operations
- Identifying available nodes for processing
- Tracking scene modifications

---

## 2. slicer://volumes

Loaded imaging volumes with metadata.

**Returns:**
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
    }
  ],
  "total_count": 1
}
```

**Use cases:**
- Identifying available volumes for analysis
- Checking image properties before processing
- Validating data loaded correctly

---

## 3. slicer://status

Health status and connection information.

**Returns:**
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

**Use cases:**
- Verifying Slicer connection before operations
- Debugging connection issues
- Monitoring system health

---

## Resource Access Pattern

Resources are read-only and fetched fresh on each request (no caching in MVP):

```python
@mcp.resource("slicer://status")
def get_status() -> dict:
    """Slicer connection status."""
    client = get_client()
    return client.health_check()
```

**HTTP calls:** Resources use `GET /slicer/mrml/*` and `POST /slicer/exec` internally.
