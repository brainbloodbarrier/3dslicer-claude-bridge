# Performance Benchmarks Reference

Expected latencies and performance characteristics.

## Expected Response Times

### Screenshot Capture

| View Type | Average | P95 | Notes |
|-----------|---------|-----|-------|
| Axial (Red) | 50-100ms | 150ms | 512x512 PNG |
| Sagittal (Yellow) | 50-100ms | 150ms | 512x512 PNG |
| Coronal (Green) | 50-100ms | 150ms | 512x512 PNG |
| 3D View | 100-200ms | 300ms | Depends on scene |
| Full Layout | 200-400ms | 600ms | All views |

### Volume Measurement

| Segmentation Size | Voxel Count | Average | P95 |
|-------------------|-------------|---------|-----|
| Small | <10k | 300-500ms | 800ms |
| Medium | 10k-100k | 500ms-2s | 3s |
| Large | 100k-500k | 2-4s | 6s |
| Very Large | 500k-1M | 4-8s | 12s |

### Scene Operations

| Operation | Average | P95 |
|-----------|---------|-----|
| Health check | 10-30ms | 50ms |
| List scene nodes | 50-200ms | 400ms |
| Execute Python (simple) | 20-50ms | 80ms |
| Execute Python (complex) | 100-500ms | 1s |
| Load sample data | 2-10s | 30s |
| Set layout | 50-150ms | 300ms |

### Network Overhead (localhost)

| Component | Overhead |
|-----------|----------|
| HTTP request/response | 1-5ms |
| JSON serialization | 1-10ms |
| Base64 encoding | 5-20ms |
| **Total per request** | 10-35ms |

## Retry Behavior

Connection errors retry with exponential backoff:

```
Attempt 1: Immediate
Attempt 2: +1s (total: 1s)
Attempt 3: +2s (total: 3s)
Attempt 4: +4s (total: 7s)
Final Error: ~7s from start
```

**Timeouts are NOT retried** - Slicer may be frozen.

## Performance Tips

### Screenshots
- Capture only the view you need (not "full")
- Hide unnecessary nodes before capture
- Lower rendering quality for faster captures

### Volume Measurements
- Pass `segment_name` to measure specific segments
- Cache results for repeated measurements
- Simplify segmentations if precision allows

### Scene Operations
- Batch operations into single `execute_python` calls
- Cache node IDs instead of repeated lookups
- Filter by node type instead of listing all

## Run Benchmarks

```bash
# With Slicer running
uv run pytest tests/benchmarks/ -v -s -m integration --durations=0
```
