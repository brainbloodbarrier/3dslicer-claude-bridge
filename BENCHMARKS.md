# Performance Benchmarks

This document provides performance expectations and benchmarking guidance for the 3D Slicer MCP Bridge.

> **Note**: Values in this document are **estimated targets** based on typical hardware.
> Run `pytest tests/benchmarks/ -v -s` with a live Slicer connection to measure
> actual performance on your system.

## Test Environment Requirements

### Minimum Requirements
- **CPU**: Any modern multi-core processor
- **RAM**: 8 GB (16 GB recommended for large datasets)
- **GPU**: OpenGL 3.2+ capable (for 3D view rendering)
- **Disk**: SSD recommended for sample data loading

### Software Requirements
- **3D Slicer**: Version 5.0.0 or higher (tested with 5.6.2)
- **Python**: 3.10+
- **Network**: localhost (127.0.0.1) communication

## Expected Performance

### Screenshot Capture

| View Type | Average Latency | P95 Latency | Output Size | Notes |
|-----------|-----------------|-------------|-------------|-------|
| Axial (Red) | 50-100ms | 150ms | ~50-200 KB | 512x512 PNG |
| Sagittal (Yellow) | 50-100ms | 150ms | ~50-200 KB | 512x512 PNG |
| Coronal (Green) | 50-100ms | 150ms | ~50-200 KB | 512x512 PNG |
| 3D View | 100-200ms | 300ms | ~100-500 KB | Depends on scene complexity |
| Full Layout | 200-400ms | 600ms | ~200-800 KB | Captures all views |

**Factors affecting screenshot performance:**
- Scene complexity (number of visible structures)
- GPU rendering capabilities
- Window/viewport size
- 3D rendering quality settings

### Volume Measurement

| Segmentation Size | Voxel Count | Average Latency | P95 Latency | Notes |
|-------------------|-------------|-----------------|-------------|-------|
| Small | <10,000 | 300-500ms | 800ms | Simple structures (small lesions) |
| Medium | 10k-100k | 500ms-2s | 3s | Typical organs (liver, kidney) |
| Large | 100k-500k | 2-4s | 6s | Large structures (lungs, brain) |
| Very Large | 500k-1M | 4-8s | 12s | Whole-body segmentations |
| Massive | >1M | 8-15s | 20s | Multiple detailed structures |

**Factors affecting volume measurement:**
- Number of segments in the segmentation
- Voxel count per segment
- Available system memory
- Slicer's SegmentStatistics module initialization

### Scene Operations

| Operation | Average Latency | P95 Latency | Notes |
|-----------|-----------------|-------------|-------|
| Health check | 10-30ms | 50ms | Simple HTTP GET + response parsing |
| List scene nodes | 50-200ms | 400ms | Varies with node count |
| Execute Python (simple) | 20-50ms | 80ms | e.g., `print("hello")` |
| Execute Python (complex) | 100-500ms | 1s | Data processing operations |
| Load sample data | 2-10s | 30s | Network download + file loading |
| Set layout | 50-150ms | 300ms | UI reconfiguration |

### Network Overhead

Since communication is over localhost HTTP:

| Component | Typical Overhead |
|-----------|------------------|
| HTTP request/response | 1-5ms |
| JSON serialization | 1-10ms (depends on payload size) |
| Base64 encoding (screenshots) | 5-20ms |
| Total network overhead | 10-35ms per request |

## Retry and Timeout Behavior

### Retry Configuration
- **Max retries**: 3 attempts (configurable via constants)
- **Backoff**: Exponential (1s, 2s, 4s)
- **Total wait on failure**: Up to 7 seconds before final failure

### Timeout Settings
- **Default timeout**: 30 seconds per request
- **Configurable via**: `SLICER_TIMEOUT` environment variable

### Failure Timeline
```
Attempt 1: Immediate
    ↓ (failure)
Wait: 1 second
Attempt 2: +1s
    ↓ (failure)
Wait: 2 seconds
Attempt 3: +3s total
    ↓ (failure)
Wait: 4 seconds
Attempt 4: +7s total
    ↓ (failure)
Final Error: ~7s from start
```

## Performance Tips

### Optimizing Screenshot Capture
1. **Use specific views**: Capture only the view you need instead of "full"
2. **Reduce scene complexity**: Hide unnecessary nodes before capture
3. **Adjust quality settings**: Lower rendering quality in Slicer for faster captures

### Optimizing Volume Measurements
1. **Measure specific segments**: Pass `segment_name` instead of measuring all
2. **Pre-compute statistics**: For repeated measurements, cache the first result
3. **Simplify segmentations**: Use smoothing to reduce voxel count if precision allows

### Optimizing Scene Operations
1. **Batch operations**: Combine multiple operations into single `execute_python` calls
2. **Use node IDs directly**: Avoid repeated lookups by caching node IDs
3. **Filter node queries**: Request specific node types instead of listing all

## Benchmarking Your System

### Quick Health Check
```bash
# Time a simple health check
time python -c "
from slicer_mcp.slicer_client import get_client
client = get_client()
print(client.health_check())
"
```

### Screenshot Benchmark
```bash
# Time screenshot capture
time python -c "
from slicer_mcp.tools import capture_screenshot
result = capture_screenshot('axial')
print(f'Image size: {len(result[\"image_base64\"])} bytes')
"
```

### Running Benchmark Tests
```bash
# Run benchmark test suite (requires live Slicer connection)
pytest tests/benchmarks/ -v -s -m integration

# Run with timing output
pytest tests/benchmarks/ -v -s --durations=0 -m integration

# Run specific benchmark category
pytest tests/benchmarks/test_performance.py::TestScreenshotBenchmark -v -s
```

The benchmark tests will output detailed latency statistics including:
- Average, median, min, max latencies
- P95 percentile (95% of requests complete within this time)
- Standard deviation

## Performance Monitoring

### Enabling Audit Logging
```bash
export SLICER_AUDIT_LOG=./slicer_audit.log
# Audit log includes timestamps for performance analysis
```

### Log Analysis
```bash
# Extract timing information from audit log
cat slicer_audit.log | jq '.timestamp' | sort
```

## Known Performance Limitations

### Single-Threaded Operations
- Slicer WebServer processes requests sequentially
- Concurrent requests queue and execute one at a time
- Large operations can block subsequent requests

### Memory Constraints
- Large segmentations may require significant memory
- Base64 encoding temporarily doubles image memory usage
- Consider Slicer's memory settings for large datasets

### First-Request Latency
- First request after Slicer startup may be slower (module initialization)
- SegmentStatistics module lazy-loads on first volume measurement
- Sample data download includes network latency

## Comparison: Expected vs Actual

Use this table to compare your system's performance:

| Operation | Expected | Your Result | Status |
|-----------|----------|-------------|--------|
| Health check | <50ms | ___ms | [ ] OK |
| Screenshot (2D) | <150ms | ___ms | [ ] OK |
| Screenshot (3D) | <300ms | ___ms | [ ] OK |
| Volume (small) | <800ms | ___ms | [ ] OK |
| Volume (medium) | <3s | ___ms | [ ] OK |
| Sample data load | <30s | ___ms | [ ] OK |

If your results significantly exceed expected values, check:
1. Slicer version compatibility
2. System resource availability (CPU, memory, GPU)
3. Network configuration (firewall, proxy)
4. Slicer WebServer extension status
