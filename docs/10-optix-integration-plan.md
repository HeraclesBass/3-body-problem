# OptiX Integration Plan for Celestial Chaos

## Research Completed (2026-02-03)

### Python Bindings Decision

**Choice: otk-pyoptix** (Official NVIDIA)
- Supports OptiX 7.6+ through 9.0
- Most maintained option
- Direct NVIDIA support

**Installation:**
```bash
pip install pyoptix
```

### Architecture: Warp → CUDA → OptiX Bridge

```
┌──────────────────────────────────────────────────────┐
│              CELESTIAL CHAOS PIPELINE                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Audio (librosa) → Physics Parameters               │
│         ↓                                            │
│  Warp N-body Simulation (CUDA kernels)              │
│         ↓                                            │
│  GPU Memory (positions, velocities, trails)         │
│         ↓                                            │
│  CUDA Device Pointer Bridge (zero-copy)             │
│         ↓                                            │
│  OptiX Geometry (custom intersector)                │
│         ↓                                            │
│  OptiX Ray Tracing (volumetric shader)              │
│         ↓                                            │
│  Frame Buffer → ffmpeg → Video                      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Key Insight:** GH200 unified memory makes this FAST
- No CPU↔GPU copies needed
- Warp and OptiX share same physical memory
- Zero-copy particle data access

### Reference Implementation

**optixParticleVolumes** (NVIDIA official sample)
- Location: https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixParticleVolumes
- Demonstrates: RBF volume rendering with 1M particles
- Technique: Custom intersection program for volumetric particles
- Perfect starting point for our bioluminescent spheres

## Implementation Phases

### Phase 1: OptiX Setup & Hello World (1-2 days)

**Tasks:**
1. Install OptiX SDK 8.1+ on GH200
2. Install otk-pyoptix Python bindings
3. Create minimal OptiX program:
   - Initialize OptiX context
   - Create simple camera
   - Render single volumetric sphere
   - Output PNG image

**Success Criteria:**
- ✅ OptiX renders a glowing sphere
- ✅ Python bindings working
- ✅ Frame buffer export functional

### Phase 2: Warp-OptiX Bridge (2-3 days)

**Tasks:**
1. Design data structure for particle array
2. Implement CUDA device pointer sharing:
   ```python
   # Warp side
   warp_positions = wp.array(...)  # GPU memory
   cuda_ptr = warp_positions.ptr   # Get CUDA device pointer

   # OptiX side
   optix_buffer = optix.Buffer(cuda_ptr, ...)  # Wrap pointer
   ```
3. Create OptiX custom intersection program for particles
4. Test with static 3-body positions

**Success Criteria:**
- ✅ Warp particle data visible in OptiX
- ✅ Zero-copy confirmed (no performance hit)
- ✅ 3 spheres render correctly

### Phase 3: Volumetric Shader (3-4 days)

**Tasks:**
1. Implement bioluminescent emission shader:
   ```glsl
   // Pseudocode
   float glow_intensity = emission / (distance * distance);
   vec3 color = mix(cold_color, hot_color, velocity_norm);
   alpha = exp(-age / decay_rate);
   ```
2. Add soft falloff (no hard edges)
3. Implement trail rendering (volumetric tubes)
4. Audio-reactive parameters (bass → glow intensity)

**Success Criteria:**
- ✅ Organic bioluminescent aesthetic achieved
- ✅ Trails render beautifully
- ✅ Matches spec color palette

### Phase 4: Integration & Performance (2-3 days)

**Tasks:**
1. Replace matplotlib renderer with OptiX
2. Benchmark FPS (target: 100+ fps for 3-body)
3. Test scale-up (100, 1000 particles)
4. Memory profiling
5. Export 4K test render

**Success Criteria:**
- ✅ 60+ fps for 1080p @ 1000 particles
- ✅ 4K render completes successfully
- ✅ Visual quality matches artistic vision

### Phase 5: Polish & Features (ongoing)

**Tasks:**
- HDR rendering + bloom post-processing
- Motion blur
- Depth of field
- Camera animation system
- Multi-act composition tools

## Technical Details

### OptiX Geometry Types

For particle rendering, we'll use **Custom Primitives** with intersection program:

```python
# Custom intersection for volumetric sphere
def intersect_sphere(ray, particle):
    # Ray-sphere intersection
    # Return t_near, t_far for volume integration
    pass

# Closest-hit program (volumetric integration)
def shade_volume(ray, t_near, t_far, particle):
    # Integrate emission along ray
    # Apply bioluminescent falloff
    # Return RGBA
    pass
```

### Memory Layout

```python
# Warp physics output (GPU)
struct ParticleData {
    vec3 position;
    vec3 velocity;
    float mass;
    float age;
}

# OptiX reads directly from this buffer
# GH200 unified memory = no copy!
```

### Performance Estimates

| Config | Particles | Resolution | Target FPS |
|--------|-----------|------------|------------|
| 3-body | 3 | 4K | 120 fps |
| Demo | 100 | 1080p | 60 fps |
| Hero | 1000 | 4K | 30 fps |
| Extreme | 10000 | 8K | 5 fps (offline) |

GH200 bandwidth (900 GB/s) + RT cores = fast volumetric rendering

## Open Questions

1. **OptiX version on Lambda GH200?**
   - Need to check what's available
   - May need to request SDK installation

2. **Denoiser integration?**
   - OptiX has built-in AI denoiser
   - Could enable higher quality at lower sample counts

3. **Multi-GPU (8x H100)?**
   - OptiX supports multi-GPU
   - Could distribute BVH for massive particle counts

## Dependencies

```txt
# Python packages
pyoptix>=0.1.0          # OptiX Python bindings
warp-lang>=1.0.0        # Already installed
numpy
pillow                  # Frame buffer I/O
```

```bash
# System requirements
OptiX SDK 8.1+          # Download from NVIDIA
CUDA Toolkit 12.x       # Should be on Lambda
Driver 570+             # ✅ Already installed
```

## References

- **OptiX Programming Guide**: https://raytracing-docs.nvidia.com/optix8/guide/
- **otk-pyoptix**: https://github.com/NVIDIA/otk-pyoptix
- **optixParticleVolumes**: https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixParticleVolumes
- **GVDB + OptiX**: https://developer.nvidia.com/gvdb-samples
- **GH200 Unified Memory**: See NVIDIA GH200 Grace Hopper documentation

---

**Next Steps:**
1. Install OptiX SDK on GH200
2. Test otk-pyoptix installation
3. Run optixParticleVolumes sample
4. Begin Phase 1 implementation
