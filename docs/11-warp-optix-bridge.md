# Warp-OptiX Data Bridge Architecture

## Problem Statement

**Challenge:** NVIDIA Warp (physics) and NVIDIA OptiX (rendering) are separate frameworks with no built-in integration.

**Goal:** Create efficient zero-copy bridge to pass particle data from Warp simulation to OptiX renderer.

## Solution: CUDA Device Pointer Sharing

### High-Level Flow

```python
# 1. Warp allocates particle data on GPU
sim = NBodySimulation(device="cuda:0")
sim.step()  # Update physics

# 2. Get CUDA device pointer from Warp array
positions_gpu = sim.positions.ptr  # uint64 CUDA pointer

# 3. OptiX wraps the same pointer (zero-copy!)
optix_buffer = optix.Buffer.from_cuda_pointer(
    ptr=positions_gpu,
    shape=(n_particles,),
    dtype=optix.Vec3
)

# 4. OptiX renders directly from Warp's memory
optix.render(scene)
```

**Key Insight:** GH200 unified memory means this pointer points to shared CPU+GPU memory. No copies anywhere in the pipeline!

## Data Structures

### Warp Physics Output

```python
# Current Warp simulation state (GPU arrays)
class NBodySimulation:
    positions: wp.array(dtype=wp.vec3)      # [n, 3] float32
    velocities: wp.array(dtype=wp.vec3)     # [n, 3] float32
    masses: wp.array(dtype=wp.float32)      # [n] float32
    trail_buffer: wp.array(dtype=wp.vec3, ndim=2)  # [n, trail_len, 3]
    trail_head: wp.array(dtype=wp.int32)    # [n] int32
```

### OptiX Renderer Input

We need a **flattened particle structure** for OptiX custom geometry:

```cpp
// OptiX-compatible struct (matches Warp layout)
struct Particle {
    float3 position;    // 12 bytes
    float3 velocity;    // 12 bytes
    float mass;         // 4 bytes
    float _pad;         // 4 bytes (alignment)
};  // Total: 32 bytes (aligned)
```

**Problem:** Warp stores separate arrays. OptiX needs interleaved struct.

**Solution 1:** Keep separate buffers, use OptiX multi-buffer custom primitive
```python
# OptiX reads from multiple buffers
geometry.set_attribute("positions", positions_buffer)
geometry.set_attribute("velocities", velocities_buffer)
geometry.set_attribute("masses", masses_buffer)
```

**Solution 2:** Add Warp kernel to pack into interleaved format
```python
@wp.kernel
def pack_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    output: wp.array(dtype=ParticleStruct)
):
    i = wp.tid()
    output[i] = ParticleStruct(
        position=positions[i],
        velocity=velocities[i],
        mass=masses[i]
    )
```

**Recommendation:** Start with Solution 1 (simpler), optimize to Solution 2 if needed.

## OptiX Custom Geometry Integration

### Geometry Creation

```python
# Create OptiX geometry for particles
particle_geometry = optix.Geometry()
particle_geometry.set_primitive_count(n_particles)
particle_geometry.set_bounding_box_program(bbox_program)
particle_geometry.set_intersection_program(intersect_program)
```

### Bounding Box Program

```python
# OptiX program to compute AABB for each particle
def compute_particle_bbox(prim_index):
    pos = positions[prim_index]
    radius = compute_radius(masses[prim_index])

    return AABB(
        min=pos - radius,
        max=pos + radius
    )
```

### Intersection Program

```python
# Ray-sphere intersection for volumetric particle
def intersect_particle(ray, prim_index):
    pos = positions[prim_index]
    radius = compute_radius(masses[prim_index])

    # Ray-sphere math
    oc = ray.origin - pos
    a = dot(ray.direction, ray.direction)
    b = 2.0 * dot(oc, ray.direction)
    c = dot(oc, oc) - radius * radius

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return False  # No intersection

    t_near = (-b - sqrt(discriminant)) / (2*a)
    t_far = (-b + sqrt(discriminant)) / (2*a)

    report_intersection(t_near, t_far, prim_index)
    return True
```

### Volumetric Shading Program

```python
# Closest-hit program for bioluminescent rendering
def shade_particle_volume(ray, t_near, t_far, prim_index):
    pos = positions[prim_index]
    vel = velocities[prim_index]
    mass = masses[prim_index]

    # Sample points along ray through sphere
    num_samples = 16
    dt = (t_far - t_near) / num_samples

    accumulated_color = vec3(0)
    accumulated_alpha = 0

    for i in range(num_samples):
        t = t_near + i * dt
        sample_pos = ray.origin + t * ray.direction
        dist_from_center = length(sample_pos - pos)

        # Bioluminescent emission (spec equation)
        emission_strength = 1.0 / (dist_from_center * dist_from_center + 0.01)

        # Velocity-based color (slow = cyan, fast = magenta)
        speed = length(vel)
        color = lerp(CYAN, MAGENTA, speed / max_speed)

        # Audio reactivity
        glow_boost = 1.0 + audio_params.bass_energy * 2.0

        # Accumulate
        sample_emission = color * emission_strength * glow_boost
        sample_alpha = emission_strength * 0.1

        accumulated_color += sample_emission * dt
        accumulated_alpha += sample_alpha * dt

    return vec4(accumulated_color, min(accumulated_alpha, 1.0))
```

## Trail Rendering

Trails are trickier - need to render tubes connecting historical positions.

### Option 1: Particles for Each Trail Point

```python
# Treat each trail point as a small particle
# Total geometry count = n_bodies * trail_length
for body in range(n_bodies):
    for trail_idx in range(trail_length):
        pos = trail_buffer[body, trail_idx]
        age = compute_age(trail_head[body], trail_idx)
        alpha = exp(-age / decay_rate)
        # Render small sphere with age-based alpha
```

**Pros:** Simple, reuses particle shader
**Cons:** High primitive count (3 bodies × 1000 trail = 3000 primitives)

### Option 2: Custom Tube Geometry

```python
# Create curved tube connecting trail points
# Use Catmull-Rom spline for smoothness
for body in range(n_bodies):
    points = get_trail_points(body)
    tube = create_tube_geometry(points, radius=0.05)
    # Render with alpha gradient along length
```

**Pros:** Beautiful smooth trails, fewer primitives
**Cons:** More complex intersection math

**Recommendation:** Start with Option 1, upgrade to Option 2 for final quality.

## Memory Management

### Update Frequency

**Every Frame:**
- Particle positions (Warp updates, OptiX reads)
- Audio parameters (CPU → GPU constant buffer)

**Once at Init:**
- Geometry BVH structure
- Shader programs
- Texture maps

### Synchronization

```python
# Warp simulation runs in CUDA stream
warp_stream = wp.get_stream()

# OptiX rendering uses same stream (automatic sync)
optix.set_cuda_stream(warp_stream)

# No manual sync needed - CUDA handles it!
```

## Performance Optimization

### BVH Refit vs Rebuild

OptiX BVH (acceleration structure) options:

**Refit (fast):** Update bounding boxes, keep tree structure
- Use when particles move but topology unchanged
- ~10x faster than rebuild
- Good for smooth motion

**Rebuild (slow but optimal):** Reconstruct entire tree
- Use when particles rearrange significantly
- Better ray tracing performance
- Do every N frames if needed

```python
# Refit for most frames (fast update)
bvh.refit()  # ~1ms for 1000 particles

# Rebuild occasionally for quality
if frame % 100 == 0:
    bvh.rebuild()  # ~10ms but better perf after
```

### Multi-Stream Parallelism

```python
# Overlap physics and rendering
physics_stream = cuda.Stream()
render_stream = cuda.Stream()

# Frame N physics (stream 1)
with physics_stream:
    sim.step()

# Frame N-1 rendering (stream 2, uses previous data)
with render_stream:
    optix.render()
    export_frame()

# Sync both before frame buffer read
cuda.synchronize()
```

**Potential speedup:** 1.5-2x for compute-heavy physics

## GH200-Specific Optimizations

### Unified Memory Advantages

```python
# Traditional GPU (separate CPU/GPU memory):
# 1. Warp computes on GPU
# 2. Copy to CPU (slow!)
# 3. Copy to OptiX on GPU (slow!)

# GH200 unified memory:
# 1. Warp computes in unified memory
# 2. OptiX reads same memory (zero-copy!)
# No copies = massive speedup
```

### Memory Bandwidth

- GH200: 900 GB/s unified bandwidth
- H100: 3000 GB/s HBM3 (but needs copies)

For our use case (small particle counts, no copies), GH200 wins!

### Page Migration

GH200 automatically migrates memory pages between CPU/GPU:

```python
# If OptiX needs data, hardware migrates to GPU
# If CPU needs data for I/O, migrates to CPU
# Transparent to application!
```

## Code Structure

```
src/
├── physics/
│   ├── nbody.py          # Existing Warp simulation
│   └── kernels.py        # Existing Warp kernels
│
├── rendering/
│   ├── optix_context.py  # OptiX initialization
│   ├── optix_bridge.py   # Warp → OptiX data bridge
│   ├── particle_geom.py  # Particle geometry + shaders
│   └── trail_geom.py     # Trail geometry + shaders
│
└── pipeline/
    └── renderer.py       # Main render loop
```

## Testing Strategy

### Phase 1: Static Test
```python
# Render static 3-body positions
sim = NBodySimulation()
sim.initialize_figure_eight()

renderer = OptiXRenderer(sim)
image = renderer.render_frame()
image.save("test_static.png")

# Verify: 3 glowing spheres visible
```

### Phase 2: Animation Test
```python
# Render 100 frames of motion
for frame in range(100):
    sim.step()
    image = renderer.render_frame()
    save_frame(image, frame)

# Verify: Smooth motion, trails appear
```

### Phase 3: Performance Test
```python
# Measure FPS for different particle counts
for n in [10, 100, 1000, 10000]:
    sim = NBodySimulation(n_bodies=n)
    fps = benchmark_fps(sim, frames=100)
    print(f"{n} particles: {fps:.1f} fps")

# Target: 60+ fps for 1000 particles @ 1080p
```

## Next Steps

1. ✅ Install OptiX SDK on GH200
2. ✅ Test otk-pyoptix installation
3. ✅ Create minimal OptiX "hello sphere" program
4. ✅ Implement bridge for static positions
5. ✅ Add volumetric shader
6. ✅ Integrate with live Warp simulation
7. ✅ Benchmark and optimize

---

**References:**
- OptiX Custom Primitives: https://raytracing-docs.nvidia.com/optix8/guide/index.html#geometry
- CUDA-OptiX Interop: https://docs.nvidia.com/gameworks/content/gameworkslibrary/optix/optix__cuda__interop_8h.html
- optixParticleVolumes: https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixParticleVolumes
