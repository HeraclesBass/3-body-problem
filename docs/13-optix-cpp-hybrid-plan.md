# OptiX C++/Python Hybrid Implementation Plan

**Status:** Documented for future implementation
**Priority:** Deferred (focus on matplotlib improvements first)
**Estimated Effort:** 1-2 days development

## Overview

Since Python bindings have ARM64/version compatibility issues, we'll create a C++/CUDA OptiX renderer callable from Python. This gives us:
- ✅ Full OptiX 9.1 features
- ✅ Optimal performance
- ✅ Zero-copy Warp integration
- ✅ Complete control over rendering pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Python Layer                          │
├─────────────────────────────────────────────────────────┤
│  Warp Physics → CUDA Device Pointer                     │
│         ↓                                                │
│  ctypes/pybind11 FFI                                    │
│         ↓                                                │
├─────────────────────────────────────────────────────────┤
│                   C++/CUDA Layer                        │
├─────────────────────────────────────────────────────────┤
│  OptiX Context                                          │
│         ↓                                                │
│  Custom Geometry (particles from Warp pointer)          │
│         ↓                                                │
│  Ray Tracing Pipeline                                   │
│         ↓                                                │
│  Volumetric Shader (bioluminescent)                     │
│         ↓                                                │
│  Frame Buffer → Python (numpy array)                    │
└─────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Phase 1: C++ OptiX Renderer (2-3 hours)

**File:** `src/rendering/optix_renderer.cu`

```cpp
// Minimal OptiX 9.1 renderer in C++/CUDA

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

class OptiXRenderer {
public:
    OptiXRenderer(int width, int height);

    // Set particle data from Warp (zero-copy)
    void setParticles(float* positions_gpu,
                     float* velocities_gpu,
                     int n_particles);

    // Render frame
    void render(float* output_rgb);

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixModule module;

    CUdeviceptr particle_positions;
    CUdeviceptr particle_velocities;
    int n_particles;

    // Frame buffer
    CUdeviceptr frame_buffer;
    int width, height;
};
```

**Key Features:**
- Direct CUDA pointer sharing with Warp (zero-copy)
- OptiX 9.1 ray tracing pipeline
- Custom intersection for volumetric spheres
- Bioluminescent shader

### Phase 2: Python Bindings (1 hour)

**File:** `src/rendering/optix_wrapper.py`

```python
import ctypes
import numpy as np
import warp as wp

# Load compiled shared library
liboptix = ctypes.CDLL('./build/liboptix_renderer.so')

# Define C function signatures
liboptix.create_renderer.argtypes = [ctypes.c_int, ctypes.c_int]
liboptix.create_renderer.restype = ctypes.c_void_p

liboptix.set_particles.argtypes = [
    ctypes.c_void_p,  # renderer
    ctypes.c_void_p,  # positions (CUDA pointer)
    ctypes.c_void_p,  # velocities (CUDA pointer)
    ctypes.c_int      # n_particles
]

liboptix.render.argtypes = [
    ctypes.c_void_p,  # renderer
    ctypes.c_void_p   # output buffer
]

class OptiXRenderer:
    def __init__(self, width=1920, height=1080):
        self.renderer = liboptix.create_renderer(width, height)
        self.width = width
        self.height = height

    def set_particles(self, warp_positions, warp_velocities):
        """Zero-copy particle data from Warp arrays."""
        n = warp_positions.shape[0]

        # Get CUDA device pointers from Warp
        pos_ptr = warp_positions.ptr
        vel_ptr = warp_velocities.ptr

        liboptix.set_particles(
            self.renderer,
            ctypes.c_void_p(pos_ptr),
            ctypes.c_void_p(vel_ptr),
            n
        )

    def render(self):
        """Render frame and return as numpy array."""
        # Allocate output buffer
        output = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        liboptix.render(
            self.renderer,
            output.ctypes.data_as(ctypes.c_void_p)
        )

        return output
```

**Usage:**
```python
# In main render loop
from rendering.optix_wrapper import OptiXRenderer

renderer = OptiXRenderer(width=1920, height=1080)

for frame in range(total_frames):
    # Physics (Warp)
    sim.step()

    # Render (OptiX C++)
    renderer.set_particles(sim.positions, sim.velocities)
    img = renderer.render()

    # Save frame
    Image.fromarray(img).save(f'frame_{frame:04d}.png')
```

### Phase 3: OptiX Shaders (2-3 hours)

**File:** `src/rendering/optix_shaders.cu`

```cuda
// OptiX programs for bioluminescent volumetric particles

#include <optix.h>

extern "C" {
__constant__ Params params;
}

// Ray-sphere intersection
extern "C" __global__ void __intersection__sphere() {
    const int prim_idx = optixGetPrimitiveIndex();

    float3 center = params.positions[prim_idx];
    float radius = params.radii[prim_idx];

    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();

    // Ray-sphere intersection math
    float3 oc = ray_orig - center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - radius * radius;

    float discriminant = b*b - 4*a*c;

    if (discriminant >= 0) {
        float t_near = (-b - sqrtf(discriminant)) / (2*a);
        float t_far = (-b + sqrtf(discriminant)) / (2*a);

        if (t_near > 0) {
            optixReportIntersection(t_near, 0);
        }
    }
}

// Volumetric shading (bioluminescent)
extern "C" __global__ void __closesthit__volume() {
    const int prim_idx = optixGetPrimitiveIndex();

    float3 center = params.positions[prim_idx];
    float3 velocity = params.velocities[prim_idx];

    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float t = optixGetRayTmax();

    float3 hit_pos = ray_orig + t * ray_dir;
    float dist_from_center = length(hit_pos - center);

    // Bioluminescent glow (spec equation)
    float emission = 1.0f / (dist_from_center * dist_from_center + 0.01f);

    // Velocity-based color (slow = cyan, fast = magenta)
    float speed = length(velocity);
    float3 cold_color = make_float3(0.0f, 1.0f, 1.0f);  // Cyan
    float3 hot_color = make_float3(1.0f, 0.0f, 1.0f);   // Magenta

    float speed_norm = fminf(speed / params.max_speed, 1.0f);
    float3 color = lerp(cold_color, hot_color, speed_norm);

    // Audio reactivity
    float glow_boost = 1.0f + params.bass_energy * 2.0f;

    // Accumulate color
    float3 result = color * emission * glow_boost;

    // Set payload (output color)
    optixSetPayload_0(__float_as_uint(result.x));
    optixSetPayload_1(__float_as_uint(result.y));
    optixSetPayload_2(__float_as_uint(result.z));
}
```

### Phase 4: Build System (30 min)

**File:** `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18)
project(optix_renderer CUDA CXX)

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 90)  # GH200 = sm_90

add_library(optix_renderer SHARED
    src/rendering/optix_renderer.cu
    src/rendering/optix_shaders.cu
)

target_include_directories(optix_renderer PRIVATE
    ${OPTIX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(optix_renderer
    ${CUDA_LIBRARIES}
)
```

**Build:**
```bash
cd ~/3-body-problem-v8
mkdir build && cd build
cmake ..
make -j$(nproc)

# Creates: build/liboptix_renderer.so
```

## Performance Expectations

### GH200 Capabilities

| Particles | Resolution | FPS | Notes |
|-----------|------------|-----|-------|
| 3-10 | 4K | 120+ | Trivial load |
| 100 | 1080p | 60+ | Smooth real-time |
| 1,000 | 1080p | 30+ | High quality |
| 10,000 | 720p | 15+ | Cinematic |

**Why Fast:**
- GH200 RT cores (hardware ray tracing)
- Zero-copy Warp → OptiX
- Unified memory (900 GB/s bandwidth)
- Volumetric shading on GPU

## Testing Strategy

### Unit Tests

```bash
# Test 1: OptiX context creation
./build/test_optix_init

# Test 2: Particle data transfer
./build/test_particle_transfer

# Test 3: Single frame render
./build/test_render_frame
```

### Integration Test

```python
# Test with Warp physics
from rendering.optix_wrapper import OptiXRenderer
from physics.nbody import NBodySimulation

sim = NBodySimulation(n_bodies=100)
sim.initialize_random()

renderer = OptiXRenderer(1920, 1080)

for i in range(100):
    sim.step()
    renderer.set_particles(sim.positions, sim.velocities)
    img = renderer.render()
    assert img.shape == (1080, 1920, 3)
    assert img.dtype == np.uint8

print("✅ Integration test passed")
```

## Fallback Strategy

If C++ implementation proves too complex:

### Option B-Alt: OptiX Prime (Simpler)

Use OptiX Prime (ray tracing only, no full pipeline):
- Simpler API
- Python-friendly
- Still GPU-accelerated

### Option B-Alt2: Custom CUDA Renderer

Skip OptiX entirely, write pure CUDA ray tracer:
- Full control
- No SDK dependencies
- Educational value

## Future Enhancements

Once basic C++ renderer works:

1. **Trails** - Volumetric tubes connecting positions
2. **Motion blur** - Temporal accumulation
3. **Depth of field** - Lens simulation
4. **HDR + Bloom** - Post-processing pipeline
5. **Denoiser** - OptiX AI denoiser for quality

## Resources

- **OptiX Programming Guide:** https://raytracing-docs.nvidia.com/optix9/guide/
- **OptiX Samples:** `/home/ubuntu/NVIDIA-OptiX-SDK-9.1.0/.../SDK/`
- **Warp CUDA Interop:** https://nvidia.github.io/warp/modules/runtime.html#cuda-interop
- **pybind11 Tutorial:** https://pybind11.readthedocs.io/

## Decision Log

**2026-02-03:** Deferred in favor of matplotlib improvements
- Python bindings incompatible (ARM64 + OptiX 9.1)
- C++ hybrid is 1-2 day effort
- Want to ship beautiful content NOW
- Will revisit when quality ceiling hit

**Next Review:** After completing matplotlib enhancements and shipping first renders

---

**Status:** Ready to implement when needed
**Prerequisites:** OptiX SDK 9.1 ✅, GH200 access ✅, Warp working ✅
**Blockers:** None (voluntary deferral for faster iteration)
