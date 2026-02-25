# Implementation Roadmap

## Technology Stack Options

### Option A: Web-Based (Recommended for iteration speed)
```
Frontend:  Three.js / WebGL / WebGPU
Physics:   Custom JS/TS or WASM (Rust)
Shaders:   GLSL
Hosting:   Hercules platform (port 80XX)
```

**Pros**: Fast iteration, shareable, runs anywhere
**Cons**: Performance ceiling for massive particle counts

### Option B: Native Application
```
Engine:    Unreal Engine 5 / Unity
Physics:   C++ or native engine physics
Compute:   CUDA / OpenCL via Lambda
Output:    Real-time + offline rendering
```

**Pros**: Maximum performance, VR support
**Cons**: Longer iteration cycles, requires builds

### Option C: Python + GPU (Research-first)
```
Compute:   JAX / CuPy / Numba
Viz:       Vispy / Mayavi / Blender Python
Render:    Blender Cycles (Lambda GPU)
```

**Pros**: Rapid prototyping, Wolfram API integration
**Cons**: Less polished real-time experience

## Recommended: Hybrid Approach

```
Phase 1: Python prototype (physics validation)
Phase 2: Three.js real-time viewer
Phase 3: Lambda GPU for high-quality renders
Phase 4: Optional native port for VR/exhibition
```

---

## Phase 1: Foundation (Research + Prototype)

### 1.1 Physics Engine Core
- [ ] N-body gravitational simulation
- [ ] Symplectic integrator (Verlet or leapfrog)
- [ ] Energy conservation validation
- [ ] Wolfram API integration for equation verification

### 1.2 Basic Visualization
- [ ] 2D trajectory plotting (matplotlib/plotly)
- [ ] Potential field heatmap
- [ ] Lagrange point computation
- [ ] Chaos divergence demonstration

### 1.3 Deliverables
- Working simulator in Python
- Validation against Wolfram results
- First artistic renders (static images)

---

## Phase 2: Real-Time Engine

### 2.1 Three.js Setup
- [ ] Project scaffold (Vite + TypeScript)
- [ ] Camera controls (orbit, pan, zoom)
- [ ] Basic scene (stars, background)
- [ ] Performance monitoring

### 2.2 Physics in Browser
- [ ] Port integrator to TypeScript
- [ ] Web Worker for physics (non-blocking)
- [ ] Interpolation for smooth rendering

### 2.3 Trail System
- [ ] Geometry shader trail rendering
- [ ] Color mapping (velocity, time, body)
- [ ] Trail decay and fade
- [ ] GPU instancing for performance

### 2.4 Potential Visualization
- [ ] Marching cubes for isosurfaces
- [ ] Gradient descent particle system
- [ ] Heat map overlay mode
- [ ] Toggle between views

---

## Phase 3: Aharonov-Bohm Module

### 3.1 Wave Function Simulation
- [ ] 2D Schrödinger equation solver
- [ ] Split-step Fourier method
- [ ] Phase extraction and visualization

### 3.2 Solenoid Setup
- [ ] Compute A⃗ field analytically
- [ ] Electron beam splitting
- [ ] Phase evolution along paths

### 3.3 Interference Visualization
- [ ] Real-time interference pattern
- [ ] Solenoid toggle effect
- [ ] Phase-to-color mapping

---

## Phase 4: Polish + Output

### 4.1 Visual Polish
- [ ] Post-processing (bloom, DOF, motion blur)
- [ ] HDR rendering pipeline
- [ ] Color grading presets
- [ ] Audio reactivity (optional)

### 4.2 Output Pipeline
- [ ] Screenshot tool (4K, 8K)
- [ ] Video recording (WebCodecs or ffmpeg)
- [ ] Lambda batch rendering for sequences

### 4.3 Interactivity
- [ ] Initial condition presets
- [ ] Famous periodic orbits (figure-8, etc.)
- [ ] User-defined systems
- [ ] Shareable URLs

---

## Phase 5: Extensions (Optional)

### 5.1 VR Experience
- [ ] WebXR integration
- [ ] 3D potential landscape you can "walk" through
- [ ] Haptic feedback for field strength

### 5.2 ML Enhancement
- [ ] Neural upscaling (ESRGAN)
- [ ] Style transfer for artistic modes
- [ ] Predictive simulation (PINN)

### 5.3 Generative Art
- [ ] Parameter randomization
- [ ] Evolutionary algorithm for "interesting" systems
- [ ] Daily render automation

---

## File Structure (Proposed)

```
3-body-problem-v8/
├── README.md
├── CLAUDE.md
├── docs/
│   └── [documentation files]
├── research/
│   └── source-transcript.txt
├── python/
│   ├── requirements.txt
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── nbody.py
│   │   ├── integrators.py
│   │   └── potentials.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py
│   │   └── renders.py
│   └── wolfram/
│       ├── __init__.py
│       └── api.py
├── web/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── main.ts
│   │   ├── physics/
│   │   ├── rendering/
│   │   └── ui/
│   └── public/
│       └── shaders/
├── shaders/
│   ├── trail.vert
│   ├── trail.frag
│   ├── potential.frag
│   └── interference.frag
└── renders/
    └── [output images/videos]
```

---

## Lambda GPU Integration

### Setup
```bash
# gpu-bridge project connection
export LAMBDA_API_KEY=...
export LAMBDA_INSTANCE_TYPE=gpu_1x_a100
```

### Use Cases
1. **Offline Rendering**: High-quality 8K frames
2. **Batch Simulation**: Monte Carlo chaos analysis
3. **ML Training**: Style transfer models
4. **Video Encoding**: Real-time to ProRes

---

## Milestones

| Milestone | Deliverable | Hardware |
|-----------|-------------|----------|
| M1 | Python prototype with Wolfram validation | Local CPU |
| M2 | Three.js real-time viewer (basic) | Local + browser |
| M3 | Trail system + potential viz | Local |
| M4 | Aharonov-Bohm module | Local |
| M5 | High-quality render pipeline | Lambda GPU |
| M6 | Public release / exhibition | All |

---

## Next Steps

1. **Choose primary stack** (recommend: Python → Three.js hybrid)
2. **Set up Python environment** with Wolfram API
3. **Implement basic N-body simulation**
4. **Generate first artistic renders**
5. **Iterate based on visual feedback**

Ready to begin when you provide direction.
