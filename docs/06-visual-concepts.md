# Visual Concepts

## Artistic Vision

> Transform mathematical physics into visceral, beautiful experiences that reveal hidden structure.

## Core Visual Modules

### Module 1: Three-Body Ballet

**Description**: Real-time gravitational simulation with artistic rendering

**Elements**:
- Three celestial bodies with mass-proportional glow
- Trajectory trails (fading over time, color-coded by velocity)
- Gravitational potential landscape (3D terrain or color overlay)
- Lagrange point indicators

**Interactions**:
- Drag to set initial positions
- Velocity vectors shown during setup
- Time controls (pause, slow-mo, fast-forward)
- Chaos mode: show divergence of nearby initial conditions

**Technical**:
- GPU-accelerated physics (Lambda)
- Symplectic integrator for energy conservation
- Trail rendering via geometry shaders

---

### Module 2: Potential Landscape Explorer

**Description**: Interactive visualization of scalar potentials

**Elements**:
- 3D terrain representing potential V(x,y)
- Gradient arrows showing field direction
- Equipotential contour lines
- Test particle rolling on landscape

**Modes**:
- Single mass (creates well)
- Binary system (two wells + saddle points)
- N-body system (complex landscape)

**Visual Styles**:
- Topographic map (contour lines)
- Heat map (color gradient)
- Glass terrain (translucent 3D)
- Wireframe mesh

---

### Module 3: Vector Field Visualizer

**Description**: Magnetic and electric field visualization

**Elements**:
- Streamlines following field direction
- Particle systems advected by field
- Curl indicators (rotation visualization)
- Line Integral Convolution (LIC) for smooth fields

**Techniques**:
- GPU particle systems (millions of particles)
- Instanced line rendering
- Real-time field computation

---

### Module 4: Aharonov-Bohm Chamber

**Description**: Interactive quantum interference visualization

**Elements**:
- Electron beam splitter
- Solenoid with tunable current
- Wave function visualization (color = phase)
- Interference pattern display
- A⃗ field vectors (ghosted, semi-transparent)

**Interactions**:
- Toggle solenoid on/off
- Adjust solenoid strength
- Watch pattern shift in real-time

**Visual Style**:
- Neon/cyberpunk aesthetic
- Wave ripples with phase coloring
- Interference fringes as light bands

---

### Module 5: Double Pendulum Chaos

**Description**: Classic chaos demonstration

**Elements**:
- Physical pendulum simulation
- Trail drawing mode
- Phase space plot (θ₁, θ̇₁) and (θ₂, θ̇₂)
- Chaos divergence visualization

**Visual Modes**:
- Single pendulum with long trail
- Multiple pendulums (butterfly effect)
- Phase space portraits

---

### Module 6: Action Principle Gallery

**Description**: Visualize "nature choosing paths"

**Elements**:
- Start and end points
- All possible paths (faded)
- Actual path (highlighted)
- Action value displayed for each path

**Concept**:
Show why nature's path minimizes action - educational + beautiful.

---

## Color Palettes

### Cosmic
```
Background: #0a0a12 (deep space)
Primary:    #4a9eff (stellar blue)
Secondary:  #ff6b6b (red giant)
Accent:     #ffd93d (solar yellow)
Trail:      gradient from primary to secondary
```

### Quantum
```
Background: #000000 (void)
Phase 0°:   #ff0040 (red)
Phase 90°:  #00ff80 (green)
Phase 180°: #0080ff (blue)
Phase 270°: #ff00ff (magenta)
```

### Topographic
```
Low potential:  #001f3f (navy depth)
Mid potential:  #3d9970 (sea level)
High potential: #ff851b (mountain)
Peak:           #ffffff (snow)
```

## Rendering Techniques

### Trail Rendering
```glsl
// Geometry shader creates ribbons from particle history
// Alpha fades based on age
// Color encodes velocity magnitude
```

### Potential Field Raymarching
```glsl
// SDF for potential isosurfaces
// Volumetric rendering for density
// Caustics for light bending
```

### Interference Patterns
```glsl
// Wave equation solution
// Phase-to-color mapping
// Additive blending for superposition
```

## Output Formats

| Format | Resolution | Use Case |
|--------|------------|----------|
| Real-time | 1920x1080 60fps | Interactive exploration |
| 4K Video | 3840x2160 60fps | Exhibition display |
| 8K Still | 7680x4320 | Print / gallery |
| VR | 2x 2160x2160 | Immersive experience |

## Hardware Utilization

| Component | Task |
|-----------|------|
| 10-core CPU | Physics simulation, preprocessing |
| 128GB RAM | Large particle systems, texture caches |
| Lambda GPU | Real-time rendering, ML upscaling |

## Inspirations

- **Ryoji Ikeda**: Data-driven minimalism
- **Refik Anadol**: AI + physics visualizations
- **teamLab**: Immersive digital nature
- **NASA Visualizations**: Scientific accuracy + beauty
- **Beeple**: Daily render discipline
