# Code Reference - Quick Navigation

**Quick lookup guide for developers working on Celestial Chaos**

---

## File Structure

```
3-body-problem-v8/
├── render_organic.py           # Main rendering script (HEAVILY ANNOTATED)
├── render_parallel.py           # Simpler/faster renderer (minimal effects)
├── render_epic.py              # Earlier version (reference)
├── serve_video.py              # HTTP server for streaming
├── deploy-to-gpu.sh            # Deploy to GH200
├── bootstrap.sh                # GH200 setup
│
├── src/
│   ├── physics/
│   │   ├── kernels.py          # Warp GPU kernels
│   │   ├── nbody.py            # Simulation manager
│   │   └── boundary_control.py # Soft containment
│   │
│   ├── audio/
│   │   ├── analyzer.py         # Basic 3-band analyzer
│   │   └── analyzer_10band.py  # Advanced 10-band analyzer
│   │
│   ├── rendering/
│   │   ├── organic_renderer.py          # Sophisticated renderer (THE ONE)
│   │   ├── advanced_color_system.py     # Multi-dimensional color mapping
│   │   ├── fast_shader_renderer.py      # Optimized version
│   │   └── shader_renderer.py           # Full shader effects (slow)
│   │
│   └── server/
│       ├── hls_stream.py       # HLS streaming server
│       └── fast_hls.py         # Optimized HLS
│
├── docs/
│   ├── ARCHITECTURE.md         # System architecture (THIS IS COMPREHENSIVE)
│   ├── ENHANCEMENT_ROADMAP.md  # Future improvements (PRIORITY-ORDERED)
│   └── CODE_REFERENCE.md       # This file
│
├── assets/audio/
│   └── still-night.mp3         # Test audio (Pretty Lights, 5.5 min)
│
└── output/
    └── organic.mp4             # Rendered videos
```

---

## Key Functions Reference

### render_organic.py

#### `precompute_physics_with_boundaries(audio_path, n_bodies, fps, output_dir)`
**Lines:** 105-239
**Purpose:** Stage 1 - GPU physics simulation with audio reactivity
**Returns:** Path to pickled physics state file (~1.1 GB)

**Key sections:**
- **Audio setup** (L144-150): Initialize 10-band analyzer
- **GPU init** (L152-160): Warp device detection
- **Simulation config** (L162-176): N-body parameters
- **Initial conditions** (L178-187): Figure-8 or random
- **Main loop** (L196-297): Physics + boundary control + state capture
- **Serialization** (L299-315): Save to pickle

**Performance:** ~887 fps on GH200 GPU

#### `render_frame_worker_organic(args)`
**Lines:** 318-393
**Purpose:** Worker function for parallel rendering
**Returns:** Frame index (for progress tracking)

**Process:**
1. Create OrganicRenderer instance
2. Render single frame (all visual layers)
3. Save PNG to disk
4. **Critical:** Close matplotlib figure (prevent memory leak)
5. Return frame index

**Memory:** ~50 MB per worker (must close figure!)

#### `parallel_render_organic(states_file, width, height, output_dir, n_workers)`
**Lines:** 396-490
**Purpose:** Stage 2 - Orchestrate parallel rendering
**Returns:** Path to frames directory

**Strategy:**
- Load physics state (1.1 GB, shared read-only)
- Create task list (one per frame)
- Spawn worker pool (32 processes)
- Distribute tasks via imap_unordered
- Progress reporting every 100 frames

**Performance:** ~119 fps on 32 cores (GH200)

#### `main()`
**Lines:** 493-629
**Purpose:** CLI interface and 3-stage orchestration

**Stages:**
1. GPU physics simulation
2. Parallel CPU rendering
3. Video encoding (ffmpeg)

**CLI args:**
- `--bodies/-n`: Particle count (default: 50)
- `--resolution/-r`: 720p/1080p/4k (default: 1080p)
- `--quality/-q`: draft/good/best (default: good)
- `--workers/-w`: Render processes (default: 32)
- `--audio/-a`: Audio file path
- `--output/-o`: Output video path

---

### src/physics/kernels.py

#### `compute_accelerations(positions, masses, accelerations, G, softening, n)`
**Lines:** 14-53
**Purpose:** N-body gravitational force calculation (O(N²))
**Device:** GPU (Warp kernel)

**Algorithm:**
```
For each particle i:
    acc_i = 0
    For each particle j (j ≠ i):
        r = pos_j - pos_i
        dist = |r| + softening
        acc_i += G * m_j * r / dist³
```

**Softening:** Prevents singularities at close approach

#### `integrate_verlet(positions, velocities, accelerations, dt, n)`
**Lines:** 56-84
**Purpose:** Velocity Verlet integration (first half-step)

**Algorithm:**
```
v(t+dt/2) = v(t) + a(t) * dt/2
x(t+dt) = x(t) + v(t+dt/2) * dt
```

#### `integrate_verlet_finish(velocities, accelerations, dt, n)`
**Lines:** 87-111
**Purpose:** Velocity Verlet integration (second half-step)

**Algorithm:**
```
v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
```

**Why two-step?** Allows acceleration recomputation at new positions (symplectic)

#### `update_trails(positions, velocities, trail_buffer, trail_velocities, trail_head, trail_length, n)`
**Lines:** 114-146
**Purpose:** Update particle trail ring buffers
**Storage:** 2000 points per particle (position + velocity)

#### `apply_audio_modulation(accelerations, bass_energy, mid_energy, treble_energy, modulation_depth, n)`
**Lines:** 192-224
**Purpose:** Modulate accelerations based on audio
**Effect:** Bass increases gravity strength

---

### src/physics/nbody.py

#### `class NBodySimulation`
**Purpose:** High-level N-body simulation manager

**Key methods:**
- `__init__(config)`: Setup simulation with config
- `initialize_random(pos_range, vel_range, seed)`: Random initial conditions
- `initialize_figure_eight(scale)`: Classic 3-body stable orbit
- `step(audio_params)`: Single timestep with audio modulation
- `get_positions()`, `get_velocities()`: Extract state from GPU
- `get_trails()`: Extract trail buffers

**Internal state:**
- `positions`: Warp array (vec3)
- `velocities`: Warp array (vec3)
- `accelerations`: Warp array (vec3)
- `masses`: Warp array (float32)
- `trail_buffer`: Ring buffer (n_bodies × trail_length × 3)

---

### src/physics/boundary_control.py

#### `apply_soft_boundary(positions, velocities, accelerations, boundary_radius, strength)`
**Purpose:** Soft inward force when particles near edge

**Algorithm:**
```python
for particle in particles:
    dist = |position|
    if dist > boundary_radius:
        excess = dist - boundary_radius
        direction = -position / dist  # Inward
        force = direction * strength * excess²
        acceleration += force
```

**Parameters:**
- `boundary_radius = 2.3` (frame edge at ±3)
- `strength = 0.3` (gentle push)

#### `apply_velocity_damping(velocities, positions, boundary_radius, damping)`
**Purpose:** Slow down particles moving outward near boundary

**Algorithm:**
```python
for particle in particles:
    dist = |position|
    if dist > boundary_radius:
        if dot(velocity, position) > 0:  # Moving outward
            velocity *= (1 - damping)
```

**Parameters:**
- `damping = 0.01` (1% reduction per substep)

---

### src/audio/analyzer_10band.py

#### `class AudioAnalyzer10Band`
**Purpose:** Decompose audio into 10 frequency bands + beat detection

**Frequency bands:**
```python
BANDS = {
    'sub_bass': (20, 60),      # Kick fundamentals
    'bass': (60, 250),         # Bass guitar
    'low_mid': (250, 500),     # Low instruments
    'mid': (500, 2000),        # Vocals
    'high_mid': (2000, 4000),  # Clarity
    'presence': (4000, 6000),  # Intelligibility
    'brilliance': (6000, 10000),  # Cymbals
    'air': (10000, 14000),     # Ambience
    'ultra': (14000, 18000),   # Overtones
    'extreme': (18000, 20000)  # Barely audible
}
```

**Key methods:**
- `get_frame(frame_idx)`: Returns `AudioFrame10Band` with all band energies
- `get_summary()`: Statistics about audio content

**Beat detection:**
```python
energy_flux = current_energy - avg_energy
beat_strength = max(0, energy_flux / std_energy)
```

---

### src/rendering/organic_renderer.py

#### `class OrganicRenderer`
**Purpose:** Sophisticated multi-layer rendering

**Rendering layers (in order):**
1. `_render_layered_background()` - Deep space gradient + nebula clouds + star field
2. `_render_spectral_decomposition()` - Frequency band visualization
3. `_render_interactions()` - Particle interaction glow fields
4. `_render_flowing_trail()` - Spline-smoothed trails (1500 points)
5. `_render_particle_organic()` - Multi-layer particle glow
6. `_render_minimal_ui()` - Time + beat indicator

**Key methods:**
- `render_frame(positions, velocities, ...)`: Main rendering entry point
- `_render_layered_background(audio_frame, time)`: 4-layer background
- `_render_flowing_trail(trail_pos, trail_vel, ...)`: Spline-smoothed trails
- `_render_particle_organic(pos, vel, acc, ...)`: 6-layer glow particles

**Visual techniques:**
- Spline smoothing (SciPy splprep/splev)
- Multi-layer glow (6 layers, decreasing alpha)
- Color gradients along trails
- Audio-reactive everything

---

### src/rendering/advanced_color_system.py

#### `class AdvancedColorSystem`
**Purpose:** Multi-dimensional physics-to-color mapping

**Color dimensions:**
1. **Velocity → Hue:** `atan2(vy, vx)` mapped to 0-360°
2. **Speed → Saturation:** Faster = more vibrant
3. **Kinetic energy + audio → Brightness:** High energy = bright
4. **Acceleration → Hue shift:** Changing direction = color shift
5. **Audio bands → Hue modulation:** Different frequencies shift hue
6. **Particle age → Evolution:** Colors pulse/breathe over time
7. **Position → Palette:** Spatial color regions

**Key methods:**
- `get_particle_color(pos, vel, acc, mass, age, audio)`: Complete color calculation
- `velocity_to_hue(velocity)`: 3D velocity → hue (0-360°)
- `speed_to_saturation(speed, audio_energy)`: Speed + audio → saturation
- `energy_to_value(kinetic_energy, audio_brilliance)`: Energy → brightness
- `get_trail_color_gradient(trail_pos, trail_vel, audio)`: Color gradient for trails

**Result:** Billions of unique colors from continuous parameters

---

## Common Workflows

### Run a Quick Test (720p Draft)
```bash
python render_organic.py -r 720p -q draft -w 16 --bodies 25
# ~3-4 minutes on GH200
```

### Production Quality (1080p Good)
```bash
python render_organic.py -r 1080p -q good -w 32 --bodies 50
# ~2 minutes on GH200
```

### Maximum Quality (4K Best)
```bash
python render_organic.py -r 4k -q best -w 64 --bodies 100
# ~15-20 minutes on GH200
```

### Test Different Audio
```bash
python render_organic.py -a assets/audio/your-track.mp3 -r 1080p
```

### Deploy to GH200
```bash
./deploy-to-gpu.sh <GPU_HOST_IP> --full --tunnel 8765
# Then view at http://<SERVER_IP>:8765
```

---

## Performance Tuning

### Workers
- **GH200 (72 cores):** Use 32-48 workers
- **Ryzen 5950X (16 cores):** Use 8-12 workers
- **Laptop (8 cores):** Use 4-6 workers

**Rule of thumb:** 50-75% of CPU cores (leave room for OS)

### Particle Count
- **3 bodies:** Classic 3-body problem, ~10,000 fps physics
- **50 bodies:** Sweet spot, ~900 fps physics, beautiful
- **100 bodies:** ~400 fps physics, stunning visuals
- **200+ bodies:** <200 fps physics, very slow but gorgeous

### Trail Length (in `precompute_physics_with_boundaries`)
```python
trail_length=2000  # Long, flowing (current)
trail_length=1000  # Shorter, faster rendering
trail_length=500   # Very short, 2x faster rendering
```

**Trade-off:** Shorter trails = less memory, faster render, but less dramatic visuals

---

## Debug Helpers

### Check Physics State Size
```bash
ls -lh /tmp/celestial_organic/physics_states_organic.pkl
# Should be ~1.1 GB for 10k frames, 50 bodies
```

### Monitor Rendering Progress
```python
# In parallel_render_organic(), change:
if completed % 100 == 0:  # Current
if completed % 10 == 0:   # More frequent updates
```

### Test Single Frame Rendering
```python
# In render_frame_worker_organic(), add:
print(f"Rendering frame {frame_idx}...")
# See which frames are slow
```

### Profile Physics Stage
```python
# In precompute_physics_with_boundaries(), add:
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# ... physics loop ...
profiler.disable()
profiler.print_stats(sort='cumtime')
```

---

## Memory Monitoring

### Check Worker Memory Usage
```bash
# While rendering is running:
ps aux | grep python | grep render_organic
# RSS column shows memory per process
```

### Expected memory:
- Main process: ~1.2 GB (physics state loaded)
- Each worker: ~50 MB (matplotlib figure)
- Total peak: ~2.7 GB (1.2 + 32×0.05)

### If memory grows unbounded:
**Problem:** Workers not closing matplotlib figures
**Solution:** Check `plt.close(renderer.fig)` is being called

---

## Common Issues & Fixes

### Issue: "Out of memory" during rendering
**Cause:** Workers not closing figures OR too many workers
**Fix:**
```python
# Ensure this is in render_frame_worker_organic():
plt.close(renderer.fig)

# Or reduce workers:
python render_organic.py -w 16  # Instead of 32
```

### Issue: Physics very slow (<100 fps)
**Cause:** Boundary control overhead (GPU↔CPU transfers)
**Fix:** Implement GPU boundary control (see ENHANCEMENT_ROADMAP.md)

### Issue: Particles escape frame
**Cause:** Boundary control parameters too weak
**Fix:**
```python
# In precompute_physics_with_boundaries():
acc = apply_soft_boundary(pos, vel, acc, boundary_radius=2.0, strength=0.5)
#                                         ↑ Smaller        ↑ Stronger
```

### Issue: Video encoding fails
**Cause:** ffmpeg not found OR frame files missing
**Fix:**
```bash
# Check ffmpeg:
which ffmpeg

# Check frames:
ls /tmp/celestial_organic/frames/
# Should have frame_00000.png through frame_NNNNN.png
```

---

## Testing Checklist

Before committing changes:

- [ ] Physics runs without errors
- [ ] Rendering completes without memory leaks
- [ ] Video encoding succeeds
- [ ] Final video plays correctly
- [ ] Audio is synchronized
- [ ] Particles stay in frame (boundary control works)
- [ ] Colors look good (not garish or washed out)
- [ ] Performance is acceptable (< 5 min for 1080p/5min video)

---

*End of Code Reference*
