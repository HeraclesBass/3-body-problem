# Enhancement Roadmap - Celestial Chaos

**Priority-Ordered Improvements for Next Sessions**

---

## Phase 1: Quick Wins (1-2 hours)

### 1.1 Increase Particle Count
**Goal:** Scale from 50 → 100 particles
**Impact:** More visual complexity, richer interactions
**Effort:** 5 minutes
**Trade-off:** Physics ~2x slower (still fast at ~400 fps)

```python
python render_organic.py --bodies 100 -r 1080p -q good
```

### 1.2 Motion Blur
**Goal:** Smooth fast motion with accumulation
**Impact:** Cinematic quality, reduces judder
**Effort:** 30 minutes
**Implementation:**
```python
# In render_frame_worker_organic()
# Accumulate multiple substeps per frame
blur_samples = 5
accumulated = np.zeros((height, width, 3), dtype=np.float32)
for sample in range(blur_samples):
    t_offset = sample / blur_samples
    # Interpolate state at t_offset
    rgb = renderer.render_frame(interpolated_state)
    accumulated += rgb.astype(np.float32)
accumulated /= blur_samples
return accumulated.astype(np.uint8)
```

### 1.3 Different Audio Tracks
**Goal:** Test with diverse music genres
**Impact:** Validate audio reactivity across styles
**Effort:** 10 minutes
**Suggestions:**
- Heavy bass (dubstep, trap) - Test gravity modulation
- Classical (orchestral) - Test harmonic color shifts
- Jazz (complex rhythms) - Test beat detection
- Ambient (sparse) - Test background evolution

```bash
python render_organic.py -a assets/audio/dubstep-drop.mp3
python render_organic.py -a assets/audio/beethoven-symphony.mp3
```

---

## Phase 2: Visual Enhancements (4-6 hours)

### 2.1 Chord Detection → Color Palette Shifts
**Goal:** Detect harmonic changes, shift color palette
**Impact:** Music-synchronized color evolution
**Effort:** 2 hours

**Approach:**
```python
# In AudioAnalyzer10Band
def detect_chord_changes(self):
    """Use chromagram to detect harmonic changes."""
    import librosa
    chromagram = librosa.feature.chroma_stft(y=self.audio, sr=self.sr)

    # Detect chord transitions (large chromagram changes)
    chord_flux = np.sum(np.diff(chromagram, axis=1)**2, axis=0)
    chord_changes = chord_flux > np.percentile(chord_flux, 90)

    return chord_changes

# In AdvancedColorSystem
def shift_palette(self, chord_change):
    """Shift base hue on chord changes."""
    if chord_change:
        self.palette_offset = (self.palette_offset + 60) % 360
```

### 2.2 Gravitational Lensing Effect
**Goal:** Space distortion around massive particles
**Impact:** Trippy, physics-accurate visual
**Effort:** 3 hours

**Approach:**
```python
# In OrganicRenderer
def apply_lensing(self, image, positions, masses):
    """Warp image based on gravitational field."""
    height, width = image.shape[:2]

    # Create displacement field
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    for pos, mass in zip(positions, masses):
        # Screen space position
        screen_x = (pos[0] + 3) / 6 * width
        screen_y = (pos[1] + 1.7) / 3.4 * height

        # Distance from particle
        dx = x - screen_x
        dy = y - screen_y
        r = np.sqrt(dx**2 + dy**2) + 1

        # Displacement (inverse square, like gravity)
        lensing_strength = mass * 100 / r**2
        x += dx * lensing_strength / r
        y += dy * lensing_strength / r

    # Remap image
    from scipy.ndimage import map_coordinates
    warped = map_coordinates(image, [y, x], order=1)
    return warped
```

### 2.3 Bloom/Glow Post-Process
**Goal:** Gaussian blur on bright areas
**Impact:** Dreamy, ethereal quality
**Effort:** 1 hour

**Approach:**
```python
from scipy.ndimage import gaussian_filter

def apply_bloom(self, image, threshold=200, sigma=10, strength=0.5):
    """Add bloom to bright areas."""
    # Extract bright areas
    bright = np.maximum(image.astype(float) - threshold, 0)

    # Blur bright areas
    bloomed = gaussian_filter(bright, sigma=(sigma, sigma, 0))

    # Composite
    result = image.astype(float) + bloomed * strength
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

## Phase 3: Audio Intelligence (6-8 hours)

### 3.1 Tempo-Synced Pulses
**Goal:** Lock particle pulses to BPM
**Impact:** Tight music synchronization
**Effort:** 2 hours

**Implementation:**
```python
# In AudioAnalyzer10Band
def detect_tempo(self):
    """Estimate BPM using beat tracking."""
    import librosa
    tempo, beats = librosa.beat.beat_track(y=self.audio, sr=self.sr)
    beat_times = librosa.frames_to_time(beats, sr=self.sr)
    return tempo, beat_times

# In OrganicRenderer
def render_particle_organic(self, position, ..., on_beat):
    """Pulse particle size/glow on beats."""
    if on_beat:
        radius *= 1.5  # Bigger on beat
        glow_intensity *= 2.0
```

### 3.2 Spectral Flux Analysis
**Goal:** Detect "novelty" - sudden audio changes
**Impact:** Visual accents on dramatic moments
**Effort:** 3 hours

**Implementation:**
```python
def compute_spectral_flux(self):
    """Measure rate of spectral change."""
    import librosa

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr)

    # Spectral flux = sum of positive differences
    flux = np.sum(np.maximum(np.diff(S, axis=1), 0), axis=0)

    # Normalize
    flux = (flux - np.mean(flux)) / np.std(flux)
    return flux

# Use for:
# - Particle bursts on novelty peaks
# - Camera shake on big moments
# - Sudden color shifts
```

### 3.3 Harmonic Content Analysis
**Goal:** Extract pitch information for color mapping
**Impact:** Different notes → different hues
**Effort:** 3 hours

**Implementation:**
```python
def extract_pitch_contour(self):
    """Extract fundamental frequency over time."""
    import librosa

    # Pitch tracking
    pitches, magnitudes = librosa.piptrack(y=self.audio, sr=self.sr)

    # Extract most prominent pitch per frame
    pitch_contour = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_contour.append(pitch)

    return np.array(pitch_contour)

# Map to hue:
# C = 0°, C# = 30°, D = 60°, ..., B = 330°
# Creates harmonic color relationships
```

---

## Phase 4: Physics Enhancements (8-12 hours)

### 4.1 GPU Boundary Control
**Goal:** Move boundary control to Warp kernel
**Impact:** 5-10x physics speedup
**Effort:** 4 hours
**Priority:** HIGH (biggest performance gain)

**Implementation:**
```python
# In src/physics/kernels.py
@wp.kernel
def apply_boundary_control(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    boundary_radius: wp.float32,
    force_strength: wp.float32,
    damping: wp.float32,
    n: wp.int32
):
    """Apply soft boundary forces and velocity damping on GPU."""
    i = wp.tid()
    if i >= n:
        return

    pos = positions[i]
    vel = velocities[i]
    acc = accelerations[i]

    # Distance from origin
    dist = wp.length(pos)

    if dist > boundary_radius:
        # Soft boundary force (inward)
        excess = dist - boundary_radius
        force_dir = -pos / dist  # Normalized inward
        boundary_force = force_dir * force_strength * excess * excess
        acc += boundary_force

        # Velocity damping (if moving outward)
        if wp.dot(vel, pos) > 0.0:
            vel = vel * (1.0 - damping)

    accelerations[i] = acc
    velocities[i] = vel

# In NBodySimulation.step()
wp.launch(
    kernel=apply_boundary_control,
    dim=self.n_bodies,
    inputs=[self.positions, self.velocities, self.accelerations, ...]
)
# No more GPU→CPU→GPU transfers!
```

**Expected result:** Physics 887 fps → 4000+ fps

### 4.2 Variable Gravity Functions
**Goal:** G(t) functions for dramatic effects
**Impact:** Gravity swells/drops for cinematic moments
**Effort:** 2 hours

**Examples:**
```python
def gravity_function_bass_drop(t, audio_frame):
    """Gravity increases during bass drops."""
    return 1.0 + audio_frame.sub_bass * 3.0

def gravity_function_buildup(t, audio_frame):
    """Gradual gravity increase (tension building)."""
    base = 1.0
    buildup = t / audio.duration  # 0 → 1 over song
    return base + buildup * 2.0

def gravity_function_oscillating(t, audio_frame):
    """Sinusoidal gravity (breathing effect)."""
    return 1.0 + np.sin(t * 0.5) * 0.5 + audio_frame.bass * 0.8
```

### 4.3 Particle Collisions
**Goal:** Particles can merge/split
**Impact:** Dynamic particle count, dramatic events
**Effort:** 6 hours

**Approach:**
```python
@wp.kernel
def detect_collisions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    collision_radius: wp.float32,
    collision_pairs: wp.array(dtype=wp.int32, ndim=2),
    n: wp.int32
):
    """Detect particle collisions."""
    i = wp.tid()
    if i >= n:
        return

    for j in range(i+1, n):
        dist = wp.length(positions[j] - positions[i])
        if dist < collision_radius:
            # Record collision pair
            # (handle merge in Python, complex logic)
            collision_pairs[...] = ...

# In Python:
def handle_collisions(positions, velocities, masses, pairs):
    """Merge colliding particles."""
    for i, j in pairs:
        # Conservation of momentum
        m_total = masses[i] + masses[j]
        v_new = (masses[i]*velocities[i] + masses[j]*velocities[j]) / m_total

        # New merged particle
        masses[i] = m_total
        velocities[i] = v_new
        positions[i] = (positions[i] + positions[j]) / 2

        # Mark j for removal
        masses[j] = 0  # Dead particle
```

---

## Phase 5: Resolution & Quality (2-4 hours)

### 5.1 4K Rendering
**Goal:** 3840×2160 output
**Impact:** Maximum quality, future-proof
**Effort:** Test + parameter tuning
**Challenge:** 4× pixels = 4× render time (83s → 330s)

```bash
python render_organic.py -r 4k -q best -w 64 --bodies 100
```

**Optimization:**
- Use 64 workers (if system has enough RAM)
- Reduce trail length (2000 → 1000 points)
- Simplify glow layers (6 → 4 layers)

### 5.2 HDR Output
**Goal:** High Dynamic Range video (Rec. 2020, 10-bit)
**Impact:** Wider color gamut, better highlights
**Effort:** 2 hours

**Implementation:**
```python
# In OrganicRenderer.render_frame()
# Use float32 linear color (not uint8 sRGB)
rgb_linear = np.zeros((height, width, 3), dtype=np.float32)

# ... render with linear colors ...

# Apply tonemapping
def tonemap_aces(linear_rgb):
    """ACES filmic tonemapping."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((linear_rgb*(a*linear_rgb+b))/(linear_rgb*(c*linear_rgb+d)+e), 0, 1)

rgb_tonemapped = tonemap_aces(rgb_linear)

# Convert to Rec. 2020 10-bit
rgb_10bit = (rgb_tonemapped * 1023).astype(np.uint16)
```

---

## Phase 6: OptiX GPU Renderer (1-2 weeks)

**See:** `docs/13-optix-cpp-hybrid-plan.md`

**Goal:** Replace matplotlib with real-time GPU renderer
**Impact:** 100x speedup (83s → 0.8s), volumetric quality
**Effort:** 1-2 weeks (C++ code + Python bindings)

**Approach:**
1. C++ OptiX renderer with volumetric ray marching
2. Python ctypes interface for physics data transfer
3. Direct GPU→GPU transfer (no CPU roundtrip)
4. Real-time parameter tweaking (interactive mode)

**Timeline:**
- Day 1-2: Basic OptiX setup + particle rendering
- Day 3-4: Volumetric trails + glow
- Day 5-6: Color system + audio integration
- Day 7-8: Optimization + Python bindings
- Day 9-10: Testing + debugging

**Result:**
- Real-time 1080p @ 60 fps
- 4K @ 30 fps
- 8K @ 15 fps (on 8× H100)

---

## Priority Matrix

| Enhancement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| **GPU Boundary Control** | ★★★★★ | 4h | **HIGHEST** |
| Motion Blur | ★★★★☆ | 0.5h | **HIGH** |
| Increase Particles (100) | ★★★★☆ | 0.1h | **HIGH** |
| Chord Detection Colors | ★★★★☆ | 2h | **HIGH** |
| Bloom Post-Process | ★★★☆☆ | 1h | Medium |
| Gravitational Lensing | ★★★★☆ | 3h | Medium |
| Tempo-Synced Pulses | ★★★☆☆ | 2h | Medium |
| 4K Rendering | ★★★☆☆ | Test | Medium |
| Variable Gravity | ★★☆☆☆ | 2h | Low |
| Particle Collisions | ★★★☆☆ | 6h | Low |
| OptiX Renderer | ★★★★★ | 1-2w | **Future** |

---

## Recommended Next Session Plan

**Session Goal:** Ship an enhanced version with 3-5 improvements

**Timeline: 4-6 hours**

1. **GPU Boundary Control** (4h) - Biggest win
2. **Increase to 100 particles** (5min)
3. **Motion blur** (30min)
4. **Chord-based palette shifts** (2h)
5. **Test with new audio track** (10min)

**Expected results:**
- 5-10x faster physics
- More visually complex (100 particles)
- Smoother motion (motion blur)
- Better music sync (chord colors)
- Proven versatility (different audio)

**Output:** New stunning render ready to share!

---

## Long-Term Vision

**3-6 Months:**
- OptiX GPU renderer → real-time performance
- VR support → immersive experience
- Interactive mode → user can perturb particles
- Multiple camera angles → cinematic camera work
- Complete "Act I" of 3-act narrative journey

**Dream:**
Live VJ performance tool - DJ plays music, particles react in real-time on big screen behind stage. Crowd goes wild. 🎉

---

*End of Enhancement Roadmap*
