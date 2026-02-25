# Project Vision: Celestial Chaos

> "Math creates beauty" - Music-driven gravitational physics as living visual art

## Creative Vision

### The Journey
An epic narrative from simplicity to cosmic chaos to quantum strangeness:

1. **Act I: Elegance** - Single 3-body system, silk-ribbon trails, perfect orbital mechanics
2. **Act II: Emergence** - N-body chaos, gravitational clustering, galaxy formation
3. **Act III: Quantum** - Aharonov-Bohm interference, wave functions visible, potentials made real

### Aesthetic: Organic Bioluminescence
- Flowing particles like deep-sea creatures
- Nature-inspired living light
- Ethereal glows and soft color bleeding
- Trails that breathe with the music

### Audio Philosophy
Music doesn't accompany the visuals - **music IS the physics**.
- Bass frequencies modulate gravitational constant
- Beats create orbital perturbations
- Chord changes trigger color palette transitions
- Drops inject chaos into stable systems

---

## Technical Decisions

### GPU Target: GH200 Grace Hopper (Primary)

| Resource | Value | Use |
|----------|-------|-----|
| GPU Memory | 96GB HBM3 | Active rendering, OptiX BVH |
| CPU Memory | 480GB LPDDR5X | Particle history, audio buffers |
| Unified BW | 900 GB/s | Zero-copy CPU↔GPU |
| Cost | $1.49/hr | Development + most renders |

Burst to **8x H100** ($23.92/hr) for:
- 8K final renders
- 100M+ particle scenes
- Maximum quality exports

### Stack: All-NVIDIA

```
┌─────────────────────────────────────────┐
│          NVIDIA ECOSYSTEM               │
├─────────────────────────────────────────┤
│  Physics:    NVIDIA Warp                │
│  Rendering:  NVIDIA OptiX               │
│  Memory:     GH200 Unified              │
│  Compute:    CUDA 12.x                  │
│  Video:      NVENC (hardware encode)    │
└─────────────────────────────────────────┘
```

### Why Warp + OptiX?

**NVIDIA Warp:**
- Built BY NVIDIA FOR their GPUs
- Native GH200 unified memory support
- Python frontend, CUDA backend
- Differentiable (can train physics!)
- 2022+ but rapidly maturing

**OptiX:**
- Hardware-accelerated ray tracing
- Perfect for volumetric trails
- BVH acceleration for millions of particles
- HDR, motion blur, depth of field built-in

---

## Output Formats

### Primary: Video Renders
- Resolution: 4K (3840×2160) standard, 8K for hero pieces
- Frame rate: 60fps (buttery smooth trails)
- Duration: 2-10 minutes per composition
- Audio: Original WAV embedded, synced

### Secondary: Real-time Capable
Architecture supports future:
- Live audio input (microphone/line-in)
- VJ parameter control
- Gallery installations
- Interactive web export (via WebGPU port)

---

## Audio-Visual Mapping

### Frequency Bands → Physics

| Band | Hz | Parameter | Visual Effect |
|------|-----|-----------|---------------|
| Sub-bass | 20-60 | Gravitational constant G | Bodies pull harder on drops |
| Bass | 60-250 | Orbital perturbation | Kicks nudge trajectories |
| Low-mid | 250-500 | Trail decay rate | Sustain = longer trails |
| Mid | 500-2k | Particle spawn rate | Melody = particle density |
| High-mid | 2k-6k | Color saturation | Brightness follows energy |
| High | 6k-20k | Glow intensity | Shimmer on cymbals/hats |

### Musical Features → Events

| Feature | Detection | Visual |
|---------|-----------|--------|
| Beat onset | librosa.onset | Gravity pulse (brief G spike) |
| Bar/measure | Beat tracking | Phase transition moment |
| Chord change | Chroma analysis | Color palette morph |
| Drop/build | RMS energy | Chaos injection / calm |
| Tempo | BPM detection | Base simulation speed |

---

## Bioluminescence Palette

### Core Colors (Organic)

```python
PALETTE = {
    "deep_void": "#0a0a12",      # Background - near black
    "biolume_cyan": "#00ffcc",   # Primary trail color
    "biolume_magenta": "#ff00aa", # Secondary trail
    "biolume_gold": "#ffcc00",   # Accent / energy
    "soft_white": "#e0f0ff",     # Highlights
    "deep_purple": "#2a0a3a",    # Shadow/depth
}
```

### Trail Rendering

```python
# Bioluminescent glow formula
glow_intensity = base_brightness * (1.0 / distance²) * audio_energy
trail_alpha = exp(-age / decay_rate) * velocity_magnitude
color = lerp(cold_color, hot_color, velocity_normalized)
```

### Particle Appearance

- Soft emissive spheres (no hard edges)
- Fresnel glow at edges
- Size varies with mass
- Color varies with velocity
- Alpha varies with age

---

## Project Milestones

### Phase 1: Foundation (First Priority)
- [ ] Warp N-body physics kernel
- [ ] Basic particle trail system
- [ ] GH200 memory optimization
- [ ] Single 3-body demo render

### Phase 2: Audio Integration
- [ ] WAV loading + FFT analysis
- [ ] Frequency → parameter mapping
- [ ] Beat detection → gravity pulses
- [ ] Audio-synced video export

### Phase 3: Visual Polish
- [ ] OptiX volumetric trails
- [ ] Bioluminescent shading
- [ ] HDR + bloom post-processing
- [ ] Motion blur

### Phase 4: Scale Up
- [ ] N-body (1000+ particles)
- [ ] 8x H100 distributed rendering
- [ ] 8K output pipeline
- [ ] Multiple composition renders

### Phase 5: Narrative Journey
- [ ] Act I: Elegance sequence
- [ ] Act II: Emergence sequence
- [ ] Act III: Quantum sequence
- [ ] Full combined piece

---

## File Naming Convention

```
celestial_chaos_[act]_[scene]_[version].[ext]

Examples:
celestial_chaos_act1_opening_v003.mp4
celestial_chaos_act2_galaxy_v001.mp4
celestial_chaos_full_journey_final.mp4
```

---

## Development Rhythm

As an **ongoing art practice**:

1. **Weekly**: Small experiments, parameter exploration
2. **Monthly**: Complete scene/sequence
3. **Quarterly**: Major feature addition
4. **Yearly**: Full narrative piece

No deadline pressure. The tool evolves. The art evolves.

---

*"Sometimes it's good not to know too much." - Yakir Aharonov*

*Applied here: Let the physics surprise you. Let the music lead. Don't over-plan the beauty.*
