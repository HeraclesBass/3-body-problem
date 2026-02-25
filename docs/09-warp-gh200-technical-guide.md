# NVIDIA Warp + GH200 Technical Guide

> Definitive reference for Celestial Chaos development on GH200 Grace Hopper

## Executive Summary

| Component | Specification |
|-----------|---------------|
| **Framework** | NVIDIA Warp 1.11.0 |
| **GPU** | GH200 Grace Hopper (Hopper GH100 + Grace ARM) |
| **GPU Memory** | 96GB HBM3 @ 4,000 GB/s |
| **CPU Memory** | 480GB LPDDR5X @ 512 GB/s |
| **Unified Memory** | 576GB total, hardware coherent |
| **CPU-GPU Link** | NVLink-C2C @ 900 GB/s |
| **Cost** | $1.49/hr (Lambda Labs) |

---

## 1. NVIDIA Warp Overview

### What Is Warp?

NVIDIA Warp is a Python framework for writing high-performance simulation and graphics code. Key characteristics:

- **JIT Compilation**: Python → CUDA kernels at runtime
- **Spatial Computing**: First-class vec3, mat44, quat, transform types
- **Automatic Differentiation**: Forward and reverse mode for ML integration
- **PyTorch/JAX Interop**: Zero-copy tensor sharing

### Why Warp for This Project?

| Requirement | Warp Capability |
|-------------|-----------------|
| N-body physics | Built-in spatial types, hash grids |
| GPU acceleration | Native CUDA compilation |
| GH200 support | Hopper architecture (sm_90) supported |
| Audio reactivity | Python ecosystem (librosa) + kernel speed |
| Differentiable | Can optimize physics parameters with ML |

---

## 2. GH200 Architecture Deep Dive

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GH200 MEMORY MODEL                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    TIER 1: GPU HBM3                          │   │
│   │         96GB @ 4,000 GB/s bandwidth                         │   │
│   │                                                             │   │
│   │   Use for:                                                  │   │
│   │   • Active particle positions/velocities                    │   │
│   │   • Current frame rendering data                            │   │
│   │   • Kernel working memory                                   │   │
│   │   • Atomic operations (reductions)                          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                    NVLink-C2C @ 900 GB/s                            │
│                    (7x faster than PCIe)                            │
│                              │                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    TIER 2: CPU LPDDR5X                       │   │
│   │        480GB @ 512 GB/s bandwidth                           │   │
│   │                                                             │   │
│   │   Use for:                                                  │   │
│   │   • Particle trail history (ring buffers)                   │   │
│   │   • Audio analysis buffers                                  │   │
│   │   • Frame output queues                                     │   │
│   │   • Large lookup tables                                     │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key GH200 Advantages for This Project

| Feature | Benefit |
|---------|---------|
| **Unified Memory** | No explicit CPU↔GPU copies for trail data |
| **576GB Total** | Store 10M+ particle trajectories in memory |
| **900 GB/s NVLink** | Real-time audio buffer access from GPU |
| **72 ARM Cores** | Parallel FFT analysis while GPU renders |
| **$1.49/hr** | 55% cheaper than standalone H100 |

### Memory Budget Calculation

```python
# Particle state (per particle)
PARTICLE_STATE = {
    'position': 12,      # vec3 (3 × float32)
    'velocity': 12,      # vec3
    'acceleration': 12,  # vec3
    'mass': 4,           # float32
    'color': 16,         # vec4 (RGBA)
    'age': 4,            # float32
    'trail_idx': 4,      # int32 (ring buffer index)
}
BYTES_PER_PARTICLE = sum(PARTICLE_STATE.values())  # 64 bytes

# Trail history (per particle)
TRAIL_POINTS = 1000
TRAIL_POINT_SIZE = 16  # vec3 + float32 (position + intensity)
BYTES_PER_TRAIL = TRAIL_POINTS * TRAIL_POINT_SIZE  # 16KB

# GPU Memory (96GB) - Active simulation
GPU_AVAILABLE = 80 * 1024**3  # 80GB (leave 16GB for overhead)
MAX_ACTIVE_PARTICLES = GPU_AVAILABLE // BYTES_PER_PARTICLE
# = 1.25 billion particles (active state only)

# For 50M particles with full trails:
TOTAL_MEMORY = 50_000_000 * (BYTES_PER_PARTICLE + BYTES_PER_TRAIL)
# = 50M × (64 + 16,000) = 803 GB
# Fits in unified memory with tiered storage!
```

---

## 3. Warp Kernel Development

### Basic Kernel Structure

```python
import warp as wp
import numpy as np

# Initialize Warp (call once at startup)
wp.init()

# Select device
device = wp.get_device("cuda:0")  # GH200 GPU

# Define a kernel (JIT compiled to CUDA)
@wp.kernel
def compute_accelerations(
    positions: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    accelerations: wp.array(dtype=wp.vec3),
    G: wp.float32,
    softening: wp.float32,
    n: wp.int32
):
    """N-body gravitational acceleration kernel"""
    i = wp.tid()  # Thread index

    if i >= n:
        return

    pos_i = positions[i]
    acc = wp.vec3(0.0, 0.0, 0.0)

    for j in range(n):
        if i != j:
            pos_j = positions[j]
            r = pos_j - pos_i
            dist_sq = wp.dot(r, r) + softening * softening
            dist = wp.sqrt(dist_sq)
            acc += G * masses[j] * r / (dist_sq * dist)

    accelerations[i] = acc
```

### Launching Kernels

```python
# Create arrays on device
n = 10000  # Number of particles
positions = wp.zeros(n, dtype=wp.vec3, device=device)
velocities = wp.zeros(n, dtype=wp.vec3, device=device)
accelerations = wp.zeros(n, dtype=wp.vec3, device=device)
masses = wp.ones(n, dtype=wp.float32, device=device)

# Initialize positions (example: random sphere)
positions_np = np.random.randn(n, 3).astype(np.float32) * 100.0
wp.copy(positions, wp.array(positions_np, dtype=wp.vec3, device=device))

# Launch kernel
wp.launch(
    kernel=compute_accelerations,
    dim=n,
    inputs=[positions, masses, accelerations, 1.0, 0.1, n],
    device=device
)

# Synchronize (wait for GPU)
wp.synchronize_device(device)
```

### Symplectic Integrator (Velocity Verlet)

```python
@wp.kernel
def integrate_verlet_step1(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    n: wp.int32
):
    """First half of Verlet: update velocities and positions"""
    i = wp.tid()
    if i >= n:
        return

    # v(t + dt/2) = v(t) + a(t) * dt/2
    vel = velocities[i] + accelerations[i] * (dt * 0.5)

    # x(t + dt) = x(t) + v(t + dt/2) * dt
    pos = positions[i] + vel * dt

    velocities[i] = vel
    positions[i] = pos


@wp.kernel
def integrate_verlet_step2(
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    n: wp.int32
):
    """Second half of Verlet: complete velocity update"""
    i = wp.tid()
    if i >= n:
        return

    # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
    velocities[i] = velocities[i] + accelerations[i] * (dt * 0.5)
```

### Full Simulation Step

```python
def simulation_step(positions, velocities, accelerations, masses, dt, G, softening, n, device):
    """Complete symplectic integration step"""

    # Step 1: Half velocity update + full position update
    wp.launch(
        integrate_verlet_step1,
        dim=n,
        inputs=[positions, velocities, accelerations, dt, n],
        device=device
    )

    # Step 2: Recompute accelerations at new positions
    wp.launch(
        compute_accelerations,
        dim=n,
        inputs=[positions, masses, accelerations, G, softening, n],
        device=device
    )

    # Step 3: Complete velocity update
    wp.launch(
        integrate_verlet_step2,
        dim=n,
        inputs=[velocities, accelerations, dt, n],
        device=device
    )
```

---

## 4. Audio-Reactive Parameters

### FFT Integration Pattern

```python
import librosa
import numpy as np

class AudioAnalyzer:
    """Real-time audio feature extraction"""

    def __init__(self, wav_path: str, fps: int = 60):
        # Load audio
        self.y, self.sr = librosa.load(wav_path, sr=44100)
        self.fps = fps
        self.hop_length = self.sr // fps  # Samples per frame

        # Pre-compute features for all frames
        self.precompute_features()

    def precompute_features(self):
        """Pre-compute all audio features for the entire track"""

        # Mel spectrogram (for frequency bands)
        self.mel = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr,
            hop_length=self.hop_length,
            n_mels=32  # 32 frequency bands
        )
        self.mel_db = librosa.power_to_db(self.mel, ref=np.max)

        # Beat tracking
        self.tempo, self.beats = librosa.beat.beat_track(
            y=self.y, sr=self.sr,
            hop_length=self.hop_length
        )

        # Onset detection (transients)
        self.onsets = librosa.onset.onset_detect(
            y=self.y, sr=self.sr,
            hop_length=self.hop_length,
            units='frames'
        )

        # RMS energy
        self.rms = librosa.feature.rms(
            y=self.y, hop_length=self.hop_length
        )[0]

        # Chroma (for key/chord detection)
        self.chroma = librosa.feature.chroma_stft(
            y=self.y, sr=self.sr,
            hop_length=self.hop_length
        )

        self.n_frames = self.mel.shape[1]

    def get_frame_features(self, frame_idx: int) -> dict:
        """Get audio features for a specific frame"""
        if frame_idx >= self.n_frames:
            frame_idx = self.n_frames - 1

        # Frequency bands (normalized 0-1)
        mel_frame = self.mel_db[:, frame_idx]
        mel_norm = (mel_frame - mel_frame.min()) / (mel_frame.max() - mel_frame.min() + 1e-8)

        return {
            'sub_bass': float(np.mean(mel_norm[0:2])),     # 20-60 Hz
            'bass': float(np.mean(mel_norm[2:6])),         # 60-250 Hz
            'low_mid': float(np.mean(mel_norm[6:10])),     # 250-500 Hz
            'mid': float(np.mean(mel_norm[10:18])),        # 500-2kHz
            'high_mid': float(np.mean(mel_norm[18:26])),   # 2k-6kHz
            'high': float(np.mean(mel_norm[26:32])),       # 6k-20kHz
            'energy': float(self.rms[frame_idx]),
            'is_beat': frame_idx in self.beats,
            'is_onset': frame_idx in self.onsets,
            'chroma': self.chroma[:, frame_idx].tolist(),
        }
```

### Audio-to-Physics Mapping

```python
class PhysicsModulator:
    """Map audio features to physics parameters"""

    def __init__(self):
        # Base physics parameters
        self.G_base = 1.0
        self.spawn_rate_base = 100
        self.trail_decay_base = 0.995

        # Modulation depths (how much audio affects each parameter)
        self.modulation = {
            'G': 0.5,              # Gravity: bass boosts G by up to 50%
            'perturbation': 0.3,   # Beat → orbital kick
            'spawn': 2.0,          # Mid energy → 2x spawn rate
            'trail_decay': 0.1,    # Low-mid sustain → longer trails
            'saturation': 0.4,     # High-mid → color saturation
            'glow': 0.6,           # High → particle brightness
        }

    def get_physics_params(self, audio_features: dict) -> dict:
        """Convert audio features to physics parameters"""

        # Gravitational constant (bass-reactive)
        G = self.G_base * (1.0 + audio_features['sub_bass'] * self.modulation['G'])

        # Beat-triggered perturbation
        perturbation = 0.0
        if audio_features['is_beat']:
            perturbation = audio_features['bass'] * self.modulation['perturbation']

        # Particle spawn rate (mid-reactive)
        spawn_rate = int(self.spawn_rate_base * (1.0 + audio_features['mid'] * self.modulation['spawn']))

        # Trail persistence (sustain-reactive)
        trail_decay = self.trail_decay_base + audio_features['low_mid'] * self.modulation['trail_decay']
        trail_decay = min(trail_decay, 0.9999)  # Clamp

        # Visual parameters
        saturation = 0.6 + audio_features['high_mid'] * self.modulation['saturation']
        glow = 0.5 + audio_features['high'] * self.modulation['glow']

        return {
            'G': G,
            'perturbation': perturbation,
            'spawn_rate': spawn_rate,
            'trail_decay': trail_decay,
            'saturation': saturation,
            'glow': glow,
            'chroma': audio_features['chroma'],  # For color palette
        }
```

---

## 5. Trail Rendering System

### Ring Buffer for Trail History

```python
@wp.kernel
def update_trail_buffer(
    positions: wp.array(dtype=wp.vec3),
    trail_buffer: wp.array2d(dtype=wp.vec3),  # [n_particles, trail_length]
    trail_indices: wp.array(dtype=wp.int32),
    trail_length: wp.int32,
    n: wp.int32
):
    """Store current position in circular trail buffer"""
    i = wp.tid()
    if i >= n:
        return

    # Get current write index
    idx = trail_indices[i]

    # Store position
    trail_buffer[i, idx] = positions[i]

    # Advance index (wrap around)
    trail_indices[i] = (idx + 1) % trail_length


@wp.kernel
def compute_trail_vertices(
    trail_buffer: wp.array2d(dtype=wp.vec3),
    trail_indices: wp.array(dtype=wp.int32),
    velocities: wp.array(dtype=wp.vec3),
    trail_vertices: wp.array(dtype=wp.vec3),
    trail_colors: wp.array(dtype=wp.vec4),
    trail_length: wp.int32,
    decay_rate: wp.float32,
    n_particles: wp.int32
):
    """Generate trail vertex data for rendering"""
    tid = wp.tid()
    particle_idx = tid // trail_length
    point_idx = tid % trail_length

    if particle_idx >= n_particles:
        return

    # Calculate actual buffer index (unwrap ring buffer)
    current_idx = trail_indices[particle_idx]
    buffer_idx = (current_idx - point_idx - 1 + trail_length) % trail_length

    # Get position
    pos = trail_buffer[particle_idx, buffer_idx]

    # Calculate age (0 = newest, 1 = oldest)
    age = wp.float32(point_idx) / wp.float32(trail_length)

    # Alpha decay based on age
    alpha = wp.pow(decay_rate, wp.float32(point_idx))

    # Velocity-based color (example: speed → hue)
    vel = velocities[particle_idx]
    speed = wp.length(vel)

    # Simple color mapping (cyan to magenta based on speed)
    hue = wp.clamp(speed / 10.0, 0.0, 1.0)
    r = hue
    g = 1.0 - hue * 0.5
    b = 1.0

    # Store output
    output_idx = particle_idx * trail_length + point_idx
    trail_vertices[output_idx] = pos
    trail_colors[output_idx] = wp.vec4(r, g, b, alpha)
```

---

## 6. GH200 Optimization Patterns

### Unified Memory Best Practices

```python
# GH200 Unified Memory: Arrays can be accessed from both CPU and GPU
# No explicit copies needed, but access patterns matter

# GOOD: Sequential GPU access (maximize bandwidth)
@wp.kernel
def good_pattern(data: wp.array(dtype=wp.float32)):
    i = wp.tid()
    data[i] = data[i] * 2.0  # Coalesced access

# BAD: Random access (kills bandwidth)
@wp.kernel
def bad_pattern(data: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.int32)):
    i = wp.tid()
    j = indices[i]
    data[i] = data[j]  # Random access → poor performance
```

### Memory Tiering Strategy

```python
class GH200MemoryManager:
    """Optimize memory placement for GH200 unified memory"""

    def __init__(self, n_particles: int, trail_length: int):
        self.n = n_particles
        self.trail_len = trail_length
        self.device = wp.get_device("cuda:0")

        # TIER 1 (GPU HBM3): Hot data, frequently accessed
        # These stay GPU-resident for maximum bandwidth
        self.positions = wp.zeros(n_particles, dtype=wp.vec3, device=self.device)
        self.velocities = wp.zeros(n_particles, dtype=wp.vec3, device=self.device)
        self.accelerations = wp.zeros(n_particles, dtype=wp.vec3, device=self.device)

        # TIER 2 (Unified): Trail history, large but sequential access
        # GH200 handles this efficiently via NVLink-C2C
        self.trail_buffer = wp.zeros(
            (n_particles, trail_length),
            dtype=wp.vec3,
            device=self.device
        )
        self.trail_indices = wp.zeros(n_particles, dtype=wp.int32, device=self.device)

        # TIER 2 (CPU): Audio buffers, read by CPU, accessed by GPU
        # Pre-loaded, streamed to GPU as needed
        self.audio_features = None  # Loaded from AudioAnalyzer

    def prefetch_audio_frame(self, frame_idx: int):
        """Prefetch audio features for upcoming frame"""
        # GH200's unified memory means this is a hint, not a copy
        # The memory system handles migration automatically
        pass
```

### Kernel Fusion for Performance

```python
# BETTER: Fuse multiple operations into single kernel
@wp.kernel
def fused_simulation_step(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    trail_buffer: wp.array2d(dtype=wp.vec3),
    trail_indices: wp.array(dtype=wp.int32),
    G: wp.float32,
    dt: wp.float32,
    softening: wp.float32,
    trail_length: wp.int32,
    n: wp.int32
):
    """Fused kernel: acceleration + integration + trail update"""
    i = wp.tid()
    if i >= n:
        return

    # 1. Compute acceleration
    pos_i = positions[i]
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(n):
        if i != j:
            r = positions[j] - pos_i
            dist_sq = wp.dot(r, r) + softening * softening
            dist = wp.sqrt(dist_sq)
            acc += G * masses[j] * r / (dist_sq * dist)

    # 2. Verlet integration (fused)
    vel = velocities[i] + acc * dt
    pos = pos_i + vel * dt

    # 3. Update trail buffer
    idx = trail_indices[i]
    trail_buffer[i, idx] = pos
    trail_indices[i] = (idx + 1) % trail_length

    # 4. Write back
    positions[i] = pos
    velocities[i] = vel
```

---

## 7. Rendering Pipeline Options

### Option A: ModernGL Preview (Development)

```python
import moderngl
import numpy as np

class TrailRenderer:
    """OpenGL-based trail rendering for real-time preview"""

    def __init__(self, width: int, height: int):
        self.ctx = moderngl.create_standalone_context()

        # Trail shader
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec4 in_color;
                out vec4 v_color;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    v_color = in_color;
                    gl_PointSize = 2.0;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec4 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = v_color;
                }
            '''
        )
```

### Option B: OptiX Ray Tracing (Production)

OptiX integration requires additional setup but provides:
- Hardware-accelerated ray tracing on GH200
- Volumetric trail rendering
- Global illumination
- Motion blur

```python
# OptiX requires separate installation and CUDA interop
# This is a placeholder for the architecture

class OptiXRenderer:
    """OptiX-based production renderer"""

    def __init__(self, width: int, height: int):
        # OptiX context setup
        # BVH construction for trails
        # Ray generation programs
        # Hit/miss shaders
        pass

    def render_frame(self, trail_vertices, trail_colors, camera):
        # Build acceleration structure
        # Launch ray tracing
        # Post-process (bloom, tone mapping)
        pass
```

### Option C: Blender Export (Offline)

```python
def export_to_blender_usd(positions_history, output_path):
    """Export simulation to USD for Blender rendering"""
    # USD format supported by Warp
    # Can be rendered in Blender Cycles
    pass
```

---

## 8. Performance Benchmarks (Expected)

### GH200 N-Body Performance

| Particles | Integration | Full Step | FPS (60Hz budget) |
|-----------|-------------|-----------|-------------------|
| 1,000 | 0.01 ms | 0.05 ms | ✓ 20,000 FPS |
| 10,000 | 0.1 ms | 1.2 ms | ✓ 830 FPS |
| 100,000 | 10 ms | 15 ms | ✓ 66 FPS |
| 1,000,000 | 150 ms | 200 ms | ✗ 5 FPS |

Note: O(n²) N-body scales poorly. For >100k particles, use:
- Barnes-Hut tree (O(n log n))
- Fast Multipole Method
- Warp's HashGrid for local interactions

### Memory Bandwidth Utilization

| Operation | Bandwidth | GH200 Limit | Utilization |
|-----------|-----------|-------------|-------------|
| Position read | 50M × 12B × 60Hz = 36 GB/s | 4,000 GB/s | 0.9% |
| Trail write | 50M × 16B × 60Hz = 48 GB/s | 4,000 GB/s | 1.2% |
| Audio features | 32 × 4B × 60Hz = 7.7 KB/s | 900 GB/s | ~0% |

**Conclusion**: GH200 is massively over-provisioned for this workload. We have headroom for:
- Higher particle counts
- More complex physics
- Real-time rendering

---

## 9. Quick Start Commands

### Local Development (CPU-only)

```bash
cd /path/to/3-body-problem
source venv/bin/activate

# Test Warp
python -c "import warp as wp; wp.init(); print(wp.get_devices())"

# Run CPU simulation
python src/test_nbody_cpu.py
```

### GH200 Deployment

```bash
# 1. Start GH200 at Lambda Labs
# https://cloud.lambdalabs.com/instances

# 2. SSH to instance
ssh ubuntu@<GH200_IP>

# 3. Clone/sync project
git clone <repo> ~/celestial-chaos
# OR
rsync -avz ./3-body-problem/ ubuntu@<IP>:~/celestial-chaos/

# 4. Setup environment
cd ~/celestial-chaos
python3 -m venv venv
source venv/bin/activate
pip install warp-lang numpy librosa soundfile scipy matplotlib

# 5. Verify GPU
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
# Should show: ['cuda:0'] with GH200

# 6. Run simulation
python src/main.py --audio input.wav --output render.mp4
```

---

## 10. File Structure

```
3-body-problem/
├── venv/                          # Python environment
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── kernels.py             # Warp kernels
│   │   ├── integrators.py         # Verlet, RK4
│   │   └── nbody.py               # N-body simulation
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── analyzer.py            # FFT, beat detection
│   │   └── modulator.py           # Audio → physics mapping
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── trails.py              # Trail system
│   │   ├── preview.py             # ModernGL preview
│   │   └── optix.py               # OptiX renderer (future)
│   └── utils/
│       ├── __init__.py
│       └── config.py              # Settings, presets
├── docs/                          # Documentation
├── research/                      # Source materials
├── shaders/                       # GLSL shaders
├── assets/                        # Textures, palettes
└── renders/                       # Output directory
```

---

## References

- [NVIDIA Warp Documentation](https://nvidia.github.io/warp/)
- [NVIDIA Warp GitHub](https://github.com/NVIDIA/warp)
- [GH200 Architecture Guide](https://docs.nvidia.com/gh200-superchip-benchmark-guide.pdf)
- [Lambda Labs GH200](https://lambdalabs.com/service/gpu-cloud)
- [librosa Documentation](https://librosa.org/doc/latest/)

---

**Version**: 1.0
**Last Updated**: 2026-02-03
**Status**: Ready for GH200 deployment
