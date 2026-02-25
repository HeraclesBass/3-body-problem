# 🌌 Celestial Chaos: 3-Body Problem V8

## Audio-Reactive Gravitational Visualization

A stunning real-time visualization of a 10-body gravitational system within a realistic 360° cosmos, where **each star responds to its own 1/10th of the music frequency spectrum**.

![Status](https://img.shields.io/badge/status-production-green)
![Renderer](https://img.shields.io/badge/renderer-Stellar%20Frequency%20Visualizer-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)

---

## ✨ What You Get

- **360° Immersive Cosmos**: Deep space with 100,000+ realistic stars, distant nebulae, dust extinction, Milky Way glow
- **10 Frequency-Reactive Stars**: Each responds to a different frequency band (sub-bass → treble)
- **Revolutionary Trail System**: Trails ONLY appear when their frequency band is active (visual music spectrum)
- **HDR Lighting**: Bloom effects, multi-layer glow, beat-triggered flashes
- **Real Physics**: N-body gravity simulation with tight orbits
- **Full Song Rendering**: Complete music videos with synchronized visuals

---

## 🚀 Quick Start

### Basic Render (30 seconds)
```bash
python render_cinematic.py -r 1080p -q good --duration 30
```

### Full Song Render
```bash
python render_cinematic.py -r 1080p -q good
```

### Custom Settings
```bash
python render_cinematic.py \
  --bodies 50 \
  --resolution 1080p \
  --quality good \
  --duration 60 \
  --audio assets/audio/still-night.mp3
```

---

## 📊 Frequency Band Mapping

Each of the 10 stars represents a frequency band:

| # | Band | Frequency | Star Color | Trail Appearance |
|---|------|-----------|-----------|-----------------|
| 0 | Sub-bass | 20-60 Hz | 🔴 Deep Red | When bass is strong |
| 1 | Bass | 60-250 Hz | 🟠 Orange-Red | When kicks/drums hit |
| 2 | Low-mid | 250-500 Hz | 🟡 Orange-Yellow | Warmth/body |
| 3 | Mid | 500-2k Hz | 🟡 Yellow | Presence/vocals |
| 4 | High-mid | 2-4k Hz | 🟢 Yellow-Green | Detail/clarity |
| 5 | Presence | 4-6k Hz | 🔵 Green-Cyan | Crispness |
| 6 | Brilliance | 6-10k Hz | 🔵 Cyan | Sizzle/air |
| 7 | Air | 10-14k Hz | 🔵 Blue | Brightness |
| 8 | Ultra | 14-18k Hz | 💜 Purple | Ultrasonics |
| 9 | Extreme | 18-20k Hz | 💜 Magenta | Extreme treble |

---

## 🎨 Features

### Star Reactivity
- **Size**: Scales 100-150% with energy + beat
- **Glow**: HDR bloom (0.8-5.0 intensity)
- **Brightness**: 50-100% responsive to frequency
- **Burst Rings**: Expanding rings on strong beats

### Frequency-Band Trails
- **Visibility**: Only appear when band energy > 50%
- **Width**: 1-5 pixels (scales with band energy)
- **Opacity**: 20-100% (newer segments brighter)
- **Colors**: 8-color rainbow cycle per segment
- **Effect**: Visual music spectrum analyzer

### Cosmic Environment
- **Distant Nebulae**: 8 far structures (red, orange, blue, purple, yellow)
- **Deep Star Field**: 100,000+ realistic stars (Milky Way distribution)
- **Dust Extinction**: 15 clouds with color-shifted reddening
- **Galactic Core**: Golden glow pulsing with beats
- **360° Projection**: Perspective-based spherical rendering

---

## 🎬 Quality Levels

| Level | Resolution | Glow | Bg Stars | Parallax | Encode | Time/min |
|-------|-----------|------|----------|----------|--------|----------|
| Draft | 720p | 6 | 300 | 2 | Fast | ~1 min |
| Good | 1080p | 12 | 600 | 4 | Normal | ~2-3 min |
| Best | 1440p | 20 | 1200 | 6 | Slow | ~3-4 min |

---

## 📁 Project Structure

```
3-body-problem-v8/
├── render_cinematic.py       # ⭐ Main renderer (Stellar Frequency Visualizer)
├── serve_video.py            # HTTP video server
├── bootstrap.sh              # GPU environment setup
├── deploy-to-gpu.sh          # Deploy to Lambda Labs GH200
│
├── src/
│   ├── physics/
│   │   ├── kernels.py        # Warp GPU kernels
│   │   ├── nbody.py          # N-body simulation
│   │   └── boundary_control.py
│   └── audio/
│       ├── analyzer.py       # Basic audio analysis
│       └── analyzer_10band.py # 10-band frequency extraction
│
├── assets/
│   └── audio/
│       └── still-night.mp3   # Default music file (331.4s)
│
├── output/                   # Rendered videos (auto-created)
├── tests/                    # Test suite
│
├── .archive/
│   ├── legacy_renderers/     # Old render scripts (v1-v7)
│   ├── legacy_tests/         # Old test files
│   └── legacy_docs/          # Historical documentation
│
├── CLAUDE.md                 # Project guidelines
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── requirements-lock.txt     # Pinned versions
```

---

## 🔧 Setup & Installation

### On CPU (Local)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### On GPU (Lambda Labs GH200)
```bash
./bootstrap.sh
```

---

## 📚 Documentation

- **CLAUDE.md**: Project guidelines, quick start, deployment
- **render_cinematic.py**: Well-commented main renderer
- **.archive/**: Historical documentation (v1-v7)

---

## 🎵 Audio Input

Supported: MP3, WAV, FLAC, OGG

**Default**: `assets/audio/still-night.mp3` (331.4 seconds)

```bash
python render_cinematic.py --audio /path/to/song.mp3
```

---

## ⚡ Performance

**CPU Rendering:**
- Speed: ~11 fps
- Time: ~14-15 min per 5.5 min song (good quality)

**GPU Rendering (NVIDIA CUDA):**
- Speed: 3-5× faster
- Time: ~3-5 min per 5.5 min song (good quality)

---

## 🎯 How It Works

1. **Audio Analysis**: Loads music, extracts 10-band energy + beat detection
2. **Physics Simulation**: N-body gravity (NVIDIA Warp), audio-modulated forces
3. **Rendering**: 360° cosmos + stellar system + frequency trails
4. **Output**: MP4 video with synchronized audio

---

## 🐛 Troubleshooting

### "No module named 'warp'"
Normal on CPU. Rendering works, just slower.

### Render is slow
- Use `-q draft` or `-r 720p`
- Use `--bodies 10`

### Memory error
Reduce resolution or quality

---

## ✅ Status

**Working:**
- ✅ 360° spherical cosmos environment
- ✅ 10 frequency-band reactive stars
- ✅ Frequency-specific trail visualization
- ✅ HDR lighting & bloom
- ✅ Beat detection & response
- ✅ Full song rendering
- ✅ CPU & GPU support

---

**Last Updated**: February 4, 2026 | **Status**: Production Ready ✅
