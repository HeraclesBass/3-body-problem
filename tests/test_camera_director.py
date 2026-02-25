"""Quick tests for Camera Director"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from rendering.camera_modes import OrbitMode, ZoomMode, DollyMode, ChaseMode
from rendering.camera_director import CameraDirector
from audio.harmonic_analyzer import HarmonicFrame

print("Testing Camera Director...")

# Create director
director = CameraDirector(
    camera_modes={
        'orbit': OrbitMode(),
        'zoom': ZoomMode(),
        'dolly': DollyMode(),
        'chase': ChaseMode(),
    },
    default_mode='orbit'
)

# Test data
particles_pos = np.random.randn(10, 3) * 10
particles_vel = np.random.randn(10, 3) * 2

# Test with beat attack
frame = HarmonicFrame(
    time=0.0,
    dominant_notes=[("A4", 440.0)],
    harmonics=[440.0, 880.0],
    chord=None,
    chord_confidence=0.0,
    beat_strength=0.9,
    beat_attack=True,
    spectral_flux=0.5,
    harmonicity=0.7
)

pos = director.update(frame, particles_pos, particles_vel, dt=1.0/30.0)
assert pos.shape == (3,), "Director failed"
assert np.all(np.isfinite(pos)), "Director returned invalid values"

stats = director.get_director_stats()
assert 'current_mode' in stats
assert 'beat_count' in stats

print("✅ Camera Director works")
print(f"   Mode: {stats['current_mode']}, Beats: {stats['beat_count']}")
print("\n✅ All tests passed!")
