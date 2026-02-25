"""Quick tests for Camera Modes"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from rendering.camera_modes import OrbitMode, ZoomMode, DollyMode, ChaseMode, SmoothTrackingMode

# Test data
particles_pos = np.random.randn(10, 3) * 10
particles_vel = np.random.randn(10, 3) * 2

print("Testing Camera Modes...")

# Test all modes
modes = [
    OrbitMode(),
    ZoomMode(),
    DollyMode(),
    ChaseMode(),
    SmoothTrackingMode()
]

for mode in modes:
    pos = mode.update(particles_pos, particles_vel, dt=1.0/30.0)
    assert pos.shape == (3,), f"{mode.__class__.__name__} failed"
    assert np.all(np.isfinite(pos)), f"{mode.__class__.__name__} returned invalid values"
    print(f"✅ {mode.__class__.__name__} works")

print("\n✅ All camera modes passed!")
