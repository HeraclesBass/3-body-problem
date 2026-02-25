"""Quick test for Moment Detector"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'analysis'))

import numpy as np
from moment_detector import MomentDetector
from audio.harmonic_analyzer import HarmonicFrame

print("Testing Moment Detector...")

detector = MomentDetector()

# Record some frames with more variation
for i in range(200):  # More frames
    time = i / 30.0
    frame = HarmonicFrame(
        time=time,
        dominant_notes=[],
        harmonics=[],
        chord=None,
        chord_confidence=0.0,
        beat_strength=0.3 + 0.7 * np.sin(i * 0.1),  # More variation
        beat_attack=(i % 10 == 0),
        spectral_flux=0.2 + 0.8 * np.sin(i * 0.2),
        harmonicity=0.5 + 0.5 * np.sin(i * 0.15)
    )
    visual_complexity = 0.3 + 0.7 * np.sin(i * 0.3)
    detector.record_frame(time, frame, particle_count=50, visual_complexity=visual_complexity)

# Detect moments
moments = detector.detect_moments()
print(f"✅ Detected {len(moments)} moments")

# Get clips
clips = detector.get_clip_markers(duration=5.0, count=3)
print(f"✅ Generated {len(clips)} clip markers")

stats = detector.get_stats()
print(f"✅ Stats: {stats.get('total_moments', 0)} total moments")

print("\n✅ All tests passed!")
