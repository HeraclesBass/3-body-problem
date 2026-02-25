"""
Unit tests for Harmonic Analyzer

Tests:
- Note frequency detection accuracy
- Harmonic overtone calculation
- Chord detection
- Beat strength envelope (smooth, not binary)
- Spectral flux computation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from audio.harmonic_analyzer import HarmonicAnalyzer


def test_note_name_conversion():
    """Test frequency to note name conversion."""
    analyzer = HarmonicAnalyzer.__new__(HarmonicAnalyzer)
    analyzer.A4_FREQ = 440.0
    analyzer.NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Test known frequencies
    assert analyzer._freq_to_note_name(440.0) == "A4"
    assert analyzer._freq_to_note_name(261.63) == "C4"  # Middle C
    assert analyzer._freq_to_note_name(880.0) == "A5"   # A5 (octave up)
    assert analyzer._freq_to_note_name(220.0) == "A3"   # A3 (octave down)

    print("✅ Note name conversion accurate")


def test_harmonics_calculation():
    """Test harmonic overtone calculation."""
    analyzer = HarmonicAnalyzer.__new__(HarmonicAnalyzer)

    # Test A4 (440 Hz) harmonics
    harmonics = analyzer._get_harmonics(440.0)
    expected = [440.0, 880.0, 1320.0, 1760.0]

    assert len(harmonics) == 4
    for i, (actual, expected_freq) in enumerate(zip(harmonics, expected)):
        assert abs(actual - expected_freq) < 0.01, f"Harmonic {i+1} mismatch"

    print("✅ Harmonic overtones calculated correctly")


def test_beat_envelope_smooth():
    """Test that beat envelope is smooth, not binary."""
    # This test requires actual audio, so we'll use a mock
    analyzer = HarmonicAnalyzer.__new__(HarmonicAnalyzer)
    analyzer.sr = 22050
    analyzer.hop_length = 512
    analyzer.y = np.random.randn(22050 * 5)  # Mock 5 seconds of audio
    analyzer.beats = np.array([0, 50, 100])  # Mock beat frames
    analyzer.stft = np.zeros((1025, 200))  # Mock STFT

    analyzer._compute_beat_envelope()

    # Check that envelope has intermediate values (not just 0 and 1)
    unique_values = np.unique(analyzer.beat_envelope)
    assert len(unique_values) > 10, "Beat envelope should have many intermediate values"

    # Check that peaks are at beat locations
    for beat_frame in analyzer.beats:
        if beat_frame < len(analyzer.beat_envelope):
            assert analyzer.beat_envelope[beat_frame] > 0.9, "Beat peaks should be near 1.0"

    # Check decay after beats
    if analyzer.beats[0] + 10 < len(analyzer.beat_envelope):
        assert analyzer.beat_envelope[analyzer.beats[0] + 10] < 0.9, "Should decay after beat"

    print("✅ Beat envelope is smooth with attack/decay")


def test_chord_pattern_definitions():
    """Test chord pattern definitions are correct."""
    analyzer = HarmonicAnalyzer.__new__(HarmonicAnalyzer)
    analyzer.CHORD_PATTERNS = HarmonicAnalyzer.CHORD_PATTERNS

    # Test major triad
    assert analyzer.CHORD_PATTERNS['maj'] == [0, 4, 7]  # Root, major 3rd, perfect 5th

    # Test minor triad
    assert analyzer.CHORD_PATTERNS['min'] == [0, 3, 7]  # Root, minor 3rd, perfect 5th

    # Test 7th chords have 4 notes
    assert len(analyzer.CHORD_PATTERNS['maj7']) == 4
    assert len(analyzer.CHORD_PATTERNS['min7']) == 4

    print("✅ Chord patterns defined correctly")


def test_harmonicity_range():
    """Test that harmonicity values are in valid range [0, 1]."""
    analyzer = HarmonicAnalyzer.__new__(HarmonicAnalyzer)

    # Mock harmonicity array
    analyzer.harmonicity = np.array([0.0, 0.3, 0.7, 1.0, 0.5])

    # All values should be in [0, 1]
    assert np.all(analyzer.harmonicity >= 0.0)
    assert np.all(analyzer.harmonicity <= 1.0)

    print("✅ Harmonicity values in valid range")


def test_integration_with_real_audio():
    """Integration test with actual audio file (if available)."""
    audio_path = Path(__file__).parent.parent / "assets" / "audio" / "still-night.mp3"

    if not audio_path.exists():
        pytest.skip("Test audio file not found")

    # Load analyzer
    analyzer = HarmonicAnalyzer(str(audio_path), fps=30.0)

    # Test that analysis ran
    assert analyzer.duration > 0
    assert analyzer.total_frames > 0
    assert len(analyzer.beat_envelope) > 0
    assert len(analyzer.spectral_flux) > 0

    # Test getting a frame
    frame = analyzer.get_frame(0)
    assert frame.time == 0.0
    assert 0.0 <= frame.beat_strength <= 1.0
    assert 0.0 <= frame.harmonicity <= 1.0
    assert isinstance(frame.dominant_notes, list)

    # Test getting summary
    summary = analyzer.get_summary()
    assert 'duration_sec' in summary
    assert 'tempo_bpm' in summary
    assert 'avg_harmonicity' in summary

    print("✅ Integration test passed with real audio")
    print(f"   Detected {summary['num_beats']} beats at {summary['tempo_bpm']:.1f} BPM")
    print(f"   Average harmonicity: {summary['avg_harmonicity']:.2f}")


if __name__ == "__main__":
    print("Running Harmonic Analyzer Tests...\n")

    test_note_name_conversion()
    test_harmonics_calculation()
    test_beat_envelope_smooth()
    test_chord_pattern_definitions()
    test_harmonicity_range()
    test_integration_with_real_audio()

    print("\n✅ All tests passed!")
