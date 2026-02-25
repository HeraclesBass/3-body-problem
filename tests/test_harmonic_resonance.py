"""
Unit tests for Harmonic Resonance Engine

Tests:
- Resonance amplitude updates
- Size modulation from resonance
- Trail intensity modulation
- Chord pattern effects
- Phase oscillation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

# Import directly to avoid physics.__init__ which imports warp
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'physics'))
from harmonic_resonance import HarmonicResonance
sys.path.pop(0)

from audio.frequency_zones import FrequencyZones
from audio.harmonic_analyzer import HarmonicFrame


def test_resonance_initialization():
    """Test resonance engine initializes correctly."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Check arrays initialized
    assert len(resonance.particle_phases) == 50
    assert len(resonance.particle_frequencies) == 50
    assert len(resonance.resonance_amplitudes) == 50

    # Phases should be in [0, 2π]
    assert np.all(resonance.particle_phases >= 0)
    assert np.all(resonance.particle_phases <= 2 * np.pi)

    # Frequencies should be in [0.5, 2.0]
    assert np.all(resonance.particle_frequencies >= 0.5)
    assert np.all(resonance.particle_frequencies <= 2.0)

    print("✅ Resonance initialization works")


def test_amplitude_update():
    """Test amplitude updates from harmonic frame."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Create harmonic frame with high bass
    frame = HarmonicFrame(
        time=0.0,
        dominant_notes=[("A4", 440.0)],
        harmonics=[440.0, 880.0, 1320.0, 1760.0],
        chord="Cmaj",
        chord_confidence=0.8,
        beat_strength=0.5,
        beat_attack=False,
        spectral_flux=0.3,
        harmonicity=0.7
    )

    # Update zones first
    from audio.analyzer_10band import AudioFrame10Band
    audio_frame = AudioFrame10Band(
        sub_bass=0.9, bass=0.8, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.5, onset_strength=0.3, tempo=120.0, time=0.0
    )
    zones.update_from_audio(audio_frame)

    # Update resonance
    resonance.update(frame)

    # Amplitudes should be non-zero
    assert np.any(resonance.resonance_amplitudes > 0.0)

    print("✅ Amplitude updates work")


def test_size_modulation():
    """Test particle size modulation."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Set up some resonance
    resonance.resonance_amplitudes[:] = 0.5

    # Get modulated sizes
    base_size = 5.0
    for i in range(10):
        size = resonance.get_particle_size(i, base_size)

        # Size should be modulated around base_size
        assert size > 0
        # Should be within modulation range
        assert size >= base_size * 0.7
        assert size <= base_size * 1.5

    print("✅ Size modulation works")


def test_trail_intensity():
    """Test trail intensity modulation."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Test with different amplitudes
    resonance.resonance_amplitudes[0] = 0.0  # Low
    resonance.resonance_amplitudes[1] = 1.0  # High

    intensity_low = resonance.get_trail_intensity(0)
    intensity_high = resonance.get_trail_intensity(1)

    # Higher amplitude should give higher intensity
    assert intensity_high > intensity_low

    # Both should be in valid range
    assert 0.0 <= intensity_low <= 1.0
    assert 0.0 <= intensity_high <= 1.0

    print("✅ Trail intensity modulation works")


def test_beat_attack_boost():
    """Test that beat attacks boost amplitudes."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Frame without beat attack
    frame_no_beat = HarmonicFrame(
        time=0.0,
        dominant_notes=[],
        harmonics=[],
        chord=None,
        chord_confidence=0.0,
        beat_strength=0.5,
        beat_attack=False,
        spectral_flux=0.1,
        harmonicity=0.5
    )

    # Frame with beat attack
    frame_with_beat = HarmonicFrame(
        time=0.1,
        dominant_notes=[],
        harmonics=[],
        chord=None,
        chord_confidence=0.0,
        beat_strength=1.0,
        beat_attack=True,
        spectral_flux=0.8,
        harmonicity=0.5
    )

    # Update zones
    from audio.analyzer_10band import AudioFrame10Band
    audio_frame = AudioFrame10Band(
        sub_bass=0.5, bass=0.5, low_mid=0.5, mid=0.5, high_mid=0.5,
        presence=0.5, brilliance=0.5, air=0.5, ultra=0.5, extreme=0.5,
        beat_strength=0.5, onset_strength=0.5, tempo=120.0, time=0.0
    )
    zones.update_from_audio(audio_frame)

    # Update without beat
    resonance.update(frame_no_beat)
    targets_no_beat = resonance.target_amplitudes.copy()

    # Update with beat
    resonance.update(frame_with_beat)
    targets_with_beat = resonance.target_amplitudes.copy()

    # Beat attack should boost targets
    assert targets_with_beat.mean() >= targets_no_beat.mean()

    print("✅ Beat attack boost works")


def test_phase_advancement():
    """Test that phases advance over time."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Store initial phases
    initial_phases = resonance.particle_phases.copy()

    # Update multiple times
    frame = HarmonicFrame(
        time=0.0,
        dominant_notes=[],
        harmonics=[],
        chord=None,
        chord_confidence=0.0,
        beat_strength=0.5,
        beat_attack=False,
        spectral_flux=0.1,
        harmonicity=0.5
    )

    for _ in range(10):
        resonance.update(frame, dt=1.0/30.0)

    # Phases should have advanced
    assert not np.allclose(resonance.particle_phases, initial_phases)

    # Phases should still be in [0, 2π]
    assert np.all(resonance.particle_phases >= 0)
    assert np.all(resonance.particle_phases <= 2 * np.pi)

    print("✅ Phase advancement works")


def test_chord_detection():
    """Test chord detection tracking."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Frame with chord
    frame = HarmonicFrame(
        time=0.0,
        dominant_notes=[("C4", 261.63), ("E4", 329.63), ("G4", 392.00)],
        harmonics=[261.63, 523.26, 784.89, 1046.52],
        chord="Cmaj",
        chord_confidence=0.9,
        beat_strength=0.5,
        beat_attack=False,
        spectral_flux=0.3,
        harmonicity=0.8
    )

    resonance.update(frame)

    # Chord should be tracked
    assert resonance.current_chord == "Cmaj"
    assert resonance.chord_confidence == 0.9

    print("✅ Chord detection tracking works")


def test_resonance_stats():
    """Test resonance statistics."""
    zones = FrequencyZones(num_particles=50, seed=42)
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones,
        seed=42
    )

    # Set some amplitudes
    resonance.resonance_amplitudes[:10] = 0.8
    resonance.resonance_amplitudes[10:] = 0.0

    stats = resonance.get_resonance_stats()

    # Check stats
    assert 'avg_amplitude' in stats
    assert 'max_amplitude' in stats
    assert 'active_particles' in stats

    # Should have 10 active particles (>0.1)
    assert stats['active_particles'] == 10

    print("✅ Resonance statistics work")


if __name__ == "__main__":
    print("Running Harmonic Resonance Tests...\n")

    test_resonance_initialization()
    test_amplitude_update()
    test_size_modulation()
    test_trail_intensity()
    test_beat_attack_boost()
    test_phase_advancement()
    test_chord_detection()
    test_resonance_stats()

    print("\n✅ All tests passed!")
