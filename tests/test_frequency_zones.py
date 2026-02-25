"""
Unit tests for Frequency Zone System

Tests:
- Particle zone assignment
- Zone energy tracking
- Color hue mapping
- Audio frame integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from audio.frequency_zones import FrequencyZones, ZoneDefinition
from audio.analyzer_10band import AudioFrame10Band


def test_zone_definitions():
    """Test that zone definitions are valid."""
    assert len(FrequencyZones.ZONES) == 10

    # Check frequency ranges don't overlap
    for i in range(len(FrequencyZones.ZONES) - 1):
        zone_current = FrequencyZones.ZONES[i]
        zone_next = FrequencyZones.ZONES[i + 1]
        assert zone_current.freq_high <= zone_next.freq_low

    # Check color hue ranges are valid
    for zone in FrequencyZones.ZONES:
        hue_min, hue_max = zone.color_hue_range
        assert 0.0 <= hue_min <= 1.0
        assert 0.0 <= hue_max <= 1.0
        assert hue_min < hue_max

    print("✅ Zone definitions valid")


def test_particle_assignment():
    """Test particle zone assignment."""
    zones = FrequencyZones(num_particles=100, seed=42)

    # All particles should have a zone
    assert len(zones.particle_zones) == 100

    # All zone IDs should be valid (0-9)
    assert np.all(zones.particle_zones >= 0)
    assert np.all(zones.particle_zones < 10)

    # Should have particles in multiple zones
    unique_zones = np.unique(zones.particle_zones)
    assert len(unique_zones) > 5  # At least half the zones used

    print("✅ Particle assignment works")


def test_zone_distribution():
    """Test that zone distribution favors bass frequencies."""
    zones = FrequencyZones(num_particles=1000, seed=42)

    # Count particles per zone
    zone_counts = [np.sum(zones.particle_zones == i) for i in range(10)]

    # Bass zones (0, 1) should have more particles than extreme zone (9)
    bass_particles = zone_counts[0] + zone_counts[1]
    extreme_particles = zone_counts[9]
    assert bass_particles > extreme_particles

    print("✅ Zone distribution favors bass")


def test_energy_update():
    """Test zone energy updates from audio."""
    zones = FrequencyZones(num_particles=50, seed=42)

    # Create mock audio frame
    audio_frame = AudioFrame10Band(
        sub_bass=0.8, bass=0.7, low_mid=0.5, mid=0.3, high_mid=0.2,
        presence=0.1, brilliance=0.05, air=0.02, ultra=0.01, extreme=0.0,
        beat_strength=0.9, onset_strength=0.5, tempo=120.0, time=1.0
    )

    # Update energies
    zones.update_from_audio(audio_frame, smoothing=0.0)

    # Check energies match audio frame (with tolerance)
    assert abs(zones.get_zone_energy(0) - 0.8) < 0.01  # Sub-bass
    assert abs(zones.get_zone_energy(1) - 0.7) < 0.01  # Bass
    assert abs(zones.get_zone_energy(9) - 0.0) < 0.01  # Extreme

    print("✅ Energy updates work")


def test_energy_smoothing():
    """Test energy smoothing over multiple frames."""
    zones = FrequencyZones(num_particles=50, seed=42)

    # First frame: high bass
    frame1 = AudioFrame10Band(
        sub_bass=1.0, bass=1.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=1.0, onset_strength=0.0, tempo=120.0, time=0.0
    )

    # Second frame: low bass
    frame2 = AudioFrame10Band(
        sub_bass=0.0, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=1.0
    )

    # Update with frame 1
    zones.update_from_audio(frame1, smoothing=0.0)
    energy_after_frame1 = zones.get_zone_energy(0)
    assert abs(energy_after_frame1 - 1.0) < 0.01

    # Update with frame 2 (with smoothing)
    zones.update_from_audio(frame2, smoothing=0.5)
    energy_after_frame2 = zones.get_zone_energy(0)

    # Should be between 0 and 1 due to smoothing
    assert 0.0 < energy_after_frame2 < 1.0

    print("✅ Energy smoothing works")


def test_particle_energy_access():
    """Test getting energy for specific particles."""
    zones = FrequencyZones(num_particles=50, seed=42)

    # Create audio frame with zone 0 active
    audio_frame = AudioFrame10Band(
        sub_bass=0.9, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    zones.update_from_audio(audio_frame, smoothing=0.0)

    # Find a particle in zone 0
    particle_in_zone0 = np.where(zones.particle_zones == 0)[0][0]

    # Get its energy
    energy = zones.get_particle_energy(particle_in_zone0)
    assert abs(energy - 0.9) < 0.01

    print("✅ Particle energy access works")


def test_color_hue_mapping():
    """Test zone to color hue mapping."""
    zones = FrequencyZones(num_particles=50, seed=42)

    # Test each zone has valid hue
    for zone_id in range(10):
        hue = zones.get_zone_color_hue(zone_id, energy=0.5)
        assert 0.0 <= hue <= 1.0

    # Test hue changes with energy
    hue_low = zones.get_zone_color_hue(0, energy=0.0)
    hue_high = zones.get_zone_color_hue(0, energy=1.0)
    assert hue_low < hue_high

    print("✅ Color hue mapping works")


def test_zone_stats():
    """Test zone statistics."""
    zones = FrequencyZones(num_particles=100, seed=42)

    # Create audio frame
    audio_frame = AudioFrame10Band(
        sub_bass=0.5, bass=0.5, low_mid=0.5, mid=0.5, high_mid=0.5,
        presence=0.5, brilliance=0.5, air=0.5, ultra=0.5, extreme=0.5,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    zones.update_from_audio(audio_frame, smoothing=0.0)

    stats = zones.get_zone_stats()

    # Should have stats for all zones
    assert len(stats) == 10

    # Each zone should have expected keys
    for zone_name, zone_stats in stats.items():
        assert 'particles' in zone_stats
        assert 'energy' in zone_stats
        assert 'freq_range' in zone_stats

    print("✅ Zone statistics work")


if __name__ == "__main__":
    print("Running Frequency Zone Tests...\n")

    test_zone_definitions()
    test_particle_assignment()
    test_zone_distribution()
    test_energy_update()
    test_energy_smoothing()
    test_particle_energy_access()
    test_color_hue_mapping()
    test_zone_stats()

    print("\n✅ All tests passed!")
