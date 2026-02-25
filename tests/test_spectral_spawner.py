"""
Unit tests for Spectral Spawner

Tests:
- Particle spawning on frequency spikes
- Lifetime management
- Particle aging and death
- Spawn throttling (cooldown)
- Maximum particle cap
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

# Import directly to avoid warp
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'physics'))
from spectral_spawner import SpectralSpawner, SpawnedParticle
sys.path.pop(0)

from audio.frequency_zones import FrequencyZones
from audio.harmonic_analyzer import HarmonicFrame


def test_spawner_initialization():
    """Test spawner initializes correctly."""
    zones = FrequencyZones(num_particles=50, seed=42)
    spawner = SpectralSpawner(
        max_particles=100,
        frequency_zones=zones,
        seed=42
    )

    assert spawner.max_particles == 100
    assert len(spawner.particles) == 0
    assert spawner.next_id == 0
    assert len(spawner.spawn_thresholds) == 10
    assert len(spawner.zone_cooldowns) == 10

    print("✅ Spawner initialization works")


def test_particle_spawning_on_spike():
    """Test that particles spawn on energy spikes."""
    zones = FrequencyZones(num_particles=50, seed=42)
    spawner = SpectralSpawner(
        max_particles=100,
        frequency_zones=zones,
        seed=42
    )

    # Update zones with audio to create initial state
    from audio.analyzer_10band import AudioFrame10Band
    audio_low = AudioFrame10Band(
        sub_bass=0.0, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    zones.update_from_audio(audio_low)

    # First update (establish baseline)
    frame = HarmonicFrame(
        time=0.0,
        dominant_notes=[("A4", 440.0)],
        harmonics=[440.0, 880.0],
        chord=None,
        chord_confidence=0.0,
        beat_strength=0.0,
        beat_attack=False,
        spectral_flux=0.1,
        harmonicity=0.5
    )
    spawner.update(frame)
    initial_count = spawner.get_particle_count()

    # Create energy spike in bass zone
    audio_high = AudioFrame10Band(
        sub_bass=0.9, bass=0.9, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.1
    )
    zones.update_from_audio(audio_high)

    # Second update (should spawn particles)
    new_spawns = spawner.update(frame, dt=0.1)

    # Should have spawned particles
    assert len(new_spawns) > 0
    assert spawner.get_particle_count() > initial_count

    print("✅ Particle spawning on spike works")


def test_particle_lifetime():
    """Test particle lifetime and aging."""
    zones = FrequencyZones(num_particles=50, seed=42)
    spawner = SpectralSpawner(
        max_particles=100,
        frequency_zones=zones,
        seed=42
    )

    # Manually create a particle with short lifetime
    from audio.harmonic_analyzer import HarmonicFrame
    particle = SpawnedParticle(
        id=0,
        zone_id=0,
        spawn_time=0.0,
        lifetime=0.5,  # 0.5 seconds
        age=0.0,
        frequency=440.0,
        initial_energy=0.8
    )
    spawner.particles.append(particle)

    # Age the particle
    frame = HarmonicFrame(
        time=0.0,
        dominant_notes=[],
        harmonics=[],
        chord=None,
        chord_confidence=0.0,
        beat_strength=0.0,
        beat_attack=False,
        spectral_flux=0.0,
        harmonicity=0.5
    )

    # Update several times
    for _ in range(10):
        spawner.update(frame, dt=0.1)

    # Particle should be dead after 1 second (lifetime was 0.5s)
    assert spawner.get_particle_count() == 0

    print("✅ Particle lifetime works")


def test_spawn_cooldown():
    """Test that spawn cooldown prevents rapid spawning."""
    zones = FrequencyZones(num_particles=50, seed=42)
    spawner = SpectralSpawner(
        max_particles=100,
        frequency_zones=zones,
        seed=42
    )

    # Set up for spawning
    from audio.analyzer_10band import AudioFrame10Band
    audio_low = AudioFrame10Band(
        sub_bass=0.0, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    zones.update_from_audio(audio_low)

    frame = HarmonicFrame(
        time=0.0,
        dominant_notes=[("A4", 440.0)],
        harmonics=[440.0],
        chord=None,
        chord_confidence=0.0,
        beat_strength=0.0,
        beat_attack=False,
        spectral_flux=0.1,
        harmonicity=0.5
    )
    spawner.update(frame, dt=0.01)

    # Create spike
    audio_high = AudioFrame10Band(
        sub_bass=0.9, bass=0.0, low_mid=0.0, mid=0.0, high_mid=0.0,
        presence=0.0, brilliance=0.0, air=0.0, ultra=0.0, extreme=0.0,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.01
    )
    zones.update_from_audio(audio_high)

    # First spawn
    spawns1 = spawner.update(frame, dt=0.01)
    count_after_first = spawner.get_particle_count()

    # Immediate second update (should not spawn due to cooldown)
    spawns2 = spawner.update(frame, dt=0.01)
    count_after_second = spawner.get_particle_count()

    # Second spawn should be blocked
    assert len(spawns2) == 0

    print("✅ Spawn cooldown works")


def test_max_particle_cap():
    """Test that spawner respects maximum particle cap."""
    zones = FrequencyZones(num_particles=50, seed=42)
    spawner = SpectralSpawner(
        max_particles=5,  # Very low cap
        frequency_zones=zones,
        seed=42
    )

    # Manually add particles up to cap
    for i in range(5):
        particle = SpawnedParticle(
            id=i,
            zone_id=0,
            spawn_time=0.0,
            lifetime=10.0,  # Long lifetime
            age=0.0,
            frequency=440.0,
            initial_energy=0.8
        )
        spawner.particles.append(particle)

    # Try to spawn more
    from audio.analyzer_10band import AudioFrame10Band
    audio_high = AudioFrame10Band(
        sub_bass=0.9, bass=0.9, low_mid=0.9, mid=0.9, high_mid=0.9,
        presence=0.9, brilliance=0.9, air=0.9, ultra=0.9, extreme=0.9,
        beat_strength=0.0, onset_strength=0.0, tempo=120.0, time=0.0
    )
    zones.update_from_audio(audio_high)

    frame = HarmonicFrame(
        time=0.0,
        dominant_notes=[],
        harmonics=[],
        chord=None,
        chord_confidence=0.0,
        beat_strength=0.0,
        beat_attack=False,
        spectral_flux=0.9,
        harmonicity=0.5
    )

    spawner.update(frame)

    # Should not exceed cap
    assert spawner.get_particle_count() <= 5

    print("✅ Maximum particle cap works")


def test_spawner_stats():
    """Test spawner statistics."""
    zones = FrequencyZones(num_particles=50, seed=42)
    spawner = SpectralSpawner(
        max_particles=100,
        frequency_zones=zones,
        seed=42
    )

    # Add some particles
    for i in range(10):
        particle = SpawnedParticle(
            id=i,
            zone_id=i % 10,
            spawn_time=0.0,
            lifetime=1.0,
            age=0.0,
            frequency=440.0,
            initial_energy=0.5
        )
        spawner.particles.append(particle)

    stats = spawner.get_spawner_stats()

    assert 'total_particles' in stats
    assert 'particles_by_zone' in stats
    assert stats['total_particles'] == 10
    assert len(stats['particles_by_zone']) == 10

    print("✅ Spawner statistics work")


if __name__ == "__main__":
    print("Running Spectral Spawner Tests...\n")

    test_spawner_initialization()
    test_particle_spawning_on_spike()
    test_particle_lifetime()
    test_spawn_cooldown()
    test_max_particle_cap()
    test_spawner_stats()

    print("\n✅ All tests passed!")
