"""
Spectral Particle Spawning System for V3

Dynamically spawns particles based on musical frequency spikes.
Creates "music made visible" effect where notes become particles.

Spawning Logic:
- Frequency spike detected → new particle spawned in that zone
- Particle lifetime tied to note sustain
- Configurable spawn thresholds per band
- Maximum particle cap to prevent overload

Usage:
    spawner = SpectralSpawner(
        max_particles=200,
        frequency_zones=zones
    )

    # Update from harmonic frame
    spawn_events = spawner.update(harmonic_frame, dt=1.0/30.0)

    # Get active particle list
    active_particles = spawner.get_active_particles()
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SpawnedParticle:
    """A dynamically spawned particle."""
    id: int
    zone_id: int
    spawn_time: float
    lifetime: float  # Duration in seconds
    age: float  # Current age in seconds
    frequency: float  # Spawning frequency in Hz
    initial_energy: float  # Energy at spawn

    @property
    def is_alive(self) -> bool:
        """Check if particle is still alive."""
        return self.age < self.lifetime

    @property
    def life_fraction(self) -> float:
        """Get fraction of lifetime remaining [0, 1]."""
        return 1.0 - (self.age / self.lifetime) if self.lifetime > 0 else 0.0


class SpectralSpawner:
    """
    Dynamic particle spawning based on frequency spikes.

    Monitors frequency zones for energy spikes and spawns particles
    that live for the duration of the musical note.
    """

    def __init__(
        self,
        max_particles: int,
        frequency_zones,
        seed: int = 42
    ):
        """
        Initialize spectral spawner.

        Args:
            max_particles: Maximum number of spawned particles
            frequency_zones: FrequencyZones instance
            seed: Random seed
        """
        self.max_particles = max_particles
        self.zones = frequency_zones
        self.rng = np.random.RandomState(seed)

        # Active spawned particles
        self.particles: List[SpawnedParticle] = []

        # Next particle ID
        self.next_id = 0

        # Previous zone energies (for spike detection)
        self.prev_zone_energies = np.zeros(10, dtype=np.float32)

        # Spawn threshold per zone (energy increase required to spawn)
        self.spawn_thresholds = np.array([
            0.15,  # Sub-bass (less frequent spawns)
            0.15,  # Bass
            0.12,  # Low-mid
            0.10,  # Mid (more responsive)
            0.10,  # High-mid
            0.08,  # Presence (very responsive)
            0.08,  # Brilliance
            0.06,  # Air
            0.06,  # Ultra
            0.05,  # Extreme (most responsive)
        ], dtype=np.float32)

        # Cooldown per zone (seconds before next spawn)
        self.zone_cooldowns = np.zeros(10, dtype=np.float32)
        self.cooldown_duration = 0.1  # 100ms cooldown

        # Lifetime parameters
        self.lifetime_base = 1.0  # Base lifetime (seconds)
        self.lifetime_range = (0.5, 2.0)  # Min/max lifetime

        # Current simulation time
        self.current_time = 0.0

    def update(
        self,
        harmonic_frame,
        dt: float = 1.0/30.0
    ) -> List[SpawnedParticle]:
        """
        Update spawner and potentially spawn new particles.

        Args:
            harmonic_frame: HarmonicFrame from harmonic analyzer
            dt: Time step (seconds)

        Returns:
            List of newly spawned particles this frame
        """
        self.current_time += dt
        new_spawns = []

        # Update zone cooldowns
        self.zone_cooldowns = np.maximum(
            0.0,
            self.zone_cooldowns - dt
        )

        # =====================================================================
        # DETECT FREQUENCY SPIKES AND SPAWN PARTICLES
        # =====================================================================
        current_energies = self.zones.zone_energies

        for zone_id in range(10):
            # Check for energy spike
            energy_increase = (
                current_energies[zone_id] -
                self.prev_zone_energies[zone_id]
            )

            # Spawn conditions:
            # 1. Energy spike exceeds threshold
            # 2. Zone is not on cooldown
            # 3. Not at particle cap
            if (energy_increase > self.spawn_thresholds[zone_id] and
                self.zone_cooldowns[zone_id] <= 0.0 and
                len(self.particles) < self.max_particles):

                # Spawn particle
                particle = self._spawn_particle(
                    zone_id=zone_id,
                    energy=current_energies[zone_id],
                    harmonic_frame=harmonic_frame
                )

                self.particles.append(particle)
                new_spawns.append(particle)

                # Set cooldown for this zone
                self.zone_cooldowns[zone_id] = self.cooldown_duration

        # Store current energies for next frame
        self.prev_zone_energies = current_energies.copy()

        # =====================================================================
        # AGE PARTICLES AND REMOVE DEAD ONES
        # =====================================================================
        # Age all particles
        for particle in self.particles:
            particle.age += dt

        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive]

        return new_spawns

    def _spawn_particle(
        self,
        zone_id: int,
        energy: float,
        harmonic_frame
    ) -> SpawnedParticle:
        """
        Spawn a new particle.

        Args:
            zone_id: Zone to spawn in
            energy: Zone energy at spawn
            harmonic_frame: Current harmonic frame

        Returns:
            New SpawnedParticle
        """
        # Determine lifetime based on energy and harmonicity
        # High energy + high harmonicity → longer lifetime (sustained note)
        # Low energy or low harmonicity → shorter lifetime (transient)
        lifetime_factor = (energy + harmonic_frame.harmonicity) / 2.0
        lifetime = np.interp(
            lifetime_factor,
            [0.0, 1.0],
            self.lifetime_range
        )

        # Get frequency from dominant notes
        frequency = 440.0  # Default A4
        if harmonic_frame.dominant_notes:
            frequency = harmonic_frame.dominant_notes[0][1]

        # Create particle
        particle = SpawnedParticle(
            id=self.next_id,
            zone_id=zone_id,
            spawn_time=self.current_time,
            lifetime=lifetime,
            age=0.0,
            frequency=frequency,
            initial_energy=energy
        )

        self.next_id += 1
        return particle

    def get_active_particles(self) -> List[SpawnedParticle]:
        """Get list of currently active particles."""
        return self.particles.copy()

    def get_particle_count(self) -> int:
        """Get number of active particles."""
        return len(self.particles)

    def get_zone_particle_count(self, zone_id: int) -> int:
        """Get number of active particles in a zone."""
        return sum(1 for p in self.particles if p.zone_id == zone_id)

    def clear_all_particles(self):
        """Remove all spawned particles."""
        self.particles.clear()

    def get_spawner_stats(self) -> dict:
        """Get spawner statistics."""
        zone_counts = [
            self.get_zone_particle_count(i) for i in range(10)
        ]

        return {
            'total_particles': len(self.particles),
            'particles_by_zone': zone_counts,
            'avg_lifetime': np.mean([p.lifetime for p in self.particles]) if self.particles else 0.0,
            'avg_age': np.mean([p.age for p in self.particles]) if self.particles else 0.0,
            'next_id': self.next_id,
        }


class SpawnVisualizer:
    """Utility for visualizing particle spawns."""

    @staticmethod
    def get_spawn_timeline(
        spawner: SpectralSpawner,
        duration: float = 10.0,
        resolution: int = 100
    ) -> np.ndarray:
        """
        Generate timeline of particle spawns.

        Args:
            spawner: SpectralSpawner instance
            duration: Timeline duration (seconds)
            resolution: Number of time bins

        Returns:
            2D array (zones, time) of spawn counts
        """
        timeline = np.zeros((10, resolution), dtype=np.int32)

        for particle in spawner.particles:
            # Calculate time bin
            time_bin = int(
                (particle.spawn_time / duration) * resolution
            )
            if 0 <= time_bin < resolution:
                timeline[particle.zone_id, time_bin] += 1

        return timeline
