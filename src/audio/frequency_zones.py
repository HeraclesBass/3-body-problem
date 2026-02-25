"""
Frequency Zone Assignment System for V3

Assigns particles to frequency bands and tracks zone energy.
Creates visual coherence by grouping particles that react to similar frequencies.

Zone Structure (matching 10-band analyzer):
- Zone 0: Sub-bass (20-60 Hz) - Deep bass particles
- Zone 1: Bass (60-150 Hz) - Kick drum particles
- Zone 2: Low-mid (150-300 Hz) - Low melodic particles
- Zone 3: Mid (300-600 Hz) - Vocal range particles
- Zone 4: High-mid (600-1200 Hz) - Bright melodic particles
- Zone 5: Presence (1.2k-2.5k Hz) - Clarity particles
- Zone 6: Brilliance (2.5k-5k Hz) - Bright sparkle particles
- Zone 7: Air (5k-10k Hz) - Shimmer particles
- Zone 8: Ultra (10k-16k Hz) - High sparkle particles
- Zone 9: Extreme (16k-20k Hz) - Ultra-high detail particles

Usage:
    zones = FrequencyZones()

    # Assign particle to zone
    zone_id = zones.assign_particle_zone(particle_index)

    # Update zone energies from audio
    zones.update_from_audio(audio_frame)

    # Get zone energy
    energy = zones.get_zone_energy(zone_id)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ZoneDefinition:
    """Definition of a frequency zone."""
    id: int
    name: str
    freq_low: float
    freq_high: float
    description: str

    # Visual characteristics (used by unified palette)
    color_hue_range: Tuple[float, float]  # (min_hue, max_hue) in [0, 1]


class FrequencyZones:
    """
    Manages particle assignment to frequency zones and zone energy tracking.
    """

    # Zone definitions matching 10-band analyzer
    ZONES = [
        ZoneDefinition(
            id=0, name='sub_bass', freq_low=20, freq_high=60,
            description='Deep bass foundation',
            color_hue_range=(0.0, 0.05)  # Deep red
        ),
        ZoneDefinition(
            id=1, name='bass', freq_low=60, freq_high=150,
            description='Kick drum energy',
            color_hue_range=(0.05, 0.15)  # Red-orange
        ),
        ZoneDefinition(
            id=2, name='low_mid', freq_low=150, freq_high=300,
            description='Low melodic content',
            color_hue_range=(0.1, 0.2)  # Orange-yellow
        ),
        ZoneDefinition(
            id=3, name='mid', freq_low=300, freq_high=600,
            description='Vocal range',
            color_hue_range=(0.15, 0.35)  # Yellow-green
        ),
        ZoneDefinition(
            id=4, name='high_mid', freq_low=600, freq_high=1200,
            description='Bright melodic',
            color_hue_range=(0.3, 0.5)  # Green-cyan
        ),
        ZoneDefinition(
            id=5, name='presence', freq_low=1200, freq_high=2500,
            description='Clarity and presence',
            color_hue_range=(0.45, 0.6)  # Cyan-blue
        ),
        ZoneDefinition(
            id=6, name='brilliance', freq_low=2500, freq_high=5000,
            description='Bright sparkle',
            color_hue_range=(0.55, 0.7)  # Blue
        ),
        ZoneDefinition(
            id=7, name='air', freq_low=5000, freq_high=10000,
            description='Shimmer and air',
            color_hue_range=(0.65, 0.8)  # Blue-purple
        ),
        ZoneDefinition(
            id=8, name='ultra', freq_low=10000, freq_high=16000,
            description='High sparkle',
            color_hue_range=(0.75, 0.9)  # Purple-magenta
        ),
        ZoneDefinition(
            id=9, name='extreme', freq_low=16000, freq_high=20000,
            description='Ultra-high detail',
            color_hue_range=(0.85, 1.0)  # Magenta-red
        ),
    ]

    def __init__(self, num_particles: int, seed: int = 42):
        """
        Initialize frequency zones.

        Args:
            num_particles: Total number of particles
            seed: Random seed for reproducible zone assignment
        """
        self.num_particles = num_particles
        self.rng = np.random.RandomState(seed)

        # Particle to zone mapping
        self.particle_zones = np.zeros(num_particles, dtype=np.int32)

        # Current energy per zone [0, 1]
        self.zone_energies = np.zeros(len(self.ZONES), dtype=np.float32)

        # Previous energy (for smooth transitions)
        self.zone_energies_prev = np.zeros(len(self.ZONES), dtype=np.float32)

        # Assign particles to zones
        self._assign_particles_to_zones()

    def _assign_particles_to_zones(self):
        """
        Assign particles to zones.

        Distribution strategy:
        - More particles in bass zones (visually dominant)
        - Fewer particles in extreme highs (subtle accents)
        """
        # Distribution weights (favor bass/mid frequencies)
        weights = np.array([
            0.15,  # Sub-bass (lots of energy)
            0.20,  # Bass (very prominent)
            0.15,  # Low-mid (melodic)
            0.15,  # Mid (vocal range, important)
            0.12,  # High-mid
            0.10,  # Presence
            0.07,  # Brilliance
            0.04,  # Air
            0.02,  # Ultra (rare)
            0.01,  # Extreme (very rare)
        ])

        # Normalize weights
        weights = weights / weights.sum()

        # Assign zones based on weights
        self.particle_zones = self.rng.choice(
            len(self.ZONES),
            size=self.num_particles,
            p=weights
        )

        # Log distribution
        print(f"Particle zone distribution:")
        for zone in self.ZONES:
            count = np.sum(self.particle_zones == zone.id)
            pct = 100 * count / self.num_particles
            print(f"  Zone {zone.id} ({zone.name:12s}): {count:3d} particles ({pct:5.1f}%)")

    def assign_particle_zone(self, particle_idx: int) -> int:
        """
        Get the zone assignment for a particle.

        Args:
            particle_idx: Particle index

        Returns:
            Zone ID (0-9)
        """
        return int(self.particle_zones[particle_idx])

    def update_from_audio(self, audio_frame, smoothing: float = 0.3):
        """
        Update zone energies from audio frame.

        Args:
            audio_frame: AudioFrame10Band from analyzer
            smoothing: Smoothing factor (0=instant, 1=no change)
        """
        # Store previous energies
        self.zone_energies_prev = self.zone_energies.copy()

        # Extract energies from audio frame (matches zone order)
        new_energies = np.array([
            audio_frame.sub_bass,
            audio_frame.bass,
            audio_frame.low_mid,
            audio_frame.mid,
            audio_frame.high_mid,
            audio_frame.presence,
            audio_frame.brilliance,
            audio_frame.air,
            audio_frame.ultra,
            audio_frame.extreme,
        ], dtype=np.float32)

        # Smooth transition
        self.zone_energies = (
            smoothing * self.zone_energies +
            (1.0 - smoothing) * new_energies
        )

    def get_zone_energy(self, zone_id: int) -> float:
        """
        Get current energy for a zone.

        Args:
            zone_id: Zone ID (0-9)

        Returns:
            Energy value [0, 1]
        """
        return float(self.zone_energies[zone_id])

    def get_particle_energy(self, particle_idx: int) -> float:
        """
        Get energy for a specific particle's zone.

        Args:
            particle_idx: Particle index

        Returns:
            Energy value [0, 1]
        """
        zone_id = self.particle_zones[particle_idx]
        return self.get_zone_energy(zone_id)

    def get_zone_color_hue(self, zone_id: int, energy: float = 0.5) -> float:
        """
        Get color hue for a zone based on energy.

        Args:
            zone_id: Zone ID (0-9)
            energy: Energy value [0, 1] (interpolates within hue range)

        Returns:
            Hue value [0, 1]
        """
        zone = self.ZONES[zone_id]
        hue_min, hue_max = zone.color_hue_range

        # Interpolate hue based on energy
        return hue_min + energy * (hue_max - hue_min)

    def get_particle_color_hue(self, particle_idx: int) -> float:
        """
        Get color hue for a particle based on its zone and energy.

        Args:
            particle_idx: Particle index

        Returns:
            Hue value [0, 1]
        """
        zone_id = self.particle_zones[particle_idx]
        energy = self.get_zone_energy(zone_id)
        return self.get_zone_color_hue(zone_id, energy)

    def get_zone_stats(self) -> dict:
        """Get zone statistics."""
        stats = {}
        for zone in self.ZONES:
            count = np.sum(self.particle_zones == zone.id)
            stats[zone.name] = {
                'particles': int(count),
                'energy': float(self.zone_energies[zone.id]),
                'freq_range': f"{zone.freq_low}-{zone.freq_high} Hz"
            }
        return stats
