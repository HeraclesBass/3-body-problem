"""
Harmonic Resonance Engine for V3

Makes particles vibrate and respond to detected musical frequencies.
Creates "music made visible" effect where particles physically resonate with notes.

Resonance Effects:
- Size oscillation based on frequency amplitude
- Trail intensity modulation with harmonics
- Harmonic waves create visible patterns
- Musical chords → geometric formations

Usage:
    resonance = HarmonicResonance(
        num_particles=50,
        frequency_zones=zones
    )

    # Update from harmonic analysis
    resonance.update(harmonic_frame, dt=1.0/30.0)

    # Get modulated particle size
    size = resonance.get_particle_size(particle_idx, base_size=5.0)

    # Get trail intensity
    intensity = resonance.get_trail_intensity(particle_idx)
"""

import numpy as np
from typing import List, Tuple


class HarmonicResonance:
    """
    Particle resonance with musical frequencies.

    Each particle resonates with frequencies in its zone,
    creating visible oscillations synchronized with music.
    """

    def __init__(
        self,
        num_particles: int,
        frequency_zones,
        seed: int = 42
    ):
        """
        Initialize harmonic resonance engine.

        Args:
            num_particles: Number of particles
            frequency_zones: FrequencyZones instance
            seed: Random seed
        """
        self.num_particles = num_particles
        self.zones = frequency_zones
        self.rng = np.random.RandomState(seed)

        # Per-particle resonance phase (for oscillation)
        self.particle_phases = self.rng.uniform(
            0, 2 * np.pi,
            size=num_particles
        )

        # Per-particle resonance frequency (how fast they oscillate)
        # Particles resonate at different rates for visual variety
        self.particle_frequencies = self.rng.uniform(
            0.5, 2.0,  # 0.5x to 2x the driving frequency
            size=num_particles
        )

        # Current resonance amplitudes per particle [0, 1]
        self.resonance_amplitudes = np.zeros(num_particles, dtype=np.float32)

        # Target amplitudes (for smooth transitions)
        self.target_amplitudes = np.zeros(num_particles, dtype=np.float32)

        # Amplitude velocities (spring physics)
        self.amplitude_velocities = np.zeros(num_particles, dtype=np.float32)

        # Physics parameters
        self.spring_strength = 0.15  # How quickly to reach target
        self.damping = 0.88          # Prevent jitter

        # Size modulation parameters
        self.size_modulation_range = (0.7, 1.5)  # Min/max size multiplier

        # Trail intensity parameters
        self.trail_intensity_range = (0.3, 1.0)  # Min/max trail brightness

        # Harmonic content tracking (for chord detection effects)
        self.current_chord = None
        self.chord_confidence = 0.0

        # Chord formation patterns (geometric arrangements)
        self.chord_patterns = {
            'maj': self._generate_major_pattern(),
            'min': self._generate_minor_pattern(),
            '7': self._generate_seventh_pattern(),
        }

    def _generate_major_pattern(self) -> np.ndarray:
        """Generate pattern multipliers for major chord (happy/bright)."""
        # Major chords → particles expand outward
        return np.ones(self.num_particles) * 1.2

    def _generate_minor_pattern(self) -> np.ndarray:
        """Generate pattern multipliers for minor chord (sad/dark)."""
        # Minor chords → particles contract inward
        return np.ones(self.num_particles) * 0.8

    def _generate_seventh_pattern(self) -> np.ndarray:
        """Generate pattern multipliers for 7th chord (complex/jazzy)."""
        # 7th chords → particles create wave patterns
        phases = np.linspace(0, 2*np.pi, self.num_particles)
        return 1.0 + 0.2 * np.sin(phases * 3)

    def update(self, harmonic_frame, dt: float = 1.0/30.0):
        """
        Update resonance from harmonic analysis.

        Args:
            harmonic_frame: HarmonicFrame from harmonic analyzer
            dt: Time step (seconds)
        """
        # Update chord tracking
        self.current_chord = harmonic_frame.chord
        self.chord_confidence = harmonic_frame.chord_confidence

        # =====================================================================
        # UPDATE TARGET AMPLITUDES FROM FREQUENCY ZONES
        # =====================================================================
        # Each particle's target amplitude is its zone's energy
        for particle_idx in range(self.num_particles):
            zone_id = self.zones.particle_zones[particle_idx]
            zone_energy = self.zones.get_zone_energy(zone_id)

            # Target amplitude from zone energy
            self.target_amplitudes[particle_idx] = zone_energy

            # Boost amplitude on beat attacks
            if harmonic_frame.beat_attack:
                self.target_amplitudes[particle_idx] = min(
                    1.0,
                    zone_energy + 0.3  # +30% on beat
                )

            # Boost amplitude based on spectral flux (note onsets)
            if harmonic_frame.spectral_flux > 0.7:
                self.target_amplitudes[particle_idx] = min(
                    1.0,
                    self.target_amplitudes[particle_idx] + 0.2
                )

        # =====================================================================
        # SMOOTH AMPLITUDE TRANSITIONS (SPRING PHYSICS)
        # =====================================================================
        for particle_idx in range(self.num_particles):
            # Spring force toward target
            force = (
                (self.target_amplitudes[particle_idx] -
                 self.resonance_amplitudes[particle_idx]) *
                self.spring_strength
            )

            # Update velocity
            self.amplitude_velocities[particle_idx] += force
            self.amplitude_velocities[particle_idx] *= self.damping

            # Update amplitude
            self.resonance_amplitudes[particle_idx] += (
                self.amplitude_velocities[particle_idx] * dt * 30.0
            )

            # Clamp to [0, 1]
            self.resonance_amplitudes[particle_idx] = np.clip(
                self.resonance_amplitudes[particle_idx],
                0.0, 1.0
            )

        # =====================================================================
        # ADVANCE OSCILLATION PHASES
        # =====================================================================
        # Phase advances based on current beat strength
        # (particles oscillate faster during strong beats)
        phase_speed = 2.0 * np.pi * 2.0  # 2 Hz base frequency

        # Modulate speed by beat strength
        beat_multiplier = 1.0 + harmonic_frame.beat_strength * 2.0

        # Advance phases
        self.particle_phases += (
            phase_speed * beat_multiplier * dt * self.particle_frequencies
        )

        # Wrap phases to [0, 2π]
        self.particle_phases = self.particle_phases % (2 * np.pi)

    def get_particle_size(
        self,
        particle_idx: int,
        base_size: float = 5.0
    ) -> float:
        """
        Get resonance-modulated particle size.

        Args:
            particle_idx: Particle index
            base_size: Base particle size

        Returns:
            Modulated size
        """
        # Get resonance amplitude and phase
        amplitude = self.resonance_amplitudes[particle_idx]
        phase = self.particle_phases[particle_idx]

        # Oscillation value: [-1, 1]
        oscillation = np.sin(phase)

        # Modulation factor based on amplitude and oscillation
        # High amplitude → larger oscillations
        modulation = 1.0 + (amplitude * oscillation * 0.3)

        # Apply chord pattern if active
        if self.current_chord and self.chord_confidence > 0.6:
            chord_type = self._extract_chord_type(self.current_chord)
            if chord_type in self.chord_patterns:
                pattern_mult = self.chord_patterns[chord_type][particle_idx]
                modulation *= pattern_mult

        # Clamp modulation to valid range
        modulation = np.clip(
            modulation,
            self.size_modulation_range[0],
            self.size_modulation_range[1]
        )

        return base_size * modulation

    def get_trail_intensity(
        self,
        particle_idx: int
    ) -> float:
        """
        Get trail intensity for particle based on resonance.

        Args:
            particle_idx: Particle index

        Returns:
            Trail intensity [0, 1]
        """
        # Trail intensity driven by amplitude
        amplitude = self.resonance_amplitudes[particle_idx]

        # Map amplitude to trail intensity range
        intensity = np.interp(
            amplitude,
            [0.0, 1.0],
            self.trail_intensity_range
        )

        return float(intensity)

    def get_resonance_strength(
        self,
        particle_idx: int
    ) -> float:
        """
        Get overall resonance strength for particle.

        Args:
            particle_idx: Particle index

        Returns:
            Resonance strength [0, 1]
        """
        return float(self.resonance_amplitudes[particle_idx])

    def _extract_chord_type(self, chord_name: str) -> str:
        """Extract chord type from chord name (e.g., 'Cmaj' -> 'maj')."""
        if 'maj' in chord_name:
            return 'maj'
        elif 'min' in chord_name:
            return 'min'
        elif '7' in chord_name:
            return '7'
        return 'maj'  # Default

    def get_resonance_stats(self) -> dict:
        """Get resonance statistics for debugging."""
        return {
            'avg_amplitude': float(self.resonance_amplitudes.mean()),
            'max_amplitude': float(self.resonance_amplitudes.max()),
            'active_particles': int(np.sum(self.resonance_amplitudes > 0.1)),
            'current_chord': self.current_chord,
            'chord_confidence': float(self.chord_confidence),
        }


class ResonanceVisualizer:
    """Utility for visualizing resonance patterns."""

    @staticmethod
    def get_resonance_heatmap(
        resonance: HarmonicResonance,
        particle_positions: np.ndarray,
        grid_size: int = 100
    ) -> np.ndarray:
        """
        Generate 2D heatmap of resonance amplitudes.

        Args:
            resonance: HarmonicResonance instance
            particle_positions: Array of shape (N, 2) with x,y positions
            grid_size: Heatmap resolution

        Returns:
            2D array of resonance intensities
        """
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Normalize positions to grid
        positions_norm = particle_positions.copy()
        pos_min = positions_norm.min(axis=0)
        pos_max = positions_norm.max(axis=0)
        positions_norm = (positions_norm - pos_min) / (pos_max - pos_min)
        positions_norm *= (grid_size - 1)
        positions_norm = positions_norm.astype(int)

        # Splat resonance amplitudes onto grid
        for i in range(len(particle_positions)):
            x, y = positions_norm[i]
            if 0 <= x < grid_size and 0 <= y < grid_size:
                amplitude = resonance.resonance_amplitudes[i]
                heatmap[y, x] = max(heatmap[y, x], amplitude)

        return heatmap
