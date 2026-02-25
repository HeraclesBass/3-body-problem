"""
Frequency Zone Visual System for V3

Extends V2's UnifiedPalette with zone-based color mapping.
Each frequency zone gets a color region that evolves with music.

Color Strategy:
- Bass zones (0-2): Warm colors (red, orange, yellow)
- Mid zones (3-5): Neutral colors (green, cyan)
- Treble zones (6-9): Cool colors (blue, purple, magenta)
- Smooth transitions between zones and over time
- Individual particle variation within zone for distinction

This creates visual frequency separation while maintaining coherence.
"""

import numpy as np
from colorsys import hsv_to_rgb
from typing import Tuple


class ZonedPalette:
    """
    Frequency zone-aware color palette.

    Integrates with FrequencyZones to assign colors based on
    particle frequency assignments and zone energy levels.
    """

    def __init__(self, frequency_zones):
        """
        Initialize zoned palette.

        Args:
            frequency_zones: FrequencyZones instance
        """
        self.zones = frequency_zones

        # Current zone hue offsets (deviations from base zone hue)
        self.zone_hue_offsets = np.zeros(10, dtype=np.float32)

        # Target offsets (for smooth transitions)
        self.zone_hue_targets = np.zeros(10, dtype=np.float32)

        # Velocities for spring physics
        self.zone_hue_velocities = np.zeros(10, dtype=np.float32)

        # Physics parameters
        self.spring_strength = 0.08  # Gentle transitions
        self.damping = 0.90          # Smooth, no jitter

        # Overall palette shift (global hue rotation)
        self.global_hue_shift = 0.0
        self.global_hue_velocity = 0.0

        # Saturation and value ranges
        self.saturation_range = (0.65, 0.95)  # Vibrant
        self.value_range = (0.50, 1.0)        # Dim to bright

        # Particle variation (degrees of hue variation within zone)
        self.particle_hue_variation = 15.0  # ±15° for distinction

        # Generate per-particle hue offsets
        self._generate_particle_offsets()

    def _generate_particle_offsets(self):
        """Generate random hue offsets for each particle (for variation)."""
        num_particles = len(self.zones.particle_zones)
        # Random offsets in [-variation, +variation]
        self.particle_offsets = np.random.uniform(
            -self.particle_hue_variation,
            self.particle_hue_variation,
            size=num_particles
        )

    def update(self, audio_frame, dt=1.0/30.0):
        """
        Update palette based on audio content.

        Args:
            audio_frame: AudioFrame10Band from analyzer
            dt: Time step (seconds)
        """
        # Update zone energies
        self.zones.update_from_audio(audio_frame, smoothing=0.3)

        # =====================================================================
        # ZONE HUE TARGETS FROM AUDIO
        # =====================================================================
        # Each zone's hue shifts based on its energy
        # High energy → shift toward zone's color peak
        # Low energy → shift toward zone's color base

        for zone_id in range(10):
            energy = self.zones.get_zone_energy(zone_id)

            # Target offset scales with energy
            # ±20° shift at full energy
            self.zone_hue_targets[zone_id] = (energy - 0.5) * 40.0

        # =====================================================================
        # GLOBAL HUE SHIFT FROM OVERALL SPECTRUM
        # =====================================================================
        # Global shift rotates entire palette based on dominant frequencies

        # Calculate spectral centroid (weighted average frequency)
        zone_energies = self.zones.zone_energies
        zone_weights = np.arange(10)  # 0-9
        if zone_energies.sum() > 0:
            spectral_centroid = np.average(zone_weights, weights=zone_energies)
        else:
            spectral_centroid = 5.0  # Midpoint

        # Map centroid to hue shift
        # Low centroid (bass) → negative shift (warmer)
        # High centroid (treble) → positive shift (cooler)
        target_global_shift = (spectral_centroid - 5.0) * 10.0

        # =====================================================================
        # SMOOTH TRANSITIONS (SPRING PHYSICS)
        # =====================================================================
        # Update zone hue offsets
        for zone_id in range(10):
            # Spring force toward target
            force = (self.zone_hue_targets[zone_id] -
                    self.zone_hue_offsets[zone_id]) * self.spring_strength

            # Update velocity
            self.zone_hue_velocities[zone_id] += force
            self.zone_hue_velocities[zone_id] *= self.damping

            # Update position
            self.zone_hue_offsets[zone_id] += self.zone_hue_velocities[zone_id] * dt * 30.0

        # Update global shift
        global_force = (target_global_shift - self.global_hue_shift) * self.spring_strength
        self.global_hue_velocity += global_force
        self.global_hue_velocity *= self.damping
        self.global_hue_shift += self.global_hue_velocity * dt * 30.0

    def get_particle_color(
        self,
        particle_idx: int,
        energy_boost: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Get RGB color for a particle based on its zone.

        Args:
            particle_idx: Particle index
            energy_boost: Additional energy boost [0, 1] for special effects

        Returns:
            (r, g, b) tuple with values [0, 1]
        """
        # Get particle's zone
        zone_id = self.zones.particle_zones[particle_idx]

        # Get base hue from zone definition
        zone_hue_degrees = self.zones.get_zone_color_hue(
            zone_id, energy=0.5
        ) * 360.0

        # Apply zone offset
        zone_hue_degrees += self.zone_hue_offsets[zone_id]

        # Apply global shift
        zone_hue_degrees += self.global_hue_shift

        # Apply particle-specific variation
        zone_hue_degrees += self.particle_offsets[particle_idx]

        # Wrap to [0, 360)
        zone_hue_degrees = zone_hue_degrees % 360.0

        # Convert to [0, 1] for HSV
        hue = zone_hue_degrees / 360.0

        # Get zone energy
        zone_energy = self.zones.get_zone_energy(zone_id)

        # Saturation increases with energy
        saturation = np.interp(
            zone_energy + energy_boost,
            [0.0, 1.0],
            self.saturation_range
        )

        # Value (brightness) also increases with energy
        value = np.interp(
            zone_energy + energy_boost,
            [0.0, 1.0],
            self.value_range
        )

        # Clamp
        saturation = np.clip(saturation, 0.0, 1.0)
        value = np.clip(value, 0.0, 1.0)

        # Convert HSV to RGB
        r, g, b = hsv_to_rgb(hue, saturation, value)

        return (r, g, b)

    def get_particle_color_array(
        self,
        particle_indices: np.ndarray,
        energy_boosts: np.ndarray = None
    ) -> np.ndarray:
        """
        Get colors for multiple particles efficiently.

        Args:
            particle_indices: Array of particle indices
            energy_boosts: Optional array of energy boosts per particle

        Returns:
            Array of shape (N, 3) with RGB colors
        """
        if energy_boosts is None:
            energy_boosts = np.zeros(len(particle_indices))

        colors = np.zeros((len(particle_indices), 3))
        for i, idx in enumerate(particle_indices):
            colors[i] = self.get_particle_color(idx, energy_boosts[i])

        return colors

    def get_zone_color(self, zone_id: int) -> Tuple[float, float, float]:
        """
        Get representative color for a zone.

        Args:
            zone_id: Zone ID (0-9)

        Returns:
            (r, g, b) tuple
        """
        # Base zone hue
        zone_hue_degrees = self.zones.get_zone_color_hue(zone_id, 0.5) * 360.0

        # Apply zone offset and global shift
        zone_hue_degrees += self.zone_hue_offsets[zone_id]
        zone_hue_degrees += self.global_hue_shift
        zone_hue_degrees = zone_hue_degrees % 360.0

        hue = zone_hue_degrees / 360.0

        # Zone energy
        energy = self.zones.get_zone_energy(zone_id)

        # Saturation and value from energy
        saturation = np.interp(energy, [0.0, 1.0], self.saturation_range)
        value = np.interp(energy, [0.0, 1.0], self.value_range)

        r, g, b = hsv_to_rgb(hue, saturation, value)
        return (r, g, b)

    def get_palette_summary(self) -> dict:
        """Get palette state summary for debugging."""
        return {
            'global_hue_shift': float(self.global_hue_shift),
            'zone_hue_offsets': self.zone_hue_offsets.tolist(),
            'zone_energies': self.zones.zone_energies.tolist(),
        }


class PalettePreview:
    """Utility for previewing palette colors (for debugging)."""

    @staticmethod
    def generate_preview_image(
        zoned_palette: ZonedPalette,
        width: int = 1000,
        height: int = 100
    ) -> np.ndarray:
        """
        Generate preview image of current palette.

        Args:
            zoned_palette: ZonedPalette instance
            width: Image width
            height: Image height

        Returns:
            RGB image array [0, 255] uint8
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Show 10 zones as vertical stripes
        zone_width = width // 10

        for zone_id in range(10):
            x_start = zone_id * zone_width
            x_end = (zone_id + 1) * zone_width

            # Get zone color
            r, g, b = zoned_palette.get_zone_color(zone_id)

            # Fill stripe
            image[:, x_start:x_end, 0] = int(r * 255)
            image[:, x_start:x_end, 1] = int(g * 255)
            image[:, x_start:x_end, 2] = int(b * 255)

        return image
