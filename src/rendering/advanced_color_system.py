"""
Advanced Color System - Billions of Color Possibilities

Multi-dimensional color mapping for organic, evolving visuals:
- Velocity vector → Hue (360 degrees)
- Speed → Saturation (energy)
- Audio energy → Brightness
- Acceleration → Color temperature shift
- Particle age → Color evolution
- Orbital characteristics → Palette selection
- Harmonic content → Spectral tint

This creates smooth, organic color transitions driven by physics and music.
"""

import numpy as np
from colorsys import hsv_to_rgb, rgb_to_hsv


class AdvancedColorSystem:
    """
    Sophisticated color engine with multi-dimensional mapping.

    Generates billions of unique colors based on particle state and audio.
    """

    def __init__(self):
        """Initialize color palettes and mapping functions."""

        # Base palettes for different energy states
        self.palettes = {
            'deep_space': {
                'base_hue': 240,  # Deep blue
                'sat_range': (0.6, 0.95),
                'val_range': (0.3, 0.9)
            },
            'nebula': {
                'base_hue': 280,  # Purple-magenta
                'sat_range': (0.7, 1.0),
                'val_range': (0.5, 1.0)
            },
            'plasma': {
                'base_hue': 320,  # Magenta-red
                'sat_range': (0.8, 1.0),
                'val_range': (0.7, 1.0)
            },
            'aurora': {
                'base_hue': 160,  # Cyan-green
                'sat_range': (0.5, 0.9),
                'val_range': (0.6, 1.0)
            }
        }

        # Harmonic color relationships
        self.harmonic_shifts = {
            'root': 0,
            'third': 60,
            'fifth': 120,
            'seventh': 180,
            'ninth': 240,
            'eleventh': 300
        }

    def velocity_to_hue(self, velocity):
        """
        Convert 3D velocity vector to hue (0-360).

        Uses velocity direction in 3D space to create smooth hue transitions.
        Fast particles moving right = warm colors (red/orange)
        Slow particles moving left = cool colors (blue/cyan)
        """
        vx, vy, vz = velocity

        # Project to 2D for hue calculation
        angle = np.arctan2(vy, vx)  # -π to π

        # Map to 0-360 degrees
        hue = (np.degrees(angle) + 180) % 360

        # Modulate by z-component (depth)
        hue_shift = vz * 30  # ±30 degree shift based on z-velocity
        hue = (hue + hue_shift) % 360

        return hue

    def speed_to_saturation(self, speed, audio_energy):
        """
        Map speed and audio to saturation.

        Fast particles = high saturation (vibrant)
        Slow particles = low saturation (pale)
        Audio boosts saturation
        """
        base_sat = np.clip(speed / 3.0, 0.2, 1.0)

        # Audio boost
        audio_boost = audio_energy * 0.3

        return np.clip(base_sat + audio_boost, 0, 1)

    def energy_to_value(self, kinetic_energy, audio_brilliance):
        """
        Map particle energy and audio to brightness (value).

        High energy = bright
        Low energy = dim
        Brilliance frequencies boost brightness
        """
        # Normalize kinetic energy (assume max ~2.0)
        base_value = np.clip(kinetic_energy / 2.0, 0.3, 0.95)

        # Brilliance boost
        brilliance_boost = audio_brilliance * 0.4

        return np.clip(base_value + brilliance_boost, 0.2, 1.0)

    def get_particle_color(self, position, velocity, acceleration, mass,
                          particle_age, audio_frame):
        """
        Compute sophisticated color for a single particle.

        Args:
            position: vec3 position
            velocity: vec3 velocity
            acceleration: vec3 acceleration
            mass: scalar mass
            particle_age: time since particle creation
            audio_frame: AudioFrame10Band

        Returns:
            (r, g, b) tuple [0-1 range]
        """
        speed = np.linalg.norm(velocity)

        # 1. Base hue from velocity direction
        hue = self.velocity_to_hue(velocity)

        # 2. Acceleration modulates hue (directional change = color shift)
        accel_magnitude = np.linalg.norm(acceleration)
        accel_hue_shift = accel_magnitude * 20

        # Audio harmonic shifts
        # Use different frequency bands for different hue shifts
        audio_hue_shift = (
            audio_frame.sub_bass * 10 +
            audio_frame.bass * 15 +
            audio_frame.presence * 25
        )

        final_hue = (hue + accel_hue_shift + audio_hue_shift) % 360

        # 3. Saturation from speed + mid frequencies
        sat = self.speed_to_saturation(speed, audio_frame.mid)

        # 4. Value/brightness from kinetic energy + brilliance
        kinetic = 0.5 * mass * speed * speed
        val = self.energy_to_value(kinetic, audio_frame.brilliance)

        # 5. Age modulation (particles evolve over time)
        age_factor = np.sin(particle_age * 0.5) * 0.1
        val = np.clip(val + age_factor, 0, 1)

        # Convert HSV to RGB
        h_norm = final_hue / 360.0
        rgb = hsv_to_rgb(h_norm, sat, val)

        return rgb

    def get_trail_color_gradient(self, trail_positions, trail_velocities,
                                 audio_frame, n_samples=100):
        """
        Generate smooth color gradient along trail.

        Returns array of RGB colors, one per trail point.
        """
        n_points = len(trail_positions)
        if n_points < 2:
            return [(0.5, 0.5, 0.5)] * n_points

        # Sample colors along trail
        sample_indices = np.linspace(0, n_points-1, min(n_samples, n_points)).astype(int)
        colors = []

        for i in sample_indices:
            if i < len(trail_velocities):
                vel = trail_velocities[i]
                speed = np.linalg.norm(vel)

                # Hue from velocity
                hue = self.velocity_to_hue(vel)

                # Audio modulation
                audio_hue = (
                    audio_frame.low_mid * 30 +
                    audio_frame.high_mid * 20
                )
                hue = (hue + audio_hue) % 360

                # Saturation decreases with age (older = more pale)
                age_factor = i / n_points
                sat = 0.9 - age_factor * 0.5
                sat = np.clip(sat * (0.7 + audio_frame.brilliance * 0.3), 0, 1)

                # Value from speed
                val = np.clip(0.4 + speed / 2.0 + audio_frame.air * 0.3, 0.2, 0.95)

                rgb = hsv_to_rgb(hue / 360.0, sat, val)
                colors.append(rgb)

        return colors

    def get_background_color_field(self, x, y, time, audio_frame):
        """
        Compute background color at position based on audio + time.

        Creates evolving nebula with spectral colors.
        """
        # Distance from center
        dist = np.sqrt(x*x + y*y)

        # Base hue from position (creates color regions)
        angle = np.arctan2(y, x)
        base_hue = (np.degrees(angle) + 180) % 360

        # Time evolution (slow rotation)
        time_hue_shift = (time * 2) % 360

        # Audio modulation (ultra frequencies create shimmers)
        audio_hue = (
            audio_frame.ultra * 60 +
            audio_frame.extreme * 90 +
            np.sin(audio_frame.sub_bass * np.pi) * 30
        )

        final_hue = (base_hue + time_hue_shift + audio_hue) % 360

        # Saturation decreases with distance (center = saturated, edges = pale)
        sat = np.clip(0.8 - dist * 0.2, 0.1, 0.9)

        # Value from bass (pulses)
        bass_energy = (audio_frame.sub_bass + audio_frame.bass) / 2
        val = np.clip(0.1 + bass_energy * 0.3, 0.05, 0.4)

        rgb = hsv_to_rgb(final_hue / 360.0, sat, val)
        return rgb

    def get_palette_from_audio(self, audio_frame):
        """
        Select and blend color palettes based on audio characteristics.

        Different frequency profiles = different color moods.
        """
        # Analyze audio spectral profile
        bass_heavy = (audio_frame.sub_bass + audio_frame.bass) / 2
        mid_heavy = (audio_frame.mid + audio_frame.high_mid) / 2
        treble_heavy = (audio_frame.brilliance + audio_frame.air) / 2

        # Blend palettes based on spectral content
        if bass_heavy > 0.6:
            return 'deep_space'
        elif mid_heavy > 0.6:
            return 'nebula'
        elif treble_heavy > 0.6:
            return 'aurora'
        else:
            return 'plasma'
