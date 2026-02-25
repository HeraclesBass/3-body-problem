"""
Unified Color Palette System

Creates color coherence by maintaining a shared palette that evolves with music.
All particles derive colors from the same base palette, ensuring visual harmony.

Design Philosophy:
- Coherence over chaos
- Smooth transitions (spring physics)
- Music-driven evolution
- Individual particles distinguishable but harmonious
"""

import numpy as np
from colorsys import hsv_to_rgb, rgb_to_hsv


class UnifiedPalette:
    """
    Music-driven color palette with smooth evolution.

    All particles share a base hue that transitions smoothly based on
    musical content. Individual particles get slight variations (±60°)
    to remain distinguishable while maintaining overall harmony.
    """

    def __init__(self):
        """Initialize palette with default deep space colors."""
        # Current base hue (0-360°)
        self.base_hue = 240.0  # Start with deep blue

        # Target hue (what we're transitioning toward)
        self.target_hue = 240.0

        # Velocity for smooth spring-damped transitions
        self.hue_velocity = 0.0

        # Physics parameters for smooth transitions
        self.spring_strength = 0.1  # How quickly to chase target
        self.damping = 0.85         # Prevents overshooting

        # Color ranges
        self.saturation_range = (0.6, 0.9)  # Vibrant but not garish
        self.value_range = (0.5, 1.0)       # Dim to bright

        # Palette mode memory
        self.last_dominant = 'balanced'

    def update(self, audio_frame, dt=1.0/30.0):
        """
        Evolve palette based on audio content.

        Args:
            audio_frame: AudioFrame10Band with frequency analysis
            dt: Time step (seconds)

        Palette Logic:
        - Bass-heavy (deep kick, sub) → Deep blue-purple (260°)
        - Mid-heavy (vocals, guitars) → Warm orange (20°)
        - Treble-heavy (cymbals, hi-hats) → Cool cyan (180°)
        - Balanced → Magenta (280°)

        Transitions use spring physics for smooth, organic movement.
        """
        # =====================================================================
        # DETERMINE TARGET HUE FROM AUDIO SPECTRUM
        # =====================================================================
        # Weight frequency bands to determine dominant character
        bass = (audio_frame.sub_bass + audio_frame.bass) / 2
        mid = (audio_frame.mid + audio_frame.high_mid) / 2
        treble = (audio_frame.brilliance + audio_frame.air) / 2

        # Determine dominant spectrum (with hysteresis to prevent flickering)
        threshold = 0.5  # Needs to exceed this to switch modes

        if bass > threshold and bass > mid and bass > treble:
            dominant = 'bass'
            self.target_hue = 260  # Deep blue-purple (cosmic, heavy)
        elif mid > threshold and mid > bass and mid > treble:
            dominant = 'mid'
            self.target_hue = 20   # Warm orange (energetic, vocal)
        elif treble > threshold and treble > bass and treble > mid:
            dominant = 'treble'
            self.target_hue = 180  # Cool cyan (ethereal, bright)
        else:
            dominant = 'balanced'
            self.target_hue = 280  # Magenta (mysterious, balanced)

        # Hysteresis: only switch if different from last mode
        if dominant != self.last_dominant:
            # Switching modes - target hue set above
            self.last_dominant = dominant

        # =====================================================================
        # SMOOTH HUE TRANSITION (SPRING PHYSICS)
        # =====================================================================
        # Calculate shortest angular distance (handle wraparound)
        hue_diff = (self.target_hue - self.base_hue + 180) % 360 - 180

        # Spring acceleration toward target
        acceleration = hue_diff * self.spring_strength

        # Update velocity with damping
        self.hue_velocity += acceleration * dt
        self.hue_velocity *= self.damping

        # Update hue position
        self.base_hue = (self.base_hue + self.hue_velocity) % 360

    def get_particle_color(self, particle_index, n_particles, energy, audio_frame):
        """
        Get color for specific particle within unified palette.

        Args:
            particle_index: Index of particle (0 to n_particles-1)
            n_particles: Total number of particles
            energy: Particle kinetic energy (for brightness)
            audio_frame: Current audio frame (for subtle modulation)

        Returns:
            (r, g, b) tuple in range [0, 1]

        Color Distribution:
        - All particles spread across ±60° from base hue
        - Evenly spaced to create harmonic color relationships
        - Energy modulates brightness (fast = bright)
        - Audio subtly modulates saturation
        """
        # =====================================================================
        # HUE: Spread particles across palette range
        # =====================================================================
        # Particle's offset from base (-0.5 to +0.5)
        offset_fraction = (particle_index / max(n_particles - 1, 1)) - 0.5

        # Map to ±60° spread (harmonic range, not full rainbow)
        hue_offset = offset_fraction * 120.0

        # Final hue
        hue = (self.base_hue + hue_offset) % 360

        # =====================================================================
        # SATURATION: Energy-based with audio modulation
        # =====================================================================
        # Base saturation from energy (0-2 typical kinetic energy)
        base_sat = np.interp(energy, [0, 2], self.saturation_range)

        # Subtle audio modulation (mid frequencies boost saturation)
        audio_sat_boost = audio_frame.mid * 0.15

        saturation = np.clip(base_sat + audio_sat_boost, 0, 1)

        # =====================================================================
        # VALUE (BRIGHTNESS): Energy-based with audio boost
        # =====================================================================
        # Base value from energy
        base_val = np.interp(energy, [0, 2], self.value_range)

        # Audio brilliance boost
        audio_val_boost = audio_frame.brilliance * 0.2

        value = np.clip(base_val + audio_val_boost, 0, 1)

        # =====================================================================
        # CONVERT HSV TO RGB
        # =====================================================================
        rgb = hsv_to_rgb(hue / 360.0, saturation, value)

        return rgb

    def get_trail_color(self, particle_index, n_particles, trail_age, audio_frame):
        """
        Get color for particle trail point.

        Trails fade and cool with age for depth perception.

        Args:
            particle_index: Index of particle
            n_particles: Total particles
            trail_age: Age factor (0=newest, 1=oldest)
            audio_frame: Current audio frame

        Returns:
            (r, g, b) tuple in range [0, 1]
        """
        # Start with particle's base hue
        offset_fraction = (particle_index / max(n_particles - 1, 1)) - 0.5
        hue_offset = offset_fraction * 120.0
        base_hue = (self.base_hue + hue_offset) % 360

        # Age shifts toward blue (cooler = older, depth cue)
        hue_age_shift = trail_age * 40  # 40° shift toward blue
        hue = (base_hue + hue_age_shift) % 360

        # Saturation decreases with age (fades to gray)
        saturation = np.interp(trail_age, [0, 1], [0.8, 0.3])

        # Value decreases with age (dimmer)
        value = np.interp(trail_age, [0, 1], [0.9, 0.2])

        # Audio modulation (subtle on trails)
        value *= (0.7 + audio_frame.brilliance * 0.3)

        rgb = hsv_to_rgb(hue / 360.0, saturation, value)

        return rgb

    def get_background_color(self, time, audio_frame):
        """
        Get color for background elements (nebula, glow fields).

        Args:
            time: Simulation time (for slow evolution)
            audio_frame: Current audio frame

        Returns:
            (r, g, b) tuple in range [0, 1]
        """
        # Background uses base hue with slight rotation over time
        hue = (self.base_hue + time * 2) % 360  # Very slow rotation

        # Low saturation (subtle)
        saturation = 0.3 + audio_frame.sub_bass * 0.2

        # Low value (dark background)
        value = 0.1 + (audio_frame.sub_bass + audio_frame.bass) * 0.15

        rgb = hsv_to_rgb(hue / 360.0, saturation, value)

        return rgb

    def get_current_hue(self):
        """Get current base hue (for debugging/visualization)."""
        return self.base_hue

    def get_palette_info(self):
        """Get palette state for debugging."""
        return {
            'base_hue': self.base_hue,
            'target_hue': self.target_hue,
            'hue_velocity': self.hue_velocity,
            'mode': self.last_dominant
        }
