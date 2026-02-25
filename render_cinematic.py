#!/usr/bin/env python3
"""
Celestial Chaos - STELLAR FREQUENCY VISUALIZER

Music visualization with stars reacting to 10 frequency bands.
Each star is assigned a frequency band and pulses with that frequency.

Features:
- All bodies are stars (red dwarf → blue giant spectrum)
- 10-band frequency visualization (each star = one band)
- Deep parallax starfield background
- Dramatic audio-reactive glow and color
- Strong gravity, tight orbits
"""

import sys
import time
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from enum import Enum
import colorsys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from physics.nbody import NBodySimulation, SimulationConfig
from audio.analyzer import AudioAnalyzer
from audio.analyzer_10band import AudioAnalyzer10Band


# =============================================================================
# FREQUENCY BAND COLORS (10 bands - music visualization palette)
# =============================================================================

# Each band gets a distinct color from warm (bass) to cool (treble)
BAND_COLORS = [
    (255, 50, 30),    # 0: Sub-bass (20-60Hz) - Deep red
    (255, 100, 20),   # 1: Bass (60-250Hz) - Orange-red
    (255, 180, 40),   # 2: Low-mid (250-500Hz) - Orange-yellow
    (255, 255, 80),   # 3: Mid (500-2kHz) - Yellow
    (180, 255, 100),  # 4: High-mid (2-4kHz) - Yellow-green
    (80, 255, 150),   # 5: Presence (4-6kHz) - Green-cyan
    (50, 220, 255),   # 6: Brilliance (6-10kHz) - Cyan
    (80, 150, 255),   # 7: Air (10-14kHz) - Blue
    (150, 100, 255),  # 8: Ultra (14-18kHz) - Purple
    (255, 80, 200),   # 9: Extreme (18-20kHz) - Magenta
]

BAND_NAMES = [
    "SUB", "BASS", "LOW", "MID", "HIGH",
    "PRES", "BRIL", "AIR", "ULTR", "EXTR"
]


# =============================================================================
# STAR TYPES (based on frequency band assignment)
# =============================================================================

class StarType(Enum):
    RED_DWARF = 0      # Sub-bass - small, deep red
    ORANGE_DWARF = 1   # Bass - orange
    YELLOW_DWARF = 2   # Low-mid - yellow-orange
    YELLOW_STAR = 3    # Mid - yellow (sun-like)
    WHITE_STAR = 4     # High-mid - white-yellow
    WHITE_BLUE = 5     # Presence - white-blue
    BLUE_STAR = 6      # Brilliance - blue
    BLUE_GIANT = 7     # Air - bright blue
    VIOLET_STAR = 8    # Ultra - violet
    PLASMA_STAR = 9    # Extreme - hot pink/magenta


# =============================================================================
# QUALITY SETTINGS
# =============================================================================

class QualityPreset(Enum):
    DRAFT = "draft"
    GOOD = "good"
    BEST = "best"

QUALITY_SETTINGS = {
    QualityPreset.DRAFT: {
        'glow_layers': 6,
        'bg_star_count': 300,
        'parallax_layers': 2,
        'blur_passes': 0,
        'ffmpeg_preset': 'ultrafast',
        'ffmpeg_crf': '28',
    },
    QualityPreset.GOOD: {
        'glow_layers': 12,
        'bg_star_count': 600,
        'parallax_layers': 4,
        'blur_passes': 1,
        'ffmpeg_preset': 'fast',
        'ffmpeg_crf': '22',
    },
    QualityPreset.BEST: {
        'glow_layers': 20,
        'bg_star_count': 1200,
        'parallax_layers': 6,
        'blur_passes': 2,
        'ffmpeg_preset': 'slow',
        'ffmpeg_crf': '18',
    },
}


# =============================================================================
# SPHERICAL COSMIC ENVIRONMENT - 360° REALISTIC SPACE
# =============================================================================

class SphericalCosmos:
    """
    360-degree realistic space environment with:
    - Distant nebula structures (very far away)
    - Deep star fields with realistic magnitude distribution
    - Dust lanes with color extinction
    - Milky Way galactic structure
    - Spherical parallax projection
    - HDR color mapping
    """

    def __init__(self, width: int, height: int, quality: QualityPreset):
        self.width = width
        self.height = height
        self.quality = quality

        rng = np.random.RandomState(42)

        # ===== LAYER 1: Distant Nebula Structures (10,000+ ly away) =====
        nebula_colors = [
            (150, 50, 80),   # Emission nebula - red
            (200, 100, 50),  # Supernova remnant - orange
            (100, 150, 200), # Reflection nebula - blue
            (180, 100, 200), # Planetary nebula - purple
            (220, 180, 100), # Dust reflection - yellow
        ]
        self.distant_nebulae = []
        for _ in range(8):
            self.distant_nebulae.append({
                'x': rng.uniform(-1, 1),  # Spherical coords [-1, 1]
                'y': rng.uniform(-1, 1),
                'z': rng.uniform(-1, 1),
                'size': rng.uniform(0.3, 0.8),  # Angular size
                'color': nebula_colors[int(rng.uniform(0, len(nebula_colors)))],
                'intensity': rng.uniform(0.3, 0.7),
                'rotation': rng.uniform(0, 2 * np.pi),
            })
            # Normalize to unit sphere
            norm = np.sqrt(self.distant_nebulae[-1]['x']**2 +
                          self.distant_nebulae[-1]['y']**2 +
                          self.distant_nebulae[-1]['z']**2)
            if norm > 0:
                self.distant_nebulae[-1]['x'] /= norm
                self.distant_nebulae[-1]['y'] /= norm
                self.distant_nebulae[-1]['z'] /= norm

        # ===== LAYER 2: Deep Star Field (realistic magnitude distribution) =====
        # Millions of stars with proper Milky Way distribution
        base_star_count = QUALITY_SETTINGS[quality]['bg_star_count'] * 3
        self.deep_stars = []

        # Brightest stars (rare) - proper Milky Way coordinates
        for _ in range(int(base_star_count * 0.01)):
            self._add_star(rng, mag=rng.uniform(0, 2), weight=3.0)

        # Bright stars - concentrated around galactic plane
        for _ in range(int(base_star_count * 0.05)):
            self._add_star(rng, mag=rng.uniform(2, 4), weight=2.0)

        # Medium stars
        for _ in range(int(base_star_count * 0.20)):
            self._add_star(rng, mag=rng.uniform(4, 5), weight=1.0)

        # Faint stars - all around
        for _ in range(int(base_star_count * 0.74)):
            self._add_star(rng, mag=rng.uniform(5, 7), weight=0.5)

        # ===== LAYER 3: Dust & Color =====
        self.dust_map = []
        for _ in range(15):
            # Large dust clouds that cause reddening
            angle = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0, 1)
            self.dust_map.append({
                'x': np.cos(angle) * rad,
                'y': np.sin(angle) * (rng.uniform(-0.3, 0.3)),  # Flatter distribution
                'z': np.sin(angle) * rad * rng.uniform(-0.5, 0.5),
                'size': rng.uniform(0.3, 1.0),
                'opacity': rng.uniform(0.1, 0.4),
                'color_shift': (rng.uniform(0.8, 1.0), rng.uniform(0.6, 0.9), rng.uniform(0.4, 0.7)),
            })
            # Normalize
            norm = np.sqrt(self.dust_map[-1]['x']**2 + self.dust_map[-1]['y']**2 + self.dust_map[-1]['z']**2)
            if norm > 0:
                self.dust_map[-1]['x'] /= norm
                self.dust_map[-1]['y'] /= norm
                self.dust_map[-1]['z'] /= norm

        # ===== LAYER 4: Milky Way Structure =====
        # Galactic core glow and disk
        self.galactic_center = {
            'x': 0.1, 'y': 0.0, 'z': 0.0,  # Offset from center
            'glow_size': 0.5,
            'color': (200, 150, 80),  # Golden/orange
        }

        # ===== LAYER 5: Shader Effects (Hyper-dimensional Background) =====
        # Distant gravitational lens centers (very far away, subtle effect)
        self.lens_fields = []
        for _ in range(3):  # 3 subtle warp points
            self.lens_fields.append({
                'x': rng.uniform(-1, 1),
                'y': rng.uniform(-1, 1),
                'z': rng.uniform(-1, 1),
                'strength': rng.uniform(0.05, 0.15),  # Very subtle
                'band_id': int(rng.uniform(0, 10)),
                'frequency': rng.uniform(1.0, 2.0),  # Oscillation frequency
            })
            # Normalize
            norm = np.sqrt(self.lens_fields[-1]['x']**2 +
                          self.lens_fields[-1]['y']**2 +
                          self.lens_fields[-1]['z']**2)
            if norm > 0:
                self.lens_fields[-1]['x'] /= norm
                self.lens_fields[-1]['y'] /= norm
                self.lens_fields[-1]['z'] /= norm

        # ===== LAYER 6: Distant Stellar Remnants (Particle System) =====
        # Scattered particles representing supernovae far away
        self.distant_remnants = []
        for _ in range(50):  # Many tiny distant particles
            self.distant_remnants.append({
                'x': rng.uniform(-1, 1),
                'y': rng.uniform(-1, 1),
                'z': rng.uniform(-1, 1),
                'brightness': rng.uniform(0.3, 0.8),
                'color_idx': rng.choice([0, 1, 2]),  # Red, orange, purple
                'phase': rng.uniform(0, 2 * np.pi),
                'pulse_speed': rng.uniform(0.5, 1.5),
            })
            # Normalize to unit sphere (far background)
            norm = np.sqrt(self.distant_remnants[-1]['x']**2 +
                          self.distant_remnants[-1]['y']**2 +
                          self.distant_remnants[-1]['z']**2)
            if norm > 0:
                self.distant_remnants[-1]['x'] /= norm
                self.distant_remnants[-1]['y'] /= norm
                self.distant_remnants[-1]['z'] /= norm

        # Audio reactivity
        self.band_energies = [0.5] * 10
        self.beat_strength = 0.0
        self.frame_time = 0.0

    def _add_star(self, rng, mag, weight):
        """Add a star to the deep field with realistic distribution."""
        # Milky Way disk concentration
        latitude = rng.normal(0, 0.15) if rng.random() < 0.7 else rng.uniform(-1, 1)
        longitude = rng.uniform(0, 2 * np.pi)

        # Convert to Cartesian on unit sphere
        lat_rad = latitude
        lon_rad = longitude
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        # Brightness from magnitude
        brightness = 0.01 + 0.12 * (1 - mag / 7) ** 2.5
        brightness *= weight

        # Realistic stellar colors
        temp_k = 10000 - mag * 1000  # Hotter = bluer
        if temp_k > 7500:
            color = (200, 210, 255)  # Blue
        elif temp_k > 6000:
            color = (255, 255, 240)  # White
        elif temp_k > 4500:
            color = (255, 255, 200)  # Yellow
        elif temp_k > 3500:
            color = (255, 200, 100)  # Orange
        else:
            color = (255, 100, 50)   # Red dwarf

        self.deep_stars.append({
            'x': x, 'y': y, 'z': z,
            'mag': mag,
            'brightness': brightness,
            'color': color,
            'twinkle_phase': rng.uniform(0, 2 * np.pi),
            'twinkle_speed': rng.uniform(0.5, 2.0),
            'twinkle_amp': 0.05 + rng.uniform(0, 0.1),
        })

    def spherical_to_screen(self, x: float, y: float, z: float) -> tuple:
        """Project spherical coordinates to screen with perspective."""
        # Distance from viewer (deeper = smaller on screen)
        depth = z + 1.5  # Shift so we're inside the sphere
        if depth < 0.1:
            return None, None

        # Perspective projection
        scale = 1.0 / (depth + 0.5)
        sx = int((x * scale + 1) * self.width / 2)
        sy = int((y * scale + 1) * self.height / 2)

        # Clamp to screen
        if sx < 0 or sx >= self.width or sy < 0 or sy >= self.height:
            return None, None

        return sx, sy, depth

    def _perlin_noise_2d(self, x: float, y: float, scale: float = 1.0) -> float:
        """
        Simple Perlin-like noise using sine waves and hashing.
        Creates organic-looking nebula cloud patterns.
        """
        # Hash function for reproducible random values
        def hash_coord(xi, yi):
            n = np.sin(xi * 12.9898 + yi * 78.233) * 43758.5453
            return n - np.floor(n)

        # Scale coordinates
        xf = x * scale
        yf = y * scale

        # Integer and fractional parts
        xi = int(np.floor(xf))
        yi = int(np.floor(yf))
        xfrac = xf - xi
        yfrac = yf - yi

        # Smooth interpolation
        u = xfrac * xfrac * (3.0 - 2.0 * xfrac)
        v = yfrac * yfrac * (3.0 - 2.0 * yfrac)

        # Hash values at corners
        n00 = hash_coord(xi, yi)
        n10 = hash_coord(xi + 1, yi)
        n01 = hash_coord(xi, yi + 1)
        n11 = hash_coord(xi + 1, yi + 1)

        # Interpolate
        nx0 = n00 * (1 - u) + n10 * u
        nx1 = n01 * (1 - u) + n11 * u
        return nx0 * (1 - v) + nx1 * v

    def _render_gravitational_lens(self, draw, time_offset: float):
        """
        Render subtle gravitational lens distortion effects.
        Creates a warping of the background based on distant black hole-like objects.
        """
        for lens in self.lens_fields:
            energy = self.band_energies[lens['band_id']]

            # Subtle oscillating strength based on audio
            oscillation = np.sin(time_offset * lens['frequency']) * 0.5 + 0.5
            strength = lens['strength'] * (0.5 + oscillation * 0.5) * (0.7 + energy * 0.3)

            # Project lens position to screen
            sx, sy, depth = self.spherical_to_screen(lens['x'], lens['y'], lens['z'])

            if sx is not None:
                # Create subtle warp glow (very faint)
                warp_size = 80 * (1.0 / (depth + 0.5))
                glow_intensity = strength * 0.08

                for layer in range(2):
                    layer_size = warp_size * (1 - layer * 0.3)
                    layer_color = int(glow_intensity * 20 * (2 - layer))

                    ri = int(layer_size)
                    if ri > 2:
                        draw.ellipse([sx - ri, sy - ri, sx + ri, sy + ri],
                                    outline=(layer_color, layer_color // 2, layer_color // 2), width=1)

    def _render_distant_remnants(self, draw, time_offset: float):
        """
        Render distant stellar remnant particles.
        Many tiny, faint particles suggesting distant supernovae and nebulosity.
        """
        remnant_colors = [
            (255, 80, 50),    # Red emission
            (255, 150, 50),   # Orange
            (180, 100, 200),  # Purple
        ]

        for remnant in self.distant_remnants:
            # Project to screen
            sx, sy, depth = self.spherical_to_screen(remnant['x'], remnant['y'], remnant['z'])

            if sx is not None:
                # Pulsing brightness based on frequency band
                band_energy = self.band_energies[int(remnant['brightness'] * 10) % 10]
                pulse = np.sin(time_offset * remnant['pulse_speed'] + remnant['phase']) * 0.5 + 0.5
                brightness = remnant['brightness'] * (0.4 + pulse * 0.4) * (0.8 + band_energy * 0.2)

                # Perspective fade
                brightness *= (0.5 / (depth + 0.5))

                if brightness > 0.01:
                    color = remnant_colors[remnant['color_idx']]
                    r = int(color[0] * brightness)
                    g = int(color[1] * brightness)
                    b = int(color[2] * brightness)

                    # Draw as tiny point
                    draw.point((sx, sy), fill=(r, g, b))

    def _render_nebula_clouds(self, draw, time_offset: float):
        """
        Render simplified nebula clouds (optimized for performance).
        Uses simple gradient patterns instead of expensive Perlin noise.
        """
        # Simplified nebula: just a few audio-reactive gradient blobs
        # Much faster than Perlin noise while maintaining atmosphere

        # Only render 3-4 large gradient blobs based on audio
        for i in range(4):
            # Position based on layer index
            x_pos = (i / 4.0) * self.width
            y_pos = self.height * (0.3 + 0.4 * ((i % 2) * 2 - 0.5))

            # Get audio reactivity from different bands
            band_idx = i * 2 % 10
            energy = self.band_energies[band_idx]

            # Size oscillates gently
            blob_size = 150 + energy * 100

            # Color based on energy
            if energy > 0.5:
                color = (int(100 * energy), 80, int(200 * energy))
            else:
                color = (180, int(60 * energy), 40)

            # Draw as soft gradient blob
            for layer in range(5):
                alpha = 0.08 * (5 - layer) / 5 * energy
                layer_size = blob_size * (1 - layer * 0.2)
                layer_color = tuple(int(c * alpha) for c in color)

                if layer_size > 5:
                    draw.ellipse([
                        x_pos - layer_size, y_pos - layer_size,
                        x_pos + layer_size, y_pos + layer_size
                    ], fill=layer_color)

    def update(self, audio_frame, band_energies):
        """Update with audio data and time tracking."""
        self.band_energies = band_energies
        if hasattr(audio_frame, 'beat_strength'):
            self.beat_strength = audio_frame.beat_strength
        else:
            self.beat_strength = 0.0

        # Increment frame time for supernova aging
        self.frame_time += 1.0

    def render(self, draw, time_offset: float, viewport_offset=None):
        """Render 360° cosmic environment."""

        # Deep black void with subtle gradient
        draw.rectangle([0, 0, self.width, self.height], fill=(3, 2, 8))

        # Add space gradient - subtle blue at edges, darker at center
        for y in range(self.height):
            t = (y / self.height - 0.5) * 2  # -1 to 1
            brightness = int(5 + 8 * abs(t) ** 2)  # Darker center, brighter edges
            draw.rectangle([0, y, self.width, y + 1], fill=(brightness // 2, brightness // 3, brightness))

        # ===== Render Distant Nebulae =====
        nebula_depths = []
        for nebula in self.distant_nebulae:
            sx, sy, depth = self.spherical_to_screen(nebula['x'], nebula['y'], nebula['z'])
            if sx is not None:
                nebula_depths.append((depth, nebula, sx, sy))

        # Draw nebulae back-to-front (z-sorting)
        for depth, nebula, sx, sy in sorted(nebula_depths, key=lambda x: x[0]):
            # Nebula size scales with perspective
            size = nebula['size'] * 200 * (1.0 / (depth + 0.3))

            # Audio-reactive intensity
            energy = self.band_energies[int(nebula['color'][0] / 255 * 10) % 10]
            intensity = nebula['intensity'] * (0.5 + energy * 0.5)

            # Draw nebula as soft gradient
            color = nebula['color']
            for layer in range(5):
                alpha = intensity * (5 - layer) / 5
                layer_size = size * (1 - layer * 0.15)
                layer_color = tuple(int(c * alpha) for c in color)

                ri = int(layer_size)
                if ri > 1:
                    draw.ellipse([sx - ri, sy - ri, sx + ri, sy + ri], fill=layer_color)

        # ===== Render Deep Star Field (back-to-front) =====
        star_depths = []
        for star in self.deep_stars:
            sx, sy, depth = self.spherical_to_screen(star['x'], star['y'], star['z'])
            if sx is not None:
                star_depths.append((depth, star, sx, sy))

        for depth, star, sx, sy in sorted(star_depths, key=lambda x: x[0])[:200]:  # Top 200 brightest visible

            # Twinkle effect
            twinkle = 1.0 - star['twinkle_amp'] * (1 - np.sin(time_offset * star['twinkle_speed'] + star['twinkle_phase']))

            # Perspective: farther stars are dimmer
            brightness = star['brightness'] * twinkle * (0.3 + 0.7 / (depth + 1))

            if brightness < 0.01:
                continue

            # Apply color
            r = int(star['color'][0] * brightness)
            g = int(star['color'][1] * brightness)
            b = int(star['color'][2] * brightness)

            # Draw star with subtle glow for bright ones
            if star['mag'] < 3:
                draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=(r // 2, g // 2, b // 2))
            draw.point((sx, sy), fill=(min(255, r), min(255, g), min(255, b)))

        # ===== Render Dust Extinction =====
        for dust in self.dust_map:
            sx, sy, depth = self.spherical_to_screen(dust['x'], dust['y'], dust['z'])
            if sx is not None:
                size = dust['size'] * 300 * (1.0 / (depth + 0.3))

                # Audio-reactive dust glow
                energy = self.band_energies[int(dust['color_shift'][0] * 10) % 10]
                opacity = dust['opacity'] * (0.3 + energy * 0.7)

                # Draw dust as semi-transparent regions
                for layer in range(3):
                    alpha = opacity * (3 - layer) / 3
                    layer_size = size * (1 - layer * 0.2)
                    layer_color = tuple(int(c * alpha * 50) for c in dust['color_shift'])  # Very dark

                    ri = int(layer_size)
                    if ri > 1:
                        draw.ellipse([sx - ri, sy - ri, sx + ri, sy + ri], fill=layer_color)

        # ===== Render Shader Effects =====
        # Subtle gravitational lens distortion (very faint, atmospheric)
        self._render_gravitational_lens(draw, time_offset)

        # Perlin noise nebula clouds (organic, layered)
        self._render_nebula_clouds(draw, time_offset)

        # Distant stellar remnants (tiny particles, far background)
        self._render_distant_remnants(draw, time_offset)

        # ===== Render Milky Way Glow =====
        core = self.galactic_center
        csx, csy, cdepth = self.spherical_to_screen(core['x'], core['y'], core['z'])
        if csx is not None:
            # Large soft glow
            glow_size = core['glow_size'] * 400
            glow_color = core['color']

            for layer in range(8):
                alpha = 0.1 * (8 - layer) / 8 * (0.4 + self.beat_strength * 0.6)
                layer_size = glow_size * (1 - layer * 0.1)
                layer_color = tuple(int(c * alpha) for c in glow_color)

                ri = int(layer_size)
                if ri > 5:
                    draw.ellipse([csx - ri, csy - ri, csx + ri, csy + ri], fill=layer_color)

        # ===== Beat Flash Overlay =====
        if self.beat_strength > 0.5:
            flash_intensity = (self.beat_strength - 0.5) * 2
            for y in range(self.height):
                val = int(flash_intensity * 15)
                draw.rectangle([0, y, self.width, y + 1], fill=(val // 2, val // 3, val))


# =============================================================================
# PARALLAX STARFIELD BACKGROUND - NASA DEEP SPACE REALISTIC
# =============================================================================

class ParallaxStarfield:
    """
    Realistic NASA-style deep space background.

    Features:
    - True black void with subtle gradient
    - Realistic star colors (spectral classes O B A F G K M)
    - Proper magnitude distribution (many dim, few bright)
    - Subtle dust lanes (dark regions)
    - Very minimal, non-distracting
    """

    # Realistic stellar colors by spectral class (Kelvin -> RGB)
    SPECTRAL_COLORS = [
        (155, 176, 255),  # O - Blue
        (170, 191, 255),  # B - Blue-white
        (202, 215, 255),  # A - White
        (248, 247, 255),  # F - Yellow-white
        (255, 244, 234),  # G - Yellow (Sun-like)
        (255, 210, 161),  # K - Orange
        (255, 204, 111),  # M - Red-orange
    ]

    def __init__(self, width: int, height: int, quality: QualityPreset):
        self.width = width
        self.height = height
        self.num_layers = QUALITY_SETTINGS[quality]['parallax_layers']

        # More stars for realism
        base_count = QUALITY_SETTINGS[quality]['bg_star_count']

        rng = np.random.RandomState(7749)  # Seed for reproducibility

        # Generate realistic star field
        self.layers = []

        for layer_idx in range(self.num_layers):
            stars = []
            depth = layer_idx / max(1, self.num_layers - 1)

            # Far layers have more, dimmer stars
            count = int(base_count * (1.5 - depth * 0.5))

            for _ in range(count):
                x = rng.uniform(0, width)
                y = rng.uniform(0, height)

                # Magnitude distribution: exponentially more dim stars
                # magnitude 0-6 scale (0=bright, 6=barely visible)
                magnitude = rng.exponential(2.5)
                magnitude = min(6, magnitude)

                # Brightness from magnitude (logarithmic)
                brightness = 0.02 + 0.15 * (1 - magnitude / 6) ** 2.5

                # Size: most stars are point-like, few are resolvable
                if magnitude < 1:
                    size = 1.5 + rng.uniform(0, 0.5)
                elif magnitude < 2:
                    size = 1.0 + rng.uniform(0, 0.3)
                elif magnitude < 3.5:
                    size = 0.7
                else:
                    size = 0  # Point source

                # Spectral class distribution (realistic)
                # Most stars are K and M (red/orange), few are O and B (blue)
                spectral_roll = rng.random()
                if spectral_roll < 0.003:
                    spectral = 0  # O - very rare
                elif spectral_roll < 0.01:
                    spectral = 1  # B - rare
                elif spectral_roll < 0.03:
                    spectral = 2  # A
                elif spectral_roll < 0.08:
                    spectral = 3  # F
                elif spectral_roll < 0.20:
                    spectral = 4  # G
                elif spectral_roll < 0.45:
                    spectral = 5  # K
                else:
                    spectral = 6  # M - most common

                # Very subtle twinkle (atmospheric scintillation)
                twinkle_phase = rng.uniform(0, 2 * np.pi)
                twinkle_speed = rng.uniform(0.5, 2.0)  # Slower, subtler
                twinkle_amp = 0.05 + rng.uniform(0, 0.1)  # Very subtle

                stars.append({
                    'x': x, 'y': y,
                    'size': size,
                    'brightness': brightness * (0.3 + depth * 0.7),
                    'spectral': spectral,
                    'twinkle_phase': twinkle_phase,
                    'twinkle_speed': twinkle_speed,
                    'twinkle_amp': twinkle_amp,
                    'depth': depth,
                })

            self.layers.append(stars)

        # Generate subtle dust lanes (dark regions)
        self.dust_lanes = []
        for _ in range(3):
            self.dust_lanes.append({
                'x': rng.uniform(0, width),
                'y': rng.uniform(0, height),
                'angle': rng.uniform(0, np.pi),
                'width': rng.uniform(100, 300),
                'length': rng.uniform(400, 800),
                'opacity': rng.uniform(0.1, 0.25),
            })

        # Nebula clouds (audio-reactive)
        self.nebula_clouds = []
        for _ in range(6):  # 6 nebula regions (increased from 4)
            self.nebula_clouds.append({
                'x': rng.uniform(0, width),
                'y': rng.uniform(0, height),
                'radius': rng.uniform(200, 500),
                'color_bias': rng.choice([0, 1, 2, 3, 4]),  # More frequency bands
                'intensity': rng.uniform(0.15, 0.35),  # Increased for HDR
                'rotation': rng.uniform(0, 2 * np.pi),
            })

        # Initialize audio reactivity
        self.band_energies = [0.5] * 10  # Initialize with neutral values
        self.beat_strength = 0.0

    def update(self, audio_frame, band_energies):
        """Audio-reactive background updates."""
        # Store band energies for rendering
        self.band_energies = band_energies

        # Store beat strength for audio-reactive effects
        if hasattr(audio_frame, 'beat_strength'):
            self.beat_strength = audio_frame.beat_strength
        else:
            self.beat_strength = 0.0

    def render(self, draw, time_offset: float, viewport_offset: tuple):
        """Render realistic deep space background with HDR-like brightness."""

        # Much brighter blue-purple base (HDR effect)
        draw.rectangle([0, 0, self.width, self.height], fill=(20, 15, 35))

        # Strong gradient (simulates Milky Way glow on horizon) - much brighter
        for y in range(self.height - 150, self.height):
            t = (y - (self.height - 150)) / 150
            # Stronger warm glow at bottom for HDR effect
            val = int(t * 25)
            if val > 0:
                draw.rectangle([0, y, self.width, y + 1], fill=(val, val // 2, int(val * 0.6)))

        # Render nebula clouds (audio-reactive)
        self._render_nebula_clouds(draw)

        # Audio-reactive gradient enhancement
        if self.beat_strength > 0.5:
            # Flash subtle color on strong beats
            beat_color = int((self.beat_strength - 0.5) * 2 * 8)  # 0-8
            for y in range(self.height - 100, self.height):
                t = (y - (self.height - 100)) / 100
                val = int(t * beat_color)
                if val > 0:
                    draw.rectangle([0, y, self.width, y + 1],
                                  fill=(val, val // 2, val // 3))  # Warm beat flash

        # Render dust lanes (dark patches that obscure stars)
        # These are rendered by darkening regions, applied via star dimming

        # Render star layers with parallax
        for layer_idx, stars in enumerate(self.layers):
            depth = layer_idx / max(1, self.num_layers - 1)

            # Parallax: far layers move slower
            parallax = 0.05 + depth * 0.15  # Reduced parallax for realism
            ox = viewport_offset[0] * parallax * 30
            oy = viewport_offset[1] * parallax * 30

            for star in stars:
                sx = (star['x'] + ox) % self.width
                sy = (star['y'] + oy) % self.height

                # Check if in dust lane (dim the star)
                dust_factor = 1.0
                for dust in self.dust_lanes:
                    dx = sx - dust['x']
                    dy = sy - dust['y']
                    # Rotate to dust lane frame
                    cos_a, sin_a = np.cos(dust['angle']), np.sin(dust['angle'])
                    rx = dx * cos_a + dy * sin_a
                    ry = -dx * sin_a + dy * cos_a
                    # Check if inside ellipse
                    if abs(rx) < dust['length'] / 2 and abs(ry) < dust['width'] / 2:
                        dist = np.sqrt((rx / (dust['length']/2))**2 + (ry / (dust['width']/2))**2)
                        if dist < 1:
                            dust_factor *= (1 - dust['opacity'] * (1 - dist))

                # Very subtle twinkle
                twinkle = 1.0 - star['twinkle_amp'] * (1 - np.sin(time_offset * star['twinkle_speed'] + star['twinkle_phase']))

                # Final brightness
                brightness = star['brightness'] * twinkle * dust_factor

                if brightness < 0.01:
                    continue

                # Get spectral color
                base_color = self.SPECTRAL_COLORS[star['spectral']]
                r = int(base_color[0] * brightness)
                g = int(base_color[1] * brightness)
                b = int(base_color[2] * brightness)

                # Clamp
                r, g, b = min(255, r), min(255, g), min(255, b)

                if r == 0 and g == 0 and b == 0:
                    continue

                # Draw star
                size = star['size']
                if size <= 0:
                    # Point source
                    draw.point((int(sx), int(sy)), fill=(r, g, b))
                else:
                    # Small disk for brighter stars
                    ri = max(1, int(size))
                    # Draw with slight glow for brightest
                    if brightness > 0.08:
                        # Subtle glow
                        glow_r = ri + 1
                        glow_color = (r // 4, g // 4, b // 4)
                        draw.ellipse([sx - glow_r, sy - glow_r, sx + glow_r, sy + glow_r], fill=glow_color)
                    draw.ellipse([sx - ri, sy - ri, sx + ri, sy + ri], fill=(r, g, b))

    def _render_nebula_clouds(self, draw):
        """Render audio-reactive nebula clouds."""
        for cloud in self.nebula_clouds:
            # Get audio energy for this cloud's frequency band
            energy = self.band_energies[cloud['color_bias']]

            # Only render if energy above threshold (subtle appearance)
            if energy < 0.15:  # Lower threshold for better visibility
                continue

            # Cloud intensity modulates with audio (MUCH BRIGHTER for HDR)
            intensity = cloud['intensity'] * energy * 1.2  # Increased from 0.8

            # Choose color based on frequency band (BRIGHTER COLORS)
            if cloud['color_bias'] == 0:  # Sub-bass → Red
                color = (int(150 * intensity), int(30 * intensity), int(50 * intensity))
            elif cloud['color_bias'] == 1:  # Bass → Orange
                color = (int(150 * intensity), int(80 * intensity), int(30 * intensity))
            elif cloud['color_bias'] == 2:  # Low-mid → Purple
                color = (int(90 * intensity), int(30 * intensity), int(150 * intensity))
            elif cloud['color_bias'] == 3:  # Mid → Pink
                color = (int(180 * intensity), int(50 * intensity), int(120 * intensity))
            else:  # High-mid → Cyan
                color = (int(50 * intensity), int(180 * intensity), int(180 * intensity))

            # Draw cloud as soft ellipse
            cx, cy = cloud['x'], cloud['y']
            radius = cloud['radius'] * (0.8 + energy * 0.4)  # Pulses with energy

            # Multiple layers for soft gradient
            for i in range(5):
                r = radius * (1 - i * 0.15)
                alpha_scale = (5 - i) / 5
                layer_color = tuple(int(c * alpha_scale) for c in color)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=layer_color)


# =============================================================================
# FREQUENCY STAR RENDERER
# =============================================================================

class FrequencyStar:
    """A star that pulses with its assigned frequency band."""

    def __init__(self, star_id: int, band_id: int, quality: QualityPreset):
        self.id = star_id
        self.band_id = band_id
        self.quality = quality
        self.glow_layers = QUALITY_SETTINGS[quality]['glow_layers']

        # Base color from band
        self.base_color = BAND_COLORS[band_id]

        # Smoothed energy for this band (avoids jitter)
        self.smoothed_energy = 0.0
        self.energy_attack = 0.3   # Fast attack
        self.energy_release = 0.08  # Slow release

        # Pulse state
        self.pulse_phase = 0.0
        self.last_beat_time = 0.0

    def update(self, band_energy: float, beat_strength: float, dt: float):
        """Update star state based on its frequency band energy."""
        # Smooth the energy with asymmetric attack/release
        if band_energy > self.smoothed_energy:
            self.smoothed_energy += self.energy_attack * (band_energy - self.smoothed_energy)
        else:
            self.smoothed_energy += self.energy_release * (band_energy - self.smoothed_energy)

        # Pulse phase advances based on energy
        self.pulse_phase += dt * (2 + self.smoothed_energy * 4)

    def get_render_params(self, base_radius: float):
        """Get rendering parameters based on current state."""
        energy = self.smoothed_energy

        # Size pulses smoothly with energy only (no beat spike) - half as reactive
        size_mult = 1.0 + energy * 0.25  # 0-25% smooth scaling (reduced from 50%)
        radius = base_radius * size_mult

        # Glow intensity smooth response
        glow_intensity = 0.8 + energy * 3.0  # Smooth, elegant

        # Brightness smooth and stable
        brightness = 0.5 + energy * 0.5  # 0.5-1.0 range
        brightness = min(1.0, brightness)

        # Subtle hue shift when very active
        hue_shift = energy * 0.1

        return {
            'radius': radius,
            'glow_intensity': glow_intensity,
            'brightness': brightness,
            'hue_shift': hue_shift,
            'energy': energy,
        }


class StellarRenderer:
    """Renders frequency-reactive stars."""

    def __init__(self, width: int, height: int, n_bodies: int, quality: QualityPreset):
        self.width = width
        self.height = height
        self.quality = quality
        self.glow_layers = QUALITY_SETTINGS[quality]['glow_layers']

        # Create frequency stars - distribute bodies across bands
        self.stars = {}
        for i in range(n_bodies):
            band_id = i % 10  # Cycle through 10 bands
            self.stars[i] = FrequencyStar(i, band_id, quality)

        # Trail history
        self.trails = [[] for _ in range(n_bodies)]
        self.max_trail_length = 400  # Longer trails for dramatic effect

        # Viewport
        self.vp_x_min, self.vp_x_max = -2.5, 2.5
        self.vp_y_min, self.vp_y_max = -1.4, 1.4
        self.vp_attack = 0.15
        self.vp_release = 0.05

        # Orbital camera state
        self.orbit_enabled = True
        self.orbit_radius = 2.0      # Distance from COM
        self.orbit_speed = 0.15      # rad/s (~42s per orbit)
        self.orbit_angle = 0.0       # Current angle

        # Smoothed tracking state
        self.com_x = 0.0
        self.com_y = 0.0
        self.view_width = 5.0
        self.view_height = 2.8

    def update_viewport(self, positions: np.ndarray, dt: float):
        """Orbital camera system with smooth tracking."""
        if len(positions) == 0:
            return

        # 1. Calculate target COM
        target_com_x = positions[:, 0].mean()
        target_com_y = positions[:, 1].mean()

        # 2. Smooth COM tracking (prevents jitter)
        self.com_x += (target_com_x - self.com_x) * 0.1
        self.com_y += (target_com_y - self.com_y) * 0.1

        # 3. Calculate orbital offset
        if self.orbit_enabled:
            self.orbit_angle += self.orbit_speed * dt
            orbit_offset_x = self.orbit_radius * np.cos(self.orbit_angle)
            orbit_offset_y = self.orbit_radius * np.sin(self.orbit_angle)
        else:
            orbit_offset_x = 0.0
            orbit_offset_y = 0.0

        # 4. Camera position = COM + orbital offset
        camera_x = self.com_x + orbit_offset_x
        camera_y = self.com_y + orbit_offset_y

        # 5. Calculate zoom (18% closer than before)
        dist = np.sqrt((positions[:, 0] - self.com_x)**2 +
                       (positions[:, 1] - self.com_y)**2)
        max_dist = np.percentile(dist, 95)

        target_view_width = max(2.5, max_dist * 1.8)  # Was: max(3.0, max_dist * 2.2)
        self.view_width += (target_view_width - self.view_width) * 0.08
        self.view_height = self.view_width * 9/16

        # 6. Set viewport bounds
        self.vp_x_min = camera_x - self.view_width / 2
        self.vp_x_max = camera_x + self.view_width / 2
        self.vp_y_min = camera_y - self.view_height / 2
        self.vp_y_max = camera_y + self.view_height / 2

    def world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world to screen coords."""
        sx = int((x - self.vp_x_min) / (self.vp_x_max - self.vp_x_min) * self.width)
        sy = int((self.vp_y_max - y) / (self.vp_y_max - self.vp_y_min) * self.height)
        return sx, sy

    def get_viewport_offset(self) -> tuple:
        """Get viewport center for parallax."""
        cx = (self.vp_x_min + self.vp_x_max) / 2
        cy = (self.vp_y_min + self.vp_y_max) / 2
        return (cx, cy)

    def update_stars(self, band_energies: list, beat_strength: float, dt: float):
        """Update all star states."""
        for star_id, star in self.stars.items():
            band_energy = band_energies[star.band_id]
            star.update(band_energy, beat_strength, dt)

    def update_trails(self, positions: np.ndarray):
        """Update trail history."""
        for i in range(min(len(positions), len(self.trails))):
            self.trails[i].append(positions[i][:2].copy())
            if len(self.trails[i]) > self.max_trail_length:
                self.trails[i].pop(0)

    def render_trails(self, draw, band_energies: list):
        """Render orbital trails - each star draws its frequency band's response."""
        for i, trail in enumerate(self.trails):
            if len(trail) < 2:
                continue

            star = self.stars.get(i)
            if not star:
                continue

            # FREQUENCY-BAND VISIBILITY: Only show trail when THIS band is active
            # Each star "draws" when its frequency band pulses
            band_energy = band_energies[star.band_id]

            # Only render trail when band energy crosses threshold (music happening in this frequency)
            VISIBILITY_THRESHOLD = 0.4  # Trails only visible when frequency band is strong (40%+)
            if band_energy < VISIBILITY_THRESHOLD:
                continue  # No trail - frequency band is quiet

            # Trail width and intensity scale with BAND ENERGY
            # Strong bass = thick red trail, strong treble = thick blue trail, etc.
            n = len(trail)

            # Trail width scales with band energy (1-5 pixels)
            trail_width = int(1 + band_energy * 4)

            # Get the star's color (matches its frequency band)
            base_color = star.base_color

            for j in range(1, n, 2):
                # Progress along trail (newer segments = brighter)
                trail_progress = j / n  # 0 (old) to 1 (new)

                # Opacity scales with BAND ENERGY (the frequency response)
                # High energy = bright trail, low energy = faint trail
                energy_opacity = (band_energy - VISIBILITY_THRESHOLD) / (1.0 - VISIBILITY_THRESHOLD)  # Normalize threshold to 1.0
                energy_opacity = max(0.0, min(1.0, energy_opacity))  # Clamp 0-1
                # Slower fade: older segments stay visible longer
                position_opacity = 0.3 + trail_progress * 0.7  # Minimum 30% opacity, fade over full length

                alpha = 0.3 + energy_opacity * 0.7  # Always visible (min 30%) + energy boost (0-70%)

                x1, y1 = self.world_to_screen(trail[j-1][0], trail[j-1][1])
                x2, y2 = self.world_to_screen(trail[j][0], trail[j][1])

                # Trail color = band color, brightness = band energy
                color = tuple(min(255, int(c * alpha)) for c in base_color)

                # Draw glow when band energy is strong - less sensitive for subtlety
                if band_energy > 0.6:  # Higher threshold (only very strong frequencies)
                    glow_width = trail_width + 1  # Subtle glow width
                    # Less intense glow
                    glow_intensity = (band_energy - 0.6) / 0.4  # 0-1.0 for 0.6-1.0 energy
                    glow_color = tuple(int(c * alpha * glow_intensity * 0.4) for c in base_color)  # More subtle
                    draw.line([(x1, y1), (x2, y2)], fill=glow_color, width=glow_width)

                # Main trail line - bright when band is active
                draw.line([(x1, y1), (x2, y2)], fill=color, width=trail_width)

    def render_star(self, draw, sx: int, sy: int, star: FrequencyStar, base_radius: float):
        """Render a single frequency star with HDR glow and improved lighting."""
        params = star.get_render_params(base_radius)
        radius = params['radius']
        glow_intensity = params['glow_intensity']
        brightness = params['brightness']
        energy = params['energy']

        # Base color with brightness
        r, g, b = star.base_color
        color = (int(r * brightness), int(g * brightness), int(b * brightness))

        # HDR BLOOM: Extra wide outer glow for bloom effect
        bloom_layers = max(5, int(self.glow_layers * 0.3))
        for i in range(bloom_layers):
            t = i / max(1, bloom_layers - 1)

            # Wide bloom glow
            bloom_r = radius * (5.0 - 2.0 * t) * glow_intensity * 0.6

            # Soft alpha for bloom
            alpha = int(20 + 40 * (1 - t ** 2) * glow_intensity * 0.5)
            alpha = min(100, alpha)

            bloom_color = (
                int(r * 0.3 * alpha / 255),
                int(g * 0.3 * alpha / 255),
                int(b * 0.3 * alpha / 255)
            )

            ri = int(bloom_r)
            if ri > 1:
                draw.ellipse([sx - ri, sy - ri, sx + ri, sy + ri], fill=bloom_color)

        # Draw main glow layers (outside to inside)
        for i in range(self.glow_layers):
            t = i / max(1, self.glow_layers - 1)

            # Radius: outer to inner
            layer_r = radius * (3.0 - 2.0 * t) * glow_intensity

            # Alpha: exponential falloff (BRIGHTER)
            alpha = int(15 + 80 * (t ** 1.5) * glow_intensity)
            alpha = min(255, alpha)

            # Color with alpha (MORE SATURATED)
            glow_color = (
                int(r * brightness * alpha / 200),
                int(g * brightness * alpha / 200),
                int(b * brightness * alpha / 200)
            )

            ri = int(layer_r)
            if ri > 0:
                draw.ellipse([sx - ri, sy - ri, sx + ri, sy + ri], fill=glow_color)

        # Bright core (MUCH BRIGHTER)
        core_r = int(radius * 0.5)
        core_color = (
            min(255, int(r * brightness * 1.8)),
            min(255, int(g * brightness * 1.8)),
            min(255, int(b * brightness * 1.8))
        )
        if core_r > 0:
            draw.ellipse([sx - core_r, sy - core_r, sx + core_r, sy + core_r],
                        fill=core_color)

        # Hot white center for high energy (BIGGER AND BRIGHTER)
        if energy > 0.5:
            center_r = max(1, int(radius * 0.2 * energy))
            white_alpha = int((energy - 0.5) / 0.5 * 255)
            white_alpha = min(255, white_alpha)
            center_color = (white_alpha, white_alpha, white_alpha)
            draw.ellipse([sx - center_r, sy - center_r, sx + center_r, sy + center_r],
                        fill=center_color)


# =============================================================================
# SPECTRUM DISPLAY
# =============================================================================

class SpectrumDisplay:
    """10-band frequency spectrum visualization."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Display area (bottom of screen)
        self.bar_width = 30
        self.bar_spacing = 10
        self.max_bar_height = 120
        self.margin_bottom = 30

        # Calculate total width and center it
        total_width = 10 * self.bar_width + 9 * self.bar_spacing
        self.start_x = (width - total_width) // 2

        # Smoothed values for display
        self.display_values = [0.0] * 10
        self.peak_values = [0.0] * 10
        self.peak_hold = [0] * 10

    def update(self, band_energies: list):
        """Update display values with smoothing."""
        for i in range(10):
            # Smooth the display value
            target = band_energies[i]
            if target > self.display_values[i]:
                self.display_values[i] += 0.4 * (target - self.display_values[i])
            else:
                self.display_values[i] += 0.1 * (target - self.display_values[i])

            # Peak hold
            if self.display_values[i] > self.peak_values[i]:
                self.peak_values[i] = self.display_values[i]
                self.peak_hold[i] = 30  # Hold for 30 frames
            else:
                self.peak_hold[i] -= 1
                if self.peak_hold[i] <= 0:
                    self.peak_values[i] *= 0.95  # Decay peak

    def render(self, draw):
        """Render spectrum bars."""
        y_base = self.height - self.margin_bottom

        for i in range(10):
            x = self.start_x + i * (self.bar_width + self.bar_spacing)

            # Bar height from energy
            height = int(self.display_values[i] * self.max_bar_height)
            height = min(height, self.max_bar_height)

            # Get band color
            color = BAND_COLORS[i]

            # Draw bar with gradient effect
            if height > 0:
                # Main bar
                for y in range(height):
                    # Gradient: brighter at top
                    t = y / max(1, height)
                    brightness = 0.4 + 0.6 * t
                    row_color = tuple(int(c * brightness) for c in color)

                    draw.rectangle([
                        x, y_base - y - 1,
                        x + self.bar_width, y_base - y
                    ], fill=row_color)

                # Glow at top
                glow_y = y_base - height
                glow_color = tuple(int(c * 0.3) for c in color)
                draw.ellipse([
                    x - 5, glow_y - 10,
                    x + self.bar_width + 5, glow_y + 10
                ], fill=glow_color)

            # Peak indicator
            peak_height = int(self.peak_values[i] * self.max_bar_height)
            if peak_height > 2:
                peak_y = y_base - peak_height
                draw.rectangle([x, peak_y - 2, x + self.bar_width, peak_y],
                              fill=color)

            # Band label
            label = BAND_NAMES[i]
            label_x = x + self.bar_width // 2 - 10
            draw.text((label_x, y_base + 5), label, fill=(100, 100, 100))


# =============================================================================
# MAIN RENDERER
# =============================================================================

class CinematicRenderer:
    """Main stellar frequency visualizer."""

    def __init__(self, width: int, height: int, n_bodies: int,
                 quality: QualityPreset, fps: float):
        self.width = width
        self.height = height
        self.quality = quality
        self.fps = fps
        self.n_bodies = n_bodies

        # Subsystems
        self.background = SphericalCosmos(width, height, quality)  # NEW: 360° realistic space
        self.stellar_renderer = StellarRenderer(width, height, n_bodies, quality)
        self.spectrum = SpectrumDisplay(width, height)

        # Time tracking
        self.time = 0.0

        # Base star radius (small!)
        self.base_radius = 4.0

    def render_frame(self, sim, audio_frame, audio_10band, frame_num: int) -> bytes:
        """Render a complete frame."""
        dt = 1.0 / self.fps
        self.time += dt

        positions = sim.get_positions()

        # Extract 10-band energies
        band_energies = [
            getattr(audio_10band, 'sub_bass', 0.5),
            getattr(audio_10band, 'bass', 0.5),
            getattr(audio_10band, 'low_mid', 0.5),
            getattr(audio_10band, 'mid', 0.5),
            getattr(audio_10band, 'high_mid', 0.5),
            getattr(audio_10band, 'presence', 0.5),
            getattr(audio_10band, 'brilliance', 0.5),
            getattr(audio_10band, 'air', 0.5),
            getattr(audio_10band, 'ultra', 0.5),
            getattr(audio_10band, 'extreme', 0.5),
        ]

        beat = audio_frame.beat_strength if hasattr(audio_frame, 'beat_strength') else 0

        # Update systems
        self.stellar_renderer.update_viewport(positions, dt)
        self.stellar_renderer.update_stars(band_energies, beat, dt)
        self.stellar_renderer.update_trails(positions)
        self.background.update(audio_frame, band_energies)
        self.spectrum.update(band_energies)

        # Create image
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 1. Background (360° Spherical Cosmos)
        self.background.render(draw, self.time, None)  # Spherical cosmos doesn't need viewport offset

        # 2. Trails (NOW AUDIO-REACTIVE)
        self.stellar_renderer.render_trails(draw, band_energies)

        # 3. Stars (sorted by Z for depth)
        z_order = np.argsort(-positions[:, 2])
        for i in z_order:
            x, y, z = positions[i]
            sx, sy = self.stellar_renderer.world_to_screen(x, y)

            star = self.stellar_renderer.stars.get(i)
            if star:
                self.stellar_renderer.render_star(draw, sx, sy, star, self.base_radius)

        # 4. Spectrum display
        self.spectrum.render(draw)

        # 5. Minimal UI
        t = audio_frame.time if hasattr(audio_frame, 'time') else frame_num / self.fps
        time_str = f'{int(t//60)}:{int(t%60):02d}'
        draw.text((30, 30), time_str, fill=(150, 150, 150))

        # Optional blur
        if QUALITY_SETTINGS[self.quality]['blur_passes'] > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        return img.tobytes()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stellar Frequency Visualizer')
    parser.add_argument('--audio', '-a', default='assets/audio/still-night.mp3')
    parser.add_argument('--output', '-o', default='output/stellar.mp4')
    parser.add_argument('--bodies', '-n', type=int, default=50)
    parser.add_argument('--resolution', '-r', default='1080p',
                       choices=['720p', '1080p', '1440p', '4k'])
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--quality', '-q', default='good',
                       choices=['draft', 'good', 'best'])
    parser.add_argument('--duration', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    # Resolution
    res_map = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '4k': (3840, 2160)
    }
    width, height = res_map[args.resolution]

    quality = QualityPreset(args.quality)
    settings = QUALITY_SETTINGS[quality]

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║       STELLAR FREQUENCY VISUALIZER                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  10-Band Audio Reactive Star Field                           ║
    ║  Output:     {args.output:<47} ║
    ║  Resolution: {args.resolution} ({width}x{height})                              ║
    ║  Stars: {args.bodies}  FPS: {args.fps}  Quality: {args.quality}                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Audio
    print("Loading audio...")
    audio = AudioAnalyzer(args.audio, fps=args.fps)
    audio_10band = AudioAnalyzer10Band(args.audio, fps=args.fps)

    total_frames = audio.total_frames
    if args.duration:
        total_frames = min(total_frames, int(args.duration * args.fps))

    print(f"  Duration: {audio.duration:.1f}s ({total_frames} frames)")

    # Physics - STRONGER GRAVITY, TIGHTER ORBITS
    import warp as wp
    wp.init()
    device = "cuda:0" if (not args.cpu and wp.is_cuda_available()) else "cpu"
    print(f"Physics device: {device}")

    config = SimulationConfig(
        n_bodies=args.bodies,
        G=0.6,              # Reduced gravity for more chaotic, looser orbits
        softening=0.02,     # Hybrid version softening (finer detail)
        dt=0.0002,          # Hybrid version timestep (more precise)
        trail_length=0,
        device=device
    )
    sim = NBodySimulation(config)

    # Initialize with original distribution
    sim.initialize_random(pos_range=1.8, vel_range=0.3, seed=42)  # Original spacing

    # Add stronger orbital velocities
    positions = sim.get_positions()
    velocities = sim.get_velocities()
    masses = sim.masses.numpy()

    # More uniform masses (all stars, similar size)
    rng = np.random.RandomState(42)
    for i in range(args.bodies):
        masses[i] = rng.uniform(0.8, 1.2)  # Uniform mass range

    center = positions.mean(axis=0)
    total_mass = masses.sum()

    for i in range(args.bodies):
        r_vec = positions[i] - center
        r = np.linalg.norm(r_vec)
        if r > 0.01:
            # Stronger orbital velocity for tighter orbits
            v_orbital = np.sqrt(config.G * total_mass * 0.5 / (r + 0.05))
            tangent = np.array([-r_vec[1], r_vec[0], 0.0])
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 0.01:
                tangent = tangent / tangent_norm
                velocities[i] = tangent * v_orbital  # Replace velocity entirely

    sim.set_state(positions, velocities, masses)
    print("  Configured for tight stellar orbits")

    # Renderer
    print("Initializing stellar renderer...")
    renderer = CinematicRenderer(width, height, args.bodies, quality, args.fps)

    # Output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(args.fps),
        '-i', 'pipe:0',
        '-i', args.audio,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', settings['ffmpeg_preset'],
        '-crf', settings['ffmpeg_crf'], '-b:v', '8M',
        '-c:a', 'aac', '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-shortest',
        str(output_path)
    ]

    print("Starting render...")
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    steps_per_frame = 20  # More physics steps for tighter simulation
    start_time = time.time()

    try:
        for frame_idx in range(total_frames):
            # Audio lead time: 100ms ahead (3 frames @ 30fps) so visuals align with perceived audio
            audio_lead_frames = 3
            audio_idx = min(frame_idx + audio_lead_frames, total_frames - 1)
            audio_frame = audio.get_frame(audio_idx)
            audio_10band_frame = audio_10band.get_frame(audio_idx)

            # Audio-modulated physics
            audio_params = {
                'bass_energy': audio_frame.bass_energy,
                'mid_energy': audio_frame.mid_energy,
                'treble_energy': audio_frame.treble_energy,
                'modulation_depth': 1.0 + audio_frame.beat_strength * 0.5
            }

            for _ in range(steps_per_frame):
                sim.step(audio_params=audio_params)

            frame_data = renderer.render_frame(sim, audio_frame, audio_10band_frame, frame_idx)

            try:
                ffmpeg.stdin.write(frame_data)
            except BrokenPipeError:
                print("\nFFmpeg error!")
                break

            if (frame_idx + 1) % int(args.fps) == 0:
                elapsed = time.time() - start_time
                fps_actual = (frame_idx + 1) / elapsed
                pct = (frame_idx + 1) / total_frames * 100
                eta = (total_frames - frame_idx - 1) / fps_actual if fps_actual > 0 else 0

                print(f"\r  {pct:5.1f}% | {frame_idx+1}/{total_frames} | "
                      f"{fps_actual:.1f} fps | ETA: {int(eta)}s    ",
                      end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()

    total_time = time.time() - start_time
    print(f"\n\n✓ Render complete: {output_path}")
    print(f"  Time: {total_time:.1f}s ({total_frames/total_time:.1f} fps)")
    print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")


if __name__ == '__main__':
    main()
