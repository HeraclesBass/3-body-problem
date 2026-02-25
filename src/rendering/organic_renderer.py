"""
Organic Renderer - Sophisticated Audio-Reactive Visuals

Design Philosophy:
- Smooth, flowing, organic (not flashy)
- Billions of color possibilities from physics
- Deep music integration (all 10 bands + harmonics)
- Long flowing trails (2000+ points)
- Layered complexity (background, trails, particles, effects)
- Subtle, beautiful, hypnotic

Visual Techniques:
- Multi-dimensional color mapping (velocity, acceleration, energy, audio)
- Flowing trail ribbons with color gradients
- Particle interaction glow fields
- Evolving background nebula
- Harmonic resonance visualization
- Gravitational lens distortion
- Color bleeding and bloom
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
try:
    from scipy.interpolate import splprep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from rendering.advanced_color_system import AdvancedColorSystem
from colorsys import hsv_to_rgb, rgb_to_hsv


class OrganicRenderer:
    """
    Sophisticated organic renderer for bioluminescent aesthetic.

    Goes crazy with:
    - Infinite color possibilities
    - Long flowing trails (2000 points)
    - Multi-layer backgrounds
    - Particle interaction effects
    - Audio-driven everything
    """

    def __init__(self, width=1920, height=1080, dpi=100):
        self.width = width
        self.height = height
        self.dpi = dpi

        # Create figure with smooth rendering
        fig_w, fig_h = width / dpi, height / dpi
        self.fig, self.ax = plt.subplots(
            figsize=(fig_w, fig_h),
            dpi=dpi,
            facecolor='#000205'
        )
        self.ax.set_facecolor('#000205')

        # Initialize advanced color system
        self.color_system = AdvancedColorSystem()

        # Background evolution state
        self.bg_phase = 0
        self.nebula_seed = np.random.RandomState(42)

    def _render_layered_background(self, audio_frame, time):
        """
        Multi-layer evolving background nebula.

        Layers:
        1. Deep space gradient (slowest evolution)
        2. Nebula clouds (mid evolution, bass-reactive)
        3. Particle field (fast shimmer, treble-reactive)
        4. Harmonic glow fields (audio frequency-driven)
        """
        # Layer 1: Deep space gradient (subtle)
        gradient_strength = 0.05 + audio_frame.ultra * 0.03

        for radius in [4, 3, 2]:
            color = self.color_system.get_background_color_field(
                0, 0, time, audio_frame
            )
            circle = Circle(
                (0, 0), radius,
                color=color,
                alpha=gradient_strength,
                zorder=0
            )
            self.ax.add_patch(circle)

        # Layer 2: Nebula clouds (bass-reactive, organic shapes)
        bass_total = (audio_frame.sub_bass + audio_frame.bass) / 2

        if bass_total > 0.2:
            # Multiple nebula clouds at different positions
            n_clouds = 3
            self.bg_phase += bass_total * 0.02

            for i in range(n_clouds):
                angle = (i / n_clouds) * 2 * np.pi + self.bg_phase
                radius_base = 1.5 + i * 0.5
                radius = radius_base * (1 + bass_total * 0.6)

                cloud_x = np.cos(angle) * 1.2
                cloud_y = np.sin(angle) * 0.7

                # Organic color based on position + audio
                hue = ((angle * 57.3 + audio_frame.low_mid * 60) % 360) / 360.0
                sat = 0.4 + audio_frame.mid * 0.3
                val = 0.2 + bass_total * 0.2
                color = hsv_to_rgb(hue, sat, val)

                # Multiple layers for soft edges
                for r_mult, alpha in [(1.5, 0.02), (1.0, 0.04), (0.7, 0.06)]:
                    circle = Circle(
                        (cloud_x, cloud_y),
                        radius * r_mult,
                        color=color,
                        alpha=min(alpha * bass_total, 0.08),
                        zorder=1
                    )
                    self.ax.add_patch(circle)

        # Layer 3: Star field (extreme frequencies)
        if audio_frame.extreme > 0.3:
            n_stars = int(30 * audio_frame.extreme)
            x = self.nebula_seed.uniform(-3, 3, n_stars)
            y = self.nebula_seed.uniform(-1.7, 1.7, n_stars)

            # Stars twinkle with audio
            sizes = self.nebula_seed.uniform(0.5, 2.5, n_stars) * audio_frame.extreme
            alpha = 0.2 + audio_frame.air * 0.3

            self.ax.scatter(
                x, y, s=sizes,
                c='white',
                alpha=min(alpha, 0.5),
                zorder=1
            )

        # Layer 4: Harmonic glow fields (presence + high-mid)
        if audio_frame.presence > 0.4:
            # Create subtle glow fields at harmonic positions
            for i, shift in enumerate([0, 120, 240]):  # Triadic harmony
                angle = (shift + time * 10) % 360
                rad = np.radians(angle)
                x = np.cos(rad) * 2.0
                y = np.sin(rad) * 1.2

                hue = (angle + audio_frame.high_mid * 60) / 360.0
                color = hsv_to_rgb(hue, 0.6, 0.3)

                circle = Circle(
                    (x, y),
                    0.8 * (1 + audio_frame.presence * 0.5),
                    color=color,
                    alpha=min(audio_frame.presence * 0.06, 0.08),
                    zorder=1
                )
                self.ax.add_patch(circle)

    def _render_flowing_trail(self, trail_positions, trail_velocities,
                            particle_color_rgb, audio_frame):
        """
        Render long, flowing trail with smooth color evolution.

        Techniques:
        - Spline interpolation for smoothness
        - Color gradient along length
        - Width variation from velocity
        - Audio-reactive transparency
        """
        n_points = len(trail_positions)
        if n_points < 4:
            return

        # Take longer trails (up to 1500 points for flowing effect)
        trail = trail_positions[-1500:]
        n = len(trail)

        # Smooth the trail with spline (organic curves)
        if HAS_SCIPY and n > 10:
            try:
                # Spline smoothing (reduces jitter)
                tck, u = splprep([trail[:, 0], trail[:, 1]], s=0.5, k=min(3, n-1))
                u_smooth = np.linspace(0, 1, n * 2)  # Upsample for smoothness
                smooth_trail = np.array(splev(u_smooth, tck)).T
            except:
                smooth_trail = trail
        else:
            smooth_trail = trail

        # Color gradient along trail
        n_smooth = len(smooth_trail)
        colors = []
        alphas = []
        widths = []

        for i in range(n_smooth):
            # Age-based fade (older = more transparent)
            age_factor = i / n_smooth

            # Get velocity at this point (approximate)
            vel_idx = min(int(i / len(smooth_trail) * n), n-1)
            if vel_idx < len(trail_velocities):
                vel = trail_velocities[vel_idx]
                speed = np.linalg.norm(vel)

                # Color from velocity (using advanced system)
                hue = self.color_system.velocity_to_hue(vel)
                hue = (hue + audio_frame.brilliance * 40) % 360

                # Trail fades to cooler colors
                hue_shift = age_factor * 60  # Shift toward blue
                final_hue = (hue + hue_shift) % 360

                sat = 0.7 - age_factor * 0.3
                val = 0.8 - age_factor * 0.5
                val *= (0.5 + audio_frame.brilliance * 0.5)

                rgb = hsv_to_rgb(final_hue / 360.0, sat, val)
            else:
                rgb = particle_color_rgb

            colors.append(rgb)

            # Alpha fade (exp decay + audio)
            alpha_base = np.exp(-age_factor * 2.5)
            alpha = alpha_base * (0.4 + audio_frame.brilliance * 0.4)
            alphas.append(min(alpha, 0.85))

            # Width varies with velocity
            width = 0.8 + audio_frame.air * 0.6
            widths.append(width)

        # Draw as LineCollection for smooth gradients
        points = smooth_trail.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            colors=colors[:-1],
            linewidths=widths[:-1],
            alpha=0.7,
            zorder=3
        )
        self.ax.add_collection(lc)

    def _render_particle_organic(self, position, velocity, acceleration, mass,
                                particle_age, audio_frame):
        """
        Render particle with organic, sophisticated effects.

        Features:
        - Advanced color from multi-dimensional mapping
        - Soft multi-layer glow (not harsh)
        - Audio-reactive size (smooth)
        - Interaction glow with nearby particles
        - Subtle pulsing on beats
        """
        x, y, z = position
        speed = np.linalg.norm(velocity)

        # Get sophisticated color
        rgb = self.color_system.get_particle_color(
            position, velocity, acceleration, mass,
            particle_age, audio_frame
        )
        color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'

        # Size based on mass + audio (smooth, not flashy)
        base_size = 0.12 * (mass ** 0.5)

        # Subtle audio reactivity (not aggressive)
        size_mod = 1.0 + (
            audio_frame.presence * 0.3 +
            audio_frame.beat_strength * 0.5
        )
        radius = base_size * size_mod

        # Multi-layer organic glow (soft, bioluminescent)
        glow_intensity = 1.0 + (
            audio_frame.brilliance * 0.6 +
            audio_frame.beat_strength * 0.4
        )

        # Softer glow layers (more layers, lower alpha each = smoother)
        glow_layers = [
            (4.0, 0.008),
            (3.0, 0.015),
            (2.2, 0.025),
            (1.6, 0.045),
            (1.2, 0.08),
            (0.85, 0.15),
        ]

        for r_mult, alpha_base in glow_layers:
            r = radius * r_mult * glow_intensity
            alpha = alpha_base * glow_intensity

            # Color shifts slightly in outer layers (atmospheric scattering effect)
            if r_mult > 2:
                # Outer glow shifts toward blue
                h, s, v = rgb_to_hsv(*rgb)
                h = (h * 360 + 30) % 360
                outer_rgb = hsv_to_rgb(h / 360.0, s * 0.7, v)
                outer_color = f'#{int(outer_rgb[0]*255):02x}{int(outer_rgb[1]*255):02x}{int(outer_rgb[2]*255):02x}'
            else:
                outer_color = color

            circle = Circle(
                (x, y), r,
                color=outer_color,
                alpha=min(alpha, 0.3),
                zorder=5
            )
            self.ax.add_patch(circle)

        # Core particle (solid but not harsh)
        core_size = (30 + audio_frame.beat_strength * 50) * mass
        self.ax.scatter(
            [x], [y],
            c=[color],
            s=core_size,
            alpha=0.85,
            zorder=10,
            linewidths=0,
            edgecolors='none'
        )

        # Inner glow (brightest)
        self.ax.scatter(
            [x], [y],
            c=[color],
            s=core_size * 0.4,
            alpha=0.95,
            zorder=11
        )

    def _compute_particle_interactions(self, positions, masses):
        """
        Compute interaction glow between nearby particles.

        When particles get close, they create connecting glow.
        """
        n = len(positions)
        interactions = []

        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])

                # Only render if close
                if dist < 0.8:
                    strength = 1.0 - (dist / 0.8)
                    interactions.append({
                        'p1': positions[i],
                        'p2': positions[j],
                        'strength': strength,
                        'mass_avg': (masses[i] + masses[j]) / 2
                    })

        return interactions

    def _render_interactions(self, interactions, audio_frame):
        """Render subtle glow between interacting particles."""
        for inter in interactions:
            p1, p2 = inter['p1'], inter['p2']
            strength = inter['strength']

            # Line connecting particles (very subtle)
            alpha = strength * 0.15 * (0.5 + audio_frame.presence * 0.5)

            # Color from mid-point energy
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2

            hue = ((mid_x + 3) / 6 * 180 + audio_frame.high_mid * 60) % 360
            sat = 0.6 + audio_frame.mid * 0.3
            val = 0.5 + audio_frame.brilliance * 0.3

            rgb = hsv_to_rgb(hue / 360.0, sat, val)
            color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'

            self.ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                alpha=min(alpha, 0.2),
                linewidth=1.5 * strength,
                zorder=4
            )

    def _render_spectral_decomposition(self, audio_frame):
        """
        Visualize spectral content as subtle overlay.

        Creates organic patterns from frequency bands.
        """
        # Create circular pattern based on frequency bands
        bands = [
            audio_frame.sub_bass, audio_frame.bass, audio_frame.low_mid,
            audio_frame.mid, audio_frame.high_mid, audio_frame.presence,
            audio_frame.brilliance, audio_frame.air, audio_frame.ultra,
            audio_frame.extreme
        ]

        # Only render if there's significant spectral content
        if np.mean(bands) > 0.3:
            # Radial visualization (subtle)
            for i, energy in enumerate(bands):
                if energy > 0.2:
                    angle = (i / len(bands)) * 2 * np.pi
                    radius = 2.5 + energy * 0.5

                    # Position
                    x = np.cos(angle) * radius
                    y = np.sin(angle) * radius

                    # Color from frequency (low = red, high = blue)
                    hue = (i / len(bands)) * 300  # 0-300 degrees
                    rgb = hsv_to_rgb(hue / 360.0, 0.5, 0.4)
                    color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'

                    circle = Circle(
                        (x, y),
                        0.15 * energy,
                        color=color,
                        alpha=min(energy * 0.08, 0.1),
                        zorder=2
                    )
                    self.ax.add_patch(circle)

    def render_frame(self, positions, velocities, accelerations, masses,
                    trails, trail_velocities, audio_frame, time, frame_idx):
        """
        Render complete frame with all sophisticated effects.

        Args:
            positions: (n, 3) current positions
            velocities: (n, 3) current velocities
            accelerations: (n, 3) current accelerations
            masses: (n,) masses
            trails: list of (trail_len, 3) trail histories
            trail_velocities: list of (trail_len, 3) trail velocity histories
            audio_frame: AudioFrame10Band
            time: simulation time
            frame_idx: current frame number

        Returns:
            RGB numpy array (height, width, 3) uint8
        """
        # Setup frame
        self.ax.clear()
        self.ax.set_facecolor('#000205')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1.7, 1.7)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # 1. Multi-layer background
        self._render_layered_background(audio_frame, time)

        # 2. Spectral decomposition overlay
        self._render_spectral_decomposition(audio_frame)

        # 3. Particle interaction glow fields
        if frame_idx > 30:
            interactions = self._compute_particle_interactions(positions, masses)
            if len(interactions) > 0:
                self._render_interactions(interactions, audio_frame)

        # 4. Long flowing trails
        if frame_idx > 30 and trails is not None:
            for i in range(len(positions)):
                if i < len(trails):
                    # Get particle's main color
                    particle_rgb = self.color_system.get_particle_color(
                        positions[i], velocities[i], accelerations[i],
                        masses[i], time, audio_frame
                    )

                    trail_pos = trails[i]
                    trail_vel = trail_velocities[i] if i < len(trail_velocities) else None

                    mask = np.any(trail_pos != 0, axis=1)
                    if mask.sum() > 4:
                        valid_trail = trail_pos[mask]
                        valid_vel = trail_vel[mask] if trail_vel is not None else None

                        self._render_flowing_trail(
                            valid_trail, valid_vel,
                            particle_rgb, audio_frame
                        )

        # 5. Particles with organic glow
        for i in range(len(positions)):
            self._render_particle_organic(
                positions[i],
                velocities[i],
                accelerations[i],
                masses[i],
                time,  # Use as particle age
                audio_frame
            )

        # 6. Minimal UI (non-intrusive)
        self._render_minimal_ui(audio_frame)

        # Convert to RGB
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        return img[:, :, :3].copy()

    def _render_minimal_ui(self, audio_frame):
        """Subtle, non-intrusive UI elements."""
        t = audio_frame.time

        # Time (very subtle)
        self.ax.text(
            0.015, 0.98,
            f'{int(t//60)}:{int(t%60):02d}',
            transform=self.ax.transAxes,
            color='white',
            fontsize=11,
            alpha=0.25,
            fontfamily='monospace',
            verticalalignment='top'
        )

        # Beat indicator (gentle pulse, not flashy)
        if audio_frame.beat_strength > 0.6:
            alpha = audio_frame.beat_strength * 0.4
            size = 10 + audio_frame.beat_strength * 6

            self.ax.text(
                0.985, 0.98, '○',
                transform=self.ax.transAxes,
                color='#ff88aa',
                fontsize=size,
                alpha=min(alpha, 0.5),
                verticalalignment='top',
                horizontalalignment='right'
            )
