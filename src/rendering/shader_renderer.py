"""
Shader-Enhanced Matplotlib Renderer

Pushes matplotlib to its limits with GPU-shader-like effects:
- Multi-layer particle glow (additive blending)
- Velocity-based color gradients
- Dynamic sizing with audio reactivity
- Nebula shader background
- Glowing trails with color transitions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


class ShaderRenderer:
    """
    Shader-quality renderer using matplotlib.

    Simulates GPU shader effects through clever use of:
    - Alpha compositing (additive blending)
    - Radial gradients (multi-layer circles)
    - Color interpolation (velocity → hue)
    - Gaussian noise (nebula background)
    """

    def __init__(self, width=1920, height=1080, dpi=100):
        """
        Initialize renderer.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            dpi: Dots per inch for matplotlib
        """
        self.width = width
        self.height = height
        self.dpi = dpi

        # Create figure
        fig_w = width / dpi
        fig_h = height / dpi
        self.fig, self.ax = plt.subplots(
            figsize=(fig_w, fig_h),
            dpi=dpi,
            facecolor='black'
        )
        self.ax.set_facecolor('black')

        # Bioluminescent color palette
        self.colors = {
            'cyan': '#00ffcc',
            'magenta': '#ff00aa',
            'gold': '#ffcc00',
            'blue': '#0088ff',
            'green': '#00ff88',
            'purple': '#aa00ff',
            'orange': '#ff8800',
            'pink': '#ff0088',
        }

        # Velocity-based color map (slow → cold, fast → hot)
        self.velocity_cmap = LinearSegmentedColormap.from_list(
            'velocity',
            ['#0088ff', '#00ffcc', '#ffcc00', '#ff00aa', '#ff0044']
        )

    def _render_nebula_background(self, audio_frame):
        """
        Shader-like nebula background.

        Uses radial gradients + audio reactivity for living space.
        """
        # Background base glow (ultra frequencies)
        if audio_frame.ultra > 0.2:
            # Large soft glow
            intensity = audio_frame.ultra * 0.4
            for radius, alpha in [(4, 0.02), (3, 0.04), (2, 0.06)]:
                circle = plt.Circle(
                    (0, 0), radius,
                    color='#1a0033',
                    alpha=alpha * intensity
                )
                self.ax.add_patch(circle)

        # Bass pulse (sub-bass + bass)
        bass_total = (audio_frame.sub_bass + audio_frame.bass) / 2
        if bass_total > 0.25:
            # Center gravity well
            for radius, alpha in [(2.5, 0.03), (1.8, 0.06), (1.2, 0.09)]:
                circle = plt.Circle(
                    (0, 0), radius * (1 + bass_total * 0.5),
                    color='#001a33',
                    alpha=alpha * bass_total
                )
                self.ax.add_patch(circle)

        # Extreme frequency sparkles (particle field)
        if audio_frame.extreme > 0.5:
            n_sparkles = int(20 * audio_frame.extreme)
            x = np.random.uniform(-3, 3, n_sparkles)
            y = np.random.uniform(-1.7, 1.7, n_sparkles)
            sizes = np.random.uniform(1, 4, n_sparkles) * audio_frame.extreme
            self.ax.scatter(
                x, y,
                s=sizes,
                c='white',
                alpha=0.3 * audio_frame.extreme,
                zorder=1
            )

    def _get_particle_color(self, velocity, audio_frame):
        """
        Compute particle color from velocity + audio.

        Args:
            velocity: vec3 velocity
            audio_frame: Audio analysis

        Returns:
            (color_hex, saturation_multiplier)
        """
        speed = np.linalg.norm(velocity)

        # Velocity → color mapping (0-1 range)
        speed_norm = np.clip(speed / 2.0, 0, 1)

        # Use colormap
        rgba = self.velocity_cmap(speed_norm)
        color_hex = f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}'

        # High-mid frequencies boost saturation
        saturation = 1.0 + audio_frame.high_mid * 0.5

        return color_hex, saturation

    def _render_particle_shader(self, pos, vel, mass, audio_frame, base_color):
        """
        Render single particle with multi-layer shader-like glow.

        Simulates volumetric emission through layered alpha-blended circles.
        """
        x, y = pos[0], pos[1]

        # Dynamic size (mass + presence frequencies)
        base_size = 0.12 * np.sqrt(mass)
        size_mod = 1.0 + audio_frame.presence * 0.8
        radius = base_size * size_mod

        # Glow intensity (brilliance + beat)
        glow_intensity = 1.0 + audio_frame.brilliance * 1.5 + audio_frame.beat_strength * 2.0

        # Multi-layer glow (shader-like additive blending)
        glow_layers = [
            (radius * 3.0 * glow_intensity, 0.02),
            (radius * 2.0 * glow_intensity, 0.05),
            (radius * 1.4 * glow_intensity, 0.12),
            (radius * 1.0, 0.25),
            (radius * 0.6, 0.5),
        ]

        for r, alpha in glow_layers:
            circle = plt.Circle(
                (x, y), r,
                color=base_color,
                alpha=min(alpha * glow_intensity, 1.0),
                zorder=5
            )
            self.ax.add_patch(circle)

        # Core (solid)
        core_size = (50 + audio_frame.beat_strength * 120) * (mass ** 0.5)
        self.ax.scatter(
            [x], [y],
            c=[base_color],
            s=core_size,
            alpha=0.9,
            zorder=10,
            edgecolors='white',
            linewidths=0.5 * audio_frame.air
        )

    def _render_trail_shader(self, trail, velocity_history, audio_frame, particle_color):
        """
        Shader-enhanced trail rendering.

        Features:
        - Age-based alpha fadeout
        - Velocity-based color gradient
        - Audio-reactive brightness (brilliance)
        - Smooth interpolation
        """
        if len(trail) < 3:
            return

        n_points = len(trail)

        # Trail brightness (brilliance frequencies)
        brightness = 0.4 + audio_frame.brilliance * 0.6

        # Draw trail segments with gradient
        for i in range(1, n_points, 2):  # Skip every other for performance
            # Age-based alpha (newer = brighter)
            age_norm = i / n_points
            alpha = (1.0 - age_norm) * brightness

            # Velocity at this point
            if velocity_history is not None and i < len(velocity_history):
                vel = velocity_history[i]
                speed = np.linalg.norm(vel)
                speed_norm = np.clip(speed / 2.0, 0, 1)

                # Color transition along trail
                rgba = self.velocity_cmap(speed_norm)
                segment_color = f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}'
            else:
                segment_color = particle_color

            # Line width (audio-reactive)
            line_width = 1.2 + audio_frame.air * 0.8

            # Draw segment
            self.ax.plot(
                trail[i-1:i+1, 0],
                trail[i-1:i+1, 1],
                color=segment_color,
                alpha=min(alpha, 0.8),
                linewidth=line_width,
                zorder=3
            )

    def render_frame(self, positions, velocities, masses, trails, audio_frame, frame_count):
        """
        Render complete frame with all shader effects.

        Args:
            positions: numpy array (n, 3) particle positions
            velocities: numpy array (n, 3) particle velocities
            masses: numpy array (n,) particle masses
            trails: list of trail arrays per particle
            audio_frame: AudioFrame10Band with all frequency data
            frame_count: Current frame number

        Returns:
            RGB numpy array (height, width, 3) uint8
        """
        # Clear previous frame
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1.7, 1.7)  # 16:9 aspect
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # 1. Render nebula background (shader)
        self._render_nebula_background(audio_frame)

        # 2. Render trails (if enough frames elapsed)
        if frame_count > 30 and trails is not None:
            for i in range(len(positions)):
                if i < len(trails):
                    trail = trails[i]
                    velocity_history = None  # TODO: Get from simulation

                    # Mask out zero entries
                    mask = np.any(trail != 0, axis=1)
                    if mask.sum() > 2:
                        valid_trail = trail[mask][-500:]  # Last 500 points

                        # Get particle color for this trail
                        color, _ = self._get_particle_color(velocities[i], audio_frame)

                        self._render_trail_shader(
                            valid_trail,
                            velocity_history,
                            audio_frame,
                            color
                        )

        # 3. Render particles (shader-enhanced)
        for i in range(len(positions)):
            pos = positions[i]
            vel = velocities[i]
            mass = masses[i]

            # Get velocity-based color
            color, saturation = self._get_particle_color(vel, audio_frame)

            # Render with shader-like multi-layer glow
            self._render_particle_shader(pos, vel, mass, audio_frame, color)

        # 4. UI overlays
        self._render_ui(audio_frame, frame_count)

        # 5. Render to RGB bytes
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        rgb = img[:, :, :3].copy()

        return rgb

    def _render_ui(self, audio_frame, frame_count):
        """Render time, beat indicator, and frequency bars."""
        t = audio_frame.time

        # Time display
        self.ax.text(
            0.02, 0.97,
            f'{int(t//60)}:{int(t%60):02d}',
            transform=self.ax.transAxes,
            color='white',
            fontsize=16,
            alpha=0.6,
            fontfamily='monospace',
            verticalalignment='top'
        )

        # Beat indicator (pulsing circle)
        if audio_frame.beat_strength > 0.4:
            self.ax.text(
                0.98, 0.97, '●',
                transform=self.ax.transAxes,
                color='#ff4444',
                fontsize=20 + audio_frame.beat_strength * 20,
                alpha=audio_frame.beat_strength,
                verticalalignment='top',
                horizontalalignment='right'
            )

        # Frequency spectrum bars (bottom right)
        bar_x = 0.92
        bar_y_start = 0.05
        bar_width = 0.006
        bar_spacing = 0.008

        bands = [
            ('sub_bass', audio_frame.sub_bass, '#0088ff'),
            ('bass', audio_frame.bass, '#00ffcc'),
            ('low_mid', audio_frame.low_mid, '#00ff88'),
            ('mid', audio_frame.mid, '#88ff00'),
            ('high_mid', audio_frame.high_mid, '#ffcc00'),
            ('presence', audio_frame.presence, '#ff8800'),
            ('brilliance', audio_frame.brilliance, '#ff00aa'),
            ('air', audio_frame.air, '#ff0088'),
            ('ultra', audio_frame.ultra, '#aa00ff'),
            ('extreme', audio_frame.extreme, '#ff00ff'),
        ]

        for i, (name, energy, color) in enumerate(bands):
            x = bar_x - i * bar_spacing
            height = energy * 0.15  # Max 15% of screen height

            # Bar
            rect = mpatches.Rectangle(
                (x, bar_y_start), bar_width, height,
                transform=self.ax.transAxes,
                color=color,
                alpha=0.6 + energy * 0.4,
                zorder=20
            )
            self.ax.add_patch(rect)

    def get_frame_bytes(self):
        """Get current frame as bytes (for ffmpeg)."""
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        rgb = img[:, :, :3].copy()
        return rgb.tobytes()
