"""
Fast Shader-Enhanced Renderer

Optimized for speed while keeping visual quality:
- Reduced glow layers (3 instead of 5)
- Simplified trail rendering
- Cached matplotlib objects
- Batch operations where possible
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


class FastShaderRenderer:
    """Fast shader-quality renderer (10+ fps target)."""

    def __init__(self, width=1920, height=1080, dpi=100):
        self.width = width
        self.height = height
        self.dpi = dpi

        fig_w = width / dpi
        fig_h = height / dpi
        self.fig, self.ax = plt.subplots(
            figsize=(fig_w, fig_h),
            dpi=dpi,
            facecolor='#000508'  # Slightly blue-black
        )
        self.ax.set_facecolor('#000508')

        # Velocity colormap (slow → cold, fast → hot)
        self.velocity_cmap = LinearSegmentedColormap.from_list(
            'velocity',
            ['#0088ff', '#00ffcc', '#00ff88', '#ffcc00', '#ff00aa']
        )

    def _render_nebula_bg(self, audio):
        """Fast nebula background (single layer)."""
        # Bass pulse only (cheap)
        bass = (audio.sub_bass + audio.bass) / 2
        if bass > 0.3:
            r = 2.0 * (1 + bass * 0.5)
            circle = plt.Circle(
                (0, 0), r,
                color='#001133',
                alpha=min(bass * 0.15, 0.15),
                zorder=1
            )
            self.ax.add_patch(circle)

        # Extreme frequency stars (sparse)
        if audio.extreme > 0.6:
            n = int(10 * audio.extreme)
            x = np.random.uniform(-3, 3, n)
            y = np.random.uniform(-1.7, 1.7, n)
            self.ax.scatter(x, y, s=1, c='white', alpha=0.4, zorder=1)

    def _get_color(self, velocity, audio):
        """Fast velocity → color mapping."""
        speed = np.linalg.norm(velocity)
        speed_norm = np.clip(speed / 2.0, 0, 1)
        rgba = self.velocity_cmap(speed_norm)
        return f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}'

    def render_frame(self, positions, velocities, masses, trails, audio, frame_idx):
        """Fast render with essential shader effects only."""
        # Clear
        self.ax.clear()
        self.ax.set_facecolor('#000508')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1.7, 1.7)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Background
        self._render_nebula_bg(audio)

        # Trails (simplified - single line per body)
        if frame_idx > 30 and trails is not None:
            trail_alpha = 0.3 + audio.brilliance * 0.4

            for i in range(min(len(positions), len(trails))):
                trail = trails[i]
                mask = np.any(trail != 0, axis=1)

                if mask.sum() > 2:
                    valid = trail[mask][-300:]  # Last 300 points (less than 500)
                    color = self._get_color(velocities[i], audio)

                    # Single line (fast)
                    self.ax.plot(
                        valid[:, 0], valid[:, 1],
                        color=color,
                        alpha=min(trail_alpha, 0.7),
                        linewidth=0.8 + audio.air * 0.4,
                        zorder=3
                    )

        # Particles (3-layer glow instead of 5)
        glow = 1.0 + audio.presence * 1.0 + audio.beat_strength * 1.5

        for i in range(len(positions)):
            x, y = positions[i, 0], positions[i, 1]
            color = self._get_color(velocities[i], audio)
            mass = masses[i]

            r_base = 0.1 * np.sqrt(mass)

            # 3-layer glow (reduced from 5)
            for r_mult, alpha_base in [(2.5, 0.04), (1.5, 0.12), (0.8, 0.3)]:
                r = r_base * r_mult * glow
                alpha = min(alpha_base * glow, 0.9)

                circle = plt.Circle(
                    (x, y), r,
                    color=color,
                    alpha=alpha,
                    zorder=5
                )
                self.ax.add_patch(circle)

            # Core
            size = (40 + audio.beat_strength * 80) * mass
            self.ax.scatter(
                [x], [y],
                c=[color],
                s=size,
                alpha=0.95,
                zorder=10
            )

        # UI (minimal)
        t = audio.time
        self.ax.text(
            0.02, 0.97, f'{int(t//60)}:{int(t%60):02d}',
            transform=self.ax.transAxes,
            color='white', fontsize=14, alpha=0.5,
            fontfamily='monospace', verticalalignment='top'
        )

        # Beat indicator
        if audio.beat_strength > 0.5:
            self.ax.text(
                0.98, 0.97, '●',
                transform=self.ax.transAxes,
                color='#ff4444',
                fontsize=16 + audio.beat_strength * 12,
                alpha=min(audio.beat_strength, 1.0),
                verticalalignment='top',
                horizontalalignment='right'
            )

        # Render to bytes
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        return img[:, :, :3].copy()
