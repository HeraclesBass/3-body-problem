"""
Cinematic Renderer - Next Generation Visualization

Transforms chaotic random colors into smooth, cohesive, captivating visuals.

Key Features:
- Unified color palette (harmonious, not chaotic)
- 3D shaded spheres (realistic depth)
- Smooth camera tracking (cinematic framing)
- Parallax starfield (depth perception)
- Gaussian bloom (professional glow)
- Smooth audio envelopes (musical reactions)

Design Philosophy:
- Coherence over chaos
- Depth over flatness
- Follow the action
- Musical harmony
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
try:
    from scipy.interpolate import splprep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from rendering.unified_palette import UnifiedPalette
from rendering.smooth_camera import SmoothCamera, compute_center_of_mass
from audio.smoothed_envelope import MultiEnvelopeController


class ParallaxStarfield:
    """Multi-layer starfield with depth parallax."""

    def __init__(self, width, height, n_layers=3, seed=42):
        """
        Create parallax starfield.

        Args:
            width: Frame width in data coordinates
            height: Frame height in data coordinates
            n_layers: Number of depth layers (3-5 typical)
            seed: Random seed for reproducible stars
        """
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.layers = []

        # Create layers at different depths
        for i in range(n_layers):
            depth = (i + 1) / (n_layers + 1)  # 0.25, 0.5, 0.75 for n=3
            n_stars = int(100 / (depth + 0.5))  # More stars in distant layers

            # Star properties
            positions = np.column_stack([
                np.random.uniform(-width/2, width/2, n_stars),
                np.random.uniform(-height/2, height/2, n_stars)
            ])

            sizes = np.random.uniform(0.01, 0.03, n_stars)
            brightness = 0.2 + depth * 0.3  # Distant = dimmer

            self.layers.append({
                'positions': positions,
                'depth': depth,
                'sizes': sizes,
                'brightness': brightness
            })

    def render(self, ax, camera_offset):
        """
        Render stars with parallax effect.

        Args:
            ax: Matplotlib axes
            camera_offset: [x, y] camera offset from origin
        """
        for layer in self.layers:
            # Parallax: offset inversely proportional to depth
            # Distant stars move slower
            parallax_factor = 1.0 - layer['depth']
            parallax_offset = camera_offset * parallax_factor

            # Star positions in camera space
            star_positions = layer['positions'] - parallax_offset

            # Wrap positions for infinite scrolling
            # (This creates the illusion of endless space)
            x = star_positions[:, 0]
            y = star_positions[:, 1]
            x = ((x + self.width/2) % self.width) - self.width/2
            y = ((y + self.height/2) % self.height) - self.height/2

            # Draw stars
            ax.scatter(
                x, y,
                s=layer['sizes'] * 100,  # Matplotlib size scaling
                c='white',
                alpha=layer['brightness'],
                zorder=1
            )


class CinematicRenderer:
    """
    Next-generation renderer with cohesive visuals.

    Combines all improvements:
    - Unified palette
    - 3D shaded spheres
    - Smooth camera
    - Parallax background
    - Gaussian bloom
    - Smooth audio
    """

    def __init__(self, width=1920, height=1080, dpi=100):
        """
        Initialize cinematic renderer.

        Args:
            width: Output width (pixels)
            height: Output height (pixels)
            dpi: Dots per inch (affects matplotlib figure size)
        """
        self.width = width
        self.height = height
        self.dpi = dpi

        # Create matplotlib figure
        fig_w, fig_h = width / dpi, height / dpi
        self.fig, self.ax = plt.subplots(
            figsize=(fig_w, fig_h),
            dpi=dpi,
            facecolor='#000000'  # Pure black
        )
        self.ax.set_facecolor('#000000')

        # World coordinate system (aspect-corrected)
        aspect = width / height  # e.g., 16/9 = 1.78
        self.world_width = 6.0   # Total width in world units
        self.world_height = self.world_width / aspect  # Maintain aspect

        # Initialize systems
        self.palette = UnifiedPalette()
        self.camera = SmoothCamera(smoothing=0.12, damping=0.82)
        self.audio_envelopes = MultiEnvelopeController(fps=30)
        self.starfield = ParallaxStarfield(
            self.world_width,
            self.world_height,
            n_layers=3
        )

        # Frame counter for time-based effects
        self.frame_count = 0

    def _render_3d_sphere(self, center, radius, color, intensity=1.0):
        """
        Render 3D sphere with Phong shading.

        Args:
            center: [x, y] position
            radius: Sphere radius
            color: Base RGB color
            intensity: Light intensity multiplier

        Creates realistic sphere with:
        - Ambient lighting (base color)
        - Diffuse lighting (Lambertian)
        - Specular highlight (shiny spot)
        """
        # Light direction (from upper-right, slightly in front)
        light_dir = np.array([0.3, 0.3, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)

        # Create multiple glow layers for soft appearance
        # Instead of computing full Phong per-pixel (expensive),
        # we approximate with concentric circles of varying brightness

        # Core (brightest, specular highlight)
        core_color = np.array(color) * (0.8 + 0.4 * intensity)
        core_color = np.clip(core_color, 0, 1)
        core = Circle(
            center, radius * 0.3,
            color=core_color,
            alpha=1.0,
            zorder=10
        )
        self.ax.add_patch(core)

        # Mid layers (diffuse shading)
        for r_mult, brightness in [(0.5, 0.9), (0.7, 0.7), (0.9, 0.5)]:
            layer_color = np.array(color) * brightness * intensity
            layer_color = np.clip(layer_color, 0, 1)
            layer = Circle(
                center, radius * r_mult,
                color=layer_color,
                alpha=0.8,
                zorder=9
            )
            self.ax.add_patch(layer)

        # Outer glow (soft ambient)
        outer_color = np.array(color) * 0.3
        outer = Circle(
            center, radius * 1.2,
            color=outer_color,
            alpha=0.4,
            zorder=8
        )
        self.ax.add_patch(outer)

    def _render_trail(self, trail_positions, particle_index, n_particles, audio_frame):
        """
        Render smooth particle trail.

        Args:
            trail_positions: (n_points, 3) trail history
            particle_index: Index of this particle
            n_particles: Total particles
            audio_frame: Current audio frame
        """
        n_points = len(trail_positions)
        if n_points < 4:
            return

        # Take recent trail points
        trail = trail_positions[-800:]  # Shorter trails for cleaner look
        n = len(trail)

        if n < 4:
            return

        # Optional: Spline smoothing for very smooth curves
        if HAS_SCIPY and n > 10:
            try:
                tck, u = splprep([trail[:, 0], trail[:, 1]], s=0.3, k=min(3, n-1))
                u_smooth = np.linspace(0, 1, n)
                smooth_trail = np.array(splev(u_smooth, tck)).T
            except:
                smooth_trail = trail[:, :2]
        else:
            smooth_trail = trail[:, :2]

        # Trail colors from palette
        n_smooth = len(smooth_trail)
        colors = []
        for i in range(n_smooth):
            age = i / n_smooth
            rgb = self.palette.get_trail_color(
                particle_index, n_particles, age, audio_frame
            )
            colors.append(rgb)

        # Draw as line collection
        points = smooth_trail.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Width decreases with age
        widths = np.linspace(1.5, 0.3, n_smooth - 1)

        lc = LineCollection(
            segments,
            colors=colors[:-1],
            linewidths=widths,
            alpha=0.6,
            zorder=5
        )
        self.ax.add_collection(lc)

    def render_frame(self, positions, velocities, accelerations, masses,
                    trails, trail_velocities, audio_frame, time, frame_idx):
        """
        Render complete frame with all cinematic features.

        Args:
            positions: (n, 3) particle positions
            velocities: (n, 3) particle velocities
            accelerations: (n, 3) particle accelerations
            masses: (n,) particle masses
            trails: (n, trail_len, 3) trail histories
            trail_velocities: (n, trail_len, 3) trail velocity histories
            audio_frame: AudioFrame10Band
            time: Simulation time
            frame_idx: Frame number

        Returns:
            RGB numpy array (height, width, 3) uint8
        """
        # =====================================================================
        # UPDATE SYSTEMS
        # =====================================================================
        # Update color palette
        self.palette.update(audio_frame, dt=1.0/30.0)

        # Update audio envelopes
        smoothed_audio = self.audio_envelopes.update(audio_frame)

        # Update camera to follow center of mass
        com = compute_center_of_mass(positions, masses)
        camera_pos = self.camera.update(com, dt=1.0/30.0)

        # =====================================================================
        # SETUP FRAME
        # =====================================================================
        self.ax.clear()
        self.ax.set_facecolor('#000000')

        # Set view bounds (camera-centered)
        half_w = self.world_width / 2
        half_h = self.world_height / 2
        self.ax.set_xlim(camera_pos[0] - half_w, camera_pos[0] + half_w)
        self.ax.set_ylim(camera_pos[1] - half_h, camera_pos[1] + half_h)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # =====================================================================
        # LAYER 1: PARALLAX STARFIELD
        # =====================================================================
        self.starfield.render(self.ax, camera_pos)

        # =====================================================================
        # LAYER 2: SUBTLE BACKGROUND GLOW
        # =====================================================================
        # Gentle bass-reactive glow at center of mass
        if smoothed_audio['bass'] > 0.2:
            glow_color = self.palette.get_background_color(time, audio_frame)
            glow_radius = 2.0 * (1.0 + smoothed_audio['bass'] * 0.5)

            glow = Circle(
                com[:2],
                glow_radius,
                color=glow_color,
                alpha=0.08 * smoothed_audio['bass'],
                zorder=2
            )
            self.ax.add_patch(glow)

        # =====================================================================
        # LAYER 3: PARTICLE TRAILS
        # =====================================================================
        if trails is not None and frame_idx > 30:
            for i in range(len(positions)):
                if i < len(trails):
                    trail_pos = trails[i]
                    mask = np.any(trail_pos != 0, axis=1)
                    if mask.sum() > 4:
                        valid_trail = trail_pos[mask]
                        self._render_trail(valid_trail, i, len(positions), audio_frame)

        # =====================================================================
        # LAYER 4: 3D SHADED PARTICLES
        # =====================================================================
        for i in range(len(positions)):
            pos = positions[i]
            vel = velocities[i]
            mass = masses[i]

            # Particle energy
            speed = np.linalg.norm(vel)
            energy = 0.5 * mass * speed * speed

            # Get color from unified palette
            color = self.palette.get_particle_color(
                i, len(positions), energy, audio_frame
            )

            # Base size from mass
            base_radius = 0.08 * (mass ** 0.4)

            # Gentle audio-reactive size modulation
            size_mod = 1.0 + smoothed_audio['beat'] * 0.2  # Only 20% change
            radius = base_radius * size_mod

            # Light intensity from beat
            intensity = 1.0 + smoothed_audio['beat'] * 0.5

            # Render 3D sphere
            self._render_3d_sphere(pos[:2], radius, color, intensity)

        # =====================================================================
        # LAYER 5: MINIMAL UI
        # =====================================================================
        # Time display (very subtle)
        self.ax.text(
            0.015, 0.98,
            f'{int(time//60)}:{int(time%60):02d}',
            transform=self.ax.transAxes,
            color='white',
            fontsize=10,
            alpha=0.2,
            fontfamily='monospace',
            verticalalignment='top'
        )

        # Beat indicator (gentle pulse)
        if smoothed_audio['beat'] > 0.3:
            size = 8 + smoothed_audio['beat'] * 4
            self.ax.text(
                0.985, 0.98, '○',
                transform=self.ax.transAxes,
                color='#ffffff',
                fontsize=size,
                alpha=min(smoothed_audio['beat'] * 0.4, 0.4),
                verticalalignment='top',
                horizontalalignment='right'
            )

        # =====================================================================
        # CONVERT TO RGB ARRAY
        # =====================================================================
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        rgb = img[:, :, :3].copy()

        # =====================================================================
        # POST-PROCESS: GAUSSIAN BLOOM
        # =====================================================================
        rgb_float = rgb.astype(np.float32) / 255.0

        # Extract bright areas
        luminance = 0.299*rgb_float[:,:,0] + 0.587*rgb_float[:,:,1] + 0.114*rgb_float[:,:,2]
        threshold = 0.6
        bright_mask = luminance > threshold

        # Bloom only bright pixels
        bloomed = np.zeros_like(rgb_float)
        bright_image = rgb_float.copy()
        bright_image[~bright_mask] = 0

        # Gaussian blur each channel
        blur_radius = 15  # Moderate bloom
        for c in range(3):
            bloomed[:,:,c] = gaussian_filter(bright_image[:,:,c], sigma=blur_radius)

        # Additive blend
        bloom_strength = 0.6  # Moderate strength
        result = rgb_float + bloomed * bloom_strength
        result = np.clip(result, 0, 1)

        # Convert back to uint8
        final = (result * 255).astype(np.uint8)

        self.frame_count += 1
        return final
