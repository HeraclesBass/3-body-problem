"""
Intro Sequence / Hook Enhancement for V3

Optimizes the first 3 seconds for maximum impact (viral hook).

Techniques:
- Zoom from space → particle field
- High-energy particle spawn on first beats
- Full palette visible by 2 seconds
- Optional text overlay support
- Watermark positioning

Usage:
    intro = IntroSequence(duration=3.0)

    # Check if in intro period
    if intro.is_intro_active(current_time):
        # Get intro camera position
        cam_pos = intro.get_intro_camera(
            current_time,
            target_pos=center_of_mass
        )

        # Get intro particle boost
        particle_boost = intro.get_particle_energy_boost(current_time)
"""

import numpy as np
from typing import Tuple, Optional


class IntroSequence:
    """
    Manages cinematic intro sequence for first 3 seconds.

    Creates high-impact opening to capture viewer attention.
    """

    def __init__(self, duration: float = 3.0):
        """
        Initialize intro sequence.

        Args:
            duration: Duration of intro (seconds)
        """
        self.duration = duration
        self.start_distance = 500.0  # Start zoomed way out
        self.end_distance = 100.0  # End at normal distance

    def is_intro_active(self, current_time: float) -> bool:
        """Check if intro is currently active."""
        return current_time < self.duration

    def get_intro_progress(self, current_time: float) -> float:
        """
        Get intro progress [0, 1].

        Args:
            current_time: Current time (seconds)

        Returns:
            Progress value, 0 at start, 1 at end
        """
        if not self.is_intro_active(current_time):
            return 1.0

        progress = current_time / self.duration
        # Ease-out for smooth deceleration
        return 1.0 - (1.0 - progress) ** 2

    def get_intro_camera(
        self,
        current_time: float,
        target_pos: np.ndarray
    ) -> np.ndarray:
        """
        Get camera position during intro.

        Args:
            current_time: Current time (seconds)
            target_pos: Target position (center of mass)

        Returns:
            Camera position [x, y, z]
        """
        progress = self.get_intro_progress(current_time)

        # Interpolate distance from far to near
        distance = self.start_distance + (self.end_distance - self.start_distance) * progress

        # Position camera above and behind target
        camera_pos = target_pos + np.array([0, 0, distance])

        return camera_pos

    def get_particle_energy_boost(self, current_time: float) -> float:
        """
        Get particle energy boost during intro.

        First beats get extra energy for visual impact.

        Args:
            current_time: Current time (seconds)

        Returns:
            Energy boost multiplier [1.0, 2.0]
        """
        if not self.is_intro_active(current_time):
            return 1.0

        # High boost at start, decay to normal
        progress = current_time / self.duration
        boost = 1.0 + (1.0 - progress) * 1.0  # 2.0x at start, 1.0x at end

        return boost

    def get_palette_saturation_boost(self, current_time: float) -> float:
        """
        Get color saturation boost during intro.

        Ensures full palette visible quickly.

        Args:
            current_time: Current time (seconds)

        Returns:
            Saturation boost [1.0, 1.5]
        """
        if not self.is_intro_active(current_time):
            return 1.0

        # Ramp up saturation in first second
        if current_time < 1.0:
            return 1.0 + 0.5 * (current_time / 1.0)
        return 1.5


class TextOverlay:
    """Manages text overlays for social media."""

    def __init__(self, text: str, position: str = 'bottom'):
        """
        Initialize text overlay.

        Args:
            text: Text to display
            position: 'top', 'center', or 'bottom'
        """
        self.text = text
        self.position = position

    def get_position_pixels(
        self,
        frame_width: int,
        frame_height: int,
        safe_zones: dict
    ) -> Tuple[int, int]:
        """
        Get text position in pixels.

        Args:
            frame_width: Frame width
            frame_height: Frame height
            safe_zones: Dict with top/bottom/left/right percentages

        Returns:
            (x, y) position in pixels
        """
        x = frame_width // 2  # Centered horizontally

        if self.position == 'top':
            y = int(frame_height * safe_zones['top'])
        elif self.position == 'center':
            y = frame_height // 2
        else:  # bottom
            y = int(frame_height * (1.0 - safe_zones['bottom']))

        return (x, y)


class Watermark:
    """Manages watermark placement."""

    def __init__(
        self,
        logo_path: Optional[str] = None,
        position: str = 'bottom-right',
        opacity: float = 0.7
    ):
        """
        Initialize watermark.

        Args:
            logo_path: Path to logo image (optional)
            position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
            opacity: Watermark opacity [0, 1]
        """
        self.logo_path = logo_path
        self.position = position
        self.opacity = opacity

    def get_position_pixels(
        self,
        frame_width: int,
        frame_height: int,
        logo_width: int,
        logo_height: int,
        margin: int = 20
    ) -> Tuple[int, int]:
        """
        Get watermark position in pixels.

        Args:
            frame_width: Frame width
            frame_height: Frame height
            logo_width: Logo width
            logo_height: Logo height
            margin: Margin from edges (pixels)

        Returns:
            (x, y) position in pixels
        """
        if 'right' in self.position:
            x = frame_width - logo_width - margin
        else:  # left
            x = margin

        if 'bottom' in self.position:
            y = frame_height - logo_height - margin
        else:  # top
            y = margin

        return (x, y)
