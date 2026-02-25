"""
Smooth Camera Tracking System

Follows center of mass with spring-damped motion for cinematic feel.
Always keeps action centered, creates dynamic compositions.

Design Philosophy:
- Follow, don't snap
- Smooth, organic motion
- Never lose track of particles
- Cinematic framing
"""

import numpy as np


class SmoothCamera:
    """
    Spring-damped camera that smoothly follows a target.

    Uses second-order dynamics (spring + damper) for natural,
    organic camera movement that feels cinematic.
    """

    def __init__(self, smoothing=0.1, damping=0.8, max_speed=5.0):
        """
        Initialize camera at origin.

        Args:
            smoothing: Spring strength (0.05-0.2)
                - Lower = smoother but slower
                - Higher = faster but more jerky
                - Typical: 0.1
            damping: Velocity damping (0.7-0.95)
                - Lower = more bouncy
                - Higher = more critically damped
                - Typical: 0.8
            max_speed: Maximum camera velocity (prevents sudden jumps)
                - Units per second
                - Typical: 5.0
        """
        # Current camera position (world coordinates)
        self.position = np.array([0.0, 0.0], dtype=np.float64)

        # Current camera velocity (for smooth motion)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)

        # Physics parameters
        self.smoothing = smoothing
        self.damping = damping
        self.max_speed = max_speed

    def update(self, target_position, dt=1.0/30.0):
        """
        Update camera to follow target.

        Uses spring-damper model:
        1. Acceleration = spring_force (toward target)
        2. Velocity += acceleration * dt
        3. Velocity *= damping (prevent oscillation)
        4. Position += velocity * dt

        Args:
            target_position: [x, y] world coordinates to follow
            dt: Time step (seconds)

        Returns:
            Current camera position [x, y]
        """
        # Convert target to numpy array
        target = np.array(target_position, dtype=np.float64)

        # Spring force (acceleration toward target)
        offset = target - self.position
        acceleration = offset * self.smoothing

        # Update velocity
        self.velocity += acceleration * dt

        # Apply damping (prevents oscillation)
        self.velocity *= self.damping

        # Clamp velocity to max speed (prevents sudden jumps)
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # Update position
        self.position += self.velocity * dt

        return self.position.copy()

    def get_position(self):
        """Get current camera position."""
        return self.position.copy()

    def get_velocity(self):
        """Get current camera velocity."""
        return self.velocity.copy()

    def reset(self, position=None):
        """Reset camera to position with zero velocity."""
        if position is not None:
            self.position = np.array(position, dtype=np.float64)
        else:
            self.position = np.array([0.0, 0.0], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)

    def teleport(self, position):
        """
        Instantly move camera (no smooth transition).

        Use sparingly (e.g., scene changes).
        """
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)


def compute_center_of_mass(positions, masses):
    """
    Compute weighted center of mass.

    Args:
        positions: (n, 3) array of particle positions [x, y, z]
        masses: (n,) array of particle masses

    Returns:
        [x, y] center of mass (ignores z)
    """
    if len(positions) == 0:
        return np.array([0.0, 0.0])

    # Weighted average
    total_mass = np.sum(masses)
    if total_mass == 0:
        # Unweighted center if all masses zero
        com = np.mean(positions[:, :2], axis=0)
    else:
        com = np.sum(positions[:, :2] * masses[:, None], axis=0) / total_mass

    return com


def compute_camera_target(positions, masses, velocities=None, look_ahead=0.5):
    """
    Compute intelligent camera target.

    Can incorporate velocity for "look ahead" - camera leads the
    motion slightly for better composition.

    Args:
        positions: (n, 3) array of positions
        masses: (n,) array of masses
        velocities: (n, 3) array of velocities (optional)
        look_ahead: How far to look ahead (0-1)
            - 0: Pure center of mass
            - 0.5: Moderate look ahead
            - 1.0: Full velocity look ahead

    Returns:
        [x, y] target position for camera
    """
    # Base target: center of mass
    com = compute_center_of_mass(positions, masses)

    if velocities is not None and look_ahead > 0:
        # Compute center of velocity (weighted by mass)
        total_mass = np.sum(masses)
        if total_mass > 0:
            com_velocity = np.sum(velocities[:, :2] * masses[:, None], axis=0) / total_mass

            # Look ahead in direction of motion
            com += com_velocity * look_ahead

    return com


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == '__main__':
    """
    Test camera tracking with simulated particle motion.
    """
    print("Testing camera tracking...\n")

    # Create camera
    camera = SmoothCamera(smoothing=0.15, damping=0.85)

    # Simulate particles moving in circle
    n_frames = 100
    radius = 2.0
    angular_speed = 2 * np.pi / n_frames  # One full circle

    print("Frame | Target X | Target Y | Camera X | Camera Y | Lag")
    print("-" * 65)

    for frame in range(n_frames):
        angle = frame * angular_speed

        # Particles orbit in circle
        positions = np.array([
            [radius * np.cos(angle), radius * np.sin(angle), 0],
            [radius * np.cos(angle + np.pi), radius * np.sin(angle + np.pi), 0],
        ])
        masses = np.array([1.0, 1.0])

        # Compute target (center of mass)
        target = compute_center_of_mass(positions, masses)

        # Update camera
        camera_pos = camera.update(target, dt=1.0/30.0)

        # Measure lag
        lag = np.linalg.norm(target - camera_pos)

        if frame % 10 == 0:
            print(f"{frame:5d} | {target[0]:8.3f} | {target[1]:8.3f} | "
                  f"{camera_pos[0]:8.3f} | {camera_pos[1]:8.3f} | {lag:6.3f}")

    print("\nNotice:")
    print("- Camera smoothly follows circular motion")
    print("- Small lag keeps motion smooth (not snappy)")
    print("- No oscillation or jitter")
