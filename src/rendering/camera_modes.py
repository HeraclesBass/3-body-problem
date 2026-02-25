"""
Dynamic Camera Modes for V3

Implements cinematic camera movements beyond V2's simple tracking:
- Orbit: Circles center of mass
- Zoom: Smooth in/out based on particle spread
- Dolly: Forward/back motion along view axis
- Chase: Follows fastest-moving particle

All modes use spring-damped physics for smooth, organic motion.

Usage:
    # Create camera mode
    orbit = OrbitMode(radius=100, speed=0.5)

    # Update camera
    camera_pos = orbit.update(
        particles_pos=positions,
        particles_vel=velocities,
        dt=1.0/30.0
    )
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class CameraMode(ABC):
    """Base class for camera modes."""

    def __init__(self):
        """Initialize camera mode."""
        self.position = np.array([0.0, 0.0, 100.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.target = np.array([0.0, 0.0, 0.0])

        # Spring physics parameters
        self.spring_strength = 0.1
        self.damping = 0.85

    @abstractmethod
    def update(
        self,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Update camera position.

        Args:
            particles_pos: Array of shape (N, 3) with particle positions
            particles_vel: Array of shape (N, 3) with particle velocities
            dt: Time step (seconds)

        Returns:
            Camera position [x, y, z]
        """
        pass

    def _compute_center_of_mass(self, particles_pos: np.ndarray) -> np.ndarray:
        """Compute center of mass of particles."""
        if len(particles_pos) == 0:
            return np.array([0.0, 0.0, 0.0])
        return particles_pos.mean(axis=0)

    def _compute_spread(self, particles_pos: np.ndarray) -> float:
        """Compute spread (standard deviation) of particles."""
        if len(particles_pos) == 0:
            return 1.0
        center = self._compute_center_of_mass(particles_pos)
        distances = np.linalg.norm(particles_pos - center, axis=1)
        return float(np.std(distances)) if len(distances) > 0 else 1.0


class OrbitMode(CameraMode):
    """
    Camera orbits around center of mass.

    Creates circular motion for dynamic viewing angle.
    """

    def __init__(
        self,
        radius: float = 100.0,
        speed: float = 0.3,
        elevation: float = 30.0
    ):
        """
        Initialize orbit mode.

        Args:
            radius: Orbit radius
            speed: Orbit speed (radians per second)
            elevation: Elevation angle in degrees
        """
        super().__init__()
        self.radius = radius
        self.speed = speed
        self.elevation_deg = elevation
        self.elevation_rad = np.radians(elevation)

        # Current angle
        self.angle = 0.0

    def update(
        self,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Update camera in orbit around center of mass."""
        # Get center of mass
        center = self._compute_center_of_mass(particles_pos)

        # Advance angle
        self.angle += self.speed * dt
        self.angle = self.angle % (2 * np.pi)

        # Calculate target position
        x = center[0] + self.radius * np.cos(self.angle)
        y = center[1] + self.radius * np.sin(self.angle)
        z = center[2] + self.radius * np.sin(self.elevation_rad)

        self.target = np.array([x, y, z])

        # Spring toward target
        force = (self.target - self.position) * self.spring_strength
        self.velocity += force
        self.velocity *= self.damping
        self.position += self.velocity * dt * 30.0

        return self.position.copy()


class ZoomMode(CameraMode):
    """
    Camera zooms in/out based on particle spread.

    Zooms in when particles clustered, out when spread apart.
    """

    def __init__(
        self,
        base_distance: float = 100.0,
        zoom_factor: float = 2.0
    ):
        """
        Initialize zoom mode.

        Args:
            base_distance: Base distance from center
            zoom_factor: How much to zoom (multiplier on spread)
        """
        super().__init__()
        self.base_distance = base_distance
        self.zoom_factor = zoom_factor

        # Current zoom level
        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.zoom_velocity = 0.0

    def update(
        self,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Update camera zoom based on particle spread."""
        # Get center and spread
        center = self._compute_center_of_mass(particles_pos)
        spread = self._compute_spread(particles_pos)

        # Calculate target zoom level
        # Larger spread → zoom out (larger multiplier)
        self.target_zoom = 1.0 + spread / (self.base_distance * 0.5)
        self.target_zoom = np.clip(self.target_zoom, 0.5, 3.0)

        # Smooth zoom transition
        zoom_force = (self.target_zoom - self.current_zoom) * self.spring_strength
        self.zoom_velocity += zoom_force
        self.zoom_velocity *= self.damping
        self.current_zoom += self.zoom_velocity * dt * 30.0

        # Calculate camera position
        distance = self.base_distance * self.current_zoom
        self.target = center + np.array([0, 0, distance])

        # Spring toward target
        force = (self.target - self.position) * self.spring_strength
        self.velocity += force
        self.velocity *= self.damping
        self.position += self.velocity * dt * 30.0

        return self.position.copy()


class DollyMode(CameraMode):
    """
    Camera moves forward/back along view axis.

    Creates cinematic push/pull effect.
    """

    def __init__(
        self,
        base_distance: float = 100.0,
        dolly_speed: float = 20.0
    ):
        """
        Initialize dolly mode.

        Args:
            base_distance: Base distance from center
            dolly_speed: Speed of dolly movement
        """
        super().__init__()
        self.base_distance = base_distance
        self.dolly_speed = dolly_speed

        # Dolly position (0 = base distance)
        self.dolly_position = 0.0
        self.dolly_direction = 1.0  # 1=forward, -1=backward

    def update(
        self,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Update camera dolly movement."""
        # Get center
        center = self._compute_center_of_mass(particles_pos)

        # Move dolly position
        self.dolly_position += self.dolly_direction * self.dolly_speed * dt

        # Reverse direction at limits
        if self.dolly_position > 50.0:
            self.dolly_direction = -1.0
        elif self.dolly_position < -50.0:
            self.dolly_direction = 1.0

        # Calculate camera position
        distance = self.base_distance + self.dolly_position
        self.target = center + np.array([0, 0, distance])

        # Spring toward target
        force = (self.target - self.position) * self.spring_strength
        self.velocity += force
        self.velocity *= self.damping
        self.position += self.velocity * dt * 30.0

        return self.position.copy()


class ChaseMode(CameraMode):
    """
    Camera follows fastest-moving particle.

    Creates dynamic, action-focused framing.
    """

    def __init__(
        self,
        follow_distance: float = 80.0,
        follow_offset: np.ndarray = np.array([0, 0, 30.0])
    ):
        """
        Initialize chase mode.

        Args:
            follow_distance: Distance to maintain from target
            follow_offset: Offset from target position
        """
        super().__init__()
        self.follow_distance = follow_distance
        self.follow_offset = follow_offset

        # Currently tracked particle index
        self.tracked_particle_idx = 0

    def update(
        self,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Update camera to chase fastest particle."""
        if len(particles_pos) == 0:
            return self.position.copy()

        # Find fastest-moving particle
        speeds = np.linalg.norm(particles_vel, axis=1)
        fastest_idx = np.argmax(speeds)

        # Get target particle position
        target_particle = particles_pos[fastest_idx]

        # Calculate camera target position (behind and above)
        if speeds[fastest_idx] > 0.1:
            # Follow behind velocity vector
            direction = particles_vel[fastest_idx] / (speeds[fastest_idx] + 1e-6)
            self.target = target_particle - direction * self.follow_distance + self.follow_offset
        else:
            # Static offset if particle not moving
            self.target = target_particle + self.follow_offset

        # Spring toward target
        force = (self.target - self.position) * self.spring_strength
        self.velocity += force
        self.velocity *= self.damping
        self.position += self.velocity * dt * 30.0

        self.tracked_particle_idx = fastest_idx
        return self.position.copy()


class SmoothTrackingMode(CameraMode):
    """
    Enhanced version of V2's tracking camera.

    Follows center of mass with smooth spring physics.
    This is the baseline mode for V3.
    """

    def __init__(
        self,
        distance: float = 100.0,
        offset: np.ndarray = np.array([0, 0, 0])
    ):
        """
        Initialize smooth tracking mode.

        Args:
            distance: Distance from center of mass
            offset: Offset from center position
        """
        super().__init__()
        self.distance = distance
        self.offset = offset

    def update(
        self,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Update camera to smoothly track center of mass."""
        # Get center of mass
        center = self._compute_center_of_mass(particles_pos)

        # Target position: center + offset + distance along Z
        self.target = center + self.offset + np.array([0, 0, self.distance])

        # Spring toward target
        force = (self.target - self.position) * self.spring_strength
        self.velocity += force
        self.velocity *= self.damping
        self.position += self.velocity * dt * 30.0

        return self.position.copy()
