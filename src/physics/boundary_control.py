"""
Boundary Control for Keeping Particles In-Frame

Solutions:
1. Soft restraining force (gentle push toward center)
2. Periodic boundary conditions (wrap-around)
3. Dynamic camera tracking
4. Elastic boundaries

We use #1 (soft restraining) for organic, non-jarring behavior.
"""

import numpy as np


def apply_soft_boundary(positions, velocities, accelerations, boundary_radius=2.5, strength=0.5):
    """
    Apply soft restraining force to keep particles in view.

    Particles beyond boundary_radius experience gentle push toward origin.
    Force increases smoothly with distance (not harsh boundary).

    Args:
        positions: (n, 3) positions
        velocities: (n, 3) velocities
        accelerations: (n, 3) accelerations (modified in-place)
        boundary_radius: Distance where soft force begins
        strength: Force strength multiplier

    Returns:
        Modified accelerations
    """
    n = len(positions)

    for i in range(n):
        pos = positions[i]
        dist_2d = np.sqrt(pos[0]**2 + pos[1]**2)  # Distance in viewing plane

        if dist_2d > boundary_radius:
            # Distance beyond boundary
            excess = dist_2d - boundary_radius

            # Soft restoring force (proportional to excess distance)
            force_magnitude = -strength * excess

            # Direction toward origin
            direction = pos / (dist_2d + 1e-6)

            # Apply force (only in x-y plane for viewing)
            restoring_force = direction * force_magnitude
            restoring_force[2] = 0  # Don't constrain z

            accelerations[i] += restoring_force

    return accelerations


def apply_velocity_damping(velocities, positions, boundary_radius=2.5, damping=0.02):
    """
    Gently damp velocities of particles near boundary.

    Prevents particles from escaping to infinity.
    """
    n = len(velocities)

    for i in range(n):
        dist_2d = np.sqrt(positions[i, 0]**2 + positions[i, 1]**2)

        if dist_2d > boundary_radius:
            # Progressive damping (stronger further out)
            excess = dist_2d - boundary_radius
            damp_factor = 1.0 - min(damping * excess, 0.5)

            velocities[i] *= damp_factor

    return velocities


def get_dynamic_view_bounds(positions, padding=0.5):
    """
    Calculate optimal view bounds to keep all particles visible.

    Alternative to soft boundaries - camera follows the action.
    """
    if len(positions) == 0:
        return (-3, 3, -1.7, 1.7)

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    # Enforce aspect ratio (16:9)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    width = x_max - x_min
    height = y_max - y_min

    target_aspect = 16 / 9
    current_aspect = width / height

    if current_aspect > target_aspect:
        # Too wide, increase height
        new_height = width / target_aspect
        y_min = center_y - new_height / 2
        y_max = center_y + new_height / 2
    else:
        # Too tall, increase width
        new_width = height * target_aspect
        x_min = center_x - new_width / 2
        x_max = center_x + new_width / 2

    return (x_min, x_max, y_min, y_max)
