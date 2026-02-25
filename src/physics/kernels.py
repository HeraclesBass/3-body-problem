"""
NVIDIA Warp kernels for N-body gravitational simulation.

Optimized for GH200 Grace Hopper unified memory architecture.
CPU-compatible for development/testing.
"""

import warp as wp

# Initialize Warp (auto-detects CUDA availability)
wp.init()


@wp.kernel
def compute_accelerations(
    positions: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    accelerations: wp.array(dtype=wp.vec3),
    G: wp.float32,
    softening: wp.float32,
    n: wp.int32
):
    """
    Compute gravitational accelerations for all bodies.

    O(n²) direct summation - suitable for up to ~100k particles on GH200.
    Uses softening parameter to prevent singularities at close approach.

    Args:
        positions: Body positions (vec3 array)
        masses: Body masses (float32 array)
        accelerations: Output accelerations (vec3 array)
        G: Gravitational constant
        softening: Softening length to prevent singularities
        n: Number of bodies
    """
    i = wp.tid()
    if i >= n:
        return

    pos_i = positions[i]
    acc = wp.vec3(0.0, 0.0, 0.0)

    for j in range(n):
        if i != j:
            pos_j = positions[j]
            r = pos_j - pos_i
            dist_sq = wp.dot(r, r) + softening * softening
            dist = wp.sqrt(dist_sq)
            # F = G * m_i * m_j / r² → a = G * m_j / r² * r_hat
            acc += G * masses[j] * r / (dist_sq * dist)

    accelerations[i] = acc


@wp.kernel
def integrate_verlet(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    n: wp.int32
):
    """
    Velocity Verlet integration (first half-step).

    v(t + dt/2) = v(t) + a(t) * dt/2
    x(t + dt) = x(t) + v(t + dt/2) * dt

    Args:
        positions: Body positions (updated in place)
        velocities: Body velocities (updated in place)
        accelerations: Current accelerations
        dt: Time step
        n: Number of bodies
    """
    i = wp.tid()
    if i >= n:
        return

    # Half-step velocity update
    velocities[i] = velocities[i] + accelerations[i] * (dt * 0.5)
    # Full-step position update
    positions[i] = positions[i] + velocities[i] * dt


@wp.kernel
def integrate_verlet_finish(
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    n: wp.int32
):
    """
    Velocity Verlet integration (second half-step).

    v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2

    Called after recomputing accelerations at new positions.

    Args:
        velocities: Body velocities (updated in place)
        accelerations: New accelerations at updated positions
        dt: Time step
        n: Number of bodies
    """
    i = wp.tid()
    if i >= n:
        return

    velocities[i] = velocities[i] + accelerations[i] * (dt * 0.5)


@wp.kernel
def update_trails(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    trail_buffer: wp.array(dtype=wp.vec3, ndim=2),
    trail_velocities: wp.array(dtype=wp.vec3, ndim=2),
    trail_head: wp.array(dtype=wp.int32),
    trail_length: wp.int32,
    n: wp.int32
):
    """
    Update particle trail ring buffer.

    Stores position and velocity history for trail rendering.
    Velocity stored for color mapping (speed → hue).

    Args:
        positions: Current body positions
        velocities: Current body velocities
        trail_buffer: Ring buffer [n_bodies, trail_length] positions
        trail_velocities: Ring buffer [n_bodies, trail_length] velocities
        trail_head: Current write index per body
        trail_length: Maximum trail length
        n: Number of bodies
    """
    i = wp.tid()
    if i >= n:
        return

    head = trail_head[i]
    trail_buffer[i, head] = positions[i]
    trail_velocities[i, head] = velocities[i]
    trail_head[i] = (head + 1) % trail_length


@wp.kernel
def compute_energy(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    kinetic: wp.array(dtype=wp.float32),
    potential: wp.array(dtype=wp.float32),
    G: wp.float32,
    n: wp.int32
):
    """
    Compute kinetic and potential energy per body.

    Used for energy conservation validation and visual effects.

    Args:
        positions: Body positions
        velocities: Body velocities
        masses: Body masses
        kinetic: Output kinetic energy per body
        potential: Output potential energy per body
        G: Gravitational constant
        n: Number of bodies
    """
    i = wp.tid()
    if i >= n:
        return

    # Kinetic: 0.5 * m * v²
    v = velocities[i]
    kinetic[i] = 0.5 * masses[i] * wp.dot(v, v)

    # Potential: -G * m_i * m_j / r (sum over j > i to avoid double counting)
    pot = wp.float32(0.0)
    for j in range(i + 1, n):
        r = positions[j] - positions[i]
        dist = wp.length(r)
        if dist > 0.0:
            pot -= G * masses[i] * masses[j] / dist

    potential[i] = pot


@wp.kernel
def apply_audio_modulation(
    accelerations: wp.array(dtype=wp.vec3),
    bass_energy: wp.float32,
    mid_energy: wp.float32,
    treble_energy: wp.float32,
    modulation_depth: wp.float32,
    n: wp.int32
):
    """
    Modulate accelerations based on audio energy bands.

    Bass → gravity strength
    Mid → orbital perturbation
    Treble → velocity damping/excitation

    Args:
        accelerations: Body accelerations (modified in place)
        bass_energy: Low frequency energy (20-250 Hz)
        mid_energy: Mid frequency energy (250-4000 Hz)
        treble_energy: High frequency energy (4000-20000 Hz)
        modulation_depth: Overall modulation intensity
        n: Number of bodies
    """
    i = wp.tid()
    if i >= n:
        return

    # Bass amplifies gravity
    gravity_mod = 1.0 + bass_energy * modulation_depth

    # Apply modulation
    accelerations[i] = accelerations[i] * gravity_mod
