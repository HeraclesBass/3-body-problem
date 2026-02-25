"""
10-Band Audio Modulation for Physics

Maps frequency bands to physics parameters for maximum reactivity.
"""

import numpy as np


def compute_audio_physics_params(audio_frame):
    """
    Convert 10-band audio to physics parameters.

    Mapping:
    - Sub-bass (20-60 Hz) → Gravitational constant
    - Bass (60-150 Hz) → Acceleration bursts
    - Low-mid (150-300 Hz) → Trail decay rate
    - Mid (300-600 Hz) → Particle spawn/chaos
    - High-mid (600-1200 Hz) → Orbital perturbation
    - Presence (1.2k-2.5k Hz) → Energy injection
    - Brilliance (2.5k-5k Hz) → Velocity damping
    - Air (5k-10k Hz) → Particle shimmer
    - Ultra (10k-16k Hz) → Background energy
    - Extreme (16k-20k Hz) → Quantum fluctuations

    Args:
        audio_frame: AudioFrame10Band

    Returns:
        dict of physics parameters
    """

    # Base gravity modulation (sub-bass)
    gravity_mod = 1.0 + audio_frame.sub_bass * 2.0

    # Acceleration bursts on bass hits
    accel_boost = 1.0 + audio_frame.bass * 3.0

    # Orbital chaos (high-mid)
    chaos_factor = audio_frame.high_mid

    # Energy injection (presence + brilliance)
    energy_injection = (audio_frame.presence + audio_frame.brilliance) / 2

    # Damping (air frequencies)
    velocity_damping = 1.0 - audio_frame.air * 0.3

    # Beat perturbation
    beat_kick = audio_frame.beat_strength * 2.0

    return {
        'gravity_multiplier': gravity_mod,
        'acceleration_boost': accel_boost,
        'chaos_factor': chaos_factor,
        'energy_injection': energy_injection,
        'velocity_damping': velocity_damping,
        'beat_perturbation': beat_kick,

        # For rendering
        'bass_energy': audio_frame.bass,
        'mid_energy': audio_frame.mid,
        'treble_energy': audio_frame.brilliance,
        'glow_intensity': audio_frame.presence,
        'trail_brightness': audio_frame.brilliance,
        'shimmer': audio_frame.air,
        'sparkle': audio_frame.extreme,
    }


def apply_audio_to_acceleration(accelerations, audio_params):
    """
    Apply audio-reactive modulation to accelerations.

    Args:
        accelerations: numpy array (n, 3) of particle accelerations
        audio_params: dict from compute_audio_physics_params

    Returns:
        Modified accelerations
    """
    # Gravity modulation (all particles)
    acc = accelerations * audio_params['gravity_multiplier']

    # Acceleration boost
    acc *= audio_params['acceleration_boost']

    # Beat perturbation (random kick)
    if audio_params['beat_perturbation'] > 0.5:
        # Random perturbation on beat
        n = len(accelerations)
        kick = np.random.randn(n, 3) * 0.1 * audio_params['beat_perturbation']
        acc += kick

    # Chaos injection (high-mid)
    if audio_params['chaos_factor'] > 0.3:
        n = len(accelerations)
        chaos = np.random.randn(n, 3) * 0.05 * audio_params['chaos_factor']
        acc += chaos

    return acc


def apply_audio_to_velocity(velocities, audio_params, dt):
    """
    Apply audio-reactive damping/boosting to velocities.

    Args:
        velocities: numpy array (n, 3)
        audio_params: dict
        dt: time step

    Returns:
        Modified velocities
    """
    # Velocity damping (air frequencies)
    vel = velocities * audio_params['velocity_damping']

    # Energy injection (presence)
    if audio_params['energy_injection'] > 0.5:
        # Boost velocity magnitude slightly
        speeds = np.linalg.norm(vel, axis=1, keepdims=True)
        directions = vel / (speeds + 1e-6)
        boosted_speeds = speeds * (1.0 + audio_params['energy_injection'] * 0.1)
        vel = directions * boosted_speeds

    return vel
