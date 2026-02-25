"""
Beat-Synchronized Camera Director for V3

Automatically switches between camera modes based on musical beats.
Creates dynamic, music-driven cinematography.

Director Logic:
- Kick drum → zoom in (focus on action)
- Snare → orbit shift (new perspective)
- Build-ups → slow zoom out (anticipation)
- Drops → snap cut to chase mode (excitement)

Usage:
    director = CameraDirector(
        camera_modes={
            'orbit': OrbitMode(),
            'zoom': ZoomMode(),
            'dolly': DollyMode(),
            'chase': ChaseMode(),
        }
    )

    # Update from harmonic frame
    camera_pos = director.update(
        harmonic_frame=frame,
        particles_pos=positions,
        particles_vel=velocities,
        dt=1.0/30.0
    )
"""

import numpy as np
from typing import Dict, Optional
from .camera_modes import CameraMode


class CameraDirector:
    """
    Automatic camera director synchronized to music beats.

    Intelligently switches between camera modes based on musical
    content to create dynamic, engaging cinematography.
    """

    def __init__(
        self,
        camera_modes: Dict[str, CameraMode],
        default_mode: str = 'orbit'
    ):
        """
        Initialize camera director.

        Args:
            camera_modes: Dictionary of camera mode instances
            default_mode: Name of default camera mode
        """
        self.camera_modes = camera_modes
        self.current_mode_name = default_mode
        self.current_mode = camera_modes[default_mode]

        # Mode history (for preventing rapid switching)
        self.mode_history = [default_mode]
        self.mode_switch_cooldown = 0.0
        self.cooldown_duration = 1.0  # 1 second minimum between switches

        # Beat tracking
        self.last_beat_strength = 0.0
        self.beat_count = 0

        # Build-up detection
        self.energy_window = []
        self.window_size = 30  # Track last 30 frames

        # Mode preferences based on music state
        self.mode_weights = {
            'orbit': 1.0,
            'zoom': 1.0,
            'dolly': 1.0,
            'chase': 1.0,
        }

    def update(
        self,
        harmonic_frame,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Update camera director and get camera position.

        Args:
            harmonic_frame: HarmonicFrame from harmonic analyzer
            particles_pos: Particle positions
            particles_vel: Particle velocities
            dt: Time step (seconds)

        Returns:
            Camera position [x, y, z]
        """
        # Update cooldown
        if self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= dt

        # Update energy window for build-up detection
        overall_energy = self._compute_overall_energy(particles_vel)
        self.energy_window.append(overall_energy)
        if len(self.energy_window) > self.window_size:
            self.energy_window.pop(0)

        # =====================================================================
        # DETECT MUSICAL EVENTS AND CHOOSE CAMERA MODE
        # =====================================================================
        new_mode = None

        # 1. Beat Attack Detection
        if harmonic_frame.beat_attack and self.mode_switch_cooldown <= 0:
            beat_strength = harmonic_frame.beat_strength

            # Strong beat (kick drum) → Zoom in
            if beat_strength > 0.8:
                new_mode = 'zoom'
                self.beat_count += 1

            # Medium beat (snare) → Orbit shift
            elif beat_strength > 0.5:
                new_mode = 'orbit'
                self.beat_count += 1

        # 2. Build-Up Detection
        if self._is_building_up() and self.mode_switch_cooldown <= 0:
            # During build-ups, use dolly for tension
            new_mode = 'dolly'

        # 3. Drop Detection (sudden energy increase)
        if self._is_drop(harmonic_frame) and self.mode_switch_cooldown <= 0:
            # On drops, snap to chase mode for excitement
            new_mode = 'chase'

        # 4. High Harmonicity → Orbit (showcase musical content)
        if (harmonic_frame.harmonicity > 0.8 and
            self.beat_count % 4 == 0 and
            self.mode_switch_cooldown <= 0):
            new_mode = 'orbit'

        # =====================================================================
        # SWITCH MODE IF APPROPRIATE
        # =====================================================================
        if new_mode and new_mode in self.camera_modes:
            if new_mode != self.current_mode_name:
                self._switch_mode(new_mode)
                self.mode_switch_cooldown = self.cooldown_duration

        # =====================================================================
        # UPDATE CURRENT CAMERA MODE
        # =====================================================================
        camera_pos = self.current_mode.update(
            particles_pos,
            particles_vel,
            dt
        )

        # Store beat strength for next frame
        self.last_beat_strength = harmonic_frame.beat_strength

        return camera_pos

    def _switch_mode(self, new_mode_name: str):
        """
        Switch to a new camera mode.

        Args:
            new_mode_name: Name of new mode
        """
        self.current_mode_name = new_mode_name
        self.current_mode = self.camera_modes[new_mode_name]
        self.mode_history.append(new_mode_name)

        # Keep history limited
        if len(self.mode_history) > 10:
            self.mode_history.pop(0)

    def _compute_overall_energy(self, particles_vel: np.ndarray) -> float:
        """Compute overall system energy from velocities."""
        if len(particles_vel) == 0:
            return 0.0
        speeds = np.linalg.norm(particles_vel, axis=1)
        return float(speeds.mean())

    def _is_building_up(self) -> bool:
        """Detect if music is building up (gradual energy increase)."""
        if len(self.energy_window) < self.window_size:
            return False

        # Check if energy is consistently increasing
        recent = self.energy_window[-10:]
        older = self.energy_window[-20:-10]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        # Building up if recent energy > older energy by threshold
        return recent_avg > older_avg * 1.2

    def _is_drop(self, harmonic_frame) -> bool:
        """Detect music drop (sudden energy spike)."""
        # Drop = high spectral flux + beat attack + high beat strength
        return (
            harmonic_frame.spectral_flux > 0.7 and
            harmonic_frame.beat_attack and
            harmonic_frame.beat_strength > 0.8
        )

    def get_current_mode_name(self) -> str:
        """Get name of current camera mode."""
        return self.current_mode_name

    def get_mode_history(self) -> list:
        """Get recent mode switch history."""
        return self.mode_history.copy()

    def get_director_stats(self) -> dict:
        """Get director statistics."""
        return {
            'current_mode': self.current_mode_name,
            'total_switches': len(self.mode_history) - 1,
            'beat_count': self.beat_count,
            'mode_history': self.mode_history[-5:],  # Last 5 modes
        }


class TransitionManager:
    """
    Manages smooth transitions between camera positions.

    When camera modes switch, provides smooth blending rather than
    abrupt cuts (optional feature).
    """

    def __init__(self, transition_duration: float = 0.5):
        """
        Initialize transition manager.

        Args:
            transition_duration: Duration of transitions (seconds)
        """
        self.transition_duration = transition_duration
        self.is_transitioning = False
        self.transition_progress = 0.0
        self.start_pos = np.array([0.0, 0.0, 0.0])
        self.end_pos = np.array([0.0, 0.0, 0.0])

    def start_transition(self, from_pos: np.ndarray, to_pos: np.ndarray):
        """
        Start a transition between positions.

        Args:
            from_pos: Starting position
            to_pos: Target position
        """
        self.is_transitioning = True
        self.transition_progress = 0.0
        self.start_pos = from_pos.copy()
        self.end_pos = to_pos.copy()

    def update(self, target_pos: np.ndarray, dt: float) -> np.ndarray:
        """
        Update transition and return blended position.

        Args:
            target_pos: Target position from new mode
            dt: Time step

        Returns:
            Blended camera position
        """
        if not self.is_transitioning:
            return target_pos

        # Advance transition
        self.transition_progress += dt / self.transition_duration

        if self.transition_progress >= 1.0:
            # Transition complete
            self.is_transitioning = False
            return target_pos

        # Smooth interpolation (ease-in-out)
        t = self.transition_progress
        smooth_t = t * t * (3.0 - 2.0 * t)  # Smoothstep

        # Blend positions
        blended = self.start_pos * (1.0 - smooth_t) + target_pos * smooth_t

        return blended
