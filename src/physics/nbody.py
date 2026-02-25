"""
N-body simulation manager for Celestial Chaos.

Handles array allocation, simulation stepping, and state management.
Optimized for GH200 unified memory but CPU-compatible.
"""

import numpy as np
import warp as wp
from typing import Optional, Tuple
from dataclasses import dataclass

from . import kernels


@dataclass
class SimulationConfig:
    """Configuration for N-body simulation."""
    n_bodies: int = 3
    G: float = 1.0  # Gravitational constant (normalized units)
    softening: float = 0.01  # Softening length
    dt: float = 0.001  # Time step
    trail_length: int = 1000  # Trail history length
    device: str = "cuda:0"  # "cuda:0" or "cpu"


class NBodySimulation:
    """
    N-body gravitational simulation using NVIDIA Warp.

    Features:
    - Velocity Verlet integration for energy conservation
    - Particle trail tracking with ring buffer
    - Energy monitoring for validation
    - Audio modulation interface

    Example:
        config = SimulationConfig(n_bodies=3, device="cpu")
        sim = NBodySimulation(config)
        sim.initialize_figure_eight()

        for _ in range(1000):
            sim.step()
            positions = sim.get_positions()
    """

    def __init__(self, config: SimulationConfig):
        """Initialize simulation with given configuration."""
        self.config = config
        self.n = config.n_bodies
        self.time = 0.0

        # Determine device
        if config.device == "cpu" or not wp.is_cuda_available():
            self.device = "cpu"
            if config.device != "cpu":
                print("CUDA not available, falling back to CPU")
        else:
            self.device = config.device

        # Allocate arrays on device
        self._allocate_arrays()

        # Initialize trail tracking
        self.trail_enabled = config.trail_length > 0

    def _allocate_arrays(self):
        """Allocate Warp arrays on the target device."""
        n = self.n
        device = self.device
        trail_len = self.config.trail_length

        # Core simulation arrays
        self.positions = wp.zeros(n, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(n, dtype=wp.vec3, device=device)
        self.accelerations = wp.zeros(n, dtype=wp.vec3, device=device)
        self.masses = wp.zeros(n, dtype=wp.float32, device=device)

        # Energy tracking
        self.kinetic = wp.zeros(n, dtype=wp.float32, device=device)
        self.potential = wp.zeros(n, dtype=wp.float32, device=device)

        # Trail buffers (if enabled)
        if trail_len > 0:
            self.trail_buffer = wp.zeros((n, trail_len), dtype=wp.vec3, device=device)
            self.trail_velocities = wp.zeros((n, trail_len), dtype=wp.vec3, device=device)
            self.trail_head = wp.zeros(n, dtype=wp.int32, device=device)

    def set_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray
    ):
        """
        Set simulation state from numpy arrays.

        Args:
            positions: Shape (n, 3) body positions
            velocities: Shape (n, 3) body velocities
            masses: Shape (n,) body masses
        """
        assert positions.shape == (self.n, 3), f"Expected positions shape ({self.n}, 3)"
        assert velocities.shape == (self.n, 3), f"Expected velocities shape ({self.n}, 3)"
        assert masses.shape == (self.n,), f"Expected masses shape ({self.n},)"

        # Convert numpy arrays to warp vec3 arrays
        # Warp expects list of tuples for vec3 initialization
        pos_list = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        vel_list = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in velocities]

        # Create new warp arrays with data
        self.positions = wp.array(pos_list, dtype=wp.vec3, device=self.device)
        self.velocities = wp.array(vel_list, dtype=wp.vec3, device=self.device)
        self.masses = wp.array(masses.astype(np.float32), dtype=wp.float32, device=self.device)

        # Reset accelerations
        self.accelerations = wp.zeros(self.n, dtype=wp.vec3, device=self.device)

        self.time = 0.0

    def initialize_figure_eight(self, scale: float = 1.0):
        """
        Initialize the classic figure-8 three-body solution.

        Discovered by Moore (1993), proven stable by Chenciner & Montgomery (2000).
        One of few known periodic solutions to the three-body problem.

        Args:
            scale: Spatial scale factor
        """
        if self.n != 3:
            raise ValueError("Figure-8 solution requires exactly 3 bodies")

        # Figure-8 initial conditions (normalized units)
        # Positions on the figure-8 curve
        positions = np.array([
            [-0.97000436, 0.24308753, 0.0],
            [0.97000436, -0.24308753, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32) * scale

        # Velocities (equal masses chase each other)
        v = np.array([0.4662036850, 0.4323657300, 0.0], dtype=np.float32)
        velocities = np.array([
            v / 2,
            v / 2,
            -v
        ], dtype=np.float32)

        # Equal masses
        masses = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self.set_state(positions, velocities, masses)

    def initialize_random(
        self,
        pos_range: float = 10.0,
        vel_range: float = 1.0,
        mass_range: Tuple[float, float] = (0.5, 2.0),
        seed: Optional[int] = None
    ):
        """
        Initialize with random positions and velocities.

        Args:
            pos_range: Position range [-pos_range, pos_range]
            vel_range: Velocity range [-vel_range, vel_range]
            mass_range: Mass range (min, max)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        positions = np.random.uniform(-pos_range, pos_range, (self.n, 3)).astype(np.float32)
        velocities = np.random.uniform(-vel_range, vel_range, (self.n, 3)).astype(np.float32)
        masses = np.random.uniform(mass_range[0], mass_range[1], self.n).astype(np.float32)

        # Center of mass correction (zero total momentum)
        total_mass = masses.sum()
        com_vel = (masses[:, np.newaxis] * velocities).sum(axis=0) / total_mass
        velocities -= com_vel

        self.set_state(positions, velocities, masses)

    def step(self, audio_params: Optional[dict] = None):
        """
        Advance simulation by one time step using Velocity Verlet.

        The Verlet scheme preserves energy better than Euler:
        1. v(t + dt/2) = v(t) + a(t) * dt/2
        2. x(t + dt) = x(t) + v(t + dt/2) * dt
        3. Compute a(t + dt) from new positions
        4. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2

        Args:
            audio_params: Optional dict with bass_energy, mid_energy, treble_energy
        """
        cfg = self.config

        # Step 1-2: Half-step velocity, full-step position
        wp.launch(
            kernels.integrate_verlet,
            dim=self.n,
            inputs=[
                self.positions,
                self.velocities,
                self.accelerations,
                cfg.dt,
                self.n
            ],
            device=self.device
        )

        # Step 3: Compute new accelerations
        wp.launch(
            kernels.compute_accelerations,
            dim=self.n,
            inputs=[
                self.positions,
                self.masses,
                self.accelerations,
                cfg.G,
                cfg.softening,
                self.n
            ],
            device=self.device
        )

        # Optional: Apply audio modulation
        if audio_params is not None:
            wp.launch(
                kernels.apply_audio_modulation,
                dim=self.n,
                inputs=[
                    self.accelerations,
                    audio_params.get('bass_energy', 0.0),
                    audio_params.get('mid_energy', 0.0),
                    audio_params.get('treble_energy', 0.0),
                    audio_params.get('modulation_depth', 0.5),
                    self.n
                ],
                device=self.device
            )

        # Step 4: Finish velocity update
        wp.launch(
            kernels.integrate_verlet_finish,
            dim=self.n,
            inputs=[
                self.velocities,
                self.accelerations,
                cfg.dt,
                self.n
            ],
            device=self.device
        )

        # Update trails
        if self.trail_enabled:
            wp.launch(
                kernels.update_trails,
                dim=self.n,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.trail_buffer,
                    self.trail_velocities,
                    self.trail_head,
                    self.config.trail_length,
                    self.n
                ],
                device=self.device
            )

        self.time += cfg.dt

    def get_positions(self) -> np.ndarray:
        """Get current positions as numpy array (n, 3)."""
        return self.positions.numpy()

    def get_velocities(self) -> np.ndarray:
        """Get current velocities as numpy array (n, 3)."""
        return self.velocities.numpy()

    def get_trails(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trail history as numpy arrays.

        Returns:
            positions: Shape (n, trail_length, 3)
            velocities: Shape (n, trail_length, 3)
        """
        if not self.trail_enabled:
            raise ValueError("Trails not enabled in config")

        return self.trail_buffer.numpy(), self.trail_velocities.numpy()
        return pos, vel

    def compute_total_energy(self) -> Tuple[float, float, float]:
        """
        Compute and return total system energy.

        Returns:
            (kinetic, potential, total) energy
        """
        wp.launch(
            kernels.compute_energy,
            dim=self.n,
            inputs=[
                self.positions,
                self.velocities,
                self.masses,
                self.kinetic,
                self.potential,
                self.config.G,
                self.n
            ],
            device=self.device
        )

        ke = float(self.kinetic.numpy().sum())
        pe = float(self.potential.numpy().sum())
        return ke, pe, ke + pe

    def get_device_info(self) -> dict:
        """Get information about the compute device."""
        info = {
            'device': self.device,
            'cuda_available': wp.is_cuda_available(),
            'n_bodies': self.n,
            'trail_length': self.config.trail_length if self.trail_enabled else 0
        }

        if wp.is_cuda_available() and self.device != "cpu":
            # Get CUDA device properties if available
            try:
                props = wp.get_cuda_device_properties(0)
                info['gpu_name'] = props['name']
                info['gpu_memory_gb'] = props['total_global_mem'] / (1024**3)
            except:
                pass

        return info
