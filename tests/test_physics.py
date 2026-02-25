#!/usr/bin/env python3
"""
Test script for N-body physics simulation.

Run on CPU to verify kernels work before deploying to GH200.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics.nbody import NBodySimulation, SimulationConfig


def test_figure_eight():
    """Test the classic figure-8 three-body solution."""
    print("\n" + "=" * 60)
    print("TEST: Figure-8 Three-Body Solution")
    print("=" * 60)

    # Configure for CPU testing
    config = SimulationConfig(
        n_bodies=3,
        G=1.0,
        softening=0.01,
        dt=0.0001,  # Small timestep for accuracy
        trail_length=100,
        device="cpu"
    )

    sim = NBodySimulation(config)
    sim.initialize_figure_eight()

    # Get initial state
    pos0 = sim.get_positions().copy()
    ke0, pe0, E0 = sim.compute_total_energy()

    print(f"\nInitial positions:\n{pos0}")
    print(f"\nInitial energy: KE={ke0:.6f}, PE={pe0:.6f}, Total={E0:.6f}")

    # Run for many steps
    n_steps = 10000
    print(f"\nRunning {n_steps} steps...")

    for i in range(n_steps):
        sim.step()
        if (i + 1) % 2500 == 0:
            ke, pe, E = sim.compute_total_energy()
            drift = abs(E - E0) / abs(E0) * 100
            print(f"  Step {i+1}: Energy drift = {drift:.6f}%")

    # Check final state
    pos_final = sim.get_positions()
    ke_f, pe_f, E_f = sim.compute_total_energy()

    print(f"\nFinal positions:\n{pos_final}")
    print(f"Final energy: KE={ke_f:.6f}, PE={pe_f:.6f}, Total={E_f:.6f}")

    # Energy conservation check
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    print(f"\nTotal energy drift: {energy_drift:.4f}%")

    if energy_drift < 1.0:
        print("✓ PASS: Energy conserved within 1%")
        return True
    else:
        print("✗ FAIL: Energy drift too large")
        return False


def test_random_nbody():
    """Test N-body with random initial conditions."""
    print("\n" + "=" * 60)
    print("TEST: Random N-body (10 particles)")
    print("=" * 60)

    config = SimulationConfig(
        n_bodies=10,
        G=1.0,
        softening=0.5,
        dt=0.01,
        trail_length=50,
        device="cpu"
    )

    sim = NBodySimulation(config)
    sim.initialize_random(pos_range=10.0, vel_range=0.5, seed=42)

    ke0, pe0, E0 = sim.compute_total_energy()
    print(f"Initial energy: {E0:.6f}")

    # Run simulation
    n_steps = 500
    positions_history = []

    for i in range(n_steps):
        sim.step()
        if i % 50 == 0:
            positions_history.append(sim.get_positions().copy())

    ke_f, pe_f, E_f = sim.compute_total_energy()
    print(f"Final energy: {E_f:.6f}")

    energy_drift = abs(E_f - E0) / abs(E0) * 100 if E0 != 0 else 0
    print(f"Energy drift: {energy_drift:.4f}%")

    print(f"✓ Simulation completed successfully")
    return True


def test_device_info():
    """Display device information."""
    print("\n" + "=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)

    config = SimulationConfig(n_bodies=3, device="cpu")
    sim = NBodySimulation(config)

    info = sim.get_device_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    return True


def visualize_figure_eight():
    """Visualize the figure-8 solution."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Figure-8 Orbit")
    print("=" * 60)

    config = SimulationConfig(
        n_bodies=3,
        G=1.0,
        softening=0.01,
        dt=0.0005,
        trail_length=0,  # We'll track manually
        device="cpu"
    )

    sim = NBodySimulation(config)
    sim.initialize_figure_eight()

    # Collect trajectory
    n_steps = 20000
    trajectory = [[] for _ in range(3)]

    print(f"Simulating {n_steps} steps...")
    for i in range(n_steps):
        sim.step()
        if i % 10 == 0:  # Sample every 10 steps
            pos = sim.get_positions()
            for j in range(3):
                trajectory[j].append(pos[j].copy())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    for j in range(3):
        traj = np.array(trajectory[j])
        ax.plot(traj[:, 0], traj[:, 1], color=colors[j], alpha=0.7, linewidth=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[j], s=100, zorder=5)

    ax.set_aspect('equal')
    ax.set_title("Figure-8 Three-Body Solution", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    output_path = Path(__file__).parent.parent / "outputs" / "figure_eight.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# CELESTIAL CHAOS - Physics Engine Tests")
    print("# Running on CPU (GH200 deployment pending)")
    print("#" * 60)

    tests = [
        ("Device Info", test_device_info),
        ("Figure-8 Solution", test_figure_eight),
        ("Random N-body", test_random_nbody),
        ("Visualization", visualize_figure_eight),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
