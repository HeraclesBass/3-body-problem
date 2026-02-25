# The Three-Body Problem

## Historical Context

> "A juicy, juicy problem, which occupied literally generations, hundreds and hundreds of years of incredibly ambitious, talented mathematicians, physicists, and astronomers."

### Timeline

| Year | Contributor | Achievement |
|------|-------------|-------------|
| ~1687 | Newton | Solved two-body problem completely |
| 1770s | Lagrange | Developed potential approach, found Lagrange points |
| 1887 | Bruns | Proved three-body problem is analytically unsolvable |
| Modern | Computers | Numerical simulation is the only practical approach |

## Why Two Bodies Are Easy

In the two-body problem:
- Forces always point toward the shared center of mass
- Motion is predictable (ellipses, parabolas, hyperbolas)
- Closed-form solution exists

## Why Three Bodies Break Everything

With three bodies:
- Forces are "extremely dynamic"
- Must track both **magnitude AND direction** of all forces
- Results in "chaotic mess of vectors"
- No closed-form solution possible (proven by Bruns, 1887)

## The Equations

### Lagrangian Formulation
```
L = T - V

T = ½m₁(ẋ₁² + ẏ₁² + ż₁²) + ½m₂(ẋ₂² + ẏ₂² + ż₂²) + ½m₃(ẋ₃² + ẏ₃² + ż₃²)

V = -G(m₁m₂/d₁₂ + m₁m₃/d₁₃ + m₂m₃/d₂₃)
```

Where:
- `dᵢⱼ = √[(xⱼ-xᵢ)² + (yⱼ-yᵢ)² + (zⱼ-zᵢ)²]`

### Equations of Motion (9 coupled differential equations)

For mass 1 (x-component):
```
m₁ẍ₁ = Gm₁m₂(x₂-x₁)/d₁₂³ + Gm₁m₃(x₃-x₁)/d₁₃³
```

Similar equations for y₁, z₁, and all components of masses 2 and 3.

## Lagrange Points

When Lagrange studied the combined potential of two bodies (e.g., Sun-Earth), he found 5 points where the gradient is zero:

| Point | Location | Stability |
|-------|----------|-----------|
| L1 | Between bodies | Unstable |
| L2 | Beyond smaller body | Unstable |
| L3 | Beyond larger body | Unstable |
| L4 | 60° ahead in orbit | Stable |
| L5 | 60° behind in orbit | Stable |

**Visual opportunity**: Show how a tiny third body placed at these points maintains stable orbit.

## Chaos and Beauty

The three-body problem exhibits:
- **Sensitive dependence on initial conditions** - tiny changes → wildly different outcomes
- **Strange attractors** - patterns emerge from chaos
- **Periodic orbits** - rare but beautiful solutions exist

### Known Periodic Solutions
- Figure-8 orbit (discovered 1993)
- Choreographic solutions (all bodies trace same path)
- Various resonance orbits

## Visualization Concepts

1. **Potential Landscape** - 3D terrain showing combined gravitational wells
2. **Trajectory Traces** - Glowing trails following each body
3. **Chaos Visualization** - Show divergence of nearby initial conditions
4. **Lagrange Point Stability** - Animate test particles near each point

## Computational Approach

Since analytical solutions don't exist:
```python
# Numerical integration (simplified)
def step(positions, velocities, masses, dt):
    forces = compute_gravitational_forces(positions, masses)
    velocities += forces * dt / masses
    positions += velocities * dt
    return positions, velocities
```

### Integration Methods
- **Euler** (simple, inaccurate)
- **Runge-Kutta 4** (good balance)
- **Verlet** (energy-conserving, good for long simulations)
- **Symplectic integrators** (best for orbital mechanics)

## Why It Matters for Art

The three-body problem is the perfect subject for computational art because:
1. **Impossible to predict** - every simulation is unique
2. **Visually stunning** - chaotic but structured patterns
3. **Physically real** - not abstract, represents actual celestial mechanics
4. **Philosophically deep** - determinism vs unpredictability
