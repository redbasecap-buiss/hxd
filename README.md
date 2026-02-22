# FLUX 🌊

**State-of-the-art Lattice Boltzmann Method (LBM) fluid dynamics solver in pure Rust — university-grade CFD.**

[![CI](https://github.com/redbasecap-buiss/flux/actions/workflows/ci.yml/badge.svg)](https://github.com/redbasecap-buiss/flux/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What is this?

FLUX implements the Lattice Boltzmann Method for computational fluid dynamics. Unlike traditional Navier-Stokes solvers, LBM evolves particle distribution functions on a discrete lattice, making it naturally parallelizable and well-suited for complex geometries.

## Theory

### The Lattice Boltzmann Equation

The LBM evolves distribution functions $f_i(\mathbf{x}, t)$ according to:

$$f_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) = f_i(\mathbf{x}, t) + \Omega_i$$

where $\mathbf{e}_i$ are the discrete velocities and $\Omega_i$ is the collision operator.

### Equilibrium Distribution

The Maxwell-Boltzmann equilibrium distribution (truncated to 2nd order):

$$f_i^{eq} = w_i \rho \left(1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u})^2}{2c_s^4} - \frac{\mathbf{u} \cdot \mathbf{u}}{2c_s^2}\right)$$

### BGK Collision Operator

Single relaxation time (Bhatnagar-Gross-Krook):

$$\Omega_i^{BGK} = -\frac{1}{\tau}(f_i - f_i^{eq})$$

where $\tau$ is related to kinematic viscosity: $\nu = c_s^2(\tau - \tfrac{1}{2})$

### MRT Collision Operator

Multiple Relaxation Time transforms to moment space for independent relaxation:

$$\Omega^{MRT} = -\mathbf{M}^{-1}\mathbf{S}(\mathbf{m} - \mathbf{m}^{eq})$$

where $\mathbf{M}$ is the transformation matrix and $\mathbf{S}$ is the diagonal relaxation matrix.

### Lattice Models

| Model | Dimensions | Velocities | Use Case |
|-------|-----------|------------|----------|
| D2Q9  | 2D        | 9          | Standard 2D flows |
| D3Q19 | 3D        | 19         | 3D simulations |

## Features

- **D2Q9** and **D3Q19** lattice models
- **BGK** and **MRT** collision operators
- **Smagorinsky** subgrid model for Large Eddy Simulation (LES)
- **Boundary conditions**: bounce-back, Zou-He, periodic, open, moving walls
- **Geometry**: circles, spheres, rectangles, STL import, signed distance fields
- **Output**: VTK (ParaView), CSV, PPM image
- **Parallelism**: Rayon-based shared memory
- **Checkpoint/restart** support
- **Validation suite** against analytical solutions

## Installation

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
```

## Usage

### Generate example config

```bash
flux init
```

### Run a simulation

```bash
flux run flux.toml --output results/
```

### Performance benchmark

```bash
flux benchmark -n 256 --steps 1000
```

### Run validation suite

```bash
flux validate
```

## Configuration

```toml
[domain]
nx = 256
ny = 256
nz = 1  # 1 = 2D mode

[physics]
reynolds = 100
mach = 0.1

[solver]
collision = "bgk"  # or "mrt"
max_steps = 50000
convergence_threshold = 1e-6
checkpoint_interval = 1000

[output]
format = "vtk"
interval = 100
fields = ["velocity", "density", "vorticity"]

[boundary]
north = "wall"
south = "wall"
east = "pressure"
west = "velocity"
```

## Validation Results

FLUX has been validated against classical CFD benchmarks:

| Test Case | Reference | Error | Status |
|-----------|-----------|-------|--------|
| Poiseuille Flow | Analytical parabolic profile | < 1% | ✅ |
| Couette Flow | Analytical linear profile | < 1% | ✅ |
| Lid-Driven Cavity | Ghia et al. (1982) | < 10% avg | ✅ |
| Taylor-Green Vortex | Analytical energy decay | < 5% | ✅ |
| Convergence Order | 2nd order spatial | > 1.5 | ✅ |

## Architecture

```
src/
├── main.rs        # CLI interface
├── lattice.rs     # D2Q9/D3Q19 models, collision, streaming
├── boundary.rs    # Boundary conditions (bounce-back, Zou-He, etc.)
├── geometry.rs    # Domain geometry, obstacles, STL import
├── physics.rs     # Unit conversion, Reynolds/Mach numbers
├── output.rs      # VTK, CSV, PPM output
├── solver.rs      # Simulation engine, config, checkpointing
├── parallel.rs    # Rayon parallelism utilities
└── validation.rs  # Validation test cases
```

## References

1. **Krüger, T. et al.** (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer.
2. **Ghia, U., Ghia, K.N., & Shin, C.T.** (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.
3. **Zou, Q. & He, X.** (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591-1598.
4. **Lallemand, P. & Luo, L.-S.** (2000). Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability. *Physical Review E*, 61(6), 6546.

## License

MIT
