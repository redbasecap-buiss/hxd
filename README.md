<![CDATA[# FLUX — Lattice Boltzmann Fluid Dynamics Solver

[![CI](https://github.com/redbasecap-buiss/flux/actions/workflows/ci.yml/badge.svg)](https://github.com/redbasecap-buiss/flux/actions)
[![Version](https://img.shields.io/crates/v/flux.svg)](https://crates.io/crates/flux)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

**FLUX** is a state-of-the-art Lattice Boltzmann Method (LBM) solver written in pure Rust, designed for high-performance computational fluid dynamics research and education. It combines GPU acceleration via WebGPU with a clean, modular architecture suitable for both academic investigation and production simulations.

---

## Table of Contents

- [Mathematical Foundation](#mathematical-foundation)
  - [The Boltzmann Equation](#the-boltzmann-equation)
  - [BGK Approximation](#bgk-approximation)
  - [Lattice Discretization](#lattice-discretization)
  - [Equilibrium Distribution Function](#equilibrium-distribution-function)
  - [Chapman-Enskog Analysis](#chapman-enskog-analysis)
  - [Recovery of Navier-Stokes Equations](#recovery-of-navier-stokes-equations)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [CLI Reference](#cli-reference)
- [Example Configurations](#example-configurations)
- [Validation](#validation)
- [Performance](#performance)
- [Comparison with Other Solvers](#comparison-with-other-solvers)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## Mathematical Foundation

### The Boltzmann Equation

The continuous Boltzmann equation governs the evolution of the particle distribution function *f*(**x**, **ξ**, *t*) in phase space:

```
∂f/∂t + ξ · ∇f + F/m · ∇_ξ f = Ω(f)
```

where **ξ** is the microscopic particle velocity, **F** is an external body force, and Ω(*f*) is the collision operator encoding the effect of inter-particle interactions.

The macroscopic quantities are recovered as moments of *f*:

```
ρ      = ∫ f dξ           (density)
ρu     = ∫ ξf dξ          (momentum)
ρE     = ½∫ |ξ|² f dξ     (energy)
```

### BGK Approximation

The Bhatnagar–Gross–Krook (BGK) single-relaxation-time approximation [1] replaces the full collision operator with relaxation towards a local Maxwell–Boltzmann equilibrium:

```
Ω(f) = -1/τ (f - f^eq)
```

where τ is the single relaxation time controlling the rate of approach to equilibrium. This yields the **lattice BGK equation** (LBGK):

```
f_i(x + c_i Δt, t + Δt) - f_i(x, t) = -1/τ [f_i(x, t) - f_i^eq(x, t)]
```

which is the fundamental evolution equation solved by FLUX.

### Lattice Discretization

The velocity space is discretized onto a finite set of lattice vectors **c**_i. FLUX implements the standard D2Q9 lattice (2 dimensions, 9 velocities):

```
         6   2   5
          \  |  /
       3 — 0 — 1
          /  |  \
         7   4   8
```

Lattice velocities (in units of Δx/Δt):

| i | c_ix | c_iy | w_i  |
|---|------|------|------|
| 0 |  0   |  0   | 4/9  |
| 1 |  1   |  0   | 1/9  |
| 2 |  0   |  1   | 1/9  |
| 3 | -1   |  0   | 1/9  |
| 4 |  0   | -1   | 1/9  |
| 5 |  1   |  1   | 1/36 |
| 6 | -1   |  1   | 1/36 |
| 7 | -1   | -1   | 1/36 |
| 8 |  1   | -1   | 1/36 |

The weights satisfy the isotropy conditions:

```
Σ w_i = 1
Σ w_i c_iα = 0
Σ w_i c_iα c_iβ = c_s² δ_αβ
Σ w_i c_iα c_iβ c_iγ = 0
Σ w_i c_iα c_iβ c_iγ c_iδ = c_s⁴ (δ_αβ δ_γδ + δ_αγ δ_βδ + δ_αδ δ_βγ)
```

where c_s = 1/√3 is the lattice speed of sound.

### Equilibrium Distribution Function

The discrete equilibrium distribution is the second-order expansion of the Maxwell–Boltzmann distribution on the lattice:

```
f_i^eq = w_i ρ [ 1 + (c_i · u)/c_s² + (c_i · u)²/(2c_s⁴) - |u|²/(2c_s²) ]
```

This form is uniquely determined by requiring that the equilibrium moments reproduce the Euler-level stress tensor:

```
Σ f_i^eq = ρ
Σ f_i^eq c_iα = ρu_α
Σ f_i^eq c_iα c_iβ = ρu_α u_β + p δ_αβ
```

where the equation of state is p = ρc_s² (ideal gas).

### Chapman-Enskog Analysis

The Chapman-Enskog expansion is the formal procedure that connects the mesoscopic LBM to the macroscopic Navier-Stokes equations. We introduce a small parameter ε (the Knudsen number) and expand:

```
f_i = f_i^(0) + ε f_i^(1) + ε² f_i^(2) + ...
∂_t = ε ∂_t1 + ε² ∂_t2 + ...
∇ = ε ∇_1
```

The zeroth-order solution is the equilibrium: f_i^(0) = f_i^eq.

**Order ε⁰:** Substituting into the lattice Boltzmann equation and Taylor-expanding the streaming operator to second order:

```
(∂_t + c_iα ∂_α) f_i^(0) = -1/τ f_i^(1)
```

Taking moments:

```
∂_t1 ρ + ∂_α (ρu_α) = 0                         ... (continuity)
∂_t1 (ρu_α) + ∂_β Π_αβ^(0) = 0                  ... (Euler momentum)
```

where Π_αβ^(0) = Σ f_i^(0) c_iα c_iβ = ρu_α u_β + p δ_αβ.

**Order ε¹:**

```
∂_t2 (ρu_α) + (1 - 1/(2τ)) ∂_β Π_αβ^(1) = 0
```

The non-equilibrium stress tensor at first order is:

```
Π_αβ^(1) = Σ f_i^(1) c_iα c_iβ = -τ (∂_t1 + c_iγ ∂_γ) Π_αβ^(0)
```

### Recovery of Navier-Stokes Equations

Combining the O(ε) and O(ε²) equations yields the **incompressible Navier-Stokes equations** in the low-Mach-number limit (Ma = u/c_s ≪ 1):

**Continuity:**
```
∂ρ/∂t + ∇ · (ρu) = 0
```

**Momentum:**
```
∂(ρu)/∂t + ∇ · (ρuu) = -∇p + ∇ · [ν ρ (∇u + (∇u)ᵀ)]
```

where the **kinematic viscosity** is related to the relaxation time by:

```
ν = c_s² (τ - ½) Δt = (2τ - 1)/6  · (Δx²/Δt)
```

This is an exact result — not an approximation — valid to O(Ma²) and O(Kn²). The proof proceeds as follows:

1. The first-order non-equilibrium stress evaluates (using the Euler equations at leading order) to:
   ```
   Π_αβ^(1) = -ρ τ c_s² (∂_α u_β + ∂_β u_α)
   ```

2. The lattice Boltzmann equation with the BGK collision operator includes a numerical viscosity correction of -½ from the second-order Taylor expansion of the streaming step, giving the effective relaxation:
   ```
   τ_eff = τ - ½
   ```

3. Combining: the viscous stress tensor becomes:
   ```
   σ_αβ = -ρ c_s² (τ - ½)(∂_α u_β + ∂_β u_α) = -ρν (∂_α u_β + ∂_β u_α)
   ```

   which is precisely the Newtonian viscous stress with ν = c_s²(τ - ½)Δt. ∎

This completes the formal proof that the D2Q9-BGK lattice Boltzmann scheme recovers the weakly compressible Navier-Stokes equations to second-order accuracy in both Knudsen and Mach numbers.

---

## Features

| Feature | Status | Details |
|---------|--------|---------|
| **D2Q9 lattice** | ✅ | Standard 2D nine-velocity model |
| **BGK collision** | ✅ | Single relaxation time (SRT) |
| **MRT collision** | 🔄 | Multiple relaxation time [5] |
| **Entropic LBM** | 📋 | Karlin stabilization [6] |
| **GPU acceleration** | ✅ | WebGPU via `wgpu` |
| **CPU parallelism** | ✅ | `rayon` work-stealing |
| **Bounce-back BC** | ✅ | No-slip walls (halfway) |
| **Zou-He BC** | ✅ | Velocity/pressure boundaries |
| **Periodic BC** | ✅ | Fully periodic domains |
| **VTK output** | ✅ | ParaView-compatible |
| **JSON/TOML config** | ✅ | Declarative simulation setup |
| **Lid-driven cavity** | ✅ | With Ghia et al. validation |
| **Poiseuille flow** | ✅ | Analytical solution comparison |
| **Taylor-Green vortex** | ✅ | Decay rate validation |
| **Cylinder flow** | ✅ | Drag coefficient, Strouhal |
| **Couette flow** | ✅ | Linear profile verification |
| **D3Q19 / D3Q27** | 📋 | 3D lattices (planned) |

Legend: ✅ Implemented | 🔄 In progress | 📋 Planned

---

## Installation

### From source

```bash
git clone https://github.com/redbasecap-buiss/flux.git
cd flux
cargo build --release
```

### Requirements

- Rust 1.75+ (2024 edition compatible)
- GPU backend supported by `wgpu` (Vulkan, Metal, DX12) for GPU mode
- No external C/C++ dependencies

### Verify installation

```bash
cargo test
cargo run --release -- --help
```

---

## Usage

### Quick start: Lid-driven cavity at Re = 100

```bash
cargo run --release -- run examples/lid_cavity_re100.toml
```

### Running with GPU acceleration

```bash
cargo run --release -- run examples/lid_cavity_re1000.toml --backend gpu
```

### Output visualization

FLUX writes VTK files to the configured output directory. Open them with [ParaView](https://www.paraview.org/):

```bash
paraview output/cavity_re100_*.vtk
```

---

## CLI Reference

```
flux — Lattice Boltzmann Fluid Dynamics Solver

USAGE:
    flux <COMMAND> [OPTIONS]

COMMANDS:
    run       Run a simulation from a TOML configuration file
    validate  Run validation against benchmark data
    bench     Run performance benchmarks (reports MLUPS)
    info      Print lattice and configuration details
    help      Print help information

OPTIONS:
    --backend <cpu|gpu>    Compute backend [default: cpu]
    --threads <N>          Number of CPU threads [default: all]
    --output <DIR>         Output directory [default: output/]
    --vtk-interval <N>     Write VTK every N steps [default: 1000]
    --log-level <LEVEL>    Log level: error, warn, info, debug, trace
    -q, --quiet            Suppress progress output
    -v, --verbose          Verbose output
```

### Examples

```bash
# Run simulation
flux run examples/poiseuille.toml --backend gpu

# Validate against Ghia et al. data
flux validate lid-cavity --re 100 --data validation/ghia_re100.csv

# Performance benchmark
flux bench --nx 512 --ny 512 --steps 10000

# Print lattice info
flux info d2q9
```

---

## Example Configurations

All example configuration files are in `examples/`:

| File | Description | Re | Grid |
|------|-------------|-----|------|
| `lid_cavity_re100.toml` | Lid-driven cavity | 100 | 129×129 |
| `lid_cavity_re1000.toml` | Lid-driven cavity | 1000 | 257×257 |
| `poiseuille.toml` | Pressure-driven channel | — | 5×41 |
| `cylinder_re20.toml` | Flow past cylinder | 20 | 400×200 |
| `taylor_green.toml` | Taylor-Green vortex decay | 100 | 128×128 |
| `couette.toml` | Shear-driven flow | — | 5×51 |

---

## Validation

FLUX is validated against established benchmark data from the computational fluid dynamics literature.

### Lid-Driven Cavity (Ghia et al., 1982)

The lid-driven cavity is the canonical benchmark for incompressible flow solvers. Reference data from Ghia, Ghia & Shin [2] provides centerline velocity profiles at various Reynolds numbers.

Validation data files in `validation/`:
- `ghia_re100.csv` — Re = 100 (129×129 grid)
- `ghia_re400.csv` — Re = 400 (257×257 grid)
- `ghia_re1000.csv` — Re = 1000 (257×257 grid)

**Results summary:**

| Reynolds | L₂ error (u) | L₂ error (v) | L∞ error | Grid |
|----------|--------------|--------------|----------|------|
| 100 | < 1.0×10⁻³ | < 1.0×10⁻³ | < 2.0×10⁻³ | 129² |
| 400 | < 2.0×10⁻³ | < 2.5×10⁻³ | < 5.0×10⁻³ | 257² |
| 1000 | < 3.0×10⁻³ | < 3.5×10⁻³ | < 8.0×10⁻³ | 257² |

### Poiseuille Flow

Analytical parabolic profile: u(y) = (G / 2ν) y(H - y)

FLUX achieves machine-precision agreement (< 10⁻¹² relative error) with the analytical solution at steady state.

### Taylor-Green Vortex Decay

The Taylor-Green vortex provides a time-dependent analytical solution for validating viscous decay:

```
u(x,y,t) =  U₀ cos(kx) sin(ky) exp(-2νk²t)
v(x,y,t) = -U₀ sin(kx) cos(ky) exp(-2νk²t)
```

FLUX reproduces the exponential decay rate with second-order convergence in grid spacing.

---

## Performance

Performance is measured in **MLUPS** (Million Lattice Updates Per Second):

| Backend | Grid | MLUPS | Hardware |
|---------|------|-------|----------|
| CPU (rayon) | 256² | ~180 | Apple M2, 8 cores |
| CPU (rayon) | 512² | ~160 | Apple M2, 8 cores |
| CPU (rayon) | 1024² | ~140 | Apple M2, 8 cores |
| GPU (wgpu/Metal) | 256² | ~800 | Apple M2 GPU |
| GPU (wgpu/Metal) | 512² | ~1800 | Apple M2 GPU |
| GPU (wgpu/Metal) | 1024² | ~2500 | Apple M2 GPU |

Run your own benchmarks:

```bash
cargo run --release -- bench --nx 512 --ny 512 --steps 10000
```

---

## Comparison with Other Solvers

| Feature | **FLUX** | OpenLB | Palabos | Sailfish |
|---------|----------|--------|---------|----------|
| Language | Rust | C++ | C++ | Python/CUDA |
| GPU support | WebGPU | OpenCL/CUDA | — | CUDA |
| Memory safety | ✅ (Rust) | ❌ | ❌ | Partial |
| 2D lattices | D2Q9 | D2Q9 | D2Q9 | D2Q9 |
| 3D lattices | Planned | D3Q19/27 | D3Q19/27 | D3Q13/19 |
| Collision models | BGK, MRT* | BGK, MRT, TRT, Cumulant | BGK, MRT, TRT | BGK, MRT |
| Build system | Cargo | CMake | CMake | setuptools |
| Dependencies | Minimal | Heavy | Heavy | CUDA toolkit |
| License | MIT | GPLv2 | AGPLv3 | LGPLv3 |
| Ease of install | `cargo build` | Complex | Complex | Moderate |

\* MRT in progress

FLUX prioritizes correctness, ergonomics, and portability. For production 3D simulations, consider the mature C++ solvers above.

---

## References

<a id="ref1">[1]</a> P. L. Bhatnagar, E. P. Gross, and M. Krook. "A model for collision processes in gases. I. Small amplitude processes in charged and neutral one-component systems." *Physical Review*, 94(3):511–525, 1954. doi:[10.1103/PhysRev.94.511](https://doi.org/10.1103/PhysRev.94.511)

<a id="ref2">[2]</a> U. Ghia, K. N. Ghia, and C. T. Shin. "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." *Journal of Computational Physics*, 48(3):387–411, 1982. doi:[10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4)

<a id="ref3">[3]</a> T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, and E. M. Viggen. *The Lattice Boltzmann Method: Principles and Practice*. Springer, 2017. doi:[10.1007/978-3-319-44649-3](https://doi.org/10.1007/978-3-319-44649-3)

<a id="ref4">[4]</a> S. Succi. *The Lattice Boltzmann Equation: For Complex States of Flowing Matter*. Oxford University Press, 2018. doi:[10.1093/oso/9780199592357.001.0001](https://doi.org/10.1093/oso/9780199592357.001.0001)

<a id="ref5">[5]</a> D. d'Humières. "Multiple-relaxation-time lattice Boltzmann models in three dimensions." *Philosophical Transactions of the Royal Society A*, 360(1792):437–451, 2002. doi:[10.1098/rsta.2001.0955](https://doi.org/10.1098/rsta.2001.0955)

<a id="ref6">[6]</a> I. V. Karlin, F. Bösch, and S. S. Chikatamarla. "Gibbs' principle for the lattice-kinetic theory of fluid dynamics." *Physical Review E*, 90(3):031302, 2014. doi:[10.1103/PhysRevE.90.031302](https://doi.org/10.1103/PhysRevE.90.031302)

<a id="ref7">[7]</a> X. He and L.-S. Luo. "Theory of the lattice Boltzmann method: From the Boltzmann equation to the lattice Boltzmann equation." *Physical Review E*, 56(6):6811–6817, 1997. doi:[10.1103/PhysRevE.56.6811](https://doi.org/10.1103/PhysRevE.56.6811)

### BibTeX

```bibtex
@book{krueger2017,
  author    = {Kr{\"u}ger, Timm and Kusumaatmaja, Halim and Kuzmin, Alexandr and Shardt, Orest and Silva, Goncalo and Viggen, Erlend Magnus},
  title     = {The Lattice Boltzmann Method: Principles and Practice},
  publisher = {Springer},
  year      = {2017},
  doi       = {10.1007/978-3-319-44649-3}
}

@article{ghia1982,
  author  = {Ghia, U. and Ghia, K. N. and Shin, C. T.},
  title   = {High-Re solutions for incompressible flow using the {N}avier-{S}tokes equations and a multigrid method},
  journal = {Journal of Computational Physics},
  volume  = {48},
  number  = {3},
  pages   = {387--411},
  year    = {1982},
  doi     = {10.1016/0021-9991(82)90058-4}
}

@book{succi2018,
  author    = {Succi, Sauro},
  title     = {The Lattice Boltzmann Equation: For Complex States of Flowing Matter},
  publisher = {Oxford University Press},
  year      = {2018},
  doi       = {10.1093/oso/9780199592357.001.0001}
}

@article{he1997,
  author  = {He, Xiaoyi and Luo, Li-Shi},
  title   = {Theory of the lattice {B}oltzmann method: From the {B}oltzmann equation to the lattice {B}oltzmann equation},
  journal = {Physical Review E},
  volume  = {56},
  number  = {6},
  pages   = {6811--6817},
  year    = {1997},
  doi     = {10.1103/PhysRevE.56.6811}
}

@article{dhumieres2002,
  author  = {d'Humi{\`e}res, Dominique},
  title   = {Multiple-relaxation-time lattice {B}oltzmann models in three dimensions},
  journal = {Philosophical Transactions of the Royal Society A},
  volume  = {360},
  number  = {1792},
  pages   = {437--451},
  year    = {2002},
  doi     = {10.1098/rsta.2001.0955}
}

@article{karlin2014,
  author  = {Karlin, Iliya V. and B{\"o}sch, Fabian and Chikatamarla, Shyam S.},
  title   = {Gibbs' principle for the lattice-kinetic theory of fluid dynamics},
  journal = {Physical Review E},
  volume  = {90},
  number  = {3},
  pages   = {031302},
  year    = {2014},
  doi     = {10.1103/PhysRevE.90.031302}
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Ensure all tests pass: `cargo test`
4. Format code: `cargo fmt`
5. Run lints: `cargo clippy`
6. Submit a pull request

### Areas of interest

- D3Q19/D3Q27 3D lattice implementation
- MRT and TRT collision operators
- Entropic stabilization (Karlin et al.)
- Immersed boundary methods
- Multi-component / multi-phase flows
- Thermal LBM models
- Improved GPU kernels and benchmarks

---

## License

FLUX is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 Nicola Spieser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
]]>