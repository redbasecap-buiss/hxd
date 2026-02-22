# FLUX Python Visualization Examples

Beautiful **Lattice Boltzmann Method (LBM)** simulations implemented in pure Python with stunning HD video output.

## 🌊 Simulations

### `vortex_street.py` — von Kármán Vortex Street
- **Flow around a cylinder** creating spectacular vortex shedding
- **D2Q9 LBM implementation** from scratch 
- **Vorticity visualization** with `RdBu_r` colormap
- **Output**: `vortex_street.mp4` (1920×720, 30fps, 16.7s)

```bash
python3 vortex_street.py
```

### `lid_cavity.py` — Lid-Driven Cavity Flow
- **Classical CFD benchmark** (Ghia et al. 1982)
- **Dual visualization**: velocity magnitude + streamlines
- **Moving wall boundary conditions**
- **Output**: `lid_cavity.mp4` (1920×960, 20fps, 20s)

```bash
python3 lid_cavity.py
```

### `simple_lbm_test.py` — Physics Validation
- **Taylor-Green vortex** decay test
- **Periodic boundaries** for numerical stability
- **Mass/energy conservation** checks
- Validates core LBM implementation

```bash
python3 simple_lbm_test.py
```

## 🎬 Video Gallery

| Simulation | Preview | Physics |
|------------|---------|---------|
| **Vortex Street** | Spectacular vortex shedding behind cylinder | Re=100, D2Q9, BGK collision |
| **Lid Cavity** | Recirculating flow with streamlines | Re=400, Moving wall BC |

## 🔬 Theory

Both simulations implement the **Lattice Boltzmann Method** with:

- **D2Q9 velocity set**: 9 discrete velocities in 2D
- **BGK collision operator**: Single relaxation time τ
- **Maxwell-Boltzmann equilibrium**: Truncated to 2nd order
- **Bounce-back boundaries**: For solid walls
- **Zou-He inlet BC**: For prescribed velocity (vortex street)

### Key Physics Parameters

| Parameter | Vortex Street | Lid Cavity |
|-----------|---------------|------------|
| Grid | 400×100 | 128×128 |
| Reynolds number | 100 | 400 |
| Inlet velocity | 0.05 | 0.05 (lid) |
| Relaxation time τ | 0.515 | 0.548 |

## 🛠 Requirements

```bash
pip3 install numpy matplotlib
# or
brew install python-matplotlib
```

FFmpeg required for MP4 output (usually via homebrew).

## 🎯 Educational Value

These implementations prioritize **clarity over performance**:

- **Pure Python**: No external LBM libraries
- **Well-commented**: Explains each physics step  
- **Publication-quality output**: HD videos with proper colormaps
- **Benchmark cases**: Compare with literature (Ghia et al.)

Perfect for:
- **Learning LBM theory** 
- **CFD course demonstrations**
- **Research visualization**
- **Complementing the high-performance Rust solver**

## 🚀 Performance

| Simulation | Time | Frames | Physics Steps |
|------------|------|--------|---------------|
| Vortex Street | 44.8s | 500 | 10,000 |
| Lid Cavity | 124.3s | 400 | 20,000 |

*Tested on Apple M-series chip*

## 📚 References

1. **Krüger, T. et al.** (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer.
2. **Ghia, U., Ghia, K.N., & Shin, C.T.** (1982). High-Re solutions for incompressible flow. *Journal of Computational Physics*.
3. **Zou, Q. & He, X.** (1997). On pressure and velocity boundary conditions for LBM BGK model. *Physics of Fluids*.

---

*Part of the **FLUX** project — University-grade LBM fluid dynamics solver*