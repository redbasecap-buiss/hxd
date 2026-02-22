//! Turbulence models for LBM simulations.
//!
//! # Smagorinsky Large Eddy Simulation (LES)
//!
//! The Smagorinsky subgrid-scale model computes a turbulent (eddy) viscosity:
//!
//! νt = (Cs · Δ)² · |S|
//!
//! where:
//! - Cs is the Smagorinsky constant (typically 0.1–0.2)
//! - Δ is the grid spacing (= 1 in lattice units)
//! - |S| is the magnitude of the strain rate tensor
//!
//! The effective relaxation time becomes:
//!
//! τ_eff = 0.5 * (τ₀ + √(τ₀² + 18·Cs²·|Π_neq|/(ρ·cs⁴)))
//!
//! # References
//! - Smagorinsky, J. (1963). "General circulation experiments with the primitive equations."
//! - Hou, S. et al. (1996). "Simulation of Cavity Flow by the Lattice Boltzmann Method."
//! - Yu, H. et al. (2005). "LES of turbulent square jet flow using an MRT lattice Boltzmann model."

use crate::lattice::{Lattice2D, CS2, D2Q9_E};

/// Smagorinsky LES model parameters
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SmagorinskyModel {
    /// Smagorinsky constant Cs (typically 0.1–0.2)
    pub cs: f64,
    /// Grid spacing Δ (= 1 in lattice units)
    pub delta: f64,
}

impl SmagorinskyModel {
    /// Create a new Smagorinsky model with given constant.
    pub fn new(cs: f64) -> Self {
        Self { cs, delta: 1.0 }
    }

    /// Create with custom grid spacing (for non-uniform grids).
    pub fn with_delta(cs: f64, delta: f64) -> Self {
        Self { cs, delta }
    }

    /// Compute the non-equilibrium stress tensor components from distributions.
    ///
    /// Π_neq_αβ = Σ_i (f_i - f_i^eq) · e_iα · e_iβ
    #[inline]
    pub fn non_equilibrium_stress(fi: &[f64; 9], feq: &[f64; 9]) -> (f64, f64, f64) {
        let mut pi_xx = 0.0;
        let mut pi_xy = 0.0;
        let mut pi_yy = 0.0;
        for q in 0..9 {
            let fneq = fi[q] - feq[q];
            let ex = D2Q9_E[q].0 as f64;
            let ey = D2Q9_E[q].1 as f64;
            pi_xx += ex * ex * fneq;
            pi_xy += ex * ey * fneq;
            pi_yy += ey * ey * fneq;
        }
        (pi_xx, pi_xy, pi_yy)
    }

    /// Compute the magnitude of the non-equilibrium stress tensor.
    ///
    /// |Π_neq| = √(Π_xx² + 2·Π_xy² + Π_yy²)
    #[inline]
    pub fn stress_magnitude(pi_xx: f64, pi_xy: f64, pi_yy: f64) -> f64 {
        (pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy).sqrt()
    }

    /// Compute the strain rate magnitude |S| from non-equilibrium stress.
    ///
    /// For D2Q9: |S| = |Π_neq| / (2·ρ·cs²·τ)
    #[inline]
    pub fn strain_rate_from_stress(pi_mag: f64, rho: f64, tau: f64) -> f64 {
        pi_mag / (2.0 * rho * CS2 * tau)
    }

    /// Compute the subgrid eddy viscosity.
    ///
    /// νt = (Cs · Δ)² · |S|
    #[inline]
    pub fn eddy_viscosity(&self, strain_rate: f64) -> f64 {
        (self.cs * self.delta).powi(2) * strain_rate
    }

    /// Compute effective tau incorporating Smagorinsky SGS model.
    ///
    /// Uses the formulation from Hou et al. (1996):
    /// τ_eff = 0.5 * (τ₀ + √(τ₀² + 18·Cs²·Δ²·|Π_neq|/(ρ·cs⁴)))
    #[inline]
    pub fn effective_tau(&self, tau0: f64, pi_mag: f64, rho: f64) -> f64 {
        let discriminant = tau0 * tau0
            + 18.0 * self.cs * self.cs * self.delta * self.delta * pi_mag / (rho * CS2 * CS2);
        0.5 * (tau0 + discriminant.sqrt())
    }

    /// Compute the effective tau for a specific lattice node.
    pub fn effective_tau_at(&self, lattice: &Lattice2D, x: usize, y: usize) -> f64 {
        let mut fi = [0.0; 9];
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        for q in 0..9 {
            let val = lattice.f[lattice.idx(q, x, y)];
            fi[q] = val;
            rho += val;
            ux += val * D2Q9_E[q].0 as f64;
            uy += val * D2Q9_E[q].1 as f64;
        }
        if rho > 1e-15 {
            ux /= rho;
            uy /= rho;
        }
        let feq = Lattice2D::equilibrium(rho, ux, uy);
        let (pi_xx, pi_xy, pi_yy) = Self::non_equilibrium_stress(&fi, &feq);
        let pi_mag = Self::stress_magnitude(pi_xx, pi_xy, pi_yy);
        self.effective_tau(lattice.tau, pi_mag, rho)
    }

    /// Compute the eddy viscosity field for the entire lattice.
    pub fn eddy_viscosity_field(&self, lattice: &Lattice2D) -> Vec<f64> {
        let nx = lattice.nx;
        let ny = lattice.ny;
        let mut nu_t = vec![0.0; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let tau_eff = self.effective_tau_at(lattice, x, y);
                let nu_eff = CS2 * (tau_eff - 0.5);
                let nu_base = CS2 * (lattice.tau - 0.5);
                nu_t[y * nx + x] = (nu_eff - nu_base).max(0.0);
            }
        }
        nu_t
    }
}

/// Compute the Q-criterion for 2D flows (useful for vortex identification).
///
/// Q = 0.5 * (|Ω|² - |S|²)
///
/// where Ω is the vorticity tensor and S is the strain rate tensor.
/// In 2D: Q = -(∂u/∂x)² - (∂v/∂y)² - 2·(∂u/∂y)·(∂v/∂x) ... simplified
/// Actually Q = -0.5 * (S_ij S_ij) when using just strain (for vortex detection).
pub fn q_criterion_field(lattice: &Lattice2D) -> Vec<f64> {
    let nx = lattice.nx;
    let ny = lattice.ny;
    let mut q_field = vec![0.0; nx * ny];

    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            let (_, ux_r, uy_r) = macroscopic(lattice, x + 1, y);
            let (_, ux_l, uy_l) = macroscopic(lattice, x - 1, y);
            let (_, ux_u, uy_u) = macroscopic(lattice, x, y + 1);
            let (_, ux_d, uy_d) = macroscopic(lattice, x, y - 1);

            let dudx = (ux_r - ux_l) / 2.0;
            let dudy = (ux_u - ux_d) / 2.0;
            let dvdx = (uy_r - uy_l) / 2.0;
            let dvdy = (uy_u - uy_d) / 2.0;

            // Strain rate tensor: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
            let s11 = dudx;
            let s22 = dvdy;
            let s12 = 0.5 * (dudy + dvdx);

            // Vorticity tensor: Ω_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
            let omega12 = 0.5 * (dudy - dvdx);

            // Q = 0.5 * (|Ω|² - |S|²)
            let omega_sq = 2.0 * omega12 * omega12;
            let s_sq = s11 * s11 + s22 * s22 + 2.0 * s12 * s12;
            q_field[y * nx + x] = 0.5 * (omega_sq - s_sq);
        }
    }
    q_field
}

/// Compute turbulent kinetic energy (TKE) from velocity fluctuations.
///
/// TKE = 0.5 * (u'² + v'²)
/// where u' = u - ū (fluctuation from mean).
pub fn turbulent_kinetic_energy(
    ux: &[f64],
    uy: &[f64],
    ux_mean: &[f64],
    uy_mean: &[f64],
) -> Vec<f64> {
    ux.iter()
        .zip(uy.iter())
        .zip(ux_mean.iter())
        .zip(uy_mean.iter())
        .map(|(((&u, &v), &um), &vm)| {
            let up = u - um;
            let vp = v - vm;
            0.5 * (up * up + vp * vp)
        })
        .collect()
}

/// Helper: get macroscopic quantities at a node
#[inline]
fn macroscopic(lattice: &Lattice2D, x: usize, y: usize) -> (f64, f64, f64) {
    let rho = lattice.density(x, y);
    let (ux, uy) = lattice.velocity(x, y);
    (rho, ux, uy)
}

/// Compute the resolved strain rate tensor field.
/// Returns (S11, S12, S22) at each point using central differences.
pub fn strain_rate_field(lattice: &Lattice2D) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nx = lattice.nx;
    let ny = lattice.ny;
    let mut s11 = vec![0.0; nx * ny];
    let mut s12 = vec![0.0; nx * ny];
    let mut s22 = vec![0.0; nx * ny];

    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            let (_, ux_r, uy_r) = macroscopic(lattice, x + 1, y);
            let (_, ux_l, uy_l) = macroscopic(lattice, x - 1, y);
            let (_, ux_u, uy_u) = macroscopic(lattice, x, y + 1);
            let (_, ux_d, uy_d) = macroscopic(lattice, x, y - 1);

            let idx = y * nx + x;
            s11[idx] = (ux_r - ux_l) / 2.0;
            s22[idx] = (uy_u - uy_d) / 2.0;
            s12[idx] = 0.25 * ((ux_u - ux_d) + (uy_r - uy_l));
        }
    }
    (s11, s12, s22)
}

/// Jet colormap: blue → cyan → green → yellow → red
/// Classic "jet" colormap for scientific visualization.
pub fn colormap_jet(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let r;
    let g;
    let b;

    if t < 0.125 {
        r = 0.0;
        g = 0.0;
        b = 0.5 + t / 0.125 * 0.5;
    } else if t < 0.375 {
        let s = (t - 0.125) / 0.25;
        r = 0.0;
        g = s;
        b = 1.0;
    } else if t < 0.625 {
        let s = (t - 0.375) / 0.25;
        r = s;
        g = 1.0;
        b = 1.0 - s;
    } else if t < 0.875 {
        let s = (t - 0.625) / 0.25;
        r = 1.0;
        g = 1.0 - s;
        b = 0.0;
    } else {
        let s = (t - 0.875) / 0.125;
        r = 1.0 - s * 0.5;
        g = 0.0;
        b = 0.0;
    }

    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Write a PPM image using jet colormap for a scalar field.
pub fn write_ppm_jet(
    path: &std::path::Path,
    nx: usize,
    ny: usize,
    field: &[f64],
) -> std::io::Result<()> {
    use std::io::Write;
    let max_v = field.iter().cloned().fold(0.0f64, f64::max).max(1e-10);
    let min_v = field.iter().cloned().fold(f64::MAX, f64::min);
    let range = (max_v - min_v).max(1e-10);

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", nx, ny)?;
    writeln!(file, "255")?;

    for y in (0..ny).rev() {
        for x in 0..nx {
            let t = (field[y * nx + x] - min_v) / range;
            let (r, g, b) = colormap_jet(t);
            file.write_all(&[r, g, b])?;
        }
    }
    Ok(())
}

/// Write velocity magnitude as PPM with jet colormap.
pub fn write_velocity_ppm_jet(
    path: &std::path::Path,
    nx: usize,
    ny: usize,
    ux: &[f64],
    uy: &[f64],
) -> std::io::Result<()> {
    let vmag: Vec<f64> = ux
        .iter()
        .zip(uy.iter())
        .map(|(&u, &v)| (u * u + v * v).sqrt())
        .collect();
    write_ppm_jet(path, nx, ny, &vmag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{CollisionOperator, Lattice2D};

    #[test]
    fn test_smagorinsky_new() {
        let smag = SmagorinskyModel::new(0.1);
        assert!((smag.cs - 0.1).abs() < 1e-14);
        assert!((smag.delta - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_smagorinsky_eddy_viscosity_zero_strain() {
        let smag = SmagorinskyModel::new(0.1);
        let nu_t = smag.eddy_viscosity(0.0);
        assert!((nu_t).abs() < 1e-14);
    }

    #[test]
    fn test_smagorinsky_eddy_viscosity_positive_strain() {
        let smag = SmagorinskyModel::new(0.15);
        let strain = 0.05;
        let nu_t = smag.eddy_viscosity(strain);
        // νt = (0.15 * 1.0)^2 * 0.05 = 0.0225 * 0.05 = 0.001125
        assert!((nu_t - 0.001125).abs() < 1e-10);
    }

    #[test]
    fn test_effective_tau_no_stress() {
        let smag = SmagorinskyModel::new(0.1);
        let tau0 = 0.8;
        let tau_eff = smag.effective_tau(tau0, 0.0, 1.0);
        // With zero stress: √(τ₀²) = τ₀, so τ_eff = 0.5*(τ₀+τ₀) = τ₀
        assert!((tau_eff - tau0).abs() < 1e-14);
    }

    #[test]
    fn test_effective_tau_increases_with_stress() {
        let smag = SmagorinskyModel::new(0.15);
        let tau0 = 0.8;
        let tau1 = smag.effective_tau(tau0, 0.01, 1.0);
        let tau2 = smag.effective_tau(tau0, 0.1, 1.0);
        assert!(tau1 > tau0);
        assert!(tau2 > tau1);
    }

    #[test]
    fn test_non_equilibrium_stress_at_equilibrium() {
        let rho = 1.0;
        let ux = 0.05;
        let uy = 0.02;
        let feq = Lattice2D::equilibrium(rho, ux, uy);
        let fi = feq; // At equilibrium, fi == feq
        let (pi_xx, pi_xy, pi_yy) = SmagorinskyModel::non_equilibrium_stress(&fi, &feq);
        assert!(pi_xx.abs() < 1e-14);
        assert!(pi_xy.abs() < 1e-14);
        assert!(pi_yy.abs() < 1e-14);
    }

    #[test]
    fn test_stress_magnitude() {
        let mag = SmagorinskyModel::stress_magnitude(3.0, 0.0, 4.0);
        // √(9 + 0 + 16) = 5
        assert!((mag - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_effective_tau_at_equilibrium_lattice() {
        let lat = Lattice2D::new(10, 10, 0.8, CollisionOperator::Bgk);
        let smag = SmagorinskyModel::new(0.1);
        let tau_eff = smag.effective_tau_at(&lat, 5, 5);
        // At equilibrium, stress is ~0, so tau_eff ≈ tau0
        assert!((tau_eff - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_eddy_viscosity_field_uniform() {
        let lat = Lattice2D::new(10, 10, 0.8, CollisionOperator::Bgk);
        let smag = SmagorinskyModel::new(0.1);
        let nu_t = smag.eddy_viscosity_field(&lat);
        // At equilibrium, eddy viscosity should be ~0 everywhere
        for &v in &nu_t {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_q_criterion_uniform_flow() {
        let lat = Lattice2D::new(10, 10, 0.8, CollisionOperator::Bgk);
        let q = q_criterion_field(&lat);
        // Uniform flow at rest → zero Q
        for y in 1..9 {
            for x in 1..9 {
                assert!(q[y * 10 + x].abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_tke_zero_fluctuation() {
        let ux = vec![0.1; 10];
        let uy = vec![0.05; 10];
        let tke = turbulent_kinetic_energy(&ux, &uy, &ux, &uy);
        for &k in &tke {
            assert!(k.abs() < 1e-14);
        }
    }

    #[test]
    fn test_tke_with_fluctuations() {
        let ux = vec![0.12, 0.08];
        let uy = vec![0.06, 0.04];
        let ux_mean = vec![0.1, 0.1];
        let uy_mean = vec![0.05, 0.05];
        let tke = turbulent_kinetic_energy(&ux, &uy, &ux_mean, &uy_mean);
        // tke[0] = 0.5 * (0.02² + 0.01²) = 0.5 * 0.0005 = 0.00025
        assert!((tke[0] - 0.00025).abs() < 1e-10);
        // tke[1] = 0.5 * ((-0.02)² + (-0.01)²) = 0.00025
        assert!((tke[1] - 0.00025).abs() < 1e-10);
    }

    #[test]
    fn test_strain_rate_field_uniform() {
        let lat = Lattice2D::new(10, 10, 0.8, CollisionOperator::Bgk);
        let (s11, s12, s22) = strain_rate_field(&lat);
        for y in 1..9 {
            for x in 1..9 {
                let idx = y * 10 + x;
                assert!(s11[idx].abs() < 1e-14);
                assert!(s12[idx].abs() < 1e-14);
                assert!(s22[idx].abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_colormap_jet_bounds() {
        // Test at several points — ensure no panics and values are valid u8
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let (r, g, b) = colormap_jet(t);
            // Values are u8 so always in [0, 255]; just verify the function runs
            let _ = (r, g, b);
        }
    }

    #[test]
    fn test_colormap_jet_blue_at_zero() {
        let (r, g, b) = colormap_jet(0.0);
        assert!(b > r, "Should be blue-ish at t=0: r={r}, g={g}, b={b}");
    }

    #[test]
    fn test_colormap_jet_red_at_one() {
        let (r, g, b) = colormap_jet(1.0);
        assert!(r > b, "Should be red-ish at t=1: r={r}, g={g}, b={b}");
    }

    #[test]
    fn test_write_ppm_jet() {
        let dir = std::env::temp_dir().join("flux_test_jet");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_jet.ppm");
        let field: Vec<f64> = (0..64).map(|i| i as f64 / 63.0).collect();
        write_ppm_jet(&path, 8, 8, &field).unwrap();
        assert!(path.exists());
        let data = std::fs::read(&path).unwrap();
        // PPM header + 8*8*3 bytes
        assert!(data.len() > 8 * 8 * 3);
    }

    #[test]
    fn test_write_velocity_ppm_jet() {
        let dir = std::env::temp_dir().join("flux_test_vel_jet");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vel_jet.ppm");
        let ux = vec![0.1; 16];
        let uy = vec![0.05; 16];
        write_velocity_ppm_jet(&path, 4, 4, &ux, &uy).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_smagorinsky_with_delta() {
        let smag = SmagorinskyModel::with_delta(0.1, 2.0);
        let nu_t = smag.eddy_viscosity(0.1);
        // νt = (0.1 * 2.0)^2 * 0.1 = 0.04 * 0.1 = 0.004
        assert!((nu_t - 0.004).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_c_smag_field() {
        // Test the existing c_smag integration in Lattice2D
        let mut lat = Lattice2D::new(10, 10, 0.8, CollisionOperator::Bgk);
        lat.c_smag = 0.15;
        // Perturb and collide — should not panic
        lat.set_equilibrium(5, 5, 1.1, 0.05, 0.02);
        lat.collide_bgk();
        // Verify density conservation
        let rho = lat.density(5, 5);
        assert!((rho - 1.1).abs() < 1e-10);
    }
}
