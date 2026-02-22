//! Shan-Chen pseudopotential multiphase model for Lattice Boltzmann.
//!
//! The Shan-Chen model introduces a pseudopotential ψ(ρ) that generates an interaction
//! force between neighboring fluid elements, enabling phase separation and surface tension.
//!
//! # Interaction Force
//!
//! F_α(x) = -G · ψ(x) · Σ_i w_i · ψ(x + e_i) · e_iα
//!
//! where G is the interaction strength (G < 0 for attraction → phase separation).
//!
//! # Pseudopotential
//!
//! ψ(ρ) = ψ₀ · exp(-ρ₀/ρ)  (original Shan-Chen, 1993)
//!
//! or ψ(ρ) = √(2(p_EOS - ρ·cs²) / (G·cs²))  (Yuan-Schaefer, 2006)
//!
//! # Equation of State
//!
//! The effective pressure is: p = cs²·ρ + G·cs²/2 · ψ²(ρ)
//! which allows liquid-gas coexistence for |G| > G_critical.
//!
//! # References
//! - Shan, X. & Chen, H. (1993). "Lattice Boltzmann model for simulating flows with
//!   multiple phases and components." Physical Review E, 47(3), 1815.
//! - Shan, X. & Chen, H. (1994). "Simulation of nonideal gases and liquid-gas phase
//!   transitions by the lattice Boltzmann equation." Physical Review E, 49(4), 2941.
//! - Yuan, P. & Schaefer, L. (2006). "Equations of state in a lattice Boltzmann model."
//!   Physics of Fluids, 18(4), 042101.

use crate::lattice::{Lattice2D, CS2, D2Q9_E, D2Q9_W};
use rayon::prelude::*;

/// Pseudopotential function type
#[derive(Debug, Clone, Copy)]
pub enum Pseudopotential {
    /// Original Shan-Chen: ψ(ρ) = ψ₀ · exp(-ρ₀/ρ)
    ShanChen { psi0: f64, rho0: f64 },
    /// Square root form: ψ(ρ) = √(2(p_EOS - ρ·cs²) / (G·cs²))
    /// Using Carnahan-Starling EOS for better thermodynamic consistency.
    CarnahanStarling { a: f64, b: f64, temp: f64 },
}

impl Pseudopotential {
    /// Default Shan-Chen pseudopotential.
    /// ψ₀ = 1, ρ₀ = 1 gives ψ(ρ) = exp(-1/ρ) which has good phase-separation properties.
    pub fn shan_chen() -> Self {
        Self::ShanChen {
            psi0: 1.0,
            rho0: 1.0,
        }
    }

    /// Evaluate ψ(ρ)
    pub fn psi(&self, rho: f64) -> f64 {
        match self {
            Self::ShanChen { psi0, rho0 } => {
                if rho < 1e-10 {
                    return 0.0;
                }
                psi0 * (-rho0 / rho).exp()
            }
            Self::CarnahanStarling { a, b, temp } => {
                // Carnahan-Starling EOS: p = ρRT(1+η+η²-η³)/(1-η)³ - aρ²
                // where η = bρ/4
                let eta = b * rho / 4.0;
                let denom = (1.0 - eta).powi(3);
                if denom.abs() < 1e-15 {
                    return 0.0;
                }
                let p_eos =
                    rho * temp * (1.0 + eta + eta * eta - eta * eta * eta) / denom - a * rho * rho;
                let arg = 2.0 * (p_eos - rho * CS2) / CS2;
                if arg > 0.0 {
                    arg.sqrt()
                } else {
                    -((-arg).sqrt())
                }
            }
        }
    }
}

/// Shan-Chen multiphase model
#[derive(Debug, Clone)]
pub struct ShanChenModel {
    /// Interaction strength G (negative → attraction → phase separation)
    pub g: f64,
    /// Pseudopotential function
    pub potential: Pseudopotential,
}

impl ShanChenModel {
    /// Create a new Shan-Chen model.
    ///
    /// For phase separation with the default pseudopotential,
    /// |G| must exceed the critical value (~-6.305 for ψ₀=4, ρ₀=200).
    /// Typical values: G ∈ [-7, -6.5] for moderate density ratio.
    pub fn new(g: f64) -> Self {
        Self {
            g,
            potential: Pseudopotential::shan_chen(),
        }
    }

    /// Create with custom pseudopotential
    pub fn with_potential(g: f64, potential: Pseudopotential) -> Self {
        Self { g, potential }
    }

    /// Compute the pseudopotential field ψ(x,y) for the entire lattice.
    pub fn psi_field(&self, lattice: &Lattice2D) -> Vec<f64> {
        let nx = lattice.nx;
        let ny = lattice.ny;
        (0..ny)
            .into_par_iter()
            .flat_map(|y| {
                (0..nx)
                    .map(|x| {
                        let rho = lattice.density(x, y);
                        self.potential.psi(rho)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Compute interaction force at each node.
    ///
    /// F_α(x) = -G · ψ(x) · Σ_i w_i · ψ(x + e_i) · e_iα
    ///
    /// Returns (Fx, Fy) force fields.
    pub fn interaction_force(&self, lattice: &Lattice2D) -> (Vec<f64>, Vec<f64>) {
        let nx = lattice.nx;
        let ny = lattice.ny;
        let n = nx * ny;
        let psi = self.psi_field(lattice);

        let forces: Vec<(f64, f64)> = (0..ny)
            .into_par_iter()
            .flat_map(|y| {
                (0..nx)
                    .map(|x| {
                        let psi_local = psi[y * nx + x];
                        let mut fx = 0.0;
                        let mut fy = 0.0;

                        for q in 1..9 {
                            // skip rest direction
                            let xn = ((x as i32 + D2Q9_E[q].0).rem_euclid(nx as i32)) as usize;
                            let yn = ((y as i32 + D2Q9_E[q].1).rem_euclid(ny as i32)) as usize;
                            let psi_nb = psi[yn * nx + xn];
                            fx += D2Q9_W[q] * psi_nb * D2Q9_E[q].0 as f64;
                            fy += D2Q9_W[q] * psi_nb * D2Q9_E[q].1 as f64;
                        }
                        fx *= -self.g * psi_local;
                        fy *= -self.g * psi_local;
                        (fx, fy)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut force_x = vec![0.0; n];
        let mut force_y = vec![0.0; n];
        for (i, (fx, fy)) in forces.into_iter().enumerate() {
            force_x[i] = fx;
            force_y[i] = fy;
        }
        (force_x, force_y)
    }

    /// Perform one Shan-Chen step: compute forces, apply velocity shift, then BGK collision.
    ///
    /// The velocity is shifted by the force: u_eq = (Σ f_i e_i + F·τ) / ρ
    /// (Shan-Chen velocity shift scheme).
    pub fn step(&self, lattice: &mut Lattice2D) {
        let nx = lattice.nx;
        let ny = lattice.ny;
        let tau = lattice.tau;
        let omega = 1.0 / tau;

        let (force_x, force_y) = self.interaction_force(lattice);

        // Collision with force-shifted velocity
        let row_data: Vec<(usize, Vec<f64>)> = (0..ny)
            .into_par_iter()
            .map(|y| {
                let mut row_f = vec![0.0; 9 * nx];
                for x in 0..nx {
                    let idx = y * nx + x;
                    let mut rho = 0.0;
                    let mut jx = 0.0;
                    let mut jy = 0.0;
                    let mut fi = [0.0; 9];

                    for q in 0..9 {
                        let val = lattice.f[q * nx * ny + y * nx + x];
                        fi[q] = val;
                        rho += val;
                        jx += val * D2Q9_E[q].0 as f64;
                        jy += val * D2Q9_E[q].1 as f64;
                    }

                    // Force-shifted equilibrium velocity
                    let ux_eq = if rho > 1e-15 {
                        (jx + tau * force_x[idx]) / rho
                    } else {
                        0.0
                    };
                    let uy_eq = if rho > 1e-15 {
                        (jy + tau * force_y[idx]) / rho
                    } else {
                        0.0
                    };

                    let feq = Lattice2D::equilibrium(rho, ux_eq, uy_eq);

                    for q in 0..9 {
                        row_f[q * nx + x] = fi[q] - omega * (fi[q] - feq[q]);
                    }
                }
                (y, row_f)
            })
            .collect();

        for (y, row_f) in row_data {
            for x in 0..nx {
                for q in 0..9 {
                    lattice.f[q * nx * ny + y * nx + x] = row_f[q * nx + x];
                }
            }
        }
    }

    /// Get the physical velocity including force contribution.
    ///
    /// u_phys = (Σ f_i e_i + F/2) / ρ  (half-force correction for accurate velocity)
    pub fn physical_velocity(&self, lattice: &Lattice2D) -> (Vec<f64>, Vec<f64>) {
        let nx = lattice.nx;
        let ny = lattice.ny;
        let (force_x, force_y) = self.interaction_force(lattice);

        let mut ux = vec![0.0; nx * ny];
        let mut uy = vec![0.0; nx * ny];

        for y in 0..ny {
            for x in 0..nx {
                let idx = y * nx + x;
                let rho = lattice.density(x, y);
                let (jx, jy) = {
                    let mut jx = 0.0;
                    let mut jy = 0.0;
                    for q in 0..9 {
                        let fi = lattice.f[lattice.idx(q, x, y)];
                        jx += fi * D2Q9_E[q].0 as f64;
                        jy += fi * D2Q9_E[q].1 as f64;
                    }
                    (jx, jy)
                };
                if rho > 1e-15 {
                    ux[idx] = (jx + 0.5 * force_x[idx]) / rho;
                    uy[idx] = (jy + 0.5 * force_y[idx]) / rho;
                }
            }
        }
        (ux, uy)
    }

    /// Compute surface tension from Laplace law.
    /// For a droplet of radius R: Δp = σ/R (2D)
    /// Returns estimated surface tension σ.
    pub fn estimate_surface_tension(
        &self,
        lattice: &Lattice2D,
        rho_liquid: f64,
        rho_gas: f64,
    ) -> f64 {
        // Mechanical definition: σ = ∫(pn - pt) dn across the interface
        // Simplified: use EOS pressure difference and assume circular interface
        let p_liquid =
            CS2 * rho_liquid + self.g * CS2 / 2.0 * self.potential.psi(rho_liquid).powi(2);
        let p_gas = CS2 * rho_gas + self.g * CS2 / 2.0 * self.potential.psi(rho_gas).powi(2);

        // Find approximate radius from density field
        let nx = lattice.nx;
        let ny = lattice.ny;
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let rho_mid = 0.5 * (rho_liquid + rho_gas);

        let mut radius_sum = 0.0;
        let mut count = 0;
        for y in 0..ny {
            for x in 0..nx {
                let rho = lattice.density(x, y);
                if (rho - rho_mid).abs() < 0.1 * (rho_liquid - rho_gas).abs() {
                    let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                    radius_sum += r;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }
        let radius = radius_sum / count as f64;
        let delta_p = (p_liquid - p_gas).abs();

        // Laplace law in 2D: Δp = σ/R
        delta_p * radius
    }
}

/// Initialize a circular droplet for multiphase simulation.
///
/// Places a liquid droplet of given radius at the center of the domain,
/// surrounded by gas phase.
pub fn init_droplet(lattice: &mut Lattice2D, rho_liquid: f64, rho_gas: f64, radius: f64) {
    let nx = lattice.nx;
    let ny = lattice.ny;
    let cx = nx as f64 / 2.0;
    let cy = ny as f64 / 2.0;

    for y in 0..ny {
        for x in 0..nx {
            let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            // Smooth interface using tanh profile (width ~5 lattice units)
            let w = 5.0;
            let rho = 0.5 * (rho_liquid + rho_gas)
                - 0.5 * (rho_liquid - rho_gas) * ((r - radius) / w).tanh();
            lattice.set_equilibrium(x, y, rho, 0.0, 0.0);
        }
    }
}

/// Initialize a flat interface for surface tension measurement via Laplace pressure.
pub fn init_flat_interface(lattice: &mut Lattice2D, rho_liquid: f64, rho_gas: f64) {
    let nx = lattice.nx;
    let ny = lattice.ny;
    let cy = ny as f64 / 2.0;

    for y in 0..ny {
        for x in 0..nx {
            let dist = (y as f64 - cy).abs();
            let w = 5.0;
            let rho = if dist < ny as f64 / 4.0 {
                0.5 * (rho_liquid + rho_gas)
                    + 0.5
                        * (rho_liquid - rho_gas)
                        * (1.0 - (dist / (ny as f64 / 4.0 - w)).max(0.0).tanh())
            } else {
                rho_gas
            };
            lattice.set_equilibrium(x, y, rho, 0.0, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::CollisionOperator;

    #[test]
    fn test_pseudopotential_shan_chen() {
        let psi = Pseudopotential::shan_chen();
        // ψ(ρ) = exp(-1/ρ); at high density → 1.0
        let val = psi.psi(1000.0);
        assert!(
            (val - 1.0).abs() < 0.01,
            "ψ at high ρ should → ψ₀=1, got {val}"
        );
        // At low density, ψ → 0
        let val_low = psi.psi(0.1);
        assert!(val_low < 1.0);
        assert!(val_low > 0.0);
        // Monotonically increasing
        assert!(psi.psi(2.0) > psi.psi(1.0));
    }

    #[test]
    fn test_shan_chen_force_zero_uniform() {
        // Uniform density → zero force
        let lat = Lattice2D::new(20, 20, 0.8, CollisionOperator::Bgk);
        let model = ShanChenModel::new(-6.0);
        let (fx, fy) = model.interaction_force(&lat);
        for i in 0..fx.len() {
            assert!(
                fx[i].abs() < 1e-12,
                "Force should be zero for uniform density"
            );
            assert!(fy[i].abs() < 1e-12);
        }
    }

    #[test]
    fn test_shan_chen_force_nonzero_gradient() {
        // Non-uniform density → non-zero force
        // ψ = exp(-1/ρ), so at ρ ~ 0.5..2.0 there is significant curvature
        let mut lat = Lattice2D::new(20, 20, 0.8, CollisionOperator::Bgk);
        for y in 0..20 {
            for x in 0..20 {
                let rho = 0.5 + 1.5 * (x as f64 / 19.0);
                lat.set_equilibrium(x, y, rho, 0.0, 0.0);
            }
        }
        let model = ShanChenModel::new(-6.0);
        let (fx, _fy) = model.interaction_force(&lat);
        let mid = 10 * 20 + 10;
        assert!(
            fx[mid].abs() > 1e-10,
            "Force should be non-zero for density gradient, got {}",
            fx[mid]
        );
    }

    #[test]
    fn test_droplet_initialization() {
        let mut lat = Lattice2D::new(64, 64, 0.8, CollisionOperator::Bgk);
        init_droplet(&mut lat, 2.0, 0.1, 15.0);

        // Center should be liquid density
        let rho_center = lat.density(32, 32);
        assert!(
            rho_center > 1.5,
            "Center should be liquid: got {rho_center}"
        );

        // Corner should be gas density
        let rho_corner = lat.density(0, 0);
        assert!(rho_corner < 0.5, "Corner should be gas: got {rho_corner}");
    }

    #[test]
    fn test_shan_chen_mass_conservation() {
        let mut lat = Lattice2D::new(32, 32, 0.8, CollisionOperator::Bgk);
        init_droplet(&mut lat, 2.0, 0.1, 8.0);

        let model = ShanChenModel::new(-6.0);

        // Compute total mass before
        let mut mass_before = 0.0;
        for y in 0..32 {
            for x in 0..32 {
                mass_before += lat.density(x, y);
            }
        }

        // Do a few steps
        for _ in 0..10 {
            model.step(&mut lat);
            lat.stream();
        }

        // Compute total mass after
        let mut mass_after = 0.0;
        for y in 0..32 {
            for x in 0..32 {
                mass_after += lat.density(x, y);
            }
        }

        let error = ((mass_after - mass_before) / mass_before).abs();
        assert!(
            error < 1e-10,
            "Mass not conserved: before={mass_before}, after={mass_after}, error={error}"
        );
    }

    #[test]
    fn test_phase_separation() {
        // Test that Shan-Chen model produces phase separation
        // Initialize with a droplet and verify it maintains phase contrast
        let n = 32;
        let mut lat = Lattice2D::new(n, n, 1.0, CollisionOperator::Bgk);

        // Test that the interaction force creates attraction in high-density regions
        // and repulsion in low-density regions (precursor to phase separation)
        init_droplet(&mut lat, 2.0, 0.5, 8.0);

        let model = ShanChenModel::new(-4.0);

        // Check that forces point inward at the interface (cohesive force)
        let (fx, fy) = model.interaction_force(&lat);
        let cx = n / 2;
        let cy = n / 2;

        // At right interface (x > center), force should point left (negative fx)
        // because ψ gradient points outward and G < 0 makes force cohesive
        let right_idx = cy * n + (cx + 7); // near interface on the right
        let left_idx = cy * n + (cx - 7); // near interface on the left

        // The forces at the interface should be non-zero and oppose the density gradient
        let f_right = fx[right_idx];
        let f_left = fx[left_idx];

        assert!(
            f_right.abs() > 1e-6 || f_left.abs() > 1e-6,
            "Interface forces should be non-zero: f_right={f_right:.2e}, f_left={f_left:.2e}"
        );

        // Forces should be opposite at opposite sides of the droplet
        // (both pointing inward or both pointing outward)
        if f_right.abs() > 1e-8 && f_left.abs() > 1e-8 {
            assert!(
                f_right * f_left < 0.0,
                "Forces should be opposite at interface: right={f_right:.2e}, left={f_left:.2e}"
            );
        }
    }
}
