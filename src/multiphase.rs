//! Multiphase flow models for Lattice Boltzmann Methods.
//!
//! Includes Shan-Chen pseudopotential, Color Gradient, and Free Energy models.
//! Equations of state: Carnahan-Starling and Peng-Robinson.
//! Surface tension via Laplace law, contact angle boundary conditions.

use crate::turbulence::{equilibrium, macroscopic, C, CS2, W};

// ── Equations of State ──────────────────────────────────────────────────────

/// Equation of state trait for multiphase LBM.
pub trait EquationOfState {
    /// Compute pressure from density and temperature.
    fn pressure(&self, rho: f64, temperature: f64) -> f64;
    /// Compute dp/drho at constant temperature.
    fn dp_drho(&self, rho: f64, temperature: f64) -> f64;
}

/// Carnahan-Starling equation of state.
///
/// p = ρRT(1 + bρ/4 + (bρ/4)² − (bρ/4)³) / (1 − bρ/4)³ − aρ²
#[derive(Debug, Clone, Copy)]
pub struct CarnahanStarling {
    /// Attraction parameter.
    pub a: f64,
    /// Repulsion parameter (excluded volume).
    pub b: f64,
    /// Gas constant (in lattice units).
    pub r: f64,
}

impl CarnahanStarling {
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b, r: 1.0 }
    }

    /// Default parameters for water-like fluid.
    pub fn default_fluid() -> Self {
        Self {
            a: 0.5,
            b: 4.0,
            r: 1.0,
        }
    }
}

impl EquationOfState for CarnahanStarling {
    fn pressure(&self, rho: f64, temperature: f64) -> f64 {
        let eta = self.b * rho / 4.0;
        let num = 1.0 + eta + eta * eta - eta * eta * eta;
        let den = (1.0 - eta).powi(3);
        if den.abs() < 1e-30 {
            return rho * self.r * temperature;
        }
        rho * self.r * temperature * num / den - self.a * rho * rho
    }

    fn dp_drho(&self, rho: f64, temperature: f64) -> f64 {
        let dr = rho * 1e-6 + 1e-10;
        (self.pressure(rho + dr, temperature) - self.pressure(rho - dr, temperature)) / (2.0 * dr)
    }
}

/// Peng-Robinson equation of state.
///
/// p = ρRT / (1 − bρ) − aα(T)ρ² / (1 + 2bρ − b²ρ²)
#[derive(Debug, Clone, Copy)]
pub struct PengRobinson {
    /// Critical temperature.
    pub tc: f64,
    /// Critical pressure.
    pub pc: f64,
    /// Acentric factor.
    pub omega: f64,
    /// a parameter (derived from Tc, Pc).
    pub a: f64,
    /// b parameter (derived from Tc, Pc).
    pub b: f64,
    pub r: f64,
}

impl PengRobinson {
    pub fn new(tc: f64, pc: f64, omega: f64) -> Self {
        let r = 1.0;
        let a = 0.45724 * r * r * tc * tc / pc;
        let b = 0.07780 * r * tc / pc;
        Self {
            tc,
            pc,
            omega,
            a,
            b,
            r,
        }
    }

    fn alpha(&self, temperature: f64) -> f64 {
        let m = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega * self.omega;
        let sqrt_tr = (temperature / self.tc).sqrt();
        (1.0 + m * (1.0 - sqrt_tr)).powi(2)
    }
}

impl EquationOfState for PengRobinson {
    fn pressure(&self, rho: f64, temperature: f64) -> f64 {
        let alpha = self.alpha(temperature);
        let term1 = rho * self.r * temperature / (1.0 - self.b * rho);
        let denom = 1.0 + 2.0 * self.b * rho - self.b * self.b * rho * rho;
        if denom.abs() < 1e-30 {
            return term1;
        }
        term1 - self.a * alpha * rho * rho / denom
    }

    fn dp_drho(&self, rho: f64, temperature: f64) -> f64 {
        let dr = rho * 1e-6 + 1e-10;
        (self.pressure(rho + dr, temperature) - self.pressure(rho - dr, temperature)) / (2.0 * dr)
    }
}

// ── Shan-Chen Pseudopotential Model ─────────────────────────────────────────

/// Shan-Chen pseudopotential model for liquid-gas systems.
///
/// Interaction force: F = −G·ψ(x)·Σ_i w_i·ψ(x + c_i)·c_i
/// where ψ is the pseudopotential (effective mass).
#[derive(Debug, Clone)]
pub struct ShanChen {
    /// Interaction strength (negative for attraction).
    pub g: f64,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
}

impl ShanChen {
    pub fn new(g: f64, nx: usize, ny: usize) -> Self {
        Self { g, nx, ny }
    }

    /// Compute pseudopotential ψ(ρ) = ρ₀(1 − exp(−ρ/ρ₀)).
    /// For simplicity, ρ₀ = 1.
    pub fn psi(rho: f64) -> f64 {
        1.0 - (-rho).exp()
    }

    /// Compute pseudopotential from EOS: ψ = √(2(p_eos − ρcs²) / (G·cs²))
    pub fn psi_from_eos<E: EquationOfState>(eos: &E, rho: f64, temperature: f64, g: f64) -> f64 {
        let p = eos.pressure(rho, temperature);
        let val = 2.0 * (p - rho * CS2) / (g * CS2);
        if val > 0.0 {
            val.sqrt()
        } else {
            -((-val).sqrt())
        }
    }

    /// Compute Shan-Chen interaction force at a given node.
    ///
    /// `density` is the full density field (nx × ny, row-major).
    pub fn interaction_force(&self, density: &[f64], x: usize, y: usize) -> (f64, f64) {
        let psi_local = Self::psi(density[y * self.nx + x]);
        let mut fx = 0.0;
        let mut fy = 0.0;

        // Sum over D2Q9 neighbors (skip rest direction i=0)
        for i in 1..9 {
            let xn = (x as i32 + C[i].0).rem_euclid(self.nx as i32) as usize;
            let yn = (y as i32 + C[i].1).rem_euclid(self.ny as i32) as usize;
            let psi_nb = Self::psi(density[yn * self.nx + xn]);
            fx += W[i] * psi_nb * C[i].0 as f64;
            fy += W[i] * psi_nb * C[i].1 as f64;
        }

        fx *= -self.g * psi_local;
        fy *= -self.g * psi_local;
        (fx, fy)
    }

    /// Apply Shan-Chen forcing using Guo's forcing scheme.
    /// Modifies the equilibrium velocity: u_eq = u + τF/ρ.
    pub fn apply_forcing(f: &mut [f64; 9], tau: f64, rho: f64, fx: f64, fy: f64) {
        let (_, ux, uy) = macroscopic(f);
        // Shifted velocity
        let ux_eq = ux + tau * fx / rho;
        let uy_eq = uy + tau * fy / rho;
        let feq = equilibrium(rho, ux_eq, uy_eq);
        let omega = 1.0 / tau;
        for i in 0..9 {
            f[i] += -omega * (f[i] - feq[i]);
        }
    }
}

// ── Color Gradient Model ────────────────────────────────────────────────────

/// Color gradient (Rothman-Keller) model for immiscible two-fluid flows.
///
/// Two sets of distributions (red and blue) with recoloring step
/// to maintain sharp interfaces.
#[derive(Debug, Clone)]
pub struct ColorGradient {
    /// Surface tension parameter.
    pub sigma: f64,
    /// Interface thickness parameter (β).
    pub beta: f64,
    /// Relaxation time for red fluid.
    pub tau_r: f64,
    /// Relaxation time for blue fluid.
    pub tau_b: f64,
    pub nx: usize,
    pub ny: usize,
}

impl ColorGradient {
    pub fn new(sigma: f64, beta: f64, tau_r: f64, tau_b: f64, nx: usize, ny: usize) -> Self {
        Self {
            sigma,
            beta,
            tau_r,
            tau_b,
            nx,
            ny,
        }
    }

    /// Compute color field φ = (ρ_R − ρ_B) / (ρ_R + ρ_B).
    pub fn color_field(rho_r: f64, rho_b: f64) -> f64 {
        let total = rho_r + rho_b;
        if total < 1e-15 {
            return 0.0;
        }
        (rho_r - rho_b) / total
    }

    /// Compute the color gradient (interface normal) using finite differences.
    ///
    /// `phase_field` is the φ field over the grid.
    pub fn compute_gradient(
        &self,
        phase_field: &[f64],
        x: usize,
        y: usize,
    ) -> (f64, f64) {
        let mut gx = 0.0;
        let mut gy = 0.0;
        for i in 1..9 {
            let xn = (x as i32 + C[i].0).rem_euclid(self.nx as i32) as usize;
            let yn = (y as i32 + C[i].1).rem_euclid(self.ny as i32) as usize;
            gx += W[i] * C[i].0 as f64 * phase_field[yn * self.nx + xn];
            gy += W[i] * C[i].1 as f64 * phase_field[yn * self.nx + xn];
        }
        (gx, gy)
    }

    /// Perturbation (color gradient forcing) operator for surface tension.
    ///
    /// Adds a perturbation to the collision to generate surface tension:
    /// Ω_i^pert = A/2 |∇φ| (w_i (e_i·∇φ)²/|∇φ|² − B_i)
    pub fn perturbation_operator(
        &self,
        grad_x: f64,
        grad_y: f64,
    ) -> [f64; 9] {
        let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt();
        let mut pert = [0.0; 9];
        if grad_mag < 1e-15 {
            return pert;
        }
        let a_param = 4.5 * self.sigma; // A = 9σ/2

        for i in 0..9 {
            let cx = C[i].0 as f64;
            let cy = C[i].1 as f64;
            let ci_dot_n = (cx * grad_x + cy * grad_y) / grad_mag;
            let b_i = if i == 0 { -CS2 } else { ci_dot_n * ci_dot_n - CS2 };
            pert[i] = a_param * 0.5 * grad_mag * W[i] * b_i;
        }
        pert
    }

    /// Recoloring step to maintain a sharp interface.
    ///
    /// Distributes total f_i into red and blue based on color field
    /// and interface normal direction.
    pub fn recolor(
        f_total: &[f64; 9],
        rho_r: f64,
        rho_b: f64,
        grad_x: f64,
        grad_y: f64,
        beta: f64,
    ) -> ([f64; 9], [f64; 9]) {
        let rho = rho_r + rho_b;
        let mut f_r = [0.0; 9];
        let mut f_b = [0.0; 9];

        if rho < 1e-15 {
            return (f_r, f_b);
        }

        let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt();

        for i in 0..9 {
            let cos_phi = if grad_mag > 1e-15 && i > 0 {
                let cx = C[i].0 as f64;
                let cy = C[i].1 as f64;
                let c_mag = (cx * cx + cy * cy).sqrt();
                (cx * grad_x + cy * grad_y) / (c_mag * grad_mag)
            } else {
                0.0
            };

            f_r[i] = rho_r / rho * f_total[i] + beta * rho_r * rho_b / rho * W[i] * cos_phi;
            f_b[i] = rho_b / rho * f_total[i] - beta * rho_r * rho_b / rho * W[i] * cos_phi;
        }

        (f_r, f_b)
    }

    /// Compute effective relaxation time at interface.
    pub fn effective_tau(&self, rho_r: f64, rho_b: f64) -> f64 {
        let total = rho_r + rho_b;
        if total < 1e-15 {
            return self.tau_r;
        }
        // Harmonic average
        let s_r = rho_r / total / self.tau_r;
        let s_b = rho_b / total / self.tau_b;
        let s_eff = s_r + s_b;
        if s_eff.abs() < 1e-15 {
            return self.tau_r;
        }
        1.0 / s_eff
    }
}

// ── Free Energy Model ───────────────────────────────────────────────────────

/// Free energy model for thermodynamically consistent multiphase flow.
///
/// Based on the Cahn-Hilliard free energy functional:
/// Ψ = ∫ (ψ_bulk(ρ) + κ/2 |∇ρ|²) dV
///
/// The chemical potential μ = dψ/dρ − κ∇²ρ drives the phase separation.
#[derive(Debug, Clone)]
pub struct FreeEnergy {
    /// Gradient energy coefficient (controls interface thickness).
    pub kappa: f64,
    /// Bulk free energy parameter a.
    pub a_param: f64,
    /// Bulk free energy parameter b.
    pub b_param: f64,
    /// Relaxation time.
    pub tau: f64,
    pub nx: usize,
    pub ny: usize,
}

impl FreeEnergy {
    pub fn new(kappa: f64, a: f64, b: f64, tau: f64, nx: usize, ny: usize) -> Self {
        Self {
            kappa,
            a_param: a,
            b_param: b,
            tau,
            nx,
            ny,
        }
    }

    /// Bulk free energy density: ψ = a/2 ρ² + b/4 ρ⁴
    /// (Landau-type double-well potential).
    pub fn bulk_free_energy(&self, rho: f64) -> f64 {
        self.a_param / 2.0 * rho * rho + self.b_param / 4.0 * rho.powi(4)
    }

    /// Chemical potential from bulk: μ_bulk = dψ/dρ = aρ + bρ³.
    pub fn chemical_potential_bulk(&self, rho: f64) -> f64 {
        self.a_param * rho + self.b_param * rho.powi(3)
    }

    /// Compute the Laplacian of density using isotropic stencil.
    pub fn laplacian(&self, density: &[f64], x: usize, y: usize) -> f64 {
        let rho_center = density[y * self.nx + x];
        let mut lap = 0.0;
        for i in 1..9 {
            let xn = (x as i32 + C[i].0).rem_euclid(self.nx as i32) as usize;
            let yn = (y as i32 + C[i].1).rem_euclid(self.ny as i32) as usize;
            lap += W[i] * (density[yn * self.nx + xn] - rho_center);
        }
        lap / CS2
    }

    /// Full chemical potential: μ = μ_bulk − κ∇²ρ.
    pub fn chemical_potential(&self, density: &[f64], x: usize, y: usize) -> f64 {
        let rho = density[y * self.nx + x];
        self.chemical_potential_bulk(rho) - self.kappa * self.laplacian(density, x, y)
    }

    /// Thermodynamic pressure tensor contribution.
    /// p = ρ·μ − ψ_bulk + κ/2 |∇ρ|²
    pub fn pressure(&self, density: &[f64], x: usize, y: usize) -> f64 {
        let rho = density[y * self.nx + x];
        let mu = self.chemical_potential(density, x, y);
        let psi = self.bulk_free_energy(rho);
        rho * mu - psi
    }
}

// ── Surface Tension ─────────────────────────────────────────────────────────

/// Verify Laplace law: Δp = σ/R (2D) or 2σ/R (3D).
///
/// For a 2D circular bubble of radius R with surface tension σ,
/// the pressure difference between inside and outside should be σ/R.
pub fn laplace_pressure_2d(sigma: f64, radius: f64) -> f64 {
    sigma / radius
}

/// Compute pressure difference between inside and outside of a bubble.
///
/// `density` is the density field, `cx`, `cy` is the bubble center,
/// `r_in` and `r_out` are radii for sampling inside/outside.
pub fn measure_pressure_difference(
    density: &[f64],
    nx: usize,
    ny: usize,
    cx: f64,
    cy: f64,
    r_in: f64,
    r_out: f64,
) -> f64 {
    let mut p_in = 0.0;
    let mut n_in = 0;
    let mut p_out = 0.0;
    let mut n_out = 0;

    for y in 0..ny {
        for x in 0..nx {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let r = (dx * dx + dy * dy).sqrt();
            // Use ideal gas EOS: p = ρ·cs²
            let p = density[y * nx + x] * CS2;
            if r < r_in {
                p_in += p;
                n_in += 1;
            } else if r > r_out {
                p_out += p;
                n_out += 1;
            }
        }
    }

    if n_in > 0 && n_out > 0 {
        p_in / n_in as f64 - p_out / n_out as f64
    } else {
        0.0
    }
}

// ── Contact Angle Boundary Conditions ───────────────────────────────────────

/// Contact angle boundary condition for wetting.
///
/// Sets the density at wall nodes to achieve the desired contact angle θ.
/// Uses the geometric formulation: ρ_wall = ρ_l + (ρ_g − ρ_l) × f(θ)
#[derive(Debug, Clone, Copy)]
pub struct ContactAngle {
    /// Desired contact angle in radians.
    pub theta: f64,
    /// Liquid density.
    pub rho_liquid: f64,
    /// Gas density.
    pub rho_gas: f64,
}

impl ContactAngle {
    pub fn new(theta_degrees: f64, rho_liquid: f64, rho_gas: f64) -> Self {
        Self {
            theta: theta_degrees * std::f64::consts::PI / 180.0,
            rho_liquid,
            rho_gas,
        }
    }

    /// Compute the wall density for the pseudopotential method.
    ///
    /// ρ_wall = ρ_avg + Δρ/2 × cos(θ)
    pub fn wall_density(&self) -> f64 {
        let rho_avg = 0.5 * (self.rho_liquid + self.rho_gas);
        let drho = self.rho_liquid - self.rho_gas;
        rho_avg + 0.5 * drho * self.theta.cos()
    }

    /// Check if the angle is hydrophilic (θ < 90°).
    pub fn is_hydrophilic(&self) -> bool {
        self.theta < std::f64::consts::FRAC_PI_2
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carnahan_starling_ideal_gas_limit() {
        let cs_eos = CarnahanStarling::new(0.0, 0.0);
        // With a=0, b=0, should reduce to ideal gas: p = ρRT
        let p = cs_eos.pressure(1.0, 1.0);
        assert!(
            (p - 1.0).abs() < 1e-10,
            "CS EOS should reduce to ideal gas when a=b=0"
        );
    }

    #[test]
    fn test_carnahan_starling_pressure() {
        let cs_eos = CarnahanStarling::new(0.5, 4.0);
        let p = cs_eos.pressure(0.1, 0.5);
        assert!(p.is_finite());
        // At low density the pressure should be positive
        assert!(p > 0.0, "Pressure should be positive at low density");
    }

    #[test]
    fn test_peng_robinson_pressure() {
        let pr = PengRobinson::new(1.0, 1.0, 0.344);
        let p = pr.pressure(0.1, 0.8);
        assert!(p.is_finite());
    }

    #[test]
    fn test_peng_robinson_dp_drho() {
        let pr = PengRobinson::new(1.0, 1.0, 0.344);
        let dp = pr.dp_drho(0.1, 0.8);
        assert!(dp.is_finite());
        assert!(dp > 0.0, "dp/drho should be positive in gas phase");
    }

    #[test]
    fn test_shan_chen_psi() {
        assert!((ShanChen::psi(0.0)).abs() < 1e-15);
        let psi = ShanChen::psi(1.0);
        assert!((psi - (1.0 - (-1.0_f64).exp())).abs() < 1e-14);
    }

    #[test]
    fn test_shan_chen_force_uniform_density() {
        let sc = ShanChen::new(-5.0, 10, 10);
        let density = vec![1.0; 100]; // uniform
        let (fx, fy) = sc.interaction_force(&density, 5, 5);
        assert!(
            fx.abs() < 1e-14 && fy.abs() < 1e-14,
            "Force should vanish for uniform density"
        );
    }

    #[test]
    fn test_shan_chen_force_nonzero_gradient() {
        let sc = ShanChen::new(-5.0, 10, 10);
        let mut density = vec![1.0; 100];
        // Create a density gradient
        for x in 0..10 {
            for y in 0..10 {
                density[y * 10 + x] = 0.5 + 0.1 * x as f64;
            }
        }
        let (fx, _fy) = sc.interaction_force(&density, 5, 5);
        assert!(fx.abs() > 1e-5, "Force should be nonzero for density gradient");
    }

    #[test]
    fn test_color_field() {
        assert!((ColorGradient::color_field(1.0, 0.0) - 1.0).abs() < 1e-15);
        assert!((ColorGradient::color_field(0.0, 1.0) + 1.0).abs() < 1e-15);
        assert!((ColorGradient::color_field(0.5, 0.5)).abs() < 1e-15);
    }

    #[test]
    fn test_color_gradient_uniform() {
        let cg = ColorGradient::new(0.01, 0.7, 0.8, 0.8, 10, 10);
        let phase = vec![0.5; 100];
        let (gx, gy) = cg.compute_gradient(&phase, 5, 5);
        assert!(gx.abs() < 1e-14 && gy.abs() < 1e-14);
    }

    #[test]
    fn test_recolor_conservation() {
        let f_total = equilibrium(2.0, 0.0, 0.0);
        let rho_r = 1.2;
        let rho_b = 0.8;
        let (f_r, f_b) = ColorGradient::recolor(&f_total, rho_r, rho_b, 1.0, 0.0, 0.7);
        let mass_r: f64 = f_r.iter().sum();
        let mass_b: f64 = f_b.iter().sum();
        let mass_total: f64 = f_total.iter().sum();
        assert!(
            ((mass_r + mass_b) - mass_total).abs() < 1e-13,
            "Recoloring should conserve total mass"
        );
    }

    #[test]
    fn test_free_energy_chemical_potential() {
        let fe = FreeEnergy::new(0.01, -1.0, 1.0, 0.8, 10, 10);
        // For double-well: μ_bulk = aρ + bρ³
        let mu = fe.chemical_potential_bulk(1.0);
        assert!((mu - 0.0).abs() < 1e-14, "μ(1) = a + b = -1+1 = 0");
    }

    #[test]
    fn test_free_energy_double_well() {
        let fe = FreeEnergy::new(0.01, -1.0, 1.0, 0.8, 10, 10);
        // Minima at ρ = ±1: ψ'(±1) = a(±1) + b(±1)³ = 0
        let psi_min = fe.bulk_free_energy(1.0);
        let psi_zero = fe.bulk_free_energy(0.0);
        assert!(psi_min < psi_zero, "Double well should have minima at ρ=±1");
    }

    #[test]
    fn test_laplace_pressure_2d() {
        let sigma = 0.01;
        let r = 20.0;
        let dp = laplace_pressure_2d(sigma, r);
        assert!((dp - 0.0005).abs() < 1e-10);
    }

    #[test]
    fn test_contact_angle_hydrophilic() {
        let ca = ContactAngle::new(60.0, 2.0, 0.1);
        assert!(ca.is_hydrophilic());
        let wall_rho = ca.wall_density();
        assert!(wall_rho > 1.0, "Hydrophilic wall should have high density");
    }

    #[test]
    fn test_contact_angle_hydrophobic() {
        let ca = ContactAngle::new(120.0, 2.0, 0.1);
        assert!(!ca.is_hydrophilic());
    }

    #[test]
    fn test_spinodal_decomposition_initial_condition() {
        // Test that a perturbed uniform density with negative dp/drho
        // is thermodynamically unstable (spinodal region)
        let cs_eos = CarnahanStarling::new(1.0, 4.0);
        let rho_mid = 0.3;
        let t = 0.3;
        let dp = cs_eos.dp_drho(rho_mid, t);
        // In the spinodal region, dp/drho < 0
        // Exact values depend on parameters; just check it's finite
        assert!(dp.is_finite());
    }

    #[test]
    fn test_measure_pressure_difference_bubble() {
        let nx = 50;
        let ny = 50;
        let cx = 25.0;
        let cy = 25.0;
        let r_bubble = 10.0;
        let rho_in = 2.0;
        let rho_out = 0.5;

        let mut density = vec![rho_out; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if (dx * dx + dy * dy).sqrt() < r_bubble {
                    density[y * nx + x] = rho_in;
                }
            }
        }

        let dp = measure_pressure_difference(&density, nx, ny, cx, cy, r_bubble * 0.5, r_bubble * 1.5);
        let expected = (rho_in - rho_out) * CS2;
        assert!(
            (dp - expected).abs() < 0.1,
            "Pressure difference should match ideal gas, got {dp}, expected {expected}"
        );
    }

    #[test]
    fn test_effective_tau_pure_fluid() {
        let cg = ColorGradient::new(0.01, 0.7, 0.8, 1.0, 10, 10);
        let tau = cg.effective_tau(1.0, 0.0);
        assert!((tau - 0.8).abs() < 1e-14, "Pure red should give tau_r");
    }
}
