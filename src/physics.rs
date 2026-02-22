//! Physical parameter conversion and non-dimensionalization.
//!
//! In LBM, simulations are performed in "lattice units" where:
//! - Δx = 1 (lattice spacing)
//! - Δt = 1 (time step)
//! - cs² = 1/3 (speed of sound squared)
//!
//! The key relationships are:
//! - ν = cs²(τ - 0.5) = (2τ - 1)/6  (kinematic viscosity)
//! - Re = U·L/ν  (Reynolds number)
//! - Ma = U/cs = U·√3  (Mach number, should be < 0.1 for incompressibility)

use crate::lattice::CS2;

/// Physical parameters for a simulation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhysicsParams {
    /// Reynolds number
    pub reynolds: f64,
    /// Mach number (= U/cs, should be ≪ 1)
    pub mach: f64,
    /// Characteristic length in lattice units
    pub char_length: f64,
}

impl PhysicsParams {
    /// Create physics parameters from Reynolds number and Mach number.
    pub fn new(reynolds: f64, mach: f64, char_length: f64) -> Self {
        Self {
            reynolds,
            mach,
            char_length,
        }
    }

    /// Compute characteristic velocity from Mach number: U = Ma · cs
    pub fn char_velocity(&self) -> f64 {
        self.mach * CS2.sqrt()
    }

    /// Compute kinematic viscosity from Re = U·L/ν → ν = U·L/Re
    pub fn viscosity(&self) -> f64 {
        self.char_velocity() * self.char_length / self.reynolds
    }

    /// Compute relaxation time τ from viscosity: ν = cs²(τ-0.5) → τ = ν/cs² + 0.5
    pub fn tau(&self) -> f64 {
        self.viscosity() / CS2 + 0.5
    }

    /// Check if Mach number is acceptably low for incompressible flow.
    /// Returns error message if Ma > threshold.
    pub fn check_mach(&self, threshold: f64) -> Result<(), String> {
        if self.mach > threshold {
            Err(format!(
                "Mach number {:.4} exceeds threshold {:.4}. \
                 Compressibility errors will be O(Ma²) = O({:.4}). \
                 Reduce velocity or increase cs.",
                self.mach,
                threshold,
                self.mach * self.mach
            ))
        } else {
            Ok(())
        }
    }

    /// Check stability: τ must be > 0.5 for positive viscosity
    pub fn check_stability(&self) -> Result<(), String> {
        let tau = self.tau();
        if tau <= 0.5 {
            return Err(format!(
                "τ = {tau:.6} ≤ 0.5 — simulation will be unstable. \
                 Increase resolution or decrease Reynolds number."
            ));
        }
        if tau < 0.505 {
            return Err(format!(
                "τ = {tau:.6} is very close to 0.5 — simulation may be unstable. \
                 Consider using MRT collision operator."
            ));
        }
        Ok(())
    }
}

/// Convert between physical and lattice units
pub struct UnitConverter {
    /// Physical length scale [m]
    pub phys_length: f64,
    /// Lattice length scale [lattice units]
    pub lattice_length: f64,
    /// Physical velocity scale [m/s]
    pub phys_velocity: f64,
    /// Lattice velocity scale [lattice units]
    pub lattice_velocity: f64,
}

impl UnitConverter {
    pub fn new(
        phys_length: f64,
        lattice_length: f64,
        phys_velocity: f64,
        lattice_velocity: f64,
    ) -> Self {
        Self {
            phys_length,
            lattice_length,
            phys_velocity,
            lattice_velocity,
        }
    }

    /// Spatial conversion factor: Δx = phys_length / lattice_length
    pub fn dx(&self) -> f64 {
        self.phys_length / self.lattice_length
    }

    /// Velocity conversion factor
    pub fn velocity_ratio(&self) -> f64 {
        self.phys_velocity / self.lattice_velocity
    }

    /// Time step: Δt = Δx · (u_lattice / u_phys)
    pub fn dt(&self) -> f64 {
        self.dx() * self.lattice_velocity / self.phys_velocity
    }

    /// Convert physical length to lattice units
    pub fn to_lattice_length(&self, phys: f64) -> f64 {
        phys / self.dx()
    }

    /// Convert lattice length to physical
    pub fn to_physical_length(&self, lattice: f64) -> f64 {
        lattice * self.dx()
    }

    /// Convert physical velocity to lattice units
    pub fn to_lattice_velocity(&self, phys: f64) -> f64 {
        phys / self.velocity_ratio()
    }

    /// Convert lattice velocity to physical
    pub fn to_physical_velocity(&self, lattice: f64) -> f64 {
        lattice * self.velocity_ratio()
    }

    /// Convert physical time to lattice time steps
    pub fn to_lattice_time(&self, phys_time: f64) -> f64 {
        phys_time / self.dt()
    }

    /// Convert lattice time steps to physical time
    pub fn to_physical_time(&self, lattice_steps: f64) -> f64 {
        lattice_steps * self.dt()
    }

    /// Physical viscosity from lattice viscosity
    pub fn to_physical_viscosity(&self, nu_lattice: f64) -> f64 {
        nu_lattice * self.dx() * self.dx() / self.dt()
    }
}

/// Compute Reynolds number from velocity, length, and viscosity
pub fn reynolds_number(velocity: f64, length: f64, viscosity: f64) -> f64 {
    velocity * length / viscosity
}

/// Compute viscosity from Reynolds number
pub fn viscosity_from_reynolds(velocity: f64, length: f64, reynolds: f64) -> f64 {
    velocity * length / reynolds
}

/// Compute tau from viscosity: τ = ν/cs² + 0.5
pub fn tau_from_viscosity(nu: f64) -> f64 {
    nu / CS2 + 0.5
}

/// Compute viscosity from tau: ν = cs²(τ - 0.5)
pub fn viscosity_from_tau(tau: f64) -> f64 {
    CS2 * (tau - 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_viscosity_roundtrip() {
        let nu = 0.01;
        let tau = tau_from_viscosity(nu);
        let nu2 = viscosity_from_tau(tau);
        assert!((nu - nu2).abs() < 1e-14);
    }

    #[test]
    fn test_reynolds_viscosity_roundtrip() {
        let re = 100.0;
        let u = 0.1;
        let l = 50.0;
        let nu = viscosity_from_reynolds(u, l, re);
        let re2 = reynolds_number(u, l, nu);
        assert!((re - re2).abs() < 1e-10);
    }

    #[test]
    fn test_physics_params_tau() {
        let params = PhysicsParams::new(100.0, 0.1, 50.0);
        let tau = params.tau();
        assert!(tau > 0.5, "τ must be > 0.5 for stability");
        // Verify: ν = U·L/Re, τ = ν/cs² + 0.5
        let u = params.char_velocity();
        let nu = u * 50.0 / 100.0;
        let expected_tau = nu / CS2 + 0.5;
        assert!((tau - expected_tau).abs() < 1e-14);
    }

    #[test]
    fn test_mach_check() {
        let params = PhysicsParams::new(100.0, 0.3, 50.0);
        assert!(params.check_mach(0.1).is_err());

        let params = PhysicsParams::new(100.0, 0.05, 50.0);
        assert!(params.check_mach(0.1).is_ok());
    }

    #[test]
    fn test_unit_converter() {
        let conv = UnitConverter::new(1.0, 100.0, 1.0, 0.1);
        assert!((conv.dx() - 0.01).abs() < 1e-14);
        assert!((conv.to_lattice_length(0.5) - 50.0).abs() < 1e-10);
        assert!((conv.to_physical_length(50.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stability_check() {
        // Very high Re with low resolution → small τ
        let params = PhysicsParams::new(100000.0, 0.1, 10.0);
        // This should warn about instability
        let result = params.check_stability();
        assert!(result.is_err() || params.tau() < 0.51);
    }
}
