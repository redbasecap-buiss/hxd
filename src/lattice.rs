//! Lattice Boltzmann Method core: D2Q9 and D3Q19 models, collision operators, streaming.
//!
//! # Lattice Models
//!
//! ## D2Q9 (2D, 9 velocities)
//! Discrete velocities: e_i = { (0,0), (±1,0), (0,±1), (±1,±1) }
//! Weights: w_0 = 4/9, w_{1..4} = 1/9, w_{5..8} = 1/36
//!
//! ## D3Q19 (3D, 19 velocities)
//! Standard 3D lattice with 19 discrete velocities.

use rayon::prelude::*;

// ── D2Q9 Constants ──────────────────────────────────────────────────────────

/// D2Q9 discrete velocity vectors (x, y)
pub const D2Q9_E: [(i32, i32); 9] = [
    (0, 0),   // 0: rest
    (1, 0),   // 1: east
    (0, 1),   // 2: north
    (-1, 0),  // 3: west
    (0, -1),  // 4: south
    (1, 1),   // 5: north-east
    (-1, 1),  // 6: north-west
    (-1, -1), // 7: south-west
    (1, -1),  // 8: south-east
];

/// D2Q9 weights
pub const D2Q9_W: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// D2Q9 opposite direction indices (for bounce-back)
pub const D2Q9_OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Speed of sound squared for D2Q9: cs² = 1/3
pub const CS2: f64 = 1.0 / 3.0;

// ── D3Q19 Constants ─────────────────────────────────────────────────────────

/// D3Q19 discrete velocity vectors (x, y, z)
pub const D3Q19_E: [(i32, i32, i32); 19] = [
    (0, 0, 0),   // 0: rest
    (1, 0, 0),   // 1
    (-1, 0, 0),  // 2
    (0, 1, 0),   // 3
    (0, -1, 0),  // 4
    (0, 0, 1),   // 5
    (0, 0, -1),  // 6
    (1, 1, 0),   // 7
    (-1, 1, 0),  // 8
    (1, -1, 0),  // 9
    (-1, -1, 0), // 10
    (1, 0, 1),   // 11
    (-1, 0, 1),  // 12
    (1, 0, -1),  // 13
    (-1, 0, -1), // 14
    (0, 1, 1),   // 15
    (0, -1, 1),  // 16
    (0, 1, -1),  // 17
    (0, -1, -1), // 18
];

/// D3Q19 weights
pub const D3Q19_W: [f64; 19] = [
    1.0 / 3.0, // rest
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0, // axis-aligned
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0, // diagonal xy
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0, // diagonal xz
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0, // diagonal yz
];

/// D3Q19 opposite direction indices
pub const D3Q19_OPP: [usize; 19] = [
    0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
];

// ── Collision Type ──────────────────────────────────────────────────────────

// ── D3Q27 Constants ─────────────────────────────────────────────────────────

/// D3Q27 discrete velocity vectors (x, y, z)
pub const D3Q27_E: [(i32, i32, i32); 27] = [
    (0, 0, 0),    // 0: rest
    (1, 0, 0),    // 1
    (-1, 0, 0),   // 2
    (0, 1, 0),    // 3
    (0, -1, 0),   // 4
    (0, 0, 1),    // 5
    (0, 0, -1),   // 6
    (1, 1, 0),    // 7
    (-1, 1, 0),   // 8
    (1, -1, 0),   // 9
    (-1, -1, 0),  // 10
    (1, 0, 1),    // 11
    (-1, 0, 1),   // 12
    (1, 0, -1),   // 13
    (-1, 0, -1),  // 14
    (0, 1, 1),    // 15
    (0, -1, 1),   // 16
    (0, 1, -1),   // 17
    (0, -1, -1),  // 18
    (1, 1, 1),    // 19
    (-1, 1, 1),   // 20
    (1, -1, 1),   // 21
    (-1, -1, 1),  // 22
    (1, 1, -1),   // 23
    (-1, 1, -1),  // 24
    (1, -1, -1),  // 25
    (-1, -1, -1), // 26
];

/// D3Q27 weights: w_0 = 8/27, w_{1..6} = 2/27, w_{7..18} = 1/54, w_{19..26} = 1/216
pub const D3Q27_W: [f64; 27] = [
    8.0 / 27.0,  // rest
    2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, // axis
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, // edge xy
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, // edge xz
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, // edge yz
    1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, // corner +z
    1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, // corner -z
];

/// D3Q27 opposite direction indices
pub const D3Q27_OPP: [usize; 27] = [
    0,  2,  1,  4,  3,  6,  5,   // rest, axis
    10, 9,  8,  7,               // edge xy
    14, 13, 12, 11,              // edge xz
    18, 17, 16, 15,              // edge yz
    26, 25, 24, 23, 22, 21, 20, 19, // corners
];

// ── Collision Type ──────────────────────────────────────────────────────────

/// Collision operator selection
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CollisionOperator {
    Bgk,
    Mrt,
    /// Two-Relaxation-Time: uses symmetric (τ+) and antisymmetric (τ-) relaxation.
    /// τ+ controls viscosity, τ- is a free parameter for stability.
    /// Magic parameter Λ = (τ+ - 0.5)(τ- - 0.5) ≈ 1/4 gives wall-location independence.
    Trt,
    /// Cumulant collision: relaxation in cumulant space.
    /// State-of-the-art operator with Galilean invariance and adjustable bulk viscosity.
    /// Reference: Geier et al. (2015). "The cumulant lattice Boltzmann equation in three
    /// dimensions: Theory and validation."
    Cumulant,
}

// ── D2Q9 Lattice ────────────────────────────────────────────────────────────

/// 2D Lattice Boltzmann field on a D2Q9 lattice.
///
/// Distribution functions are stored as `f[q * nx * ny + y * nx + x]`
/// for cache-friendly streaming along x.
#[derive(Clone)]
pub struct Lattice2D {
    pub nx: usize,
    pub ny: usize,
    pub f: Vec<f64>,
    pub f_tmp: Vec<f64>,
    /// Relaxation parameter τ (related to viscosity: ν = cs²(τ - 0.5))
    pub tau: f64,
    pub collision: CollisionOperator,
    /// Smagorinsky constant (0 = no LES)
    pub c_smag: f64,
}

impl Lattice2D {
    /// Create a new 2D lattice initialized to equilibrium at rest with ρ=1.
    pub fn new(nx: usize, ny: usize, tau: f64, collision: CollisionOperator) -> Self {
        let n = 9 * nx * ny;
        let mut f = vec![0.0; n];
        // Initialize to equilibrium at rest (ρ=1, u=0)
        for y in 0..ny {
            for x in 0..nx {
                for q in 0..9 {
                    f[q * nx * ny + y * nx + x] = D2Q9_W[q];
                }
            }
        }
        Self {
            nx,
            ny,
            f: f.clone(),
            f_tmp: f,
            tau,
            collision,
            c_smag: 0.0,
        }
    }

    #[inline]
    pub fn idx(&self, q: usize, x: usize, y: usize) -> usize {
        q * self.nx * self.ny + y * self.nx + x
    }

    /// Compute macroscopic density at (x, y)
    #[inline]
    pub fn density(&self, x: usize, y: usize) -> f64 {
        let mut rho = 0.0;
        for q in 0..9 {
            rho += self.f[self.idx(q, x, y)];
        }
        rho
    }

    /// Compute macroscopic velocity at (x, y)
    #[inline]
    pub fn velocity(&self, x: usize, y: usize) -> (f64, f64) {
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        for q in 0..9 {
            let fi = self.f[self.idx(q, x, y)];
            rho += fi;
            ux += fi * D2Q9_E[q].0 as f64;
            uy += fi * D2Q9_E[q].1 as f64;
        }
        if rho > 1e-15 {
            (ux / rho, uy / rho)
        } else {
            (0.0, 0.0)
        }
    }

    /// Compute equilibrium distribution for D2Q9
    ///
    /// f_eq_i = w_i * ρ * (1 + (e_i · u)/cs² + (e_i · u)²/(2·cs⁴) - u·u/(2·cs²))
    #[inline]
    pub fn equilibrium(rho: f64, ux: f64, uy: f64) -> [f64; 9] {
        let usq = ux * ux + uy * uy;
        let mut feq = [0.0; 9];
        for q in 0..9 {
            let eu = D2Q9_E[q].0 as f64 * ux + D2Q9_E[q].1 as f64 * uy;
            feq[q] = D2Q9_W[q]
                * rho
                * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - usq / (2.0 * CS2));
        }
        feq
    }

    /// Compute the local strain rate tensor magnitude |S| for Smagorinsky model.
    #[inline]
    #[allow(dead_code)]
    fn strain_rate_magnitude(&self, x: usize, y: usize, rho: f64, ux: f64, uy: f64) -> f64 {
        let feq = Self::equilibrium(rho, ux, uy);
        let mut pi_neq_xx = 0.0;
        let mut pi_neq_xy = 0.0;
        let mut pi_neq_yy = 0.0;
        for q in 0..9 {
            let fi = self.f[self.idx(q, x, y)];
            let fneq = fi - feq[q];
            let ex = D2Q9_E[q].0 as f64;
            let ey = D2Q9_E[q].1 as f64;
            pi_neq_xx += ex * ex * fneq;
            pi_neq_xy += ex * ey * fneq;
            pi_neq_yy += ey * ey * fneq;
        }
        let q_tensor = pi_neq_xx * pi_neq_xx + 2.0 * pi_neq_xy * pi_neq_xy + pi_neq_yy * pi_neq_yy;
        q_tensor.sqrt()
    }

    /// Compute effective tau with Smagorinsky LES model
    #[inline]
    #[allow(dead_code)]
    fn effective_tau(&self, x: usize, y: usize, rho: f64, ux: f64, uy: f64) -> f64 {
        if self.c_smag <= 0.0 {
            return self.tau;
        }
        let s_mag = self.strain_rate_magnitude(x, y, rho, ux, uy);
        let tau0 = self.tau;
        // τ_eff = 0.5 * (τ + √(τ² + 18·C_s²·|Π_neq| / (ρ·cs⁴)))
        let discriminant =
            tau0 * tau0 + 18.0 * self.c_smag * self.c_smag * s_mag / (rho * CS2 * CS2);
        0.5 * (tau0 + discriminant.sqrt())
    }

    /// BGK collision: f_i ← f_i - (f_i - f_i^eq) / τ
    pub fn collide_bgk(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let f = &mut self.f;
        let tau = self.tau;
        let c_smag = self.c_smag;

        // We process rows in parallel
        let row_data: Vec<(usize, Vec<f64>)> = (0..ny)
            .into_par_iter()
            .map(|y| {
                let mut row_f = vec![0.0; 9 * nx];
                for x in 0..nx {
                    let mut rho = 0.0;
                    let mut ux = 0.0;
                    let mut uy = 0.0;
                    let mut fi = [0.0; 9];
                    for q in 0..9 {
                        let val = f[q * nx * ny + y * nx + x];
                        fi[q] = val;
                        rho += val;
                        ux += val * D2Q9_E[q].0 as f64;
                        uy += val * D2Q9_E[q].1 as f64;
                    }
                    if rho > 1e-15 {
                        ux /= rho;
                        uy /= rho;
                    }

                    let effective_tau = if c_smag > 0.0 {
                        let feq = Self::equilibrium(rho, ux, uy);
                        let mut pi_neq_xx = 0.0;
                        let mut pi_neq_xy = 0.0;
                        let mut pi_neq_yy = 0.0;
                        for q in 0..9 {
                            let fneq = fi[q] - feq[q];
                            let ex = D2Q9_E[q].0 as f64;
                            let ey = D2Q9_E[q].1 as f64;
                            pi_neq_xx += ex * ex * fneq;
                            pi_neq_xy += ex * ey * fneq;
                            pi_neq_yy += ey * ey * fneq;
                        }
                        let q_tensor = pi_neq_xx * pi_neq_xx
                            + 2.0 * pi_neq_xy * pi_neq_xy
                            + pi_neq_yy * pi_neq_yy;
                        let s_mag = q_tensor.sqrt();
                        let discriminant =
                            tau * tau + 18.0 * c_smag * c_smag * s_mag / (rho * CS2 * CS2);
                        0.5 * (tau + discriminant.sqrt())
                    } else {
                        tau
                    };

                    let feq = Self::equilibrium(rho, ux, uy);
                    let omega = 1.0 / effective_tau;
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
                    f[q * nx * ny + y * nx + x] = row_f[q * nx + x];
                }
            }
        }
    }

    /// MRT collision operator for D2Q9.
    ///
    /// Transforms to moment space, relaxes each moment independently, transforms back.
    /// Moment basis: {ρ, e, ε, jx, qx, jy, qy, pxx, pxy}
    pub fn collide_mrt(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let tau = self.tau;
        let omega_nu = 1.0 / tau;

        // MRT relaxation rates (standard choices for stability)
        let s = [
            0.0,      // s0: density (conserved)
            1.4,      // s1: energy
            1.4,      // s2: energy square
            0.0,      // s3: jx (conserved)
            1.2,      // s4: qx
            0.0,      // s5: jy (conserved)
            1.2,      // s6: qy
            omega_nu, // s7: pxx (viscous)
            omega_nu, // s8: pxy (viscous)
        ];

        // Transformation matrix M for D2Q9 (Lallemand & Luo, 2000)
        #[rustfmt::skip]
        let m_mat: [[f64; 9]; 9] = [
            [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], // ρ
            [-4.0, -1.0, -1.0, -1.0, -1.0,  2.0,  2.0,  2.0,  2.0], // e
            [ 4.0, -2.0, -2.0, -2.0, -2.0,  1.0,  1.0,  1.0,  1.0], // ε
            [ 0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0], // jx
            [ 0.0, -2.0,  0.0,  2.0,  0.0,  1.0, -1.0, -1.0,  1.0], // qx
            [ 0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0], // jy
            [ 0.0,  0.0, -2.0,  0.0,  2.0,  1.0,  1.0, -1.0, -1.0], // qy
            [ 0.0,  1.0, -1.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0], // pxx
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0, -1.0], // pxy
        ];

        // Inverse transformation matrix M^-1 (precomputed)
        #[rustfmt::skip]
        let m_inv: [[f64; 9]; 9] = [
            [ 1.0/9.0, -1.0/9.0,  1.0/9.0,  0.0,      0.0,      0.0,      0.0,      0.0,      0.0     ],
            [ 1.0/9.0, -1.0/36.0,-2.0/36.0, 1.0/6.0, -1.0/6.0,  0.0,      0.0,      1.0/4.0,  0.0     ],
            [ 1.0/9.0, -1.0/36.0,-2.0/36.0, 0.0,      0.0,      1.0/6.0, -1.0/6.0, -1.0/4.0,  0.0     ],
            [ 1.0/9.0, -1.0/36.0,-2.0/36.0,-1.0/6.0,  1.0/6.0,  0.0,      0.0,      1.0/4.0,  0.0     ],
            [ 1.0/9.0, -1.0/36.0,-2.0/36.0, 0.0,      0.0,     -1.0/6.0,  1.0/6.0, -1.0/4.0,  0.0     ],
            [ 1.0/9.0,  2.0/36.0, 1.0/36.0, 1.0/6.0,  1.0/12.0, 1.0/6.0,  1.0/12.0, 0.0,      1.0/4.0 ],
            [ 1.0/9.0,  2.0/36.0, 1.0/36.0,-1.0/6.0, -1.0/12.0, 1.0/6.0,  1.0/12.0, 0.0,     -1.0/4.0 ],
            [ 1.0/9.0,  2.0/36.0, 1.0/36.0,-1.0/6.0, -1.0/12.0,-1.0/6.0, -1.0/12.0, 0.0,      1.0/4.0 ],
            [ 1.0/9.0,  2.0/36.0, 1.0/36.0, 1.0/6.0,  1.0/12.0,-1.0/6.0, -1.0/12.0, 0.0,     -1.0/4.0 ],
        ];

        let f = &mut self.f;
        let row_data: Vec<(usize, Vec<f64>)> = (0..ny)
            .into_par_iter()
            .map(|y| {
                let mut row_f = vec![0.0; 9 * nx];
                for x in 0..nx {
                    // Get populations
                    let mut fi = [0.0; 9];
                    for q in 0..9 {
                        fi[q] = f[q * nx * ny + y * nx + x];
                    }

                    // Transform to moment space: m = M · f
                    let mut m = [0.0; 9];
                    for i in 0..9 {
                        for j in 0..9 {
                            m[i] += m_mat[i][j] * fi[j];
                        }
                    }

                    // Compute equilibrium moments
                    let rho = m[0];
                    let jx = m[3];
                    let jy = m[5];
                    let mut meq = [0.0; 9];
                    meq[0] = rho;
                    meq[1] = -2.0 * rho + 3.0 * (jx * jx + jy * jy) / rho;
                    meq[2] = rho - 3.0 * (jx * jx + jy * jy) / rho;
                    meq[3] = jx;
                    meq[4] = -jx;
                    meq[5] = jy;
                    meq[6] = -jy;
                    meq[7] = (jx * jx - jy * jy) / rho;
                    meq[8] = jx * jy / rho;

                    // Relax in moment space: m* = m - S(m - meq)
                    for i in 0..9 {
                        m[i] -= s[i] * (m[i] - meq[i]);
                    }

                    // Transform back: f = M^-1 · m*
                    for i in 0..9 {
                        let mut val = 0.0;
                        for j in 0..9 {
                            val += m_inv[i][j] * m[j];
                        }
                        row_f[i * nx + x] = val;
                    }
                }
                (y, row_f)
            })
            .collect();

        for (y, row_f) in row_data {
            for x in 0..nx {
                for q in 0..9 {
                    f[q * nx * ny + y * nx + x] = row_f[q * nx + x];
                }
            }
        }
    }

    /// TRT collision operator: Two-Relaxation-Time.
    ///
    /// Decomposes distributions into symmetric and antisymmetric parts:
    ///   f_i^+ = (f_i + f_ī) / 2,  f_i^- = (f_i - f_ī) / 2
    /// and relaxes each with its own rate:
    ///   f_i* = f_i - (f_i^+ - f_i^{eq+}) / τ+ - (f_i^- - f_i^{eq-}) / τ-
    ///
    /// τ+ = τ (controls viscosity), τ- chosen via magic parameter Λ:
    ///   Λ = (τ+ - 0.5)(τ- - 0.5)
    /// Λ = 1/4 gives exact wall location for bounce-back (recommended).
    ///
    /// Reference: Ginzburg, I. (2005). "Equilibrium-type and link-type lattice Boltzmann models
    /// for generic advection and anisotropic-dispersion equation."
    pub fn collide_trt(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let tau_plus = self.tau;
        let omega_plus = 1.0 / tau_plus;
        // Magic parameter Λ = 1/4 for wall-location independence
        let lambda = 0.25;
        let tau_minus = 0.5 + lambda / (tau_plus - 0.5);
        let omega_minus = 1.0 / tau_minus;
        let f = &mut self.f;

        let row_data: Vec<(usize, Vec<f64>)> = (0..ny)
            .into_par_iter()
            .map(|y| {
                let mut row_f = vec![0.0; 9 * nx];
                for x in 0..nx {
                    let mut rho = 0.0;
                    let mut ux = 0.0;
                    let mut uy = 0.0;
                    let mut fi = [0.0; 9];
                    for q in 0..9 {
                        let val = f[q * nx * ny + y * nx + x];
                        fi[q] = val;
                        rho += val;
                        ux += val * D2Q9_E[q].0 as f64;
                        uy += val * D2Q9_E[q].1 as f64;
                    }
                    if rho > 1e-15 {
                        ux /= rho;
                        uy /= rho;
                    }

                    let feq = Self::equilibrium(rho, ux, uy);

                    for q in 0..9 {
                        let qbar = D2Q9_OPP[q];
                        // Symmetric and antisymmetric parts
                        let f_plus = 0.5 * (fi[q] + fi[qbar]);
                        let f_minus = 0.5 * (fi[q] - fi[qbar]);
                        let feq_plus = 0.5 * (feq[q] + feq[qbar]);
                        let feq_minus = 0.5 * (feq[q] - feq[qbar]);

                        row_f[q * nx + x] = fi[q]
                            - omega_plus * (f_plus - feq_plus)
                            - omega_minus * (f_minus - feq_minus);
                    }
                }
                (y, row_f)
            })
            .collect();

        for (y, row_f) in row_data {
            for x in 0..nx {
                for q in 0..9 {
                    f[q * nx * ny + y * nx + x] = row_f[q * nx + x];
                }
            }
        }
    }

    /// Cumulant collision operator for D2Q9.
    ///
    /// Transforms populations to central moments, then to cumulants,
    /// relaxes cumulants, and transforms back. Provides Galilean invariance
    /// and tunable bulk viscosity.
    ///
    /// Reference: Geier, M. et al. (2006). "Cascaded digital lattice Boltzmann
    /// automata for high Reynolds number flow." Physical Review E, 73(6).
    pub fn collide_cumulant(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let tau = self.tau;
        let omega = 1.0 / tau;
        // Bulk viscosity relaxation rate (can be tuned; here = shear rate for simplicity)
        let omega_bulk = omega;
        let f = &mut self.f;

        let row_data: Vec<(usize, Vec<f64>)> = (0..ny)
            .into_par_iter()
            .map(|y| {
                let mut row_f = vec![0.0; 9 * nx];
                for x in 0..nx {
                    let mut rho = 0.0;
                    let mut ux = 0.0;
                    let mut uy = 0.0;
                    let mut fi = [0.0; 9];
                    for q in 0..9 {
                        let val = f[q * nx * ny + y * nx + x];
                        fi[q] = val;
                        rho += val;
                        ux += val * D2Q9_E[q].0 as f64;
                        uy += val * D2Q9_E[q].1 as f64;
                    }
                    if rho > 1e-15 {
                        ux /= rho;
                        uy /= rho;
                    }

                    // Compute central moments: κ_ab = Σ fi (ei_x - ux)^a (ei_y - uy)^b
                    let mut k00 = 0.0;
                    let mut _k10 = 0.0;
                    let mut _k01 = 0.0;
                    let mut k20 = 0.0;
                    let mut k02 = 0.0;
                    let mut k11 = 0.0;
                    let mut k21 = 0.0;
                    let mut k12 = 0.0;
                    let mut k22 = 0.0;

                    for q in 0..9 {
                        let cx = D2Q9_E[q].0 as f64 - ux;
                        let cy = D2Q9_E[q].1 as f64 - uy;
                        k00 += fi[q];
                        _k10 += fi[q] * cx;
                        _k01 += fi[q] * cy;
                        k20 += fi[q] * cx * cx;
                        k02 += fi[q] * cy * cy;
                        k11 += fi[q] * cx * cy;
                        k21 += fi[q] * cx * cx * cy;
                        k12 += fi[q] * cx * cy * cy;
                        k22 += fi[q] * cx * cx * cy * cy;
                    }

                    // Normalized central moments (divide by rho)
                    let _nk00 = k00 / rho; // = 1
                    // k10, k01 = 0 by definition of central moments
                    let nk20 = k20 / rho;
                    let nk02 = k02 / rho;
                    let nk11 = k11 / rho;
                    let nk21 = k21 / rho;
                    let nk12 = k12 / rho;
                    let nk22 = k22 / rho;

                    // Cumulants from central moments (for 2D, 2nd order cumulants = central moments)
                    // c_20 = nk20, c_02 = nk02, c_11 = nk11
                    // c_21 = nk21, c_12 = nk12
                    // c_22 = nk22 - nk20*nk02 - 2*nk11*nk11  (connected part)

                    let c20 = nk20;
                    let c02 = nk02;
                    let c11 = nk11;
                    let c21 = nk21;
                    let c12 = nk12;
                    let c22 = nk22 - nk20 * nk02 - 2.0 * nk11 * nk11;

                    // Equilibrium cumulants
                    let c20_eq = CS2; // 1/3
                    let c02_eq = CS2;
                    let c11_eq = 0.0;
                    let c21_eq = 0.0;
                    let c12_eq = 0.0;
                    let c22_eq = 0.0;

                    // Relax cumulants
                    // Trace (bulk): (c20 + c02) relaxed with omega_bulk
                    // Normal stress difference: (c20 - c02) relaxed with omega (shear)
                    // Shear stress c11: relaxed with omega
                    let trace = c20 + c02;
                    let diff = c20 - c02;
                    let trace_eq = c20_eq + c02_eq;
                    let diff_eq = c20_eq - c02_eq;

                    let trace_new = trace - omega_bulk * (trace - trace_eq);
                    let diff_new = diff - omega * (diff - diff_eq);
                    let c11_new = c11 - omega * (c11 - c11_eq);

                    let c20_new = 0.5 * (trace_new + diff_new);
                    let c02_new = 0.5 * (trace_new - diff_new);

                    // Higher-order cumulants: relax to equilibrium with rate 1 (no free parameter)
                    let c21_new = c21 - 1.0 * (c21 - c21_eq);
                    let c12_new = c12 - 1.0 * (c12 - c12_eq);
                    let c22_new = c22 - 1.0 * (c22 - c22_eq);

                    // Convert back: cumulants → normalized central moments
                    let nk20_new = c20_new;
                    let nk02_new = c02_new;
                    let nk11_new = c11_new;
                    let nk21_new = c21_new;
                    let nk12_new = c12_new;
                    let nk22_new = c22_new + nk20_new * nk02_new + 2.0 * nk11_new * nk11_new;

                    // Central moments (multiply by rho)
                    let k20_n = rho * nk20_new;
                    let k02_n = rho * nk02_new;
                    let k11_n = rho * nk11_new;
                    let k21_n = rho * nk21_new;
                    let k12_n = rho * nk12_new;
                    let k22_n = rho * nk22_new;

                    let mut fi_new = [0.0; 9];

                    // Convert central moments back to raw moments:
                    // M_ab = Σ_{p,q} C(a,p) C(b,q) ux^(a-p) uy^(b-q) κ_pq
                    let m00 = rho;
                    let m10 = rho * ux; // k10=0, so m10 = rho*ux
                    let m01 = rho * uy;
                    let m20 = k20_n + rho * ux * ux;
                    let m02 = k02_n + rho * uy * uy;
                    let m11 = k11_n + rho * ux * uy;
                    let m21 = k21_n + 2.0 * ux * k11_n + uy * k20_n + rho * ux * ux * uy;
                    let m12 = k12_n + 2.0 * uy * k11_n + ux * k02_n + rho * ux * uy * uy;
                    let m22 = k22_n
                        + 2.0 * ux * k12_n
                        + 2.0 * uy * k21_n
                        + ux * ux * k02_n
                        + uy * uy * k20_n
                        + 4.0 * ux * uy * k11_n
                        + rho * ux * ux * uy * uy;

                    // D2Q9 raw-moment-to-population inversion:
                    // Derived from m_ab = Σ_q f_q · e_qx^a · e_qy^b
                    // Diagonal populations: m11=f5-f6+f7-f8, m21=f5-f6-f7+f8,
                    //   m12=f5+f6-f7-f8, m22=f5+f6+f7+f8
                    // Axis: f1+f3=m20-m22, f1-f3=m10-m21, f2+f4=m02-m22, f2-f4=m01-m12

                    fi_new[0] = m00 - m20 - m02 + m22;
                    fi_new[1] = 0.5 * ((m20 - m22) + (m10 - m21));
                    fi_new[2] = 0.5 * ((m02 - m22) + (m01 - m12));
                    fi_new[3] = 0.5 * ((m20 - m22) - (m10 - m21));
                    fi_new[4] = 0.5 * ((m02 - m22) - (m01 - m12));
                    fi_new[5] = 0.25 * (m22 + m21 + m12 + m11);
                    fi_new[6] = 0.25 * (m22 - m21 + m12 - m11);
                    fi_new[7] = 0.25 * (m22 - m21 - m12 + m11);
                    fi_new[8] = 0.25 * (m22 + m21 - m12 - m11);

                    for q in 0..9 {
                        row_f[q * nx + x] = fi_new[q];
                    }
                }
                (y, row_f)
            })
            .collect();

        for (y, row_f) in row_data {
            for x in 0..nx {
                for q in 0..9 {
                    f[q * nx * ny + y * nx + x] = row_f[q * nx + x];
                }
            }
        }
    }

    /// Perform collision step based on configured operator
    pub fn collide(&mut self) {
        match self.collision {
            CollisionOperator::Bgk => self.collide_bgk(),
            CollisionOperator::Mrt => self.collide_mrt(),
            CollisionOperator::Trt => self.collide_trt(),
            CollisionOperator::Cumulant => self.collide_cumulant(),
        }
    }

    /// Streaming step: propagate distributions to neighbors.
    /// Uses f_tmp as temporary buffer to avoid read-write conflicts.
    /// Periodic boundary conditions are applied by default.
    pub fn stream(&mut self) {
        let nx = self.nx;
        let ny = self.ny;

        self.f_tmp
            .par_chunks_mut(nx * ny)
            .enumerate()
            .for_each(|(q, chunk)| {
                let ex = D2Q9_E[q].0;
                let ey = D2Q9_E[q].1;
                for y in 0..ny {
                    for x in 0..nx {
                        // Source position (with periodic wrapping)
                        let sx = ((x as i32 - ex).rem_euclid(nx as i32)) as usize;
                        let sy = ((y as i32 - ey).rem_euclid(ny as i32)) as usize;
                        chunk[y * nx + x] = self.f[q * nx * ny + sy * nx + sx];
                    }
                }
            });

        std::mem::swap(&mut self.f, &mut self.f_tmp);
    }

    /// Set distribution at a node to equilibrium with given macroscopic values
    pub fn set_equilibrium(&mut self, x: usize, y: usize, rho: f64, ux: f64, uy: f64) {
        let feq = Self::equilibrium(rho, ux, uy);
        for q in 0..9 {
            let idx = self.idx(q, x, y);
            self.f[idx] = feq[q];
        }
    }

    /// Get all macroscopic fields
    pub fn macroscopic_fields(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.nx * self.ny;
        let mut rho = vec![0.0; n];
        let mut ux = vec![0.0; n];
        let mut uy = vec![0.0; n];

        for y in 0..self.ny {
            for x in 0..self.nx {
                let idx = y * self.nx + x;
                let (r, u, v) = self.macroscopic(x, y);
                rho[idx] = r;
                ux[idx] = u;
                uy[idx] = v;
            }
        }
        (rho, ux, uy)
    }

    #[inline]
    fn macroscopic(&self, x: usize, y: usize) -> (f64, f64, f64) {
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        for q in 0..9 {
            let fi = self.f[self.idx(q, x, y)];
            rho += fi;
            ux += fi * D2Q9_E[q].0 as f64;
            uy += fi * D2Q9_E[q].1 as f64;
        }
        if rho > 1e-15 {
            (rho, ux / rho, uy / rho)
        } else {
            (rho, 0.0, 0.0)
        }
    }

    /// Compute vorticity field (duy/dx - dux/dy) using central differences
    pub fn vorticity_field(&self) -> Vec<f64> {
        let nx = self.nx;
        let ny = self.ny;
        let mut vort = vec![0.0; nx * ny];

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let (_, _, uy_right) = self.macroscopic(x + 1, y);
                let (_, _, uy_left) = self.macroscopic(x.wrapping_sub(1), y);
                let (_, ux_up, _) = self.macroscopic(x, y + 1);
                let (_, ux_down, _) = self.macroscopic(x, y.wrapping_sub(1));
                vort[y * nx + x] = (uy_right - uy_left) / 2.0 - (ux_up - ux_down) / 2.0;
            }
        }
        vort
    }
}

// ── D3Q19 Lattice ───────────────────────────────────────────────────────────

/// 3D Lattice Boltzmann field on a D3Q19 lattice.
#[derive(Clone)]
pub struct Lattice3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub f: Vec<f64>,
    pub f_tmp: Vec<f64>,
    pub tau: f64,
    pub collision: CollisionOperator,
}

impl Lattice3D {
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64, collision: CollisionOperator) -> Self {
        let n = 19 * nx * ny * nz;
        let mut f = vec![0.0; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    for q in 0..19 {
                        f[q * nx * ny * nz + z * nx * ny + y * nx + x] = D3Q19_W[q];
                    }
                }
            }
        }
        Self {
            nx,
            ny,
            nz,
            f: f.clone(),
            f_tmp: f,
            tau,
            collision,
        }
    }

    #[inline]
    pub fn idx(&self, q: usize, x: usize, y: usize, z: usize) -> usize {
        q * self.nx * self.ny * self.nz + z * self.nx * self.ny + y * self.nx + x
    }

    /// Compute macroscopic density at (x, y, z)
    pub fn density(&self, x: usize, y: usize, z: usize) -> f64 {
        let mut rho = 0.0;
        for q in 0..19 {
            rho += self.f[self.idx(q, x, y, z)];
        }
        rho
    }

    /// Compute macroscopic velocity at (x, y, z)
    pub fn velocity(&self, x: usize, y: usize, z: usize) -> (f64, f64, f64) {
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        let mut uz = 0.0;
        for q in 0..19 {
            let fi = self.f[self.idx(q, x, y, z)];
            rho += fi;
            ux += fi * D3Q19_E[q].0 as f64;
            uy += fi * D3Q19_E[q].1 as f64;
            uz += fi * D3Q19_E[q].2 as f64;
        }
        if rho > 1e-15 {
            (ux / rho, uy / rho, uz / rho)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// D3Q19 equilibrium distribution
    pub fn equilibrium(rho: f64, ux: f64, uy: f64, uz: f64) -> [f64; 19] {
        let usq = ux * ux + uy * uy + uz * uz;
        let mut feq = [0.0; 19];
        for q in 0..19 {
            let eu = D3Q19_E[q].0 as f64 * ux + D3Q19_E[q].1 as f64 * uy + D3Q19_E[q].2 as f64 * uz;
            feq[q] = D3Q19_W[q]
                * rho
                * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - usq / (2.0 * CS2));
        }
        feq
    }

    /// BGK collision for 3D
    pub fn collide_bgk(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nxy = nx * ny;
        let tau = self.tau;
        let omega = 1.0 / tau;
        let f = &mut self.f;

        let updates: Vec<(usize, Vec<f64>)> = (0..nz)
            .into_par_iter()
            .map(|z| {
                let mut slice = vec![0.0; 19 * nxy];
                for y in 0..ny {
                    for x in 0..nx {
                        let mut rho = 0.0;
                        let mut ux = 0.0;
                        let mut uy = 0.0;
                        let mut uz = 0.0;
                        let mut fi = [0.0; 19];
                        for q in 0..19 {
                            let val = f[q * nxy * nz + z * nxy + y * nx + x];
                            fi[q] = val;
                            rho += val;
                            ux += val * D3Q19_E[q].0 as f64;
                            uy += val * D3Q19_E[q].1 as f64;
                            uz += val * D3Q19_E[q].2 as f64;
                        }
                        if rho > 1e-15 {
                            ux /= rho;
                            uy /= rho;
                            uz /= rho;
                        }
                        let feq = Self::equilibrium(rho, ux, uy, uz);
                        for q in 0..19 {
                            slice[q * nxy + y * nx + x] = fi[q] - omega * (fi[q] - feq[q]);
                        }
                    }
                }
                (z, slice)
            })
            .collect();

        for (z, slice) in updates {
            for q in 0..19 {
                for y in 0..ny {
                    for x in 0..nx {
                        f[q * nxy * nz + z * nxy + y * nx + x] = slice[q * nxy + y * nx + x];
                    }
                }
            }
        }
    }

    /// Perform collision step
    pub fn collide(&mut self) {
        self.collide_bgk();
    }

    /// Streaming step with periodic boundaries
    pub fn stream(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nxy = nx * ny;

        self.f_tmp
            .par_chunks_mut(nxy * nz)
            .enumerate()
            .for_each(|(q, chunk)| {
                let ex = D3Q19_E[q].0;
                let ey = D3Q19_E[q].1;
                let ez = D3Q19_E[q].2;
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            let sx = ((x as i32 - ex).rem_euclid(nx as i32)) as usize;
                            let sy = ((y as i32 - ey).rem_euclid(ny as i32)) as usize;
                            let sz = ((z as i32 - ez).rem_euclid(nz as i32)) as usize;
                            chunk[z * nxy + y * nx + x] =
                                self.f[q * nxy * nz + sz * nxy + sy * nx + sx];
                        }
                    }
                }
            });

        std::mem::swap(&mut self.f, &mut self.f_tmp);
    }

    /// Set equilibrium at a node
    pub fn set_equilibrium(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        rho: f64,
        ux: f64,
        uy: f64,
        uz: f64,
    ) {
        let feq = Self::equilibrium(rho, ux, uy, uz);
        for q in 0..19 {
            let idx = self.idx(q, x, y, z);
            self.f[idx] = feq[q];
        }
    }
}

// ── D3Q27 Lattice ───────────────────────────────────────────────────────────

/// 3D Lattice Boltzmann field on a D3Q27 lattice.
///
/// The D3Q27 model includes all 27 discrete velocities (rest + 6 axis + 12 edge + 8 corner).
/// It provides higher isotropy than D3Q19 and is required for certain advanced collision
/// operators (e.g., cumulant method in 3D).
#[derive(Clone)]
pub struct Lattice3D27 {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub f: Vec<f64>,
    pub f_tmp: Vec<f64>,
    pub tau: f64,
    pub collision: CollisionOperator,
}

impl Lattice3D27 {
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64, collision: CollisionOperator) -> Self {
        let n = 27 * nx * ny * nz;
        let mut f = vec![0.0; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    for q in 0..27 {
                        f[q * nx * ny * nz + z * nx * ny + y * nx + x] = D3Q27_W[q];
                    }
                }
            }
        }
        Self {
            nx,
            ny,
            nz,
            f: f.clone(),
            f_tmp: f,
            tau,
            collision,
        }
    }

    #[inline]
    pub fn idx(&self, q: usize, x: usize, y: usize, z: usize) -> usize {
        q * self.nx * self.ny * self.nz + z * self.nx * self.ny + y * self.nx + x
    }

    /// Compute macroscopic density
    pub fn density(&self, x: usize, y: usize, z: usize) -> f64 {
        let mut rho = 0.0;
        for q in 0..27 {
            rho += self.f[self.idx(q, x, y, z)];
        }
        rho
    }

    /// Compute macroscopic velocity
    pub fn velocity(&self, x: usize, y: usize, z: usize) -> (f64, f64, f64) {
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        let mut uz = 0.0;
        for q in 0..27 {
            let fi = self.f[self.idx(q, x, y, z)];
            rho += fi;
            ux += fi * D3Q27_E[q].0 as f64;
            uy += fi * D3Q27_E[q].1 as f64;
            uz += fi * D3Q27_E[q].2 as f64;
        }
        if rho > 1e-15 {
            (ux / rho, uy / rho, uz / rho)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// D3Q27 equilibrium distribution
    pub fn equilibrium(rho: f64, ux: f64, uy: f64, uz: f64) -> [f64; 27] {
        let usq = ux * ux + uy * uy + uz * uz;
        let mut feq = [0.0; 27];
        for q in 0..27 {
            let eu =
                D3Q27_E[q].0 as f64 * ux + D3Q27_E[q].1 as f64 * uy + D3Q27_E[q].2 as f64 * uz;
            feq[q] = D3Q27_W[q]
                * rho
                * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - usq / (2.0 * CS2));
        }
        feq
    }

    /// BGK collision for D3Q27
    pub fn collide_bgk(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nxy = nx * ny;
        let tau = self.tau;
        let omega = 1.0 / tau;
        let f = &mut self.f;

        let updates: Vec<(usize, Vec<f64>)> = (0..nz)
            .into_par_iter()
            .map(|z| {
                let mut slice = vec![0.0; 27 * nxy];
                for y in 0..ny {
                    for x in 0..nx {
                        let mut rho = 0.0;
                        let mut ux = 0.0;
                        let mut uy = 0.0;
                        let mut uz = 0.0;
                        let mut fi = [0.0; 27];
                        for q in 0..27 {
                            let val = f[q * nxy * nz + z * nxy + y * nx + x];
                            fi[q] = val;
                            rho += val;
                            ux += val * D3Q27_E[q].0 as f64;
                            uy += val * D3Q27_E[q].1 as f64;
                            uz += val * D3Q27_E[q].2 as f64;
                        }
                        if rho > 1e-15 {
                            ux /= rho;
                            uy /= rho;
                            uz /= rho;
                        }
                        let feq = Self::equilibrium(rho, ux, uy, uz);
                        for q in 0..27 {
                            slice[q * nxy + y * nx + x] = fi[q] - omega * (fi[q] - feq[q]);
                        }
                    }
                }
                (z, slice)
            })
            .collect();

        for (z, slice) in updates {
            for q in 0..27 {
                for y in 0..ny {
                    for x in 0..nx {
                        f[q * nxy * nz + z * nxy + y * nx + x] = slice[q * nxy + y * nx + x];
                    }
                }
            }
        }
    }

    /// Perform collision step
    pub fn collide(&mut self) {
        self.collide_bgk();
    }

    /// Streaming step with periodic boundaries
    pub fn stream(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nxy = nx * ny;

        self.f_tmp
            .par_chunks_mut(nxy * nz)
            .enumerate()
            .for_each(|(q, chunk)| {
                let ex = D3Q27_E[q].0;
                let ey = D3Q27_E[q].1;
                let ez = D3Q27_E[q].2;
                for z in 0..nz {
                    for y in 0..ny {
                        for x in 0..nx {
                            let sx = ((x as i32 - ex).rem_euclid(nx as i32)) as usize;
                            let sy = ((y as i32 - ey).rem_euclid(ny as i32)) as usize;
                            let sz = ((z as i32 - ez).rem_euclid(nz as i32)) as usize;
                            chunk[z * nxy + y * nx + x] =
                                self.f[q * nxy * nz + sz * nxy + sy * nx + sx];
                        }
                    }
                }
            });

        std::mem::swap(&mut self.f, &mut self.f_tmp);
    }

    /// Set equilibrium at a node
    pub fn set_equilibrium(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        rho: f64,
        ux: f64,
        uy: f64,
        uz: f64,
    ) {
        let feq = Self::equilibrium(rho, ux, uy, uz);
        for q in 0..27 {
            let idx = self.idx(q, x, y, z);
            self.f[idx] = feq[q];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d2q9_weights_sum_to_one() {
        let sum: f64 = D2Q9_W.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "D2Q9 weights sum = {sum}");
    }

    #[test]
    fn test_d3q19_weights_sum_to_one() {
        let sum: f64 = D3Q19_W.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "D3Q19 weights sum = {sum}");
    }

    #[test]
    fn test_d2q9_opposite_directions() {
        for i in 0..9 {
            let opp = D2Q9_OPP[i];
            assert_eq!(D2Q9_E[i].0, -D2Q9_E[opp].0);
            assert_eq!(D2Q9_E[i].1, -D2Q9_E[opp].1);
        }
    }

    #[test]
    fn test_d3q19_opposite_directions() {
        for i in 0..19 {
            let opp = D3Q19_OPP[i];
            assert_eq!(D3Q19_E[i].0, -D3Q19_E[opp].0);
            assert_eq!(D3Q19_E[i].1, -D3Q19_E[opp].1);
            assert_eq!(D3Q19_E[i].2, -D3Q19_E[opp].2);
        }
    }

    #[test]
    fn test_equilibrium_conserves_mass() {
        let rho = 1.5;
        let ux = 0.1;
        let uy = -0.05;
        let feq = Lattice2D::equilibrium(rho, ux, uy);
        let sum: f64 = feq.iter().sum();
        assert!(
            (sum - rho).abs() < 1e-14,
            "Mass not conserved: sum={sum}, rho={rho}"
        );
    }

    #[test]
    fn test_equilibrium_conserves_momentum() {
        let rho = 1.2;
        let ux = 0.1;
        let uy = 0.05;
        let feq = Lattice2D::equilibrium(rho, ux, uy);
        let mut jx = 0.0;
        let mut jy = 0.0;
        for q in 0..9 {
            jx += feq[q] * D2Q9_E[q].0 as f64;
            jy += feq[q] * D2Q9_E[q].1 as f64;
        }
        assert!((jx - rho * ux).abs() < 1e-14);
        assert!((jy - rho * uy).abs() < 1e-14);
    }

    #[test]
    fn test_3d_equilibrium_conserves_mass() {
        let rho = 2.0;
        let feq = Lattice3D::equilibrium(rho, 0.05, -0.03, 0.01);
        let sum: f64 = feq.iter().sum();
        assert!((sum - rho).abs() < 1e-14);
    }

    #[test]
    fn test_lattice2d_init_uniform_density() {
        let lat = Lattice2D::new(10, 10, 1.0, CollisionOperator::Bgk);
        for y in 0..10 {
            for x in 0..10 {
                assert!((lat.density(x, y) - 1.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_lattice2d_init_zero_velocity() {
        let lat = Lattice2D::new(10, 10, 1.0, CollisionOperator::Bgk);
        for y in 0..10 {
            for x in 0..10 {
                let (ux, uy) = lat.velocity(x, y);
                assert!(ux.abs() < 1e-14);
                assert!(uy.abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_bgk_collision_preserves_mass() {
        let mut lat = Lattice2D::new(5, 5, 0.8, CollisionOperator::Bgk);
        // Perturb one node
        lat.set_equilibrium(2, 2, 1.1, 0.05, 0.02);
        let rho_before = lat.density(2, 2);
        lat.collide_bgk();
        let rho_after = lat.density(2, 2);
        assert!(
            (rho_before - rho_after).abs() < 1e-14,
            "BGK doesn't conserve mass"
        );
    }

    #[test]
    fn test_trt_collision_preserves_mass() {
        let mut lat = Lattice2D::new(5, 5, 0.8, CollisionOperator::Trt);
        lat.set_equilibrium(2, 2, 1.1, 0.05, 0.02);
        let rho_before = lat.density(2, 2);
        lat.collide_trt();
        let rho_after = lat.density(2, 2);
        assert!(
            (rho_before - rho_after).abs() < 1e-14,
            "TRT doesn't conserve mass: before={rho_before}, after={rho_after}"
        );
    }

    #[test]
    fn test_trt_collision_preserves_momentum() {
        let mut lat = Lattice2D::new(5, 5, 0.8, CollisionOperator::Trt);
        lat.set_equilibrium(2, 2, 1.1, 0.05, 0.02);
        let (ux_before, uy_before) = lat.velocity(2, 2);
        lat.collide_trt();
        let (ux_after, uy_after) = lat.velocity(2, 2);
        assert!(
            (ux_before - ux_after).abs() < 1e-14,
            "TRT doesn't conserve x-momentum"
        );
        assert!(
            (uy_before - uy_after).abs() < 1e-14,
            "TRT doesn't conserve y-momentum"
        );
    }

    #[test]
    fn test_trt_reduces_to_bgk_at_equilibrium() {
        // At equilibrium, TRT and BGK should produce identical results
        let mut lat_bgk = Lattice2D::new(5, 5, 0.8, CollisionOperator::Bgk);
        let mut lat_trt = Lattice2D::new(5, 5, 0.8, CollisionOperator::Trt);
        // Both start at equilibrium (ρ=1, u=0), so collision is a no-op
        lat_bgk.collide();
        lat_trt.collide();
        for i in 0..lat_bgk.f.len() {
            assert!(
                (lat_bgk.f[i] - lat_trt.f[i]).abs() < 1e-14,
                "TRT and BGK differ at equilibrium at index {i}"
            );
        }
    }

    #[test]
    fn test_mrt_collision_preserves_mass() {
        let mut lat = Lattice2D::new(5, 5, 0.8, CollisionOperator::Mrt);
        lat.set_equilibrium(2, 2, 1.1, 0.05, 0.02);
        let rho_before = lat.density(2, 2);
        lat.collide_mrt();
        let rho_after = lat.density(2, 2);
        assert!(
            (rho_before - rho_after).abs() < 1e-12,
            "MRT doesn't conserve mass: before={rho_before}, after={rho_after}"
        );
    }

    #[test]
    fn test_streaming_periodic() {
        let mut lat = Lattice2D::new(5, 5, 1.0, CollisionOperator::Bgk);
        // Set a perturbation
        lat.set_equilibrium(0, 0, 2.0, 0.0, 0.0);
        let f1_before = lat.f[lat.idx(1, 0, 0)]; // east direction at (0,0)
        lat.stream();
        // f1 (east) at (0,0) should have moved to (1,0)
        let f1_after = lat.f[lat.idx(1, 1, 0)];
        assert!((f1_before - f1_after).abs() < 1e-14);
    }

    #[test]
    fn test_streaming_wraps_periodic() {
        let mut lat = Lattice2D::new(5, 5, 1.0, CollisionOperator::Bgk);
        lat.set_equilibrium(4, 0, 2.0, 0.0, 0.0);
        let f1_before = lat.f[lat.idx(1, 4, 0)]; // east at (4,0)
        lat.stream();
        let f1_after = lat.f[lat.idx(1, 0, 0)]; // should wrap to (0,0)
        assert!((f1_before - f1_after).abs() < 1e-14);
    }

    #[test]
    fn test_3d_streaming() {
        let mut lat = Lattice3D::new(5, 5, 5, 1.0, CollisionOperator::Bgk);
        lat.set_equilibrium(0, 0, 0, 2.0, 0.0, 0.0, 0.0);
        let f1_before = lat.f[lat.idx(1, 0, 0, 0)];
        lat.stream();
        let f1_after = lat.f[lat.idx(1, 1, 0, 0)];
        assert!((f1_before - f1_after).abs() < 1e-14);
    }

    // ── D3Q27 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_d3q27_weights_sum_to_one() {
        let sum: f64 = D3Q27_W.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "D3Q27 weights sum = {sum}");
    }

    #[test]
    fn test_d3q27_opposite_directions() {
        for i in 0..27 {
            let opp = D3Q27_OPP[i];
            assert_eq!(D3Q27_E[i].0, -D3Q27_E[opp].0, "x mismatch at {i}");
            assert_eq!(D3Q27_E[i].1, -D3Q27_E[opp].1, "y mismatch at {i}");
            assert_eq!(D3Q27_E[i].2, -D3Q27_E[opp].2, "z mismatch at {i}");
        }
    }

    #[test]
    fn test_d3q27_equilibrium_conserves_mass() {
        let rho = 1.7;
        let feq = Lattice3D27::equilibrium(rho, 0.04, -0.02, 0.03);
        let sum: f64 = feq.iter().sum();
        assert!((sum - rho).abs() < 1e-13, "D3Q27 mass: sum={sum}, rho={rho}");
    }

    #[test]
    fn test_d3q27_equilibrium_conserves_momentum() {
        let rho = 1.3;
        let (ux, uy, uz) = (0.05, -0.03, 0.02);
        let feq = Lattice3D27::equilibrium(rho, ux, uy, uz);
        let mut jx = 0.0;
        let mut jy = 0.0;
        let mut jz = 0.0;
        for q in 0..27 {
            jx += feq[q] * D3Q27_E[q].0 as f64;
            jy += feq[q] * D3Q27_E[q].1 as f64;
            jz += feq[q] * D3Q27_E[q].2 as f64;
        }
        assert!((jx - rho * ux).abs() < 1e-13);
        assert!((jy - rho * uy).abs() < 1e-13);
        assert!((jz - rho * uz).abs() < 1e-13);
    }

    #[test]
    fn test_d3q27_init_density() {
        let lat = Lattice3D27::new(4, 4, 4, 1.0, CollisionOperator::Bgk);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    assert!((lat.density(x, y, z) - 1.0).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn test_d3q27_streaming() {
        let mut lat = Lattice3D27::new(5, 5, 5, 1.0, CollisionOperator::Bgk);
        lat.set_equilibrium(0, 0, 0, 2.0, 0.0, 0.0, 0.0);
        let f1_before = lat.f[lat.idx(1, 0, 0, 0)];
        lat.stream();
        let f1_after = lat.f[lat.idx(1, 1, 0, 0)];
        assert!((f1_before - f1_after).abs() < 1e-14);
    }

    #[test]
    fn test_d3q27_collision_preserves_mass() {
        let mut lat = Lattice3D27::new(4, 4, 4, 0.8, CollisionOperator::Bgk);
        lat.set_equilibrium(2, 2, 2, 1.3, 0.04, -0.02, 0.01);
        let rho_before = lat.density(2, 2, 2);
        lat.collide();
        let rho_after = lat.density(2, 2, 2);
        assert!(
            (rho_before - rho_after).abs() < 1e-13,
            "D3Q27 mass not conserved: {rho_before} vs {rho_after}"
        );
    }

    // ── Cumulant collision tests ────────────────────────────────────────

    #[test]
    fn test_cumulant_collision_preserves_mass() {
        let mut lat = Lattice2D::new(5, 5, 0.8, CollisionOperator::Cumulant);
        lat.set_equilibrium(2, 2, 1.1, 0.05, 0.02);
        let rho_before = lat.density(2, 2);
        lat.collide_cumulant();
        let rho_after = lat.density(2, 2);
        assert!(
            (rho_before - rho_after).abs() < 1e-12,
            "Cumulant doesn't conserve mass: {rho_before} vs {rho_after}"
        );
    }

    #[test]
    fn test_cumulant_collision_preserves_momentum() {
        let mut lat = Lattice2D::new(5, 5, 0.8, CollisionOperator::Cumulant);
        lat.set_equilibrium(2, 2, 1.1, 0.05, 0.02);
        let (ux_before, uy_before) = lat.velocity(2, 2);
        lat.collide_cumulant();
        let (ux_after, uy_after) = lat.velocity(2, 2);
        assert!(
            (ux_before - ux_after).abs() < 1e-12,
            "Cumulant x-momentum: {ux_before} vs {ux_after}"
        );
        assert!(
            (uy_before - uy_after).abs() < 1e-12,
            "Cumulant y-momentum: {uy_before} vs {uy_after}"
        );
    }

    #[test]
    fn test_cumulant_at_equilibrium_is_noop() {
        let mut lat = Lattice2D::new(5, 5, 0.8, CollisionOperator::Cumulant);
        let f_before: Vec<f64> = lat.f.clone();
        lat.collide_cumulant();
        for i in 0..f_before.len() {
            assert!(
                (lat.f[i] - f_before[i]).abs() < 1e-13,
                "Cumulant changed equilibrium at index {i}: {} vs {}",
                f_before[i],
                lat.f[i]
            );
        }
    }
}
