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

    /// Perform collision step based on configured operator
    pub fn collide(&mut self) {
        match self.collision {
            CollisionOperator::Bgk => self.collide_bgk(),
            CollisionOperator::Mrt => self.collide_mrt(),
            CollisionOperator::Trt => self.collide_trt(),
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
}
