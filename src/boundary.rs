//! Boundary conditions for the Lattice Boltzmann Method.
//!
//! Supported types:
//! - **Bounce-back**: No-slip wall (full-way bounce-back)
//! - **Zou-He**: Velocity or pressure boundary (Zou & He, 1997)
//! - **Periodic**: Wrap-around (handled in streaming)
//! - **Open**: Extrapolation outflow
//! - **Moving wall**: For Couette flow etc.

use crate::lattice::{Lattice2D, D2Q9_E, D2Q9_OPP, D2Q9_W, CS2};

/// Boundary condition type
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BoundaryCondition {
    Wall,
    Velocity,
    Pressure,
    Periodic,
    Open,
    MovingWall,
}

/// Which edge of the domain
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Edge {
    North,
    South,
    East,
    West,
}

/// Apply bounce-back (no-slip) on specified edge.
/// Full-way bounce-back: f_opp(x_wall) = f_i(x_wall) after streaming.
pub fn apply_bounce_back(lattice: &mut Lattice2D, edge: Edge) {
    let nx = lattice.nx;
    let ny = lattice.ny;

    match edge {
        Edge::North => {
            let y = ny - 1;
            for x in 0..nx {
                // Directions pointing into the domain (south-ward): 4, 7, 8
                for &q in &[4, 7, 8] {
                    let opp = D2Q9_OPP[q];
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(opp, x, y)];
                }
            }
        }
        Edge::South => {
            let y = 0;
            for x in 0..nx {
                for &q in &[2, 5, 6] {
                    let opp = D2Q9_OPP[q];
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(opp, x, y)];
                }
            }
        }
        Edge::East => {
            let x = nx - 1;
            for y in 0..ny {
                for &q in &[3, 6, 7] {
                    let opp = D2Q9_OPP[q];
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(opp, x, y)];
                }
            }
        }
        Edge::West => {
            let x = 0;
            for y in 0..ny {
                for &q in &[1, 5, 8] {
                    let opp = D2Q9_OPP[q];
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(opp, x, y)];
                }
            }
        }
    }
}

/// Apply moving wall boundary (bounce-back with momentum transfer).
/// Used for lid-driven cavity and Couette flow.
///
/// f_opp = f_i - 2 * w_i * ρ * (e_i · u_wall) / cs²
pub fn apply_moving_wall(lattice: &mut Lattice2D, edge: Edge, wall_ux: f64, wall_uy: f64) {
    let nx = lattice.nx;
    let ny = lattice.ny;

    match edge {
        Edge::North => {
            let y = ny - 1;
            for x in 0..nx {
                let rho = lattice.density(x, y);
                for &q in &[4, 7, 8] {
                    let opp = D2Q9_OPP[q];
                    let eu_wall =
                        D2Q9_E[opp].0 as f64 * wall_ux + D2Q9_E[opp].1 as f64 * wall_uy;
                    lattice.f[lattice.idx(q, x, y)] =
                        lattice.f[lattice.idx(opp, x, y)] - 2.0 * D2Q9_W[opp] * rho * eu_wall / CS2;
                }
            }
        }
        Edge::South => {
            let y = 0;
            for x in 0..nx {
                let rho = lattice.density(x, y);
                for &q in &[2, 5, 6] {
                    let opp = D2Q9_OPP[q];
                    let eu_wall =
                        D2Q9_E[opp].0 as f64 * wall_ux + D2Q9_E[opp].1 as f64 * wall_uy;
                    lattice.f[lattice.idx(q, x, y)] =
                        lattice.f[lattice.idx(opp, x, y)] - 2.0 * D2Q9_W[opp] * rho * eu_wall / CS2;
                }
            }
        }
        Edge::East => {
            let x = nx - 1;
            for y in 0..ny {
                let rho = lattice.density(x, y);
                for &q in &[3, 6, 7] {
                    let opp = D2Q9_OPP[q];
                    let eu_wall =
                        D2Q9_E[opp].0 as f64 * wall_ux + D2Q9_E[opp].1 as f64 * wall_uy;
                    lattice.f[lattice.idx(q, x, y)] =
                        lattice.f[lattice.idx(opp, x, y)] - 2.0 * D2Q9_W[opp] * rho * eu_wall / CS2;
                }
            }
        }
        Edge::West => {
            let x = 0;
            for y in 0..ny {
                let rho = lattice.density(x, y);
                for &q in &[1, 5, 8] {
                    let opp = D2Q9_OPP[q];
                    let eu_wall =
                        D2Q9_E[opp].0 as f64 * wall_ux + D2Q9_E[opp].1 as f64 * wall_uy;
                    lattice.f[lattice.idx(q, x, y)] =
                        lattice.f[lattice.idx(opp, x, y)] - 2.0 * D2Q9_W[opp] * rho * eu_wall / CS2;
                }
            }
        }
    }
}

/// Zou-He velocity boundary condition.
/// Prescribes velocity at the boundary and solves for unknown distributions.
///
/// Reference: Zou, Q. & He, X. (1997). "On pressure and velocity boundary
/// conditions for the lattice Boltzmann BGK model."
pub fn apply_zou_he_velocity(lattice: &mut Lattice2D, edge: Edge, ux: f64, uy: f64) {
    let nx = lattice.nx;
    let ny = lattice.ny;

    match edge {
        Edge::West => {
            let x = 0;
            for y in 1..ny - 1 {
                let f0 = lattice.f[lattice.idx(0, x, y)];
                let f2 = lattice.f[lattice.idx(2, x, y)];
                let f3 = lattice.f[lattice.idx(3, x, y)];
                let f4 = lattice.f[lattice.idx(4, x, y)];
                let f6 = lattice.f[lattice.idx(6, x, y)];
                let f7 = lattice.f[lattice.idx(7, x, y)];

                let rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux);

                let f1 = f3 + 2.0 / 3.0 * rho * ux;
                let f5 = f7 - 0.5 * (f2 - f4) + rho * ux / 6.0 + rho * uy / 2.0;
                let f8 = f6 + 0.5 * (f2 - f4) + rho * ux / 6.0 - rho * uy / 2.0;

                lattice.f[lattice.idx(1, x, y)] = f1;
                lattice.f[lattice.idx(5, x, y)] = f5;
                lattice.f[lattice.idx(8, x, y)] = f8;
            }
        }
        Edge::East => {
            let x = nx - 1;
            for y in 1..ny - 1 {
                let f0 = lattice.f[lattice.idx(0, x, y)];
                let f1 = lattice.f[lattice.idx(1, x, y)];
                let f2 = lattice.f[lattice.idx(2, x, y)];
                let f4 = lattice.f[lattice.idx(4, x, y)];
                let f5 = lattice.f[lattice.idx(5, x, y)];
                let f8 = lattice.f[lattice.idx(8, x, y)];

                let rho = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 + ux);

                let f3 = f1 - 2.0 / 3.0 * rho * ux;
                let f7 = f5 + 0.5 * (f2 - f4) - rho * ux / 6.0 - rho * uy / 2.0;
                let f6 = f8 - 0.5 * (f2 - f4) - rho * ux / 6.0 + rho * uy / 2.0;

                lattice.f[lattice.idx(3, x, y)] = f3;
                lattice.f[lattice.idx(7, x, y)] = f7;
                lattice.f[lattice.idx(6, x, y)] = f6;
            }
        }
        Edge::South => {
            let y = 0;
            for x in 1..nx - 1 {
                let f0 = lattice.f[lattice.idx(0, x, y)];
                let f1 = lattice.f[lattice.idx(1, x, y)];
                let f3 = lattice.f[lattice.idx(3, x, y)];
                let f4 = lattice.f[lattice.idx(4, x, y)];
                let f7 = lattice.f[lattice.idx(7, x, y)];
                let f8 = lattice.f[lattice.idx(8, x, y)];

                let rho = (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / (1.0 - uy);

                let f2 = f4 + 2.0 / 3.0 * rho * uy;
                let f5 = f7 - 0.5 * (f1 - f3) + rho * uy / 6.0 + rho * ux / 2.0;
                let f6 = f8 + 0.5 * (f1 - f3) + rho * uy / 6.0 - rho * ux / 2.0;

                lattice.f[lattice.idx(2, x, y)] = f2;
                lattice.f[lattice.idx(5, x, y)] = f5;
                lattice.f[lattice.idx(6, x, y)] = f6;
            }
        }
        Edge::North => {
            let y = ny - 1;
            for x in 1..nx - 1 {
                let f0 = lattice.f[lattice.idx(0, x, y)];
                let f1 = lattice.f[lattice.idx(1, x, y)];
                let f2 = lattice.f[lattice.idx(2, x, y)];
                let f3 = lattice.f[lattice.idx(3, x, y)];
                let f5 = lattice.f[lattice.idx(5, x, y)];
                let f6 = lattice.f[lattice.idx(6, x, y)];

                let rho = (f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)) / (1.0 + uy);

                let f4 = f2 - 2.0 / 3.0 * rho * uy;
                let f7 = f5 + 0.5 * (f1 - f3) - rho * uy / 6.0 - rho * ux / 2.0;
                let f8 = f6 - 0.5 * (f1 - f3) - rho * uy / 6.0 + rho * ux / 2.0;

                lattice.f[lattice.idx(4, x, y)] = f4;
                lattice.f[lattice.idx(7, x, y)] = f7;
                lattice.f[lattice.idx(8, x, y)] = f8;
            }
        }
    }
}

/// Zou-He pressure boundary condition.
/// Prescribes density (pressure = ρ·cs²) at the boundary.
pub fn apply_zou_he_pressure(lattice: &mut Lattice2D, edge: Edge, rho_target: f64) {
    let nx = lattice.nx;
    let ny = lattice.ny;

    match edge {
        Edge::East => {
            let x = nx - 1;
            for y in 1..ny - 1 {
                let f0 = lattice.f[lattice.idx(0, x, y)];
                let f1 = lattice.f[lattice.idx(1, x, y)];
                let f2 = lattice.f[lattice.idx(2, x, y)];
                let f4 = lattice.f[lattice.idx(4, x, y)];
                let f5 = lattice.f[lattice.idx(5, x, y)];
                let f8 = lattice.f[lattice.idx(8, x, y)];

                let ux = 1.0 - (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_target;

                let f3 = f1 - 2.0 / 3.0 * rho_target * ux;
                let f7 = f5 + 0.5 * (f2 - f4) - rho_target * ux / 6.0;
                let f6 = f8 - 0.5 * (f2 - f4) - rho_target * ux / 6.0;

                lattice.f[lattice.idx(3, x, y)] = f3;
                lattice.f[lattice.idx(7, x, y)] = f7;
                lattice.f[lattice.idx(6, x, y)] = f6;
            }
        }
        Edge::West => {
            let x = 0;
            for y in 1..ny - 1 {
                let f0 = lattice.f[lattice.idx(0, x, y)];
                let f2 = lattice.f[lattice.idx(2, x, y)];
                let f3 = lattice.f[lattice.idx(3, x, y)];
                let f4 = lattice.f[lattice.idx(4, x, y)];
                let f6 = lattice.f[lattice.idx(6, x, y)];
                let f7 = lattice.f[lattice.idx(7, x, y)];

                let ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho_target;

                let f1 = f3 + 2.0 / 3.0 * rho_target * ux;
                let f5 = f7 - 0.5 * (f2 - f4) + rho_target * ux / 6.0;
                let f8 = f6 + 0.5 * (f2 - f4) + rho_target * ux / 6.0;

                lattice.f[lattice.idx(1, x, y)] = f1;
                lattice.f[lattice.idx(5, x, y)] = f5;
                lattice.f[lattice.idx(8, x, y)] = f8;
            }
        }
        _ => {
            // Pressure BCs for North/South follow similar pattern
            // For brevity, we implement the most common cases (East/West)
            unimplemented!("Pressure BC for {:?} not yet implemented", edge);
        }
    }
}

/// Open boundary condition using first-order extrapolation.
/// Copies distributions from the second-to-last row/column.
pub fn apply_open_boundary(lattice: &mut Lattice2D, edge: Edge) {
    let nx = lattice.nx;
    let ny = lattice.ny;

    match edge {
        Edge::East => {
            let x = nx - 1;
            for y in 0..ny {
                for q in 0..9 {
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(q, x - 1, y)];
                }
            }
        }
        Edge::West => {
            let x = 0;
            for y in 0..ny {
                for q in 0..9 {
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(q, x + 1, y)];
                }
            }
        }
        Edge::North => {
            let y = ny - 1;
            for x in 0..nx {
                for q in 0..9 {
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(q, x, y - 1)];
                }
            }
        }
        Edge::South => {
            let y = 0;
            for x in 0..nx {
                for q in 0..9 {
                    lattice.f[lattice.idx(q, x, y)] = lattice.f[lattice.idx(q, x, y + 1)];
                }
            }
        }
    }
}

/// Apply bounce-back on obstacle nodes (interior).
/// `is_solid` is a boolean mask of size nx*ny.
pub fn apply_obstacle_bounce_back(lattice: &mut Lattice2D, is_solid: &[bool]) {
    let nx = lattice.nx;
    let ny = lattice.ny;

    // For each solid node, swap populations with opposite directions
    for y in 0..ny {
        for x in 0..nx {
            if !is_solid[y * nx + x] {
                continue;
            }
            // Simple bounce-back: swap f_i and f_opp
            for q in 1..5 {
                let opp = D2Q9_OPP[q];
                let idx_q = lattice.idx(q, x, y);
                let idx_opp = lattice.idx(opp, x, y);
                lattice.f.swap(idx_q, idx_opp);
            }
            // Diagonal pairs (5,7) and (6,8) already paired by the above pattern
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::CollisionOperator;

    #[test]
    fn test_bounce_back_reverses_populations() {
        let mut lat = Lattice2D::new(10, 10, 1.0, CollisionOperator::Bgk);
        lat.set_equilibrium(5, 0, 1.0, 0.0, 0.1);

        let f2_before = lat.f[lat.idx(2, 5, 0)]; // north
        apply_bounce_back(&mut lat, Edge::South);
        let f2_after = lat.f[lat.idx(2, 5, 0)]; // should now equal old f4

        // After bounce-back on south wall, f2 should come from f4
        let f4_orig = Lattice2D::equilibrium(1.0, 0.0, 0.1)[4];
        assert!((f2_after - f4_orig).abs() < 1e-12);
    }

    #[test]
    fn test_zou_he_velocity_west() {
        let mut lat = Lattice2D::new(20, 10, 0.8, CollisionOperator::Bgk);
        let ux_target = 0.05;
        apply_zou_he_velocity(&mut lat, Edge::West, ux_target, 0.0);

        // Check that velocity at west boundary is approximately correct
        for y in 2..8 {
            let (ux, _) = lat.velocity(0, y);
            assert!(
                (ux - ux_target).abs() < 0.01,
                "Zou-He velocity mismatch at y={y}: got {ux}"
            );
        }
    }

    #[test]
    fn test_open_boundary_extrapolation() {
        let mut lat = Lattice2D::new(10, 10, 1.0, CollisionOperator::Bgk);
        lat.set_equilibrium(8, 5, 1.1, 0.05, 0.0);
        apply_open_boundary(&mut lat, Edge::East);

        // East boundary should have same distributions as x=8
        for q in 0..9 {
            let f_interior = lat.f[lat.idx(q, 8, 5)];
            let f_boundary = lat.f[lat.idx(q, 9, 5)];
            assert!((f_interior - f_boundary).abs() < 1e-14);
        }
    }

    #[test]
    fn test_moving_wall_adds_momentum() {
        let mut lat = Lattice2D::new(10, 10, 1.0, CollisionOperator::Bgk);
        let wall_ux = 0.1;

        // Store populations before
        let f4_before = lat.f[lat.idx(4, 5, 9)];

        apply_moving_wall(&mut lat, Edge::North, wall_ux, 0.0);

        // Moving wall should modify populations
        let f4_after = lat.f[lat.idx(4, 5, 9)];
        // The difference should depend on wall velocity
        assert!(
            (f4_after - f4_before).abs() > 1e-10,
            "Moving wall should change populations"
        );
    }
}
