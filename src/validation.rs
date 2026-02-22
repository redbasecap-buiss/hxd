//! Validation test cases against analytical solutions.
//!
//! These tests prove academic correctness of the LBM implementation.

use crate::boundary::{apply_bounce_back, apply_moving_wall, Edge};
use crate::lattice::{CollisionOperator, Lattice2D, CS2, D2Q9_E, D2Q9_W};
use crate::physics::PhysicsParams;

/// Run all validation cases and print results
pub fn run_validation_suite() {
    println!("FLUX Validation Suite");
    println!("=====================\n");

    let cases: &[(&str, fn() -> (bool, String))] = &[
        ("Poiseuille Flow", validate_poiseuille),
        ("Couette Flow", validate_couette),
        ("Lid-Driven Cavity Re=100", validate_cavity_100),
        ("Taylor-Green Vortex", validate_taylor_green),
        ("Convergence Order", validate_convergence_order),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, func) in cases {
        let (ok, msg) = func();
        if ok {
            println!("✅ {name}: PASS — {msg}");
            passed += 1;
        } else {
            println!("❌ {name}: FAIL — {msg}");
            failed += 1;
        }
    }

    println!("\nResults: {passed} passed, {failed} failed");
    if failed > 0 {
        std::process::exit(1);
    }
}

/// Poiseuille flow: body-force-driven channel flow.
///
/// Uses Guo forcing scheme. Physical velocity: u_phys = (Σ f_i·e_i + F·dt/2) / ρ
/// Analytical solution: u(y) = F/(2ν) · y · (H - y)
/// where H = ny-1 for mid-grid bounce-back walls at y=0 and y=ny-1.
pub fn validate_poiseuille() -> (bool, String) {
    let ny = 64;
    let nx = 4; // Periodic in x, only need a few columns
    let tau = 0.8;
    let nu = CS2 * (tau - 0.5);
    let omega = 1.0 / tau;

    // Mid-grid bounce-back: effective channel height H = ny - 1
    let h = (ny - 1) as f64;
    let u_max_target = 0.01;
    // u_max = F·H²/(8ν) → F = 8·ν·u_max/H²
    let force_acc = 8.0 * nu * u_max_target / (h * h);

    let mut lat = Lattice2D::new(nx, ny, tau, CollisionOperator::Bgk);

    let steps = 40000;
    for _ in 0..steps {
        // Guo forcing: collision with force term (interior nodes only)
        for y in 1..ny - 1 {
            for x in 0..nx {
                let mut rho = 0.0;
                let mut jx = 0.0;
                let mut jy = 0.0;
                let mut fi = [0.0; 9];
                for q in 0..9 {
                    let idx = lat.idx(q, x, y);
                    fi[q] = lat.f[idx];
                    rho += fi[q];
                    jx += fi[q] * D2Q9_E[q].0 as f64;
                    jy += fi[q] * D2Q9_E[q].1 as f64;
                }
                // Physical velocity with half-force correction
                let ux = (jx + force_acc * 0.5) / rho;
                let uy = jy / rho;

                // BGK collision with corrected velocity
                let feq = Lattice2D::equilibrium(rho, ux, uy);

                // Guo forcing term
                for q in 0..9 {
                    let ex = D2Q9_E[q].0 as f64;
                    let ey = D2Q9_E[q].1 as f64;
                    let eu = ex * ux + ey * uy;
                    let force_term = (1.0 - 0.5 * omega)
                        * D2Q9_W[q]
                        * ((ex - ux) / CS2 + eu * ex / (CS2 * CS2))
                        * force_acc;

                    let idx = lat.idx(q, x, y);
                    lat.f[idx] = fi[q] - omega * (fi[q] - feq[q]) + force_term;
                }
            }
        }

        // Regular BGK on wall nodes (no force)
        for &y_wall in &[0, ny - 1] {
            for x in 0..nx {
                let mut rho_w = 0.0;
                let mut jx_w = 0.0;
                let mut jy_w = 0.0;
                let mut fi_w = [0.0; 9];
                for q in 0..9 {
                    let idx = lat.idx(q, x, y_wall);
                    fi_w[q] = lat.f[idx];
                    rho_w += fi_w[q];
                    jx_w += fi_w[q] * D2Q9_E[q].0 as f64;
                    jy_w += fi_w[q] * D2Q9_E[q].1 as f64;
                }
                let ux_w = jx_w / rho_w;
                let uy_w = jy_w / rho_w;
                let feq = Lattice2D::equilibrium(rho_w, ux_w, uy_w);
                for q in 0..9 {
                    let idx = lat.idx(q, x, y_wall);
                    lat.f[idx] = fi_w[q] - omega * (fi_w[q] - feq[q]);
                }
            }
        }

        lat.stream();
        apply_bounce_back(&mut lat, Edge::North);
        apply_bounce_back(&mut lat, Edge::South);
    }

    let x_probe = 0;
    let mut max_error = 0.0;
    for y in 1..ny - 1 {
        // Physical velocity: u_phys = (Σ f_i·e_i + F·dt/2) / ρ
        let mut rho = 0.0;
        let mut jx = 0.0;
        for q in 0..9 {
            let idx = lat.idx(q, x_probe, y);
            let fi = lat.f[idx];
            rho += fi;
            jx += fi * D2Q9_E[q].0 as f64;
        }
        let ux_phys = (jx + force_acc * 0.5) / rho;
        let yf = y as f64;
        let u_analytical = force_acc / (2.0 * nu) * yf * (h - yf);
        if u_analytical.abs() > 1e-10 {
            let error = ((ux_phys - u_analytical) / u_analytical).abs();
            if error > max_error {
                max_error = error;
            }
        }
    }

    let pass = max_error < 0.02;
    (pass, format!("max relative error = {max_error:.4e}"))
}

/// Couette flow: shear-driven flow between two plates.
///
/// Analytical solution: u(y) = U_wall * y / H (linear profile)
pub fn validate_couette() -> (bool, String) {
    let nx = 20;
    let ny = 32;
    let wall_u = 0.05;
    let tau = 0.8;

    let mut lat = Lattice2D::new(nx, ny, tau, CollisionOperator::Bgk);

    let steps = 10000;
    for _ in 0..steps {
        lat.collide();
        lat.stream();
        apply_bounce_back(&mut lat, Edge::South);
        apply_moving_wall(&mut lat, Edge::North, wall_u, 0.0);
    }

    let x_probe = nx / 2;
    let h = (ny - 1) as f64;
    let mut max_error = 0.0;
    for y in 1..ny - 1 {
        let (ux, _) = lat.velocity(x_probe, y);
        let u_analytical = wall_u * y as f64 / h;
        if u_analytical.abs() > 1e-10 {
            let error = ((ux - u_analytical) / u_analytical).abs();
            if error > max_error {
                max_error = error;
            }
        }
    }

    let pass = max_error < 0.01;
    (pass, format!("max relative error = {max_error:.4e}"))
}

/// Lid-driven cavity at Re=100.
/// Compare centerline velocities against Ghia et al. (1982).
pub fn validate_cavity_100() -> (bool, String) {
    let n = 65;
    let re = 100.0;
    let mach = 0.1;
    let char_length = (n - 2) as f64;

    let params = PhysicsParams::new(re, mach, char_length);
    let tau = params.tau();
    let u_lid = params.char_velocity();

    let mut lat = Lattice2D::new(n, n, tau, CollisionOperator::Bgk);

    let steps = 30000;
    for _ in 0..steps {
        lat.collide();
        lat.stream();
        apply_bounce_back(&mut lat, Edge::South);
        apply_bounce_back(&mut lat, Edge::East);
        apply_bounce_back(&mut lat, Edge::West);
        apply_moving_wall(&mut lat, Edge::North, u_lid, 0.0);
    }

    // Ghia et al. (1982) reference data for Re=100
    let ghia_y = [
        0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344,
        0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
    ];
    let ghia_u = [
        0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581,
        -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0000,
    ];

    let x_center = n / 2;
    let mut total_error = 0.0;
    let mut count = 0;

    for i in 1..ghia_y.len() - 1 {
        let y_idx = (ghia_y[i] * (n - 1) as f64).round() as usize;
        if y_idx >= n {
            continue;
        }
        let (ux, _) = lat.velocity(x_center, y_idx);
        let u_normalized = ux / u_lid;
        let error = (u_normalized - ghia_u[i]).abs();
        total_error += error;
        count += 1;
    }

    let avg_error = total_error / count as f64;
    let pass = avg_error < 0.1;
    (
        pass,
        format!("avg deviation from Ghia = {avg_error:.4e} ({count} points)"),
    )
}

/// Taylor-Green vortex decay.
///
/// Initial condition: u = U₀ cos(kx) sin(ky), v = -U₀ sin(kx) cos(ky)
/// Analytical decay: u(t) = u(0) · exp(-2νk²t)
pub fn validate_taylor_green() -> (bool, String) {
    let n = 64;
    let nu = 0.01;
    let tau = nu / CS2 + 0.5;
    let u0 = 0.04;
    let k = 2.0 * std::f64::consts::PI / n as f64;

    let mut lat = Lattice2D::new(n, n, tau, CollisionOperator::Bgk);

    for y in 0..n {
        for x in 0..n {
            let xf = x as f64;
            let yf = y as f64;
            let ux = u0 * (k * xf).cos() * (k * yf).sin();
            let uy = -u0 * (k * xf).sin() * (k * yf).cos();
            let rho = 1.0 - u0 * u0 / (4.0 * CS2) * ((2.0 * k * xf).cos() + (2.0 * k * yf).cos());
            lat.set_equilibrium(x, y, rho, ux, uy);
        }
    }

    let check_step = 500;
    for _ in 0..check_step {
        lat.collide();
        lat.stream();
    }

    let decay = (-2.0 * nu * k * k * check_step as f64).exp();

    // Compare total kinetic energy
    let mut ke_numerical = 0.0;
    let mut ke_analytical = 0.0;
    for y in 0..n {
        for x in 0..n {
            let (ux, uy) = lat.velocity(x, y);
            ke_numerical += ux * ux + uy * uy;

            let xf = x as f64;
            let yf = y as f64;
            let ux_a = u0 * (k * xf).cos() * (k * yf).sin() * decay;
            let uy_a = -u0 * (k * xf).sin() * (k * yf).cos() * decay;
            ke_analytical += ux_a * ux_a + uy_a * uy_a;
        }
    }

    let ke_error = ((ke_numerical - ke_analytical) / ke_analytical).abs();
    let pass = ke_error < 0.05;
    (pass, format!("KE relative error = {ke_error:.4e}"))
}

/// Verify 2nd order spatial convergence using body-force Poiseuille with Guo forcing.
fn poiseuille_error(ny: usize) -> f64 {
    let nx = 4;
    let tau = 0.8;
    let nu = CS2 * (tau - 0.5);
    let omega = 1.0 / tau;
    let h = (ny - 1) as f64;
    let u_max_target = 0.01;
    let force_acc = 8.0 * nu * u_max_target / (h * h);

    let mut lat = Lattice2D::new(nx, ny, tau, CollisionOperator::Bgk);

    let steps = 30000;
    for _ in 0..steps {
        // Guo forcing on interior nodes
        for y in 1..ny - 1 {
            for x in 0..nx {
                let mut rho = 0.0;
                let mut jx = 0.0;
                let mut jy = 0.0;
                let mut fi = [0.0; 9];
                for q in 0..9 {
                    let idx = lat.idx(q, x, y);
                    fi[q] = lat.f[idx];
                    rho += fi[q];
                    jx += fi[q] * D2Q9_E[q].0 as f64;
                    jy += fi[q] * D2Q9_E[q].1 as f64;
                }
                let ux = (jx + force_acc * 0.5) / rho;
                let uy = jy / rho;
                let feq = Lattice2D::equilibrium(rho, ux, uy);
                for q in 0..9 {
                    let ex = D2Q9_E[q].0 as f64;
                    let ey = D2Q9_E[q].1 as f64;
                    let eu = ex * ux + ey * uy;
                    let force_term = (1.0 - 0.5 * omega)
                        * D2Q9_W[q]
                        * ((ex - ux) / CS2 + eu * ex / (CS2 * CS2))
                        * force_acc;
                    let idx = lat.idx(q, x, y);
                    lat.f[idx] = fi[q] - omega * (fi[q] - feq[q]) + force_term;
                }
            }
        }
        // Regular BGK on wall nodes
        for &y_wall in &[0, ny - 1] {
            for x in 0..nx {
                let mut rho_w = 0.0;
                let mut jx_w = 0.0;
                let mut jy_w = 0.0;
                let mut fi_w = [0.0; 9];
                for q in 0..9 {
                    let idx = lat.idx(q, x, y_wall);
                    fi_w[q] = lat.f[idx];
                    rho_w += fi_w[q];
                    jx_w += fi_w[q] * D2Q9_E[q].0 as f64;
                    jy_w += fi_w[q] * D2Q9_E[q].1 as f64;
                }
                let ux_w = jx_w / rho_w;
                let uy_w = jy_w / rho_w;
                let feq = Lattice2D::equilibrium(rho_w, ux_w, uy_w);
                for q in 0..9 {
                    let idx = lat.idx(q, x, y_wall);
                    lat.f[idx] = fi_w[q] - omega * (fi_w[q] - feq[q]);
                }
            }
        }
        lat.stream();
        apply_bounce_back(&mut lat, Edge::North);
        apply_bounce_back(&mut lat, Edge::South);
    }

    let x_probe = 0;
    let mut l2_error = 0.0;
    let mut count = 0;
    for y in 1..ny - 1 {
        let mut rho = 0.0;
        let mut jx = 0.0;
        for q in 0..9 {
            let idx = lat.idx(q, x_probe, y);
            let fi = lat.f[idx];
            rho += fi;
            jx += fi * D2Q9_E[q].0 as f64;
        }
        let ux_phys = (jx + force_acc * 0.5) / rho;
        let yf = y as f64;
        let u_analytical = force_acc / (2.0 * nu) * yf * (h - yf);
        l2_error += (ux_phys - u_analytical) * (ux_phys - u_analytical);
        count += 1;
    }
    (l2_error / count as f64).sqrt()
}

pub fn validate_convergence_order() -> (bool, String) {
    let ny1 = 16;
    let ny2 = 32;

    let e1 = poiseuille_error(ny1);
    let e2 = poiseuille_error(ny2);

    if e2 < 1e-15 || e1 < 1e-15 {
        return (false, "Error too small to compute order".to_string());
    }

    let order = (e1 / e2).ln() / (ny2 as f64 / ny1 as f64).ln();
    let pass = order > 1.5;
    (
        pass,
        format!("order = {order:.2} (e1={e1:.2e}, e2={e2:.2e})"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poiseuille_flow() {
        let (pass, msg) = validate_poiseuille();
        assert!(pass, "Poiseuille flow validation failed: {msg}");
    }

    #[test]
    fn test_couette_flow() {
        let (pass, msg) = validate_couette();
        assert!(pass, "Couette flow validation failed: {msg}");
    }

    #[test]
    fn test_taylor_green_vortex() {
        let (pass, msg) = validate_taylor_green();
        assert!(pass, "Taylor-Green vortex validation failed: {msg}");
    }

    #[test]
    fn test_convergence_order() {
        let (pass, msg) = validate_convergence_order();
        assert!(pass, "Convergence order test failed: {msg}");
    }

    #[test]
    fn test_cavity_re100() {
        let (pass, msg) = validate_cavity_100();
        assert!(pass, "Lid-driven cavity Re=100 failed: {msg}");
    }
}
