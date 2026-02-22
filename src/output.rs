//! Output and visualization for LBM simulations.
//!
//! Supported formats:
//! - **VTK** (Legacy format, ParaView compatible)
//! - **CSV** (velocity/density fields)
//! - **PPM** (direct image output, no external deps)

use crate::lattice::Lattice2D;
use byteorder::{BigEndian, WriteBytesExt};
use std::io::Write;
use std::path::Path;

/// Output format selection
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Vtk,
    Csv,
    Ppm,
}

/// Which fields to output
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputField {
    Velocity,
    Density,
    Vorticity,
}

/// Compute velocity magnitude field
pub fn velocity_magnitude(ux: &[f64], uy: &[f64]) -> Vec<f64> {
    ux.iter()
        .zip(uy.iter())
        .map(|(&u, &v)| (u * u + v * v).sqrt())
        .collect()
}

/// Write 2D field data in VTK legacy format (structured points).
pub fn write_vtk(
    path: &Path,
    nx: usize,
    ny: usize,
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
    vorticity: Option<&[f64]>,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // VTK legacy header
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "FLUX LBM output")?;
    writeln!(file, "BINARY")?;
    writeln!(file, "DATASET STRUCTURED_POINTS")?;
    writeln!(file, "DIMENSIONS {} {} 1", nx, ny)?;
    writeln!(file, "ORIGIN 0 0 0")?;
    writeln!(file, "SPACING 1 1 1")?;
    writeln!(file, "POINT_DATA {}", nx * ny)?;

    // Density (scalar)
    writeln!(file, "SCALARS density float 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    for &r in rho {
        file.write_f32::<BigEndian>(r as f32)?;
    }
    writeln!(file)?;

    // Velocity (vector)
    writeln!(file, "VECTORS velocity float")?;
    for i in 0..nx * ny {
        file.write_f32::<BigEndian>(ux[i] as f32)?;
        file.write_f32::<BigEndian>(uy[i] as f32)?;
        file.write_f32::<BigEndian>(0.0)?;
    }
    writeln!(file)?;

    // Velocity magnitude (scalar)
    writeln!(file, "SCALARS velocity_magnitude float 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    let vmag = velocity_magnitude(ux, uy);
    for &v in &vmag {
        file.write_f32::<BigEndian>(v as f32)?;
    }
    writeln!(file)?;

    // Vorticity (scalar, optional)
    if let Some(vort) = vorticity {
        writeln!(file, "SCALARS vorticity float 1")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        for &v in vort {
            file.write_f32::<BigEndian>(v as f32)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

/// Write field data as CSV.
pub fn write_csv(
    path: &Path,
    nx: usize,
    ny: usize,
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "x,y,rho,ux,uy,umag")?;
    for y in 0..ny {
        for x in 0..nx {
            let i = y * nx + x;
            let umag = (ux[i] * ux[i] + uy[i] * uy[i]).sqrt();
            writeln!(
                file,
                "{},{},{:.8},{:.8},{:.8},{:.8}",
                x, y, rho[i], ux[i], uy[i], umag
            )?;
        }
    }
    Ok(())
}

/// Write velocity magnitude as a PPM image (P6 binary).
/// Uses a blue-white-red diverging colormap.
pub fn write_ppm(path: &Path, nx: usize, ny: usize, ux: &[f64], uy: &[f64]) -> std::io::Result<()> {
    let vmag = velocity_magnitude(ux, uy);
    let max_v = vmag.iter().cloned().fold(0.0f64, f64::max).max(1e-10);

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", nx, ny)?;
    writeln!(file, "255")?;

    // Write pixels top-to-bottom
    for y in (0..ny).rev() {
        for x in 0..nx {
            let t = vmag[y * nx + x] / max_v;
            let (r, g, b) = colormap_viridis(t);
            file.write_all(&[r, g, b])?;
        }
    }

    Ok(())
}

/// Viridis-inspired colormap: dark blue → teal → green → yellow
fn colormap_viridis(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let r;
    let g;
    let b;

    if t < 0.25 {
        let s = t / 0.25;
        r = 68.0 * (1.0 - s) + 59.0 * s;
        g = 1.0 * (1.0 - s) + 82.0 * s;
        b = 84.0 * (1.0 - s) + 139.0 * s;
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        r = 59.0 * (1.0 - s) + 33.0 * s;
        g = 82.0 * (1.0 - s) + 145.0 * s;
        b = 139.0 * (1.0 - s) + 140.0 * s;
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        r = 33.0 * (1.0 - s) + 94.0 * s;
        g = 145.0 * (1.0 - s) + 201.0 * s;
        b = 140.0 * (1.0 - s) + 98.0 * s;
    } else {
        let s = (t - 0.75) / 0.25;
        r = 94.0 * (1.0 - s) + 253.0 * s;
        g = 201.0 * (1.0 - s) + 231.0 * s;
        b = 98.0 * (1.0 - s) + 37.0 * s;
    }

    (r as u8, g as u8, b as u8)
}

/// Compute streamline starting points and trace them.
/// Returns list of streamlines, each as a Vec of (x, y) points.
pub fn compute_streamlines(
    nx: usize,
    ny: usize,
    ux: &[f64],
    uy: &[f64],
    num_seeds: usize,
    max_steps: usize,
) -> Vec<Vec<(f64, f64)>> {
    let mut streamlines = Vec::new();
    let step = ny / (num_seeds + 1);

    for i in 0..num_seeds {
        let seed_y = (i + 1) * step;
        let mut x = 0.5;
        let mut y = seed_y as f64;
        let mut points = vec![(x, y)];

        for _ in 0..max_steps {
            let ix = (x as usize).min(nx - 1);
            let iy = (y as usize).min(ny - 1);
            let idx = iy * nx + ix;
            let u = ux[idx];
            let v = uy[idx];
            let speed = (u * u + v * v).sqrt();
            if speed < 1e-10 {
                break;
            }
            // RK1 integration
            let dt = 0.5;
            x += u * dt;
            y += v * dt;
            if x < 0.0 || x >= nx as f64 || y < 0.0 || y >= ny as f64 {
                break;
            }
            points.push((x, y));
        }

        if points.len() > 1 {
            streamlines.push(points);
        }
    }

    streamlines
}

/// Write output from a Lattice2D based on format selection
pub fn write_output(
    lattice: &Lattice2D,
    path: &Path,
    format: &OutputFormat,
    fields: &[OutputField],
) -> std::io::Result<()> {
    let (rho, ux, uy) = lattice.macroscopic_fields();
    let include_vorticity = fields.iter().any(|f| matches!(f, OutputField::Vorticity));
    let vorticity = if include_vorticity {
        Some(lattice.vorticity_field())
    } else {
        None
    };

    match format {
        OutputFormat::Vtk => write_vtk(
            path,
            lattice.nx,
            lattice.ny,
            &rho,
            &ux,
            &uy,
            vorticity.as_deref(),
        ),
        OutputFormat::Csv => write_csv(path, lattice.nx, lattice.ny, &rho, &ux, &uy),
        OutputFormat::Ppm => write_ppm(path, lattice.nx, lattice.ny, &ux, &uy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_magnitude() {
        let ux = vec![3.0, 0.0];
        let uy = vec![4.0, 0.0];
        let vmag = velocity_magnitude(&ux, &uy);
        assert!((vmag[0] - 5.0).abs() < 1e-14);
        assert!((vmag[1] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_colormap_bounds() {
        // u8 values are always in [0, 255]; just ensure no panics
        let (_r, _g, _b) = colormap_viridis(0.0);
        let (_r, _g, _b) = colormap_viridis(1.0);
    }

    #[test]
    fn test_write_csv() {
        let dir = std::env::temp_dir().join("flux_test_csv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");
        let rho = vec![1.0; 4];
        let ux = vec![0.1, 0.2, 0.1, 0.2];
        let uy = vec![0.0; 4];
        write_csv(&path, 2, 2, &rho, &ux, &uy).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("x,y,rho,ux,uy,umag"));
        assert!(content.lines().count() == 5); // header + 4 data rows
    }

    #[test]
    fn test_write_ppm() {
        let dir = std::env::temp_dir().join("flux_test_ppm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.ppm");
        let ux = vec![0.1; 16];
        let uy = vec![0.05; 16];
        write_ppm(&path, 4, 4, &ux, &uy).unwrap();
        assert!(path.exists());
    }
}
