//! Enhanced I/O: VTK, binary formats, image output, terminal visualisation, animation export.

use crate::gpu::CpuLbmSolver;
use std::io::Write;
use std::path::Path;

// ---------------------------------------------------------------------------
// Colormaps
// ---------------------------------------------------------------------------

/// Supported colormaps for scalar field visualisation.
#[derive(Clone, Copy, Debug)]
pub enum Colormap {
    Jet,
    Viridis,
    Plasma,
    Coolwarm,
}

/// Map a value in [0,1] to an RGB triple.
pub fn colormap(val: f32, cmap: Colormap) -> [u8; 3] {
    let t = val.clamp(0.0, 1.0);
    match cmap {
        Colormap::Jet => jet(t),
        Colormap::Viridis => viridis(t),
        Colormap::Plasma => plasma(t),
        Colormap::Coolwarm => coolwarm(t),
    }
}

fn jet(t: f32) -> [u8; 3] {
    let r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn viridis(t: f32) -> [u8; 3] {
    // Simplified viridis approximation
    let r = (0.267004 + t * (0.993248 - 0.267004)).clamp(0.0, 1.0);
    let g = (0.004874 + t * (0.906157 - 0.004874)).clamp(0.0, 1.0);
    let b = (0.329415 + t * (0.143936 - 0.329415)).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn plasma(t: f32) -> [u8; 3] {
    let r = (0.050383 + t * (0.940015 - 0.050383)).clamp(0.0, 1.0);
    let g = (0.029803 + t * (0.975158 - 0.029803)).clamp(0.0, 1.0);
    let b = (0.527975 + t * (0.12394 - 0.527975)).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn coolwarm(t: f32) -> [u8; 3] {
    // Blue (0) -> White (0.5) -> Red (1)
    let r = if t < 0.5 {
        0.2298 + t * 2.0 * (1.0 - 0.2298)
    } else {
        1.0 - (t - 0.5) * 2.0 * (1.0 - 0.7059)
    };
    let g = if t < 0.5 {
        0.2987 + t * 2.0 * (1.0 - 0.2987)
    } else {
        1.0 - (t - 0.5) * 2.0 * (1.0 - 0.0156)
    };
    let b = if t < 0.5 {
        0.7537 + t * 2.0 * (1.0 - 0.7537)
    } else {
        1.0 - (t - 0.5) * 2.0 * (1.0 - 0.1502)
    };
    [
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

// ---------------------------------------------------------------------------
// PPM image output
// ---------------------------------------------------------------------------

/// Write a scalar field to a PPM image file.
pub fn write_ppm<P: AsRef<Path>>(
    path: P,
    data: &[f32],
    nx: usize,
    ny: usize,
    cmap: Colormap,
) -> std::io::Result<()> {
    let (min_v, max_v) = min_max(data);
    let range = if (max_v - min_v).abs() < 1e-12 {
        1.0
    } else {
        max_v - min_v
    };

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{nx} {ny}")?;
    writeln!(file, "255")?;
    for y in (0..ny).rev() {
        for x in 0..nx {
            let t = (data[y * nx + x] - min_v) / range;
            let rgb = colormap(t, cmap);
            file.write_all(&rgb)?;
        }
    }
    Ok(())
}

/// Write a scalar field to a PNG file (using PPM internally, minimal deps).
/// Returns PPM data as bytes (can be written as .ppm or converted).
pub fn write_png_as_ppm(data: &[f32], nx: usize, ny: usize, cmap: Colormap) -> Vec<u8> {
    let (min_v, max_v) = min_max(data);
    let range = if (max_v - min_v).abs() < 1e-12 {
        1.0
    } else {
        max_v - min_v
    };

    let mut buf = Vec::new();
    writeln!(buf, "P6").unwrap();
    writeln!(buf, "{nx} {ny}").unwrap();
    writeln!(buf, "255").unwrap();
    for y in (0..ny).rev() {
        for x in 0..nx {
            let t = (data[y * nx + x] - min_v) / range;
            let rgb = colormap(t, cmap);
            buf.extend_from_slice(&rgb);
        }
    }
    buf
}

fn min_max(data: &[f32]) -> (f32, f32) {
    let mut mn = f32::MAX;
    let mut mx = f32::MIN;
    for &v in data {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    (mn, mx)
}

// ---------------------------------------------------------------------------
// VTK XML output (.vtu)
// ---------------------------------------------------------------------------

/// Write a 2D scalar field as VTK unstructured grid (XML .vtu format).
pub fn write_vtk<P: AsRef<Path>>(
    path: P,
    density: &[f32],
    velocity_x: &[f32],
    velocity_y: &[f32],
    nx: usize,
    ny: usize,
) -> std::io::Result<()> {
    let n = nx * ny;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "<?xml version=\"1.0\"?>")?;
    writeln!(
        f,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(f, "  <UnstructuredGrid>")?;
    writeln!(
        f,
        "    <Piece NumberOfPoints=\"{n}\" NumberOfCells=\"{n}\">"
    )?;

    // Points
    writeln!(f, "      <Points>")?;
    writeln!(
        f,
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for y in 0..ny {
        for x in 0..nx {
            write!(f, "          {} {} 0\n", x, y)?;
        }
    }
    writeln!(f, "        </DataArray>")?;
    writeln!(f, "      </Points>")?;

    // Cells (one vertex per cell for simplicity)
    writeln!(f, "      <Cells>")?;
    writeln!(
        f,
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">"
    )?;
    for i in 0..n {
        write!(f, "          {i}\n")?;
    }
    writeln!(f, "        </DataArray>")?;
    writeln!(
        f,
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
    )?;
    for i in 1..=n {
        write!(f, "          {i}\n")?;
    }
    writeln!(f, "        </DataArray>")?;
    writeln!(
        f,
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    for _ in 0..n {
        write!(f, "          1\n")?; // VTK_VERTEX
    }
    writeln!(f, "        </DataArray>")?;
    writeln!(f, "      </Cells>")?;

    // Point data
    writeln!(f, "      <PointData Scalars=\"density\">")?;
    writeln!(
        f,
        "        <DataArray type=\"Float32\" Name=\"density\" format=\"ascii\">"
    )?;
    for v in density {
        write!(f, "          {v}\n")?;
    }
    writeln!(f, "        </DataArray>")?;
    writeln!(
        f,
        "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for i in 0..n {
        write!(f, "          {} {} 0\n", velocity_x[i], velocity_y[i])?;
    }
    writeln!(f, "        </DataArray>")?;
    writeln!(f, "      </PointData>")?;

    writeln!(f, "    </Piece>")?;
    writeln!(f, "  </UnstructuredGrid>")?;
    writeln!(f, "</VTKFile>")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Binary format (HDF5-like)
// ---------------------------------------------------------------------------

/// A simple binary format for large datasets.
/// Header: magic(4) + nx(u32) + ny(u32) + nfields(u32) + field_names
/// Data: raw f32 arrays
const BINARY_MAGIC: &[u8; 4] = b"FLX1";

/// Write solver state in a compact binary format.
pub fn write_binary<P: AsRef<Path>>(path: P, solver: &CpuLbmSolver) -> std::io::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};

    let mut f = std::fs::File::create(path)?;
    f.write_all(BINARY_MAGIC)?;
    f.write_u32::<LittleEndian>(solver.nx as u32)?;
    f.write_u32::<LittleEndian>(solver.ny as u32)?;
    f.write_u32::<LittleEndian>(2)?; // number of fields: density, velocity_mag

    // Field 0: density
    let density = solver.density_field();
    for v in &density {
        f.write_f32::<LittleEndian>(*v)?;
    }

    // Field 1: velocity magnitude
    let vel = solver.velocity_magnitude_field();
    for v in &vel {
        f.write_f32::<LittleEndian>(*v)?;
    }

    Ok(())
}

/// Read binary format, returns (nx, ny, fields).
pub fn read_binary<P: AsRef<Path>>(path: P) -> std::io::Result<(usize, usize, Vec<Vec<f32>>)> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut f, &mut magic)?;
    if &magic != BINARY_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Bad magic",
        ));
    }
    let nx = f.read_u32::<LittleEndian>()? as usize;
    let ny = f.read_u32::<LittleEndian>()? as usize;
    let nfields = f.read_u32::<LittleEndian>()? as usize;
    let n = nx * ny;

    let mut fields = Vec::new();
    for _ in 0..nfields {
        let mut field = vec![0.0f32; n];
        for v in &mut field {
            *v = f.read_f32::<LittleEndian>()?;
        }
        fields.push(field);
    }
    Ok((nx, ny, fields))
}

// ---------------------------------------------------------------------------
// Terminal visualisation
// ---------------------------------------------------------------------------

/// Render a scalar field to the terminal using Unicode block characters.
pub fn terminal_render(data: &[f32], nx: usize, ny: usize, term_width: usize) -> String {
    let blocks = [' ', '░', '▒', '▓', '█'];
    let (min_v, max_v) = min_max(data);
    let range = if (max_v - min_v).abs() < 1e-12 {
        1.0
    } else {
        max_v - min_v
    };

    // Scale to fit terminal width
    let step_x = (nx as f32 / term_width as f32).max(1.0) as usize;
    let step_y = (step_x * 2).max(1); // Terminal chars are ~2:1 aspect

    let mut out = String::new();
    let mut y = ny;
    while y > 0 {
        y = y.saturating_sub(step_y);
        for x in (0..nx).step_by(step_x) {
            let t = (data[y * nx + x] - min_v) / range;
            let idx = ((t * 4.0) as usize).min(4);
            out.push(blocks[idx]);
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Animation export
// ---------------------------------------------------------------------------

/// Export solver state for a single frame (VTK + PPM).
pub fn export_frame<P: AsRef<Path>>(
    dir: P,
    solver: &CpuLbmSolver,
    frame: usize,
    cmap: Colormap,
) -> std::io::Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;

    let density = solver.density_field();
    let vel_mag = solver.velocity_magnitude_field();

    // VTK frame
    let mut vx = vec![0.0f32; solver.nx * solver.ny];
    let mut vy = vec![0.0f32; solver.nx * solver.ny];
    for y in 0..solver.ny {
        for x in 0..solver.nx {
            let [ux, uy] = solver.velocity(x, y);
            vx[y * solver.nx + x] = ux;
            vy[y * solver.nx + x] = uy;
        }
    }
    write_vtk(
        dir.join(format!("frame_{frame:06}.vtu")),
        &density,
        &vx,
        &vy,
        solver.nx,
        solver.ny,
    )?;

    // PPM frame
    write_ppm(
        dir.join(format!("frame_{frame:06}.ppm")),
        &vel_mag,
        solver.nx,
        solver.ny,
        cmap,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colormap_jet_bounds() {
        let [r, g, b] = colormap(0.0, Colormap::Jet);
        assert!(r <= 255 && g <= 255 && b <= 255);
        let [r, g, b] = colormap(1.0, Colormap::Jet);
        assert!(r <= 255 && g <= 255 && b <= 255);
    }

    #[test]
    fn test_colormap_all_variants() {
        for cmap in [Colormap::Jet, Colormap::Viridis, Colormap::Plasma, Colormap::Coolwarm] {
            for i in 0..=10 {
                let t = i as f32 / 10.0;
                let _rgb = colormap(t, cmap);
            }
        }
    }

    #[test]
    fn test_write_ppm() {
        let data = vec![0.0, 0.5, 1.0, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6];
        let dir = std::env::temp_dir().join("flux_test_ppm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.ppm");
        write_ppm(&path, &data, 3, 3, Colormap::Jet).unwrap();
        assert!(path.exists());
        let content = std::fs::read(&path).unwrap();
        assert!(content.len() > 10);
    }

    #[test]
    fn test_write_vtk() {
        let n = 4;
        let density = vec![1.0f32; n * n];
        let vx = vec![0.1f32; n * n];
        let vy = vec![0.0f32; n * n];
        let dir = std::env::temp_dir().join("flux_test_vtk");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.vtu");
        write_vtk(&path, &density, &vx, &vy, n, n).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("VTKFile"));
        assert!(content.contains("density"));
    }

    #[test]
    fn test_binary_roundtrip() {
        let solver = CpuLbmSolver::new(8, 6, 1.0);
        let dir = std::env::temp_dir().join("flux_test_bin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.flx");
        write_binary(&path, &solver).unwrap();
        let (nx, ny, fields) = read_binary(&path).unwrap();
        assert_eq!(nx, 8);
        assert_eq!(ny, 6);
        assert_eq!(fields.len(), 2);
        // Density should be ~1.0
        assert!((fields[0][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_terminal_render() {
        let data = vec![0.0, 0.5, 1.0, 0.3];
        let out = terminal_render(&data, 2, 2, 40);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_export_frame() {
        let solver = CpuLbmSolver::new(5, 5, 1.0);
        let dir = std::env::temp_dir().join("flux_test_anim");
        export_frame(&dir, &solver, 0, Colormap::Viridis).unwrap();
        assert!(dir.join("frame_000000.vtu").exists());
        assert!(dir.join("frame_000000.ppm").exists());
    }
}
