//! Geometry definitions for LBM simulations.
//!
//! Provides tools for defining solid boundaries in the domain:
//! - Rectangular domains
//! - Circle/sphere obstacles
//! - STL import for complex 3D geometries
//! - Signed distance fields for curved boundaries

/// A 2D geometry represented as a boolean solid mask.
#[derive(Clone)]
pub struct Geometry2D {
    pub nx: usize,
    pub ny: usize,
    /// true = solid, false = fluid
    pub solid: Vec<bool>,
}

impl Geometry2D {
    /// Create an empty (all fluid) domain
    pub fn new(nx: usize, ny: usize) -> Self {
        Self {
            nx,
            ny,
            solid: vec![false; nx * ny],
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    #[inline]
    pub fn is_solid(&self, x: usize, y: usize) -> bool {
        self.solid[self.idx(x, y)]
    }

    #[inline]
    pub fn set_solid(&mut self, x: usize, y: usize, solid: bool) {
        let idx = self.idx(x, y);
        self.solid[idx] = solid;
    }

    /// Add a circular obstacle centered at (cx, cy) with radius r.
    pub fn add_circle(&mut self, cx: f64, cy: f64, r: f64) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if dx * dx + dy * dy <= r * r {
                    self.set_solid(x, y, true);
                }
            }
        }
    }

    /// Add a rectangular obstacle from (x0, y0) to (x1, y1) inclusive.
    pub fn add_rectangle(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        for y in y0..=y1.min(self.ny - 1) {
            for x in x0..=x1.min(self.nx - 1) {
                self.set_solid(x, y, true);
            }
        }
    }

    /// Add channel walls (solid top and bottom rows).
    pub fn add_channel_walls(&mut self) {
        for x in 0..self.nx {
            self.set_solid(x, 0, true);
            self.set_solid(x, self.ny - 1, true);
        }
    }

    /// Compute signed distance field from the solid boundary.
    /// Positive = inside fluid, negative = inside solid.
    /// Uses brute-force nearest-boundary search (fine for moderate domains).
    pub fn signed_distance_field(&self) -> Vec<f64> {
        let n = self.nx * self.ny;
        let mut sdf = vec![f64::MAX; n];

        // Collect boundary cells (fluid cells adjacent to solid)
        let mut boundary: Vec<(usize, usize)> = Vec::new();
        for y in 0..self.ny {
            for x in 0..self.nx {
                if !self.is_solid(x, y) {
                    continue;
                }
                // Check if adjacent to fluid
                let neighbors = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx < self.nx && ny < self.ny && !self.is_solid(nx, ny) {
                        boundary.push((x, y));
                        break;
                    }
                }
            }
        }

        for y in 0..self.ny {
            for x in 0..self.nx {
                let mut min_dist = f64::MAX;
                for &(bx, by) in &boundary {
                    let dx = x as f64 - bx as f64;
                    let dy = y as f64 - by as f64;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                let sign = if self.is_solid(x, y) { -1.0 } else { 1.0 };
                sdf[self.idx(x, y)] = sign * min_dist;
            }
        }
        sdf
    }

    /// Count fluid nodes
    pub fn fluid_count(&self) -> usize {
        self.solid.iter().filter(|&&s| !s).count()
    }

    /// Count solid nodes
    pub fn solid_count(&self) -> usize {
        self.solid.iter().filter(|&&s| *s).count()
    }
}

/// A 3D geometry
#[derive(Clone)]
pub struct Geometry3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub solid: Vec<bool>,
}

impl Geometry3D {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            solid: vec![false; nx * ny * nz],
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.nx * self.ny + y * self.nx + x
    }

    #[inline]
    pub fn is_solid(&self, x: usize, y: usize, z: usize) -> bool {
        self.solid[self.idx(x, y, z)]
    }

    pub fn set_solid(&mut self, x: usize, y: usize, z: usize, solid: bool) {
        let idx = self.idx(x, y, z);
        self.solid[idx] = solid;
    }

    /// Add a sphere obstacle
    pub fn add_sphere(&mut self, cx: f64, cy: f64, cz: f64, r: f64) {
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let dz = z as f64 - cz;
                    if dx * dx + dy * dy + dz * dz <= r * r {
                        self.set_solid(x, y, z, true);
                    }
                }
            }
        }
    }

    /// Import geometry from STL binary file.
    /// Voxelizes the STL mesh into the grid.
    pub fn import_stl(&mut self, path: &std::path::Path, scale: f64) -> Result<(), String> {
        let data = std::fs::read(path).map_err(|e| format!("Failed to read STL: {e}"))?;

        if data.len() < 84 {
            return Err("STL file too small".to_string());
        }

        // Binary STL: 80 byte header + 4 byte triangle count
        let num_triangles =
            u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
        let expected_size = 84 + num_triangles * 50;
        if data.len() < expected_size {
            return Err(format!(
                "STL file truncated: expected {expected_size} bytes, got {}",
                data.len()
            ));
        }

        // Parse triangles and voxelize
        for i in 0..num_triangles {
            let offset = 84 + i * 50;
            // Skip normal (12 bytes), read 3 vertices (36 bytes)
            let mut vertices = [(0.0f64, 0.0f64, 0.0f64); 3];
            for v in 0..3 {
                let vo = offset + 12 + v * 12;
                let x = f32::from_le_bytes([data[vo], data[vo + 1], data[vo + 2], data[vo + 3]])
                    as f64
                    * scale;
                let y = f32::from_le_bytes([
                    data[vo + 4],
                    data[vo + 5],
                    data[vo + 6],
                    data[vo + 7],
                ]) as f64
                    * scale;
                let z = f32::from_le_bytes([
                    data[vo + 8],
                    data[vo + 9],
                    data[vo + 10],
                    data[vo + 11],
                ]) as f64
                    * scale;
                vertices[v] = (x, y, z);
            }

            // Simple voxelization: mark cells that contain triangle vertices
            for &(vx, vy, vz) in &vertices {
                let ix = vx.round() as i64;
                let iy = vy.round() as i64;
                let iz = vz.round() as i64;
                if ix >= 0
                    && ix < self.nx as i64
                    && iy >= 0
                    && iy < self.ny as i64
                    && iz >= 0
                    && iz < self.nz as i64
                {
                    self.set_solid(ix as usize, iy as usize, iz as usize, true);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_geometry() {
        let geo = Geometry2D::new(10, 10);
        assert_eq!(geo.fluid_count(), 100);
        assert_eq!(geo.solid_count(), 0);
    }

    #[test]
    fn test_add_circle() {
        let mut geo = Geometry2D::new(50, 50);
        geo.add_circle(25.0, 25.0, 5.0);
        assert!(geo.is_solid(25, 25));
        assert!(!geo.is_solid(0, 0));
        assert!(geo.solid_count() > 0);
    }

    #[test]
    fn test_circle_area() {
        let mut geo = Geometry2D::new(200, 200);
        geo.add_circle(100.0, 100.0, 20.0);
        let solid = geo.solid_count() as f64;
        let expected = std::f64::consts::PI * 20.0 * 20.0;
        // Should be within ~5% for this resolution
        assert!(
            (solid - expected).abs() / expected < 0.05,
            "Circle area mismatch: got {solid}, expected {expected}"
        );
    }

    #[test]
    fn test_add_rectangle() {
        let mut geo = Geometry2D::new(20, 20);
        geo.add_rectangle(5, 5, 10, 10);
        assert!(geo.is_solid(7, 7));
        assert!(!geo.is_solid(0, 0));
        assert_eq!(geo.solid_count(), 36); // 6x6
    }

    #[test]
    fn test_channel_walls() {
        let mut geo = Geometry2D::new(20, 10);
        geo.add_channel_walls();
        for x in 0..20 {
            assert!(geo.is_solid(x, 0));
            assert!(geo.is_solid(x, 9));
            assert!(!geo.is_solid(x, 5));
        }
    }

    #[test]
    fn test_signed_distance_field() {
        let mut geo = Geometry2D::new(20, 20);
        geo.add_circle(10.0, 10.0, 3.0);
        let sdf = geo.signed_distance_field();
        // Center of circle should be negative (inside solid)
        assert!(sdf[geo.idx(10, 10)] < 0.0);
        // Far corner should be positive (fluid)
        assert!(sdf[geo.idx(0, 0)] > 0.0);
    }

    #[test]
    fn test_3d_sphere() {
        let mut geo = Geometry3D::new(30, 30, 30);
        geo.add_sphere(15.0, 15.0, 15.0, 5.0);
        assert!(geo.is_solid(15, 15, 15));
        assert!(!geo.is_solid(0, 0, 0));
    }
}
