//! Advanced geometry: STL import, point-in-mesh tests, boundary classification,
//! immersed boundary method, rotating geometry support.

use std::io::Read;
use std::path::Path;

// ---------------------------------------------------------------------------
// Triangle / Mesh types
// ---------------------------------------------------------------------------

/// A 3D point/vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

/// A triangle with vertices and normal.
#[derive(Clone, Debug)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub normal: Vec3,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let e1 = v1.sub(v0);
        let e2 = v2.sub(v0);
        let normal = e1.cross(e2);
        let len = normal.length();
        let normal = if len > 1e-12 {
            normal.scale(1.0 / len)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };
        Self { v0, v1, v2, normal }
    }

    /// Compute centroid.
    pub fn centroid(&self) -> Vec3 {
        Vec3::new(
            (self.v0.x + self.v1.x + self.v2.x) / 3.0,
            (self.v0.y + self.v1.y + self.v2.y) / 3.0,
            (self.v0.z + self.v1.z + self.v2.z) / 3.0,
        )
    }
}

/// A triangle mesh.
#[derive(Clone, Debug)]
pub struct TriMesh {
    pub triangles: Vec<Triangle>,
}

impl TriMesh {
    pub fn new(triangles: Vec<Triangle>) -> Self {
        Self { triangles }
    }

    /// Bounding box: (min, max).
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        for tri in &self.triangles {
            for v in [tri.v0, tri.v1, tri.v2] {
                min.x = min.x.min(v.x);
                min.y = min.y.min(v.y);
                min.z = min.z.min(v.z);
                max.x = max.x.max(v.x);
                max.y = max.y.max(v.y);
                max.z = max.z.max(v.z);
            }
        }
        (min, max)
    }
}

// ---------------------------------------------------------------------------
// STL import
// ---------------------------------------------------------------------------

/// Parse an STL file (auto-detects binary vs ASCII).
pub fn load_stl<P: AsRef<Path>>(path: P) -> std::io::Result<TriMesh> {
    let data = std::fs::read(path)?;
    if data.len() > 5 && &data[0..5] == b"solid" {
        // Could be ASCII, but binary files sometimes start with "solid" too
        // Check if it's actually binary by looking at expected size
        if data.len() > 84 {
            let ntri = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
            let expected = 84 + ntri * 50;
            if data.len() == expected {
                return parse_stl_binary(&data);
            }
        }
        parse_stl_ascii(&data)
    } else {
        parse_stl_binary(&data)
    }
}

/// Parse STL from bytes (auto-detect).
pub fn parse_stl(data: &[u8]) -> std::io::Result<TriMesh> {
    if data.len() > 5 && &data[0..5] == b"solid" {
        if data.len() > 84 {
            let ntri = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
            let expected = 84 + ntri * 50;
            if data.len() == expected {
                return parse_stl_binary(data);
            }
        }
        parse_stl_ascii(data)
    } else {
        parse_stl_binary(data)
    }
}

fn parse_stl_binary(data: &[u8]) -> std::io::Result<TriMesh> {
    if data.len() < 84 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "STL binary too short",
        ));
    }
    let ntri = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let mut triangles = Vec::with_capacity(ntri);
    let mut offset = 84;
    for _ in 0..ntri {
        if offset + 50 > data.len() {
            break;
        }
        let _normal = read_vec3(&data[offset..]);
        let v0 = read_vec3(&data[offset + 12..]);
        let v1 = read_vec3(&data[offset + 24..]);
        let v2 = read_vec3(&data[offset + 36..]);
        triangles.push(Triangle::new(v0, v1, v2));
        offset += 50;
    }
    Ok(TriMesh::new(triangles))
}

fn read_vec3(data: &[u8]) -> Vec3 {
    let x = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let y = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let z = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    Vec3::new(x, y, z)
}

fn parse_stl_ascii(data: &[u8]) -> std::io::Result<TriMesh> {
    let text = String::from_utf8_lossy(data);
    let mut triangles = Vec::new();
    let mut vertices: Vec<Vec3> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("vertex") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                vertices.push(Vec3::new(x, y, z));
                if vertices.len() == 3 {
                    triangles.push(Triangle::new(vertices[0], vertices[1], vertices[2]));
                    vertices.clear();
                }
            }
        }
    }
    Ok(TriMesh::new(triangles))
}

// ---------------------------------------------------------------------------
// Point-in-polygon (2D) / Point-in-mesh (3D ray casting)
// ---------------------------------------------------------------------------

/// 2D point-in-polygon test using ray casting (XY plane).
pub fn point_in_polygon_2d(px: f32, py: f32, polygon: &[(f32, f32)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// 3D ray-casting point-in-mesh test (assumes watertight mesh).
/// Casts ray along +X and counts intersections.
pub fn point_in_mesh(point: Vec3, mesh: &TriMesh) -> bool {
    let mut count = 0u32;
    let ray_dir = Vec3::new(1.0, 0.0, 0.0);
    for tri in &mesh.triangles {
        if ray_triangle_intersect(point, ray_dir, tri) {
            count += 1;
        }
    }
    count % 2 == 1
}

/// Möller–Trumbore ray-triangle intersection.
fn ray_triangle_intersect(origin: Vec3, dir: Vec3, tri: &Triangle) -> bool {
    let eps = 1e-7f32;
    let e1 = tri.v1.sub(tri.v0);
    let e2 = tri.v2.sub(tri.v0);
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < eps {
        return false;
    }
    let f = 1.0 / a;
    let s = origin.sub(tri.v0);
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = f * e2.dot(q);
    t > eps
}

// ---------------------------------------------------------------------------
// Boundary classification
// ---------------------------------------------------------------------------

/// Boundary node type for lattice classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryType {
    Fluid,
    Wall,
    Inlet,
    Outlet,
}

/// Classify lattice nodes based on a 2D polygon boundary.
/// Returns a grid of boundary types (row-major, nx × ny).
pub fn classify_boundary_2d(
    nx: usize,
    ny: usize,
    polygon: &[(f32, f32)],
) -> Vec<BoundaryType> {
    let mut flags = vec![BoundaryType::Fluid; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            if !point_in_polygon_2d(x as f32 + 0.5, y as f32 + 0.5, polygon) {
                flags[y * nx + x] = BoundaryType::Wall;
            }
        }
    }
    flags
}

/// Classify nodes using a 3D mesh (XY slice at z=0).
pub fn classify_boundary_mesh(nx: usize, ny: usize, mesh: &TriMesh) -> Vec<BoundaryType> {
    let mut flags = vec![BoundaryType::Fluid; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            if point_in_mesh(p, mesh) {
                flags[y * nx + x] = BoundaryType::Wall;
            }
        }
    }
    flags
}

// ---------------------------------------------------------------------------
// Immersed Boundary Method (IBM) for curved surfaces
// ---------------------------------------------------------------------------

/// An immersed boundary point with position and velocity.
#[derive(Clone, Debug)]
pub struct IbPoint {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
}

/// Discrete delta function (2-point) for IBM force spreading.
fn delta_h(r: f32, h: f32) -> f32 {
    let r_abs = (r / h).abs();
    if r_abs >= 1.0 {
        0.0
    } else {
        (1.0 - r_abs) / h
    }
}

/// Compute IBM force density contribution at lattice node (lx, ly)
/// from an immersed boundary point.
pub fn ib_force_at_node(ib: &IbPoint, lx: f32, ly: f32, h: f32) -> [f32; 2] {
    let dx = delta_h(lx - ib.x, h);
    let dy = delta_h(ly - ib.y, h);
    let weight = dx * dy;
    [ib.vx * weight, ib.vy * weight]
}

/// Generate immersed boundary points for a circle.
pub fn ib_circle(cx: f32, cy: f32, radius: f32, n_points: usize) -> Vec<IbPoint> {
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (n_points as f32);
        points.push(IbPoint {
            x: cx + radius * theta.cos(),
            y: cy + radius * theta.sin(),
            vx: 0.0,
            vy: 0.0,
        });
    }
    points
}

// ---------------------------------------------------------------------------
// Rotating geometry
// ---------------------------------------------------------------------------

/// Rotate a set of IB points around a centre by angle (radians).
pub fn rotate_ib_points(points: &mut [IbPoint], cx: f32, cy: f32, angle: f32) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    for p in points.iter_mut() {
        let dx = p.x - cx;
        let dy = p.y - cy;
        p.x = cx + dx * cos_a - dy * sin_a;
        p.y = cy + dx * sin_a + dy * cos_a;
    }
}

/// Rotate a mesh around the Z axis by angle (radians) about centre (cx, cy).
pub fn rotate_mesh(mesh: &mut TriMesh, cx: f32, cy: f32, angle: f32) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    for tri in &mut mesh.triangles {
        for v in [&mut tri.v0, &mut tri.v1, &mut tri.v2] {
            let dx = v.x - cx;
            let dy = v.y - cy;
            v.x = cx + dx * cos_a - dy * sin_a;
            v.y = cy + dx * sin_a + dy * cos_a;
        }
        // Recompute normal
        let e1 = tri.v1.sub(tri.v0);
        let e2 = tri.v2.sub(tri.v0);
        let n = e1.cross(e2);
        let len = n.length();
        tri.normal = if len > 1e-12 {
            n.scale(1.0 / len)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!((c.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_normal() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!((tri.normal.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_point_in_polygon_square() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(point_in_polygon_2d(5.0, 5.0, &square));
        assert!(!point_in_polygon_2d(15.0, 5.0, &square));
        assert!(!point_in_polygon_2d(-1.0, 5.0, &square));
    }

    #[test]
    fn test_point_in_polygon_triangle() {
        let tri = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        assert!(point_in_polygon_2d(5.0, 3.0, &tri));
        assert!(!point_in_polygon_2d(0.0, 10.0, &tri));
    }

    #[test]
    fn test_classify_boundary_2d() {
        let polygon = vec![(1.0, 1.0), (8.0, 1.0), (8.0, 8.0), (1.0, 8.0)];
        let flags = classify_boundary_2d(10, 10, &polygon);
        assert_eq!(flags[5 * 10 + 5], BoundaryType::Fluid); // inside
        assert_eq!(flags[0 * 10 + 0], BoundaryType::Wall); // outside
    }

    #[test]
    fn test_ib_circle() {
        let points = ib_circle(5.0, 5.0, 3.0, 32);
        assert_eq!(points.len(), 32);
        // Check radius
        for p in &points {
            let r = ((p.x - 5.0).powi(2) + (p.y - 5.0).powi(2)).sqrt();
            assert!((r - 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rotate_ib_points() {
        let mut points = vec![IbPoint {
            x: 2.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
        }];
        rotate_ib_points(&mut points, 0.0, 0.0, std::f32::consts::FRAC_PI_2);
        assert!((points[0].x).abs() < 1e-5);
        assert!((points[0].y - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_stl_ascii_parse() {
        let stl = b"solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test";
        let mesh = parse_stl(stl).unwrap();
        assert_eq!(mesh.triangles.len(), 1);
    }

    #[test]
    fn test_stl_binary_parse() {
        // Create a minimal binary STL with 1 triangle
        let mut data = vec![0u8; 84 + 50];
        // header: 80 bytes (zeros)
        // triangle count
        data[80] = 1;
        data[81] = 0;
        data[82] = 0;
        data[83] = 0;
        // normal (0,0,1)
        let one = 1.0f32.to_le_bytes();
        data[92] = one[0];
        data[93] = one[1];
        data[94] = one[2];
        data[95] = one[3];
        // v0 = (0,0,0) already zeros
        // v1 = (1,0,0)
        let offset = 84 + 12;
        data[offset] = one[0];
        data[offset + 1] = one[1];
        data[offset + 2] = one[2];
        data[offset + 3] = one[3];
        // v2 = (0,1,0)
        let offset2 = 84 + 24;
        data[offset2 + 4] = one[0];
        data[offset2 + 5] = one[1];
        data[offset2 + 6] = one[2];
        data[offset2 + 7] = one[3];

        let mesh = parse_stl_binary(&data).unwrap();
        assert_eq!(mesh.triangles.len(), 1);
    }

    #[test]
    fn test_bounding_box() {
        let mesh = TriMesh::new(vec![Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 2.0),
        )]);
        let (min, max) = mesh.bounding_box();
        assert!((min.x - 0.0).abs() < 1e-6);
        assert!((max.x - 5.0).abs() < 1e-6);
        assert!((max.y - 3.0).abs() < 1e-6);
        assert!((max.z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_function() {
        let h = 1.0;
        assert!((delta_h(0.0, h) - 1.0).abs() < 1e-6);
        assert!((delta_h(1.0, h) - 0.0).abs() < 1e-6);
        assert!((delta_h(0.5, h) - 0.5).abs() < 1e-6);
    }
}
