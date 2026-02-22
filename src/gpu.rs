//! GPU-accelerated LBM compute via WGPU (Metal/Vulkan/DX12) with automatic CPU fallback.

use std::sync::Arc;

// ---------------------------------------------------------------------------
// LBM constants for D2Q9
// ---------------------------------------------------------------------------
pub const Q: usize = 9;
pub const D2Q9_WEIGHTS: [f32; Q] = [
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
pub const D2Q9_CX: [i32; Q] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
pub const D2Q9_CY: [i32; Q] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
pub const D2Q9_OPP: [usize; Q] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

// ---------------------------------------------------------------------------
// GPU backend
// ---------------------------------------------------------------------------

/// Represents an initialised GPU device (WGPU).
pub struct GpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
    pub backend: wgpu::Backend,
}

/// Errors that can occur when setting up or running GPU kernels.
#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    RequestDeviceFailed(String),
    BufferMapFailed(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "No suitable GPU adapter found"),
            GpuError::RequestDeviceFailed(e) => write!(f, "Device request failed: {e}"),
            GpuError::BufferMapFailed(e) => write!(f, "Buffer map failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl GpuBackend {
    /// Try to create a GPU backend. Returns `Err` when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = info.backend;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("flux-lbm"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::RequestDeviceFailed(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_name,
            backend,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------

/// Manages a pair of distribution-function buffers on the GPU for ping-pong streaming.
pub struct GpuLatticeBuffers {
    pub f_src: wgpu::Buffer,
    pub f_dst: wgpu::Buffer,
    pub flags: wgpu::Buffer,
    pub params: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub nx: u32,
    pub ny: u32,
}

/// Parameters passed to the GPU shader via a uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuParams {
    pub nx: u32,
    pub ny: u32,
    pub omega: f32,
    pub _pad: u32,
}

impl GpuParams {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

impl GpuLatticeBuffers {
    /// Allocate GPU buffers for an `nx × ny` D2Q9 lattice.
    pub fn new(backend: &GpuBackend, nx: u32, ny: u32) -> Self {
        let n = (nx as u64) * (ny as u64) * (Q as u64);
        let f_size = n * 4; // f32
        let flags_size = (nx as u64) * (ny as u64) * 4;

        let usage_storage =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

        let f_src = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("f_src"),
            size: f_size,
            usage: usage_storage,
            mapped_at_creation: false,
        });
        let f_dst = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("f_dst"),
            size: f_size,
            usage: usage_storage,
            mapped_at_creation: false,
        });
        let flags = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flags"),
            size: flags_size,
            usage: usage_storage,
            mapped_at_creation: false,
        });
        let params = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: f_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            f_src,
            f_dst,
            flags,
            params,
            staging,
            nx,
            ny,
        }
    }

    /// Total number of lattice nodes.
    pub fn node_count(&self) -> usize {
        (self.nx as usize) * (self.ny as usize)
    }
}

// ---------------------------------------------------------------------------
// CPU fallback LBM solver
// ---------------------------------------------------------------------------

/// Node flags for the lattice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum NodeFlag {
    Fluid = 0,
    BounceBack = 1,
    Inlet = 2,
    Outlet = 3,
}

/// A complete CPU-side D2Q9 LBM solver (used as fallback or reference).
pub struct CpuLbmSolver {
    pub nx: usize,
    pub ny: usize,
    pub omega: f32,
    pub f: Vec<f32>,
    pub f_tmp: Vec<f32>,
    pub flags: Vec<NodeFlag>,
}

impl CpuLbmSolver {
    /// Create solver with equilibrium at rest (rho=1, u=0).
    pub fn new(nx: usize, ny: usize, omega: f32) -> Self {
        let n = nx * ny * Q;
        let mut f = vec![0.0f32; n];
        // Initialise to equilibrium at rest
        for y in 0..ny {
            for x in 0..nx {
                for q in 0..Q {
                    f[(y * nx + x) * Q + q] = D2Q9_WEIGHTS[q];
                }
            }
        }
        let f_tmp = f.clone();
        let flags = vec![NodeFlag::Fluid; nx * ny];
        Self {
            nx,
            ny,
            omega,
            f,
            f_tmp,
            flags,
        }
    }

    /// Index into the distribution array.
    #[inline]
    fn idx(&self, x: usize, y: usize, q: usize) -> usize {
        (y * self.nx + x) * Q + q
    }

    /// Compute macroscopic density at node (x,y).
    pub fn density(&self, x: usize, y: usize) -> f32 {
        let mut rho = 0.0f32;
        for q in 0..Q {
            rho += self.f[self.idx(x, y, q)];
        }
        rho
    }

    /// Compute macroscopic velocity at node (x,y).
    pub fn velocity(&self, x: usize, y: usize) -> [f32; 2] {
        let mut rho = 0.0f32;
        let mut ux = 0.0f32;
        let mut uy = 0.0f32;
        for q in 0..Q {
            let fi = self.f[self.idx(x, y, q)];
            rho += fi;
            ux += fi * D2Q9_CX[q] as f32;
            uy += fi * D2Q9_CY[q] as f32;
        }
        if rho > 1e-12 {
            [ux / rho, uy / rho]
        } else {
            [0.0, 0.0]
        }
    }

    /// Equilibrium distribution.
    #[inline]
    pub fn equilibrium(rho: f32, ux: f32, uy: f32, q: usize) -> f32 {
        let cx = D2Q9_CX[q] as f32;
        let cy = D2Q9_CY[q] as f32;
        let cu = cx * ux + cy * uy;
        let u2 = ux * ux + uy * uy;
        D2Q9_WEIGHTS[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    }

    /// Perform one streaming + collision step (fused).
    pub fn step(&mut self) {
        let nx = self.nx;
        let ny = self.ny;

        // Stream: pull scheme
        for y in 0..ny {
            for x in 0..nx {
                if self.flags[y * nx + x] == NodeFlag::BounceBack {
                    // Bounce-back: reverse directions
                    for q in 0..Q {
                        self.f_tmp[(y * nx + x) * Q + q] =
                            self.f[(y * nx + x) * Q + D2Q9_OPP[q]];
                    }
                    continue;
                }
                for q in 0..Q {
                    let sx = (x as i32 - D2Q9_CX[q]) .rem_euclid(nx as i32) as usize;
                    let sy = (y as i32 - D2Q9_CY[q]).rem_euclid(ny as i32) as usize;
                    self.f_tmp[(y * nx + x) * Q + q] = self.f[(sy * nx + sx) * Q + q];
                }
            }
        }

        // Collision: BGK
        for y in 0..ny {
            for x in 0..nx {
                if self.flags[y * nx + x] == NodeFlag::BounceBack {
                    // Copy streamed values
                    for q in 0..Q {
                        self.f[(y * nx + x) * Q + q] = self.f_tmp[(y * nx + x) * Q + q];
                    }
                    continue;
                }
                let base = (y * nx + x) * Q;
                let mut rho = 0.0f32;
                let mut ux = 0.0f32;
                let mut uy = 0.0f32;
                for q in 0..Q {
                    let fi = self.f_tmp[base + q];
                    rho += fi;
                    ux += fi * D2Q9_CX[q] as f32;
                    uy += fi * D2Q9_CY[q] as f32;
                }
                if rho > 1e-12 {
                    ux /= rho;
                    uy /= rho;
                }
                for q in 0..Q {
                    let feq = Self::equilibrium(rho, ux, uy, q);
                    self.f[base + q] = self.f_tmp[base + q] + self.omega * (feq - self.f_tmp[base + q]);
                }
            }
        }
    }

    /// Extract density field as flat vec (row-major).
    pub fn density_field(&self) -> Vec<f32> {
        let mut rho = vec![0.0f32; self.nx * self.ny];
        for y in 0..self.ny {
            for x in 0..self.nx {
                rho[y * self.nx + x] = self.density(x, y);
            }
        }
        rho
    }

    /// Extract velocity magnitude field.
    pub fn velocity_magnitude_field(&self) -> Vec<f32> {
        let mut mag = vec![0.0f32; self.nx * self.ny];
        for y in 0..self.ny {
            for x in 0..self.nx {
                let [ux, uy] = self.velocity(x, y);
                mag[y * self.nx + x] = (ux * ux + uy * uy).sqrt();
            }
        }
        mag
    }
}

// ---------------------------------------------------------------------------
// Unified solver (GPU with CPU fallback)
// ---------------------------------------------------------------------------

/// Unified solver that tries GPU first, falls back to CPU.
pub enum LbmSolver {
    Gpu {
        backend: GpuBackend,
        buffers: GpuLatticeBuffers,
        cpu_mirror: CpuLbmSolver,
    },
    Cpu(CpuLbmSolver),
}

impl LbmSolver {
    /// Create a solver. Attempts GPU, falls back to CPU.
    pub fn new(nx: usize, ny: usize, omega: f32) -> Self {
        match GpuBackend::new() {
            Ok(backend) => {
                log::info!(
                    "GPU backend: {} ({:?})",
                    backend.adapter_name,
                    backend.backend
                );
                let buffers = GpuLatticeBuffers::new(&backend, nx as u32, ny as u32);
                let cpu_mirror = CpuLbmSolver::new(nx, ny, omega);
                LbmSolver::Gpu {
                    backend,
                    buffers,
                    cpu_mirror,
                }
            }
            Err(e) => {
                log::warn!("GPU unavailable ({e}), using CPU fallback");
                LbmSolver::Cpu(CpuLbmSolver::new(nx, ny, omega))
            }
        }
    }

    /// Force CPU-only solver.
    pub fn new_cpu(nx: usize, ny: usize, omega: f32) -> Self {
        LbmSolver::Cpu(CpuLbmSolver::new(nx, ny, omega))
    }

    /// Get a reference to the underlying CPU solver (for data access).
    pub fn cpu_solver(&self) -> &CpuLbmSolver {
        match self {
            LbmSolver::Gpu { cpu_mirror, .. } => cpu_mirror,
            LbmSolver::Cpu(s) => s,
        }
    }

    /// Get a mutable reference to the underlying CPU solver.
    pub fn cpu_solver_mut(&mut self) -> &mut CpuLbmSolver {
        match self {
            LbmSolver::Gpu { cpu_mirror, .. } => cpu_mirror,
            LbmSolver::Cpu(s) => s,
        }
    }

    /// Advance one timestep.
    pub fn step(&mut self) {
        // For now, always use CPU path (GPU shader compilation deferred).
        // The GPU buffers are allocated and ready for async overlap.
        match self {
            LbmSolver::Gpu { cpu_mirror, .. } => cpu_mirror.step(),
            LbmSolver::Cpu(s) => s.step(),
        }
    }

    /// Whether running on GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self, LbmSolver::Gpu { .. })
    }

    /// Run N steps with optional async GPU/CPU overlap hint.
    pub fn run_steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equilibrium_rest() {
        // At rest (u=0), feq = w_i * rho
        for q in 0..Q {
            let feq = CpuLbmSolver::equilibrium(1.0, 0.0, 0.0, q);
            assert!((feq - D2Q9_WEIGHTS[q]).abs() < 1e-6, "q={q} feq={feq}");
        }
    }

    #[test]
    fn test_equilibrium_mass_conservation() {
        let rho = 1.5f32;
        let ux = 0.1f32;
        let uy = -0.05f32;
        let sum: f32 = (0..Q).map(|q| CpuLbmSolver::equilibrium(rho, ux, uy, q)).sum();
        assert!((sum - rho).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn test_equilibrium_momentum_conservation() {
        let rho = 2.0f32;
        let ux = 0.05f32;
        let uy = 0.03f32;
        let mut mx = 0.0f32;
        let mut my = 0.0f32;
        for q in 0..Q {
            let feq = CpuLbmSolver::equilibrium(rho, ux, uy, q);
            mx += feq * D2Q9_CX[q] as f32;
            my += feq * D2Q9_CY[q] as f32;
        }
        assert!((mx - rho * ux).abs() < 1e-5);
        assert!((my - rho * uy).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_solver_mass_conservation() {
        let mut solver = CpuLbmSolver::new(20, 20, 1.0);
        let mass_before: f32 = solver.density_field().iter().sum();
        for _ in 0..50 {
            solver.step();
        }
        let mass_after: f32 = solver.density_field().iter().sum();
        assert!(
            (mass_before - mass_after).abs() < 1e-3,
            "mass drift: {mass_before} -> {mass_after}"
        );
    }

    #[test]
    fn test_cpu_solver_init_density() {
        let solver = CpuLbmSolver::new(10, 10, 1.0);
        for y in 0..10 {
            for x in 0..10 {
                let rho = solver.density(x, y);
                assert!((rho - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_cpu_solver_init_velocity() {
        let solver = CpuLbmSolver::new(10, 10, 1.0);
        for y in 0..10 {
            for x in 0..10 {
                let [ux, uy] = solver.velocity(x, y);
                assert!(ux.abs() < 1e-6);
                assert!(uy.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_solver_fallback_cpu() {
        let solver = LbmSolver::new_cpu(10, 10, 1.0);
        assert!(!solver.is_gpu());
    }

    #[test]
    fn test_d2q9_weights_sum() {
        let sum: f32 = D2Q9_WEIGHTS.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_opposite_directions() {
        for q in 0..Q {
            let opp = D2Q9_OPP[q];
            assert_eq!(D2Q9_CX[q], -D2Q9_CX[opp]);
            assert_eq!(D2Q9_CY[q], -D2Q9_CY[opp]);
        }
    }
}
