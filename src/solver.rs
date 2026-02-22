//! Simulation engine: initialization, time-stepping, convergence, checkpointing.

use crate::boundary::{
    apply_bounce_back, apply_moving_wall, apply_obstacle_bounce_back, apply_open_boundary,
    apply_zou_he_pressure, apply_zou_he_velocity, BoundaryCondition, Edge,
};
use crate::geometry::Geometry2D;
use crate::lattice::{CollisionOperator, Lattice2D};
use crate::output::{write_output, OutputField, OutputFormat};
use crate::parallel::compute_mlups;
use crate::physics::PhysicsParams;
use std::path::Path;
use std::time::Instant;

/// Simulation configuration (loaded from TOML)
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SimConfig {
    pub domain: DomainConfig,
    pub physics: PhysicsConfig,
    pub solver: SolverConfig,
    pub output: OutputConfig,
    pub boundary: BoundaryConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DomainConfig {
    pub nx: usize,
    pub ny: usize,
    #[serde(default = "default_nz")]
    pub nz: usize,
}

fn default_nz() -> usize {
    1
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PhysicsConfig {
    pub reynolds: f64,
    #[serde(default = "default_mach")]
    pub mach: f64,
}

fn default_mach() -> f64 {
    0.1
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SolverConfig {
    #[serde(default = "default_collision")]
    pub collision: CollisionOperator,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_convergence")]
    pub convergence_threshold: f64,
    #[serde(default = "default_checkpoint")]
    pub checkpoint_interval: usize,
}

fn default_collision() -> CollisionOperator {
    CollisionOperator::Bgk
}
fn default_max_steps() -> usize {
    50000
}
fn default_convergence() -> f64 {
    1e-6
}
fn default_checkpoint() -> usize {
    1000
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OutputConfig {
    #[serde(default = "default_format")]
    pub format: OutputFormat,
    #[serde(default = "default_interval")]
    pub interval: usize,
    #[serde(default = "default_fields")]
    pub fields: Vec<OutputField>,
}

fn default_format() -> OutputFormat {
    OutputFormat::Vtk
}
fn default_interval() -> usize {
    100
}
fn default_fields() -> Vec<OutputField> {
    vec![OutputField::Velocity, OutputField::Density]
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct BoundaryConfig {
    #[serde(default = "default_wall")]
    pub north: BoundaryCondition,
    #[serde(default = "default_wall")]
    pub south: BoundaryCondition,
    #[serde(default = "default_wall")]
    pub east: BoundaryCondition,
    #[serde(default = "default_wall")]
    pub west: BoundaryCondition,
}

fn default_wall() -> BoundaryCondition {
    BoundaryCondition::Wall
}

/// Simulation state
pub struct Simulation {
    pub lattice: Lattice2D,
    pub geometry: Geometry2D,
    pub config: SimConfig,
    pub physics: PhysicsParams,
    pub step: usize,
    pub converged: bool,
}

impl Simulation {
    /// Create a new simulation from config
    pub fn from_config(config: SimConfig) -> Result<Self, String> {
        let nx = config.domain.nx;
        let ny = config.domain.ny;
        let char_length = ny as f64;

        let physics = PhysicsParams::new(config.physics.reynolds, config.physics.mach, char_length);
        physics.check_mach(0.3)?;

        let tau = physics.tau();
        if tau <= 0.5 {
            return Err(format!(
                "τ = {tau:.6} ≤ 0.5 — unstable. Increase resolution or decrease Re."
            ));
        }

        let lattice = Lattice2D::new(nx, ny, tau, config.solver.collision);
        let geometry = Geometry2D::new(nx, ny);

        Ok(Self {
            lattice,
            geometry,
            config,
            physics,
            step: 0,
            converged: false,
        })
    }

    /// Apply boundary conditions based on config
    pub fn apply_boundaries(&mut self) {
        self.apply_edge_bc(Edge::North, &self.config.boundary.north.clone());
        self.apply_edge_bc(Edge::South, &self.config.boundary.south.clone());
        self.apply_edge_bc(Edge::East, &self.config.boundary.east.clone());
        self.apply_edge_bc(Edge::West, &self.config.boundary.west.clone());

        // Apply obstacle bounce-back if geometry has solids
        if self.geometry.solid_count() > 0 {
            apply_obstacle_bounce_back(&mut self.lattice, &self.geometry.solid);
        }
    }

    fn apply_edge_bc(&mut self, edge: Edge, bc: &BoundaryCondition) {
        match bc {
            BoundaryCondition::Wall => apply_bounce_back(&mut self.lattice, edge),
            BoundaryCondition::Velocity => {
                let u = self.physics.char_velocity();
                apply_zou_he_velocity(&mut self.lattice, edge, u, 0.0);
            }
            BoundaryCondition::Pressure => {
                apply_zou_he_pressure(&mut self.lattice, edge, 1.0);
            }
            BoundaryCondition::Periodic => {} // Handled in streaming
            BoundaryCondition::Open => apply_open_boundary(&mut self.lattice, edge),
            BoundaryCondition::MovingWall => {
                let u = self.physics.char_velocity();
                apply_moving_wall(&mut self.lattice, edge, u, 0.0);
            }
        }
    }

    /// Perform a single time step
    pub fn step(&mut self) {
        self.lattice.collide();
        self.lattice.stream();
        self.apply_boundaries();
        self.step += 1;
    }

    /// Check convergence based on velocity field change
    pub fn check_convergence(&self, prev_ux: &[f64], prev_uy: &[f64]) -> f64 {
        let (_, ux, uy) = self.lattice.macroscopic_fields();
        let mut diff = 0.0;
        let mut norm = 0.0;
        for i in 0..ux.len() {
            let du = ux[i] - prev_ux[i];
            let dv = uy[i] - prev_uy[i];
            diff += du * du + dv * dv;
            norm += ux[i] * ux[i] + uy[i] * uy[i];
        }
        if norm > 1e-20 {
            (diff / norm).sqrt()
        } else {
            diff.sqrt()
        }
    }

    /// Run the full simulation
    pub fn run(&mut self, output_dir: &Path) -> Result<SimulationResult, String> {
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create output dir: {e}"))?;

        let nx = self.config.domain.nx;
        let ny = self.config.domain.ny;
        let max_steps = self.config.solver.max_steps;
        let conv_threshold = self.config.solver.convergence_threshold;
        let output_interval = self.config.output.interval;
        let checkpoint_interval = self.config.solver.checkpoint_interval;

        println!("FLUX LBM Solver v{}", env!("CARGO_PKG_VERSION"));
        println!("Domain: {}×{}", nx, ny);
        println!(
            "Re = {:.1}, Ma = {:.3}",
            self.physics.reynolds, self.physics.mach
        );
        println!(
            "τ = {:.6}, ν = {:.6}",
            self.physics.tau(),
            self.physics.viscosity()
        );
        println!("Collision: {:?}", self.config.solver.collision);
        println!("Max steps: {max_steps}");
        println!();

        let start = Instant::now();
        let mut prev_ux = vec![0.0; nx * ny];
        let mut prev_uy = vec![0.0; nx * ny];
        let mut convergence = f64::MAX;

        for t in 0..max_steps {
            self.step();

            // Check convergence periodically
            if t % 100 == 0 && t > 0 {
                convergence = self.check_convergence(&prev_ux, &prev_uy);
                let (_, ux, uy) = self.lattice.macroscopic_fields();
                prev_ux.copy_from_slice(&ux);
                prev_uy.copy_from_slice(&uy);

                if convergence < conv_threshold {
                    self.converged = true;
                    println!("Converged at step {t}: residual = {convergence:.2e}");
                    break;
                }
            }

            // Output
            if t % output_interval == 0 && t > 0 {
                let path = output_dir.join(format!("output_{:06}.vtk", t));
                write_output(
                    &self.lattice,
                    &path,
                    &self.config.output.format,
                    &self.config.output.fields,
                )
                .map_err(|e| format!("Output error: {e}"))?;
            }

            // Checkpoint
            if t % checkpoint_interval == 0 && t > 0 {
                let path = output_dir.join("checkpoint.bin");
                self.save_checkpoint(&path)
                    .map_err(|e| format!("Checkpoint error: {e}"))?;
            }

            // Progress
            if t % 1000 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let mlups = compute_mlups(nx * ny, t + 1, elapsed);
                println!(
                    "Step {t:>6}/{max_steps} | residual: {convergence:.2e} | {mlups:.1} MLUPS"
                );
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let total_steps = self.step;
        let mlups = compute_mlups(nx * ny, total_steps, elapsed);

        // Final output
        let path = output_dir.join("final.vtk");
        write_output(
            &self.lattice,
            &path,
            &self.config.output.format,
            &self.config.output.fields,
        )
        .map_err(|e| format!("Final output error: {e}"))?;

        println!();
        println!("Simulation complete:");
        println!("  Steps: {total_steps}");
        println!("  Converged: {}", self.converged);
        println!("  Time: {elapsed:.2}s");
        println!("  Performance: {mlups:.1} MLUPS");

        Ok(SimulationResult {
            steps: total_steps,
            converged: self.converged,
            elapsed_secs: elapsed,
            mlups,
            final_residual: convergence,
        })
    }

    /// Save checkpoint (serialize lattice state)
    pub fn save_checkpoint(&self, path: &Path) -> std::io::Result<()> {
        let data = CheckpointData {
            step: self.step,
            nx: self.lattice.nx,
            ny: self.lattice.ny,
            tau: self.lattice.tau,
            f: self.lattice.f.clone(),
        };
        let json = serde_json::to_vec(&data)?;
        std::fs::write(path, json)
    }

    /// Load checkpoint
    pub fn load_checkpoint(&mut self, path: &Path) -> std::io::Result<()> {
        let data = std::fs::read(path)?;
        let checkpoint: CheckpointData = serde_json::from_slice(&data)?;
        self.step = checkpoint.step;
        self.lattice.f = checkpoint.f;
        println!("Loaded checkpoint from step {}", self.step);
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CheckpointData {
    step: usize,
    nx: usize,
    ny: usize,
    tau: f64,
    f: Vec<f64>,
}

/// Result of a simulation run
#[derive(Debug)]
pub struct SimulationResult {
    pub steps: usize,
    pub converged: bool,
    pub elapsed_secs: f64,
    pub mlups: f64,
    pub final_residual: f64,
}

/// Generate an example configuration file
pub fn generate_example_config() -> String {
    r#"# FLUX LBM Solver Configuration

[domain]
nx = 256
ny = 256
nz = 1  # 1 = 2D mode

[physics]
reynolds = 100
mach = 0.1

[solver]
collision = "bgk"  # or "mrt"
max_steps = 50000
convergence_threshold = 1e-6
checkpoint_interval = 1000

[output]
format = "vtk"
interval = 100
fields = ["velocity", "density", "vorticity"]

[boundary]
north = "wall"
south = "wall"
east = "pressure"
west = "velocity"
"#
    .to_string()
}

/// Load configuration from TOML file
pub fn load_config(path: &Path) -> Result<SimConfig, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read config: {e}"))?;
    toml::from_str(&content).map_err(|e| format!("Failed to parse config: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_example_config() {
        let config_str = generate_example_config();
        let config: SimConfig =
            toml::from_str(&config_str).expect("Failed to parse example config");
        assert_eq!(config.domain.nx, 256);
        assert_eq!(config.domain.ny, 256);
        assert_eq!(config.physics.reynolds, 100.0);
        assert_eq!(config.solver.collision, CollisionOperator::Bgk);
    }

    #[test]
    fn test_simulation_creation() {
        let config_str = r#"
[domain]
nx = 32
ny = 32

[physics]
reynolds = 100
mach = 0.1

[solver]
collision = "bgk"
max_steps = 100

[output]
format = "vtk"
interval = 50
fields = ["velocity"]

[boundary]
north = "wall"
south = "wall"
east = "wall"
west = "wall"
"#;
        let config: SimConfig = toml::from_str(config_str).unwrap();
        let sim = Simulation::from_config(config);
        assert!(sim.is_ok());
    }

    #[test]
    fn test_single_step() {
        let config_str = r#"
[domain]
nx = 16
ny = 16

[physics]
reynolds = 100
mach = 0.05

[solver]
collision = "bgk"
max_steps = 10

[output]
format = "vtk"
interval = 100
fields = ["velocity"]

[boundary]
north = "wall"
south = "wall"
east = "wall"
west = "wall"
"#;
        let config: SimConfig = toml::from_str(config_str).unwrap();
        let mut sim = Simulation::from_config(config).unwrap();
        sim.step();
        assert_eq!(sim.step, 1);
    }

    #[test]
    fn test_convergence_check() {
        let config_str = r#"
[domain]
nx = 8
ny = 8

[physics]
reynolds = 10
mach = 0.05

[solver]
collision = "bgk"
max_steps = 10

[output]
format = "csv"
interval = 100
fields = ["velocity"]

[boundary]
north = "wall"
south = "wall"
east = "wall"
west = "wall"
"#;
        let config: SimConfig = toml::from_str(config_str).unwrap();
        let sim = Simulation::from_config(config).unwrap();
        let (_, ux, uy) = sim.lattice.macroscopic_fields();
        let conv = sim.check_convergence(&ux, &uy);
        // Same field → zero convergence
        assert!(conv < 1e-14);
    }
}
