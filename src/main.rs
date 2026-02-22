//! FLUX — Lattice Boltzmann Method fluid dynamics solver
//!
//! A state-of-the-art LBM solver in pure Rust for university-grade CFD.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    dead_code
)]

mod boundary;
mod geometry;
mod lattice;
mod output;
mod parallel;
mod physics;
mod solver;
mod turbulence;
mod validation;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "flux")]
#[command(about = "FLUX — Lattice Boltzmann Method fluid dynamics solver")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a simulation from a config file
    Run {
        /// Path to the configuration TOML file
        config: PathBuf,
        /// Output directory
        #[arg(short, long, default_value = "output")]
        output: PathBuf,
    },
    /// Run performance benchmark (MLUPS)
    Benchmark {
        /// Grid size (NxN)
        #[arg(short = 'n', long, default_value = "256")]
        size: usize,
        /// Number of steps
        #[arg(short, long, default_value = "1000")]
        steps: usize,
    },
    /// Run validation suite against analytical solutions
    Validate,
    /// Generate an example configuration file
    Init {
        /// Output path for the config file
        #[arg(default_value = "flux.toml")]
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { config, output } => {
            let cfg = match solver::load_config(&config) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let mut sim = match solver::Simulation::from_config(cfg) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            match sim.run(&output) {
                Ok(result) => {
                    if !result.converged {
                        eprintln!("Warning: simulation did not converge within max steps");
                    }
                }
                Err(e) => {
                    eprintln!("Simulation error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Benchmark { size, steps } => {
            run_benchmark(size, steps);
        }
        Commands::Validate => {
            validation::run_validation_suite();
        }
        Commands::Init { path } => {
            let config = solver::generate_example_config();
            std::fs::write(&path, config).expect("Failed to write config file");
            println!("Generated example config at {}", path.display());
        }
    }
}

fn run_benchmark(size: usize, steps: usize) {
    use lattice::CollisionOperator;
    use std::time::Instant;

    println!("FLUX Benchmark");
    println!("==============");
    println!("Grid: {size}×{size}");
    println!("Steps: {steps}");
    println!();

    for collision in &[CollisionOperator::Bgk, CollisionOperator::Mrt] {
        let mut lat = lattice::Lattice2D::new(size, size, 0.8, *collision);

        let start = Instant::now();
        for _ in 0..steps {
            lat.collide();
            lat.stream();
        }
        let elapsed = start.elapsed().as_secs_f64();
        let mlups = parallel::compute_mlups(size * size, steps, elapsed);
        println!("{:?}: {mlups:.1} MLUPS ({elapsed:.2}s)", collision);
    }
}

#[cfg(test)]
mod tests {
    // Integration tests are in the validation module
}
