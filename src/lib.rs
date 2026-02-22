//! FLUX — Lattice Boltzmann Method fluid dynamics solver
//!
//! A state-of-the-art LBM solver in pure Rust for university-grade CFD.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    dead_code
)]

pub mod boundary;
pub mod geometry;
pub mod lattice;
pub mod multiphase;
pub mod output;
pub mod parallel;
pub mod physics;
pub mod solver;
pub mod thermal;
pub mod turbulence;
pub mod validation;
