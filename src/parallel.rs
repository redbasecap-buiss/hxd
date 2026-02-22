//! Parallelization utilities for LBM simulations.
//!
//! Uses Rayon for shared-memory parallelism with domain decomposition
//! for cache efficiency.

use rayon::prelude::*;

/// Configure the Rayon thread pool.
pub fn configure_threads(num_threads: Option<usize>) {
    if let Some(n) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // Ignore error if already initialized
    }
}

/// Decompose a 2D domain into row-based chunks for parallel processing.
/// Returns (start_y, end_y) pairs.
pub fn decompose_rows(ny: usize, num_chunks: usize) -> Vec<(usize, usize)> {
    let chunk_size = ny / num_chunks;
    let remainder = ny % num_chunks;
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut start = 0;
    for i in 0..num_chunks {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + chunk_size + extra;
        chunks.push((start, end));
        start = end;
    }
    chunks
}

/// Parallel reduction: compute global sum of a field.
pub fn parallel_sum(data: &[f64]) -> f64 {
    data.par_iter().sum()
}

/// Parallel reduction: compute global max of a field.
pub fn parallel_max(data: &[f64]) -> f64 {
    data.par_iter()
        .cloned()
        .reduce(|| f64::NEG_INFINITY, f64::max)
}

/// Parallel reduction: compute global min of a field.
pub fn parallel_min(data: &[f64]) -> f64 {
    data.par_iter().cloned().reduce(|| f64::INFINITY, f64::min)
}

/// Compute MLUPS (Million Lattice Updates Per Second)
pub fn compute_mlups(num_nodes: usize, num_steps: usize, elapsed_secs: f64) -> f64 {
    (num_nodes as f64 * num_steps as f64) / (elapsed_secs * 1e6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_rows_even() {
        let chunks = decompose_rows(100, 4);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], (0, 25));
        assert_eq!(chunks[1], (25, 50));
        assert_eq!(chunks[2], (50, 75));
        assert_eq!(chunks[3], (75, 100));
    }

    #[test]
    fn test_decompose_rows_uneven() {
        let chunks = decompose_rows(10, 3);
        assert_eq!(chunks.len(), 3);
        // 10 / 3 = 3 remainder 1
        assert_eq!(chunks[0], (0, 4)); // gets extra
        assert_eq!(chunks[1], (4, 7));
        assert_eq!(chunks[2], (7, 10));
    }

    #[test]
    fn test_parallel_sum() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let sum = parallel_sum(&data);
        assert!((sum - 499500.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_max() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        assert!((parallel_max(&data) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_parallel_min() {
        let data = vec![3.0, 1.0, 4.0, 1.5, 2.0];
        assert!((parallel_min(&data) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_compute_mlups() {
        let mlups = compute_mlups(1_000_000, 100, 1.0);
        assert!((mlups - 100.0).abs() < 1e-6);
    }
}
