/// Byte-level statistics for a buffer.
pub struct ByteStats {
    pub size: usize,
    pub null_count: usize,
    pub printable_count: usize,
    pub entropy: f64,
}

/// Format byte count as human-readable size (e.g. "1.5 KiB").
fn format_size(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes == 0 {
        return "0 B".into();
    }
    let mut size = bytes as f64;
    for &unit in UNITS {
        if size < 1024.0 || unit == "TiB" {
            return if size.fract() < 0.05 || unit == "B" {
                format!("{:.0} {}", size, unit)
            } else {
                format!("{:.1} {}", size, unit)
            };
        }
        size /= 1024.0;
    }
    format!("{} B", bytes)
}

/// Compute byte-level statistics for the given data slice.
pub fn compute(data: &[u8]) -> ByteStats {
    if data.is_empty() {
        return ByteStats {
            size: 0,
            null_count: 0,
            printable_count: 0,
            entropy: 0.0,
        };
    }

    let mut freq = [0u64; 256];
    let mut null_count = 0usize;
    let mut printable_count = 0usize;

    for &b in data {
        freq[b as usize] += 1;
        if b == 0 {
            null_count += 1;
        }
        if b.is_ascii_graphic() || b == b' ' {
            printable_count += 1;
        }
    }

    let len = data.len() as f64;
    let entropy = freq
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum::<f64>()
        + 0.0; // avoid -0.0

    ByteStats {
        size: data.len(),
        null_count,
        printable_count,
        entropy,
    }
}

impl ByteStats {
    /// Format as a human-readable summary string.
    pub fn summary(&self) -> String {
        if self.size == 0 {
            return "Empty buffer".into();
        }
        let pct_null = 100.0 * self.null_count as f64 / self.size as f64;
        let pct_print = 100.0 * self.printable_count as f64 / self.size as f64;
        format!(
            "{} ({}) | entropy {:.2}/8 | null {:.1}% | printable {:.1}%",
            format_size(self.size), self.size, self.entropy, pct_null, pct_print
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let s = compute(b"");
        assert_eq!(s.size, 0);
        assert_eq!(s.entropy, 0.0);
        assert_eq!(s.summary(), "Empty buffer");
    }

    #[test]
    fn test_uniform() {
        // All same byte → entropy 0
        let data = vec![0xAA; 100];
        let s = compute(&data);
        assert_eq!(s.size, 100);
        assert_eq!(s.null_count, 0);
        assert!((s.entropy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_values() {
        // Equal split of two values → entropy 1.0
        let mut data = vec![0u8; 100];
        for i in 0..50 {
            data[i] = 1;
        }
        let s = compute(&data);
        assert!((s.entropy - 1.0).abs() < 1e-10);
        assert_eq!(s.null_count, 50);
    }

    #[test]
    fn test_all_256() {
        // All 256 byte values equally → entropy 8.0
        let data: Vec<u8> = (0..=255).collect();
        let s = compute(&data);
        assert!((s.entropy - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_printable() {
        let s = compute(b"Hello World!");
        assert_eq!(s.printable_count, 12);
        assert_eq!(s.null_count, 0);
    }

    #[test]
    fn test_summary_format() {
        let s = compute(b"AAAA");
        let summary = s.summary();
        assert!(summary.contains("4 B"));
        assert!(summary.contains("(4)"));
        assert!(summary.contains("entropy 0.00/8"));
        assert!(summary.contains("printable 100.0%"));
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
    }

    #[test]
    fn test_format_size_kib() {
        assert_eq!(format_size(1024), "1 KiB");
        assert_eq!(format_size(1536), "1.5 KiB");
    }

    #[test]
    fn test_format_size_mib() {
        assert_eq!(format_size(1048576), "1 MiB");
        assert_eq!(format_size(2621440), "2.5 MiB");
    }
}
