use std::io::{self, Write};

pub struct DiffResult {
    pub total: usize,
    pub diffs: usize,
}

pub fn compare(a: &[u8], b: &[u8]) -> DiffResult {
    let total = a.len().max(b.len());
    let diffs = (0..total).filter(|&i| a.get(i) != b.get(i)).count();
    DiffResult { total, diffs }
}

fn ascii_char(b: u8) -> char {
    if b.is_ascii_graphic() || b == b' ' {
        b as char
    } else {
        '.'
    }
}

pub fn print_diff(a: &[u8], b: &[u8], cols: usize) -> io::Result<()> {
    let mut o = io::stdout().lock();
    let max = a.len().max(b.len());
    for rs in (0..max).step_by(cols) {
        write!(o, "\x1b[36m{:08x}\x1b[0m  ", rs)?;
        // Left hex
        for j in 0..cols {
            let off = rs + j;
            if off < a.len() {
                let d = b.get(off) != Some(&a[off]);
                if d {
                    write!(o, "\x1b[31m{:02x}\x1b[0m ", a[off])?;
                } else {
                    write!(o, "{:02x} ", a[off])?;
                }
            } else {
                write!(o, "   ")?;
            }
        }
        // Left ASCII
        write!(o, " \x1b[2m|\x1b[0m")?;
        for j in 0..cols {
            let off = rs + j;
            if off < a.len() {
                let d = b.get(off) != Some(&a[off]);
                let c = ascii_char(a[off]);
                if d {
                    write!(o, "\x1b[31m{}\x1b[0m", c)?;
                } else {
                    write!(o, "{}", c)?;
                }
            } else {
                write!(o, " ")?;
            }
        }
        write!(o, "\x1b[2m|\x1b[0m \u{2502} ")?;
        // Right hex
        for j in 0..cols {
            let off = rs + j;
            if off < b.len() {
                let d = a.get(off) != Some(&b[off]);
                if d {
                    write!(o, "\x1b[31m{:02x}\x1b[0m ", b[off])?;
                } else {
                    write!(o, "{:02x} ", b[off])?;
                }
            } else {
                write!(o, "   ")?;
            }
        }
        // Right ASCII
        write!(o, " \x1b[2m|\x1b[0m")?;
        for j in 0..cols {
            let off = rs + j;
            if off < b.len() {
                let d = a.get(off) != Some(&b[off]);
                let c = ascii_char(b[off]);
                if d {
                    write!(o, "\x1b[31m{}\x1b[0m", c)?;
                } else {
                    write!(o, "{}", c)?;
                }
            } else {
                write!(o, " ")?;
            }
        }
        writeln!(o, "\x1b[2m|\x1b[0m")?;
    }
    let r = compare(a, b);
    writeln!(o, "\n{} bytes, {} differences", r.total, r.diffs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_same() {
        assert_eq!(compare(b"hi", b"hi").diffs, 0);
    }
    #[test]
    fn test_diff() {
        assert_eq!(compare(b"hi", b"ho").diffs, 1);
    }
    #[test]
    fn test_len() {
        assert!(compare(b"a", b"abc").diffs >= 2);
    }
    #[test]
    fn test_empty() {
        assert_eq!(compare(b"", b"").diffs, 0);
    }
    #[test]
    fn test_ascii_char() {
        assert_eq!(ascii_char(b'A'), 'A');
        assert_eq!(ascii_char(b' '), ' ');
        assert_eq!(ascii_char(0x00), '.');
        assert_eq!(ascii_char(0xFF), '.');
    }
}
