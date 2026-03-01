use std::io::{self, Write};

pub fn hex_dump(data: &[u8], cols: usize) -> String {
    let mut out = String::new();
    for (i, chunk) in data.chunks(cols).enumerate() {
        let off = i * cols;
        out.push_str(&format!("{:08x}  ", off));
        for (j, &b) in chunk.iter().enumerate() {
            if j == cols / 2 {
                out.push(' ');
            }
            out.push_str(&format!("{:02x} ", b));
        }
        for j in 0..(cols - chunk.len()) {
            if chunk.len() + j == cols / 2 {
                out.push(' ');
            }
            out.push_str("   ");
        }
        out.push_str(" |");
        for &b in chunk {
            out.push(if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '.'
            });
        }
        out.push_str("|\n");
    }
    if !data.is_empty() {
        out.push_str(&format!("{:08x}\n", data.len()));
    }
    out
}

fn bcol(b: u8) -> &'static str {
    match b {
        0 => "\x1b[2m",
        0x20..=0x7E => "\x1b[32m",
        0x01..=0x1F | 0x7F => "\x1b[31m",
        _ => "\x1b[33m",
    }
}

pub fn print_colored_dump(data: &[u8], cols: usize) -> io::Result<()> {
    let mut o = io::stdout().lock();
    let r = "\x1b[0m";
    for (i, chunk) in data.chunks(cols).enumerate() {
        write!(o, "\x1b[36m{:08x}\x1b[0m  ", i * cols)?;
        for (j, &b) in chunk.iter().enumerate() {
            if j == cols / 2 {
                write!(o, " ")?;
            }
            write!(o, "{}{:02x}{} ", bcol(b), b, r)?;
        }
        for j in 0..(cols - chunk.len()) {
            if chunk.len() + j == cols / 2 {
                write!(o, " ")?;
            }
            write!(o, "   ")?;
        }
        write!(o, " \x1b[2m|\x1b[0m")?;
        for &b in chunk {
            let c = if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '.'
            };
            write!(o, "{}{}{}", bcol(b), c, r)?;
        }
        writeln!(o, "\x1b[2m|\x1b[0m")?;
    }
    if !data.is_empty() {
        writeln!(o, "\x1b[36m{:08x}\x1b[0m", data.len())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_basic() {
        let o = hex_dump(b"Hi", 16);
        assert!(o.contains("48 69"));
    }
    #[test]
    fn test_empty() {
        assert_eq!(hex_dump(b"", 16), "");
    }
    #[test]
    fn test_dots() {
        let o = hex_dump(&[0, 1], 16);
        assert!(o.contains(".."));
    }
}
