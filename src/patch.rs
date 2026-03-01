use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub fn parse_hex_string(s: &str) -> Result<Vec<u8>, String> {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if !clean.len().is_multiple_of(2) {
        return Err("Odd number of hex digits".into());
    }
    (0..clean.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

pub fn parse_offset(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).map_err(|e| e.to_string())
    } else {
        s.parse::<u64>().map_err(|e| e.to_string())
    }
}

pub fn patch_file(path: &Path, offset: u64, bytes: &[u8]) -> io::Result<()> {
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    let len = f.seek(SeekFrom::End(0))?;
    if offset + bytes.len() as u64 > len {
        return Err(io::Error::other("Beyond EOF"));
    }
    f.seek(SeekFrom::Start(offset))?;
    let mut old = vec![0u8; bytes.len()];
    f.read_exact(&mut old)?;
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hex() {
        assert_eq!(parse_hex_string("DEAD").unwrap(), vec![0xDE, 0xAD]);
    }
    #[test]
    fn test_hex_sp() {
        assert_eq!(parse_hex_string("DE AD").unwrap(), vec![0xDE, 0xAD]);
    }
    #[test]
    fn test_hex_odd() {
        assert!(parse_hex_string("ABC").is_err());
    }
    #[test]
    fn test_off_d() {
        assert_eq!(parse_offset("1024").unwrap(), 1024);
    }
    #[test]
    fn test_off_h() {
        assert_eq!(parse_offset("0xFF").unwrap(), 255);
    }
    #[test]
    fn test_off_bad() {
        assert!(parse_offset("xyz").is_err());
    }
    #[test]
    fn test_patch() {
        let d = std::env::temp_dir().join("hxd_patch");
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("t.bin");
        std::fs::write(&p, &[0u8; 8]).unwrap();
        patch_file(&p, 2, &[0xAB, 0xCD]).unwrap();
        let r = std::fs::read(&p).unwrap();
        assert_eq!(r[2], 0xAB);
        std::fs::remove_dir_all(&d).ok();
    }
}
