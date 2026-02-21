pub struct MagicInfo {
    pub name: &'static str,
    pub description: &'static str,
}

static SIGS: &[(&[u8], MagicInfo)] = &[
    (
        b"\x89PNG\r\n\x1a\n",
        MagicInfo {
            name: "PNG",
            description: "PNG Image",
        },
    ),
    (
        b"\xFF\xD8\xFF",
        MagicInfo {
            name: "JPEG",
            description: "JPEG Image",
        },
    ),
    (
        b"PK\x03\x04",
        MagicInfo {
            name: "ZIP",
            description: "ZIP Archive",
        },
    ),
    (
        b"%PDF",
        MagicInfo {
            name: "PDF",
            description: "PDF Document",
        },
    ),
    (
        b"\x7FELF",
        MagicInfo {
            name: "ELF",
            description: "ELF Executable",
        },
    ),
    (
        b"MZ",
        MagicInfo {
            name: "PE",
            description: "PE Executable",
        },
    ),
    (
        b"\x1f\x8b",
        MagicInfo {
            name: "GZIP",
            description: "Gzip",
        },
    ),
    (
        b"\xcf\xfa\xed\xfe",
        MagicInfo {
            name: "Mach-O",
            description: "Mach-O 64-bit",
        },
    ),
    (
        b"SQLite format 3",
        MagicInfo {
            name: "SQLite",
            description: "SQLite DB",
        },
    ),
];

pub fn detect(data: &[u8]) -> Option<&'static MagicInfo> {
    for (sig, info) in SIGS {
        if data.len() >= sig.len() && &data[..sig.len()] == *sig {
            return Some(info);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_png() {
        assert_eq!(detect(b"\x89PNG\r\n\x1a\nxx").unwrap().name, "PNG");
    }
    #[test]
    fn test_elf() {
        assert_eq!(detect(b"\x7FELF\x02").unwrap().name, "ELF");
    }
    #[test]
    fn test_zip() {
        assert_eq!(detect(b"PK\x03\x04x").unwrap().name, "ZIP");
    }
    #[test]
    fn test_pdf() {
        assert_eq!(detect(b"%PDF-1.7").unwrap().name, "PDF");
    }
    #[test]
    fn test_none() {
        assert!(detect(b"random").is_none());
    }
    #[test]
    fn test_empty() {
        assert!(detect(b"").is_none());
    }
    #[test]
    fn test_macho() {
        assert_eq!(detect(b"\xcf\xfa\xed\xfex").unwrap().name, "Mach-O");
    }
}
