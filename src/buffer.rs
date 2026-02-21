use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub struct Buffer {
    data: Vec<u8>,
    path: Option<PathBuf>,
    modified: bool,
    undo_stack: Vec<(usize, u8, u8)>,
    redo_stack: Vec<(usize, u8, u8)>,
    clipboard: Vec<u8>,
}

impl Buffer {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            path: None,
            modified: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            clipboard: Vec::new(),
        }
    }

    pub fn from_file(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self {
            data: mmap.to_vec(),
            path: Some(path.to_path_buf()),
            modified: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            clipboard: Vec::new(),
        })
    }

    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            data,
            path: None,
            modified: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            clipboard: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn is_modified(&self) -> bool {
        self.modified
    }
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn get(&self, i: usize) -> Option<u8> {
        self.data.get(i).copied()
    }

    pub fn set_byte(&mut self, i: usize, val: u8) {
        if i < self.data.len() {
            let old = self.data[i];
            if old != val {
                self.undo_stack.push((i, old, val));
                self.redo_stack.clear();
                self.data[i] = val;
                self.modified = true;
            }
        }
    }

    pub fn undo(&mut self) -> bool {
        if let Some((i, old, new_val)) = self.undo_stack.pop() {
            self.data[i] = old;
            self.redo_stack.push((i, old, new_val));
            self.modified = !self.undo_stack.is_empty();
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if let Some((i, old, new_val)) = self.redo_stack.pop() {
            self.data[i] = new_val;
            self.undo_stack.push((i, old, new_val));
            self.modified = true;
            true
        } else {
            false
        }
    }

    pub fn yank(&mut self, start: usize, end: usize) {
        if start < end && end <= self.data.len() {
            self.clipboard = self.data[start..end].to_vec();
        }
    }

    pub fn clipboard(&self) -> &[u8] {
        &self.clipboard
    }

    pub fn save(&mut self) -> io::Result<()> {
        if let Some(ref path) = self.path {
            let mut f = OpenOptions::new().write(true).truncate(true).open(path)?;
            f.write_all(&self.data)?;
            self.modified = false;
            Ok(())
        } else {
            Err(io::Error::other("No file path"))
        }
    }

    pub fn search_bytes(&self, pat: &[u8], from: usize) -> Option<usize> {
        if pat.is_empty() || from + pat.len() > self.data.len() {
            return None;
        }
        self.data[from..]
            .windows(pat.len())
            .position(|w| w == pat)
            .map(|p| p + from)
    }

    pub fn search_bytes_rev(&self, pat: &[u8], from: usize) -> Option<usize> {
        if pat.is_empty() || from == 0 {
            return None;
        }
        let end = from.min(self.data.len());
        self.data[..end].windows(pat.len()).rposition(|w| w == pat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new() {
        assert!(Buffer::new().is_empty());
    }
    #[test]
    fn test_from_bytes() {
        let b = Buffer::from_bytes(vec![1, 2, 3]);
        assert_eq!(b.len(), 3);
        assert_eq!(b.get(1), Some(2));
    }
    #[test]
    fn test_set_undo_redo() {
        let mut b = Buffer::from_bytes(vec![0; 4]);
        b.set_byte(1, 0xFF);
        assert_eq!(b.get(1), Some(0xFF));
        assert!(b.undo());
        assert_eq!(b.get(1), Some(0));
        assert!(b.redo());
        assert_eq!(b.get(1), Some(0xFF));
    }
    #[test]
    fn test_multi_undo() {
        let mut b = Buffer::from_bytes(vec![0; 3]);
        b.set_byte(0, 1);
        b.set_byte(1, 2);
        b.set_byte(2, 3);
        assert!(b.undo());
        assert!(b.undo());
        assert!(b.undo());
        assert!(!b.undo());
    }
    #[test]
    fn test_search() {
        let b = Buffer::from_bytes(vec![0, 0xFF, 0xAB, 0xFF, 0xAB]);
        assert_eq!(b.search_bytes(&[0xFF, 0xAB], 0), Some(1));
        assert_eq!(b.search_bytes(&[0xFF, 0xAB], 2), Some(3));
    }
    #[test]
    fn test_search_rev() {
        let b = Buffer::from_bytes(vec![0xAA, 0xBB, 0xAA, 0xBB]);
        assert_eq!(b.search_bytes_rev(&[0xAA, 0xBB], 4), Some(2));
    }
    #[test]
    fn test_yank() {
        let mut b = Buffer::from_bytes(vec![10, 20, 30]);
        b.yank(0, 2);
        assert_eq!(b.clipboard(), &[10, 20]);
    }
    #[test]
    fn test_file_io() {
        let d = std::env::temp_dir().join("hxd_buf_t");
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("t.bin");
        std::fs::write(&p, &[0xDE, 0xAD]).unwrap();
        let mut b = Buffer::from_file(&p).unwrap();
        assert_eq!(b.get(0), Some(0xDE));
        b.set_byte(0, 0x00);
        b.save().unwrap();
        assert_eq!(std::fs::read(&p).unwrap()[0], 0x00);
        std::fs::remove_dir_all(&d).ok();
    }
}
