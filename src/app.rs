use crate::buffer::Buffer;
use crate::stats;
use crate::magic;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Normal,
    Edit,
    Command,
    Search,
    Visual,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    Hex,
    Binary,
    Octal,
    Decimal,
}

impl ViewMode {
    pub fn next(self) -> Self {
        match self {
            Self::Hex => Self::Binary,
            Self::Binary => Self::Octal,
            Self::Octal => Self::Decimal,
            Self::Decimal => Self::Hex,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Hex => "HEX",
            Self::Binary => "BIN",
            Self::Octal => "OCT",
            Self::Decimal => "DEC",
        }
    }
    pub fn format_byte(self, b: u8) -> String {
        match self {
            Self::Hex => format!("{:02x}", b),
            Self::Binary => format!("{:08b}", b),
            Self::Octal => format!("{:03o}", b),
            Self::Decimal => format!("{:3}", b),
        }
    }
    pub fn cell_width(self) -> usize {
        match self {
            Self::Hex => 3,
            Self::Binary => 9,
            Self::Octal => 4,
            Self::Decimal => 4,
        }
    }
}

pub struct App {
    pub buffer: Buffer,
    pub cursor: usize,
    pub scroll_offset: usize,
    pub cols: usize,
    pub mode: Mode,
    pub view_mode: ViewMode,
    pub command_input: String,
    pub search_input: String,
    pub status_message: String,
    pub visual_start: Option<usize>,
    pub bookmarks: HashMap<char, usize>,
    pub pending_bookmark: bool,
    pub pending_goto_bookmark: bool,
    pub edit_nibble_high: bool,
    pub should_quit: bool,
    pub magic_info: Option<&'static str>,
    pub rows_visible: usize,
}

impl App {
    pub fn new(buffer: Buffer, cols: usize) -> Self {
        let mi = magic::detect(buffer.data()).map(|m| m.description);
        Self {
            buffer,
            cursor: 0,
            scroll_offset: 0,
            cols,
            mode: Mode::Normal,
            view_mode: ViewMode::Hex,
            command_input: String::new(),
            search_input: String::new(),
            status_message: String::new(),
            visual_start: None,
            bookmarks: HashMap::new(),
            pending_bookmark: false,
            pending_goto_bookmark: false,
            edit_nibble_high: true,
            should_quit: false,
            magic_info: mi,
            rows_visible: 24,
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.ensure_visible();
        }
    }
    pub fn move_right(&mut self) {
        if self.cursor + 1 < self.buffer.len() {
            self.cursor += 1;
            self.ensure_visible();
        }
    }
    pub fn move_up(&mut self) {
        if self.cursor >= self.cols {
            self.cursor -= self.cols;
            self.ensure_visible();
        }
    }
    pub fn move_down(&mut self) {
        if self.cursor + self.cols < self.buffer.len() {
            self.cursor += self.cols;
            self.ensure_visible();
        }
    }
    pub fn goto_start(&mut self) {
        self.cursor = 0;
        self.scroll_offset = 0;
    }
    pub fn goto_end(&mut self) {
        if !self.buffer.is_empty() {
            self.cursor = self.buffer.len() - 1;
            self.ensure_visible();
        }
    }
    pub fn page_down(&mut self) {
        self.cursor =
            (self.cursor + self.rows_visible * self.cols).min(self.buffer.len().saturating_sub(1));
        self.ensure_visible();
    }
    pub fn page_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(self.rows_visible * self.cols);
        self.ensure_visible();
    }

    fn ensure_visible(&mut self) {
        let row = self.cursor / self.cols;
        if row < self.scroll_offset {
            self.scroll_offset = row;
        }
        if row >= self.scroll_offset + self.rows_visible {
            self.scroll_offset = row - self.rows_visible + 1;
        }
    }

    pub fn enter_edit_mode(&mut self) {
        self.mode = Mode::Edit;
        self.edit_nibble_high = true;
        self.status_message = "-- EDIT --".into();
    }
    pub fn exit_edit_mode(&mut self) {
        self.mode = Mode::Normal;
        self.status_message.clear();
    }

    pub fn edit_input(&mut self, c: char) {
        if let Some(n) = c.to_digit(16) {
            let n = n as u8;
            if let Some(cur) = self.buffer.get(self.cursor) {
                let new = if self.edit_nibble_high {
                    (n << 4) | (cur & 0x0F)
                } else {
                    (cur & 0xF0) | n
                };
                self.buffer.set_byte(self.cursor, new);
                if !self.edit_nibble_high {
                    self.move_right();
                }
                self.edit_nibble_high = !self.edit_nibble_high;
            }
        }
    }

    pub fn enter_visual_mode(&mut self) {
        self.mode = Mode::Visual;
        self.visual_start = Some(self.cursor);
        self.status_message = "-- VISUAL --".into();
    }

    pub fn yank_visual(&mut self) {
        if let Some(start) = self.visual_start {
            let (s, e) = if start <= self.cursor {
                (start, self.cursor + 1)
            } else {
                (self.cursor, start + 1)
            };
            self.buffer.yank(s, e);
            self.status_message = format!("{} bytes yanked", e - s);
        }
        self.visual_start = None;
        self.mode = Mode::Normal;
    }


    pub fn paste_at_cursor(&mut self) {
        if self.buffer.clipboard().is_empty() {
            self.status_message = "Nothing to paste".into();
            return;
        }
        let clip = self.buffer.clipboard().to_vec();
        let mut count = 0;
        for (i, &b) in clip.iter().enumerate() {
            let pos = self.cursor + i;
            if pos < self.buffer.len() {
                self.buffer.set_byte(pos, b);
                count += 1;
            } else {
                break;
            }
        }
        self.status_message = format!("{} bytes pasted", count);
    }

    pub fn fill_range(&mut self, offset: usize, count: usize, byte: u8) {
        let mut filled = 0;
        for i in 0..count {
            let pos = offset + i;
            if pos < self.buffer.len() {
                self.buffer.set_byte(pos, byte);
                filled += 1;
            } else {
                break;
            }
        }
        self.status_message = format!("Filled {} bytes with 0x{:02X}", filled, byte);
    }
    pub fn visual_range(&self) -> Option<(usize, usize)> {
        self.visual_start.map(|s| {
            if s <= self.cursor {
                (s, self.cursor)
            } else {
                (self.cursor, s)
            }
        })
    }

    pub fn search_hex(&mut self) {
        let input = self.search_input.clone();
        if let Ok(bytes) = crate::patch::parse_hex_string(&input) {
            self.do_search(&bytes, true);
        } else {
            self.do_search(input.as_bytes(), true);
        }
    }

    fn do_search(&mut self, pat: &[u8], fwd: bool) {
        if pat.is_empty() {
            return;
        }
        let r = if fwd {
            self.buffer
                .search_bytes(pat, self.cursor + 1)
                .or_else(|| self.buffer.search_bytes(pat, 0))
        } else {
            self.buffer
                .search_bytes_rev(pat, self.cursor)
                .or_else(|| self.buffer.search_bytes_rev(pat, self.buffer.len()))
        };
        match r {
            Some(p) => {
                self.cursor = p;
                self.ensure_visible();
                self.status_message = format!("Found at 0x{:x}", p);
            }
            None => {
                self.status_message = "Not found".into();
            }
        }
    }

    pub fn search_next(&mut self) {
        let i = self.search_input.clone();
        if let Ok(b) = crate::patch::parse_hex_string(&i) {
            self.do_search(&b, true);
        } else {
            self.do_search(i.as_bytes(), true);
        }
    }

    pub fn search_prev(&mut self) {
        let i = self.search_input.clone();
        if let Ok(b) = crate::patch::parse_hex_string(&i) {
            self.do_search(&b, false);
        } else {
            self.do_search(i.as_bytes(), false);
        }
    }

    pub fn execute_command(&mut self) {
        let cmd = self.command_input.trim().to_string();
        self.command_input.clear();
        self.mode = Mode::Normal;
        match cmd.as_str() {
            "q" => {
                if self.buffer.is_modified() {
                    self.status_message = "Unsaved! :q! to force".into();
                } else {
                    self.should_quit = true;
                }
            }
            "q!" => self.should_quit = true,
            "inspect" => {
                self.status_message = self.inspect_at_cursor();
            }
            "stats" => {
                self.status_message = stats::compute(self.buffer.data()).summary();
            }
            "help" => {
                self.status_message = "q w wq :off :/pat n N i v m\x27 p Tab :fill :stats :inspect".into();
            }
            "w" => match self.buffer.save() {
                Ok(()) => self.status_message = "Written".into(),
                Err(e) => self.status_message = format!("{}", e),
            },
            "wq" => match self.buffer.save() {
                Ok(()) => self.should_quit = true,
                Err(e) => self.status_message = format!("{}", e),
            },
            _ => {
                if let Ok(off) = crate::patch::parse_offset(&cmd) {
                    if (off as usize) < self.buffer.len() {
                        self.cursor = off as usize;
                        self.ensure_visible();
                    } else {
                        self.status_message = "Beyond EOF".into();
                    }
                } else if cmd.starts_with("fill ") {
                    let parts: Vec<&str> = cmd.strip_prefix("fill ").unwrap_or("").split_whitespace().collect();
                    if parts.len() == 3 {
                        if let (Ok(off), Ok(cnt), Ok(byte)) = (
                            crate::patch::parse_offset(parts[0]),
                            parts[1].parse::<usize>(),
                            u8::from_str_radix(parts[2].trim_start_matches("0x"), 16),
                        ) {
                            self.fill_range(off as usize, cnt, byte);
                        } else {
                            self.status_message = "Usage: fill <offset> <count> <hex_byte>".into();
                        }
                    } else {
                        self.status_message = "Usage: fill <offset> <count> <hex_byte>".into();
                    }
                } else {
                    self.status_message = format!("Unknown: {}", cmd);
                }
            }
        }
    }

    pub fn set_bookmark(&mut self, k: char) {
        self.bookmarks.insert(k, self.cursor);
        self.status_message = format!("Bookmark '{}'", k);
        self.pending_bookmark = false;
    }
    pub fn goto_bookmark(&mut self, k: char) {
        if let Some(&p) = self.bookmarks.get(&k) {
            self.cursor = p;
            self.ensure_visible();
        } else {
            self.status_message = format!("No bookmark '{}'", k);
        }
        self.pending_goto_bookmark = false;
    }


    /// Show multi-byte integer values at cursor (LE and BE).
    pub fn inspect_at_cursor(&self) -> String {
        let data = self.buffer.data();
        let pos = self.cursor;
        let remaining = data.len().saturating_sub(pos);
        if remaining == 0 {
            return "No data at cursor".into();
        }
        let mut parts: Vec<String> = Vec::new();
        if remaining >= 2 {
            let le = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let be = u16::from_be_bytes([data[pos], data[pos + 1]]);
            parts.push(format!("u16 LE:{} BE:{}", le, be));
        }
        if remaining >= 4 {
            let le = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
            let be = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
            parts.push(format!("u32 LE:{} BE:{}", le, be));
        }
        if remaining >= 8 {
            let mut le_bytes = [0u8; 8];
            let mut be_bytes = [0u8; 8];
            le_bytes.copy_from_slice(&data[pos..pos+8]);
            be_bytes.copy_from_slice(&data[pos..pos+8]);
            let le = u64::from_le_bytes(le_bytes);
            let be = u64::from_be_bytes(be_bytes);
            parts.push(format!("u64 LE:{} BE:{}", le, be));
        }
        if parts.is_empty() {
            format!("u8: {}", data[pos])
        } else {
            parts.join(" | ")
        }
    }

    pub fn goto_row_start(&mut self) {
        self.cursor = (self.cursor / self.cols) * self.cols;
        self.ensure_visible();
    }

    pub fn goto_row_end(&mut self) {
        let row_end = ((self.cursor / self.cols) + 1) * self.cols - 1;
        self.cursor = row_end.min(self.buffer.len().saturating_sub(1));
        self.ensure_visible();
    }

    pub fn current_byte_info(&self) -> String {
        if let Some(b) = self.buffer.get(self.cursor) {
            let ch = if b.is_ascii_graphic() || b == b' ' {
                format!("'{}'", b as char)
            } else {
                "N/A".into()
            };
            format!("0x{:02X} {} 0o{:03o} 0b{:08b} {}", b, b, b, b, ch)
        } else {
            "---".into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn app() -> App {
        App::new(Buffer::from_bytes((0..=255).collect()), 16)
    }
    #[test]
    fn test_move() {
        let mut a = app();
        a.move_right();
        assert_eq!(a.cursor, 1);
        a.move_down();
        assert_eq!(a.cursor, 17);
        a.move_up();
        assert_eq!(a.cursor, 1);
        a.move_left();
        assert_eq!(a.cursor, 0);
    }
    #[test]
    fn test_goto() {
        let mut a = app();
        a.goto_end();
        assert_eq!(a.cursor, 255);
        a.goto_start();
        assert_eq!(a.cursor, 0);
    }
    #[test]
    fn test_edit() {
        let mut a = app();
        a.enter_edit_mode();
        a.edit_input('F');
        a.edit_input('A');
        assert_eq!(a.buffer.get(0), Some(0xFA));
    }
    #[test]
    fn test_visual() {
        let mut a = app();
        a.enter_visual_mode();
        a.move_right();
        a.move_right();
        a.yank_visual();
        assert_eq!(a.buffer.clipboard(), &[0, 1, 2]);
    }
    #[test]
    fn test_bookmark() {
        let mut a = app();
        a.cursor = 100;
        a.set_bookmark('x');
        a.cursor = 0;
        a.goto_bookmark('x');
        assert_eq!(a.cursor, 100);
    }
    #[test]
    fn test_view() {
        assert_eq!(ViewMode::Hex.next(), ViewMode::Binary);
        assert_eq!(ViewMode::Decimal.next(), ViewMode::Hex);
    }
    #[test]
    fn test_fmt() {
        assert_eq!(ViewMode::Hex.format_byte(0xFF), "ff");
        assert_eq!(ViewMode::Binary.format_byte(0x0F), "00001111");
    }
    #[test]
    fn test_goto_cmd() {
        let mut a = app();
        a.command_input = "0x80".into();
        a.execute_command();
        assert_eq!(a.cursor, 0x80);
    }
    #[test]
    fn test_quit_mod() {
        let mut a = app();
        a.buffer.set_byte(0, 0xFF);
        a.command_input = "q".into();
        a.execute_command();
        assert!(!a.should_quit);
    }
    #[test]
    fn test_fquit() {
        let mut a = app();
        a.buffer.set_byte(0, 0xFF);
        a.command_input = "q!".into();
        a.execute_command();
        assert!(a.should_quit);
    }
    #[test]
    fn test_page() {
        let mut a = app();
        a.rows_visible = 4;
        a.page_down();
        assert_eq!(a.cursor, 64);
        a.page_up();
        assert_eq!(a.cursor, 0);
    }

    #[test]
    fn test_paste() {
        let mut a = app();
        a.enter_visual_mode();
        a.move_right();
        a.move_right();
        a.yank_visual();
        a.cursor = 10;
        a.paste_at_cursor();
        assert_eq!(a.buffer.get(10), Some(0));
        assert_eq!(a.buffer.get(11), Some(1));
        assert_eq!(a.buffer.get(12), Some(2));
    }
    #[test]
    fn test_fill() {
        let mut a = app();
        a.fill_range(0, 4, 0xFF);
        assert_eq!(a.buffer.get(0), Some(0xFF));
        assert_eq!(a.buffer.get(3), Some(0xFF));
        assert_eq!(a.buffer.get(4), Some(4));
    }

    #[test]
    fn test_inspect_at_cursor() {
        let mut a = app();
        a.cursor = 0;
        let msg = a.inspect_at_cursor();
        // Data is 0,1,2,...255 so u16 LE at 0 = 0x0100 = 256
        assert!(msg.contains("u16 LE:256 BE:1"));
        assert!(msg.contains("u32"));
        assert!(msg.contains("u64"));
    }

    #[test]
    fn test_inspect_near_end() {
        let a = App::new(Buffer::from_bytes(vec![0xAA, 0xBB]), 16);
        let mut a2 = a;
        a2.cursor = 0;
        let msg = a2.inspect_at_cursor();
        assert!(msg.contains("u16"));
        assert!(!msg.contains("u32")); // only 2 bytes
    }

    #[test]
    fn test_inspect_command() {
        let mut a = app();
        a.command_input = "inspect".into();
        a.execute_command();
        assert!(a.status_message.contains("u16"));
    }

    #[test]
    fn test_goto_row_start() {
        let mut a = app();
        a.cursor = 5;
        a.goto_row_start();
        assert_eq!(a.cursor, 0);
        a.cursor = 20;
        a.goto_row_start();
        assert_eq!(a.cursor, 16);
    }

    #[test]
    fn test_goto_row_end() {
        let mut a = app();
        a.cursor = 5;
        a.goto_row_end();
        assert_eq!(a.cursor, 15);
        a.cursor = 250;
        a.goto_row_end();
        assert_eq!(a.cursor, 255);
    }
}
