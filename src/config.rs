//! Configurable colors via ~/.config/hxd/config.toml

use ratatui::style::Color;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Top-level config file structure
#[derive(Deserialize, Debug, Clone, Default)]
pub struct ConfigFile {
    #[serde(default)]
    pub colors: ColorConfig,
}

/// Color configuration section
#[derive(Deserialize, Debug, Clone)]
pub struct ColorConfig {
    /// Null bytes (0x00)
    #[serde(default = "default_null")]
    pub null: String,
    /// Printable ASCII (0x20..0x7E)
    #[serde(default = "default_printable")]
    pub printable: String,
    /// Control characters (0x01..0x1F, 0x7F)
    #[serde(default = "default_control")]
    pub control: String,
    /// High bytes (0x80..0xFF)
    #[serde(default = "default_high")]
    pub high: String,
    /// Offset column
    #[serde(default = "default_offset")]
    pub offset: String,
    /// Cursor foreground
    #[serde(default = "default_cursor_fg")]
    pub cursor_fg: String,
    /// Cursor background
    #[serde(default = "default_cursor_bg")]
    pub cursor_bg: String,
    /// Header/info bar background
    #[serde(default = "default_bar_bg")]
    pub bar_bg: String,
    /// Header/info bar foreground
    #[serde(default = "default_bar_fg")]
    pub bar_fg: String,
    /// Visual selection background
    #[serde(default = "default_visual_bg")]
    pub visual_bg: String,
}

fn default_null() -> String {
    "darkgray".into()
}
fn default_printable() -> String {
    "green".into()
}
fn default_control() -> String {
    "red".into()
}
fn default_high() -> String {
    "yellow".into()
}
fn default_offset() -> String {
    "cyan".into()
}
fn default_cursor_fg() -> String {
    "black".into()
}
fn default_cursor_bg() -> String {
    "white".into()
}
fn default_bar_bg() -> String {
    "darkgray".into()
}
fn default_bar_fg() -> String {
    "white".into()
}
fn default_visual_bg() -> String {
    "blue".into()
}

impl Default for ColorConfig {
    fn default() -> Self {
        Self {
            null: default_null(),
            printable: default_printable(),
            control: default_control(),
            high: default_high(),
            offset: default_offset(),
            cursor_fg: default_cursor_fg(),
            cursor_bg: default_cursor_bg(),
            bar_bg: default_bar_bg(),
            bar_fg: default_bar_fg(),
            visual_bg: default_visual_bg(),
        }
    }
}

/// Resolved colors ready for use in the UI
#[derive(Debug, Clone)]
pub struct ResolvedColors {
    pub null: Color,
    pub printable: Color,
    pub control: Color,
    pub high: Color,
    pub offset: Color,
    pub cursor_fg: Color,
    pub cursor_bg: Color,
    pub bar_bg: Color,
    pub bar_fg: Color,
    pub visual_bg: Color,
}

impl Default for ResolvedColors {
    fn default() -> Self {
        Self::from_config(&ColorConfig::default())
    }
}

impl ResolvedColors {
    pub fn from_config(cfg: &ColorConfig) -> Self {
        Self {
            null: parse_color(&cfg.null),
            printable: parse_color(&cfg.printable),
            control: parse_color(&cfg.control),
            high: parse_color(&cfg.high),
            offset: parse_color(&cfg.offset),
            cursor_fg: parse_color(&cfg.cursor_fg),
            cursor_bg: parse_color(&cfg.cursor_bg),
            bar_bg: parse_color(&cfg.bar_bg),
            bar_fg: parse_color(&cfg.bar_fg),
            visual_bg: parse_color(&cfg.visual_bg),
        }
    }

    /// Get the color for a byte value
    pub fn byte_color(&self, b: u8) -> Color {
        match b {
            0 => self.null,
            0x20..=0x7E => self.printable,
            0x01..=0x1F | 0x7F => self.control,
            _ => self.high,
        }
    }
}

/// Parse a color string into a ratatui Color
fn parse_color(s: &str) -> Color {
    match s.to_lowercase().as_str() {
        "black" => Color::Black,
        "red" => Color::Red,
        "green" => Color::Green,
        "yellow" => Color::Yellow,
        "blue" => Color::Blue,
        "magenta" => Color::Magenta,
        "cyan" => Color::Cyan,
        "gray" | "grey" => Color::Gray,
        "darkgray" | "darkgrey" | "dark_gray" | "dark_grey" => Color::DarkGray,
        "lightred" | "light_red" => Color::LightRed,
        "lightgreen" | "light_green" => Color::LightGreen,
        "lightyellow" | "light_yellow" => Color::LightYellow,
        "lightblue" | "light_blue" => Color::LightBlue,
        "lightmagenta" | "light_magenta" => Color::LightMagenta,
        "lightcyan" | "light_cyan" => Color::LightCyan,
        "white" => Color::White,
        "reset" => Color::Reset,
        hex if hex.starts_with('#') && hex.len() == 7 => {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[1..3], 16),
                u8::from_str_radix(&hex[3..5], 16),
                u8::from_str_radix(&hex[5..7], 16),
            ) {
                Color::Rgb(r, g, b)
            } else {
                Color::White
            }
        }
        _ => Color::White,
    }
}

/// Get the config file path
pub fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hxd")
        .join("config.toml")
}

/// Load config from file, returning defaults if file doesn't exist
pub fn load_config() -> ConfigFile {
    load_config_from(&config_path())
}

/// Load config from a specific path
pub fn load_config_from(path: &std::path::Path) -> ConfigFile {
    match std::fs::read_to_string(path) {
        Ok(content) => toml::from_str(&content).unwrap_or_default(),
        Err(_) => ConfigFile::default(),
    }
}

/// Thread-safe shared config handle
pub type SharedColors = Arc<RwLock<ResolvedColors>>;

pub fn new_shared_colors() -> SharedColors {
    let cfg = load_config();
    Arc::new(RwLock::new(ResolvedColors::from_config(&cfg.colors)))
}

/// Generate an example config.toml content
pub fn example_config() -> &'static str {
    r##"# hxd color configuration
# Colors: black, red, green, yellow, blue, magenta, cyan, gray,
#          darkgray, lightred, lightgreen, lightyellow, lightblue,
#          lightmagenta, lightcyan, white, reset
# Hex colors: "#RRGGBB"

[colors]
null = "darkgray"
printable = "green"
control = "red"
high = "yellow"
offset = "cyan"
cursor_fg = "black"
cursor_bg = "white"
bar_bg = "darkgray"
bar_fg = "white"
visual_bg = "blue"
"##
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_named_colors() {
        assert_eq!(parse_color("red"), Color::Red);
        assert_eq!(parse_color("Green"), Color::Green);
        assert_eq!(parse_color("DARKGRAY"), Color::DarkGray);
        assert_eq!(parse_color("dark_gray"), Color::DarkGray);
    }

    #[test]
    fn test_parse_hex_color() {
        assert_eq!(parse_color("#FF0000"), Color::Rgb(255, 0, 0));
        assert_eq!(parse_color("#00ff00"), Color::Rgb(0, 255, 0));
    }

    #[test]
    fn test_parse_invalid_color() {
        assert_eq!(parse_color("notacolor"), Color::White);
        assert_eq!(parse_color("#ZZZZZZ"), Color::White);
    }

    #[test]
    fn test_default_colors() {
        let c = ResolvedColors::default();
        assert_eq!(c.null, Color::DarkGray);
        assert_eq!(c.printable, Color::Green);
    }

    #[test]
    fn test_byte_color() {
        let c = ResolvedColors::default();
        assert_eq!(c.byte_color(0), Color::DarkGray);
        assert_eq!(c.byte_color(b'A'), Color::Green);
        assert_eq!(c.byte_color(0x01), Color::Red);
        assert_eq!(c.byte_color(0x80), Color::Yellow);
    }

    #[test]
    fn test_load_toml() {
        let toml_str = r##"
[colors]
null = "#111111"
printable = "lightgreen"
"##;
        let cfg: ConfigFile = toml::from_str(toml_str).unwrap();
        let resolved = ResolvedColors::from_config(&cfg.colors);
        assert_eq!(resolved.null, Color::Rgb(0x11, 0x11, 0x11));
        assert_eq!(resolved.printable, Color::LightGreen);
        // defaults for unset
        assert_eq!(resolved.control, Color::Red);
    }

    #[test]
    fn test_example_config_parses() {
        let cfg: ConfigFile = toml::from_str(example_config()).unwrap();
        let _ = ResolvedColors::from_config(&cfg.colors);
    }
}
