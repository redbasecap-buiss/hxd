# hxd ⚡🔬

A modern terminal hex editor written in pure Rust.

**xxd meets vim** — interactive editing, not just viewing.

## Features

- 🖥️ **TUI** — three-column layout: offset | hex | ASCII with ratatui
- ⌨️ **Vim keybindings** — hjkl, gg/G, Ctrl-D/U, /search, n/N, :commands
- ✏️ **Edit mode** — `i` to edit bytes, type hex digits, Esc to return
- 🔍 **Search** — hex pattern (`/FF AB`) or ASCII string search with wrap-around
- 📊 **Multiple views** — Tab to cycle: Hex → Binary → Octal → Decimal
- ↩️ **Undo/Redo** — `u` to undo, Ctrl-R to redo (full history)
- ✂️ **Visual selection** — `v` to select range, `y` to yank, `d` to delete
- 🔖 **Bookmarks** — `m` + key to mark, `'` + key to jump
- 🏗️ **Structure detection** — recognizes PNG, JPEG, ZIP, PDF, ELF, PE, Mach-O, SQLite, GZIP
- 💾 **Large file support** — memory-mapped I/O via mmap
- 🎨 **Color-coded bytes** — null (dim), printable (green), control (red), high (yellow)
- 📋 **CLI subcommands** — `dump` (colored xxd), `diff` (side-by-side), `patch` (scriptable)
- 💻 **Save** — `:w`, `:wq`, `:q!` — just like vim
- 📥 **Pipe support** — `cat file | hxd`

## Installation

```bash
cargo install hxd

# Or via Homebrew
brew tap redbasecap-buiss/tap
brew install hxd
```

## Usage

```bash
# Interactive editor
hxd myfile.bin

# Colored hex dump
hxd dump myfile.bin

# Side-by-side diff
hxd diff file1.bin file2.bin

# Patch bytes at offset
hxd patch myfile.bin 0x100 "DEADBEEF"

# From stdin
cat /bin/ls | hxd
```

## Keybindings

| Key | Action |
|-----|--------|
| `h/j/k/l` | Move left/down/up/right |
| `gg` / `G` | Go to start / end |
| `Ctrl-D/U` | Page down / up |
| `i` | Enter edit mode |
| `Esc` | Back to normal mode |
| `v` | Visual selection |
| `y` / `d` | Yank / delete selection |
| `/` | Search (hex or ASCII) |
| `n` / `N` | Next / previous match |
| `u` | Undo |
| `Ctrl-R` | Redo |
| `m` + key | Set bookmark |
| `'` + key | Jump to bookmark |
| `Tab` | Cycle view mode |
| `:w` | Save |
| `:wq` | Save & quit |
| `:q!` | Force quit |
| `q` | Quit |

## Comparison

| Feature | hxd | xxd | hexyl | hexedit |
|---------|-----|-----|-------|---------|
| Interactive editing | ✅ | ❌ | ❌ | ✅ |
| Vim keybindings | ✅ | ❌ | ❌ | ❌ |
| Undo/Redo | ✅ | ❌ | ❌ | ❌ |
| Visual selection | ✅ | ❌ | ❌ | ❌ |
| Multiple views | ✅ | ❌ | ❌ | ❌ |
| Hex diff | ✅ | ❌ | ❌ | ❌ |
| File magic detection | ✅ | ❌ | ❌ | ❌ |
| Bookmarks | ✅ | ❌ | ❌ | ❌ |
| mmap large files | ✅ | ❌ | ✅ | ✅ |
| Single binary | ✅ | ✅ | ✅ | ✅ |
| Pure Rust | ✅ | ❌ | ✅ | ❌ |

## License

MIT © 2026 Nicola Spieser
