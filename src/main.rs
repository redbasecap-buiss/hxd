#![allow(dead_code)]
mod app;
mod buffer;
mod config;
mod diff;
mod dump;
mod magic;
mod patch;
mod ui;
mod stats;
mod watcher;

use app::{App, Mode};
use buffer::Buffer;
use clap::{Parser, Subcommand};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "hxd", version, about = "Modern terminal hex editor")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    file: Option<PathBuf>,
    #[arg(short, long, default_value = "16")]
    cols: usize,
}

#[derive(Subcommand)]
enum Commands {
    Dump {
        file: PathBuf,
        #[arg(short, long, default_value = "16")]
        cols: usize,
    },
    Diff {
        file1: PathBuf,
        file2: PathBuf,
        #[arg(short, long, default_value = "8")]
        cols: usize,
    },
    Patch {
        file: PathBuf,
        offset: String,
        hex_bytes: String,
    },
    /// Generate example config at ~/.config/hxd/config.toml
    InitConfig,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Dump { file, cols }) => {
            dump::print_colored_dump(&std::fs::read(&file)?, cols)?;
        }
        Some(Commands::Diff { file1, file2, cols }) => {
            diff::print_diff(&std::fs::read(&file1)?, &std::fs::read(&file2)?, cols)?;
        }
        Some(Commands::Patch {
            file,
            offset,
            hex_bytes,
        }) => {
            let off = patch::parse_offset(&offset).map_err(|e| anyhow::anyhow!(e))?;
            let bytes = patch::parse_hex_string(&hex_bytes).map_err(|e| anyhow::anyhow!(e))?;
            patch::patch_file(&file, off, &bytes)?;
            println!("Patched {} bytes at 0x{:x}", bytes.len(), off);
        }
        Some(Commands::InitConfig) => {
            let path = config::config_path();
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, config::example_config())?;
            println!("Created config at {}", path.display());
        }
        None => {
            let buf = if let Some(ref p) = cli.file {
                Buffer::from_file(p)?
            } else {
                let mut d = Vec::new();
                io::Read::read_to_end(&mut io::stdin(), &mut d)?;
                Buffer::from_bytes(d)
            };
            run_tui(buf, cli.cols)?;
        }
    }
    Ok(())
}

fn run_tui(buffer: Buffer, cols: usize) -> io::Result<()> {
    terminal::enable_raw_mode()?;
    crossterm::execute!(io::stdout(), EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut app = App::new(buffer, cols);

    // Load colors and start watcher
    let shared_colors = config::new_shared_colors();
    let _watcher_guard = watcher::watch_config(shared_colors.clone()).ok();

    loop {
        app.rows_visible = (term.size()?.height as usize).saturating_sub(4);

        // Check for config reload
        if let Some((_, ref rx)) = _watcher_guard {
            while rx.try_recv().is_ok() {
                app.status_message = "Config reloaded".into();
            }
        }

        let colors = shared_colors.read().unwrap().clone();
        term.draw(|f| ui::draw(f, &app, &colors))?;

        if app.should_quit {
            break;
        }
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                match app.mode {
                    Mode::Normal => handle_normal(&mut app, key.code, key.modifiers),
                    Mode::Edit => handle_edit(&mut app, key.code),
                    Mode::Command => handle_cmd(&mut app, key.code),
                    Mode::Search => handle_search(&mut app, key.code),
                    Mode::Visual => handle_visual(&mut app, key.code),
                }
            }
        }
    }
    terminal::disable_raw_mode()?;
    crossterm::execute!(term.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn handle_normal(app: &mut App, key: KeyCode, mods: KeyModifiers) {
    if app.pending_bookmark {
        if let KeyCode::Char(c) = key {
            app.set_bookmark(c);
        } else {
            app.pending_bookmark = false;
        }
        return;
    }
    if app.pending_goto_bookmark {
        if let KeyCode::Char(c) = key {
            app.goto_bookmark(c);
        } else {
            app.pending_goto_bookmark = false;
        }
        return;
    }
    match key {
        KeyCode::Char('q') => {
            if app.buffer.is_modified() {
                app.status_message = "Unsaved! :q!".into();
            } else {
                app.should_quit = true;
            }
        }
        KeyCode::Char('h') | KeyCode::Left => app.move_left(),
        KeyCode::Char('l') | KeyCode::Right => app.move_right(),
        KeyCode::Char('k') | KeyCode::Up => app.move_up(),
        KeyCode::Char('j') | KeyCode::Down => app.move_down(),
        KeyCode::Char('g') => app.goto_start(),
        KeyCode::Char('G') => app.goto_end(),
        KeyCode::Char('0') => app.goto_row_start(),
        KeyCode::Char('$') => app.goto_row_end(),
        KeyCode::Char('d') if mods.contains(KeyModifiers::CONTROL) => app.page_down(),
        KeyCode::Char('u') if mods.contains(KeyModifiers::CONTROL) => app.page_up(),
        KeyCode::Char('i') => app.enter_edit_mode(),
        KeyCode::Char('v') => app.enter_visual_mode(),
        KeyCode::Char(':') => {
            app.mode = Mode::Command;
            app.command_input.clear();
        }
        KeyCode::Char('/') => {
            app.mode = Mode::Search;
            app.search_input.clear();
        }
        KeyCode::Char('n') => app.search_next(),
        KeyCode::Char('N') => app.search_prev(),
        KeyCode::Char('u') => {
            if app.buffer.undo() {
                app.status_message = "Undo".into();
            }
        }
        KeyCode::Char('r') if mods.contains(KeyModifiers::CONTROL) => {
            if app.buffer.redo() {
                app.status_message = "Redo".into();
            }
        }
        KeyCode::Char('m') => app.pending_bookmark = true,
        KeyCode::Char('p') => app.paste_at_cursor(),
        KeyCode::Char('\'') => app.pending_goto_bookmark = true,
        KeyCode::Tab => {
            app.view_mode = app.view_mode.next();
            app.status_message = format!("View: {}", app.view_mode.label());
        }
        _ => {}
    }
}

fn handle_edit(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Esc => app.exit_edit_mode(),
        KeyCode::Char(c) if c.is_ascii_hexdigit() => app.edit_input(c),
        KeyCode::Left => app.move_left(),
        KeyCode::Right => app.move_right(),
        KeyCode::Up => app.move_up(),
        KeyCode::Down => app.move_down(),
        _ => {}
    }
}

fn handle_cmd(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Enter => app.execute_command(),
        KeyCode::Esc => {
            app.mode = Mode::Normal;
            app.command_input.clear();
        }
        KeyCode::Backspace => {
            app.command_input.pop();
            if app.command_input.is_empty() {
                app.mode = Mode::Normal;
            }
        }
        KeyCode::Char(c) => app.command_input.push(c),
        _ => {}
    }
}

fn handle_search(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Enter => {
            app.mode = Mode::Normal;
            app.search_hex();
        }
        KeyCode::Esc => {
            app.mode = Mode::Normal;
            app.search_input.clear();
        }
        KeyCode::Backspace => {
            app.search_input.pop();
        }
        KeyCode::Char(c) => app.search_input.push(c),
        _ => {}
    }
}

fn handle_visual(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Esc => {
            app.visual_start = None;
            app.mode = Mode::Normal;
            app.status_message.clear();
        }
        KeyCode::Char('h') | KeyCode::Left => app.move_left(),
        KeyCode::Char('l') | KeyCode::Right => app.move_right(),
        KeyCode::Char('k') | KeyCode::Up => app.move_up(),
        KeyCode::Char('j') | KeyCode::Down => app.move_down(),
        KeyCode::Char('y') | KeyCode::Char('d') => app.yank_visual(),
        KeyCode::Char('G') => app.goto_end(),
        KeyCode::Char('0') => app.goto_row_start(),
        KeyCode::Char('$') => app.goto_row_end(),
        KeyCode::Char('g') => app.goto_start(),
        _ => {}
    }
}
