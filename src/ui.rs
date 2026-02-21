use crate::app::{App, Mode};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(f.area());
    draw_header(f, app, chunks[0]);
    draw_hex(f, app, chunks[1]);
    draw_info(f, app, chunks[2]);
    draw_status(f, app, chunks[3]);
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let name = app
        .buffer
        .path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "[stdin]".into());
    let m = if app.buffer.is_modified() { " [+]" } else { "" };
    let mg = app
        .magic_info
        .map(|s| format!(" ({})", s))
        .unwrap_or_default();
    let t = format!(
        " {} {}{} — {} bytes{}",
        app.view_mode.label(),
        name,
        m,
        app.buffer.len(),
        mg
    );
    f.render_widget(
        Paragraph::new(t).style(Style::default().bg(Color::DarkGray).fg(Color::White)),
        area,
    );
}

fn draw_hex(f: &mut Frame, app: &App, area: Rect) {
    let rows = area.height as usize;
    let mut lines = Vec::new();
    for row in 0..rows {
        let ro = (app.scroll_offset + row) * app.cols;
        if ro >= app.buffer.len() {
            lines.push(Line::from(Span::styled(
                "~",
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }
        let mut spans = Vec::new();
        spans.push(Span::styled(
            format!("{:08x}  ", ro),
            Style::default().fg(Color::Cyan),
        ));
        for col in 0..app.cols {
            let off = ro + col;
            if col == app.cols / 2 {
                spans.push(Span::raw(" "));
            }
            if off < app.buffer.len() {
                let b = app.buffer.data()[off];
                let txt = format!("{} ", app.view_mode.format_byte(b));
                let mut st = bstyle(b);
                if off == app.cursor {
                    st = st.bg(Color::White).fg(Color::Black);
                } else if app.mode == Mode::Visual {
                    if let Some((vs, ve)) = app.visual_range() {
                        if off >= vs && off <= ve {
                            st = st.bg(Color::Blue);
                        }
                    }
                }
                spans.push(Span::styled(txt, st));
            } else {
                spans.push(Span::raw(" ".repeat(app.view_mode.cell_width())));
            }
        }
        spans.push(Span::styled(" |", Style::default().fg(Color::DarkGray)));
        for col in 0..app.cols {
            let off = ro + col;
            if off < app.buffer.len() {
                let b = app.buffer.data()[off];
                let c = if b.is_ascii_graphic() || b == b' ' {
                    b as char
                } else {
                    '.'
                };
                let mut st = bstyle(b);
                if off == app.cursor {
                    st = st.bg(Color::White).fg(Color::Black);
                }
                spans.push(Span::styled(format!("{}", c), st));
            }
        }
        spans.push(Span::styled("|", Style::default().fg(Color::DarkGray)));
        lines.push(Line::from(spans));
    }
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::NONE)),
        area,
    );
}

fn draw_info(f: &mut Frame, app: &App, area: Rect) {
    let t = format!(
        " 0x{:08X} ({}) | {}",
        app.cursor,
        app.cursor,
        app.current_byte_info()
    );
    f.render_widget(
        Paragraph::new(t).style(Style::default().bg(Color::DarkGray).fg(Color::White)),
        area,
    );
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let t = match app.mode {
        Mode::Command => format!(":{}", app.command_input),
        Mode::Search => format!("/{}", app.search_input),
        _ if !app.status_message.is_empty() => app.status_message.clone(),
        _ => {
            let m = match app.mode {
                Mode::Normal => "NORMAL",
                Mode::Edit => "EDIT",
                Mode::Visual => "VISUAL",
                _ => "",
            };
            format!(" {} | q i v / : Tab m '", m)
        }
    };
    let st = match app.mode {
        Mode::Edit => Style::default().fg(Color::Green),
        Mode::Visual => Style::default().fg(Color::Blue),
        Mode::Command | Mode::Search => Style::default().fg(Color::Yellow),
        _ => Style::default(),
    };
    f.render_widget(Paragraph::new(t).style(st), area);
}

fn bstyle(b: u8) -> Style {
    match b {
        0 => Style::default().fg(Color::DarkGray),
        0x20..=0x7E => Style::default().fg(Color::Green),
        0x01..=0x1F | 0x7F => Style::default().fg(Color::Red),
        _ => Style::default().fg(Color::Yellow),
    }
}
