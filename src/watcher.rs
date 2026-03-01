//! File watcher for auto-reloading config on change

use crate::config::{self, ResolvedColors, SharedColors};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc;
use std::thread;

/// Start watching the config file for changes.
/// Returns the watcher handle (must be kept alive) and a channel receiver
/// that signals when a reload happened.
pub fn watch_config(
    colors: SharedColors,
) -> anyhow::Result<(RecommendedWatcher, mpsc::Receiver<()>)> {
    let config_path = config::config_path();
    let (notify_tx, notify_rx) = mpsc::channel::<()>();

    // Create parent dir if needed so we can watch it
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let watch_path = config_path.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx)?;

    // Watch the parent directory (file might not exist yet)
    let dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    watcher.watch(dir, RecursiveMode::NonRecursive)?;

    // Spawn a thread to process events
    thread::spawn(move || {
        for Event { kind, paths, .. } in rx.into_iter().flatten() {
            let dominated = matches!(kind, EventKind::Modify(_) | EventKind::Create(_));
            if dominated && paths.iter().any(|p| p.ends_with("config.toml")) {
                // Small delay — editors do rename dances
                thread::sleep(std::time::Duration::from_millis(50));
                let cfg = config::load_config_from(&watch_path);
                let resolved = ResolvedColors::from_config(&cfg.colors);
                if let Ok(mut lock) = colors.write() {
                    *lock = resolved;
                }
                let _ = notify_tx.send(());
            }
        }
    });

    Ok((watcher, notify_rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: watcher tests are inherently timing-sensitive.
    // We test the reload logic directly instead.
    #[test]
    fn test_reload_logic() {
        let colors = config::new_shared_colors();
        {
            let c = colors.read().unwrap();
            assert_eq!(c.null, ratatui::style::Color::DarkGray);
        }
        // Simulate what the watcher does
        let toml_str = r#"
[colors]
null = "red"
"#;
        let cfg: config::ConfigFile = toml::from_str(toml_str).unwrap();
        let resolved = ResolvedColors::from_config(&cfg.colors);
        {
            let mut lock = colors.write().unwrap();
            *lock = resolved;
        }
        {
            let c = colors.read().unwrap();
            assert_eq!(c.null, ratatui::style::Color::Red);
        }
    }
}
