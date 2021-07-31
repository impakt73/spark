#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use spark::engine::Engine;

fn init_logging() {
    use tracing_subscriber::{prelude::*, registry::Registry};

    let registry = Registry::default();

    // Line-oriented text output to stdout
    let registry = registry.with(tracing_subscriber::fmt::layer());

    // Register our tracing subscriber
    tracing::subscriber::set_global_default(registry)
        .expect("Failed to install the tracing subscriber");
}

fn main() {
    init_logging();

    Engine::run();
}
