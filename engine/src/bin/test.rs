#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use spark::engine::{DemoConfig, Engine, WindowConfig};

/// Default path for the demo config file
const DEFAULT_DEMO_CONFIG_PATH: &str = "res/data/demo.json";

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config = DemoConfig::from_path(DEFAULT_DEMO_CONFIG_PATH)?;
    Engine::run_with_config(
        Some(WindowConfig::Fullscreen),
        Some(String::from(DEFAULT_DEMO_CONFIG_PATH)),
        Some(config),
    );
}
