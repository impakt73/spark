#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use spark::engine::Engine;

fn main() {
    Engine::run()
}
