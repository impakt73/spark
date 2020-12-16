#![windows_subsystem = "windows"]

use spark::engine::Engine;
fn main() {
    let engine = Engine::new();
    engine.run();
}
