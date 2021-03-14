use std::collections::HashMap;
use winit::event::{ElementState, VirtualKeyCode};

pub struct InputState {
    cur_keys: HashMap<VirtualKeyCode, ElementState>,
    prev_keys: HashMap<VirtualKeyCode, ElementState>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            cur_keys: HashMap::new(),
            prev_keys: HashMap::new(),
        }
    }

    pub fn get_cur_key_state(&self, key: VirtualKeyCode) -> ElementState {
        self.cur_keys
            .get(&key)
            .copied()
            .unwrap_or(ElementState::Released)
    }

    pub fn get_prev_key_state(&self, key: VirtualKeyCode) -> ElementState {
        self.prev_keys
            .get(&key)
            .copied()
            .unwrap_or(ElementState::Released)
    }

    pub fn update_key_state(&mut self, key: VirtualKeyCode, state: ElementState) {
        self.cur_keys.insert(key, state);
    }

    pub fn is_key_pressed(&self, key: VirtualKeyCode) -> bool {
        self.get_cur_key_state(key) == ElementState::Pressed
    }

    pub fn was_key_pressed(&self, key: VirtualKeyCode) -> bool {
        self.get_prev_key_state(key) == ElementState::Pressed
    }

    pub fn next_frame(&mut self) {
        self.prev_keys = self.cur_keys.clone()
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}
