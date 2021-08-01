use imgui::{im_str, Slider};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    event::{Event, StartCause, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, Window, WindowBuilder},
};

use crate::audio_device::{AudioDevice, AudioTrack};
use crate::renderer::Renderer;
use crate::{
    input_state::InputState,
    render_graph::{RenderGraph, RenderGraphDesc},
};

use serde::{Deserialize, Serialize};

use crate::log::*;

/// Number of "Rows" per second used by GNU Rocket Sync Tool
const ROWS_PER_SECOND: u32 = 100;

pub enum WindowConfig {
    Windowed { width: u32, height: u32 },
    Fullscreen,
}

impl Default for WindowConfig {
    fn default() -> Self {
        WindowConfig::Windowed {
            width: 1280,
            height: 720,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct SyncConfig {
    tracks: Vec<String>,
    data_path: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct DemoConfig {
    group_name: String,
    demo_name: String,
    track_path: String,
    sync: SyncConfig,
    graph: RenderGraphDesc,
}

impl DemoConfig {
    /// Attempts to load a demo config structure from the provided path
    fn from_path(path: &str) -> Result<DemoConfig, Box<dyn std::error::Error>> {
        let demo_config_str = std::fs::read_to_string(path)?;
        let demo_config: DemoConfig = serde_json::from_str(&demo_config_str)?;

        Ok(demo_config)
    }

    /// Returns the resource directory based the file path of a demo config
    fn query_res_dir(config_path: &str) -> String {
        let config_path = Path::new(&config_path);
        Path::parent(config_path)
            .expect("Unexpected demo config path format")
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Creates a resource path based the file path of a demo config
    fn make_res_path(config_path: &Option<String>, path: &str) -> String {
        if let Some(config_path) = config_path {
            let config_path = Path::new(config_path);
            let res_dir = PathBuf::from(&Self::query_res_dir(config_path.to_str().unwrap()));
            return res_dir.join(path).to_str().unwrap().to_string();
        } else {
            // Just return the original path if there's no demo config path to reference
            path.to_string()
        }
    }
}

/// Helper for measuring frame timer
///
/// Consumes frame time once per frame and manages the historical sample set over time
/// Provides simple averaging functionality
struct FrameTimer {
    /// Set of all valid frame time samples
    ///
    /// The number of elements in this set is limited to max_samples
    frame_times: VecDeque<Duration>,

    /// The maximum number of frame time samples to store before dropping old data
    max_samples: usize,
}

impl FrameTimer {
    /// Creates a new frame timer object with the specified max number of samples
    pub fn new(max_samples: usize) -> Self {
        FrameTimer {
            frame_times: Default::default(),
            max_samples,
        }
    }

    /// Inserts the provided frame time into the timer
    ///
    /// This will automatically bump out old values if necessary to stay under the max sample count
    pub fn push_sample(&mut self, frame_time: Duration) {
        self.frame_times.push_back(frame_time);

        // Remove an old sample if necessary
        if self.frame_times.len() > self.max_samples {
            self.frame_times.pop_front();

            // We should only ever be a single item over under normal operation
            assert!(self.frame_times.len() == self.max_samples);
        }
    }

    /// Calculates the current average frame time and returns it
    pub fn calculate_average(&self) -> Duration {
        let sum: Duration = self.frame_times.iter().sum();
        let avg_us = (sum.as_micros() as f64 / self.frame_times.len() as f64) as u64;
        Duration::from_micros(avg_us)
    }

    /// Returns a reference to the internal frame times queue
    ///
    /// This can be used by applications directly if they need more than averages
    pub fn get_frame_times(&self) -> &VecDeque<Duration> {
        &self.frame_times
    }
}

/// Returns a text representation of a duration suitable for display in the UI
fn format_duration(duration: &Duration) -> String {
    let total_secs = duration.as_secs_f64();
    let mins = (total_secs / 60.0).floor();
    let secs = total_secs % 60.0;
    format!("{:.0}:{:0>5.2}", mins, secs)
}

/// Builds a window title string based on the input demo config
fn build_window_title(demo_config: &Option<DemoConfig>) -> String {
    if let Some(config) = demo_config {
        format!("{} - {}", config.group_name, config.demo_name)
    } else {
        String::from("Spark Demo Engine")
    }
}

/// Enumerates different possible sources for synchronization data
enum SyncSource {
    /// Synchronization data sourced from a connected tool
    Client(rust_rocket::RocketClient),

    /// Synchronization data sourced from local memory
    Player(rust_rocket::RocketPlayer),

    /// No synchronization data
    None,
}

pub struct Engine {
    sync_source: SyncSource,
    current_sync_row: u32,
    imgui_context: imgui::Context,
    imgui_platform: WinitPlatform,
    graph: Option<RenderGraph>,
    renderer: Renderer,
    audio_track: Option<AudioTrack>,
    audio_device: Arc<AudioDevice>,
    input_state: InputState,
    window: Window,
    last_frame_time: Instant,
    frame_index: usize,
    exit_requested: bool,
    timer: FrameTimer,
    demo_config_path: Option<String>,
    demo_config: Option<DemoConfig>,
}

impl Engine {
    fn new(window: Window, demo_config_path: Option<String>) -> Self {
        let mut imgui_context = imgui::Context::create();
        imgui_context.set_renderer_name(Some(imgui::ImString::from(String::from("Spark"))));
        imgui_context
            .io_mut()
            .backend_flags
            .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);

        let mut imgui_platform = WinitPlatform::init(&mut imgui_context);
        imgui_platform.attach_window(imgui_context.io_mut(), &window, HiDpiMode::Default);

        let audio_device = Arc::new(AudioDevice::new().expect("Failed to create audio device"));

        let is_debug_build = cfg!(debug_assertions);
        let renderer = Renderer::new(&window, is_debug_build, &mut imgui_context)
            .expect("Failed to create renderer");

        window.set_visible(true);

        Engine {
            sync_source: SyncSource::None,
            current_sync_row: 0,
            imgui_context,
            imgui_platform,
            graph: None,
            renderer,
            audio_track: None,
            audio_device,
            input_state: InputState::default(),
            window,
            last_frame_time: Instant::now(),
            frame_index: 0,
            exit_requested: false,
            timer: FrameTimer::new(64),
            demo_config_path,
            demo_config: None,
        }
    }

    /// Reloads the current demo config from disk
    fn reload_demo_config_file(&mut self) {
        if let Some(path) = &self.demo_config_path {
            // Attempt to load a new demo config
            match DemoConfig::from_path(path) {
                Ok(demo_config) => {
                    self.demo_config = Some(demo_config);

                    // Reload the demo config if we've replaced it
                    self.reload_demo_config();
                }
                Err(err) => {
                    error!("Failed to load demo config: {} ({})", path, err);
                }
            }
        }
    }

    fn reload_demo_config(&mut self) {
        if let Some(demo_config) = &self.demo_config {
            self.window
                .set_title(&build_window_title(&self.demo_config));

            let track_path =
                DemoConfig::make_res_path(&self.demo_config_path, &demo_config.track_path);
            match AudioTrack::from_path(self.audio_device.clone(), &track_path) {
                Ok(mut audio_track) => {
                    // Attempt to copy the old track's state into the new track if possible
                    if let Some(old_track) = &self.audio_track {
                        audio_track
                            .set_position(&old_track.get_position().unwrap_or_default())
                            .ok();
                        if old_track.is_playing() {
                            audio_track.play().ok();
                        }
                    }
                    self.audio_track = Some(audio_track);
                }
                Err(err) => {
                    error!(
                        "Failed to load audio track: {} ({})",
                        demo_config.track_path, err
                    );
                }
            }

            let is_debug_build = cfg!(debug_assertions);
            if is_debug_build {
                match rust_rocket::RocketClient::new() {
                    Ok(client) => {
                        self.sync_source = SyncSource::Client(client);
                    }
                    Err(err) => {
                        error!("Failed to create sync client: {}", err);
                    }
                }
            } else {
                match std::fs::read(&demo_config.sync.data_path) {
                    Ok(data) => {
                        match serde_json::from_slice::<Vec<rust_rocket::track::Track>>(&data) {
                            Ok(tracks) => {
                                self.sync_source =
                                    SyncSource::Player(rust_rocket::RocketPlayer::new(tracks));
                            }
                            Err(err) => {
                                error!("Failed to parse sync data: {}", err);
                            }
                        }
                    }
                    Err(err) => {
                        error!("Failed to load sync data: {}", err);
                    }
                }
            }

            // Idle the renderer before we modify any rendering resources
            self.renderer.wait_for_idle();

            let resource_dir = self
                .demo_config_path
                .as_ref()
                .map(|path| DemoConfig::query_res_dir(path));

            match RenderGraph::new(
                &demo_config.graph,
                resource_dir.as_deref(),
                &mut self.renderer,
            ) {
                Ok(graph) => {
                    self.graph = Some(graph);
                }
                Err(err) => {
                    error!("Failed to load render graph: {}", err);
                }
            }
        }
    }

    fn destroy(&mut self) {
        self.renderer.wait_for_idle();
        if let Some(track) = &mut self.audio_track {
            if let Err(err) = track.stop() {
                error!("Failed to stop audio track: {}", err);
            }
        }
    }

    fn resize(&mut self) {
        // TODO: This code needs to be updated to properly handle minimized windows
        //       When a window is minimized, it resizes to 0x0 which causes all sorts of problems
        //       inside the graphics api. This basically results in crashes on minimize. :/
        //       This will be fixed in a future change.
        self.renderer.recreate_swapchain(&self.window).unwrap();

        // We need to reload the current demo config whenever the window resizes because it contains
        // resources that are resolution/swapchain-size dependent.
        self.reload_demo_config();
    }

    fn handle_event(&mut self, event: &Event<()>, control_flow: &mut ControlFlow) {
        match event {
            Event::NewEvents(StartCause::Init) => {
                *control_flow = ControlFlow::Poll;

                // The demo config needs to be initialized once the window is created
                self.reload_demo_config_file();
            }
            _ => {
                self.imgui_platform
                    .handle_event(self.imgui_context.io_mut(), &self.window, event);

                match event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(_new_size) => {
                            self.resize();
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            if let Some(keycode) = input.virtual_keycode {
                                self.input_state.update_key_state(keycode, input.state);
                            }
                        }
                        _ => {}
                    },
                    Event::MainEventsCleared => {
                        self.run_frame();

                        if self.exit_requested {
                            *control_flow = ControlFlow::Exit;
                        }
                    }
                    Event::LoopDestroyed => {
                        self.destroy();
                    }
                    _ => {}
                }
            }
        }

        if self.input_state.is_key_pressed(VirtualKeyCode::Escape) {
            *control_flow = ControlFlow::Exit;
        }
    }

    fn run_frame(&mut self) {
        self.renderer.begin_frame();

        let now = Instant::now();
        let delta_time = now - self.last_frame_time;

        let is_debug_build = cfg!(debug_assertions);
        if !is_debug_build {
            if self.frame_index == 0 {
                if let Some(track) = &mut self.audio_track {
                    track.play().expect("Failed to play audio track");
                }
            }
        } else {
            // Keyboard based audio track control is only avaiable in debug builds
            if let Some(track) = &mut self.audio_track {
                if self.input_state.is_key_pressed(VirtualKeyCode::Space)
                    && !self.input_state.was_key_pressed(VirtualKeyCode::Space)
                {
                    if let Err(err) = track.toggle_pause() {
                        error!("Failed to toggle audio track playback: {}", err);
                    }
                }
                if !track.is_playing() {
                    let mut modifier = 0.05;
                    if self.input_state.is_key_pressed(VirtualKeyCode::LShift)
                        || self.input_state.is_key_pressed(VirtualKeyCode::RShift)
                    {
                        if self.input_state.is_key_pressed(VirtualKeyCode::LAlt)
                            || self.input_state.is_key_pressed(VirtualKeyCode::RAlt)
                        {
                            modifier = 25.0;
                        } else {
                            modifier = 5.0;
                        }
                    }

                    let offset = Duration::from_secs_f64(delta_time.as_secs_f64() * modifier);
                    if self.input_state.is_key_pressed(VirtualKeyCode::Left) {
                        if let Err(err) = track.subtract_position_offset(&offset) {
                            error!("Failed to rewind audio track: {}", err);
                        }
                    } else if self.input_state.is_key_pressed(VirtualKeyCode::Right) {
                        if let Err(err) = track.add_position_offset(&offset) {
                            error!("Failed to fast-forward audio track: {}", err);
                        }
                    }
                }
            }
        }

        self.current_sync_row = if let Some(track) = &self.audio_track {
            (track.get_position().unwrap_or_default().as_secs_f32() * ROWS_PER_SECOND as f32) as u32
        } else {
            0
        };

        if let SyncSource::Client(client) = &mut self.sync_source {
            if let Err(err) = client.set_row(self.current_sync_row) {
                error!("Failed to set rocket's row {}", err);
            }

            while let Ok(Some(event)) = client.poll_events() {
                match event {
                    rust_rocket::client::Event::SetRow(row) => {
                        let track_time = row as f32 / ROWS_PER_SECOND as f32;

                        if let Some(track) = &mut self.audio_track {
                            if let Err(err) =
                                track.set_position(&Duration::from_secs_f32(track_time))
                            {
                                error!("Failed to set audio track via rocket: {}", err);
                            }
                        }
                    }
                    rust_rocket::client::Event::Pause(pause) => {
                        if let Some(track) = &mut self.audio_track {
                            if pause {
                                if let Err(err) = track.pause() {
                                    error!("Failed to pause audio track via rocket: {}", err);
                                }
                            } else if let Err(err) = track.play() {
                                error!("Failed to play audio track via rocket: {}", err);
                            }
                        }
                    }
                    rust_rocket::client::Event::SaveTracks => {
                        let tracks = client.save_tracks();
                        std::fs::write(
                            "res/sync.json",
                            serde_json::to_string_pretty(&tracks).unwrap(),
                        )
                        .unwrap();
                    }
                }
            }
        }

        self.imgui_context.io_mut().update_delta_time(delta_time);
        self.last_frame_time = now;

        self.timer.push_sample(delta_time);

        self.imgui_platform
            .prepare_frame(self.imgui_context.io_mut(), &self.window)
            .expect("Failed to prepare frame");

        let ui = self.imgui_context.frame();

        let avg_frame_time_us = self.timer.calculate_average().as_micros() as f64;

        let mut reload_demo = false;

        // CTRL+R can be used as a hotkey for a demo reload operation
        if self.input_state.is_key_pressed(VirtualKeyCode::LControl)
            && self.input_state.is_key_pressed(VirtualKeyCode::R)
        {
            reload_demo = true;
        }

        // Render UI
        if let Some(main_menu_bar) = ui.begin_main_menu_bar() {
            if let Some(file_menu) = ui.begin_menu(imgui::im_str!("File"), true) {
                if imgui::MenuItem::new(imgui::im_str!("Exit")).build(&ui) {
                    self.exit_requested = true;
                }

                file_menu.end(&ui);
            }

            if let Some(demo_menu) = ui.begin_menu(imgui::im_str!("Demo"), true) {
                if imgui::MenuItem::new(imgui::im_str!("Load")).build(&ui) {
                    if let Some(path) = tinyfiledialogs::open_file_dialog(
                        "Load Demo Config",
                        "",
                        Some((&["*.json"], "Demo Config Files (*.json)")),
                    ) {
                        self.demo_config_path = Some(path);
                        reload_demo = true;
                    }
                }
                if imgui::MenuItem::new(imgui::im_str!("Reload")).build(&ui) {
                    reload_demo = true;
                }

                demo_menu.end(&ui);
            }

            ui.text(format!("CPU: {:.2}ms", avg_frame_time_us / 1000.0));
            {
                let frame_times_us = self
                    .timer
                    .get_frame_times()
                    .iter()
                    .map(|time| (time.as_micros() as f64 / 1000.0) as f32)
                    .collect::<Vec<f32>>();
                ui.plot_histogram(imgui::im_str!(""), &frame_times_us)
                    .scale_min(0.0)
                    .scale_max(16.7)
                    .graph_size([64.0, 20.0])
                    .build();
            }
            if let Some(track) = &mut self.audio_track {
                let audio_pos = track.get_position().unwrap();
                let audio_length = track.get_length();

                ui.text(format!(
                    "Track: {} [{}]",
                    format_duration(&audio_pos),
                    format_duration(&audio_length)
                ));

                let mut audio_pos_in_seconds = audio_pos.as_secs_f32();
                let audio_length_in_seconds = audio_length.as_secs_f32();

                if Slider::new(im_str!(""))
                    .display_format(im_str!("%.2f"))
                    .range(0.0..=audio_length_in_seconds)
                    .flags(imgui::SliderFlags::ALWAYS_CLAMP)
                    .build(&ui, &mut audio_pos_in_seconds)
                {
                    track
                        .set_position(&Duration::from_secs_f32(audio_pos_in_seconds))
                        .unwrap();
                }
            } else {
                ui.text("No Track Loaded");
            }
            main_menu_bar.end(&ui);
        }

        self.imgui_platform.prepare_render(&ui, &self.window);
        let draw_data = ui.render();

        let cur_swapchain_idx = self.renderer.get_cur_swapchain_idx();
        let cur_time = self
            .audio_track
            .as_ref()
            .map_or(Duration::default(), |track| track.get_position().unwrap());

        let mut render_graph_image = false;
        if let Some(graph) = &mut self.graph {
            if let Some(output_image_view) = graph.get_output_image(cur_swapchain_idx) {
                self.renderer.update_graph_image(output_image_view);

                render_graph_image = true;
            }

            // Produce a sync buffer here and pass it to the graph execution
            let mut sync_buffer_data = Vec::new();
            if let Some(config) = &self.demo_config {
                match &mut self.sync_source {
                    SyncSource::Client(client) => {
                        for track_name in &config.sync.tracks {
                            if let Ok(track) = client.get_track_mut(track_name) {
                                let track_val = track.get_value(self.current_sync_row as f32);
                                sync_buffer_data.push(track_val);
                            }
                        }
                    }
                    SyncSource::Player(player) => {
                        for track_name in &config.sync.tracks {
                            if let Some(track) = player.get_track(track_name) {
                                let track_val = track.get_value(self.current_sync_row as f32);
                                sync_buffer_data.push(track_val);
                            }
                        }
                    }
                    _ => {}
                }
            }

            self.renderer
                .execute_graph(graph, &cur_time, &sync_buffer_data)
                .expect("Failed to execute render graph");
        }

        self.renderer.begin_render();

        if render_graph_image {
            self.renderer.render_graph_image();
        }

        let is_debug_build = cfg!(debug_assertions);
        if is_debug_build {
            self.renderer.render_ui(draw_data);
        }

        self.renderer.end_render();

        self.renderer.end_frame();

        self.input_state.next_frame();

        // If a demo reload was requested via the UI, perform the operation once the current frame has been submitted
        if reload_demo {
            self.reload_demo_config_file();
        }

        self.frame_index += 1;
    }

    pub fn run() -> ! {
        Self::run_with_config(None, None);
    }

    pub fn run_with_config(
        window_config: Option<WindowConfig>,
        demo_config_path: Option<String>,
    ) -> ! {
        let mut builder = WindowBuilder::new();
        builder = builder.with_visible(false);
        builder = builder.with_title(&build_window_title(&None));

        let window_config = window_config.unwrap_or_default();
        if let WindowConfig::Windowed { width, height } = window_config {
            builder = builder.with_inner_size(winit::dpi::PhysicalSize::new(width, height));
        } else {
            builder = builder.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        let event_loop = EventLoop::new();
        let window = builder.build(&event_loop).expect("Failed to create window");

        let mut engine = Self::new(window, demo_config_path);

        event_loop.run(move |event, _, control_flow| {
            engine.handle_event(&event, control_flow);
        });
    }
}
