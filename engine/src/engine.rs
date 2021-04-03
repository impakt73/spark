use ash::vk;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::{
    collections::VecDeque,
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
    render_graph::{
        RenderGraph, RenderGraphBufferParams, RenderGraphDesc, RenderGraphDispatchDimensions,
        RenderGraphImageParams, RenderGraphNodeDesc, RenderGraphPipelineSource,
        RenderGraphResourceDesc, RenderGraphResourceParams,
    },
};

use serde::{Deserialize, Serialize};

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
pub struct DemoConfig {
    group_name: String,
    demo_name: String,
    track_path: String,
    graph: GraphConfig,
}

impl DemoConfig {
    /// Attempts to load a demo config structure from the provided path
    pub fn from_path(path: &str) -> Result<DemoConfig, Box<dyn std::error::Error>> {
        let demo_config_str = std::fs::read_to_string(path)?;
        let demo_config: DemoConfig = serde_json::from_str(&demo_config_str)?;

        Ok(demo_config)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphConfig {
    resources: Vec<GraphConfigResource>,
    output_resource: Option<String>,
    nodes: Vec<GraphConfigNode>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphConfigResource {
    name: String,
    params: GraphConfigResourceParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum GraphConfigResourceParams {
    Image,
    Buffer,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphConfigNode {
    name: String,
    pipeline: String,
    refs: Vec<String>,
    deps: Vec<String>,
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

/// Attempts to load a demo config structure from the provided path
fn load_config(path: &str) -> Result<DemoConfig, Box<dyn std::error::Error>> {
    let demo_config_str = std::fs::read_to_string(path)?;
    let demo_config: DemoConfig = serde_json::from_str(&demo_config_str)?;

    Ok(demo_config)
}

/// Builds a window title string based on the input demo config
fn build_window_title(demo_config: &Option<DemoConfig>) -> String {
    if let Some(config) = demo_config {
        format!("{} - {}", config.group_name, config.demo_name)
    } else {
        String::from("Spark Demo Engine")
    }
}

pub struct Engine {
    imgui_context: imgui::Context,
    imgui_platform: WinitPlatform,
    graph: Option<RenderGraph>,
    renderer: Renderer,
    audio_track: Option<AudioTrack>,
    audio_device: Arc<AudioDevice>,
    input_state: InputState,
    window: Window,
    last_frame_time: Instant,
    exit_requested: bool,
    timer: FrameTimer,
    demo_config_path: Option<String>,
    demo_config: Option<DemoConfig>,
}

impl Engine {
    fn new(
        window: Window,
        demo_config_path: Option<String>,
        demo_config: Option<DemoConfig>,
    ) -> Self {
        let mut imgui_context = imgui::Context::create();
        imgui_context.set_renderer_name(Some(imgui::ImString::from(String::from("Spark"))));
        imgui_context
            .io_mut()
            .backend_flags
            .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);

        let mut imgui_platform = WinitPlatform::init(&mut imgui_context);
        imgui_platform.attach_window(imgui_context.io_mut(), &window, HiDpiMode::Default);

        let audio_device = Arc::new(AudioDevice::new().expect("Failed to create audio device"));

        let enable_validation = cfg!(debug_assertions);
        let renderer = Renderer::new(&window, enable_validation, &mut imgui_context)
            .expect("Failed to create renderer");

        window.set_visible(true);

        Engine {
            imgui_context,
            imgui_platform,
            graph: None,
            renderer,
            audio_track: None,
            audio_device,
            input_state: InputState::default(),
            window,
            last_frame_time: Instant::now(),
            exit_requested: false,
            timer: FrameTimer::new(64),
            demo_config_path,
            demo_config,
        }
    }

    /// Reloads the current demo config from disk
    fn reload_demo_config_file(&mut self) {
        if let Some(path) = &self.demo_config_path {
            // Attempt to load a new demo config
            match load_config(&path) {
                Ok(demo_config) => {
                    self.demo_config = Some(demo_config);

                    // Reload the demo config if we've replaced it
                    self.reload_demo_config();
                }
                Err(err) => {
                    println!("Failed to load demo config: {} ({})", path, err);
                }
            }
        }
    }

    fn reload_demo_config(&mut self) {
        if let Some(demo_config) = &self.demo_config {
            self.window
                .set_title(&build_window_title(&self.demo_config));

            match AudioTrack::from_path(self.audio_device.clone(), &demo_config.track_path) {
                Ok(audio_track) => {
                    self.audio_track = Some(audio_track);
                }
                Err(err) => {
                    println!(
                        "Failed to load audio track: {} ({})",
                        demo_config.track_path, err
                    );
                }
            }

            // Idle the renderer before we modify any rendering resources
            self.renderer.wait_for_idle();

            let (swapchain_width, swapchain_height) = self.renderer.get_swapchain_resolution();

            let dispatch_dims = RenderGraphDispatchDimensions {
                num_groups_x: ((swapchain_width + 7) & !7) / 8,
                num_groups_y: ((swapchain_height + 3) & !3) / 4,
                num_groups_z: 1,
            };

            let nodes = demo_config
                .graph
                .nodes
                .iter()
                .map(|config_node| RenderGraphNodeDesc {
                    name: config_node.name.clone(),
                    pipeline: RenderGraphPipelineSource::File(config_node.pipeline.clone()),
                    refs: config_node.refs.clone(),
                    dims: dispatch_dims,
                    deps: config_node.deps.clone(),
                })
                .collect::<Vec<RenderGraphNodeDesc>>();

            let resources = demo_config
                .graph
                .resources
                .iter()
                .map(|config_resource| {
                    let params = match config_resource.params {
                        GraphConfigResourceParams::Image => {
                            RenderGraphResourceParams::Image(RenderGraphImageParams {
                                width: swapchain_width,
                                height: swapchain_height,
                                format: vk::Format::R8G8B8A8_UNORM,
                            })
                        }
                        GraphConfigResourceParams::Buffer => {
                            // TODO: Support buffer resources
                            RenderGraphResourceParams::Buffer(RenderGraphBufferParams { size: 0 })
                        }
                    };
                    RenderGraphResourceDesc {
                        name: config_resource.name.clone(),
                        params,
                    }
                })
                .collect::<Vec<RenderGraphResourceDesc>>();

            let output_image_name = demo_config.graph.output_resource.clone();

            let render_graph_desc = RenderGraphDesc {
                resources,
                nodes,
                output_image_name,
            };

            match RenderGraph::new(&render_graph_desc, &mut self.renderer) {
                Ok(graph) => {
                    self.graph = Some(graph);
                }
                Err(err) => {
                    println!("Failed to load render graph: {}", err);
                }
            }
        }
    }

    fn destroy(&mut self) {
        self.renderer.wait_for_idle();
        if let Some(track) = &mut self.audio_track {
            if let Err(err) = track.stop() {
                println!("Failed to stop audio track: {}", err);
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
                self.reload_demo_config();
            }
            _ => {
                self.imgui_platform
                    .handle_event(self.imgui_context.io_mut(), &self.window, &event);

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

        // Audio track control
        if let Some(track) = &mut self.audio_track {
            if self.input_state.is_key_pressed(VirtualKeyCode::Space)
                && !self.input_state.was_key_pressed(VirtualKeyCode::Space)
            {
                if let Err(err) = track.toggle_pause() {
                    println!("Failed to toggle audio track playback: {}", err);
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
                        println!("Failed to rewind audio track: {}", err);
                    }
                } else if self.input_state.is_key_pressed(VirtualKeyCode::Right) {
                    if let Err(err) = track.add_position_offset(&offset) {
                        println!("Failed to fast-forward audio track: {}", err);
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

            self.renderer
                .execute_graph(graph, &cur_time)
                .expect("Failed to execute render graph");
        }

        self.renderer.begin_render();

        if render_graph_image {
            self.renderer.render_graph_image();
        }

        self.renderer.render_ui(draw_data);

        self.renderer.end_render();

        self.renderer.end_frame();

        self.input_state.next_frame();

        // If a demo reload was requested via the UI, perform the operation once the current frame has been submitted
        if reload_demo {
            self.reload_demo_config_file();
        }
    }

    pub fn run() -> ! {
        Self::run_with_config(None, None, None);
    }

    pub fn run_with_config(
        window_config: Option<WindowConfig>,
        demo_config_path: Option<String>,
        demo_config: Option<DemoConfig>,
    ) -> ! {
        let mut builder = WindowBuilder::new();
        builder = builder.with_visible(false);
        builder = builder.with_title(&build_window_title(&demo_config));

        let window_config = window_config.unwrap_or_default();
        if let WindowConfig::Windowed { width, height } = window_config {
            builder = builder.with_inner_size(winit::dpi::PhysicalSize::new(width, height));
        } else {
            builder = builder.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        let event_loop = EventLoop::new();
        let window = builder.build(&event_loop).expect("Failed to create window");

        let mut engine = Self::new(window, demo_config_path, demo_config);

        event_loop.run(move |event, _, control_flow| {
            engine.handle_event(&event, control_flow);
        });
    }
}
