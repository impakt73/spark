use ash::vk;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};
use winit::{
    event::{DeviceEvent, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::audio_device::{AudioDevice, AudioTrack};
use crate::renderer::Renderer;
use crate::{
    input_state::InputState,
    render_graph::{
        RenderGraph, RenderGraphDesc, RenderGraphDispatchDimensions, RenderGraphImageParams,
        RenderGraphNodeDesc, RenderGraphPipelineSource, RenderGraphResourceDesc,
        RenderGraphResourceParams,
    },
};

use align_data::include_aligned;

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

pub struct Engine {
    imgui_context: Option<imgui::Context>,
    imgui_platform: Option<WinitPlatform>,
    graph: Option<RenderGraph>,
    renderer: Option<Renderer>,
    audio_track: Option<AudioTrack>,
    audio_device: Option<AudioDevice>,
    input_state: InputState,
    last_frame_time: Instant,
    exit_requested: bool,
    timer: FrameTimer,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            imgui_context: None,
            imgui_platform: None,
            graph: None,
            renderer: None,
            audio_track: None,
            audio_device: None,
            input_state: InputState::default(),
            last_frame_time: Instant::now(),
            exit_requested: false,
            timer: FrameTimer::new(64),
        }
    }

    pub fn init(&mut self, window: &Window) {
        let mut context = imgui::Context::create();
        context.set_renderer_name(Some(imgui::ImString::from(String::from("Spark"))));
        context
            .io_mut()
            .backend_flags
            .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);

        let mut platform = WinitPlatform::init(&mut context);
        platform.attach_window(context.io_mut(), &window, HiDpiMode::Default);

        let audio_device = AudioDevice::new().expect("Failed to create audio device");
        let audio_track =
            AudioTrack::from_path("res/audio/test.mp3").expect("Failed to create audio track");

        let enable_validation = cfg!(debug_assertions);
        let renderer = Renderer::new(&window, enable_validation, &mut context)
            .expect("Failed to create renderer");

        self.imgui_context = Some(context);
        self.imgui_platform = Some(platform);
        self.audio_track = Some(audio_track);
        self.audio_device = Some(audio_device);
        self.renderer = Some(renderer);

        self.init_render_graph();
    }

    fn init_render_graph(&mut self) {
        // Test graph
        let (swapchain_width, swapchain_height) =
            self.renderer.as_ref().unwrap().get_swapchain_resolution();

        let dispatch_dims = RenderGraphDispatchDimensions {
            num_groups_x: ((swapchain_width + 7) & !7) / 8,
            num_groups_y: ((swapchain_height + 3) & !3) / 4,
            num_groups_z: 1,
        };

        let mut nodes = Vec::new();

        nodes.push(RenderGraphNodeDesc {
            name: String::from("Red"),
            pipeline: RenderGraphPipelineSource::Buffer(unsafe {
                include_aligned!(u32, "../spv/Red.comp.spv")
                    .align_to::<u32>()
                    .1
            }),
            refs: vec![String::from("RedImage")],
            dims: dispatch_dims,
            deps: Vec::new(),
        });

        nodes.push(RenderGraphNodeDesc {
            name: String::from("Green"),
            pipeline: RenderGraphPipelineSource::Buffer(unsafe {
                include_aligned!(u32, "../spv/Green.comp.spv")
                    .align_to::<u32>()
                    .1
            }),
            refs: vec![String::from("GreenImage")],
            dims: dispatch_dims,
            deps: Vec::new(),
        });

        nodes.push(RenderGraphNodeDesc {
            name: String::from("Yellow"),
            pipeline: RenderGraphPipelineSource::Buffer(unsafe {
                include_aligned!(u32, "../spv/Yellow.comp.spv")
                    .align_to::<u32>()
                    .1
            }),
            refs: vec![
                String::from("RedImage"),
                String::from("GreenImage"),
                String::from("YellowImage"),
            ],
            dims: dispatch_dims,
            deps: vec![String::from("Red"), String::from("Green")],
        });

        let mut resources = Vec::new();

        resources.push(RenderGraphResourceDesc {
            name: String::from("RedImage"),
            params: RenderGraphResourceParams::Image(RenderGraphImageParams {
                width: swapchain_width,
                height: swapchain_height,
                format: vk::Format::R8G8B8A8_UNORM,
            }),
        });

        resources.push(RenderGraphResourceDesc {
            name: String::from("GreenImage"),
            params: RenderGraphResourceParams::Image(RenderGraphImageParams {
                width: swapchain_width,
                height: swapchain_height,
                format: vk::Format::R8G8B8A8_UNORM,
            }),
        });

        resources.push(RenderGraphResourceDesc {
            name: String::from("YellowImage"),
            params: RenderGraphResourceParams::Image(RenderGraphImageParams {
                width: swapchain_width,
                height: swapchain_height,
                format: vk::Format::R8G8B8A8_UNORM,
            }),
        });

        let output_image_name = Some(String::from("YellowImage"));

        let render_graph_desc = RenderGraphDesc {
            resources,
            nodes,
            output_image_name,
        };
        self.graph = Some(
            RenderGraph::new(&render_graph_desc, self.renderer.as_mut().unwrap())
                .expect("Failed to create render graph"),
        );
    }

    fn destroy(&mut self) {
        self.renderer.as_mut().unwrap().wait_for_idle();
        if let Err(err) = self.audio_track.as_mut().unwrap().stop() {
            println!("Failed to stop audio track: {}", err);
        }
    }

    fn resize(&mut self, window: &Window) {
        // TODO: This code needs to be updated to properly handle minimized windows
        //       When a window is minimized, it resizes to 0x0 which causes all sorts of problems
        //       inside the graphics api. This basically results in crashes on minimize. :/
        //       This will be fixed in a future change.
        self.renderer
            .as_mut()
            .unwrap()
            .recreate_swapchain(&window)
            .unwrap();

        self.init_render_graph();
    }

    fn handle_event(
        &mut self,
        window: &mut Window,
        event: &Event<()>,
        control_flow: &mut ControlFlow,
    ) {
        match event {
            Event::NewEvents(StartCause::Init) => {
                *control_flow = ControlFlow::Poll;

                self.init(&window);
            }
            _ => {
                self.imgui_platform.as_mut().unwrap().handle_event(
                    self.imgui_context.as_mut().unwrap().io_mut(),
                    &window,
                    &event,
                );

                match event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(_new_size) => {
                            self.resize(window);
                        }
                        _ => {}
                    },
                    Event::MainEventsCleared => {
                        self.run_frame(window);

                        if self.exit_requested {
                            *control_flow = ControlFlow::Exit;
                        }
                    }
                    Event::LoopDestroyed => {
                        self.destroy();
                    }
                    Event::DeviceEvent { event, .. } => {
                        if let DeviceEvent::Key(KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        }) = event
                        {
                            self.input_state.update_key_state(*keycode, *state);
                        }
                    }
                    _ => {}
                }
            }
        }

        if self.input_state.is_key_pressed(VirtualKeyCode::Escape) {
            *control_flow = ControlFlow::Exit;
        }
    }

    fn run_frame(&mut self, window: &mut Window) {
        self.renderer.as_mut().unwrap().begin_frame();

        let now = Instant::now();
        let delta_time = now - self.last_frame_time;

        // Audio track control
        if self.input_state.is_key_pressed(VirtualKeyCode::Space)
            && !self.input_state.was_key_pressed(VirtualKeyCode::Space)
        {
            if let Err(err) = self.audio_track.as_mut().unwrap().toggle_pause() {
                println!("Failed to toggle audio track playback: {}", err);
            }
        }
        if !self.audio_track.as_ref().unwrap().is_playing() {
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
                if let Err(err) = self
                    .audio_track
                    .as_mut()
                    .unwrap()
                    .subtract_position_offset(&offset)
                {
                    println!("Failed to rewind audio track: {}", err);
                }
            } else if self.input_state.is_key_pressed(VirtualKeyCode::Right) {
                if let Err(err) = self
                    .audio_track
                    .as_mut()
                    .unwrap()
                    .add_position_offset(&offset)
                {
                    println!("Failed to fast-forward audio track: {}", err);
                }
            }
        }

        self.imgui_context
            .as_mut()
            .unwrap()
            .io_mut()
            .update_delta_time(delta_time);
        self.last_frame_time = now;

        self.timer.push_sample(delta_time);

        self.imgui_platform
            .as_mut()
            .unwrap()
            .prepare_frame(self.imgui_context.as_mut().unwrap().io_mut(), &window)
            .expect("Failed to prepare frame");

        let ui = self.imgui_context.as_mut().unwrap().frame();

        let avg_frame_time_us = self.timer.calculate_average().as_micros() as f64;

        // Render UI
        if let Some(main_menu_bar) = ui.begin_main_menu_bar() {
            if let Some(file_menu) = ui.begin_menu(imgui::im_str!("File"), true) {
                if imgui::MenuItem::new(imgui::im_str!("Exit")).build(&ui) {
                    self.exit_requested = true;
                }

                file_menu.end(&ui);
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
            let audio_pos = self.audio_track.as_ref().unwrap().get_position().unwrap();
            let audio_length = self.audio_track.as_ref().unwrap().get_length();
            ui.text(format!(
                "Track: {} [{}]",
                format_duration(&audio_pos),
                format_duration(&audio_length)
            ));
            main_menu_bar.end(&ui);
        }

        self.imgui_platform
            .as_mut()
            .unwrap()
            .prepare_render(&ui, &window);
        let draw_data = ui.render();

        let cur_swapchain_idx = self.renderer.as_ref().unwrap().get_cur_swapchain_idx();
        let mut render_graph_image = false;
        if let Some(output_image_view) = self
            .graph
            .as_ref()
            .unwrap()
            .get_output_image(cur_swapchain_idx)
        {
            self.renderer
                .as_mut()
                .unwrap()
                .update_graph_image(output_image_view);

            render_graph_image = true;
        }

        let cur_time = self.audio_track.as_ref().unwrap().get_position().unwrap();

        self.renderer
            .as_mut()
            .unwrap()
            .execute_graph(self.graph.as_ref().unwrap(), &cur_time)
            .expect("Failed to execute render graph");

        self.renderer.as_mut().unwrap().begin_render();

        if render_graph_image {
            self.renderer.as_mut().unwrap().render_graph_image();
        }

        self.renderer.as_mut().unwrap().render_ui(draw_data);

        self.renderer.as_mut().unwrap().end_render();

        self.renderer.as_mut().unwrap().end_frame();

        self.input_state.next_frame();
    }

    pub fn run(mut self) -> ! {
        let window_width = 1280;
        let window_height = 720;

        let event_loop = EventLoop::new();
        let mut window = WindowBuilder::new()
            .with_title("Spark Engine Editor")
            .with_inner_size(winit::dpi::PhysicalSize::new(window_width, window_height))
            .build(&event_loop)
            .expect("Failed to create window");

        event_loop.run(move |event, _, control_flow| {
            self.handle_event(&mut window, &event, control_flow);
        });
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
