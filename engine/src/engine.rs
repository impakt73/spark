use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::time::Instant;
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::renderer::Renderer;

pub struct Engine {
    imgui_context: Option<imgui::Context>,
    imgui_platform: Option<WinitPlatform>,
    renderer: Option<Renderer>,
    last_frame_time: Instant,
    exit_requested: bool,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            imgui_context: None,
            imgui_platform: None,
            renderer: None,
            last_frame_time: Instant::now(),
            exit_requested: false,
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

        let renderer =
            Renderer::new(&window, true, &mut context).expect("Failed to create renderer");

        self.imgui_context = Some(context);
        self.imgui_platform = Some(platform);
        self.renderer = Some(renderer);
    }

    fn destroy(&mut self) {
        self.renderer.as_mut().unwrap().wait_for_idle();
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
                    event => match event {
                        Event::DeviceEvent { event, .. } => match event {
                            DeviceEvent::Key(KeyboardInput {
                                virtual_keycode: Some(keycode),
                                state,
                                ..
                            }) => match (keycode, state) {
                                (VirtualKeyCode::Escape, ElementState::Released) => {
                                    *control_flow = ControlFlow::Exit
                                }
                                _ => (),
                            },
                            _ => (),
                        },
                        _ => {}
                    },
                }
            }
        }
    }

    fn run_frame(&mut self, window: &mut Window) {
        self.renderer.as_mut().unwrap().begin_frame();

        let now = Instant::now();
        self.imgui_context
            .as_mut()
            .unwrap()
            .io_mut()
            .update_delta_time(now - self.last_frame_time);
        self.last_frame_time = now;

        self.imgui_platform
            .as_mut()
            .unwrap()
            .prepare_frame(self.imgui_context.as_mut().unwrap().io_mut(), &window)
            .expect("Failed to prepare frame");

        let ui = self.imgui_context.as_mut().unwrap().frame();

        self.renderer.as_mut().unwrap().begin_render();

        // Render UI
        if let Some(main_menu_bar) = ui.begin_main_menu_bar() {
            if let Some(file_menu) = ui.begin_menu(imgui::im_str!("File"), true) {
                if imgui::MenuItem::new(imgui::im_str!("Exit")).build(&ui) {
                    self.exit_requested = true;
                }

                file_menu.end(&ui);
            }
            main_menu_bar.end(&ui);
        }

        self.imgui_platform
            .as_mut()
            .unwrap()
            .prepare_render(&ui, &window);
        let draw_data = ui.render();

        self.renderer.as_mut().unwrap().render_ui(draw_data);

        self.renderer.as_mut().unwrap().end_render();

        self.renderer.as_mut().unwrap().end_frame();
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
