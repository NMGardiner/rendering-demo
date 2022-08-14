use rendering_engine::*;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct Renderer {
    surface: Surface,
    instance: Instance,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    log::info!("Initialised logging.");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendering Demo")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();

    let instance = Instance::builder()
        .application_name("Rendering Demo")
        .application_version(0, 1, 0)
        .window_handle(&window)
        .enable_validation_layers(cfg!(debug_assertions))
        .build()
        .unwrap();

    let surface = Surface::new(&window, &instance).unwrap();

    // Group the renderer components to drop them all at once.
    let renderer = Renderer { surface, instance };

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => {
                // Draw a frame.
            }
            Event::WindowEvent { event, .. } => {
                // Handle the window event.

                if event == WindowEvent::CloseRequested {
                    *control_flow = ControlFlow::Exit;

                    // Drop the renderer.
                    let _ = &renderer;
                }
            }
            _ => (),
        }
    });
}
