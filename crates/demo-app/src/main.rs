mod camera;

mod mesh;

mod renderer;
use renderer::*;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    log::info!("Initialised logging.");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendering Demo")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();

    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);

    let mut renderer = Renderer::new(&window).unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        // Handle the events first, then pass them on to the renderer.
        if let Event::WindowEvent { ref event, .. } = event {
            if *event == WindowEvent::CloseRequested {
                *control_flow = ControlFlow::Exit;
            }
        }

        renderer.handle_event(&event, &window).unwrap();
    });
}
