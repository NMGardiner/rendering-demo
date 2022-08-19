use rendering_engine::*;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct Renderer {
    frames_in_flight: Vec<FrameResources>,
    swapchain: Swapchain,
    device: Device,
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

    let device = Device::new(&instance, Some(&surface)).unwrap();

    let swapchain = Swapchain::new(
        (window.inner_size().width, window.inner_size().height),
        &instance,
        &surface,
        &device,
        None,
    )
    .unwrap();

    let mut frames_in_flight: Vec<FrameResources> = vec![];
    for _i in 0..MAX_FRAMES_IN_FLIGHT {
        frames_in_flight.push(FrameResources::new(&device).unwrap());
    }

    // Group the renderer components to drop them all at once.
    let renderer = Renderer {
        frames_in_flight,
        swapchain,
        device,
        surface,
        instance,
    };

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

                    for frame in renderer.frames_in_flight.iter() {
                        frame.destroy(&renderer.device);
                    }

                    renderer.swapchain.destroy(&renderer.device);
                }
            }
            _ => (),
        }
    });
}
