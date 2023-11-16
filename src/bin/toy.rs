use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(any(target_arch = "wasm32", not(feature = "winit")))]
    return Err("must be compiled with winit feature to run".into());

    #[cfg(all(not(target_arch = "wasm32"), feature = "winit"))]
    return winit::main();
}

#[cfg(all(not(target_arch = "wasm32"), feature = "winit"))]
mod winit {
    use std::error::Error;
    use wgputoy::context::init_wgpu;
    use wgputoy::shader::{FolderLoader, WebLoader, load_shader};
    use wgputoy::WgpuToyRenderer;

    async fn init() -> Result<WgpuToyRenderer, Box<dyn Error>> {
        let name = if std::env::args().len() > 1 {
            std::env::args().nth(1).unwrap()
        } else {
            "default".to_string()
        };

        let source_loader = FolderLoader::new("./examples".to_string());
        let texture_loader = WebLoader::new();

        let shader = load_shader(&source_loader, &texture_loader, &name)?;

        let wgpu = init_wgpu(1280, 720, "").await?;
        let mut wgputoy = WgpuToyRenderer::new(wgpu);

        wgputoy.load_shader(shader).await?;

        Ok(wgputoy)
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let mut wgputoy = runtime.block_on(init())?;
        let screen_size = wgputoy.wgpu.window.inner_size();
        let start_time = std::time::Instant::now();
        let event_loop = std::mem::take(&mut wgputoy.wgpu.event_loop).unwrap();
        let device_clone = wgputoy.wgpu.device.clone();
        std::thread::spawn(move || loop {
            device_clone.poll(wgpu::Maintain::Wait);
        });
        event_loop.run(move |event, _, control_flow| {
            *control_flow = winit::event_loop::ControlFlow::Poll;
            match event {
                winit::event::Event::RedrawRequested(_) => {
                    let time = start_time.elapsed().as_micros() as f32 * 1e-6;
                    wgputoy.set_time_elapsed(time);
                    let future = wgputoy.render_async();
                    runtime.block_on(future);
                }
                winit::event::Event::MainEventsCleared => {
                    wgputoy.wgpu.window.request_redraw();
                }
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::CloseRequested,
                    ..
                } => *control_flow = winit::event_loop::ControlFlow::Exit,
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::CursorMoved { position, .. },
                    ..
                } => wgputoy.set_mouse_pos(
                    position.x as f32 / screen_size.width as f32,
                    position.y as f32 / screen_size.height as f32,
                ),
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::Resized(size),
                    ..
                } => {
                    if size.width != 0 && size.height != 0 {
                        wgputoy.resize(size.width, size.height, 1.);
                    }
                }
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::MouseInput { state, .. },
                    ..
                } => wgputoy.set_mouse_click(state == winit::event::ElementState::Pressed),
                _ => (),
            }
        });
    }
}
