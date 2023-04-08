use std::error::Error;
use wgputoy::context::init_wgpu;
use wgputoy::WgpuToyRenderer;

async fn run() -> Result<(), Box<dyn Error>> {
    let wgpu = init_wgpu(800, 600, "").await?;
    let mut wgputoy = WgpuToyRenderer::new(wgpu);
    let filename = if std::env::args().len() > 1 {
        std::env::args().nth(1).unwrap()
    } else {
        "examples/default.wgsl".to_string()
    };
    let shader = std::fs::read_to_string(filename)?;
    if let Some(source) = wgputoy.preprocess_async(&shader).await {
        wgputoy.compile(source);
    }
    let start_time = std::time::Instant::now();
    let event_loop = std::mem::take(&mut wgputoy.wgpu.event_loop).unwrap();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::RedrawRequested(_) => {
                let time = start_time.elapsed().as_micros() as f32 * 1e-6;
                wgputoy.set_time_elapsed(time);
                let future = wgputoy.render_async();
                let executor = async_executor::LocalExecutor::new();
                executor.spawn(future).detach();
                while executor.try_tick() {}
            }
            winit::event::Event::MainEventsCleared => {
                wgputoy.wgpu.window.request_redraw();
            }
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                ..
            } => *control_flow = winit::event_loop::ControlFlow::Exit,
            _ => (),
        }
    });
}

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}
