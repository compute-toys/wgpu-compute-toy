use std::error::Error;
use wgputoy::context::init_wgpu;
use wgputoy::WgpuToyRenderer;

const SHADER: &str = r#"
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + float3(0.,2.,4.));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
"#;

async fn run() -> Result<(), Box<dyn Error>> {
    let wgpu = init_wgpu(800, 600, "").await?;
    let mut wgputoy = WgpuToyRenderer::new(wgpu);
    if let Some(source) = wgputoy.preprocess_async(SHADER).await {
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
