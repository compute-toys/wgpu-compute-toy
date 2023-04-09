use std::error::Error;
use wgputoy::context::init_wgpu;
use wgputoy::WgpuToyRenderer;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct ShaderMeta {
    uniforms: Vec<Uniform>,
    textures: Vec<Texture>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Uniform {
}

#[derive(Serialize, Deserialize, Debug)]
struct Texture {
    img: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let wgpu = init_wgpu(800, 600, "").await?;
    let mut wgputoy = WgpuToyRenderer::new(wgpu);

    let filename = if std::env::args().len() > 1 {
        std::env::args().nth(1).unwrap()
    } else {
        "examples/default.wgsl".to_string()
    };
    let shader = std::fs::read_to_string(&filename)?;
    if let Some(source) = wgputoy.preprocess_async(&shader).await {
        wgputoy.compile(source);
    }

    let client = reqwest_middleware::ClientBuilder::new(reqwest::Client::new())
        .with(reqwest_middleware_cache::Cache {
            mode: reqwest_middleware_cache::CacheMode::Default,
            cache_manager: reqwest_middleware_cache::managers::CACacheManager::default(),
        })
        .build();

    if let Ok(json) = std::fs::read_to_string(std::format!("{filename}.json")) {
        let metadata: ShaderMeta = serde_json::from_str(&json)?;
        for (i, texture) in metadata.textures.iter().enumerate() {
            let img = if texture.img.starts_with("http") {
                let resp = client.get(&texture.img).send().await?;
                resp.bytes().await?.to_vec()
            } else {
                std::fs::read(&std::format!("site/public/{}", texture.img))?
            };
            if texture.img.ends_with(".hdr") {
                wgputoy.load_channel_hdr(i, &img)?;
            } else {
                wgputoy.load_channel(i, &img);
            }
        }
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
