use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "winit"))]
    return Err("must be compiled with winit feature to run".into());

    #[cfg(all(feature = "winit", not(target_arch = "wasm32")))]
    return winit::main();

    #[cfg(all(feature = "winit", target_arch = "wasm32"))]
    return Err("winit not supported on wasm target".into());
}

#[cfg(all(feature = "winit", not(target_arch = "wasm32")))]
mod winit {
    use http_cache_reqwest::{CACacheManager, Cache, CacheMode, HttpCache, HttpCacheOptions};
    use serde::{Deserialize, Serialize};
    use std::error::Error;
    use wgputoy::context::init_wgpu;
    use wgputoy::WgpuToyRenderer;
    use winit::{
        event::{ElementState, Event, KeyEvent, WindowEvent},
        event_loop::ControlFlow,
        keyboard::{KeyCode, PhysicalKey},
    };

    #[cfg(not(wasm_platform))]
    use std::time;
    #[cfg(wasm_platform)]
    use web_time as time;

    const POLL_SLEEP_TIME: time::Duration = time::Duration::from_millis(100);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Poll,
    }

    #[derive(Serialize, Deserialize, Debug)]
    #[serde(rename_all = "camelCase")]
    struct ShaderMeta {
        uniforms: Vec<Uniform>,
        textures: Vec<Texture>,
        #[serde(default)]
        float32_enabled: bool,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct Uniform {
        name: String,
        value: f32,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct Texture {
        img: String,
    }

    async fn init() -> Result<WgpuToyRenderer, Box<dyn Error>> {
        let wgpu = init_wgpu(1280, 720, "").await?;
        let mut wgputoy = WgpuToyRenderer::new(wgpu);

        let filename = if std::env::args().len() > 1 {
            std::env::args().nth(1).unwrap()
        } else {
            "examples/default.wgsl".to_string()
        };
        let shader = std::fs::read_to_string(&filename)?;

        let client = reqwest_middleware::ClientBuilder::new(reqwest::Client::new())
            .with(Cache(HttpCache {
                mode: CacheMode::Default,
                manager: CACacheManager::default(),
                options: HttpCacheOptions::default(),
            }))
            .build();

        if let Ok(json) = std::fs::read_to_string(std::format!("{filename}.json")) {
            let metadata: ShaderMeta = serde_json::from_str(&json)?;
            println!("{:?}", metadata);

            for (i, texture) in metadata.textures.iter().enumerate() {
                let url = if texture.img.starts_with("http") {
                    texture.img.clone()
                } else {
                    std::format!("https://compute.toys/{}", texture.img)
                };
                let resp = client.get(&url).send().await?;
                let img = resp.bytes().await?.to_vec();
                if texture.img.ends_with(".hdr") {
                    wgputoy.load_channel_hdr(i, &img)?;
                } else {
                    wgputoy.load_channel(i, &img);
                }
            }

            let uniform_names: Vec<String> =
                metadata.uniforms.iter().map(|u| u.name.clone()).collect();
            let uniform_values: Vec<f32> = metadata.uniforms.iter().map(|u| u.value).collect();
            if !uniform_names.is_empty() {
                wgputoy.set_custom_floats(uniform_names, uniform_values);
            }

            wgputoy.set_pass_f32(metadata.float32_enabled);
        }

        if let Some(source) = wgputoy.preprocess_async(&shader).await {
            println!("{}", source.source);
            wgputoy.compile(source);
        }
        Ok(wgputoy)
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let mut wgputoy = runtime.block_on(init())?;
        let screen_size = wgputoy.wgpu.window.inner_size();
        let event_loop = std::mem::take(&mut wgputoy.wgpu.event_loop).unwrap();
        let device_clone = wgputoy.wgpu.device.clone();
        std::thread::spawn(move || loop {
            device_clone.poll(wgpu::Maintain::Wait);
        });

        let mode = Mode::Poll;
        let mut close_requested = false;
        let mut paused = false;
        let mut current_instant = std::time::Instant::now();
        let mut reference_time = 0.0; // to handle pause and resume

        let filename = if std::env::args().len() > 1 {
            std::env::args().nth(1).unwrap()
        } else {
            "examples/default.wgsl".to_string()
        };

        // for file watching
        let mut last_modified = std::fs::metadata(&filename)?.modified()?;
        let mut last_check = std::time::Instant::now();
        let check_interval = std::time::Duration::from_secs(2);

        let _ = event_loop.run(move |event, elwt| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    device_id: _,
                    event:
                        KeyEvent {
                            state: ElementState::Released,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => {
                    close_requested = true;
                }
                WindowEvent::KeyboardInput {
                    device_id: _,
                    event:
                        KeyEvent {
                            state: ElementState::Released,
                            physical_key: PhysicalKey::Code(KeyCode::Backspace),
                            ..
                        },
                    ..
                } => {
                    // reset time
                    paused = false;
                    reference_time = 0.0;
                    current_instant = std::time::Instant::now();
                    println!("reset time");
                }
                WindowEvent::KeyboardInput {
                    device_id: _,
                    event:
                        KeyEvent {
                            state: ElementState::Released,
                            physical_key: PhysicalKey::Code(KeyCode::Space),
                            ..
                        },
                    ..
                } => {
                    // toggle pause and reset time
                    paused = !paused;
                    if !paused {
                        current_instant = std::time::Instant::now();
                    } else {
                        reference_time = reference_time + current_instant.elapsed().as_secs_f32();
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    wgputoy.set_mouse_pos(
                        position.x as f32 / screen_size.width as f32,
                        position.y as f32 / screen_size.height as f32,
                    );
                }
                WindowEvent::MouseInput { state, .. } => {
                    wgputoy.set_mouse_click(state == ElementState::Pressed);
                }
                WindowEvent::Resized(size) => {
                    if size.width != 0 && size.height != 0 {
                        wgputoy.resize(size.width, size.height, 1.);
                    }
                }
                WindowEvent::RedrawRequested => {
                    if !paused {
                        let time = reference_time + current_instant.elapsed().as_secs_f32();
                        wgputoy.set_time_elapsed(time);
                    }
                    let future = wgputoy.render_async();
                    runtime.block_on(future);
                }
                _ => (),
            },
            Event::AboutToWait => {
                // Check for file changes at a specific interval once every second or two is probably enough
                if last_check.elapsed() >= check_interval {
                    if let Ok(metadata) = std::fs::metadata(&filename) {
                        if let Ok(modified) = metadata.modified() {
                            if modified > last_modified {
                                println!("file {} changed, reloading shader", filename);
                                if let Ok(shader) = std::fs::read_to_string(&filename) {
                                    if let Some(source) =
                                        runtime.block_on(wgputoy.preprocess_async(&shader))
                                    {
                                        println!("{}", source.source);
                                        wgputoy.compile(source);

                                        // even in paused mode, we want to redraw to see the changes
                                        wgputoy.wgpu.window.request_redraw();
                                    }
                                }
                                last_modified = modified;
                            }
                        }
                    }
                    last_check = std::time::Instant::now();
                }

                if !paused {
                    wgputoy.wgpu.window.request_redraw();
                }
                match mode {
                    Mode::Poll => {
                        std::thread::sleep(POLL_SLEEP_TIME);
                        elwt.set_control_flow(ControlFlow::Poll);
                    }
                    _ => (),
                };

                if close_requested {
                    elwt.exit();
                }
            }
            _ => (),
        });
        Ok(())
    }
}
