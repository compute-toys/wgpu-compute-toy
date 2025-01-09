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
    use std::{
        error::Error,
        path::Path,
        sync::atomic::{AtomicBool, Ordering},
    };
    use wgputoy::context::init_wgpu;
    use wgputoy::WgpuToyRenderer;
    use winit::{
        event::{ElementState, Event, KeyEvent, WindowEvent},
        event_loop::ControlFlow,
        keyboard::{KeyCode, PhysicalKey},
    };

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

    const APPLICATION_TITLE: &str = "WgpuToy";
    const APPLICATION_TITLE_PAUSED: &str = "WgpuToy - Paused";

    async fn init(filename: &str) -> Result<WgpuToyRenderer, Box<dyn Error>> {
        let wgpu = init_wgpu(1280, 720, "").await?;
        let mut wgputoy = WgpuToyRenderer::new(wgpu);
        let shader = std::fs::read_to_string(filename)?;

        wgputoy.wgpu.window.set_title(APPLICATION_TITLE);

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

    static NEEDS_REBUILD: AtomicBool = AtomicBool::new(false);

    pub fn main() -> Result<(), Box<dyn Error>> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let filename = if std::env::args().len() > 1 {
            std::env::args().nth(1).unwrap()
        } else {
            "examples/default.wgsl".to_string()
        };

        let mut wgputoy = runtime.block_on(init(&filename))?;
        let screen_size = wgputoy.wgpu.window.inner_size();
        let event_loop = std::mem::take(&mut wgputoy.wgpu.event_loop).unwrap();
        let device_clone = wgputoy.wgpu.device.clone();
        std::thread::spawn(move || loop {
            device_clone.poll(wgpu::Maintain::Wait);
            std::thread::yield_now();
        });

        let mut watcher;
        'watch: {
            use notify::{RecursiveMode, Result, Watcher};

            let event_loop_proxy = event_loop.create_proxy();
            let watcher_res =
                notify::recommended_watcher(move |event: Result<notify::Event>| match event {
                    Ok(_) => {
                        let res = NEEDS_REBUILD.store(true, Ordering::Relaxed);
                        // for the event loop to run
                        event_loop_proxy.send_event(()).unwrap();
                        res
                    }
                    Err(err) => log::error!("Error watching file: {err}"),
                });

            watcher = match watcher_res {
                Ok(watcher) => watcher,
                Err(err) => {
                    log::error!("Error creating watcher: {err:?}");
                    break 'watch;
                }
            };

            let path = Path::new(&filename);
            if let Err(err) = watcher.watch(path, RecursiveMode::NonRecursive) {
                log::error!("Error watching file: {:?}", err);
                break 'watch;
            }

            log::info!("Watching file: {path:?}");
        }

        let mut close_requested = false;
        let mut paused = false;
        let mut current_instant = std::time::Instant::now();
        let mut reference_time = 0.0; // to handle pause and resume

        let _ = event_loop.run(move |event, elwt| {
            if NEEDS_REBUILD.swap(false, Ordering::Relaxed) {
                let shader = std::fs::read_to_string(&filename).unwrap();
                if let Some(source) = runtime.block_on(wgputoy.preprocess_async(&shader)) {
                    // println!("{}", source.source); // commented as it's annoying to have the file source bloat the console, maybe it should be a debug log ?
                    wgputoy.compile(source);
                    // force redraw to update the shader with the new changes
                    wgputoy.wgpu.window.request_redraw();
                }
            };

            match event {
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
                        wgputoy.reset();
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
                            wgputoy.wgpu.window.set_title(APPLICATION_TITLE);
                        } else {
                            reference_time =
                                reference_time + current_instant.elapsed().as_secs_f32();
                            wgputoy.wgpu.window.set_title(APPLICATION_TITLE_PAUSED);
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
                    if !paused {
                        wgputoy.wgpu.window.request_redraw();
                    }
                    if close_requested {
                        elwt.exit();
                    }
                }
                _ => (),
            }
        });
        Ok(())
    }
}
