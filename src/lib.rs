mod bind;
mod blit;
pub mod context;
mod utils;

use context::WgpuContext;
use naga::front::wgsl;
use num::Integer;
use std::collections::HashMap;
use std::mem::{size_of, take};
use std::sync::atomic::{AtomicBool, Ordering};
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[derive(Clone)]
struct ErrorCallback(Option<js_sys::Function>);

impl ErrorCallback {
    fn call(&self, summary: &str, row: usize, col: usize) {
        match self.0 {
            None => log::error!("No error callback registered"),
            Some(ref callback) => {
                let res = callback.call3(
                    &JsValue::NULL,
                    &JsValue::from(summary),
                    &JsValue::from(row),
                    &JsValue::from(col),
                );
                match res {
                    Err(error) => log::error!("Error calling registered error callback: {error:?}"),
                    _ => (),
                };
            }
        }
    }
}

#[derive(Clone)]
struct SuccessCallback(Option<js_sys::Function>);

impl SuccessCallback {
    fn call(&self, entry_points: Vec<String>) {
        match self.0 {
            None => log::error!("No success callback registered"),
            Some(ref callback) => {
                let res = callback.call1(
                    &JsValue::NULL,
                    &JsValue::from(
                        entry_points
                            .into_iter()
                            .map(JsValue::from)
                            .collect::<js_sys::Array>(),
                    ),
                );
                match res {
                    Err(error) => log::error!("Error calling registered error callback: {error:?}"),
                    _ => (),
                };
            }
        }
    }
}

// safe because wasm is single-threaded: https://github.com/rustwasm/wasm-bindgen/issues/1505
unsafe impl Send for ErrorCallback {}
unsafe impl Sync for ErrorCallback {}

struct ComputePipeline {
    name: String,
    workgroup_size: [u32; 3],
    thread_count: Option<[u32; 3]>,
    dispatch_count: u8,
    pipeline: wgpu::ComputePipeline,
}

#[wasm_bindgen]
pub struct WgpuToyRenderer {
    #[wasm_bindgen(skip)]
    pub wgpu: WgpuContext,
    screen_width: u32,
    screen_height: u32,
    thread_count: HashMap<String, [u32; 3]>,
    dispatch_count: HashMap<String, u8>,
    bindings: bind::Bindings,
    compute_pipeline_layout: wgpu::PipelineLayout,
    last_compute_pipelines: Option<Vec<ComputePipeline>>,
    compute_pipelines: Vec<ComputePipeline>,
    compute_bind_group: wgpu::BindGroup,
    staging_belt: wgpu::util::StagingBelt,
    on_error_cb: ErrorCallback,
    on_success_cb: SuccessCallback,
    pass_f32: bool,
    screen_blitter: blit::Blitter,
    query_set: Option<wgpu::QuerySet>,
    last_stats: instant::Instant,
}

static SHADER_ERROR: AtomicBool = AtomicBool::new(false);

// https://llogiq.github.io/2016/09/24/newline.html
fn count_newlines(s: &str) -> usize {
    s.as_bytes().iter().filter(|&&c| c == b'\n').count()
}

#[wasm_bindgen]
impl WgpuToyRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(wgpu: WgpuContext) -> WgpuToyRenderer {
        let size = wgpu.window.inner_size();
        let bindings = bind::Bindings::new(&wgpu, size.width, size.height, false);

        WgpuToyRenderer {
            compute_pipeline_layout: bindings.create_pipeline_layout(&wgpu),
            compute_bind_group: bindings.create_bind_group(&wgpu),
            last_compute_pipelines: None,
            compute_pipelines: vec![],
            screen_width: size.width,
            screen_height: size.height,
            staging_belt: wgpu::util::StagingBelt::new(4096),
            screen_blitter: blit::Blitter::new(
                &wgpu,
                &bindings.tex_screen.view(),
                blit::ColourSpace::Linear,
                wgpu.surface_format,
                wgpu::FilterMode::Nearest,
            ),
            wgpu,
            bindings,
            on_error_cb: ErrorCallback(None),
            on_success_cb: SuccessCallback(None),
            thread_count: HashMap::new(),
            dispatch_count: HashMap::new(),
            pass_f32: false,
            query_set: None,
            last_stats: instant::Instant::now(),
        }
    }

    pub fn render(&mut self) {
        match self.wgpu.surface.get_current_texture() {
            Err(e) => log::error!("Unable to get framebuffer: {e}"),
            Ok(f) => self.render_to(f),
        }
    }

    fn render_to(&mut self, frame: wgpu::SurfaceTexture) {
        let stats_period = 100;
        let mut encoder = self.wgpu.device.create_command_encoder(&Default::default());
        self.bindings
            .stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder);
        if self.bindings.time.host.frame % stats_period == 0 {
            //encoder.clear_buffer(&self.uniforms.debug_buffer, 0, None); // not yet implemented in web backend
            self.staging_belt
                .write_buffer(
                    &mut encoder,
                    &self.bindings.debug_buffer.buffer(),
                    0,
                    wgpu::BufferSize::new(size_of::<[u32; bind::NUM_ASSERT_COUNTERS]>() as u64)
                        .unwrap(),
                    &self.wgpu.device,
                )
                .copy_from_slice(&bytemuck::bytes_of(&[0u32; bind::NUM_ASSERT_COUNTERS]));

            if self.bindings.time.host.frame > 0 {
                let mean = self.last_stats.elapsed().as_secs_f32() / stats_period as f32;
                self.last_stats = instant::Instant::now();
                log::info!("{} fps ({} ms)", 1. / mean, 1e3 * mean);
            }
        }
        self.staging_belt.finish();
        if SHADER_ERROR.swap(false, Ordering::SeqCst) {
            match take(&mut self.last_compute_pipelines) {
                None => log::warn!("unable to rollback shader after error"),
                Some(vec) => {
                    self.compute_pipelines = vec;
                }
            }
        }
        let mut dispatch_counter = 0;
        for (pass_index, p) in self.compute_pipelines.iter().enumerate() {
            for i in 0..p.dispatch_count as u32 {
                let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                if let Some(q) = &self.query_set {
                    compute_pass.write_timestamp(q, 2 * pass_index as u32);
                }
                let thread_count =
                    p.thread_count
                        .unwrap_or([self.screen_width, self.screen_height, 1]);
                compute_pass.set_pipeline(&p.pipeline);
                self.wgpu.queue.write_buffer(
                    &self.bindings.dispatch_info.buffer(),
                    bind::OFFSET_ALIGNMENT as u64 * dispatch_counter,
                    bytemuck::bytes_of(&i),
                );
                compute_pass.set_bind_group(
                    0,
                    &self.compute_bind_group,
                    &[bind::OFFSET_ALIGNMENT as u32 * dispatch_counter as u32],
                );
                dispatch_counter += 1;
                compute_pass.dispatch(
                    thread_count[0].div_ceil(&p.workgroup_size[0]),
                    thread_count[1].div_ceil(&p.workgroup_size[1]),
                    thread_count[2].div_ceil(&p.workgroup_size[2]),
                );
                if let Some(q) = &self.query_set {
                    compute_pass.write_timestamp(q, 2 * pass_index as u32 + 1);
                }
                drop(compute_pass);
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.bindings.tex_write.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &self.bindings.tex_read.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.screen_width,
                        height: self.screen_height,
                        depth_or_array_layers: 4,
                    },
                );
            }
        }
        let mut staging_buffer = None;
        let query_offset = bind::NUM_ASSERT_COUNTERS * size_of::<u32>();
        let query_count = 2 * self.compute_pipelines.len();
        if self.bindings.time.host.frame % stats_period == stats_period - 1 {
            let buf = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (query_offset + query_count * size_of::<u64>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                &self.bindings.debug_buffer.buffer(),
                0,
                &buf,
                0,
                query_offset as wgpu::BufferAddress,
            );
            if let Some(q) = &self.query_set {
                encoder.resolve_query_set(
                    q,
                    0..query_count as u32,
                    &buf,
                    query_offset as wgpu::BufferAddress,
                );
            }
            staging_buffer = Some(buf);
        }
        self.bindings.time.host.frame += 1;
        self.screen_blitter.blit(
            &mut encoder,
            &frame.texture.create_view(&Default::default()),
        );
        self.wgpu.queue.submit(Some(encoder.finish()));
        if let Some(buf) = staging_buffer {
            self.wgpu.device.poll(wgpu::Maintain::Wait);
            let numthreads = self.screen_width * self.screen_height;
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(async move {
                let buffer_slice = buf.slice(..);
                match buffer_slice.map_async(wgpu::MapMode::Read).await {
                    Err(e) => log::error!("{e}"),
                    Ok(()) => {
                        let data = buffer_slice.get_mapped_range();
                        let assertions: &[u32] = bytemuck::cast_slice(&data[0..query_offset]);
                        let timestamps: &[u64] = bytemuck::cast_slice(&data[query_offset..]);
                        for (i, count) in assertions.iter().enumerate() {
                            if count > &0 {
                                let percent =
                                    *count as f32 / (numthreads * stats_period) as f32 * 100.0;
                                log::warn!("Assertion {i} failed in {percent}% of threads");
                            }
                        }
                    }
                }
                buf.unmap();
            });
        }
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(self.staging_belt.recall());
        frame.present();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let executor = async_executor::LocalExecutor::new();
            executor.spawn(self.staging_belt.recall()).detach();
            executor.try_tick();
        }
    }

    pub fn prelude(&self) -> String {
        let mut s = String::new();
        for (a, t) in [("int", "i32"), ("uint", "u32"), ("float", "f32")] {
            s.push_str(&format!("type {a} = {t};\n"));
            for n in [2, 3, 4] {
                s.push_str(&format!("type {a}{n} = vec{n}<{t}>;\n"));
            }
        }
        s.push_str(
            r#"
struct Time { frame: uint, elapsed: float };
struct Mouse { pos: uint2, click: int };
"#,
        );
        s.push_str("struct Custom {\n");
        let (custom_names, _) = &self.bindings.custom.host;
        for name in custom_names {
            s.push_str("    ");
            s.push_str(name);
            s.push_str(": float,\n");
        }
        s.push_str("};\n");
        s.push_str("struct Data {");
        for (key, val) in self.bindings.user_data.host.iter() {
            let n = val.len();
            s.push_str(&format!("{key}: array<u32,{n}>,"));
        }
        s.push_str("};\n");
        s.push_str(&self.bindings.to_wgsl());
        s.push_str(
            r#"
fn keyDown(keycode: uint) -> bool {
    return ((_keyboard[keycode / 128u][(keycode % 128u) / 32u] >> (keycode % 32u)) & 1u) == 1u;
}

fn assert(index: int, success: bool) {
    if (!success) {
        atomicAdd(&_assert_counts[index], 1u);
    }
}

fn passStore(pass: int, coord: int2, value: float4) {
    textureStore(pass_out, coord, pass, value);
}

fn passLoad(pass: int, coord: int2, lod: int) -> float4 {
    return textureLoad(pass_in, coord, pass, lod);
}

fn passSampleLevelBilinearRepeat(pass: int, uv: float2, lod: float) -> float4 {"#,
        );
        if self.pass_f32 {
            // https://iquilezles.org/articles/hwinterpolation/
            s.push_str(
                r#"
    let res = float2(textureDimensions(pass_in));
    let st = uv * res - 0.5;
    let iuv = floor(st);
    let fuv = fract(st);
    let a = textureSampleLevel(pass_in, nearest, fract((iuv + float2(0.5,0.5)) / res), pass, lod);
    let b = textureSampleLevel(pass_in, nearest, fract((iuv + float2(1.5,0.5)) / res), pass, lod);
    let c = textureSampleLevel(pass_in, nearest, fract((iuv + float2(0.5,1.5)) / res), pass, lod);
    let d = textureSampleLevel(pass_in, nearest, fract((iuv + float2(1.5,1.5)) / res), pass, lod);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
"#,
            );
        } else {
            s.push_str(
                r#"
    return textureSampleLevel(pass_in, bilinear, fract(uv), pass, lod);
"#,
            );
        }
        s.push_str("}");
        return s;
    }

    fn handle_success(&self, entry_points: Vec<String>) {
        #[cfg(target_arch = "wasm32")]
        self.on_success_cb.call(entry_points);
    }

    fn preprocess(&mut self, shader: &str) -> Option<(String, Vec<usize>)> {
        self.bindings.user_data.host = HashMap::from([("_dummy".into(), vec![0])]); // clear
        self.thread_count.clear();
        let mut sourcemap = vec![0];
        let mut wgsl = String::new();
        let mut push_line = |n, s| {
            sourcemap.push(n);
            wgsl.push_str(s);
            wgsl.push_str("\n");
        };
        for (line, n) in shader.lines().zip(1..) {
            if line.chars().nth(0) == Some('#') {
                let tokens: Vec<&str> = line.split(" ").collect();
                match tokens[..] {
                    _ => {
                        self.on_error_cb
                            .call("Unrecognised preprocessor directive", n, 1);
                        return None;
                    }
                }
            } else {
                push_line(n, line);
            }
        }
        Some((wgsl, sourcemap))
    }

    pub fn set_shader(&mut self, shader: &str) {
        let now = instant::Instant::now();
        if let Some((source, sourcemap)) = self.preprocess(shader) {
            let prelude = self.prelude(); // prelude must be generated after preprocessor has run

            // FIXME: remove pending resolution of this issue: https://github.com/gfx-rs/wgpu/issues/2130
            let prelude_len = count_newlines(&prelude);
            let re_parser = lazy_regex::regex!(r"(?s):(\d+):(\d+) (.*)");
            let re_invalid = lazy_regex::regex!(r"\[Invalid \w+\] is invalid.");
            let on_error_cb = self.on_error_cb.clone();
            let sourcemap_clone = sourcemap.clone();
            self.wgpu.device.on_uncaptured_error(move |e: wgpu::Error| {
                let err = &e.to_string();
                if re_invalid.is_match(err) {
                    return;
                }
                match re_parser.captures(err) {
                    None => {
                        log::error!("{e}");
                        on_error_cb.call(err, 0, 0);
                    }
                    Some(cap) => {
                        let row = cap[1].parse().unwrap_or(prelude_len);
                        let col = cap[2].parse().unwrap_or(0);
                        let summary = &cap[3];
                        let mut n = 0;
                        if row >= prelude_len {
                            n = row - prelude_len;
                        }
                        if n < sourcemap_clone.len() {
                            n = sourcemap_clone[n];
                        }
                        on_error_cb.call(summary, n, col);
                        SHADER_ERROR.store(true, Ordering::SeqCst);
                    }
                }
            });

            let wgsl = prelude + &source;
            match wgsl::parse_str(&wgsl) {
                Ok(module) => {
                    let entry_points: Vec<_> = module
                        .entry_points
                        .iter()
                        .filter(|f| f.stage == naga::ShaderStage::Compute)
                        .collect();
                    let entry_point_names: Vec<String> = entry_points
                        .iter()
                        .map(|entry_point| entry_point.name.clone())
                        .collect();
                    self.handle_success(entry_point_names);
                    let compute_shader =
                        self.wgpu
                            .device
                            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                                label: None,
                                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&wgsl)),
                            });
                    self.last_compute_pipelines = Some(take(&mut self.compute_pipelines));
                    self.compute_pipelines = entry_points
                        .iter()
                        .map(|entry_point| ComputePipeline {
                            name: entry_point.name.clone(),
                            workgroup_size: entry_point.workgroup_size,
                            thread_count: self
                                .thread_count
                                .get(&entry_point.name)
                                .map(|t| t.clone()),
                            dispatch_count: *self
                                .dispatch_count
                                .get(&entry_point.name)
                                .unwrap_or(&1),
                            pipeline: self.wgpu.device.create_compute_pipeline(
                                &wgpu::ComputePipelineDescriptor {
                                    label: None,
                                    layout: Some(&self.compute_pipeline_layout),
                                    module: &compute_shader,
                                    entry_point: &entry_point.name,
                                },
                            ),
                        })
                        .collect();
                    self.query_set = if !self
                        .wgpu
                        .device
                        .features()
                        .contains(wgpu::Features::TIMESTAMP_QUERY)
                    {
                        None
                    } else {
                        Some(
                            self.wgpu
                                .device
                                .create_query_set(&wgpu::QuerySetDescriptor {
                                    label: None,
                                    count: 2 * self.compute_pipelines.len() as u32,
                                    ty: wgpu::QueryType::Timestamp,
                                }),
                        )
                    };
                    log::info!(
                        "Shader compiled in {}s",
                        now.elapsed().as_micros() as f32 * 1e-6
                    );
                }
                Err(e) => {
                    log::error!("Error parsing WGSL: {e}");
                    let (row, col) = e.location(&wgsl);
                    let summary = e.emit_to_string(&wgsl);
                    let mut n = 0;
                    if row >= prelude_len {
                        n = row - prelude_len;
                    }
                    if n < sourcemap.len() {
                        n = sourcemap[n];
                    }
                    self.on_error_cb.call(&summary, n, col);
                }
            }
        }
    }

    pub fn set_time_elapsed(&mut self, t: f32) {
        self.bindings.time.host.elapsed = t;
    }

    pub fn set_mouse_pos(&mut self, x: u32, y: u32) {
        self.bindings.mouse.host.pos = [x, y];
    }

    pub fn set_mouse_click(&mut self, click: bool) {
        self.bindings.mouse.host.click = if click { 1 } else { 0 };
    }

    pub fn set_keydown(&mut self, keycode: usize, keydown: bool) {
        self.bindings.keys.host.set(keycode, keydown);
    }

    pub fn set_custom_floats(&mut self, names: Vec<js_sys::JsString>, values: Vec<f32>) {
        self.bindings.custom.host = (names.iter().map(From::from).collect(), values);
    }

    pub fn set_pass_f32(&mut self, pass_f32: bool) {
        self.pass_f32 = pass_f32;
        self.reset();
    }

    pub fn resize(&mut self, width: u32, height: u32, scale: f32) {
        self.screen_width = (width as f32 * scale) as u32;
        self.screen_height = (height as f32 * scale) as u32;
        self.reset();
        self.wgpu
            .window
            .set_inner_size(winit::dpi::PhysicalSize::new(width, height));
    }

    pub fn reset(&mut self) {
        let mut bindings = bind::Bindings::new(
            &self.wgpu,
            self.screen_width,
            self.screen_height,
            self.pass_f32,
        );
        std::mem::swap(&mut self.bindings, &mut bindings);
        self.bindings.custom.host = bindings.custom.host.clone();
        self.bindings.user_data.host = bindings.user_data.host.clone();
        self.bindings.channels = take(&mut bindings.channels);
        self.compute_pipeline_layout = self.bindings.create_pipeline_layout(&self.wgpu);
        self.compute_bind_group = self.bindings.create_bind_group(&self.wgpu);
        self.screen_blitter = blit::Blitter::new(
            &self.wgpu,
            &self.bindings.tex_screen.view(),
            blit::ColourSpace::Linear,
            self.wgpu.surface_format,
            wgpu::FilterMode::Linear,
        );
    }

    pub fn on_error(&mut self, callback: js_sys::Function) {
        self.on_error_cb = ErrorCallback(Some(callback));
    }

    pub fn on_success(&mut self, callback: js_sys::Function) {
        self.on_success_cb = SuccessCallback(Some(callback));
    }

    pub fn load_channel(&mut self, index: usize, bytes: &[u8]) {
        let now = instant::Instant::now();
        match image::load_from_memory(bytes) {
            Err(e) => log::error!("load_channel: {e}"),
            Ok(im) => {
                use image::GenericImageView;
                let (width, height) = im.dimensions();
                self.bindings.channels[index].set_texture(
                    blit::Blitter::new(
                        &self.wgpu,
                        &create_texture_from_image(
                            &self.wgpu,
                            &im.to_rgba8(),
                            width,
                            height,
                            wgpu::TextureFormat::Rgba8UnormSrgb,
                        )
                        .create_view(&Default::default()),
                        blit::ColourSpace::Linear,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                        wgpu::FilterMode::Linear,
                    )
                    .create_texture(
                        &self.wgpu,
                        width,
                        height,
                        1 + (std::cmp::max(width, height) as f32).log2() as u32,
                    ),
                );
                self.compute_bind_group = self.bindings.create_bind_group(&self.wgpu);
            }
        }
        log::info!(
            "Channel {index} loaded in {}s",
            now.elapsed().as_micros() as f32 * 1e-6
        );
    }

    pub fn load_channel_hdr(&mut self, index: usize, bytes: &[u8]) -> Result<(), String> {
        let now = instant::Instant::now();
        let decoder = image::codecs::hdr::HdrDecoder::new(bytes).map_err(|e| e.to_string())?;
        let meta = decoder.metadata();
        let pixels = decoder.read_image_native().map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = pixels
            .iter()
            .flat_map(|p| [p.c[0], p.c[1], p.c[2], p.e])
            .collect();
        self.bindings.channels[index].set_texture(
            blit::Blitter::new(
                &self.wgpu,
                &create_texture_from_image(
                    &self.wgpu,
                    &bytes,
                    meta.width,
                    meta.height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )
                .create_view(&Default::default()),
                blit::ColourSpace::Rgbe,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::FilterMode::Linear,
            )
            .create_texture(
                &self.wgpu,
                meta.width,
                meta.height,
                1 + (std::cmp::max(meta.width, meta.height) as f32).log2() as u32,
            ),
        );
        self.compute_bind_group = self.bindings.create_bind_group(&self.wgpu);
        log::info!(
            "Channel {index} loaded in {}s",
            now.elapsed().as_micros() as f32 * 1e-6
        );
        Ok(())
    }
}

fn create_texture_from_image(
    wgpu: &WgpuContext,
    rgba: &[u8],
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let texture = wgpu.device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: None,
    });
    wgpu.queue.write_texture(
        texture.as_image_copy(),
        rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * width),
            rows_per_image: std::num::NonZeroU32::new(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}
