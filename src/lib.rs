mod bind;
mod blit;
pub mod context;
mod pp;
mod utils;

#[cfg(feature = "winit")]
use context::init_wgpu;
use context::WgpuContext;
use lazy_regex::regex;
use num::Integer;
use pp::{SourceMap, WGSLError};
use wgpu::{Maintain, SubmissionIndex};
use std::collections::HashMap;
use std::mem::{size_of, take};
use std::sync::atomic::{AtomicBool, Ordering};
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct SuccessCallback(Option<js_sys::Function>);

#[cfg(target_arch = "wasm32")]
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

#[cfg(not(target_arch = "wasm32"))]
struct SuccessCallback(Option<()>);

struct ComputePipeline {
    name: String,
    workgroup_size: [u32; 3],
    workgroup_count: Option<[u32; 3]>,
    dispatch_count: u32,
    pipeline: wgpu::ComputePipeline,
}

#[wasm_bindgen]
pub struct WgpuToyRenderer {
    #[wasm_bindgen(skip)]
    pub wgpu: WgpuContext,
    screen_width: u32,
    screen_height: u32,
    bindings: bind::Bindings,
    compute_pipeline_layout: wgpu::PipelineLayout,
    last_compute_pipelines: Option<Vec<ComputePipeline>>,
    compute_pipelines: Vec<ComputePipeline>,
    compute_bind_group: wgpu::BindGroup,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    on_success_cb: SuccessCallback,
    pass_f32: bool,
    screen_blitter: blit::Blitter,
    query_set: Option<wgpu::QuerySet>,
    last_stats: instant::Instant,
    source: SourceMap,
}

const STATS_PERIOD: u32 = 100;
const ASSERTS_SIZE: usize = bind::NUM_ASSERT_COUNTERS * size_of::<u32>();

static SHADER_ERROR: AtomicBool = AtomicBool::new(false);

// https://llogiq.github.io/2016/09/24/newline.html
fn count_newlines(s: &str) -> usize {
    s.as_bytes().iter().filter(|&&c| c == b'\n').count()
}

// FIXME: async fn(&str) doesn't currently work with wasm_bindgen: https://stackoverflow.com/a/63655324/78204
#[cfg(feature = "winit")]
#[wasm_bindgen]
pub async fn create_renderer(
    width: u32,
    height: u32,
    bind_id: String,
) -> Result<WgpuToyRenderer, String> {
    let wgpu = init_wgpu(width, height, &bind_id).await?;
    Ok(WgpuToyRenderer::new(wgpu))
}

impl WgpuToyRenderer {
    pub fn new(wgpu: WgpuContext) -> WgpuToyRenderer {
        let bindings = bind::Bindings::new(
            &wgpu,
            wgpu.surface_config.width,
            wgpu.surface_config.height,
            false,
        );
        let layout = bindings.create_bind_group_layout(&wgpu);

        WgpuToyRenderer {
            compute_pipeline_layout: bindings.create_pipeline_layout(&wgpu, &layout),
            compute_bind_group: bindings.create_bind_group(&wgpu, &layout),
            compute_bind_group_layout: layout,
            last_compute_pipelines: None,
            compute_pipelines: vec![],
            screen_width: wgpu.surface_config.width,
            screen_height: wgpu.surface_config.height,
            screen_blitter: blit::Blitter::new(
                &wgpu,
                bindings.tex_screen.view(),
                blit::ColourSpace::Linear,
                wgpu.surface_config.format,
                wgpu::FilterMode::Nearest,
            ),
            wgpu,
            bindings,
            on_success_cb: SuccessCallback(None),
            pass_f32: false,
            query_set: None,
            last_stats: instant::Instant::now(),
            source: SourceMap::new(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WgpuToyRenderer {
    #[cfg(target_arch = "wasm32")]
    pub fn render(&mut self) {
        use wgpu::SurfaceError;

        match self.wgpu.surface.get_current_texture() {
            Err(err) => match err {
                SurfaceError::Lost | SurfaceError::Outdated => {
                    log::error!("Unable to get framebuffer: {err}");
                    self.wgpu
                        .surface
                        .configure(&self.wgpu.device, &self.wgpu.surface_config);
                }
                SurfaceError::OutOfMemory => log::error!("Out of GPU Memory!"),
                SurfaceError::Timeout => log::warn!("Surface Timeout"),
            },
            Ok(f) => {
                let (staging_buffer, _) = self.render_to(&f);
                f.present();
                wasm_bindgen_futures::spawn_local(Self::postrender(
                    staging_buffer,
                    self.screen_width * self.screen_height,
                    self.source.assert_map.clone(),
                ));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn render_async(&mut self) {
        use wgpu::SurfaceError;

        match self.wgpu.surface.get_current_texture() {
            Err(err) => match err {
                SurfaceError::Lost | SurfaceError::Outdated => {
                    log::error!("Unable to get framebuffer: {err}");
                    self.wgpu
                        .surface
                        .configure(&self.wgpu.device, &self.wgpu.surface_config);
                    #[cfg(feature = "winit")]
                    self.wgpu.window.request_redraw();
                }
                SurfaceError::OutOfMemory => log::error!("Out of GPU Memory!"),
                SurfaceError::Timeout => log::warn!("Surface Timeout"),
            },
            Ok(f) => {
                let (staging_buffer, _) = self.render_to(&f);
                f.present();
                Self::postrender(
                    staging_buffer,
                    self.screen_width * self.screen_height,
                    self.source.assert_map.clone(),
                )
                .await
            }
        }
    }

    pub fn render_to(&mut self, frame: &wgpu::SurfaceTexture) -> (Option<wgpu::Buffer>, SubmissionIndex) {
        let mut encoder = self.wgpu.device.create_command_encoder(&Default::default());
        self.bindings.stage(&self.wgpu.queue);
        if self.bindings.time.host.frame % STATS_PERIOD == 0 {
            //encoder.clear_buffer(&self.uniforms.debug_buffer, 0, None); // not yet implemented in web backend
            self.wgpu.queue.write_buffer(
                self.bindings.debug_buffer.buffer(),
                0,
                bytemuck::bytes_of(&[0u32; bind::NUM_ASSERT_COUNTERS]),
            );

            if self.bindings.time.host.frame > 0 {
                let mean = self.last_stats.elapsed().as_secs_f32() / STATS_PERIOD as f32;
                self.last_stats = instant::Instant::now();
                log::debug!("{} fps ({} ms)", 1. / mean, 1e3 * mean);
            }
        }
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
            for i in 0..p.dispatch_count {
                let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                if let Some(q) = &self.query_set {
                    compute_pass.write_timestamp(q, 2 * pass_index as u32);
                }
                let workgroup_count = p.workgroup_count.unwrap_or([
                    self.screen_width.div_ceil(&p.workgroup_size[0]),
                    self.screen_height.div_ceil(&p.workgroup_size[1]),
                    1,
                ]);
                compute_pass.set_pipeline(&p.pipeline);
                self.wgpu.queue.write_buffer(
                    self.bindings.dispatch_info.buffer(),
                    bind::OFFSET_ALIGNMENT as u64 * dispatch_counter,
                    bytemuck::bytes_of(&i),
                );
                compute_pass.set_bind_group(
                    0,
                    &self.compute_bind_group,
                    &[bind::OFFSET_ALIGNMENT as u32 * dispatch_counter as u32],
                );
                dispatch_counter += 1;
                compute_pass.dispatch_workgroups(
                    workgroup_count[0],
                    workgroup_count[1],
                    workgroup_count[2],
                );
                if let Some(q) = &self.query_set {
                    compute_pass.write_timestamp(q, 2 * pass_index as u32 + 1);
                }
                drop(compute_pass);
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: self.bindings.tex_write.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: self.bindings.tex_read.texture(),
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
        let query_count = 2 * self.compute_pipelines.len();
        if self.bindings.time.host.frame % STATS_PERIOD == STATS_PERIOD - 1 {
            let buf = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (ASSERTS_SIZE + query_count * size_of::<u64>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                self.bindings.debug_buffer.buffer(),
                0,
                &buf,
                0,
                ASSERTS_SIZE as wgpu::BufferAddress,
            );
            if let Some(q) = &self.query_set {
                encoder.resolve_query_set(
                    q,
                    0..query_count as u32,
                    &buf,
                    ASSERTS_SIZE as wgpu::BufferAddress,
                );
            }
            staging_buffer = Some(buf);
        }
        self.bindings.time.host.frame = self.bindings.time.host.frame.wrapping_add(1);
        self.screen_blitter.blit(
            &mut encoder,
            &frame.texture.create_view(&Default::default()),
        );

        let i = self.wgpu.queue.submit(Some(encoder.finish()));
        (staging_buffer, i)
    }

    async fn postrender(
        staging_buffer: Option<wgpu::Buffer>,
        numthreads: u32,
        assert_map: Vec<usize>,
    ) {
        if let Some(buf) = staging_buffer {
            let buffer_slice = buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| match sender.send(v) {
                Ok(()) => {}
                Err(_) => log::error!("Channel closed unexpectedly"),
            });
            match receiver.receive().await {
                None => log::error!("Channel closed unexpectedly"),
                Some(Err(e)) => log::error!("{e}"),
                Some(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();
                    let assertions: &[u32] = bytemuck::cast_slice(&data[0..ASSERTS_SIZE]);
                    let timestamps: &[u64] = bytemuck::cast_slice(&data[ASSERTS_SIZE..]);
                    for (i, count) in assertions.iter().enumerate() {
                        if count > &0 {
                            let percent =
                                *count as f32 / (numthreads * STATS_PERIOD) as f32 * 100.0;
                            log::warn!("Assertion {i} failed in {percent}% of threads");
                            if i < assert_map.len() {
                                WGSLError::handler(
                                    &format!("Assertion failed in {percent}% of threads"),
                                    assert_map[i],
                                    0,
                                );
                            }
                        }
                    }
                }
            }
            buf.unmap();
        }
    }

    pub fn prelude(&self) -> String {
        let mut s = String::new();
        for (a, t) in [("int", "i32"), ("uint", "u32"), ("float", "f32")] {
            s.push_str(&format!("alias {a} = {t};\n"));
        }
        for (a, t) in [
            ("int", "i32"),
            ("uint", "u32"),
            ("float", "f32"),
            ("bool", "bool"),
        ] {
            for n in 2..5 {
                s.push_str(&format!("alias {a}{n} = vec{n}<{t}>;\n"));
            }
        }
        for n in 2..5 {
            for m in 2..5 {
                s.push_str(&format!("alias float{n}x{m} = mat{n}x{m}<f32>;\n"));
            }
        }
        s.push_str(
            r#"
struct Time { frame: uint, elapsed: float, delta: float }
struct Mouse { pos: uint2, click: int }
struct DispatchInfo { id: uint }
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
        s.push_str("struct Data {\n");
        for (key, val) in self.bindings.user_data.host.iter() {
            let n = val.len();
            s.push_str(&format!("    {key}: array<u32,{n}>,\n"));
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

fn passStore(pass_index: int, coord: int2, value: float4) {
    textureStore(pass_out, coord, pass_index, value);
}

fn passLoad(pass_index: int, coord: int2, lod: int) -> float4 {
    return textureLoad(pass_in, coord, pass_index, lod);
}

fn passSampleLevelBilinearRepeat(pass_index: int, uv: float2, lod: float) -> float4 {"#,
        );
        if self.pass_f32 {
            // https://iquilezles.org/articles/hwinterpolation/
            s.push_str(
                r#"
    let res = float2(textureDimensions(pass_in));
    let st = uv * res - 0.5;
    let iuv = floor(st);
    let fuv = fract(st);
    let a = textureSampleLevel(pass_in, nearest, fract((iuv + float2(0.5,0.5)) / res), pass_index, lod);
    let b = textureSampleLevel(pass_in, nearest, fract((iuv + float2(1.5,0.5)) / res), pass_index, lod);
    let c = textureSampleLevel(pass_in, nearest, fract((iuv + float2(0.5,1.5)) / res), pass_index, lod);
    let d = textureSampleLevel(pass_in, nearest, fract((iuv + float2(1.5,1.5)) / res), pass_index, lod);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
"#,
            );
        } else {
            s.push_str(
                r#"
    return textureSampleLevel(pass_in, bilinear, fract(uv), pass_index, lod);
"#,
            );
        }
        s.push('}');
        s
    }

    fn handle_success(&self, entry_points: Vec<String>) {
        #[cfg(target_arch = "wasm32")]
        self.on_success_cb.call(entry_points);
        #[cfg(not(target_arch = "wasm32"))]
        log::info!("Entry points: {:?}", entry_points);
    }

    #[cfg(target_arch = "wasm32")]
    pub fn preprocess(&self, shader: &str) -> js_sys::Promise {
        let shader = shader.to_owned();
        let defines = HashMap::from([
            ("SCREEN_WIDTH".to_owned(), self.screen_width.to_string()),
            ("SCREEN_HEIGHT".to_owned(), self.screen_height.to_string()),
        ]);
        utils::promise(async move { pp::Preprocessor::new(defines).run(&shader).await })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn preprocess_async(&self, shader: &str) -> Option<SourceMap> {
        let shader = shader.to_owned();
        let defines = HashMap::from([
            ("SCREEN_WIDTH".to_owned(), self.screen_width.to_string()),
            ("SCREEN_HEIGHT".to_owned(), self.screen_height.to_string()),
        ]);
        pp::Preprocessor::new(defines).run(&shader).await
    }

    pub fn compile(&mut self, source: SourceMap) {
        let now = instant::Instant::now();
        let prelude = self.prelude(); // prelude must be generated after preprocessor has run

        // FIXME: remove pending resolution of this issue: https://github.com/gfx-rs/wgpu/issues/2130
        let prelude_len = count_newlines(&prelude);
        let re_parser = regex!(r"(?s):(\d+):(\d+) (.*)");
        let re_invalid = regex!(r"\[Invalid \w+\] is invalid.");
        let sourcemap_clone = source.map.clone();
        self.wgpu
            .device
            .on_uncaptured_error(Box::new(move |e: wgpu::Error| {
                let err = &e.to_string();
                if re_invalid.is_match(err) {
                    return;
                }
                match re_parser.captures(err) {
                    None => {
                        log::error!("{e}");
                        WGSLError::handler(err, 0, 0);
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
                        WGSLError::handler(summary, n, col);
                        SHADER_ERROR.store(true, Ordering::SeqCst);
                    }
                }
            }));

        let wgsl = &(prelude + &source.source);
        let re_entry_point = regex!(r"(?s)@compute.*?@workgroup_size\((.*?)\).*?fn\s+(\w+)");
        let entry_points: Vec<(String, [u32; 3])> = re_entry_point
            .captures_iter(&pp::strip_comments(wgsl))
            .map(|cap| {
                // TODO: Handle error if failed to parse the capture
                let mut sizes = cap[1].split(',').map(|s| s.trim().parse().unwrap_or(1));
                let workgroup_size: [u32; 3] = std::array::from_fn(|_| sizes.next().unwrap_or(1));

                (cap[2].to_owned(), workgroup_size)
            })
            .collect();
        let entry_point_names = entry_points.iter().map(|t| t.0.clone()).collect();
        self.handle_success(entry_point_names);
        let compute_shader = self
            .wgpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
            });
        self.last_compute_pipelines = Some(take(&mut self.compute_pipelines));
        self.compute_pipelines = entry_points
            .iter()
            .map(|entry_point| ComputePipeline {
                name: entry_point.0.clone(),
                workgroup_size: entry_point.1,
                workgroup_count: source.workgroup_count.get(&entry_point.0).cloned(),
                dispatch_count: *source.dispatch_count.get(&entry_point.0).unwrap_or(&1),
                pipeline: self.wgpu.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&self.compute_pipeline_layout),
                        module: &compute_shader,
                        entry_point: &entry_point.0,
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
        self.bindings.user_data.host = source.user_data.clone();
        log::info!(
            "Shader compiled in {}s",
            now.elapsed().as_micros() as f32 * 1e-6
        );
        self.source = source;
    }

    pub fn set_time_elapsed(&mut self, t: f32) {
        self.bindings.time.host.elapsed = t;
    }

    pub fn set_time_delta(&mut self, t: f32) {
        self.bindings.time.host.delta = t;
    }

    pub fn set_mouse_pos(&mut self, x: f32, y: f32) {
        if self.bindings.mouse.host.click == 1 {
            self.bindings.mouse.host.pos = [
                (x * self.screen_width as f32) as u32,
                (y * self.screen_height as f32) as u32,
            ];
        }
    }

    pub fn set_mouse_click(&mut self, click: bool) {
        self.bindings.mouse.host.click = if click { 1 } else { 0 };
    }

    pub fn set_keydown(&mut self, keycode: usize, keydown: bool) {
        self.bindings.keys.host.set(keycode, keydown);
    }

    #[cfg(target_arch = "wasm32")]
    pub fn set_custom_floats(&mut self, names: Vec<js_sys::JsString>, values: Vec<f32>) {
        self.bindings.custom.host = (names.iter().map(From::from).collect(), values);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_custom_floats(&mut self, names: Vec<String>, values: Vec<f32>) {
        self.bindings.custom.host = (names, values);
    }

    pub fn set_pass_f32(&mut self, pass_f32: bool) {
        self.pass_f32 = pass_f32;
        self.reset();
    }

    pub fn resize(&mut self, width: u32, height: u32, scale: f32) {
        self.screen_width = (width as f32 * scale) as u32;
        self.screen_height = (height as f32 * scale) as u32;
        self.wgpu.surface_config.width = self.screen_width;
        self.wgpu.surface_config.height = self.screen_height;
        self.wgpu
            .surface
            .configure(&self.wgpu.device, &self.wgpu.surface_config);
        self.reset();
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
        let layout = self.bindings.create_bind_group_layout(&self.wgpu);
        self.compute_pipeline_layout = self.bindings.create_pipeline_layout(&self.wgpu, &layout);
        self.compute_bind_group = self.bindings.create_bind_group(&self.wgpu, &layout);
        self.compute_bind_group_layout = layout;
        self.screen_blitter = blit::Blitter::new(
            &self.wgpu,
            self.bindings.tex_screen.view(),
            blit::ColourSpace::Linear,
            self.wgpu.surface_config.format,
            wgpu::FilterMode::Linear,
        );
    }

    #[cfg(target_arch = "wasm32")]
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
                self.compute_bind_group = self
                    .bindings
                    .create_bind_group(&self.wgpu, &self.compute_bind_group_layout);
            }
        }
        log::info!("Channel {index} loaded in {}s", now.elapsed().as_secs_f32());
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
        self.compute_bind_group = self
            .bindings
            .create_bind_group(&self.wgpu, &self.compute_bind_group_layout);
        log::info!("Channel {index} loaded in {}s", now.elapsed().as_secs_f32());
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
        view_formats: &[],
    });
    wgpu.queue.write_texture(
        texture.as_image_copy(),
        rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}
