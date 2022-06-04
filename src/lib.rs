mod utils;
mod blit;
pub mod context;

use context::WgpuContext;
use wasm_bindgen::prelude::*;
use naga::front::wgsl;
use num::Integer;
use bitvec::prelude::*;
use std::mem::{size_of, take};
use std::sync::atomic::{AtomicBool, Ordering};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Time {
    frame: u32,
    elapsed: f32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Mouse {
    pos: [u32; 2],
    click: i32,
}

struct Uniforms {
    time: wgpu::Buffer,
    mouse: wgpu::Buffer,
    keys: wgpu::Buffer,
    custom: wgpu::Buffer,
    user_data: wgpu::Buffer,
    atomic_storage_buffer: wgpu::Buffer,
    float_storage_buffer: wgpu::Buffer,
    debug_buffer: wgpu::Buffer,
    tex_read: wgpu::Texture,
    tex_write: wgpu::Texture,
    tex_screen: wgpu::Texture,
}

impl Drop for Uniforms {
    fn drop(&mut self) {
        self.time.destroy();
        self.mouse.destroy();
        self.keys.destroy();
        self.custom.destroy();
        self.user_data.destroy();
        self.atomic_storage_buffer.destroy();
        self.float_storage_buffer.destroy();
        self.debug_buffer.destroy();
        self.tex_read.destroy();
        self.tex_write.destroy();
        self.tex_screen.destroy();
    }
}

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
                    &JsValue::from(col)
                );
                match res {
                    Err(error) => log::error!("Error calling registered error callback: {error:?}"),
                    _ => ()
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
                    &JsValue::from(entry_points.into_iter().map(JsValue::from).collect::<js_sys::Array>())
                );
                match res {
                    Err(error) => log::error!("Error calling registered error callback: {error:?}"),
                    _ => ()
                };
            }
        }
    }
}

// safe because wasm is single-threaded: https://github.com/rustwasm/wasm-bindgen/issues/1505
unsafe impl Send for ErrorCallback {}
unsafe impl Sync for ErrorCallback {}

const NUM_KEYCODES: usize = 256;
const MAX_CUSTOM_PARAMS: usize = 32;
const NUM_ASSERT_COUNTERS: usize = 10;
const USER_DATA_BYTES: usize = 4096;

#[wasm_bindgen]
pub struct WgpuToyRenderer {
    #[wasm_bindgen(skip)]
    pub wgpu: WgpuContext,
    screen_width: u32,
    screen_height: u32,
    time: Time,
    mouse: Mouse,
    keys: BitArr!(for NUM_KEYCODES, in u8, Lsb0),
    custom_names: Vec<String>,
    custom_values: Vec<f32>,
    user_data: std::collections::HashMap<String, Vec<u32>>,
    uniforms: Uniforms,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline_layout: wgpu::PipelineLayout,
    last_compute_pipelines: Option<Vec<(wgpu::ComputePipeline, [u32; 3])>>,
    compute_pipelines: Vec<(wgpu::ComputePipeline, [u32; 3])>,
    compute_bind_group: wgpu::BindGroup,
    staging_belt: wgpu::util::StagingBelt,
    on_error_cb: ErrorCallback,
    on_success_cb: SuccessCallback,
    channels: [wgpu::Texture; 2],
    pass_f32: bool,
    screen_blitter: blit::Blitter,
    query_set: Option<wgpu::QuerySet>,
    last_stats: instant::Instant,
}

static SHADER_ERROR: AtomicBool = AtomicBool::new(false);

fn compute_bind_group_layout_entries(pass_f32: bool) -> [wgpu::BindGroupLayoutEntry; 19] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: false
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: !pass_f32 },
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: if pass_f32 { wgpu::TextureFormat::Rgba32Float } else { wgpu::TextureFormat::Rgba16Float },
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: false
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: false
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 11,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 19,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: true
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 20,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 21,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 22,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 23,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 24,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 25,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ]
}

fn create_uniforms(wgpu: &WgpuContext, width: u32, height: u32, pass_f32: bool) -> Uniforms {
    let pixels = (width * height) as usize;
    let sizes = [
        // buffers
        size_of::<Time>(),
        size_of::<Mouse>(),
        NUM_KEYCODES / 8,
        MAX_CUSTOM_PARAMS * size_of::<f32>(),
        USER_DATA_BYTES,
        134217728, // default limit (128 MiB)
        134217728, // default limit (128 MiB)
        NUM_ASSERT_COUNTERS * size_of::<u32>(),

        // textures
        pixels * 4 * if pass_f32 { size_of::<[i32; 4]>() } else { size_of::<[i16; 4]>() },
        pixels * 4 * if pass_f32 { size_of::<[i32; 4]>() } else { size_of::<[i16; 4]>() },
        pixels * 1 * size_of::<[i16; 4]>(),
    ];
    log::info!("VRAM: allocating {} MiB of buffers and textures", sizes.iter().sum::<usize>() >> 20);
    Uniforms {
        time: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<Time>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        mouse: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<Mouse>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        keys: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (NUM_KEYCODES / 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        custom: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (MAX_CUSTOM_PARAMS * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        user_data: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: USER_DATA_BYTES as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        atomic_storage_buffer: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 134217728, // default limit (128 MiB)
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        float_storage_buffer: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 134217728, // default limit (128 MiB)
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        debug_buffer: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (NUM_ASSERT_COUNTERS * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        tex_read: wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if pass_f32 { wgpu::TextureFormat::Rgba32Float } else { wgpu::TextureFormat::Rgba16Float },
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        }),
        tex_write: wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if pass_f32 { wgpu::TextureFormat::Rgba32Float } else { wgpu::TextureFormat::Rgba16Float },
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
        }),
        tex_screen: wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        }),
    }
}

fn create_compute_bind_group(wgpu: &WgpuContext, layout: &wgpu::BindGroupLayout, uniforms: &Uniforms, channels: &[wgpu::Texture]) -> wgpu::BindGroup {
    let repeat = wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        ..Default::default()
    };
    wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.custom.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniforms.time.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: uniforms.mouse.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniforms.keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&uniforms.tex_screen.create_view(&Default::default())) },
            wgpu::BindGroupEntry { binding: 5, resource: uniforms.atomic_storage_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&uniforms.tex_read.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&uniforms.tex_write.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 8, resource: uniforms.debug_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: uniforms.float_storage_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&channels[0].create_view(&Default::default())) },
            wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&channels[1].create_view(&Default::default())) },
            wgpu::BindGroupEntry { binding: 19, resource: uniforms.user_data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 20, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&Default::default())) },
            wgpu::BindGroupEntry { binding: 21, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 22, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 23, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&repeat)) },
            wgpu::BindGroupEntry { binding: 24, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..repeat
            })) },
            wgpu::BindGroupEntry { binding: 25, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..repeat
            })) },
        ],
    })
}

fn stage(staging_belt: &mut wgpu::util::StagingBelt, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, data: &[u8], buffer: &wgpu::Buffer) {
    match wgpu::BufferSize::new(data.len() as u64) {
        None => log::warn!("no data to stage"),
        Some(size) => staging_belt.write_buffer(encoder, buffer, 0, size, device)
                                  .copy_from_slice(data)
    }
}

// https://llogiq.github.io/2016/09/24/newline.html
fn count_newlines(s: &str) -> usize {
    s.as_bytes().iter().filter(|&&c| c == b'\n').count()
}

#[wasm_bindgen]
impl WgpuToyRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(wgpu: WgpuContext) -> WgpuToyRenderer {
        let size = wgpu.window.inner_size();
        let uniforms = create_uniforms(&wgpu, size.width, size.height, false);
        let compute_bind_group_layout = wgpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &compute_bind_group_layout_entries(false),
        });

        let blank = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
        };
        let channels = [
            wgpu.device.create_texture(&blank),
            wgpu.device.create_texture(&blank),
        ];

        WgpuToyRenderer {
            compute_pipeline_layout: wgpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            }),
            compute_bind_group: create_compute_bind_group(&wgpu, &compute_bind_group_layout, &uniforms, &channels),
            last_compute_pipelines: None,
            compute_pipelines: vec![],
            screen_width: size.width,
            screen_height: size.height,
            time: Time {
                frame: 0,
                elapsed: 0.,
            },
            mouse: Mouse {
                pos: [0, 0],
                click: 0,
            },
            keys: bitarr![u8, Lsb0; 0; 256],
            staging_belt: wgpu::util::StagingBelt::new(4096),
            screen_blitter: blit::Blitter::new(
                &wgpu,
                &uniforms.tex_screen.create_view(&Default::default()),
                blit::ColourSpace::Linear,
                wgpu.surface_format,
                wgpu::FilterMode::Nearest),
            wgpu,
            uniforms,
            compute_bind_group_layout,
            on_error_cb: ErrorCallback(None),
            on_success_cb: SuccessCallback(None),
            channels,
            custom_names: vec!["_dummy".into()], // just to avoid creating an empty struct in wgsl
            custom_values: vec![0.],
            user_data: std::collections::HashMap::from([("_dummy".into(), vec![0])]),
            pass_f32: false,
            query_set: None,
            last_stats: instant::Instant::now(),
        }
    }

    pub fn render(&mut self) {
        match self.wgpu.surface.get_current_texture() {
            Err(e) => log::error!("Unable to get framebuffer: {e}"),
            Ok(f) => self.render_to(f)
        }
    }

    fn render_to(&mut self, frame: wgpu::SurfaceTexture) {
        let stats_period = 100;
        let mut encoder = self.wgpu.device.create_command_encoder(&Default::default());
        let custom_bytes: Vec<u8> = self.custom_values.iter().flat_map(|x| bytemuck::bytes_of(x).iter().copied()).collect();
        let user_data: Vec<u8> = self.user_data.iter().flat_map(|(_,x)| bytemuck::cast_slice(x).iter().copied()).collect();
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, &custom_bytes, &self.uniforms.custom);
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, &user_data, &self.uniforms.user_data);
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, bytemuck::bytes_of(&self.time), &self.uniforms.time);
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, bytemuck::bytes_of(&self.mouse), &self.uniforms.mouse);
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, &self.keys.as_raw_slice(), &self.uniforms.keys);
        if self.time.frame % stats_period == 0 {
            //encoder.clear_buffer(&self.uniforms.debug_buffer, 0, None); // doesn't work for some reason
            stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, bytemuck::bytes_of(&[0u32; NUM_ASSERT_COUNTERS]), &self.uniforms.debug_buffer);

            if self.time.frame > 0 {
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
        for (pass_index, (pipeline, workgroup_size)) in self.compute_pipelines.iter().enumerate() {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            if let Some(q) = &self.query_set {
                compute_pass.write_timestamp(q, 2 * pass_index as u32);
            }
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch(self.screen_width.div_ceil(&workgroup_size[0]), self.screen_height.div_ceil(&workgroup_size[1]), 1);
            if let Some(q) = &self.query_set {
                compute_pass.write_timestamp(q, 2 * pass_index as u32 + 1);
            }
            drop(compute_pass);
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.uniforms.tex_write,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.uniforms.tex_read,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.screen_width,
                    height: self.screen_height,
                    depth_or_array_layers: 4,
                });
        }
        let mut staging_buffer = None;
        let query_offset = NUM_ASSERT_COUNTERS * size_of::<u32>();
        let query_count = 2 * self.compute_pipelines.len();
        if self.time.frame % stats_period == stats_period - 1 {
            let buf = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (query_offset + query_count * size_of::<u64>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&self.uniforms.debug_buffer, 0, &buf, 0, query_offset as wgpu::BufferAddress);
            if let Some(q) = &self.query_set {
                encoder.resolve_query_set(q, 0..query_count as u32, &buf, query_offset as wgpu::BufferAddress);
            }
            staging_buffer = Some(buf);
        }
        self.time.frame += 1;
        self.screen_blitter.blit(&mut encoder, &frame.texture.create_view(&Default::default()));
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
                                let percent = *count as f32 / (numthreads * stats_period) as f32 * 100.0;
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
        for (a,t) in [("int","i32"), ("uint","u32"), ("float","f32")] {
            s.push_str(&format!("type {a} = {t};\n"));
            for n in [2,3,4] {
                s.push_str(&format!("type {a}{n} = vec{n}<{t}>;\n"));
            }
        }
        s.push_str(r#"
struct Time { frame: uint, elapsed: float };
struct Mouse { pos: uint2, click: int };
"#);
        s.push_str("struct Custom {\n");
        for name in &self.custom_names {
            s.push_str("    ");
            s.push_str(name);
            s.push_str(": float,\n");
        }
        s.push_str("};\n");
        s.push_str("@group(0) @binding(0) var<uniform> custom: Custom;");
        s.push_str("struct Data {");
        for (key, val) in self.user_data.iter() {
            let n = val.len();
            s.push_str(&format!("{key}: array<u32,{n}>,"));
        }
        s.push_str("};");
        s.push_str("@group(0) @binding(19) var<storage,read> data: Data;");
        let pass_format = if self.pass_f32 { "rgba32float" } else { "rgba16float" };
        s.push_str(&format!(r#"
@group(0) @binding(1) var<uniform> time: Time;
@group(0) @binding(2) var<uniform> mouse: Mouse;
@group(0) @binding(3) var<uniform> _keyboard: array<vec4<u32>,2>;
@group(0) @binding(4) var screen: texture_storage_2d<rgba16float,write>;
@group(0) @binding(5) var<storage,read_write> atomic_storage: array<atomic<i32>>;
@group(0) @binding(6) var pass_in: texture_2d_array<f32>;
@group(0) @binding(7) var pass_out: texture_storage_2d_array<{pass_format},write>;
@group(0) @binding(8) var<storage,read_write> _assert_counts: array<atomic<u32>>;
@group(0) @binding(9) var<storage,read_write> float_storage: array<vec4<f32>>;
@group(0) @binding(10) var channel0: texture_2d<f32>;
@group(0) @binding(11) var channel1: texture_2d<f32>;
@group(0) @binding(20) var nearest: sampler;
@group(0) @binding(21) var bilinear: sampler;
@group(0) @binding(22) var trilinear: sampler;
@group(0) @binding(23) var nearest_repeat: sampler;
@group(0) @binding(24) var bilinear_repeat: sampler;
@group(0) @binding(25) var trilinear_repeat: sampler;
        "#));
        s.push_str(r#"
fn keyDown(keycode: uint) -> bool {
    return ((_keyboard[keycode / 128u][(keycode % 128u) / 32u] >> (keycode % 32u)) & 1u) == 1u;
}

fn assert(index: int, success: bool) {
    if (!success) {
        atomicAdd(&_assert_counts[index], 1u);
    }
}
        "#);
        s.push_str(r#"
fn passStore(pass: int, coord: int2, value: float4) {
    textureStore(pass_out, coord, pass, value);
}

fn passLoad(pass: int, coord: int2, lod: int) -> float4 {
    return textureLoad(pass_in, coord, pass, lod);
}

fn passSampleLevelBilinearRepeat(pass: int, uv: float2, lod: float) -> float4 {"#);
        if self.pass_f32 {
            // https://iquilezles.org/articles/hwinterpolation/
            s.push_str(r#"
    let res = float2(textureDimensions(pass_in));
    let st = uv * res - 0.5;
    let iuv = floor(st);
    let fuv = fract(st);
    let a = textureSampleLevel(pass_in, nearest, fract((iuv + float2(0.5,0.5)) / res), pass, lod);
    let b = textureSampleLevel(pass_in, nearest, fract((iuv + float2(1.5,0.5)) / res), pass, lod);
    let c = textureSampleLevel(pass_in, nearest, fract((iuv + float2(0.5,1.5)) / res), pass, lod);
    let d = textureSampleLevel(pass_in, nearest, fract((iuv + float2(1.5,1.5)) / res), pass, lod);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
"#);
        } else {
            s.push_str(r#"
    return textureSampleLevel(pass_in, bilinear, fract(uv), pass, lod);
"#);
        }
        s.push_str("}");
        return s;
    }

    fn handle_success(&self, entry_points: Vec<String>) {
        #[cfg(target_arch = "wasm32")]
        self.on_success_cb.call(entry_points);
    }

    fn preprocess(&mut self, shader: &str) -> Option<(String, Vec<usize>)> {
        self.user_data = std::collections::HashMap::from([("_dummy".into(), vec![0])]); // clear
        let mut sourcemap = vec![0];
        let mut wgsl = String::new();
        let mut push_line = |n, s| {
            sourcemap.push(n);
            wgsl.push_str(s);
            wgsl.push_str("\n");
        };
        for (line, n) in shader.lines().zip(1..) {
            push_line(n, line);
        }
        Some((wgsl, sourcemap))
    }

    pub fn set_shader(&mut self, shader: &str) {
        let now = instant::Instant::now();
        if let Some((source, sourcemap)) = self.preprocess(shader) {
        let prelude = self.prelude(); // prelude should be generated after preprocessor has run

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
                },
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
                let entry_points: Vec<_> = module.entry_points.iter()
                    .filter(|f| f.stage == naga::ShaderStage::Compute).collect();
                let entry_point_names: Vec<String> = entry_points.iter().map(|entry_point| {entry_point.name.clone()}).collect();
                self.handle_success(entry_point_names);
                let compute_shader = self.wgpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&wgsl)),
                });
                self.last_compute_pipelines = Some(take(&mut self.compute_pipelines));
                self.compute_pipelines = entry_points.iter().map(|entry_point| {
                    (self.wgpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&self.compute_pipeline_layout),
                        module: &compute_shader,
                        entry_point: &entry_point.name,
                    }), entry_point.workgroup_size)
                }).collect();
                self.query_set = if !self.wgpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) { None } else {
                    Some(self.wgpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: None,
                        count: 2 * self.compute_pipelines.len() as u32,
                        ty: wgpu::QueryType::Timestamp,
                    }))
                };
                log::info!("Shader compiled in {}s", now.elapsed().as_micros() as f32 * 1e-6);
            },
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
            },
        }
        }
    }

    pub fn set_time_elapsed(&mut self, t: f32) {
        self.time.elapsed = t;
    }

    pub fn set_mouse_pos(&mut self, x: u32, y: u32) {
        self.mouse.pos = [x, y];
    }

    pub fn set_mouse_click(&mut self, click: bool) {
        self.mouse.click = if click {1} else {0};
    }

    pub fn set_keydown(&mut self, keycode: usize, keydown: bool) {
        self.keys.set(keycode, keydown);
    }

    pub fn set_custom_floats(&mut self, names: Vec<js_sys::JsString>, values: Vec<f32>) {
        self.custom_names = names.iter().map(From::from).collect();
        self.custom_values = values;
    }

    pub fn set_pass_f32(&mut self, pass_f32: bool) {
        self.pass_f32 = pass_f32;
        self.reset();
    }

    pub fn resize(&mut self, width: u32, height: u32, scale: f32) {
        self.screen_width = (width as f32 * scale) as u32;
        self.screen_height = (height as f32 * scale) as u32;
        self.reset();
        self.wgpu.window.set_inner_size(winit::dpi::PhysicalSize::new(width, height));
    }

    pub fn reset(&mut self) {
        self.time.frame = 0;
        self.uniforms = create_uniforms(&self.wgpu, self.screen_width, self.screen_height, self.pass_f32);
        self.compute_bind_group_layout = self.wgpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &compute_bind_group_layout_entries(self.pass_f32),
        });
        self.compute_pipeline_layout = self.wgpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&self.compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        self.compute_bind_group = create_compute_bind_group(&self.wgpu, &self.compute_bind_group_layout, &self.uniforms, &self.channels);
        self.screen_blitter = blit::Blitter::new(
            &self.wgpu,
            &self.uniforms.tex_screen.create_view(&Default::default()),
            blit::ColourSpace::Linear,
            self.wgpu.surface_format,
            wgpu::FilterMode::Linear);
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
                self.channels[index] = blit::Blitter::new(
                    &self.wgpu,
                    &create_texture_from_image(&self.wgpu, &im.to_rgba8(), width, height, wgpu::TextureFormat::Rgba8UnormSrgb).create_view(&Default::default()),
                    blit::ColourSpace::Linear,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    wgpu::FilterMode::Linear,
                ).create_texture(&self.wgpu, width, height, 1 + (std::cmp::max(width, height) as f32).log2() as u32);
                self.compute_bind_group = create_compute_bind_group(&self.wgpu, &self.compute_bind_group_layout, &self.uniforms, &self.channels);
            }
        }
        log::info!("Channel {index} loaded in {}s", now.elapsed().as_micros() as f32 * 1e-6);
    }

    pub fn load_channel_hdr(&mut self, index: usize, bytes: &[u8]) -> Result<(), String> {
        let now = instant::Instant::now();
        let decoder = image::codecs::hdr::HdrDecoder::new(bytes).map_err(|e| e.to_string())?;
        let meta = decoder.metadata();
        let pixels = decoder.read_image_native().map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.c[0], p.c[1], p.c[2], p.e]).collect();
        self.channels[index] = blit::Blitter::new(
            &self.wgpu,
            &create_texture_from_image(&self.wgpu, &bytes, meta.width, meta.height, wgpu::TextureFormat::Rgba8Unorm).create_view(&Default::default()),
            blit::ColourSpace::Rgbe,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::FilterMode::Linear,
        ).create_texture(&self.wgpu, meta.width, meta.height, 1 + (std::cmp::max(meta.width, meta.height) as f32).log2() as u32);
        self.compute_bind_group = create_compute_bind_group(&self.wgpu, &self.compute_bind_group_layout, &self.uniforms, &self.channels);
        log::info!("Channel {index} loaded in {}s", now.elapsed().as_micros() as f32 * 1e-6);
        Ok(())
    }

}

fn create_texture_from_image(wgpu: &WgpuContext, rgba: &[u8], width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    let texture = wgpu.device.create_texture(
        &wgpu::TextureDescriptor {
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
        }
    );
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
