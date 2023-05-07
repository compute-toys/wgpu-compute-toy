use crate::WgpuContext;
use bitvec::prelude::*;
use itertools::Itertools;
use num::Integer;
use std::mem::size_of;

const NUM_KEYCODES: usize = 256;
const MAX_CUSTOM_PARAMS: usize = 32;
pub const NUM_ASSERT_COUNTERS: usize = 10;
const USER_DATA_BYTES: usize = 4096;
pub const OFFSET_ALIGNMENT: usize = 256;

trait Binding {
    fn layout(&self) -> wgpu::BindingType;
    fn binding(&self) -> wgpu::BindingResource;
    fn to_wgsl(&self) -> &str;
}

pub struct BufferBinding<H> {
    pub host: H,
    //serialise: Box<dyn for<'a> Fn(&'a H) -> &'a [u8]>,
    serialise: Box<dyn Fn(&H) -> Vec<u8>>,
    device: wgpu::Buffer,
    layout: wgpu::BindingType,
    bind: Box<dyn for<'a> Fn(&'a wgpu::Buffer) -> wgpu::BufferBinding<'a>>,
    decl: String,
}

impl<H> Drop for BufferBinding<H> {
    fn drop(&mut self) {
        self.device.destroy();
    }
}

impl<H> Binding for BufferBinding<H> {
    fn layout(&self) -> wgpu::BindingType {
        self.layout
    }
    fn binding(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer((self.bind)(&self.device))
    }
    fn to_wgsl(&self) -> &str {
        &self.decl
    }
}

impl<H> BufferBinding<H> {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.device
    }
    fn stage(&self, queue: &wgpu::Queue) {
        let data = (self.serialise)(&self.host);
        if data.len() > 0 {
            queue.write_buffer(&self.device, 0, &data)
        } else {
            log::warn!("no data to stage")
        }
    }
}

pub struct TextureBinding {
    device: wgpu::Texture,
    view: wgpu::TextureView,
    layout: wgpu::BindingType,
    decl: String,
}

impl Drop for TextureBinding {
    fn drop(&mut self) {
        self.device.destroy();
    }
}

impl Binding for TextureBinding {
    fn layout(&self) -> wgpu::BindingType {
        self.layout
    }
    fn binding(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::TextureView(&self.view)
    }
    fn to_wgsl(&self) -> &str {
        &self.decl
    }
}

impl TextureBinding {
    pub fn texture(&self) -> &wgpu::Texture {
        &self.device
    }
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
    pub fn set_texture(&mut self, texture: wgpu::Texture) {
        self.device = texture;
        self.view = self.device.create_view(&Default::default());
    }
}

struct SamplerBinding {
    layout: wgpu::BindingType,
    bind: wgpu::Sampler,
    decl: String,
}

impl Binding for SamplerBinding {
    fn layout(&self) -> wgpu::BindingType {
        self.layout
    }
    fn binding(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Sampler(&self.bind)
    }
    fn to_wgsl(&self) -> &str {
        &self.decl
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct Time {
    pub frame: u32,
    pub elapsed: f32,
    pub delta: f32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct Mouse {
    pub pos: [u32; 2],
    pub click: i32,
}

pub struct Bindings {
    pub time: BufferBinding<Time>,
    pub mouse: BufferBinding<Mouse>,
    pub keys: BufferBinding<BitArr!(for NUM_KEYCODES, in u8, Lsb0)>,
    pub custom: BufferBinding<(Vec<String>, Vec<f32>)>,
    pub user_data: BufferBinding<indexmap::IndexMap<String, Vec<u32>>>,

    pub storage1: BufferBinding<()>,
    pub storage2: BufferBinding<()>,
    pub debug_buffer: BufferBinding<()>,
    pub dispatch_info: BufferBinding<()>,

    pub tex_screen: TextureBinding,
    pub tex_read: TextureBinding,
    pub tex_write: TextureBinding,
    pub channels: Vec<TextureBinding>,

    nearest: SamplerBinding,
    bilinear: SamplerBinding,
    trilinear: SamplerBinding,
    nearest_repeat: SamplerBinding,
    bilinear_repeat: SamplerBinding,
    trilinear_repeat: SamplerBinding,
}

impl Drop for Bindings {
    fn drop(&mut self) {
        log::info!("Destroying bindings");
    }
}

fn uniform_buffer_size<T>() -> u64 {
    let size = size_of::<T>() as u64;
    return size.div_ceil(&16) * 16;
}

impl Bindings {
    pub fn new(wgpu: &WgpuContext, width: u32, height: u32, pass_f32: bool) -> Self {
        log::info!("Creating bindings");
        let uniform_buffer = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage_buffer = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let pass_format = if pass_f32 {
            "rgba32float"
        } else {
            "rgba16float"
        };
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
            view_formats: &[],
        };
        let channel_layout = wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let repeat = wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        };
        let tex_screen = wgpu.device.create_texture(&wgpu::TextureDescriptor {
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
            view_formats: &[],
        });
        let tex_read = wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if pass_f32 {
                wgpu::TextureFormat::Rgba32Float
            } else {
                wgpu::TextureFormat::Rgba16Float
            },
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tex_write = wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if pass_f32 {
                wgpu::TextureFormat::Rgba32Float
            } else {
                wgpu::TextureFormat::Rgba16Float
            },
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let channel0 = wgpu.device.create_texture(&blank);
        let channel1 = wgpu.device.create_texture(&blank);
        Bindings {
            time: BufferBinding {
                host: Time {
                    frame: 0,
                    elapsed: 0.,
                    delta: 0.,
                },
                serialise: Box::new(|h| bytemuck::bytes_of(h).to_vec()),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: uniform_buffer_size::<Time>(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }),
                layout: uniform_buffer,
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: "var<uniform> time: Time".to_string(),
            },
            mouse: BufferBinding {
                host: Mouse {
                    pos: [0, 0],
                    click: 0,
                },
                serialise: Box::new(|h| bytemuck::bytes_of(h).to_vec()),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: uniform_buffer_size::<Mouse>(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }),
                layout: uniform_buffer,
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: "var<uniform> mouse: Mouse".to_string(),
            },
            keys: BufferBinding {
                host: bitarr![u8, Lsb0; 0; 256],
                serialise: Box::new(|h| h.as_raw_slice().to_vec()),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: uniform_buffer_size::<[u8; NUM_KEYCODES / 8]>(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }),
                layout: uniform_buffer,
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: "var<uniform> _keyboard: array<vec4<u32>,2>".to_string(),
            },
            custom: BufferBinding {
                host: (
                    vec!["_dummy".into()], // just to avoid creating an empty struct in wgsl
                    vec![0.],
                ),
                serialise: Box::new(|(_, v)| {
                    v.iter()
                        .flat_map(|x| bytemuck::bytes_of(x).iter().copied())
                        .collect()
                }),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: uniform_buffer_size::<[f32; MAX_CUSTOM_PARAMS]>(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }),
                layout: uniform_buffer,
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: "var<uniform> custom: Custom".to_string(),
            },
            user_data: BufferBinding {
                host: indexmap::IndexMap::from([("_dummy".into(), vec![0])]),
                serialise: Box::new(|h| {
                    h.iter()
                        .flat_map(|(_, x)| bytemuck::cast_slice(x).iter().copied())
                        .collect()
                }),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: uniform_buffer_size::<[u8; USER_DATA_BYTES]>(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                layout: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: "var<storage,read> data: Data".to_string(),
            },

            storage1: BufferBinding {
                host: (),
                serialise: Box::new(|_| vec![]),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 134217728, // default limit (128 MiB)
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                layout: storage_buffer,
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: String::new(),
            },
            storage2: BufferBinding {
                host: (),
                serialise: Box::new(|_| vec![]),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 134217728, // default limit (128 MiB)
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                layout: storage_buffer,
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: String::new(),
            },
            debug_buffer: BufferBinding {
                host: (),
                serialise: Box::new(|_| vec![]),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: uniform_buffer_size::<[u32; NUM_ASSERT_COUNTERS]>(),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                layout: storage_buffer,
                bind: Box::new(wgpu::Buffer::as_entire_buffer_binding),
                decl: "var<storage,read_write> _assert_counts: array<atomic<u32>>".to_string(),
            },
            dispatch_info: BufferBinding {
                host: (),
                serialise: Box::new(|_| vec![]),
                device: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 256 * OFFSET_ALIGNMENT as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }),
                layout: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                bind: Box::new(|d| wgpu::BufferBinding {
                    buffer: d,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<u32>() as u64),
                }),
                decl: "var<uniform> dispatch: DispatchInfo".to_string(),
            },

            tex_screen: TextureBinding {
                view: tex_screen.create_view(&Default::default()),
                device: tex_screen,
                layout: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                decl: "var screen: texture_storage_2d<rgba16float,write>".to_string(),
            },
            tex_read: TextureBinding {
                view: tex_read.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                }),
                device: tex_read,
                layout: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float {
                        filterable: !pass_f32,
                    },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                decl: "var pass_in: texture_2d_array<f32>".to_string(),
            },
            tex_write: TextureBinding {
                view: tex_write.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                }),
                device: tex_write,
                layout: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: if pass_f32 {
                        wgpu::TextureFormat::Rgba32Float
                    } else {
                        wgpu::TextureFormat::Rgba16Float
                    },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                decl: format!("var pass_out: texture_storage_2d_array<{pass_format},write>"),
            },
            channels: vec![
                TextureBinding {
                    view: channel0.create_view(&Default::default()),
                    device: channel0,
                    layout: channel_layout,
                    decl: "var channel0: texture_2d<f32>".to_string(),
                },
                TextureBinding {
                    view: channel1.create_view(&Default::default()),
                    device: channel1,
                    layout: channel_layout,
                    decl: "var channel1: texture_2d<f32>".to_string(),
                },
            ],

            nearest: SamplerBinding {
                layout: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                bind: wgpu.device.create_sampler(&Default::default()),
                decl: "var nearest: sampler".to_string(),
            },
            bilinear: SamplerBinding {
                layout: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                bind: wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }),
                decl: "var bilinear: sampler".to_string(),
            },
            trilinear: SamplerBinding {
                layout: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                bind: wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }),
                decl: "var trilinear: sampler".to_string(),
            },
            nearest_repeat: SamplerBinding {
                layout: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                bind: wgpu.device.create_sampler(&repeat),
                decl: "var nearest_repeat: sampler".to_string(),
            },
            bilinear_repeat: SamplerBinding {
                layout: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                bind: wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..repeat
                }),
                decl: "var bilinear_repeat: sampler".to_string(),
            },
            trilinear_repeat: SamplerBinding {
                layout: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                bind: wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..repeat
                }),
                decl: "var trilinear_repeat: sampler".to_string(),
            },
        }
    }

    fn to_vec(&self) -> Vec<&dyn Binding> {
        vec![
            &self.storage1,
            &self.storage2,
            &self.time,
            &self.mouse,
            &self.keys,
            &self.custom,
            &self.user_data,
            &self.debug_buffer,
            &self.dispatch_info,
            &self.tex_screen,
            &self.tex_read,
            &self.tex_write,
            &self.channels[0],
            &self.channels[1],
            &self.nearest,
            &self.bilinear,
            &self.trilinear,
            &self.nearest_repeat,
            &self.bilinear_repeat,
            &self.trilinear_repeat,
        ]
    }

    fn create_bind_group_layout(&self, wgpu: &WgpuContext) -> wgpu::BindGroupLayout {
        wgpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &self
                    .to_vec()
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: b.layout(),
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
    }

    pub fn create_pipeline_layout(&self, wgpu: &WgpuContext) -> wgpu::PipelineLayout {
        wgpu.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&self.create_bind_group_layout(wgpu)],
                push_constant_ranges: &[],
            })
    }

    pub fn create_bind_group(&self, wgpu: &WgpuContext) -> wgpu::BindGroup {
        wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.create_bind_group_layout(wgpu),
            entries: &self
                .to_vec()
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.binding(),
                })
                .collect::<Vec<_>>(),
        })
    }

    pub fn to_wgsl(&self) -> String {
        self.to_vec()
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let d = b.to_wgsl();
                if d.is_empty() {
                    String::new()
                } else {
                    format!("@group(0) @binding({i}) {};", b.to_wgsl())
                }
            })
            .intersperse("\n".to_string())
            .collect()
    }

    pub fn stage(&self, queue: &wgpu::Queue) {
        self.custom.stage(queue);
        self.user_data.stage(queue);
        self.time.stage(queue);
        self.mouse.stage(queue);
        self.keys.stage(queue);
    }
}
