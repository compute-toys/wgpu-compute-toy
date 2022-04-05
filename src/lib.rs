mod utils;

use wasm_bindgen::prelude::*;
use naga::front::wgsl;
use naga::front::wgsl::ParseError;
use num::Integer;
use bitvec::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct WgpuContext {
    window: winit::window::Window,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Screen {
    size: [u32; 2],
}

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
    screen: wgpu::Buffer,
    time: wgpu::Buffer,
    mouse: wgpu::Buffer,
    keys: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    tex_read: wgpu::Texture,
    tex_write: wgpu::Texture,
    tex_screen: wgpu::Texture,
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
                    Err(error) => log::error!("Error calling registered error callback, error: {:?}", error),
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

#[wasm_bindgen]
pub struct WgpuToyRenderer {
    wgpu: WgpuContext,
    screen: Screen,
    time: Time,
    mouse: Mouse,
    keys: BitArr!(for NUM_KEYCODES, in u8, Lsb0),
    uniforms: Uniforms,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline_layout: wgpu::PipelineLayout,
    compute_pipelines: Vec<(wgpu::ComputePipeline, [u32; 3])>,
    compute_bind_group: wgpu::BindGroup,
    render_bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    staging_belt: wgpu::util::StagingBelt,
    on_error_cb: ErrorCallback,
    channel0: wgpu::Texture,
}


const COMPUTE_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
    label: None,
    entries: &[
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
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
    ],
};

const RENDER_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
    label: None,
    entries: &[
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
    ],
};

#[cfg(target_arch = "wasm32")]
fn init_window(bind_id: String) -> Option<winit::window::Window> {
    console_log::init_with_level(log::Level::Info).ok()?;
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let event_loop = winit::event_loop::EventLoop::new();
    let win = web_sys::window()?;
    let doc = win.document()?;
    let element = doc.get_element_by_id(&bind_id)?;
    use wasm_bindgen::JsCast;
    let canvas = element.dyn_into::<web_sys::HtmlCanvasElement>().ok()?;
    use winit::platform::web::WindowBuilderExtWebSys;
    winit::window::WindowBuilder::new()
        .with_canvas(Some(canvas))
        .build(&event_loop).ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn init_window(_: String) -> Option<winit::window::Window> {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new();
    winit::window::Window::new(&event_loop).ok()
}

// FIXME: async fn(&str) doesn't currently work with wasm_bindgen: https://stackoverflow.com/a/63655324/78204
#[wasm_bindgen]
pub async fn init_wgpu(bind_id: String) -> WgpuContext {
    let window = init_window(bind_id).expect("failed to create window");
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: Default::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("error finding adapter");
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .expect("error creating device");
    let size = window.inner_size();
    let format = surface.get_preferred_format(&adapter).unwrap();
    surface.configure(&device, &wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo, // vsync
    });
    WgpuContext {
        window,
        adapter,
        device,
        queue,
        surface,
    }
}

fn create_uniforms(wgpu: &WgpuContext, width: u32, height: u32) -> Uniforms {
    Uniforms {
        screen: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Screen>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        time: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Time>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        mouse: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Mouse>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        keys: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<bitvec::BitArr!(for NUM_KEYCODES, in u8)>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }),
        storage_buffer: wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (4 * 4 * width * height).into(),
            usage: wgpu::BufferUsages::STORAGE,
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
            format: wgpu::TextureFormat::Rgba16Float,
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
            format: wgpu::TextureFormat::Rgba16Float,
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

fn create_compute_bind_group(wgpu: &WgpuContext, layout: &wgpu::BindGroupLayout, uniforms: &Uniforms, channel0: &wgpu::Texture) -> wgpu::BindGroup {
    wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.screen.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniforms.time.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: uniforms.mouse.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniforms.keys.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&uniforms.tex_screen.create_view(&Default::default())) },
            wgpu::BindGroupEntry { binding: 5, resource: uniforms.storage_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&uniforms.tex_read.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&uniforms.tex_write.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&Default::default())) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&channel0.create_view(&Default::default())) },
        ],
    })
}

fn create_render_bind_group(wgpu: &WgpuContext, layout: &wgpu::BindGroupLayout, uniforms: &Uniforms) -> wgpu::BindGroup {
    wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&uniforms.tex_screen.create_view(&Default::default())) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&Default::default())) },
        ],
    })
}

fn stage(staging_belt: &mut wgpu::util::StagingBelt, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, data: &[u8], buffer: &wgpu::Buffer) {
    let size = wgpu::BufferSize::new(data.len() as u64).expect("size must be non-zero");
    staging_belt.write_buffer(encoder, buffer, 0, size, device)
        .copy_from_slice(data);
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
        let uniforms = create_uniforms(&wgpu, size.width, size.height);
        let compute_bind_group_layout = wgpu.device.create_bind_group_layout(&COMPUTE_BIND_GROUP_LAYOUT_DESCRIPTOR);
        let render_shader = wgpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let render_bind_group_layout = wgpu.device.create_bind_group_layout(&RENDER_BIND_GROUP_LAYOUT_DESCRIPTOR);
        let format = wgpu.surface.get_preferred_format(&wgpu.adapter).unwrap();

        let channel0 = wgpu.device.create_texture(
            &wgpu::TextureDescriptor {
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
            }
        );

        WgpuToyRenderer {
            compute_pipeline_layout: wgpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            }),
            compute_bind_group: create_compute_bind_group(&wgpu, &compute_bind_group_layout, &uniforms, &channel0),
            compute_pipelines: vec![],
            render_bind_group: create_render_bind_group(&wgpu, &render_bind_group_layout, &uniforms),
            render_pipeline: wgpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&wgpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&render_bind_group_layout],
                    push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: match format {
                        // TODO use sRGB viewFormats instead once the API stabilises?
                        wgpu::TextureFormat::Bgra8Unorm => "fs_main_srgb",
                        wgpu::TextureFormat::Bgra8UnormSrgb => "fs_main",
                        _ => panic!("unrecognised surface format")
                    },
                    targets: &[format.into()],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            }),
            screen: Screen {
                size: [size.width, size.height],
            },
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
            wgpu,
            uniforms,
            compute_bind_group_layout,
            render_bind_group_layout,
            on_error_cb: ErrorCallback(None),
            channel0,
        }
    }

    pub fn render(&mut self) {
        let frame = self.wgpu.surface
            .get_current_texture()
            .expect("error getting texture from swap chain");
        let mut encoder = self.wgpu.device.create_command_encoder(&Default::default());
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, bytemuck::bytes_of(&self.screen), &self.uniforms.screen);
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, bytemuck::bytes_of(&self.time), &self.uniforms.time);
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, bytemuck::bytes_of(&self.mouse), &self.uniforms.mouse);
        stage(&mut self.staging_belt, &self.wgpu.device, &mut encoder, &self.keys.as_raw_slice(), &self.uniforms.keys);
        self.staging_belt.finish();
        for (pipeline, workgroup_size) in &self.compute_pipelines {
            {
                let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.dispatch(self.screen.size[0].div_ceil(&workgroup_size[0]), self.screen.size[1].div_ceil(&workgroup_size[1]), 1);
            }
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
                    width: self.screen.size[0],
                    height: self.screen.size[1],
                    depth_or_array_layers: 4,
                });
        }
        self.time.frame += 1;
        // blit the output texture to the framebuffer
        {
            let format = self.wgpu.surface.get_preferred_format(&self.wgpu.adapter).unwrap();
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(format),
                ..Default::default()
            });
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        self.wgpu.queue.submit(Some(encoder.finish()));
        wasm_bindgen_futures::spawn_local(self.staging_belt.recall());
        frame.present();
    }

    fn handle_error(&self, e: ParseError, wgsl: &str) {
        let prelude: String = include_str!("prelude.wgsl").into();
        let prelude_len = count_newlines(&prelude); // in case we need to report errors
        let (row, col) = e.location(&wgsl);
        let summary = e.emit_to_string(&wgsl);
        self.on_error_cb.call(&summary, row - prelude_len, col);
    }

    pub fn set_shader(&mut self, shader: &str) {
        let mut wgsl: String = include_str!("prelude.wgsl").into();
        let shader: String = shader.into();
        wgsl.push_str(&shader);
        match wgsl::parse_str(&wgsl) {
            Ok(module) => {
                let entry_points: Vec<_> = module.entry_points.iter()
                    .filter(|f| f.stage == naga::ShaderStage::Compute).collect();
                let compute_shader = self.wgpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&wgsl)),
                });
                self.compute_pipelines = entry_points.iter().map(|entry_point| {
                    (self.wgpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&self.compute_pipeline_layout),
                        module: &compute_shader,
                        entry_point: &entry_point.name,
                    }), entry_point.workgroup_size)
                }).collect();
            },
            Err(e) => {
                log::error!("Error parsing WGSL: {}", e);
                self.handle_error(e, &wgsl);
            },
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

    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen.size = [width, height];
        self.time.frame = 0;
        self.uniforms = create_uniforms(&self.wgpu, width, height);
        self.compute_bind_group = create_compute_bind_group(&self.wgpu, &self.compute_bind_group_layout, &self.uniforms, &self.channel0);
        self.render_bind_group = create_render_bind_group(&self.wgpu, &self.render_bind_group_layout, &self.uniforms);
        self.wgpu.window.set_inner_size(winit::dpi::LogicalSize::new(width, height));
    }

    pub fn on_error(&mut self, callback: js_sys::Function) {
        self.on_error_cb = ErrorCallback(Some(callback));

        // FIXME: remove pending resolution of this issue: https://github.com/gfx-rs/wgpu/issues/2130
        let prelude: String = include_str!("prelude.wgsl").into();
        let prelude_len = count_newlines(&prelude);
        let re = regex::Regex::new(r"Parser:\s:(\d+):(\d+)\s([\s\S]*?)\s+Shader").unwrap();
        let on_error_cb = self.on_error_cb.clone();
        self.wgpu.device.on_uncaptured_error(move |e: wgpu::Error| {
            let err = &e.to_string();
            for cap in re.captures_iter(err) {
                let row = cap[1].parse().unwrap_or(prelude_len);
                let col = cap[2].parse().unwrap_or(0);
                let summary = &cap[3];
                on_error_cb.call(summary, row - prelude_len, col);
            }
        });
    }

    pub fn load_channel(&mut self, bytes: &[u8]) {
        let im = image::load_from_memory(bytes).unwrap();
        use image::GenericImageView;
        let (width, height) = im.dimensions();
        self.channel0 = self.wgpu.device.create_texture(
            &wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: None,
            }
        );
        self.wgpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.channel0,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &im.to_rgba8(),
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
    }
}
