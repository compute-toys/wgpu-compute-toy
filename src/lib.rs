mod utils;

use wasm_bindgen::prelude::*;
use js_sys::JsString;

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

#[wasm_bindgen]
pub struct WgpuToyRenderer {
    wgpu: WgpuContext,
    width: u32,
    height: u32,
    params: wgpu::Buffer,
    buf_read: wgpu::Texture,
    buf_write: wgpu::Texture,
    compute_pipeline_layout: wgpu::PipelineLayout,
    compute_pipelines: Vec<wgpu::ComputePipeline>,
    compute_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    render_pipeline_srgb: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    staging_belt: wgpu::util::StagingBelt,
    frame_count: u32,
}

#[wasm_bindgen]
pub async fn init_wgpu(bind_id: JsString) -> WgpuContext {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Debug).expect("error initializing logger");

        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas();

        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let bind_id: String = bind_id.into();
        let bound_element = document.get_element_by_id(&bind_id).unwrap();

        // Set a background color for the canvas to make it easier to tell the where the canvas is for debugging purposes.
        canvas.style().set_css_text("background-color: crimson;");
        bound_element.append_child(&canvas).unwrap();
    }

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
    window.set_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
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

#[wasm_bindgen]
impl WgpuToyRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(wgpu: WgpuContext) -> WgpuToyRenderer {
        let size = wgpu.window.inner_size();

        // uniforms
        let params = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 3 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let img = wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let buf_read = wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let buf_write = wgpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
        });
        let sb0 = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (4 * 4 * size.width * size.height).into(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // compute pipeline
        let compute_bind_group_layout = wgpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let compute_pipeline_layout = wgpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&img.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 2, resource: sb0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&buf_read.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&buf_write.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&Default::default())) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                })) },
            ],
        });

        // render pipeline
        let render_shader = wgpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let render_bind_group_layout = wgpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });
        let format = wgpu.surface.get_preferred_format(&wgpu.adapter).unwrap();
        let render_pipeline = wgpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                entry_point: "fs_main",
                targets: &[format.into()],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let render_pipeline_srgb = wgpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                entry_point: "fs_main_srgb",
                targets: &[format.into()],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let render_bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&img.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&Default::default())) },
            ],
        });

        let staging_belt = wgpu::util::StagingBelt::new(0x100);
        let frame_count: u32 = 0;

        WgpuToyRenderer {
            wgpu,
            width: size.width,
            height: size.height,
            params,
            buf_read,
            buf_write,
            compute_pipeline_layout,
            compute_pipelines: vec![],
            compute_bind_group,
            render_pipeline,
            render_pipeline_srgb,
            render_bind_group,
            staging_belt,
            frame_count,
        }
    }

    pub fn render(&mut self) {
        let frame = self.wgpu.surface
            .get_current_texture()
            .expect("error getting texture from swap chain");
        let params_data = [self.width, self.height, self.frame_count];
        let params_bytes = bytemuck::bytes_of(&params_data);
        let mut encoder = self.wgpu.device.create_command_encoder(&Default::default());
        self.staging_belt.write_buffer(&mut encoder, &self.params, 0, wgpu::BufferSize::new(params_bytes.len() as wgpu::BufferAddress).unwrap(), &self.wgpu.device).copy_from_slice(params_bytes);
        self.staging_belt.finish();
        for pipeline in &self.compute_pipelines {
            {
                let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.dispatch(self.width / 16, self.height / 16, 1);
            }
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.buf_write,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.buf_read,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 4,
                });
        }
        self.frame_count += 1;
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
            match format {
                // TODO use sRGB viewFormats instead once the API stabilises?
                wgpu::TextureFormat::Bgra8Unorm => render_pass.set_pipeline(&self.render_pipeline_srgb),
                wgpu::TextureFormat::Bgra8UnormSrgb => render_pass.set_pipeline(&self.render_pipeline),
                _ => panic!("unrecognised surface format")
            }
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        self.wgpu.queue.submit(Some(encoder.finish()));
        wasm_bindgen_futures::spawn_local(self.staging_belt.recall());
        frame.present();
    }

    pub fn set_shader(&mut self, shader: JsString, entry_points: Vec<JsString>) {
        let shader: String = shader.into();
        let compute_shader = self.wgpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&shader)),
        });
        self.compute_pipelines = entry_points.iter().map(|name| {
            let name: String = name.into();
            self.wgpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&self.compute_pipeline_layout),
                module: &compute_shader,
                entry_point: &name,
            })
        }).collect()
    }

    pub fn set_frame_count(&mut self, frame_count: u32) {
        self.frame_count = frame_count;
    }
}
