use crate::context::WgpuContext;

#[derive(Copy, Clone, Debug)]
pub enum ColourSpace {
    Linear,
    Rgbe,
}

pub struct Blitter {
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    dest_format: wgpu::TextureFormat,
}

impl Blitter {
    pub fn new(
        wgpu: &WgpuContext,
        src: &wgpu::TextureView,
        src_space: ColourSpace,
        dest_format: wgpu::TextureFormat,
        filter: wgpu::FilterMode,
    ) -> Self {
        let render_shader = wgpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
            });
        let filterable = filter == wgpu::FilterMode::Linear;
        let render_bind_group_layout =
            wgpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                sample_type: wgpu::TextureSampleType::Float { filterable },
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(if filterable {
                                wgpu::SamplerBindingType::Filtering
                            } else {
                                wgpu::SamplerBindingType::NonFiltering
                            }),
                            count: None,
                        },
                    ],
                });
        Blitter {
            render_bind_group: wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
                        min_filter: filter,
                        mag_filter: filter,
                        ..Default::default()
                    })) },
                ],
            }),
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
                    entry_point: match (src_space, dest_format) {
                        // FIXME use sRGB viewFormats instead once the API stabilises
                        (ColourSpace::Linear, wgpu::TextureFormat::Bgra8Unorm) => "fs_main_linear_to_srgb",
                        (ColourSpace::Linear, wgpu::TextureFormat::Rgba8Unorm) => "fs_main_linear_to_srgb",
                        (ColourSpace::Linear, wgpu::TextureFormat::Bgra8UnormSrgb) => "fs_main", // format automatically performs sRGB encoding
                        (ColourSpace::Linear, wgpu::TextureFormat::Rgba8UnormSrgb) => "fs_main",
                        (ColourSpace::Linear, wgpu::TextureFormat::Rgba16Float) => "fs_main",
                        (ColourSpace::Rgbe, wgpu::TextureFormat::Rgba16Float) => "fs_main_rgbe_to_linear",
                        _ => panic!("Blitter: unrecognised conversion from {src_space:?} to {dest_format:?}")
                    },
                    targets: &[Some(dest_format.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            }),
            dest_format,
        }
    }

    pub fn blit(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    pub fn create_texture(
        &self,
        wgpu: &WgpuContext,
        width: u32,
        height: u32,
        mip_level_count: u32,
    ) -> wgpu::Texture {
        let texture = wgpu.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.dest_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        });
        let mut encoder = wgpu.device.create_command_encoder(&Default::default());
        let views: Vec<wgpu::TextureView> = (0..mip_level_count)
            .map(|base_mip_level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        self.blit(&mut encoder, &views[0]);
        for target_mip in 1..mip_level_count as usize {
            Blitter::new(
                wgpu,
                &views[target_mip - 1],
                ColourSpace::Linear,
                self.dest_format,
                wgpu::FilterMode::Linear,
            )
            .blit(&mut encoder, &views[target_mip]);
        }
        wgpu.queue.submit(Some(encoder.finish()));
        texture
    }
}
