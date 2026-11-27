use std::collections::HashMap;
use std::sync::Arc;

use crate::gui_paint::{Atlas, Frame, Source};
use crate::manual::MeshTexture;
use crate::renderer::{Gpu, RenderError};

pub(crate) struct GuiPass {
    pipeline: wgpu::RenderPipeline,
    viewport: wgpu::Buffer,
    instances: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    sampler: wgpu::Sampler,
    blank: wgpu::BindGroup,
    atlas: Option<(Arc<Atlas>, wgpu::BindGroup)>,
}

impl GuiPass {
    pub(crate) fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Self {
        let device = gpu.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native GUI shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gui.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("native GUI"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let viewport = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native GUI viewport"),
            size: 16,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blank = atlas_group(gpu, &pipeline, &sampler, [1, 1], &[255]);
        Self {
            pipeline,
            viewport,
            instances: None,
            sampler,
            blank,
            atlas: None,
        }
    }

    pub(crate) fn draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: [u32; 2],
        frame: &Frame,
        textures: &HashMap<u32, MeshTexture>,
    ) -> Result<(), RenderError> {
        let rows: Vec<_> = frame
            .quads
            .iter()
            .map(|quad| {
                [
                    quad.rect,
                    quad.clip,
                    quad.fill,
                    quad.border_color,
                    quad.uv,
                    [
                        quad.radius,
                        quad.border,
                        f32::from(u8::from(quad.exact)),
                        match quad.source {
                            Source::Solid => 0.0,
                            Source::Text => 1.0,
                            Source::Image(_) => 2.0,
                        },
                    ],
                ]
            })
            .collect();
        let bytes = bytemuck::cast_slice(&rows);
        if self
            .instances
            .as_ref()
            .is_none_or(|(buffer, _)| buffer.size() < bytes.len() as u64)
        {
            let device = gpu.device();
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("native GUI quads"),
                size: bytes.len().next_power_of_two() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("native GUI quads"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.viewport.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            self.instances = Some((buffer, group));
        }
        let (buffer, group) = self.instances.as_ref().unwrap();
        gpu.queue().write_buffer(buffer, 0, bytes);
        let viewport = [size[0] as f32, size[1] as f32, 0.0, 0.0];
        gpu.queue()
            .write_buffer(&self.viewport, 0, bytemuck::cast_slice(&viewport));
        if let Some(atlas) = &frame.atlas {
            if self
                .atlas
                .as_ref()
                .is_none_or(|(previous, _)| !Arc::ptr_eq(previous, atlas))
            {
                self.atlas = Some((
                    atlas.clone(),
                    atlas_group(gpu, &self.pipeline, &self.sampler, atlas.size, &atlas.alpha),
                ));
            }
        } else {
            self.atlas = None;
        }
        let mut images = HashMap::new();
        for quad in &frame.quads {
            if let Source::Image(id) = quad.source {
                let image = textures
                    .get(&id)
                    .ok_or_else(|| RenderError::Resource(format!("unknown GUI texture {id}")))?;
                images.entry(id).or_insert_with(|| {
                    texture_group(gpu.device(), &self.pipeline, &image.view, &image.sampler)
                });
            }
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        let mut start = 0;
        while start < frame.quads.len() {
            let source = frame.quads[start].source;
            let run = frame.quads[start..]
                .iter()
                .take_while(|quad| quad.source == source)
                .count();
            let texture = match source {
                Source::Solid => &self.blank,
                Source::Text => &self.atlas.as_ref().expect("text quads have an atlas").1,
                Source::Image(id) => &images[&id],
            };
            pass.set_bind_group(1, texture, &[]);
            pass.draw(0..6, start as u32..(start + run) as u32);
            start += run;
        }
        Ok(())
    }
}

fn atlas_group(
    gpu: &Gpu,
    pipeline: &wgpu::RenderPipeline,
    sampler: &wgpu::Sampler,
    size: [u32; 2],
    alpha: &[u8],
) -> wgpu::BindGroup {
    let extent = wgpu::Extent3d {
        width: size[0],
        height: size[1],
        depth_or_array_layers: 1,
    };
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("native GUI text atlas"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        texture.as_image_copy(),
        alpha,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size[0]),
            rows_per_image: None,
        },
        extent,
    );
    texture_group(
        gpu.device(),
        pipeline,
        &texture.create_view(&Default::default()),
        sampler,
    )
}

fn texture_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native GUI image"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
