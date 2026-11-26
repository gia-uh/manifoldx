//! Pixel-space Rect drawing for already laid-out rectangles.

use crate::renderer::Gpu;

pub type RectRow = [f32; 16];

pub(crate) struct RectPass {
    pipeline: wgpu::RenderPipeline,
    viewport: wgpu::Buffer,
    instances: Option<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl RectPass {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native Rect"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rect.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("native Rect"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            label: Some("native Rect viewport"),
            size: 16,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            pipeline,
            viewport,
            instances: None,
        }
    }

    pub(crate) fn draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: [u32; 2],
        rows: &[RectRow],
    ) {
        let bytes = bytemuck::cast_slice(rows);
        if self
            .instances
            .as_ref()
            .is_none_or(|(buffer, _)| buffer.size() < bytes.len() as u64)
        {
            let device = gpu.device();
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("native Rect instances"),
                size: bytes.len().next_power_of_two() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("native Rect resources"),
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
        let (buffer, group) = self.instances.as_ref().expect("Rect instance buffer");
        gpu.queue().write_buffer(buffer, 0, bytes);
        let viewport = [size[0] as f32, size[1] as f32, 0.0, 0.0];
        gpu.queue()
            .write_buffer(&self.viewport, 0, bytemuck::cast_slice(&viewport));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rect"),
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
        pass.draw(0..6, 0..rows.len() as u32);
    }
}
