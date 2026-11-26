//! Pixel-width world-space line segments over reverse scene depth.

use std::collections::HashMap;
use std::sync::Arc;

use crate::camera::Camera;
use crate::renderer::{FrameTarget, Gpu, DEPTH_FORMAT};

pub type LineRow = [f32; 12];
pub struct LineLayer {
    pub key: String,
    pub rows: Arc<[LineRow]>,
    pub look: [f32; 2],
}

pub struct LineOptions {
    pub visible: bool,
    pub bias: f32,
    pub feather: f32,
    pub layers: Vec<LineLayer>,
}

struct GpuLayer {
    rows: Arc<[LineRow]>,
    segments: wgpu::Buffer,
    params: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    look: Option<[f32; 4]>,
}

pub(crate) struct LinePass {
    pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    layers: HashMap<String, GpuLayer>,
}

impl LinePass {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native lines"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lines.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("native lines"),
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
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        });
        let globals = uniform(device, "line view and viewport", 80);
        Self {
            pipeline,
            globals,
            layers: HashMap::new(),
        }
    }

    pub(crate) fn encode(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &FrameTarget<'_>,
        camera: &Camera,
        options: &LineOptions,
    ) -> usize {
        self.layers
            .retain(|name, _| options.layers.iter().any(|layer| layer.key == *name));
        if !options.visible || options.layers.iter().all(|layer| layer.rows.is_empty()) {
            return 0;
        }
        let device = gpu.device();
        let queue = gpu.queue();
        let mut globals = [0.0; 20];
        let [w, h] = [target.width as f32, target.height as f32];
        globals[..16].copy_from_slice(&camera.view_projection(w / h).to_columns());
        globals[16..18].copy_from_slice(&[w, h]);
        queue.write_buffer(&self.globals, 0, bytemuck::cast_slice(&globals));
        for layer in options.layers.iter().filter(|layer| !layer.rows.is_empty()) {
            let bytes = bytemuck::cast_slice(layer.rows.as_ref());
            let size = bytes.len() as u64;
            let initialize = self
                .layers
                .get(&layer.key)
                .is_none_or(|layer| layer.segments.size() < size);
            if initialize {
                let segments = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("native line segments"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let params = uniform(device, "line layer settings", 16);
                let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("native line layer"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.globals.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: segments.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
                self.layers.insert(
                    layer.key.clone(),
                    GpuLayer {
                        rows: layer.rows.clone(),
                        segments,
                        params,
                        bindings,
                        look: None,
                    },
                );
            }
            let gpu_layer = self
                .layers
                .get_mut(&layer.key)
                .expect("prepared line layer");
            if initialize || !Arc::ptr_eq(&gpu_layer.rows, &layer.rows) {
                queue.write_buffer(&gpu_layer.segments, 0, bytes);
                gpu_layer.rows = layer.rows.clone();
            }
            let [scale, alpha] = layer.look;
            let look = [options.bias, options.feather, scale, alpha];
            if gpu_layer.look != Some(look) {
                queue.write_buffer(&gpu_layer.params, 0, bytemuck::cast_slice(&look));
                gpu_layer.look = Some(look);
            }
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("native lines"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        let mut draws = 0;
        for layer in options.layers.iter().filter(|layer| !layer.rows.is_empty()) {
            pass.set_bind_group(0, &self.layers[&layer.key].bindings, &[]);
            pass.draw(0..6, 0..layer.rows.len() as u32);
            draws += 1;
        }
        draws
    }
}

fn uniform(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}
