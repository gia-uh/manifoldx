//! Hand-written material WGSL with the established mesh resource layout.

use std::collections::{BTreeSet, HashMap};

use crate::fragment::make_pipeline;
use crate::renderer::{Gpu, RenderError};

pub(crate) const GLOBAL_FLOATS: usize = 140;

pub struct ManualData {
    pub source: String,
    pub uniforms: Vec<f32>,
    pub textures: Vec<(u32, u32)>,
    pub blend: u8,
    pub axis: bool,
}

pub(crate) struct MeshTexture {
    pub(crate) group: wgpu::BindGroup,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
}

pub(crate) struct ManualPipeline {
    pub(crate) pipelines: [wgpu::RenderPipeline; 2],
    pub(crate) translucent: bool,
    pub(crate) axis: bool,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    source: String,
    values: Vec<f32>,
    textures: Vec<(u32, u32)>,
    blend: u8,
}

impl ManualPipeline {
    pub(crate) fn new(
        gpu: &Gpu,
        data: &ManualData,
        vertex: wgpu::VertexBufferLayout<'_>,
        format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let device = gpu.device();
        let topology = if data.axis {
            wgpu::PrimitiveTopology::LineList
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };
        let blend = match data.blend {
            0 => None,
            1 => Some(wgpu::BlendState::ALPHA_BLENDING),
            2 => {
                let add = wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                };
                Some(wgpu::BlendState {
                    color: add,
                    alpha: add,
                })
            }
            _ => {
                return Err(RenderError::Resource(
                    "invalid manual WGSL blend state".into(),
                ))
            }
        };
        let max = device.limits().max_bindings_per_bind_group;
        let mut slots = BTreeSet::from([0, 1, 2, 3]);
        let mut entries = vec![
            buffer_layout(0, wgpu::BufferBindingType::Uniform),
            buffer_layout(1, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer_layout(2, wgpu::BufferBindingType::Uniform),
            buffer_layout(3, wgpu::BufferBindingType::Uniform),
        ];
        for &(index, _) in &data.textures {
            if index >= max - 1 || !slots.insert(index) || !slots.insert(index + 1) {
                return Err(RenderError::Resource(format!(
                    "overlapping or unsupported manual WGSL sampler/view slots {index}/{}",
                    index.saturating_add(1)
                )));
            }
            entries.push(layout_entry(
                index,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            ));
            entries.push(layout_entry(
                index + 1,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ));
        }
        let size = uniform_size(data)?;
        if size > device.limits().max_uniform_buffer_binding_size {
            return Err(RenderError::Resource(
                "manual WGSL uniform exceeds device limits".into(),
            ));
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("manual material group 0"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manual material layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manual material WGSL"),
            source: wgpu::ShaderSource::Wgsl(data.source.as_str().into()),
        });
        let pipelines = [wgpu::FrontFace::Ccw, wgpu::FrontFace::Cw].map(|face| {
            make_pipeline(
                device,
                &module,
                &pipeline_layout,
                std::slice::from_ref(&vertex),
                format,
                "fs_main",
                (face, blend, topology),
            )
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Resource(format!(
                "manual material WGSL: {error}"
            )));
        }
        let uniform = gpu_buffer(
            device,
            "manual material uniform",
            size,
            wgpu::BufferUsages::UNIFORM,
        );
        if !data.uniforms.is_empty() {
            gpu.queue()
                .write_buffer(&uniform, 0, bytemuck::cast_slice(&data.uniforms));
        }
        Ok(Self {
            pipelines,
            translucent: blend.is_some(),
            axis: data.axis,
            layout,
            uniform,
            source: data.source.clone(),
            values: data.uniforms.clone(),
            textures: data.textures.clone(),
            blend: data.blend,
        })
    }

    pub(crate) fn update(&mut self, gpu: &Gpu, data: &ManualData) -> Result<bool, RenderError> {
        if self.source != data.source
            || self.textures != data.textures
            || self.blend != data.blend
            || self.axis != data.axis
            || uniform_size(data)? != self.uniform.size()
        {
            return Ok(false);
        }
        if self.values != data.uniforms {
            let mut values = data.uniforms.clone();
            values.resize(self.uniform.size() as usize / 4, 0.0);
            gpu.queue()
                .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&values));
            self.values.clone_from(&data.uniforms);
        }
        Ok(true)
    }

    pub(crate) fn samples(&self, target: u32) -> bool {
        self.textures.iter().any(|&(_, texture)| texture == target)
    }
}

fn uniform_size(data: &ManualData) -> Result<u64, RenderError> {
    if data.uniforms.iter().any(|value| !value.is_finite()) {
        return Err(RenderError::Resource(
            "manual WGSL uniform must be finite".into(),
        ));
    }
    Ok((data.uniforms.len() as u64 * 4)
        .max(16)
        .next_multiple_of(16))
}

pub(crate) struct ManualFrame {
    pub(crate) groups: HashMap<u32, wgpu::BindGroup>,
    globals: wgpu::Buffer,
    transforms: wgpu::Buffer,
    lights: wgpu::Buffer,
}

impl ManualFrame {
    pub(crate) fn new(device: &wgpu::Device, capacity: usize) -> Result<Self, RenderError> {
        let bytes = (capacity as u64 * 64).max(64);
        if bytes > device.limits().max_storage_buffer_binding_size {
            return Err(RenderError::Resource(
                "manual WGSL transforms exceed device limits".into(),
            ));
        }
        Ok(Self {
            groups: HashMap::new(),
            globals: gpu_buffer(
                device,
                "manual WGSL globals",
                (GLOBAL_FLOATS * 4) as u64,
                wgpu::BufferUsages::UNIFORM,
            ),
            transforms: gpu_buffer(
                device,
                "manual WGSL transforms",
                bytes,
                wgpu::BufferUsages::STORAGE,
            ),
            lights: gpu_buffer(
                device,
                "manual WGSL lights",
                128,
                wgpu::BufferUsages::UNIFORM,
            ),
        })
    }

    pub(crate) fn write(
        &self,
        queue: &wgpu::Queue,
        globals: &[f32; GLOBAL_FLOATS],
        transforms: &[[f32; 16]],
        lights: &[[f32; 8]; 4],
    ) {
        queue.write_buffer(&self.globals, 0, bytemuck::cast_slice(globals));
        queue.write_buffer(&self.transforms, 0, bytemuck::cast_slice(transforms));
        queue.write_buffer(&self.lights, 0, bytemuck::cast_slice(lights));
    }

    pub(crate) fn bind(
        &mut self,
        device: &wgpu::Device,
        id: u32,
        pipeline: &ManualPipeline,
        textures: &HashMap<u32, MeshTexture>,
    ) -> Result<(), RenderError> {
        if let std::collections::hash_map::Entry::Vacant(slot) = self.groups.entry(id) {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.transforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pipeline.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.lights.as_entire_binding(),
                },
            ];
            for &(index, key) in &pipeline.textures {
                let texture = textures.get(&key).ok_or_else(|| {
                    RenderError::Resource(format!("manual WGSL references unknown texture {key}"))
                })?;
                entries.push(wgpu::BindGroupEntry {
                    binding: index,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: index + 1,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                });
            }
            slot.insert(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("manual material resources"),
                layout: &pipeline.layout,
                entries: &entries,
            }));
        }
        Ok(())
    }
}

pub(crate) fn gpu_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn buffer_layout(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    layout_entry(
        binding,
        wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    )
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        ty,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        count: None,
    }
}
