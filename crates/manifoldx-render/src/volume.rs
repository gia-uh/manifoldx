//! Direct volume rendering with R32Float voxels and transfer textures.

use std::collections::HashMap;

use manifoldx_core::ecs::{ColumnData, EntityId, World};

use crate::manual::{gpu_buffer, GLOBAL_FLOATS};
use crate::math::{model_from_transform_row, Mat4};
use crate::renderer::{FrameTarget, Gpu, RenderError};
use crate::scene::InstanceSource;
use crate::sprite::{read, scalar, world_error};

const LUT_SIZE: u32 = 256;
const ROW_BYTES: u64 = 160;

pub struct VolumeMaterialData {
    pub range_density: [f32; 3],
    pub step: Option<f32>,
    pub max_steps: u32,
    pub color: Vec<u8>,
    pub opacity: Vec<f32>,
}

struct Voxels {
    size: wgpu::Extent3d,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct Transfer {
    color: Voxels,
    opacity: Voxels,
    data: VolumeMaterialData,
}

struct VolumeDraw {
    resource: u32,
    material: u32,
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
    values: [f32; ROW_BYTES as usize / 4],
}

struct View {
    globals: wgpu::Buffer,
    depth_group: Option<(wgpu::TextureView, wgpu::BindGroup)>,
    draws: HashMap<EntityId, VolumeDraw>,
    order: Vec<EntityId>,
}

pub(crate) struct VolumePass {
    pipeline: wgpu::RenderPipeline,
    globals_layout: wgpu::BindGroupLayout,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    volumes: HashMap<u32, Voxels>,
    materials: HashMap<u32, Transfer>,
    views: HashMap<u32, View>,
}

impl VolumePass {
    pub(crate) fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Result<Self, RenderError> {
        let device = gpu.device();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            ty,
            visibility: wgpu::ShaderStages::FRAGMENT,
            count: None,
        };
        let texture = |view_dimension| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension,
            multisampled: false,
        };
        let sampler_type = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering);
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native DVR camera layout"),
            entries: &[
                layout_entry(0, uniform),
                layout_entry(
                    1,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native DVR textures and transform"),
            entries: &[
                layout_entry(0, texture(wgpu::TextureViewDimension::D3)),
                layout_entry(1, sampler_type),
                layout_entry(2, texture(wgpu::TextureViewDimension::D2)),
                layout_entry(3, texture(wgpu::TextureViewDimension::D2)),
                layout_entry(4, sampler_type),
                layout_entry(5, uniform),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("native DVR pipeline layout"),
            bind_group_layouts: &[Some(&globals_layout), Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native volume compute along view rays"),
            source: wgpu::ShaderSource::Wgsl(include_str!("volume.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("native direct volume render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Resource(format!("DVR pipeline: {error}")));
        }
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("native nearest volume sampler"),
            ..Default::default()
        });
        Ok(Self {
            pipeline,
            globals_layout,
            layout,
            sampler,
            volumes: HashMap::new(),
            materials: HashMap::new(),
            views: HashMap::new(),
        })
    }

    pub(crate) fn voxels(
        &mut self,
        gpu: &Gpu,
        id: u32,
        dims: [u32; 3],
        bytes: &[u8],
    ) -> Result<(), RenderError> {
        let limit = gpu.device().limits().max_texture_dimension_3d;
        if id == 0
            || dims.iter().any(|size| *size == 0 || *size > limit)
            || dims.iter().map(|&n| u64::from(n)).product::<u64>() * 4 != bytes.len() as u64
        {
            return Err(RenderError::Resource(
                "invalid native volume dimensions or payload".into(),
            ));
        }
        let size = extent(dims);
        let voxels = self.volumes.entry(id).or_insert_with(|| {
            Voxels::new(
                gpu.device(),
                size,
                wgpu::TextureDimension::D3,
                wgpu::TextureFormat::R32Float,
            )
        });
        if voxels.size != size {
            return Err(RenderError::Resource(
                "native volume update must preserve voxel dimensions".into(),
            ));
        }
        voxels.write(gpu.queue(), bytes);
        Ok(())
    }

    pub(crate) fn material(
        &mut self,
        gpu: &Gpu,
        id: u32,
        data: VolumeMaterialData,
    ) -> Result<(), RenderError> {
        let [min, max, _] = data.range_density;
        if data.color.len() != LUT_SIZE as usize * 4
            || data.opacity.len() != LUT_SIZE as usize
            || data.opacity.iter().any(|v| !v.is_finite())
            || data.range_density.iter().any(|v| !v.is_finite())
            || min >= max
            || data.max_steps == 0
            || data
                .step
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(RenderError::Resource(
                "invalid DVR transfer material".into(),
            ));
        }
        if let Some(transfer) = self.materials.get_mut(&id) {
            if transfer.data.color != data.color {
                transfer.color.write(gpu.queue(), &data.color);
            }
            if transfer.data.opacity != data.opacity {
                transfer
                    .opacity
                    .write(gpu.queue(), bytemuck::cast_slice(&data.opacity));
            }
            transfer.data = data;
        } else {
            let size = extent([LUT_SIZE, 1, 1]);
            let color = Voxels::new(
                gpu.device(),
                size,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Rgba8Unorm,
            );
            let opacity = Voxels::new(
                gpu.device(),
                size,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::R32Float,
            );
            color.write(gpu.queue(), &data.color);
            opacity.write(gpu.queue(), bytemuck::cast_slice(&data.opacity));
            self.materials.insert(
                id,
                Transfer {
                    color,
                    opacity,
                    data,
                },
            );
        }
        Ok(())
    }

    pub(crate) fn drop_view(&mut self, id: u32) {
        self.views.remove(&id);
    }

    pub(crate) fn prepare(
        &mut self,
        gpu: &Gpu,
        world: &World,
        source: InstanceSource,
        layers: Option<&std::collections::BTreeSet<i32>>,
        globals: &[f32; GLOBAL_FLOATS],
        index: u32,
    ) -> Result<(), RenderError> {
        if let Some(view) = self.views.get_mut(&index) {
            view.order.clear();
        }
        let (Some(volume), Some(material)) = (world.component_index("Volume"), source.material)
        else {
            return Ok(());
        };
        let snapshot = world.snapshot();
        let members = world
            .memberships(volume, &snapshot.entities)
            .map_err(world_error)?;
        let ids: Vec<_> = snapshot
            .entities
            .into_iter()
            .zip(members)
            .filter_map(|(id, member)| member.then_some(id))
            .collect();
        if ids.is_empty() {
            return Ok(());
        }
        let models = read(world, source.transform, &ids)?;
        let materials = read(world, material, &ids)?;
        let ColumnData::F32(transforms) = models.data else {
            unreachable!("validated transform schema")
        };
        let ColumnData::U32(material_ids) = materials.data else {
            unreachable!("validated material schema")
        };
        let resources = scalar(world, "Volume", &ids, 0.0)?;
        let layer_ids = layers
            .map(|_| scalar(world, "Layer", &ids, 0.0))
            .transpose()?;
        let device = gpu.device();
        let view = self.views.entry(index).or_insert_with(|| {
            let globals = gpu_buffer(
                device,
                "native DVR globals",
                (GLOBAL_FLOATS * 4) as u64,
                wgpu::BufferUsages::UNIFORM,
            );
            View {
                globals,
                depth_group: None,
                draws: HashMap::new(),
                order: Vec::new(),
            }
        });
        gpu.queue()
            .write_buffer(&view.globals, 0, bytemuck::cast_slice(globals));
        for (i, &id) in ids.iter().enumerate() {
            if !models.membership[i]
                || !materials.membership[i]
                || layers
                    .zip(layer_ids.as_ref())
                    .is_some_and(|(filter, ids)| !filter.contains(&(ids[i] as i32)))
            {
                continue;
            }
            let resource = resources[i] as u32;
            let material = material_ids[i];
            let Some(volume) = self.volumes.get(&resource) else {
                continue;
            };
            let Some(transfer) = self.materials.get(&material) else {
                continue;
            };
            let model = model_from_transform_row(&transforms[i * 10..(i + 1) * 10]);
            let values = uniforms(model, volume.size, &transfer.data)?;
            if view
                .draws
                .get(&id)
                .is_none_or(|row| row.resource != resource || row.material != material)
            {
                let buffer = gpu_buffer(
                    device,
                    "native DVR entity transform",
                    ROW_BYTES,
                    wgpu::BufferUsages::UNIFORM,
                );
                let sampler = wgpu::BindingResource::Sampler(&self.sampler);
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("native DVR volume binding"),
                    layout: &self.layout,
                    entries: &[
                        binding(0, wgpu::BindingResource::TextureView(&volume.view)),
                        binding(1, sampler.clone()),
                        binding(2, wgpu::BindingResource::TextureView(&transfer.color.view)),
                        binding(
                            3,
                            wgpu::BindingResource::TextureView(&transfer.opacity.view),
                        ),
                        binding(4, sampler),
                        binding(5, buffer.as_entire_binding()),
                    ],
                });
                gpu.queue()
                    .write_buffer(&buffer, 0, bytemuck::cast_slice(&values));
                view.draws.insert(
                    id,
                    VolumeDraw {
                        buffer,
                        group,
                        values,
                        resource,
                        material,
                    },
                );
            } else {
                let entry = view.draws.get_mut(&id).expect("draw key checked");
                if entry.values != values {
                    gpu.queue()
                        .write_buffer(&entry.buffer, 0, bytemuck::cast_slice(&values));
                    entry.values = values;
                }
            }
            view.order.push(id);
        }
        let active: std::collections::HashSet<_> = view.order.iter().collect();
        view.draws.retain(|id, _| active.contains(id));
        Ok(())
    }

    pub(crate) fn has_draws(&self, index: u32) -> bool {
        self.views
            .get(&index)
            .is_some_and(|view| !view.order.is_empty())
    }

    pub(crate) fn encode(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &FrameTarget<'_>,
        index: u32,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> usize {
        let view = self
            .views
            .get_mut(&index)
            .expect("DVR view already prepared");
        if view
            .depth_group
            .as_ref()
            .is_none_or(|(depth, _)| depth != target.depth)
        {
            let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("native DVR camera and opaque depth"),
                layout: &self.globals_layout,
                entries: &[
                    binding(0, view.globals.as_entire_binding()),
                    binding(1, wgpu::BindingResource::TextureView(target.depth)),
                ],
            });
            view.depth_group = Some((target.depth.clone(), group));
        }
        // The prior pass stores opaque depth. Sample it without attaching it
        // to the volume pass, which only blends into the existing scene color.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("manifoldx-native-dvr"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: timestamps,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(
            0,
            &view.depth_group.as_ref().expect("DVR depth bound").1,
            &[],
        );
        for id in &view.order {
            pass.set_bind_group(1, &view.draws[id].group, &[]);
            pass.draw(0..3, 0..1);
        }
        view.order.len()
    }
}

fn extent([width, height, depth_or_array_layers]: [u32; 3]) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }
}

impl Voxels {
    fn new(
        device: &wgpu::Device,
        size: wgpu::Extent3d,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native volume field or transfer function"),
            size,
            dimension,
            format,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self {
            size,
            texture,
            view,
        }
    }

    fn write(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size.width * 4),
                rows_per_image: Some(self.size.height),
            },
            self.size,
        );
    }
}

fn binding(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn uniforms(
    model: Mat4,
    size: wgpu::Extent3d,
    material: &VolumeMaterialData,
) -> Result<[f32; 40], RenderError> {
    let columns = model.to_columns();
    let [x, y, z, translation]: [[f32; 4]; 4] = *columns
        .as_chunks::<4>()
        .0
        .first_chunk::<4>()
        .expect("a model has four columns");
    let mut inverse = Mat4::IDENTITY;
    let mut scale = [0.0; 3];
    for (index, axis) in [x, y, z].iter().enumerate() {
        let length2 = axis[..3].iter().map(|v: &f32| v * v).sum::<f32>();
        if !length2.is_finite() || length2 <= 1e-20 {
            return Err(RenderError::Resource(
                "native volume transform must be invertible".into(),
            ));
        }
        scale[index] = length2.sqrt();
        for i in 0..3 {
            inverse.0[index * 4 + i] = axis[i] / length2;
        }
        inverse.0[index * 4 + 3] = -inverse.0[index * 4..index * 4 + 3]
            .iter()
            .zip(&translation[..3])
            .map(|(a, b)| a * b)
            .sum::<f32>();
    }
    let auto = (scale[0] / size.width as f32)
        .min(scale[1] / size.height as f32)
        .min(scale[2] / size.depth_or_array_layers as f32)
        * 0.5;
    let mut values = [0.0; 40];
    values[..16].copy_from_slice(&columns);
    values[16..32].copy_from_slice(&inverse.to_columns());
    values[32..35].copy_from_slice(&material.range_density);
    values[35] = material.step.unwrap_or(auto);
    values[36] = f32::from_bits(material.max_steps);
    Ok(values)
}
