//! Billboard sprites and the colormap/starfield WGSL material path.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

use manifoldx_core::ecs::{ColumnData, ComponentReadout, EntityId, World};

use crate::fragment::{make_pipeline, upload_1d};
use crate::manual::{gpu_buffer, GLOBAL_FLOATS};
use crate::math::model_from_transform_row;
use crate::renderer::{Gpu, RenderError};
use crate::scene::InstanceSource;

const QUAD: [[f32; 3]; 6] = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
];

pub struct SpriteData {
    pub source: String,
    pub values: [f32; 4],
    pub rgba: Vec<u8>,
}

struct SpriteMaterial {
    pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    view: wgpu::TextureView,
    data: SpriteData,
}

#[derive(Default, PartialEq)]
struct Rows {
    models: Vec<[f32; 16]>,
    scalars: Vec<f32>,
    radii: Vec<f32>,
}

struct SpriteFrame {
    globals: wgpu::Buffer,
    transforms: wgpu::Buffer,
    scalars: wgpu::Buffer,
    radii: wgpu::Buffer,
    groups: HashMap<u32, wgpu::BindGroup>,
    cached: Rows,
    draws: Vec<(u32, Range<u32>)>,
}

pub(crate) struct SpritePass {
    format: wgpu::TextureFormat,
    quad: wgpu::Buffer,
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    materials: HashMap<u32, SpriteMaterial>,
    frames: [Option<SpriteFrame>; 2],
    capacity: usize,
}

impl SpritePass {
    pub(crate) fn new(
        gpu: &Gpu,
        format: wgpu::TextureFormat,
        capacity: usize,
    ) -> Result<Self, RenderError> {
        let device = gpu.device();
        if capacity as u64 * 64 > device.limits().max_storage_buffer_binding_size {
            return Err(RenderError::Resource(
                "sprite rows exceed device buffer limits".into(),
            ));
        }
        let quad = gpu_buffer(device, "native sprite quad", 72, wgpu::BufferUsages::VERTEX);
        gpu.queue()
            .write_buffer(&quad, 0, bytemuck::cast_slice(&QUAD));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprite clamp sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native sprite resources"),
            entries: &[
                buffer_binding(
                    0,
                    wgpu::BufferBindingType::Uniform,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                ),
                buffer_binding(
                    1,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    wgpu::ShaderStages::VERTEX,
                ),
                buffer_binding(
                    2,
                    wgpu::BufferBindingType::Uniform,
                    wgpu::ShaderStages::FRAGMENT,
                ),
                buffer_binding(
                    3,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    wgpu::ShaderStages::VERTEX,
                ),
                buffer_binding(
                    4,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    wgpu::ShaderStages::VERTEX,
                ),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
            ],
        });
        Ok(Self {
            format,
            quad,
            sampler,
            layout,
            materials: HashMap::new(),
            frames: [None, None],
            capacity,
        })
    }

    pub(crate) fn upload(
        &mut self,
        gpu: &Gpu,
        id: u32,
        data: SpriteData,
    ) -> Result<(), RenderError> {
        if data.rgba.len() != 256 * 4 || data.values.iter().any(|value| !value.is_finite()) {
            return Err(RenderError::Resource(
                "invalid sprite uniform or colormap".into(),
            ));
        }
        if let Some(material) = self.materials.get_mut(&id).filter(|material| {
            material.data.source == data.source && material.data.rgba == data.rgba
        }) {
            if material.data.values != data.values {
                gpu.queue()
                    .write_buffer(&material.params, 0, bytemuck::cast_slice(&data.values));
                material.data.values = data.values;
            }
            return Ok(());
        }
        let device = gpu.device();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("native sprite pipeline layout"),
            bind_group_layouts: &[Some(&self.layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native sprite WGSL"),
            source: wgpu::ShaderSource::Wgsl(data.source.as_str().into()),
        });
        let vertex = [wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }];
        let pipeline = make_pipeline(
            device,
            &shader,
            &layout,
            &vertex,
            self.format,
            "fs_main",
            (wgpu::FrontFace::Ccw, None),
        );
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Resource(format!(
                "sprite material WGSL: {error}"
            )));
        }
        let params = gpu_buffer(
            device,
            "native sprite material",
            16,
            wgpu::BufferUsages::UNIFORM,
        );
        gpu.queue()
            .write_buffer(&params, 0, bytemuck::cast_slice(&data.values));
        let view = upload_1d(gpu, &data.rgba);
        self.materials.insert(
            id,
            SpriteMaterial {
                pipeline,
                params,
                view,
                data,
            },
        );
        for frame in self.frames.iter_mut().flatten() {
            frame.groups.remove(&id);
        }
        Ok(())
    }

    pub(crate) fn prepare(
        &mut self,
        gpu: &Gpu,
        world: &World,
        source: InstanceSource,
        layers: Option<&BTreeSet<i32>>,
        globals: &[f32; GLOBAL_FLOATS],
        view: usize,
    ) -> Result<(), RenderError> {
        if let Some(frame) = &mut self.frames[view] {
            frame.draws.clear();
        }
        let Some(marker) = world.component_index("PointCloud") else {
            return Ok(());
        };
        let Some(material) = source.material else {
            return Ok(());
        };
        let snapshot = world.snapshot();
        let members = world
            .memberships(marker, &snapshot.entities)
            .map_err(world_error)?;
        let entities: Vec<_> = snapshot
            .entities
            .into_iter()
            .zip(members)
            .filter_map(|(entity, member)| member.then_some(entity))
            .collect();
        if entities.is_empty() {
            return Ok(());
        }
        let matrices = read(world, source.transform, &entities)?;
        let materials = read(world, material, &entities)?;
        let ColumnData::F32(matrix_values) = matrices.data else {
            unreachable!("validated transform schema")
        };
        let ColumnData::U32(material_ids) = materials.data else {
            unreachable!("validated material schema")
        };
        let scalars = scalar(world, "ScalarValue", &entities, 0.0)?;
        let radii = scalar(world, "Radius", &entities, 1.0)?;
        let layer_ids = layers
            .map(|_| scalar(world, "Layer", &entities, 0.0))
            .transpose()?;
        let mut groups: BTreeMap<u32, Rows> = BTreeMap::new();
        for i in 0..entities.len() {
            if !matrices.membership[i]
                || !materials.membership[i]
                || layers
                    .zip(layer_ids.as_ref())
                    .is_some_and(|(filter, ids)| !filter.contains(&(ids[i] as i32)))
            {
                continue;
            }
            let id = material_ids[i];
            if !self.materials.contains_key(&id) {
                return Err(RenderError::Resource(format!(
                    "point cloud needs a ColormapMaterial, found id {id}"
                )));
            }
            let group = groups.entry(id).or_default();
            group
                .models
                .push(model_from_transform_row(&matrix_values[i * 10..(i + 1) * 10]).to_columns());
            group.scalars.push(scalars[i]);
            group.radii.push(radii[i]);
        }
        let device = gpu.device();
        let frame = self.frames[view].get_or_insert_with(|| {
            let capacity = self.capacity as u64;
            SpriteFrame {
                globals: gpu_buffer(
                    device,
                    "native sprite globals",
                    (GLOBAL_FLOATS * 4) as u64,
                    wgpu::BufferUsages::UNIFORM,
                ),
                transforms: gpu_buffer(
                    device,
                    "native sprite models",
                    (capacity * 64).max(64),
                    wgpu::BufferUsages::STORAGE,
                ),
                scalars: gpu_buffer(
                    device,
                    "native sprite scalars",
                    (capacity * 4).max(4),
                    wgpu::BufferUsages::STORAGE,
                ),
                radii: gpu_buffer(
                    device,
                    "native sprite radii",
                    (capacity * 4).max(4),
                    wgpu::BufferUsages::STORAGE,
                ),
                groups: HashMap::new(),
                cached: Rows::default(),
                draws: Vec::new(),
            }
        });
        gpu.queue()
            .write_buffer(&frame.globals, 0, bytemuck::cast_slice(globals));
        let mut rows = Rows::default();
        for (id, mut group) in groups {
            let start = rows.models.len() as u32;
            rows.models.append(&mut group.models);
            rows.scalars.append(&mut group.scalars);
            rows.radii.append(&mut group.radii);
            frame.draws.push((id, start..rows.models.len() as u32));
            frame.groups.entry(id).or_insert_with(|| {
                let material = &self.materials[&id];
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("native sprite material"),
                    layout: &self.layout,
                    entries: &[
                        resource(0, frame.globals.as_entire_binding()),
                        resource(1, frame.transforms.as_entire_binding()),
                        resource(2, material.params.as_entire_binding()),
                        resource(3, frame.scalars.as_entire_binding()),
                        resource(4, frame.radii.as_entire_binding()),
                        resource(5, wgpu::BindingResource::TextureView(&material.view)),
                        resource(6, wgpu::BindingResource::Sampler(&self.sampler)),
                    ],
                })
            });
        }
        if frame.cached != rows {
            gpu.queue()
                .write_buffer(&frame.transforms, 0, bytemuck::cast_slice(&rows.models));
            gpu.queue()
                .write_buffer(&frame.scalars, 0, bytemuck::cast_slice(&rows.scalars));
            gpu.queue()
                .write_buffer(&frame.radii, 0, bytemuck::cast_slice(&rows.radii));
            frame.cached = rows;
        }
        Ok(())
    }

    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, view: usize) -> (usize, usize) {
        let Some(frame) = &self.frames[view] else {
            return (0, 0);
        };
        let mut triangles = 0;
        for (id, rows) in &frame.draws {
            pass.set_pipeline(&self.materials[id].pipeline);
            pass.set_bind_group(0, &frame.groups[id], &[]);
            pass.set_vertex_buffer(0, self.quad.slice(..));
            pass.draw(0..6, rows.clone());
            triangles += rows.len() * 2;
        }
        (frame.draws.len(), triangles)
    }
}

fn world_error(error: manifoldx_core::ecs::WorldError) -> RenderError {
    RenderError::Resource(error.to_string())
}

fn read(
    world: &World,
    column: usize,
    entities: &[EntityId],
) -> Result<ComponentReadout, RenderError> {
    world
        .read_component(column, Some(entities))
        .map_err(world_error)
}

fn scalar(
    world: &World,
    name: &str,
    entities: &[EntityId],
    default: f32,
) -> Result<Vec<f32>, RenderError> {
    let Some(column) = world.component_index(name) else {
        return Ok(vec![default; entities.len()]);
    };
    let result = read(world, column, entities)?;
    if let (1, ColumnData::F32(values)) = (result.schema.width, result.data) {
        return Ok(values
            .into_iter()
            .zip(result.membership)
            .map(|(value, member)| if member { value } else { default })
            .collect());
    }
    Err(RenderError::Resource(format!(
        "{name} needs one f32 column"
    )))
}

fn buffer_binding(
    binding: u32,
    ty: wgpu::BufferBindingType,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}

fn resource(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}
