//! One native wgpu device, one fixed opaque pipeline, indexed instanced draws
//! from World rows, a depth buffer and an owned offscreen readback.
//!
//! This is the base renderer slice, not the full engine renderer. Textures,
//! PBR, shadows, GUI, streaming, skinning and a window surface are later
//! slices. There is no shader compiler or material graph here: one WGSL
//! pipeline and one colour per material id.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use manifoldx_core::ecs::World;

use crate::camera::Camera;
use crate::scene::{
    read_instances, InstanceBatch, InstanceSource, MeshData, SceneError, DEFAULT_MAX_INSTANCES,
};

/// The offscreen colour format. It is sRGB, so the target encodes the
/// linear shader output once.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Reversed-depth float buffer; the pass clears 0 and compares `Greater`.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const CAMERA_BYTES: usize = 64;
const INSTANCE_FLOATS: usize = 20;
const INSTANCE_BYTES: usize = INSTANCE_FLOATS * 4;
const VERTEX_BYTES: usize = 6 * 4;
/// Largest offscreen side accepted by this slice.
pub const MAX_SIDE: u32 = 4096;

/// Renderer construction options.
#[derive(Clone, Copy, Debug)]
pub struct RendererOptions {
    /// Hard cap on instances one frame may read from the World.
    pub max_instances: usize,
    /// Accept a CPU adapter. Tests set this so a software-only CI host skips
    /// gracefully; the sample binary keeps it false.
    pub allow_software: bool,
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            max_instances: DEFAULT_MAX_INSTANCES,
            allow_software: false,
        }
    }
}

/// A copy of the adapter facts that the evidence records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterSummary {
    /// Adapter name.
    pub name: String,
    /// Backend name, such as `Vulkan`.
    pub backend: String,
    /// Device type, such as `DiscreteGpu`.
    pub device_type: String,
    /// Driver description.
    pub driver: String,
    /// Driver info string.
    pub driver_info: String,
    /// Vendor id.
    pub vendor: u32,
    /// Device id.
    pub device: u32,
}

impl AdapterSummary {
    /// One-line JSON written by the sample and the tests.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"name\":{},\"backend\":{},\"device_type\":{},\"driver\":{},\"driver_info\":{},\"vendor\":{},\"device\":{}}}",
            json_string(&self.name),
            json_string(&self.backend),
            json_string(&self.device_type),
            json_string(&self.driver),
            json_string(&self.driver_info),
            self.vendor,
            self.device
        )
    }
}

/// GPU ownership and the device facts.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: AdapterSummary,
}

impl Gpu {
    /// Create the Vulkan instance and a hardware device, or fail visibly.
    pub fn new(allow_software: bool) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|error| RenderError::Adapter(error.to_string()))?;
        let info = adapter.get_info();
        if !allow_software && matches!(info.device_type, wgpu::DeviceType::Cpu) {
            return Err(RenderError::NoHardwareAdapter(info.name));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("manifoldx-native-renderer-device"),
            ..Default::default()
        }))
        .map_err(|error| RenderError::Device(error.to_string()))?;
        Ok(Self {
            device,
            queue,
            adapter: AdapterSummary {
                name: info.name,
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
                driver: info.driver,
                driver_info: info.driver_info,
                vendor: info.vendor,
                device: info.device,
            },
        })
    }

    /// The adapter facts.
    pub fn adapter(&self) -> &AdapterSummary {
        &self.adapter
    }

    /// Borrow the device (tests and later slices).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Borrow the queue (tests and later slices).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    color: [f32; 4],
}

struct InstanceBuffer {
    buffer: wgpu::Buffer,
    capacity_bytes: u64,
}

/// Per-frame counters and wall times.
///
/// `wait_readback_ns` is CPU wall time around submission, GPU execution and
/// mapping. It is **not** a timestamp-query GPU time; this slice does not
/// enable timestamp queries.
#[derive(Clone, Debug, Default)]
pub struct RenderStats {
    /// Live entities the World read examined.
    pub entities_examined: usize,
    /// Instances after membership and geometry filtering.
    pub instances: usize,
    /// Opaque draw calls issued.
    pub draw_calls: usize,
    /// Triangles submitted.
    pub triangles: usize,
    /// Snapshot plus selected component reads plus instance assembly.
    pub read_ns: u64,
    /// Camera uniform and instance buffer copy.
    pub upload_ns: u64,
    /// Command encoding and queue submission.
    pub encode_submit_ns: u64,
    /// Poll, map, copy and unmap of the readback buffer.
    pub wait_readback_ns: u64,
    /// The whole call, from the first read to the last copied byte.
    pub total_ns: u64,
}

/// An owned RGBA image read back from the GPU.
#[derive(Clone, Debug)]
pub struct RenderOutput {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Tightly packed RGBA8 pixels, row-major, sRGB-encoded.
    pub rgba: Vec<u8>,
    /// Per-frame counters and times.
    pub stats: RenderStats,
}

/// A native offscreen renderer.
pub struct NativeRenderer {
    gpu: Gpu,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    meshes: HashMap<u32, GpuMesh>,
    materials: HashMap<u32, [f32; 4]>,
    instance_buffer: Option<InstanceBuffer>,
    max_instances: usize,
}

impl NativeRenderer {
    /// Create the device, the fixed pipeline and its camera uniform.
    pub fn new(options: RendererOptions) -> Result<Self, RenderError> {
        let gpu = Gpu::new(options.allow_software)?;
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("manifoldx-native-opaque-wgsl"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("manifoldx-native-camera-layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(CAMERA_BYTES as u64),
                        },
                        count: None,
                    }],
                });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("manifoldx-native-pipeline-layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("manifoldx-native-opaque-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[VERTEX_LAYOUT, INSTANCE_LAYOUT],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let camera_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("manifoldx-native-camera-uniform"),
            size: CAMERA_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("manifoldx-native-camera-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        Ok(Self {
            gpu,
            pipeline,
            camera_buffer,
            camera_bind_group,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            instance_buffer: None,
            max_instances: options.max_instances,
        })
    }

    /// Adapter facts for evidence.
    pub fn adapter(&self) -> &AdapterSummary {
        self.gpu.adapter()
    }

    /// Borrow the device, so a later slice can add resources without a second
    /// device API.
    pub fn device(&self) -> &wgpu::Device {
        self.gpu.device()
    }

    /// Upload one indexed geometry under the World's `Mesh` id.
    ///
    /// `color_srgb` is the geometry's default colour when an instance has no
    /// registered material. It is decoded to linear here; the sRGB target
    /// encodes the output once.
    pub fn upload_mesh(
        &mut self,
        geometry_id: u32,
        mesh: &MeshData,
        color_srgb: [f32; 3],
    ) -> Result<(), RenderError> {
        if geometry_id == 0 {
            return Err(RenderError::ReservedGeometryId);
        }
        let vertex_count = mesh.validate()?;
        let interleaved = mesh.interleaved();
        let vertex_bytes = bytemuck::cast_slice(&interleaved);
        let vertex_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("manifoldx-native-vertices"),
            size: vertex_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu.queue.write_buffer(&vertex_buffer, 0, vertex_bytes);
        let index_bytes = bytemuck::cast_slice(&mesh.indices);
        let index_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("manifoldx-native-indices"),
            size: index_bytes.len() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu.queue.write_buffer(&index_buffer, 0, index_bytes);
        debug_assert_eq!(interleaved.len(), vertex_count * 6);
        self.meshes.insert(
            geometry_id,
            GpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                color: srgb_to_linear(color_srgb),
            },
        );
        Ok(())
    }

    /// Register an opaque material colour for a `Material` row id.
    pub fn register_material(&mut self, material_id: u32, color_srgb: [f32; 3]) {
        self.materials
            .insert(material_id, srgb_to_linear(color_srgb));
    }

    /// Render the World's instances offscreen and return an owned RGBA image.
    ///
    /// `ignore_transform` is a negative-control switch only: it forces identity
    /// model matrices, so a transform change cannot alter the image.
    pub fn render(
        &mut self,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        width: u32,
        height: u32,
        ignore_transform: bool,
    ) -> Result<RenderOutput, RenderError> {
        self.render_inner(
            world,
            source,
            camera,
            width,
            height,
            ignore_transform,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_inner(
        &mut self,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        width: u32,
        height: u32,
        ignore_transform: bool,
        omit_draw: bool,
    ) -> Result<RenderOutput, RenderError> {
        let started = Instant::now();
        if !(1..=MAX_SIDE).contains(&width) || !(1..=MAX_SIDE).contains(&height) {
            return Err(RenderError::BadSize { width, height });
        }
        if source.max_instances > self.max_instances {
            return Err(RenderError::InstanceBound {
                requested: source.max_instances,
                configured: self.max_instances,
            });
        }

        let read_started = Instant::now();
        let batch = read_instances(world, source, ignore_transform)?;
        let mut stats = RenderStats {
            entities_examined: batch.entities_examined,
            ..Default::default()
        };
        let draw_groups = self.pack_instances(&batch, &mut stats)?;
        stats.read_ns = elapsed_ns(read_started.elapsed());

        let aspect = width as f32 / height as f32;
        let view_projection = camera.view_projection(aspect).to_columns();
        let upload_started = Instant::now();
        self.gpu.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&view_projection),
        );

        let instance_data = self.pack_instance_bytes(&draw_groups);
        if !instance_data.is_empty() {
            let needed = instance_data.len() as u64;
            let recreate = match &self.instance_buffer {
                Some(buffer) => buffer.capacity_bytes < needed,
                None => true,
            };
            if recreate {
                let capacity = needed.next_power_of_two();
                self.instance_buffer = Some(InstanceBuffer {
                    buffer: self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("manifoldx-native-instances"),
                        size: capacity,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    capacity_bytes: capacity,
                });
            }
            let buffer = &self.instance_buffer.as_ref().expect("created above").buffer;
            self.gpu.queue.write_buffer(buffer, 0, &instance_data);
        }
        stats.upload_ns = elapsed_ns(upload_started.elapsed());

        let color = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("manifoldx-native-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("manifoldx-native-depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let encode_started = Instant::now();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("manifoldx-native-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("manifoldx-native-opaque-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !omit_draw && !draw_groups.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                let instance_buffer = &self
                    .instance_buffer
                    .as_ref()
                    .expect("groups imply data")
                    .buffer;
                // Draw ranges index the full buffer in packed group order.
                pass.set_vertex_buffer(1, instance_buffer.slice(..));
                let mut first_instance = 0u32;
                for group in &draw_groups {
                    let mesh = self
                        .meshes
                        .get(&group.geometry_id)
                        .ok_or(RenderError::UnknownGeometry(group.geometry_id))?;
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(
                        0..mesh.index_count,
                        0,
                        first_instance..first_instance + group.instances,
                    );
                    first_instance += group.instances;
                    stats.draw_calls += 1;
                    stats.triangles += (mesh.index_count / 3) as usize * group.instances as usize;
                }
            }
        }

        let padded_bytes_per_row = align_to(width * 4, 256);
        let readback_size = padded_bytes_per_row as u64 * height as u64;
        let readback = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("manifoldx-native-readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.gpu.queue.submit([encoder.finish()]);
        stats.encode_submit_ns = elapsed_ns(encode_started.elapsed());

        let readback_started = Instant::now();
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| RenderError::Poll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|_| RenderError::ReadbackChannel)?
            .map_err(|error| RenderError::Readback(error.to_string()))?;

        let mapped = readback.get_mapped_range(..);
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for row in mapped
            .chunks_exact(padded_bytes_per_row as usize)
            .take(height as usize)
        {
            rgba.extend_from_slice(&row[..(width * 4) as usize]);
        }
        drop(mapped);
        readback.unmap();
        stats.wait_readback_ns = elapsed_ns(readback_started.elapsed());
        stats.instances = batch.instances.len();
        stats.total_ns = elapsed_ns(started.elapsed());

        Ok(RenderOutput {
            width,
            height,
            rgba,
            stats,
        })
    }

    /// Render without issuing any draw, for the negative control.
    pub fn render_without_draw(
        &mut self,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        width: u32,
        height: u32,
        ignore_transform: bool,
    ) -> Result<RenderOutput, RenderError> {
        self.render_inner(world, source, camera, width, height, ignore_transform, true)
    }

    fn pack_instances(
        &self,
        batch: &InstanceBatch,
        stats: &mut RenderStats,
    ) -> Result<Vec<DrawGroup>, RenderError> {
        let mut by_geometry: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        for instance in &batch.instances {
            let mesh = self
                .meshes
                .get(&instance.geometry_id)
                .ok_or(RenderError::UnknownGeometry(instance.geometry_id))?;
            let color = instance
                .material_id
                .and_then(|id| self.materials.get(&id).copied())
                .unwrap_or(mesh.color);
            let entry = by_geometry.entry(instance.geometry_id).or_default();
            entry.extend_from_slice(&instance.model.to_columns());
            entry.extend_from_slice(&color);
        }
        let mut groups = Vec::with_capacity(by_geometry.len());
        for (geometry_id, values) in by_geometry {
            debug_assert!(values.len().is_multiple_of(INSTANCE_FLOATS));
            let instances = (values.len() / INSTANCE_FLOATS) as u32;
            groups.push(DrawGroup {
                geometry_id,
                instances,
                values,
            });
        }
        stats.instances = batch.instances.len();
        Ok(groups)
    }

    fn pack_instance_bytes(&self, groups: &[DrawGroup]) -> Vec<u8> {
        let total: usize = groups.iter().map(|group| group.values.len()).sum();
        let mut floats = Vec::with_capacity(total);
        for group in groups {
            floats.extend_from_slice(&group.values);
        }
        bytemuck::cast_slice(&floats).to_vec()
    }
}

struct DrawGroup {
    geometry_id: u32,
    instances: u32,
    values: Vec<f32>,
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
];
const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: VERTEX_BYTES as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &VERTEX_ATTRIBUTES,
};
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 64,
        shader_location: 6,
    },
];
const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: INSTANCE_BYTES as u64,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &INSTANCE_ATTRIBUTES,
};

const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};
struct InstanceIn {
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) color: vec4<f32>,
};
struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > 1e-20 && len_sq < 1e30 {
        return v * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// The instance model is T * R * S. Its upper-left 3x3 columns are s_j * R e_j,
// so (M^-1)^T n = sum_j (n_j / |column_j|^2) * column_j. No explicit inverse is
// needed. A zero or non-finite axis is dropped; when every axis is degenerate
// the input normal is used so the result stays finite. math::transform_normal
// mirrors this function for tests and evidence.
fn transform_normal(model: mat4x4<f32>, normal: vec3<f32>) -> vec3<f32> {
    let c0 = model[0].xyz;
    let c1 = model[1].xyz;
    let c2 = model[2].xyz;
    let l0 = dot(c0, c0);
    let l1 = dot(c1, c1);
    let l2 = dot(c2, c2);
    var result = vec3<f32>(0.0);
    if l0 > 1e-20 && l0 < 1e30 {
        result += (normal.x / l0) * c0;
    }
    if l1 > 1e-20 && l1 < 1e30 {
        result += (normal.y / l1) * c1;
    }
    if l2 > 1e-20 && l2 < 1e30 {
        result += (normal.z / l2) * c2;
    }
    let len_sq = dot(result, result);
    if len_sq > 1e-20 && len_sq < 1e30 {
        return result * inverseSqrt(len_sq);
    }
    return safe_normalize(normal);
}

@vertex
fn vs_main(in: VertexIn, instance: InstanceIn) -> VertexOut {
    let model = mat4x4<f32>(instance.m0, instance.m1, instance.m2, instance.m3);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: VertexOut;
    out.clip_position = camera.view_proj * world;
    out.normal = transform_normal(model, in.normal);
    out.color = instance.color.rgb;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.35, 0.8, 0.55));
    let diffuse = max(dot(normalize(in.normal), light_dir), 0.0);
    // Flat opaque material: one albedo, one directional term, no PBR.
    let shaded = in.color * (0.28 + 0.72 * diffuse);
    return vec4<f32>(shaded, 1.0);
}
"#;

/// sRGB to linear for a colour a person chose, matching the engine's material rule.
pub fn srgb_to_linear(color: [f32; 3]) -> [f32; 4] {
    let decode = |value: f32| {
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    [decode(color[0]), decode(color[1]), decode(color[2]), 1.0]
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn elapsed_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

/// Write a binary PPM (P6), an image format that needs no extra dependency.
pub fn write_ppm(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), RenderError> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(RenderError::ImageSize)?;
    if rgba.len() != expected {
        return Err(RenderError::ImageSize);
    }
    let mut out = Vec::with_capacity(16 + (width as usize * height as usize * 3));
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for pixel in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&pixel[..3]);
    }
    std::fs::write(path, out).map_err(RenderError::Io)
}

/// Minimal JSON string escaping for the evidence objects.
pub fn json_string(value: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A renderer or scene failure.
#[derive(Debug)]
pub enum RenderError {
    /// Scene preparation failed.
    Scene(SceneError),
    /// No Vulkan adapter answered.
    Adapter(String),
    /// The adapter exists but no device could be created.
    Device(String),
    /// A CPU adapter was rejected because the binary requires hardware.
    NoHardwareAdapter(String),
    /// The offscreen size is zero or above [`MAX_SIDE`].
    BadSize {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// A source limit was above the renderer's configured bound.
    InstanceBound {
        /// Requested bound.
        requested: usize,
        /// Configured bound.
        configured: usize,
    },
    /// Geometry id 0 is reserved for "no geometry".
    ReservedGeometryId,
    /// A Mesh row named a geometry the renderer does not own.
    UnknownGeometry(u32),
    /// Device polling failed.
    Poll(String),
    /// The map callback never arrived.
    ReadbackChannel,
    /// The readback buffer could not be mapped.
    Readback(String),
    /// The image slice length did not match its dimensions.
    ImageSize,
    /// Writing an evidence file failed.
    Io(std::io::Error),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scene(error) => write!(f, "scene: {error}"),
            Self::Adapter(error) => write!(f, "no usable adapter: {error}"),
            Self::Device(error) => write!(f, "device creation failed: {error}"),
            Self::NoHardwareAdapter(name) => {
                write!(f, "hardware adapter required, got CPU adapter {name}")
            }
            Self::BadSize { width, height } => {
                write!(f, "size {width}x{height} is outside 1..={MAX_SIDE}")
            }
            Self::InstanceBound {
                requested,
                configured,
            } => write!(
                f,
                "source bound {requested} exceeds renderer bound {configured}"
            ),
            Self::ReservedGeometryId => write!(f, "geometry id 0 is reserved"),
            Self::UnknownGeometry(id) => write!(f, "mesh row references unknown geometry {id}"),
            Self::Poll(error) => write!(f, "device poll failed: {error}"),
            Self::ReadbackChannel => write!(f, "readback callback was dropped"),
            Self::Readback(error) => write!(f, "readback map failed: {error}"),
            Self::ImageSize => write!(f, "image bytes do not match its dimensions"),
            Self::Io(error) => write!(f, "io: {error}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<SceneError> for RenderError {
    fn from(error: SceneError) -> Self {
        Self::Scene(error)
    }
}
