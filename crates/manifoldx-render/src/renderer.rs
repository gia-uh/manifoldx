//! Native indexed instancing with shared drawing for desktop and offscreen targets.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use manifoldx_core::ecs::World;

use crate::camera::Camera;
use crate::math::Mat4;
use crate::scene::{
    read_instances, InstanceBatch, InstanceSource, MeshData, SceneError, DEFAULT_MAX_INSTANCES,
};

/// The offscreen colour format. It is sRGB, so the target encodes the
/// linear shader output once.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Reversed-depth float buffer; the pass clears 0 and compares `Greater`.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const GLOBAL_BYTES: usize = (16 + 4 + 4 * 8) * 4;
const INSTANCE_FLOATS: usize = 24;
const INSTANCE_BYTES: usize = INSTANCE_FLOATS * 4;
const VERTEX_BYTES: usize = 8 * 4;
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
        let instance = Self::instance();
        let adapter = Self::compatible_adapter(&instance, None)?;
        Self::from_adapter(&adapter, allow_software)
    }

    pub(crate) fn instance() -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        })
    }

    pub(crate) fn compatible_adapter(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<wgpu::Adapter, RenderError> {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        }))
        .map_err(|error| RenderError::Adapter(error.to_string()))
    }

    pub(crate) fn from_adapter(
        adapter: &wgpu::Adapter,
        allow_software: bool,
    ) -> Result<Self, RenderError> {
        let info = adapter.get_info();
        if !allow_software && matches!(info.device_type, wgpu::DeviceType::Cpu) {
            return Err(RenderError::NoHardwareAdapter(info.name));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("manifoldx-native-renderer-device"),
            required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
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

/// Supported opaque material families for the base renderer.
#[derive(Clone, Copy, Debug)]
pub enum Shading {
    Basic,
    Flat,
    Phong,
    Standard {
        roughness: f32,
        metallic: f32,
        ao: f32,
        albedo: Option<u32>,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: [f32; 3],
    /// The established point-light buffer stores the input RGB directly.
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Clone, Copy)]
struct Material {
    color: [f32; 4],
    style: [f32; 4],
    albedo: u32,
}

impl Material {
    fn diffuse(color: [f32; 4]) -> Self {
        Self {
            color,
            style: [0.0; 4],
            albedo: 0,
        }
    }
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    color: [f32; 4],
    bounds: [[f32; 3]; 2],
}

struct InstanceBuffer {
    buffer: wgpu::Buffer,
    capacity_bytes: u64,
}

/// Physical color and reverse-depth target for one frame.
pub struct FrameTarget<'a> {
    pub color: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub clear: wgpu::Color,
    pub timestamps: Option<wgpu::RenderPassTimestampWrites<'a>>,
    pub positions: Option<ResidentPositions<'a>>,
}

/// The position version published by the current frame barrier.
pub struct ResidentPositions<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub members: &'a [Option<manifoldx_core::ecs::EntityId>],
    pub row_width: u32,
    pub offset: u32,
}

struct PositionPass {
    pipeline: wgpu::ComputePipeline,
    metadata: wgpu::Buffer,
    args: wgpu::Buffer,
    capacity: u64,
}

impl PositionPass {
    fn new(device: &wgpu::Device, instance_limit: usize) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident position culling"),
            source: wgpu::ShaderSource::Wgsl(include_str!("position.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resident position culling"),
            layout: None,
            module: &module,
            entry_point: Some("place"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let capacity = instance_limit as u64 * 32;
        Self {
            pipeline,
            metadata: buffer(
                "resident position bounds",
                capacity,
                wgpu::BufferUsages::STORAGE,
            ),
            args: buffer("resident position args", 16, wgpu::BufferUsages::UNIFORM),
            capacity,
        }
    }

    fn encode(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        buffers: [&wgpu::Buffer; 3],
        args: [u32; 4],
        metadata: &[[u32; 8]],
    ) {
        debug_assert!(metadata.len() as u64 * 32 <= self.capacity);
        gpu.queue
            .write_buffer(&self.metadata, 0, bytemuck::cast_slice(metadata));
        gpu.queue
            .write_buffer(&self.args, 0, bytemuck::cast_slice(&args));
        let entries = [
            buffers[0],
            &self.metadata,
            buffers[1],
            buffers[2],
            &self.args,
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
        let binding = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident position culling"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(args[2].div_ceil(64), 1, 1);
    }
}

pub(crate) fn depth_target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let depth = device.create_texture(&wgpu::TextureDescriptor {
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
    depth.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Per-frame counters and wall times.
///
/// `wait_readback_ns` is CPU wall time around submission, GPU execution and
/// mapping. The desktop diagnostic measures GPU execution separately.
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

/// Native GPU resources and drawing for one target format.
pub struct NativeRenderer {
    gpu: Gpu,
    pipeline: wgpu::RenderPipeline,
    reversed_pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    textures: HashMap<u32, wgpu::BindGroup>,
    point_lights: [[f32; 8]; 4],
    clear: wgpu::Color,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    meshes: HashMap<u32, GpuMesh>,
    materials: HashMap<u32, Material>,
    instance_buffer: Option<InstanceBuffer>,
    max_instances: usize,
    position_pass: Option<PositionPass>,
}

impl NativeRenderer {
    /// Create the device, the fixed pipeline and its camera uniform.
    pub fn new(options: RendererOptions) -> Result<Self, RenderError> {
        let gpu = Gpu::new(options.allow_software)?;
        Ok(Self::with_gpu(gpu, COLOR_FORMAT, options.max_instances))
    }

    pub(crate) fn with_gpu(gpu: Gpu, format: wgpu::TextureFormat, max_instances: usize) -> Self {
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
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(GLOBAL_BYTES as u64),
                        },
                        count: None,
                    }],
                });
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("manifoldx-albedo-layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("manifoldx-native-pipeline-layout"),
                bind_group_layouts: &[Some(&bind_group_layout), Some(&texture_layout)],
                immediate_size: 0,
            });
        let pipeline = |front_face| {
            gpu.device
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
                        front_face,
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
                            format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let camera_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("manifoldx-native-camera-uniform"),
            size: GLOBAL_BYTES as u64,
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
        let regular = pipeline(wgpu::FrontFace::Ccw);
        let reversed = pipeline(wgpu::FrontFace::Cw);
        let mut renderer = Self {
            gpu,
            pipeline: regular,
            reversed_pipeline: reversed,
            texture_layout,
            textures: HashMap::new(),
            point_lights: [[0.0; 8]; 4],
            clear: wgpu::Color::BLACK,
            camera_buffer,
            camera_bind_group,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            instance_buffer: None,
            max_instances,
            position_pass: None,
        };
        let white = renderer.make_texture(1, 1, &[&[255, 255, 255, 255]]);
        renderer.textures.insert(0, white);
        renderer
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

    /// Borrow the queue paired with this renderer's device.
    pub fn queue(&self) -> &wgpu::Queue {
        self.gpu.queue()
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
        self.upload_mesh_with_uv(geometry_id, mesh, color_srgb, &[])
    }

    pub fn upload_mesh_with_uv(
        &mut self,
        geometry_id: u32,
        mesh: &MeshData,
        color_srgb: [f32; 3],
        uvs: &[[f32; 2]],
    ) -> Result<(), RenderError> {
        if geometry_id == 0 {
            return Err(RenderError::ReservedGeometryId);
        }
        let vertex_count = mesh.validate()?;
        if !uvs.is_empty() && uvs.len() != vertex_count {
            return Err(RenderError::Resource(
                "mesh UV count differs from vertex count".into(),
            ));
        }
        let mut interleaved = Vec::with_capacity(vertex_count * 8);
        let mut bounds = [mesh.positions[0]; 2];
        for (index, (position, normal)) in mesh.positions.iter().zip(&mesh.normals).enumerate() {
            for axis in 0..3 {
                bounds[0][axis] = bounds[0][axis].min(position[axis]);
                bounds[1][axis] = bounds[1][axis].max(position[axis]);
            }
            interleaved.extend_from_slice(position);
            interleaved.extend_from_slice(normal);
            interleaved.extend_from_slice(uvs.get(index).unwrap_or(&[0.0; 2]));
        }
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
        debug_assert_eq!(interleaved.len(), vertex_count * 8);
        self.meshes.insert(
            geometry_id,
            GpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                color: srgb_to_linear(color_srgb),
                bounds,
            },
        );
        Ok(())
    }

    /// Register an opaque material colour for a `Material` row id.
    pub fn register_material(&mut self, material_id: u32, color_srgb: [f32; 3]) {
        self.materials
            .insert(material_id, Material::diffuse(srgb_to_linear(color_srgb)));
    }

    pub fn set_material(
        &mut self,
        id: u32,
        color_srgb: [f32; 3],
        shading: Shading,
    ) -> Result<(), RenderError> {
        if id == 0 || color_srgb.iter().any(|value| !value.is_finite()) {
            return Err(RenderError::Resource("invalid material id or color".into()));
        }
        let (style, albedo) = match shading {
            Shading::Basic | Shading::Phong => ([0.0, 0.0, 0.0, 1.0], 0),
            Shading::Flat => ([0.0, 0.0, 0.0, 2.0], 0),
            Shading::Standard {
                roughness,
                metallic,
                ao,
                albedo,
            } => {
                if !(0.001..=1.0).contains(&roughness)
                    || !(0.0..=1.0).contains(&metallic)
                    || !(0.0..=1.0).contains(&ao)
                {
                    return Err(RenderError::Resource(
                        "invalid Standard material factors".into(),
                    ));
                }
                let mode = if albedo.is_some() { 4.0 } else { 3.0 };
                ([roughness, metallic, ao, mode], albedo.unwrap_or(0))
            }
        };
        if !self.textures.contains_key(&albedo) {
            return Err(RenderError::Resource(format!(
                "unknown albedo texture {albedo}"
            )));
        }
        self.materials.insert(
            id,
            Material {
                color: srgb_to_linear(color_srgb),
                style,
                albedo,
            },
        );
        Ok(())
    }

    pub fn set_background(&mut self, linear_rgb: [f32; 3]) {
        let [r, g, b] = linear_rgb.map(f64::from);
        self.clear = wgpu::Color { r, g, b, a: 1.0 };
    }

    pub fn background(&self) -> wgpu::Color {
        self.clear
    }

    pub fn set_point_lights(&mut self, lights: &[PointLight]) {
        self.point_lights = [[0.0; 8]; 4];
        // The base light buffer has four slots, as in the existing Standard material.
        for (row, light) in self.point_lights.iter_mut().zip(lights) {
            let [x, y, z] = light.position;
            let [r, g, b] = light.color;
            *row = [x, y, z, 0.0, r, g, b, light.intensity];
        }
    }

    /// Upload a complete sRGB RGBA8 chain, largest mip first.
    pub fn upload_albedo(
        &mut self,
        id: u32,
        width: u32,
        height: u32,
        levels: &[&[u8]],
    ) -> Result<(), RenderError> {
        let limit = self.device().limits().max_texture_dimension_2d;
        if id == 0 || !(1..=limit).contains(&width) || !(1..=limit).contains(&height) {
            return Err(RenderError::Resource("invalid texture id or size".into()));
        }
        let expected_levels = width.max(height).ilog2() + 1;
        if levels.len() != expected_levels as usize {
            return Err(RenderError::Resource(format!(
                "expected {expected_levels} mip levels"
            )));
        }
        for (level, pixels) in levels.iter().enumerate() {
            let w = (width >> level).max(1);
            let h = (height >> level).max(1);
            if pixels.len() != w as usize * h as usize * 4 {
                return Err(RenderError::Resource(format!(
                    "invalid byte count at mip {level}"
                )));
            }
        }
        let texture = self.make_texture(width, height, levels);
        self.textures.insert(id, texture);
        Ok(())
    }

    fn make_texture(&self, width: u32, height: u32, levels: &[&[u8]]) -> wgpu::BindGroup {
        let texture = self.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("manifoldx-albedo"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, pixels) in levels.iter().enumerate() {
            let w = (width >> level).max(1);
            let h = (height >> level).max(1);
            self.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&Default::default());
        let sampler = self.device().create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("manifoldx-albedo"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        })
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
        let depth_view = depth_target(self.device(), width, height);

        let encode_started = Instant::now();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("manifoldx-native-encoder"),
            });
        let target = FrameTarget {
            color: &color_view,
            depth: &depth_view,
            width,
            height,
            clear: self.clear,
            timestamps: None,
            positions: None,
        };
        let mut stats = self.encode_world(
            &mut encoder,
            target,
            world,
            source,
            camera,
            ignore_transform,
            omit_draw,
        )?;

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
        stats.encode_submit_ns = elapsed_ns(encode_started.elapsed())
            .saturating_sub(stats.read_ns)
            .saturating_sub(stats.upload_ns);

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
        stats.total_ns = elapsed_ns(started.elapsed());

        Ok(RenderOutput {
            width,
            height,
            rgba,
            stats,
        })
    }

    /// Encode the scene into a color and depth view owned by this device.
    pub fn encode_world_to_target(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
    ) -> Result<RenderStats, RenderError> {
        self.encode_world(encoder, target, world, source, camera, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_world(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        ignore_transform: bool,
        omit_draw: bool,
    ) -> Result<RenderStats, RenderError> {
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
        let aspect = target.width as f32 / target.height as f32;
        let view_projection = camera.view_projection(aspect);
        let draw_groups = self.pack_instances(
            &batch,
            &view_projection,
            target.positions.as_ref(),
            &mut stats,
        )?;
        stats.read_ns = elapsed_ns(read_started.elapsed());

        let upload_started = Instant::now();
        let mut globals = [0.0; GLOBAL_BYTES / 4];
        globals[..16].copy_from_slice(&view_projection.to_columns());
        globals[16..19].copy_from_slice(&camera.position);
        globals[20..].copy_from_slice(self.point_lights.as_flattened());
        self.gpu
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&globals));

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
                        usage: wgpu::BufferUsages::VERTEX
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }),
                    capacity_bytes: capacity,
                });
            }
            let buffer = &self.instance_buffer.as_ref().expect("created above").buffer;
            self.gpu.queue.write_buffer(buffer, 0, &instance_data);
        }
        if let Some(source) = &target.positions {
            let metadata: Vec<_> = draw_groups
                .iter()
                .flat_map(|group| group.positions.iter().copied())
                .collect();
            if metadata.iter().any(|row| row[0] != u32::MAX) {
                let pass = self
                    .position_pass
                    .get_or_insert_with(|| PositionPass::new(&self.gpu.device, self.max_instances));
                pass.encode(
                    &self.gpu,
                    encoder,
                    [
                        source.buffer,
                        &self.instance_buffer.as_ref().expect("instance data").buffer,
                        &self.camera_buffer,
                    ],
                    [source.row_width, source.offset, metadata.len() as u32, 0],
                    &metadata,
                );
            }
        }
        stats.upload_ns = elapsed_ns(upload_started.elapsed());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("manifoldx-native-opaque-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(target.clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: target.timestamps,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !omit_draw && !draw_groups.is_empty() {
                let instance_buffer = &self
                    .instance_buffer
                    .as_ref()
                    .expect("groups imply data")
                    .buffer;
                // Draw ranges index the full buffer in packed group order.
                pass.set_vertex_buffer(1, instance_buffer.slice(..));
                let mut first_instance = 0u32;
                for group in &draw_groups {
                    let pipeline = if group.reversed {
                        &self.reversed_pipeline
                    } else {
                        &self.pipeline
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    pass.set_bind_group(1, &self.textures[&group.albedo], &[]);
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

        Ok(stats)
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
        view_projection: &Mat4,
        positions: Option<&ResidentPositions<'_>>,
        stats: &mut RenderStats,
    ) -> Result<Vec<DrawGroup>, RenderError> {
        let mut groups = BTreeMap::<_, DrawGroup>::new();
        for instance in &batch.instances {
            let mesh = self
                .meshes
                .get(&instance.geometry_id)
                .ok_or(RenderError::UnknownGeometry(instance.geometry_id))?;
            let clip = view_projection.multiply(&instance.model);
            let slot = instance.entity.slot();
            let bound = positions.is_some_and(|source| {
                source.members.get(slot as usize) == Some(&Some(instance.entity))
            });
            if !bound && outside_frustum(&clip, mesh.bounds) {
                continue;
            }
            let material = instance
                .material_id
                .and_then(|id| self.materials.get(&id).copied())
                .unwrap_or(Material::diffuse(mesh.color));
            let geometry_id = instance.geometry_id;
            let reversed = flips_winding(&instance.model);
            let albedo = material.albedo;
            let group = groups
                .entry((geometry_id, reversed, albedo))
                .or_insert_with(|| DrawGroup {
                    geometry_id,
                    reversed,
                    albedo,
                    instances: 0,
                    values: Vec::new(),
                    positions: Vec::new(),
                });
            group.values.extend_from_slice(&instance.model.to_columns());
            group.values.extend_from_slice(&material.color);
            group.values.extend_from_slice(&material.style);
            if positions.is_some() {
                let [[lx, ly, lz], [hx, hy, hz]] = mesh.bounds;
                group.positions.push([
                    if bound { slot } else { u32::MAX },
                    lx.to_bits(),
                    ly.to_bits(),
                    lz.to_bits(),
                    hx.to_bits(),
                    hy.to_bits(),
                    hz.to_bits(),
                    0,
                ]);
            }
            group.instances += 1;
            stats.instances += 1;
        }
        Ok(groups.into_values().collect())
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

fn flips_winding(model: &Mat4) -> bool {
    let [a, b, c, _, d, e, f, _, g, h, i, _, ..] = model.0;
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g) < 0.0
}

fn outside_frustum(clip: &Mat4, [lo, hi]: [[f32; 3]; 2]) -> bool {
    let mut outside = [true; 6];
    for x in [lo[0], hi[0]] {
        for y in [lo[1], hi[1]] {
            for z in [lo[2], hi[2]] {
                let [x, y, z, w] = clip.transform_point([x, y, z]);
                for (plane, clipped) in
                    outside
                        .iter_mut()
                        .zip([x < -w, x > w, y < -w, y > w, z < 0.0, z > w])
                {
                    *plane &= clipped;
                }
            }
        }
    }
    outside.into_iter().any(|plane| plane)
}

struct DrawGroup {
    geometry_id: u32,
    reversed: bool,
    albedo: u32,
    instances: u32,
    values: Vec<f32>,
    positions: Vec<[u32; 8]>,
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
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
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 24,
        shader_location: 7,
    },
];
const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: VERTEX_BYTES as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &VERTEX_ATTRIBUTES,
};
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = [
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
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 80,
        shader_location: 8,
    },
];
const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: INSTANCE_BYTES as u64,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &INSTANCE_ATTRIBUTES,
};

const SHADER: &str = r#"
struct PointLight {
    position: vec4<f32>,
    rgb_intensity: vec4<f32>,
};
struct Camera {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
    lights: array<PointLight, 4>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var albedo_sampler: sampler;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(7) uv: vec2<f32>,
};
struct InstanceIn {
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(8) style: vec4<f32>,
};
struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) style: vec4<f32>,
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
    out.clip_position = select(vec4<f32>(0.0, 0.0, -1.0, 1.0), camera.view_proj * world, instance.m3.w != 0.0);
    out.normal = transform_normal(model, in.normal);
    out.color = instance.color.rgb;
    out.world_position = world.xyz;
    out.uv = in.uv;
    out.style = instance.style;
    return out;
}

fn geometry_schlick(ndot: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return ndot / (ndot * (1.0 - k) + k);
}

fn standard_light(
    n: vec3<f32>, v: vec3<f32>, world: vec3<f32>, albedo: vec3<f32>,
    roughness: f32, metallic: f32, light: PointLight,
) -> vec3<f32> {
    let offset = light.position.xyz - world;
    let l = safe_normalize(offset);
    let h = safe_normalize(v + l);
    let dist_sq = max(dot(offset, offset), 0.0001);
    let radiance = light.rgb_intensity.rgb * light.rgb_intensity.a / dist_sq;
    let ndotv = max(dot(n, v), 0.0);
    let ndotl = max(dot(n, l), 0.0);
    let ndoth = max(dot(n, h), 0.0);
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    let pi = 3.14159265359;
    let distribution = a2 / (pi * denom * denom);
    let geometry = geometry_schlick(ndotv, roughness) * geometry_schlick(ndotl, roughness);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - dot(h, v), 0.0, 1.0), 5.0);
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / pi;
    let specular = distribution * geometry * f / (4.0 * ndotv * ndotl + 0.0001);
    return (diffuse + specular) * radiance * ndotl;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let uv_dx = dpdx(in.uv);
    let uv_dy = dpdy(in.uv);
    let mode = u32(in.style.w);
    let n = safe_normalize(in.normal);
    if mode == 0u {
        let direction = normalize(vec3<f32>(0.35, 0.8, 0.55));
        let light = 0.28 + 0.72 * max(dot(n, direction), 0.0);
        return vec4<f32>(in.color * light, 1.0);
    }
    if mode == 1u {
        let direction = normalize(vec3<f32>(0.5773, 0.5773, 0.5773));
        let light = 0.3 + 0.7 * max(dot(n, direction), 0.0);
        return vec4<f32>(in.color * light, 1.0);
    }
    if mode == 2u {
        return vec4<f32>(in.color, 1.0);
    }
    var albedo = in.color;
    if mode == 4u {
        let texel = textureSampleGrad(albedo_texture, albedo_sampler, in.uv, uv_dx, uv_dy);
        if texel.a < 0.5 { discard; }
        albedo = texel.rgb;
    }
    let v = safe_normalize(camera.position.xyz - in.world_position);
    var color = 0.03 * albedo * in.style.z;
    for (var i = 0u; i < 4u; i++) {
        let light = camera.lights[i];
        if light.rgb_intensity.a > 0.0 {
            color += standard_light(n, v, in.world_position, albedo, in.style.x, in.style.y, light);
        }
    }
    return vec4<f32>(color / (color + vec3<f32>(1.0)), 1.0);
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
    /// The native desktop window or surface failed.
    Window(String),
    /// A resource declaration is invalid for this renderer.
    Resource(String),
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
            Self::Window(error) => write!(f, "desktop: {error}"),
            Self::Resource(error) => write!(f, "resource: {error}"),
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
