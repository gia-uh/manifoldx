//! Native indexed instancing with shared drawing for desktop and offscreen targets.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::Instant;

use manifoldx_core::ecs::{ColumnData, EntityId, World};

use crate::camera::Camera;
use crate::lines::{LineOptions, LinePass};
use crate::math::Mat4;
use crate::post::{PostOptions, PostPasses};
use crate::rect::{RectPass, RectRow};
use crate::scene::{
    read_instances, InstanceBatch, InstanceSource, MeshData, SceneError, DEFAULT_MAX_INSTANCES,
};

use crate::environment::EnvironmentData;
use crate::fragment::{FragmentData, FragmentPipeline};
use crate::manual::{gpu_buffer, ManualData, ManualFrame, ManualPipeline, MeshTexture};
use crate::shadow::{self, ShadowFrame, ShadowMap, ShadowOptions};
use crate::skin::{JointMatrices, Skin, MAX_JOINTS};
use crate::sprite::{SpriteData, SpritePass};
use crate::volume::{VolumeMaterialData, VolumePass};

/// The offscreen colour format. It is sRGB, so the target encodes the
/// linear shader output once.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Reversed-depth float buffer; the pass clears 0 and compares `Greater`.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const GLOBAL_BYTES: usize = (16 + 4 + 4 * 8 + 8 + 16 + 4 + 8 * 4 + 16 + 4 + 8 + 4) * 4;
const INSTANCE_FLOATS: usize = 40;
const PALETTE_SLOTS: usize = 8;
const SURFACE_BAKED: f32 = 12.0;
const INSTANCE_BYTES: usize = INSTANCE_FLOATS * 4;
const VERTEX_BYTES: usize = 13 * 4;
const SURFACE_FLOATS: usize = 52;
const FLAT_STYLE: [f32; 4] = [0.0, 0.0, 0.0, 2.0];
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
    Phong {
        shininess: f32,
    },
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

#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SpotLight {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub cos_inner: f32,
    pub cos_outer: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct StandardEffects {
    pub vertex_colors: bool,
    pub opacity: f32,
    pub mirror: Option<f32>,
    pub cut: Option<[f32; 4]>,
    pub sway: f32,
    pub sway_reach: f32,
    pub ripple: f32,
    pub faded: bool,
    pub tinted: bool,
    pub paletted: bool,
    pub baked: Option<u32>,
}

impl Default for StandardEffects {
    fn default() -> Self {
        Self {
            vertex_colors: false,
            opacity: 1.0,
            mirror: None,
            cut: None,
            sway: 0.0,
            sway_reach: 4.0,
            ripple: 0.0,
            faded: false,
            tinted: false,
            paletted: false,
            baked: None,
        }
    }
}

impl StandardEffects {
    pub fn validate(self) -> Result<(), RenderError> {
        if !self.sway.is_finite()
            || !self.sway_reach.is_finite()
            || !self.ripple.is_finite()
            || (self.sway > 0.0 && self.sway_reach <= 0.0)
            || !self.opacity.is_finite()
            || self.opacity < 0.0
            || self.mirror.is_some_and(|height| !height.is_finite())
            || self
                .cut
                .is_some_and(|cut| cut.iter().any(|x| !x.is_finite()))
        {
            return Err(RenderError::Resource(
                "invalid Standard material effects".into(),
            ));
        }
        Ok(())
    }
}

fn reflection(height: f32) -> Mat4 {
    let mut transform = Mat4::IDENTITY;
    transform.0[5] = -1.0;
    transform.0[7] = 2.0 * height;
    transform
}

#[derive(Clone, Copy)]
struct Material {
    color: [f32; 4],
    style: [f32; 4],
    albedo: u32,
    effects: StandardEffects,
}

impl Material {
    fn diffuse(color: [f32; 4]) -> Self {
        Self {
            color,
            style: [0.0; 4],
            albedo: 0,
            effects: StandardEffects::default(),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct DrawOptions<'a> {
    ids: Option<&'a HashMap<u32, u32>>,
    ignore_transform: bool,
    omit_draw: bool,
    secondary: Option<&'a ViewOptions>,
}

struct PackOptions<'a> {
    positions: Option<&'a ResidentPositions<'a>>,
    ids: Option<&'a HashMap<u32, u32>>,
    styles: &'a HashMap<EntityId, InstanceStyle>,
    shadow: bool,
}

#[derive(Clone, Copy)]
struct InstanceStyle {
    fade: f32,
    tint: [f32; 3],
    palette: [[f32; 4]; PALETTE_SLOTS],
    baked_layer: f32,
}

impl Default for InstanceStyle {
    fn default() -> Self {
        Self {
            fade: 1.0,
            tint: [1.0; 3],
            palette: [[1.0; 4]; PALETTE_SLOTS],
            baked_layer: -1.0,
        }
    }
}

struct SurfaceBinding {
    albedo: u32,
    baked: Option<u32>,
    values: [f32; SURFACE_FLOATS],
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
}

struct BakedArray {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl BakedArray {
    fn new(device: &wgpu::Device, size: u32, layers: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native baked color array"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: size.ilog2() + 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self { texture, view }
    }
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    color: [f32; 4],
    bounds: [[f32; 3]; 2],
    skinned: bool,
    axis: bool,
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
    fn new(device: &wgpu::Device, instance_limit: usize) -> Result<Self, RenderError> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
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
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Device(format!(
                "resident culling pipeline: {error}"
            )));
        }
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let capacity = instance_limit as u64 * 32;
        Ok(Self {
            pipeline,
            metadata: buffer(
                "resident position bounds",
                capacity,
                wgpu::BufferUsages::STORAGE,
            ),
            args: buffer("resident position args", 16, wgpu::BufferUsages::UNIFORM),
            capacity,
        })
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

struct OffscreenTarget {
    size: [u32; 2],
    color: wgpu::Texture,
    view: wgpu::TextureView,
    depth: wgpu::TextureView,
}

impl OffscreenTarget {
    fn new(device: &wgpu::Device, [width, height]: [u32; 2], format: wgpu::TextureFormat) -> Self {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("manifoldx-native-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            size: [width, height],
            color,
            view,
            depth: depth_target(device, width, height),
        }
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
    pub line_draws: usize,
    pub frame_order: Vec<&'static str>,
    /// Number of actual shadow depth layers encoded before scene colour.
    pub shadow_layers: usize,
    /// Actual encoded screen-space effects after the main pass.
    pub post_order: Vec<&'static str>,
    pub post_target_generation: u64,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewOrder {
    BeforeMain,
    Omit,
    AfterMain,
}

pub struct ViewOptions {
    pub texture_id: u32,
    pub size: [u32; 2],
    pub layers: BTreeSet<i32>,
    pub camera: Camera,
    pub clear: [f32; 3],
    pub order: ViewOrder,
}

struct SecondaryView {
    options: ViewOptions,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    buffers: ViewBuffers,
}

struct ViewBuffers {
    camera: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    palettes: Option<InstanceBuffer>,
    instances: Option<InstanceBuffer>,
}

pub struct NativeRenderer {
    gpu: Gpu,
    skin: Skin,
    camera_layout: wgpu::BindGroupLayout,
    shadow: ShadowMap,
    environment: Option<(f32, bool)>,
    fog: Option<([f32; 2], [f32; 3])>,
    skybox_pipeline: wgpu::RenderPipeline,
    shadow_options: Option<ShadowOptions>,
    shadow_pipelines: [wgpu::RenderPipeline; 2],
    shadow_layers: Vec<ViewBuffers>,
    layer_filter: Option<std::sync::Arc<BTreeSet<i32>>>,
    secondary: Option<SecondaryView>,
    spot_light: Option<SpotLight>,
    animation: [f32; 4],
    benders: [[f32; 4]; 8],
    origin_tile: [f32; 2],
    format: wgpu::TextureFormat,
    offscreen: Option<OffscreenTarget>,
    post_options: Option<std::sync::Arc<PostOptions>>,
    post_passes: Option<PostPasses>,
    rects: std::sync::Arc<[RectRow]>,
    rect_pass: Option<RectPass>,
    lines: Option<std::sync::Arc<LineOptions>>,
    line_pass: Option<LinePass>,
    rect_before_post: bool,
    post_sun: Option<DirectionalLight>,
    pipeline: wgpu::RenderPipeline,
    reversed_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    textures: HashMap<u32, MeshTexture>,
    baked_arrays: HashMap<u32, BakedArray>,
    baked_sampler: wgpu::Sampler,
    empty_baked: BakedArray,
    surface_default: wgpu::Buffer,
    surface_materials: HashMap<u32, SurfaceBinding>,
    point_lights: [[f32; 8]; 4],
    sun: [f32; 8],
    spot: [f32; 16],
    clear: wgpu::Color,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    empty_palette: wgpu::Buffer,
    palette_buffer: Option<InstanceBuffer>,
    meshes: HashMap<u32, GpuMesh>,
    materials: HashMap<u32, Material>,
    fragments: HashMap<u32, FragmentPipeline>,
    manual: HashMap<u32, ManualPipeline>,
    manual_frames: [Option<ManualFrame>; 2],
    sprites: Option<SpritePass>,
    labels: Option<SpritePass>,
    volumes: Option<VolumePass>,
    invalid_materials: HashMap<u32, String>,
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
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(GLOBAL_BYTES as u64),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(16),
                            },
                            count: None,
                        },
                    ],
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 64,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    (SURFACE_FLOATS * 4) as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });
        let surface_default = gpu_buffer(
            &gpu.device,
            "native default surface settings",
            (SURFACE_FLOATS * 4) as u64,
            wgpu::BufferUsages::UNIFORM,
        );
        let skin = Skin::new(&gpu);
        let shadow = ShadowMap::new(&gpu.device);
        let shadow_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("native shadow pipeline layout"),
                bind_group_layouts: &[
                    Some(&bind_group_layout),
                    Some(&texture_layout),
                    Some(&skin.layout),
                ],
                immediate_size: 0,
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("manifoldx-native-pipeline-layout"),
                bind_group_layouts: &[
                    Some(&bind_group_layout),
                    Some(&texture_layout),
                    Some(&skin.layout),
                    Some(&shadow.layout),
                ],
                immediate_size: 0,
            });
        let pipeline = |front_face, translucent: bool, depth_only: bool| {
            let color_target = [Some(wgpu::ColorTargetState {
                format,
                blend: Some(if translucent {
                    wgpu::BlendState::ALPHA_BLENDING
                } else {
                    wgpu::BlendState::REPLACE
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })];
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("manifoldx-native-opaque-pipeline"),
                    layout: Some(if depth_only {
                        &shadow_layout
                    } else {
                        &pipeline_layout
                    }),
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
                        cull_mode: (!translucent).then_some(wgpu::Face::Back),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: if depth_only {
                            shadow::FORMAT
                        } else {
                            DEPTH_FORMAT
                        },
                        depth_write_enabled: Some(!translucent),
                        depth_compare: Some(if depth_only {
                            wgpu::CompareFunction::Less
                        } else {
                            wgpu::CompareFunction::Greater
                        }),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(if depth_only { "fs_shadow" } else { "fs_main" }),
                        compilation_options: Default::default(),
                        targets: if depth_only { &[] } else { &color_target },
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
        let empty_palette = gpu_buffer(
            &gpu.device,
            "native empty palette",
            16,
            wgpu::BufferUsages::STORAGE,
        );
        let camera_bind_group = camera_group(
            &gpu.device,
            &bind_group_layout,
            &camera_buffer,
            &empty_palette,
        );
        let skybox_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("native environment skybox"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_sky"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_sky"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let regular = pipeline(wgpu::FrontFace::Ccw, false, false);
        let reversed = pipeline(wgpu::FrontFace::Cw, false, false);
        let translucent = pipeline(wgpu::FrontFace::Ccw, true, false);
        let shadow_pipelines = [
            pipeline(wgpu::FrontFace::Ccw, false, true),
            pipeline(wgpu::FrontFace::Cw, false, true),
        ];
        let baked_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let empty_baked = BakedArray::new(&gpu.device, 1, 1);
        let mut renderer = Self {
            gpu,
            skin,
            camera_layout: bind_group_layout,
            shadow,
            environment: None,
            fog: None,
            skybox_pipeline,
            shadow_options: None,
            shadow_pipelines,
            shadow_layers: Vec::new(),
            layer_filter: None,
            secondary: None,
            spot_light: None,
            animation: [0.0; 4],
            benders: [[0.0; 4]; 8],
            origin_tile: [0.0; 2],
            format,
            offscreen: None,
            post_options: None,
            post_passes: None,
            rects: std::sync::Arc::default(),
            rect_pass: None,
            lines: None,
            line_pass: None,
            rect_before_post: false,
            post_sun: None,
            pipeline: regular,
            reversed_pipeline: reversed,
            translucent_pipeline: translucent,
            texture_layout,
            textures: HashMap::new(),
            baked_arrays: HashMap::new(),
            baked_sampler,
            empty_baked,
            surface_default,
            surface_materials: HashMap::new(),
            point_lights: [[0.0; 8]; 4],
            sun: [0.0; 8],
            spot: [0.0; 16],
            clear: wgpu::Color::BLACK,
            camera_buffer,
            camera_bind_group,
            empty_palette,
            palette_buffer: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            fragments: HashMap::new(),
            manual: HashMap::new(),
            manual_frames: [None, None],
            sprites: None,
            labels: None,
            volumes: None,
            invalid_materials: HashMap::new(),
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

    pub fn set_environment(&mut self, environment: Option<(f32, bool)>) {
        self.environment = environment;
    }

    pub fn set_fog(&mut self, fog: Option<([f32; 2], [f32; 3])>) {
        self.fog = fog;
    }

    pub fn upload_environment(&mut self, data: &EnvironmentData) -> Result<(), RenderError> {
        self.shadow
            .upload_environment(&self.gpu.device, &self.gpu.queue, data)
    }

    pub fn upload_fragment(&mut self, id: u32, data: &FragmentData) -> Result<(), RenderError> {
        let pipeline = FragmentPipeline::new(
            &self.gpu,
            self.format,
            &[VERTEX_LAYOUT, INSTANCE_LAYOUT],
            SHADER,
            [&self.camera_layout, &self.skin.layout],
            data,
        );
        match pipeline {
            Ok(fragment) => {
                self.invalid_materials.remove(&id);
                self.fragments.insert(id, fragment);
            }
            Err(error) => {
                self.invalid_materials.insert(id, error.to_string());
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn upload_sprite(&mut self, id: u32, data: SpriteData) -> Result<(), RenderError> {
        if self.sprites.is_none() {
            self.sprites = Some(SpritePass::new(
                &self.gpu,
                self.format,
                self.max_instances,
                false,
            )?);
        }
        self.sprites
            .as_mut()
            .expect("created above")
            .upload(&self.gpu, id, data)
    }

    pub fn upload_label(&mut self, id: u32, data: SpriteData) -> Result<(), RenderError> {
        self.prepare_labels()?;
        self.labels
            .as_mut()
            .expect("prepared above")
            .upload(&self.gpu, id, data)
    }

    pub fn upload_label_tile(&mut self, layer: u32, rgba: &[u8]) -> Result<(), RenderError> {
        self.prepare_labels()?;
        self.labels
            .as_mut()
            .expect("prepared above")
            .upload_tile(&self.gpu, layer, rgba)
    }

    fn prepare_labels(&mut self) -> Result<(), RenderError> {
        if self.labels.is_none() {
            self.labels = Some(SpritePass::new(
                &self.gpu,
                self.format,
                self.max_instances,
                true,
            )?);
        }
        Ok(())
    }

    pub fn upload_volume(
        &mut self,
        id: u32,
        dims: [u32; 3],
        bytes: &[u8],
    ) -> Result<(), RenderError> {
        self.prepare_volume()?;
        self.volumes
            .as_mut()
            .expect("initialized above")
            .voxels(&self.gpu, id, dims, bytes)
    }

    pub fn upload_volume_material(
        &mut self,
        id: u32,
        data: VolumeMaterialData,
    ) -> Result<(), RenderError> {
        self.prepare_volume()?;
        self.volumes
            .as_mut()
            .expect("initialized above")
            .material(&self.gpu, id, data)
    }

    fn prepare_volume(&mut self) -> Result<(), RenderError> {
        if self.volumes.is_none() {
            self.volumes = Some(VolumePass::new(&self.gpu, self.format)?);
        }
        Ok(())
    }

    pub fn upload_manual(&mut self, id: u32, data: &ManualData) -> Result<(), RenderError> {
        let result = self.prepare_manual(id, data);
        if let Err(error) = &result {
            self.invalid_materials.insert(id, error.to_string());
        } else {
            self.invalid_materials.remove(&id);
        }
        result
    }

    fn prepare_manual(&mut self, id: u32, data: &ManualData) -> Result<(), RenderError> {
        let existing = self
            .manual
            .get_mut(&id)
            .map(|material| material.update(&self.gpu, data));
        if existing.transpose()?.unwrap_or_default() {
            return Ok(());
        }
        let attributes = [
            VERTEX_ATTRIBUTES[0],
            VERTEX_ATTRIBUTES[1],
            wgpu::VertexAttribute {
                shader_location: 2,
                ..VERTEX_ATTRIBUTES[2]
            },
        ];
        let vertex = wgpu::VertexBufferLayout {
            array_stride: if data.axis { 12 } else { VERTEX_BYTES as u64 },
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: if data.axis {
                &attributes[..1]
            } else {
                &attributes
            },
        };
        let pipeline = ManualPipeline::new(&self.gpu, data, vertex, self.format)?;
        for frame in self.manual_frames.iter_mut().flatten() {
            frame.groups.remove(&id);
        }
        self.manual.insert(id, pipeline);
        Ok(())
    }

    pub fn prepare_resident_positions(&mut self) -> Result<(), RenderError> {
        if self.position_pass.is_none() {
            self.position_pass = Some(PositionPass::new(&self.gpu.device, self.max_instances)?);
        }
        Ok(())
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
        self.upload_mesh_attributes(geometry_id, mesh, color_srgb, uvs, &[], &[])
    }

    pub fn upload_mesh_attributes(
        &mut self,
        geometry_id: u32,
        mesh: &MeshData,
        color_srgb: [f32; 3],
        uvs: &[[f32; 2]],
        colors: &[[f32; 3]],
        joints: &[u32],
    ) -> Result<(), RenderError> {
        self.upload_mesh_surfaces(geometry_id, mesh, color_srgb, uvs, (colors, &[]), joints)
    }

    pub fn upload_mesh_surfaces(
        &mut self,
        geometry_id: u32,
        mesh: &MeshData,
        color_srgb: [f32; 3],
        uvs: &[[f32; 2]],
        (colors, surfaces): (&[[f32; 3]], &[f32]),
        joints: &[u32],
    ) -> Result<(), RenderError> {
        if geometry_id == 0 {
            return Err(RenderError::ReservedGeometryId);
        }
        let vertex_count = mesh.validate()?;
        if !joints.is_empty()
            && (joints.len() != vertex_count
                || joints.iter().any(|joint| {
                    (joint & 255) as usize >= MAX_JOINTS
                        || ((joint >> 8) & 255) as usize >= MAX_JOINTS
                }))
        {
            return Err(RenderError::Resource("invalid packed mesh joints".into()));
        }
        if !colors.is_empty()
            && (colors.len() != vertex_count
                || colors.as_flattened().iter().any(|x| !x.is_finite()))
        {
            return Err(RenderError::Resource("invalid vertex colors".into()));
        }
        if !uvs.is_empty() && uvs.len() != vertex_count {
            return Err(RenderError::Resource(
                "mesh UV count differs from vertex count".into(),
            ));
        }
        if !surfaces.is_empty()
            && (surfaces.len() != vertex_count || surfaces.iter().any(|id| !id.is_finite()))
        {
            return Err(RenderError::Resource("invalid vertex surface ids".into()));
        }
        let mut interleaved = Vec::with_capacity(vertex_count * (VERTEX_BYTES / 4));
        let mut bounds = [mesh.positions[0]; 2];
        for (index, (position, normal)) in mesh.positions.iter().zip(&mesh.normals).enumerate() {
            for axis in 0..3 {
                bounds[0][axis] = bounds[0][axis].min(position[axis]);
                bounds[1][axis] = bounds[1][axis].max(position[axis]);
            }
            interleaved.extend_from_slice(position);
            interleaved.extend_from_slice(normal);
            interleaved.extend_from_slice(uvs.get(index).unwrap_or(&[0.0; 2]));
            let surface = surfaces.get(index).copied().unwrap_or_default();
            let color = colors.get(index).copied().unwrap_or([1.0; 3]);
            if surface == SURFACE_BAKED {
                // This surface carries baked UVs and a packed sRGB fallback in its color columns.
                interleaved.extend_from_slice(&color);
            } else {
                interleaved.extend_from_slice(&srgb_to_linear(color)[..3]);
            }
            interleaved.push(surface);
            interleaved.push(f32::from_bits(
                joints.get(index).copied().unwrap_or_default(),
            ));
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
        debug_assert_eq!(interleaved.len() * 4, vertex_count * VERTEX_BYTES);
        self.meshes.insert(
            geometry_id,
            GpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                color: srgb_to_linear(color_srgb),
                bounds,
                skinned: !joints.is_empty(),
                axis: false,
            },
        );
        Ok(())
    }

    pub fn upload_axis(&mut self, id: u32, points: [[f32; 3]; 2]) -> Result<(), RenderError> {
        if id == 0 || points.as_flattened().iter().any(|value| !value.is_finite()) {
            return Err(RenderError::Resource("invalid native axis geometry".into()));
        }
        let device = self.device();
        let vertex_buffer = gpu_buffer(
            device,
            "native axis endpoints",
            24,
            wgpu::BufferUsages::VERTEX,
        );
        let index_buffer = gpu_buffer(device, "native axis indices", 8, wgpu::BufferUsages::INDEX);
        self.queue()
            .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&points));
        self.queue()
            .write_buffer(&index_buffer, 0, bytemuck::cast_slice(&[0u32, 1]));
        self.meshes.insert(
            id,
            GpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: 2,
                bounds: points,
                color: [1.0; 4],
                skinned: false,
                axis: true,
            },
        );
        Ok(())
    }

    pub fn mesh_is_skinned(&self, id: u32) -> bool {
        self.meshes.get(&id).is_some_and(|mesh| mesh.skinned)
    }

    pub fn set_joints(
        &mut self,
        entity: EntityId,
        matrices: &[[f32; 16]],
    ) -> Result<(), RenderError> {
        self.skin.set(entity, matrices)
    }

    pub fn bake_poses(&mut self, clip: u32, poses: &[JointMatrices]) -> Result<(), RenderError> {
        self.skin.bake(clip, poses)
    }

    pub fn play_poses(
        &mut self,
        entity: EntityId,
        clip: u32,
        position: f64,
    ) -> Result<(), RenderError> {
        self.skin.play(entity, clip, position)
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
        self.set_material_effects(id, color_srgb, shading, StandardEffects::default())
    }

    pub fn set_material_effects(
        &mut self,
        id: u32,
        color_srgb: [f32; 3],
        shading: Shading,
        effects: StandardEffects,
    ) -> Result<(), RenderError> {
        effects.validate()?;
        if id == 0 || color_srgb.iter().any(|value| !value.is_finite()) {
            return Err(RenderError::Resource("invalid material id or color".into()));
        }
        if let Some(baked) = effects
            .baked
            .filter(|id| !self.baked_arrays.contains_key(id))
        {
            return Err(RenderError::Resource(format!(
                "unknown baked texture array {baked}"
            )));
        }
        let (style, albedo) = match shading {
            Shading::Basic => ([0.0, 0.0, 0.0, 1.0], 0),
            Shading::Phong { shininess } => {
                if !shininess.is_finite() || shininess <= 0.0 {
                    return Err(RenderError::Resource(
                        "Phong shininess must be finite and positive".into(),
                    ));
                }
                ([shininess, 0.0, 0.0, 5.0], 0)
            }
            Shading::Flat => (FLAT_STYLE, 0),
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
                effects,
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

    pub fn set_direct_lights(&mut self, sun: Option<DirectionalLight>, spot: Option<SpotLight>) {
        self.post_sun = sun;
        self.spot_light = spot;
        self.sun = sun.map_or([0.0; 8], |light| {
            let [x, y, z] = light.direction;
            let [r, g, b] = light.color;
            [x, y, z, 0.0, r, g, b, light.intensity]
        });
        self.spot = spot.map_or([0.0; 16], |light| {
            let [x, y, z] = light.position;
            let [dx, dy, dz] = light.direction;
            let [r, g, b] = light.color;
            [
                x,
                y,
                z,
                light.range,
                dx,
                dy,
                dz,
                light.cos_inner,
                r,
                g,
                b,
                light.intensity,
                light.cos_outer,
                0.0,
                0.0,
                0.0,
            ]
        });
    }

    pub fn set_animation(&mut self, time: f64, origin: [f64; 2], benders: &[[f64; 4]]) {
        self.set_frame_time(time);
        self.animation[1] = origin[0].rem_euclid(256.0) as f32;
        self.animation[2] = origin[1].rem_euclid(256.0) as f32;
        self.origin_tile = origin.map(|value| (value / 256.0).floor() as f32);
        self.benders = [[0.0; 4]; 8];
        for (target, &[x, z, radius, strength]) in self.benders.iter_mut().zip(benders) {
            *target = [
                (x - origin[0]) as f32,
                (z - origin[1]) as f32,
                radius as f32,
                strength as f32,
            ];
        }
    }

    pub fn set_frame_time(&mut self, time: f64) {
        self.animation[0] = time.rem_euclid(64.0) as f32;
    }

    pub fn set_layers(&mut self, layers: Option<std::sync::Arc<BTreeSet<i32>>>) {
        self.layer_filter = layers;
    }

    /// Declares the single test view used to verify same-frame dependencies.
    pub fn set_fixture_view(&mut self, options: ViewOptions) -> Result<(), RenderError> {
        let [width, height] = options.size;
        if options.texture_id == 0
            || self.textures.contains_key(&options.texture_id)
            || !(1..=MAX_SIDE).contains(&width)
            || !(1..=MAX_SIDE).contains(&height)
            || self.secondary.is_some()
        {
            return Err(RenderError::Resource(
                "invalid secondary fixture view".into(),
            ));
        }
        let device = self.device();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native sampled view"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = depth_target(device, width, height);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bindings = self.bind_albedo(color.clone(), sampler);
        let buffers = view_buffers(device, &self.camera_layout, &self.empty_palette);
        self.textures.insert(options.texture_id, bindings);
        self.secondary = Some(SecondaryView {
            options,
            color,
            depth,
            buffers,
        });
        Ok(())
    }

    pub fn set_shadows(&mut self, options: Option<ShadowOptions>) {
        self.shadow_options = options;
    }

    pub fn set_lines(&mut self, lines: Option<std::sync::Arc<LineOptions>>) {
        self.lines = lines;
    }

    /// Submit already laid-out, physical-pixel rectangles. The order override is for fixtures.
    pub fn set_rects(&mut self, rows: std::sync::Arc<[RectRow]>, before_post: bool) {
        self.rects = rows;
        self.rect_before_post = before_post;
    }

    pub fn set_post(&mut self, options: Option<std::sync::Arc<PostOptions>>) {
        self.post_options = options;
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

    /// Reserve an sRGB array texture with its complete mip chain.
    pub fn create_baked_array(
        &mut self,
        id: u32,
        size: u32,
        layers: u32,
    ) -> Result<(), RenderError> {
        let limits = self.device().limits();
        if id == 0
            || self.baked_arrays.contains_key(&id)
            || !(1..=limits.max_texture_dimension_2d).contains(&size)
            || !(1..=limits.max_texture_array_layers).contains(&layers)
        {
            return Err(RenderError::Resource(
                "invalid baked array id, size or layer count".into(),
            ));
        }
        let array = BakedArray::new(self.device(), size, layers);
        self.baked_arrays.insert(id, array);
        Ok(())
    }

    /// Update one layer. Validate every supplied level before writing any pixels.
    pub fn write_baked_layer(
        &self,
        id: u32,
        layer: u32,
        levels: &[Vec<u8>],
    ) -> Result<(), RenderError> {
        let array = self
            .baked_arrays
            .get(&id)
            .ok_or_else(|| RenderError::Resource(format!("unknown baked texture array {id}")))?;
        let texture = &array.texture;
        if layer >= texture.depth_or_array_layers()
            || levels.is_empty()
            || levels.len() > texture.mip_level_count() as usize
        {
            return Err(RenderError::Resource(
                "invalid baked texture layer or mip count".into(),
            ));
        }
        let side = |level: usize| (texture.width() >> level).max(1);
        for (level, pixels) in levels.iter().enumerate() {
            let size = side(level) as usize;
            if pixels.len() != size * size * 4 {
                return Err(RenderError::Resource(format!(
                    "invalid baked array mip {level} byte count"
                )));
            }
        }
        for (level, pixels) in levels.iter().enumerate() {
            let size = side(level);
            self.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(())
    }

    fn make_texture(&self, width: u32, height: u32, levels: &[&[u8]]) -> MeshTexture {
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
        self.bind_albedo(view, sampler)
    }

    pub fn set_surface_patterns(
        &mut self,
        id: u32,
        patterns: [[f32; 5]; 9],
        detail: bool,
    ) -> Result<(), RenderError> {
        let material = self.materials.get(&id).ok_or_else(|| {
            RenderError::Resource("surface patterns need a registered Standard material".into())
        })?;
        let albedo = material.albedo;
        let baked = material.effects.baked;
        if patterns
            .iter()
            .any(|row| row[0] <= 0.0 || row.iter().any(|x| !x.is_finite()))
        {
            return Err(RenderError::Resource(
                "invalid Standard surface pattern parameters".into(),
            ));
        }
        let mut data = [0.0; SURFACE_FLOATS];
        for (index, row) in patterns.iter().enumerate() {
            data[index * 4..index * 4 + 4].copy_from_slice(&row[..4]);
            data[36 + index] = row[4];
        }
        data[48] = 1.0;
        data[49] = u8::from(detail) as f32;
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        if let Some(cached) = self
            .surface_materials
            .get_mut(&id)
            .filter(|entry| entry.albedo == albedo && entry.baked == baked)
        {
            if cached.values != data {
                queue.write_buffer(&cached.uniform, 0, bytemuck::cast_slice(&data));
                cached.values = data;
            }
            return Ok(());
        }
        let uniform = gpu_buffer(
            device,
            "native surface material settings",
            (SURFACE_FLOATS * 4) as u64,
            wgpu::BufferUsages::UNIFORM,
        );
        queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        let texture = &self.textures[&albedo];
        let array = baked.map_or(&self.empty_baked, |id| &self.baked_arrays[&id]);
        let group = self.albedo_group(&texture.view, &texture.sampler, &uniform, array);
        self.surface_materials.insert(
            id,
            SurfaceBinding {
                albedo,
                baked,
                uniform,
                group,
                values: data,
            },
        );
        Ok(())
    }

    fn bind_albedo(&self, view: wgpu::TextureView, sampler: wgpu::Sampler) -> MeshTexture {
        let group = self.albedo_group(&view, &sampler, &self.surface_default, &self.empty_baked);
        MeshTexture {
            group,
            view,
            sampler,
        }
    }

    fn albedo_group(
        &self,
        view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        surface: &wgpu::Buffer,
        array: &BakedArray,
    ) -> wgpu::BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("manifoldx-albedo"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.baked_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&array.view),
                },
                wgpu::BindGroupEntry {
                    binding: 64,
                    resource: surface.as_entire_binding(),
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
            DrawOptions {
                ignore_transform,
                ..Default::default()
            },
        )
    }

    /// Capture exact group labels. Opaque unlabelled meshes still occlude.
    pub fn render_ids(
        &mut self,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        width: u32,
        height: u32,
        labels: &HashMap<u32, u32>,
    ) -> Result<RenderOutput, RenderError> {
        self.render_inner(
            world,
            source,
            camera,
            width,
            height,
            DrawOptions {
                ids: Some(labels),
                ..Default::default()
            },
        )
    }

    fn render_inner(
        &mut self,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        width: u32,
        height: u32,
        draw: DrawOptions<'_>,
    ) -> Result<RenderOutput, RenderError> {
        let started = Instant::now();
        if !(1..=MAX_SIDE).contains(&width) || !(1..=MAX_SIDE).contains(&height) {
            return Err(RenderError::BadSize { width, height });
        }
        let offscreen = self
            .offscreen
            .take()
            .filter(|target| target.size == [width, height])
            .unwrap_or_else(|| OffscreenTarget::new(self.device(), [width, height], self.format));
        let result = self.read_offscreen(world, source, camera, draw, &offscreen);
        self.offscreen = Some(offscreen);
        result.map(|mut output| {
            output.stats.total_ns = elapsed_ns(started.elapsed());
            output
        })
    }

    fn read_offscreen(
        &mut self,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        draw: DrawOptions<'_>,
        offscreen: &OffscreenTarget,
    ) -> Result<RenderOutput, RenderError> {
        let [width, height] = offscreen.size;
        let encode_started = Instant::now();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("manifoldx-native-encoder"),
            });
        let target = FrameTarget {
            color: &offscreen.view,
            depth: &offscreen.depth,
            width,
            height,
            clear: if draw.ids.is_some() {
                wgpu::Color::BLACK
            } else {
                self.clear
            },
            timestamps: None,
            positions: None,
        };
        let mut stats = self.encode_frame(&mut encoder, target, world, source, camera, draw)?;

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
                texture: &offscreen.color,
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
        if self.format == wgpu::TextureFormat::Bgra8UnormSrgb {
            for pixel in rgba.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        drop(mapped);
        readback.unmap();
        stats.wait_readback_ns = elapsed_ns(readback_started.elapsed());

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
        self.encode_frame(
            encoder,
            target,
            world,
            source,
            camera,
            DrawOptions::default(),
        )
    }

    fn encode_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        draw: DrawOptions<'_>,
    ) -> Result<RenderStats, RenderError> {
        let view_order = self
            .secondary
            .as_ref()
            .filter(|_| draw.ids.is_none() && !draw.omit_draw)
            .map(|view| view.options.order);
        let mut order = Vec::new();
        if view_order == Some(ViewOrder::BeforeMain) {
            self.draw_secondary(encoder, world, source)?;
            order.push("view");
        }
        let mut stats = self.encode_primary(encoder, target, world, source, camera, draw)?;
        if stats.shadow_layers > 0 {
            order.push("shadow");
        }
        order.append(&mut stats.frame_order);
        if stats.line_draws > 0 {
            order.push("lines");
        }
        order.extend(stats.post_order.iter().copied());
        if view_order == Some(ViewOrder::AfterMain) {
            self.draw_secondary(encoder, world, source)?;
            order.push("view");
        }
        stats.frame_order = order;
        Ok(stats)
    }

    fn draw_secondary(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
        source: InstanceSource,
    ) -> Result<(), RenderError> {
        let mut view = self.secondary.take().expect("scheduled secondary view");
        let [r, g, b] = view.options.clear.map(f64::from);
        let target = FrameTarget {
            color: &view.color,
            depth: &view.depth,
            width: view.options.size[0],
            height: view.options.size[1],
            clear: wgpu::Color { r, g, b, a: 1.0 },
            timestamps: None,
            positions: None,
        };
        // Each view has independent uniform and instance uploads. Queue writes
        // for the later main pass cannot change what the earlier pass reads.
        std::mem::swap(&mut self.camera_buffer, &mut view.buffers.camera);
        std::mem::swap(&mut self.camera_bind_group, &mut view.buffers.camera_group);
        std::mem::swap(&mut self.instance_buffer, &mut view.buffers.instances);
        std::mem::swap(&mut self.palette_buffer, &mut view.buffers.palettes);
        let draw = DrawOptions {
            secondary: Some(&view.options),
            ..Default::default()
        };
        let result = self.encode_world(encoder, target, world, source, &view.options.camera, draw);
        std::mem::swap(&mut self.camera_buffer, &mut view.buffers.camera);
        std::mem::swap(&mut self.camera_bind_group, &mut view.buffers.camera_group);
        std::mem::swap(&mut self.instance_buffer, &mut view.buffers.instances);
        std::mem::swap(&mut self.palette_buffer, &mut view.buffers.palettes);
        self.secondary = Some(view);
        result.map(|_| ())
    }

    fn encode_primary(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        draw: DrawOptions<'_>,
    ) -> Result<RenderStats, RenderError> {
        let output = target.color;
        let size = [target.width, target.height];
        let Some(options) = self.post_options.clone().filter(|_| draw.ids.is_none()) else {
            let mut stats = self.encode_world(encoder, target, world, source, camera, draw)?;
            if draw.ids.is_none() && self.draw_rects(encoder, output, size) {
                stats.post_order.push("rect");
            }
            return Ok(stats);
        };
        let mut post = self
            .post_passes
            .take()
            .unwrap_or_else(|| PostPasses::new(self.device(), self.format));
        post.prepare(self.device(), size, target.depth);
        let clear = [
            target.clear.r as f32,
            target.clear.g as f32,
            target.clear.b as f32,
        ];
        let scene = FrameTarget {
            color: post.scene(),
            ..target
        };
        let mut result = self.encode_world(encoder, scene, world, source, camera, draw);
        if let Ok(stats) = &mut result {
            stats.post_target_generation = post.generation();
            if self.rect_before_post && self.draw_rects(encoder, post.scene(), size) {
                stats.post_order.push("rect");
            }
            stats.post_order.extend(post.encode(
                self.queue(),
                encoder,
                output,
                &options,
                camera,
                self.post_sun,
                clear,
            ));
            if !self.rect_before_post && self.draw_rects(encoder, output, size) {
                stats.post_order.push("rect");
            }
        }
        self.post_passes = Some(post);
        result
    }

    fn draw_rects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        size: [u32; 2],
    ) -> bool {
        if self.rects.is_empty() {
            return false;
        }
        let pass = self
            .rect_pass
            .get_or_insert_with(|| RectPass::new(&self.gpu.device, self.format));
        pass.draw(&self.gpu, encoder, output, size, &self.rects);
        true
    }

    fn material_globals(
        &self,
        camera: &Camera,
        target: &FrameTarget<'_>,
        shadow: Option<&ShadowFrame>,
    ) -> [f32; crate::manual::GLOBAL_FLOATS] {
        let aspect = target.width as f32 / target.height as f32;
        let mut globals = [0.0; crate::manual::GLOBAL_FLOATS];
        globals[..16].copy_from_slice(&camera.view_projection(aspect).to_columns());
        globals[16..32].copy_from_slice(&camera.view().to_columns());
        globals[32..48].copy_from_slice(&camera.projection(aspect).to_columns());
        globals[48..51].copy_from_slice(&camera.position);
        globals[51] = self.animation[0];
        globals[52..54].copy_from_slice(&[target.width as f32, target.height as f32]);
        globals[54..56].copy_from_slice(&self.animation[1..3]);
        if let Some((intensity, _)) = self.environment {
            globals[56] = intensity;
            globals[57] = f32::from_bits(1);
        }
        if let Some(shadow) = shadow {
            globals[60..76].copy_from_slice(&shadow.matrices[0].to_columns());
        }
        globals[76..84].copy_from_slice(&self.sun);
        globals[88..101].copy_from_slice(&self.spot[..13]);
        if let Some((range, color)) = self.fog {
            globals[102..104].copy_from_slice(&range);
            globals[104..107].copy_from_slice(&color);
            globals[107] = f32::from_bits(1);
        }
        globals[108..].copy_from_slice(self.benders.as_flattened());
        globals
    }

    fn encode_world(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
        world: &World,
        source: InstanceSource,
        camera: &Camera,
        draw: DrawOptions<'_>,
    ) -> Result<RenderStats, RenderError> {
        if source.max_instances > self.max_instances {
            return Err(RenderError::InstanceBound {
                requested: source.max_instances,
                configured: self.max_instances,
            });
        }

        let read_started = Instant::now();
        let mut batch = read_instances(world, source, draw.ignore_transform)?;
        for marker in ["PointCloud", "TextLabel", "Volume"] {
            if let Some(column) = world
                .component_index(marker)
                .filter(|_| !batch.instances.is_empty())
            {
                let entities: Vec<_> = batch
                    .instances
                    .iter()
                    .map(|instance| instance.entity)
                    .collect();
                let mut marked = world
                    .memberships(column, &entities)
                    .map_err(|error| RenderError::Resource(error.to_string()))?
                    .into_iter();
                batch
                    .instances
                    .retain(|_| !marked.next().expect("one marker per instance"));
            }
        }
        let fades = draw.ids.is_none() && self.materials.values().any(|m| m.effects.faded);
        let tints = draw.ids.is_none() && self.materials.values().any(|m| m.effects.tinted);
        let palettes = draw.ids.is_none() && self.materials.values().any(|m| m.effects.paletted);
        let baked =
            draw.ids.is_none() && self.materials.values().any(|m| m.effects.baked.is_some());
        let styles = instance_styles(world, &batch, fades, tints, palettes, baked)?;
        let layers = draw
            .secondary
            .map(|view| &view.layers)
            .or(self.layer_filter.as_deref());
        // A layer can hide a caster from the camera without hiding its shadow.
        let view_index = usize::from(draw.secondary.is_some());
        if draw.ids.is_none()
            && !draw.omit_draw
            && (self.sprites.is_some() || self.labels.is_some() || self.volumes.is_some())
        {
            let globals = self.material_globals(camera, &target, None);
            for pass in self.sprites.iter_mut().chain(self.labels.iter_mut()) {
                pass.prepare(&self.gpu, world, source, layers, &globals, view_index)?;
            }
            if let Some(volumes) = &mut self.volumes {
                volumes.prepare(&self.gpu, world, source, layers, &globals, view_index)?;
            }
        }
        let shadow_batch = (layers.is_some()
            && self.shadow_options.is_some()
            && draw.ids.is_none()
            && draw.secondary.is_none()
            && !draw.omit_draw)
            .then(|| batch.clone());
        if let Some(layers) = layers {
            filter_layers(world, &mut batch, layers)?;
        }
        let shadow_batch = shadow_batch.as_ref().unwrap_or(&batch);
        if draw.secondary.is_some_and(|view| {
            batch.instances.iter().any(|instance| {
                instance
                    .material_id
                    .and_then(|id| self.materials.get(&id))
                    .is_some_and(|material| material.albedo == view.texture_id)
                    || instance
                        .material_id
                        .and_then(|id| self.manual.get(&id))
                        .is_some_and(|manual| manual.samples(view.texture_id))
            })
        }) {
            return Err(RenderError::Resource(
                "a secondary view cannot sample its own target".into(),
            ));
        }
        let mut stats = RenderStats {
            entities_examined: batch.entities_examined,
            ..Default::default()
        };
        let aspect = target.width as f32 / target.height as f32;
        let view_projection = camera.view_projection(aspect);
        let draw_groups = self.pack_instances(
            &batch,
            &view_projection,
            PackOptions {
                positions: target.positions.as_ref(),
                ids: draw.ids,
                styles: &styles,
                shadow: false,
            },
            &mut stats,
        )?;
        stats.read_ns = elapsed_ns(read_started.elapsed());

        let upload_started = Instant::now();
        self.skin.upload(&self.gpu);
        let frame = self
            .shadow_options
            .filter(|_| draw.ids.is_none() && !draw.omit_draw && draw.secondary.is_none())
            .and_then(|options| {
                let bounds =
                    if options.auto_fit && options.cascades == 1 && self.spot_light.is_none() {
                        self.scene_bounds(shadow_batch)
                    } else {
                        None
                    };
                ShadowFrame::new(
                    options,
                    camera,
                    aspect,
                    self.post_sun,
                    self.spot_light,
                    bounds,
                )
            });
        if draw.secondary.is_none() {
            self.shadow
                .prepare(&self.gpu.device, &self.gpu.queue, frame.as_ref())?;
            if let Some(frame) = frame {
                self.encode_shadows(encoder, shadow_batch, &frame, &styles)?;
                stats.shadow_layers = frame.layers;
            }
        }
        let manual_index = usize::from(draw.secondary.is_some());
        if draw_groups.iter().any(|group| group.manual.is_some()) {
            let manual_globals = self.material_globals(camera, &target, frame.as_ref());
            let transforms: Vec<[f32; 16]> = draw_groups
                .iter()
                .flat_map(|group| {
                    group
                        .values
                        .as_chunks::<INSTANCE_FLOATS>()
                        .0
                        .iter()
                        .map(|row| {
                            row[..16]
                                .try_into()
                                .expect("a packed row begins with a model matrix")
                        })
                })
                .collect();
            let device = &self.gpu.device;
            if self.manual_frames[manual_index].is_none() {
                self.manual_frames[manual_index] =
                    Some(ManualFrame::new(device, self.max_instances)?);
            }
            let view = self.manual_frames[manual_index]
                .as_mut()
                .expect("created above");
            view.write(
                &self.gpu.queue,
                &manual_globals,
                &transforms,
                &self.point_lights,
            );
            for id in draw_groups.iter().filter_map(|group| group.manual) {
                view.bind(device, id, &self.manual[&id], &self.textures)?;
            }
        }
        let mut globals = [0.0; GLOBAL_BYTES / 4];
        globals[..16].copy_from_slice(&view_projection.to_columns());
        globals[16..19].copy_from_slice(&camera.position);
        globals[20..52].copy_from_slice(self.point_lights.as_flattened());
        globals[52..60].copy_from_slice(&self.sun);
        globals[60..76].copy_from_slice(&self.spot);
        globals[76..80].copy_from_slice(&self.animation);
        globals[80..112].copy_from_slice(self.benders.as_flattened());
        let environment = draw.ids.is_none().then_some(self.environment).flatten();
        if let Some((intensity, _)) = environment {
            let projection = camera.projection(aspect);
            globals[112..128].copy_from_slice(&camera.view().to_columns());
            globals[128..132].copy_from_slice(&[intensity, 1.0, projection.0[0], projection.0[5]]);
        }
        if let Some((range, color)) = self.fog.filter(|_| draw.ids.is_none()) {
            globals[132..136].copy_from_slice(&[color[0], color[1], color[2], 1.0]);
            globals[136..138].copy_from_slice(&range);
        }
        globals[140..142].copy_from_slice(&self.origin_tile);
        self.gpu
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&globals));

        if draw_groups.iter().any(|group| !group.palettes.is_empty()) {
            let mut colors = Vec::with_capacity(
                draw_groups
                    .iter()
                    .map(|group| group.instances as usize)
                    .sum::<usize>()
                    * PALETTE_SLOTS,
            );
            for group in &draw_groups {
                if group.palettes.is_empty() {
                    colors.resize(
                        colors.len() + group.instances as usize * PALETTE_SLOTS,
                        [1.0; 4],
                    );
                } else {
                    colors.extend_from_slice(&group.palettes);
                }
            }
            let bytes = bytemuck::cast_slice(&colors);
            let needed = bytes.len() as u64;
            let limit = self.gpu.device.limits().max_storage_buffer_binding_size;
            if needed > limit {
                return Err(RenderError::Resource(
                    "instance palette data exceeds device limits".into(),
                ));
            }
            if self
                .palette_buffer
                .as_ref()
                .is_none_or(|buffer| buffer.capacity_bytes < needed)
            {
                let capacity = needed.next_power_of_two().min(limit);
                let buffer = gpu_buffer(
                    &self.gpu.device,
                    "native instance palettes",
                    capacity,
                    wgpu::BufferUsages::STORAGE,
                );
                self.camera_bind_group = camera_group(
                    &self.gpu.device,
                    &self.camera_layout,
                    &self.camera_buffer,
                    &buffer,
                );
                self.palette_buffer = Some(InstanceBuffer {
                    buffer,
                    capacity_bytes: capacity,
                });
            }
            let palette = self
                .palette_buffer
                .as_ref()
                .expect("palette buffer prepared");
            self.gpu.queue.write_buffer(&palette.buffer, 0, bytes);
        }
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
                let pass = self.position_pass.as_ref().ok_or_else(|| {
                    RenderError::Device("resident culling pipeline is not prepared".into())
                })?;
                pass.encode(
                    &self.gpu,
                    encoder,
                    [
                        source.buffer,
                        &self.instance_buffer.as_ref().expect("instance data").buffer,
                        &self.camera_buffer,
                    ],
                    [
                        source.row_width,
                        source.offset,
                        metadata.len() as u32,
                        INSTANCE_FLOATS as u32,
                    ],
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
            let scene = if draw.secondary.is_some() {
                &self.shadow.inactive_group
            } else {
                &self.shadow.group
            };
            if !draw.omit_draw && environment.is_some_and(|(_, sky)| sky) {
                pass.set_pipeline(&self.skybox_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, &self.textures[&0].group, &[]);
                pass.set_bind_group(2, &self.skin.group, &[]);
                pass.set_bind_group(3, scene, &[]);
                pass.draw(0..3, 0..1);
                stats.frame_order.push("skybox");
            }
            stats
                .frame_order
                .push(if draw.ids.is_some() { "ids" } else { "main" });
            if !draw.omit_draw && !draw_groups.is_empty() {
                let instance_buffer = &self
                    .instance_buffer
                    .as_ref()
                    .expect("groups imply data")
                    .buffer;
                // Draw ranges index the full buffer in packed group order.
                pass.set_vertex_buffer(1, instance_buffer.slice(..));
                let mut first_instance = 0u32;
                for group in &draw_groups {
                    if group.axis {
                        first_instance += group.instances;
                        continue;
                    }
                    if let Some(id) = group.manual {
                        let manual = &self.manual[&id];
                        let view = self.manual_frames[manual_index]
                            .as_ref()
                            .expect("manual frame prepared");
                        pass.set_pipeline(&manual.pipelines[usize::from(group.reversed)]);
                        pass.set_bind_group(0, &view.groups[&id], &[]);
                    } else if let Some(id) = group.fragment {
                        pass.set_bind_group(0, &self.camera_bind_group, &[]);
                        pass.set_bind_group(2, &self.skin.group, &[]);
                        let fragment = &self.fragments[&id];
                        pass.set_pipeline(&fragment.pipelines[usize::from(group.reversed)]);
                        pass.set_bind_group(1, &fragment.group, &[]);
                    } else {
                        pass.set_bind_group(0, &self.camera_bind_group, &[]);
                        pass.set_bind_group(2, &self.skin.group, &[]);
                        let pipeline = if group.rank.0 {
                            &self.translucent_pipeline
                        } else if group.reversed {
                            &self.reversed_pipeline
                        } else {
                            &self.pipeline
                        };
                        pass.set_pipeline(pipeline);
                        let material = group
                            .surface
                            .and_then(|id| self.surface_materials.get(&id))
                            .map(|surface| &surface.group)
                            .unwrap_or(&self.textures[&group.albedo].group);
                        pass.set_bind_group(1, material, &[]);
                        pass.set_bind_group(3, scene, &[]);
                    }
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
            if draw.ids.is_none() && !draw.omit_draw {
                if let Some(sprites) = &self.sprites {
                    let (draws, triangles) = sprites.draw(&mut pass, view_index);
                    if draws > 0 {
                        stats.frame_order.push("sprite");
                    }
                    stats.draw_calls += draws;
                    stats.triangles += triangles;
                }
                if let Some(volumes) = &self.volumes {
                    let draws = volumes.draw(&mut pass, view_index);
                    if draws > 0 {
                        stats.frame_order.push("volume");
                    }
                    stats.draw_calls += draws;
                    stats.triangles += draws;
                }
                if let Some(labels) = &self.labels {
                    let (draws, triangles) = labels.draw(&mut pass, view_index);
                    if draws > 0 {
                        stats.frame_order.push("label");
                    }
                    stats.draw_calls += draws;
                    stats.triangles += triangles;
                }
                if draw_groups.iter().any(|group| group.axis) {
                    stats.frame_order.push("axis");
                }
                let mut first = 0;
                for group in &draw_groups {
                    let end = first + group.instances;
                    if group.axis {
                        let id = group.manual.expect("axes carry a manual line material");
                        let view = self.manual_frames[manual_index]
                            .as_ref()
                            .expect("axis frame prepared");
                        let mesh = &self.meshes[&group.geometry_id];
                        pass.set_pipeline(&self.manual[&id].pipelines[0]);
                        pass.set_bind_group(0, &view.groups[&id], &[]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, first..end);
                        stats.draw_calls += 1;
                    }
                    first = end;
                }
            }
        }

        if draw.ids.is_none() && draw.secondary.is_none() && !draw.omit_draw {
            let attach = FrameTarget {
                color: target.color,
                depth: target.depth,
                width: target.width,
                height: target.height,
                clear: target.clear,
                timestamps: None,
                positions: None,
            };
            if let Some(lines) = self
                .lines
                .as_ref()
                .filter(|lines| !lines.layers.is_empty() || self.line_pass.is_some())
            {
                let pass = self
                    .line_pass
                    .get_or_insert_with(|| LinePass::new(&self.gpu.device, self.format));
                stats.line_draws = pass.encode(&self.gpu, encoder, &attach, camera, lines);
            }
        }
        Ok(stats)
    }

    fn scene_bounds(&self, batch: &InstanceBatch) -> Option<([f32; 3], f32)> {
        let mut lower = [f32::INFINITY; 3];
        let mut upper = [f32::NEG_INFINITY; 3];
        for instance in &batch.instances {
            let Some(mesh) = self
                .meshes
                .get(&instance.geometry_id)
                .filter(|mesh| !mesh.axis)
            else {
                continue;
            };
            let [lo, hi] = mesh.bounds;
            for x in [lo[0], hi[0]] {
                for y in [lo[1], hi[1]] {
                    for z in [lo[2], hi[2]] {
                        let point = instance.model.transform_point([x, y, z]);
                        for axis in 0..3 {
                            lower[axis] = lower[axis].min(point[axis]);
                            upper[axis] = upper[axis].max(point[axis]);
                        }
                    }
                }
            }
        }
        if !lower[0].is_finite() {
            return None;
        }
        let center = std::array::from_fn(|axis| 0.5 * (lower[axis] + upper[axis]));
        let diameter = (0..3)
            .map(|axis| (upper[axis] - lower[axis]).powi(2))
            .sum::<f32>()
            .sqrt();
        Some((center, (diameter * 0.5).max(0.001)))
    }

    fn encode_shadows(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        batch: &InstanceBatch,
        frame: &ShadowFrame,
        styles: &HashMap<EntityId, InstanceStyle>,
    ) -> Result<(), RenderError> {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        for index in 0..frame.layers {
            let matrix = &frame.matrices[index];
            let groups = self.pack_instances(
                batch,
                matrix,
                PackOptions {
                    positions: None,
                    ids: None,
                    styles,
                    shadow: true,
                },
                &mut RenderStats::default(),
            )?;
            let data = self.pack_instance_bytes(&groups);
            if self.shadow_layers.len() <= index {
                self.shadow_layers.push(view_buffers(
                    device,
                    &self.camera_layout,
                    &self.empty_palette,
                ));
            }
            let layer = &mut self.shadow_layers[index];
            let mut globals = [0.0; GLOBAL_BYTES / 4];
            globals[..16].copy_from_slice(&matrix.to_columns());
            globals[79] = 1.0; // Shadow positions exclude wind and benders.
            queue.write_buffer(&layer.camera, 0, bytemuck::cast_slice(&globals));
            if !data.is_empty() {
                let size = data.len() as u64;
                if layer
                    .instances
                    .as_ref()
                    .is_none_or(|buffer| buffer.capacity_bytes < size)
                {
                    let capacity_bytes = size.next_power_of_two();
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("native shadow instances"),
                        size: capacity_bytes,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    layer.instances = Some(InstanceBuffer {
                        buffer,
                        capacity_bytes,
                    });
                }
                let buffer = &layer
                    .instances
                    .as_ref()
                    .expect("shadow group implies instance buffer")
                    .buffer;
                queue.write_buffer(buffer, 0, &data);
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("native shadow layer"),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow.depth[index],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if !groups.is_empty() {
                let buffer = &layer
                    .instances
                    .as_ref()
                    .expect("shadow group implies instance buffer")
                    .buffer;
                pass.set_vertex_buffer(1, buffer.slice(..));
                let mut first = 0;
                for group in groups {
                    let mesh = &self.meshes[&group.geometry_id];
                    pass.set_pipeline(&self.shadow_pipelines[usize::from(group.reversed)]);
                    pass.set_bind_group(0, &layer.camera_group, &[]);
                    pass.set_bind_group(1, &self.textures[&0].group, &[]);
                    pass.set_bind_group(2, &self.skin.group, &[]);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, first..first + group.instances);
                    first += group.instances;
                }
            }
        }
        Ok(())
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
        self.render_inner(
            world,
            source,
            camera,
            width,
            height,
            DrawOptions {
                ignore_transform,
                omit_draw: true,
                ..Default::default()
            },
        )
    }

    fn pack_instances(
        &self,
        batch: &InstanceBatch,
        view_projection: &Mat4,
        options: PackOptions<'_>,
        stats: &mut RenderStats,
    ) -> Result<Vec<DrawGroup>, RenderError> {
        let PackOptions {
            positions,
            ids,
            styles,
            shadow,
        } = options;
        let mut groups = BTreeMap::<_, DrawGroup>::new();
        for instance in &batch.instances {
            let mesh = self
                .meshes
                .get(&instance.geometry_id)
                .ok_or(RenderError::UnknownGeometry(instance.geometry_id))?;
            if mesh.axis && (shadow || ids.is_some()) {
                continue;
            }
            if ids.is_none() {
                let invalid = instance
                    .material_id
                    .and_then(|id| self.invalid_materials.get(&id));
                if let Some(error) = invalid {
                    return Err(RenderError::Resource(error.clone()));
                }
            }
            let slot = instance.entity.slot();
            let bound = positions.is_some_and(|source| {
                source.members.get(slot as usize) == Some(&Some(instance.entity))
            });
            let material = if let Some(labels) = ids {
                let [r, g, b, _] = labels.get(&slot).copied().unwrap_or_default().to_le_bytes();
                Material {
                    color: srgb_to_linear([r, g, b].map(|value| f32::from(value) / 255.0)),
                    style: FLAT_STYLE,
                    albedo: 0,
                    effects: StandardEffects::default(),
                }
            } else {
                instance
                    .material_id
                    .and_then(|id| self.materials.get(&id).copied())
                    .unwrap_or(Material::diffuse(mesh.color))
            };
            let effects = material.effects;
            let mirrored = effects.mirror.is_some();
            let manual = instance
                .material_id
                .filter(|id| ids.is_none() && self.manual.contains_key(id));
            if mesh.axis != manual.is_some_and(|id| self.manual[&id].axis) {
                return Err(RenderError::Resource(
                    "axis geometry requires AxisMaterial and vice versa".into(),
                ));
            }
            let translucent =
                effects.opacity < 1.0 || manual.is_some_and(|id| self.manual[&id].translucent);
            if shadow && (translucent || mirrored) {
                continue;
            }
            let rank = (translucent, mirrored);
            let displayed = effects.mirror.map_or(instance.model, |height| {
                reflection(height).multiply(&instance.model)
            });
            let deformed = mesh.skinned || (!shadow && effects.sway > 0.0);
            if !bound
                && !deformed
                && !mesh.axis
                && outside_frustum(&view_projection.multiply(&displayed), mesh.bounds)
            {
                continue;
            }
            let geometry_id = instance.geometry_id;
            let reversed = flips_winding(&displayed);
            let albedo = material.albedo;
            let alpha_material = translucent.then_some(instance.material_id).flatten();
            let fragment = instance
                .material_id
                .filter(|id| !shadow && ids.is_none() && self.fragments.contains_key(id));
            let manual = manual.filter(|_| !shadow);
            let surface = instance.material_id.filter(|id| {
                !shadow
                    && ids.is_none()
                    && effects.vertex_colors
                    && self.surface_materials.contains_key(id)
            });
            let group = groups
                .entry((
                    rank,
                    geometry_id,
                    reversed,
                    albedo,
                    alpha_material,
                    fragment,
                    manual,
                    surface,
                    effects.paletted,
                ))
                .or_insert_with(|| DrawGroup {
                    geometry_id,
                    reversed,
                    albedo,
                    fragment,
                    manual,
                    axis: mesh.axis,
                    surface,
                    rank,
                    first_slot: slot,
                    instances: 0,
                    values: Vec::new(),
                    positions: Vec::new(),
                    palettes: Vec::new(),
                });
            group.values.extend_from_slice(&instance.model.to_columns());
            group.values.extend_from_slice(&material.color);
            group.values.extend_from_slice(&material.style);
            group.values.extend_from_slice(&[
                effects.opacity,
                effects.mirror.unwrap_or_default(),
                u8::from(mirrored) as f32,
                if shadow {
                    u8::from(effects.vertex_colors) as f32
                } else if effects.paletted {
                    2.0
                } else if effects.baked.is_some() {
                    3.0
                } else {
                    u8::from(effects.vertex_colors) as f32
                },
            ]);
            group
                .values
                .extend_from_slice(&effects.cut.unwrap_or_default());
            let skin = if mesh.skinned {
                self.skin.frame(instance.entity)
            } else {
                0
            };
            group.values.push(f32::from_bits(skin));
            let style = styles.get(&instance.entity).copied().unwrap_or_default();
            group.values.extend_from_slice(&[
                effects.sway,
                effects.sway_reach,
                effects.ripple,
                if effects.faded { style.fade } else { 1.0 },
            ]);
            let tint_or_baked_layer = if effects.tinted {
                style.tint
            } else if effects.baked.is_some() {
                [style.baked_layer, 0.0, 0.0]
            } else {
                [1.0; 3]
            };
            group.values.extend_from_slice(&tint_or_baked_layer);
            if effects.paletted && !shadow {
                group.palettes.extend_from_slice(&style.palette);
            }
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
                    u32::from(deformed),
                ]);
            }
            group.instances += 1;
            stats.instances += 1;
        }
        let mut groups: Vec<_> = groups.into_values().collect();
        groups.sort_by_key(|group| (group.rank, if group.rank.0 { group.first_slot } else { 0 }));
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

fn instance_styles(
    world: &World,
    batch: &InstanceBatch,
    fades: bool,
    tints: bool,
    palettes: bool,
    baked: bool,
) -> Result<HashMap<EntityId, InstanceStyle>, RenderError> {
    let mut styles = HashMap::<_, InstanceStyle>::new();
    if !fades && !tints && !palettes && !baked {
        return Ok(styles);
    }
    let ids: Vec<_> = batch
        .instances
        .iter()
        .map(|instance| instance.entity)
        .collect();
    for (enabled, name, width) in [
        (fades, "Fade", 1),
        (tints, "Tint", 3),
        (palettes, "Palette", PALETTE_SLOTS * 3),
        (baked, "Baked", 1),
    ] {
        if !enabled {
            continue;
        }
        let Some(column) = world.component_index(name) else {
            continue;
        };
        let rows = world
            .read_component(column, Some(&ids))
            .map_err(|error| RenderError::Resource(error.to_string()))?;
        let ColumnData::F32(values) = rows.data else {
            return Err(RenderError::Resource(format!("{name} has invalid storage")));
        };
        if rows.schema.width as usize != width {
            return Err(RenderError::Resource(format!("{name} has invalid width")));
        }
        for ((id, member), row) in ids
            .iter()
            .zip(rows.membership)
            .zip(values.chunks_exact(width))
        {
            if member {
                let style = styles.entry(*id).or_default();
                match name {
                    "Fade" => style.fade = row[0],
                    "Tint" => style.tint.copy_from_slice(row),
                    "Baked" => style.baked_layer = row[0],
                    _ => {
                        for (color, rgb) in style.palette.iter_mut().zip(row.as_chunks::<3>().0) {
                            color[..3].copy_from_slice(rgb);
                        }
                    }
                }
            }
        }
    }
    Ok(styles)
}

fn view_buffers(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    empty_palette: &wgpu::Buffer,
) -> ViewBuffers {
    let camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native view camera"),
        size: GLOBAL_BYTES as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let camera_group = camera_group(device, layout, &camera, empty_palette);
    ViewBuffers {
        camera,
        camera_group,
        palettes: None,
        instances: None,
    }
}

fn camera_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &wgpu::Buffer,
    palettes: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native view camera binding"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: palettes.as_entire_binding(),
            },
        ],
    })
}

fn filter_layers(
    world: &World,
    batch: &mut InstanceBatch,
    layers: &BTreeSet<i32>,
) -> Result<(), RenderError> {
    let Some(index) = world.component_index("Layer") else {
        if !layers.contains(&0) {
            batch.instances.clear();
        }
        return Ok(());
    };
    let ids: Vec<_> = batch
        .instances
        .iter()
        .map(|instance| instance.entity)
        .collect();
    let rows = world
        .read_component(index, Some(&ids))
        .map_err(|error| RenderError::Resource(error.to_string()))?;
    let ColumnData::F32(values) = rows.data else {
        return Err(RenderError::Resource(
            "Layer requires one f32 column".into(),
        ));
    };
    if rows.schema.width != 1 {
        return Err(RenderError::Resource(
            "Layer requires one f32 column".into(),
        ));
    }
    let mut values = values.into_iter();
    batch
        .instances
        .retain(|_| layers.contains(&(values.next().expect("one layer per mesh") as i32)));
    Ok(())
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
    fragment: Option<u32>,
    manual: Option<u32>,
    axis: bool,
    surface: Option<u32>,
    rank: (bool, bool),
    first_slot: u32,
    instances: u32,
    values: Vec<f32>,
    positions: Vec<[u32; 8]>,
    palettes: Vec<[f32; 4]>,
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
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
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 9,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 48,
        shader_location: 12,
    },
];
const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: VERTEX_BYTES as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &VERTEX_ATTRIBUTES,
};
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 11] = [
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
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 96,
        shader_location: 10,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 112,
        shader_location: 11,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 128,
        shader_location: 13,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 132,
        shader_location: 14,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 148,
        shader_location: 15,
    },
];
const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: INSTANCE_BYTES as u64,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &INSTANCE_ATTRIBUTES,
};

const SHADER: &str = concat!(
    r#"
struct PointLight {
    position: vec4<f32>,
    rgb_intensity: vec4<f32>,
};
struct DirectionalLight {
    direction: vec4<f32>,
    rgb_intensity: vec4<f32>,
};
struct SpotLight {
    position_range: vec4<f32>,
    direction_cos_inner: vec4<f32>,
    rgb_intensity: vec4<f32>,
    cos_outer: vec4<f32>,
};
struct Camera {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
    lights: array<PointLight, 4>,
    sun: DirectionalLight,
    spot: SpotLight,
    animation: vec4<f32>,
    benders: array<vec4<f32>, 8>,
    view: mat4x4<f32>,
    environment: vec4<f32>,
    fog_color: vec4<f32>,
    fog_range: vec4<f32>,
    origin_tile: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> palettes: array<vec4<f32>>;
@group(1) @binding(0) var albedo_sampler: sampler;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(4) var baked_sampler: sampler;
@group(1) @binding(5) var baked_texture: texture_2d_array<f32>;
struct SurfaceSettings {
    patterns: array<vec4<f32>, 9>,
    darks: array<vec4<f32>, 3>,
    options: vec4<f32>,
};
@group(1) @binding(64) var<uniform> surface_settings: SurfaceSettings;
@group(2) @binding(0) var<storage, read> joint_mats: array<mat4x4<f32>>;
struct ShadowData {
    matrices: array<mat4x4<f32>, 4>,
    settings: vec4<f32>, // caster, resolution, PCF radius, single-map bias
    splits: vec4<f32>,
    texels: vec4<f32>,
    ranges: vec4<f32>,
    camera_depth: vec4<f32>,
    layers: vec4<f32>,
};
@group(3) @binding(0) var shadow_map: texture_depth_2d_array;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> shadows: ShadowData;
@group(3) @binding(3) var env_sampler: sampler;
@group(3) @binding(4) var irradiance_map: texture_cube<f32>;
@group(3) @binding(5) var prefiltered_map: texture_cube<f32>;
@group(3) @binding(6) var brdf_lut: texture_2d<f32>;

struct SkyOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) direction: vec3<f32>,
};
@vertex
fn vs_sky(@builtin(vertex_index) index: u32) -> SkyOutput {
    let points = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    let p = points[index];
    let ray = vec3<f32>(p / camera.environment.zw, -1.0);
    return SkyOutput(vec4<f32>(p, 0.0, 1.0), vec3<f32>(
        dot(camera.view[0].xyz, ray), dot(camera.view[1].xyz, ray), dot(camera.view[2].xyz, ray),
    ));
}
@fragment
fn fs_sky(in: SkyOutput) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(prefiltered_map, env_sampler, normalize(in.direction), 0.0).rgb
        * camera.environment.x;
    return vec4<f32>(color / (color + 1.0), 1.0);
}

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(7) uv: vec2<f32>,
    @location(9) color: vec4<f32>,
    @location(12) joints: u32,
};
struct InstanceIn {
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(8) style: vec4<f32>,
    @location(10) effects: vec4<f32>,
    @location(11) cut: vec4<f32>,
    @location(13) joint_pose: u32,
    @location(14) motion: vec4<f32>,
    @location(15) tint: vec3<f32>,
};
struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) style: vec4<f32>,
    @location(5) @interpolate(flat) effects: vec4<f32>,
    @location(6) @interpolate(flat) cut: vec4<f32>,
    @location(7) @interpolate(flat) ripple: f32,
    @location(8) @interpolate(flat) fade: f32,
    @location(9) surface: f32,
    @location(10) @interpolate(flat) material_color: vec3<f32>,
    @location(11) @interpolate(flat) baked_layer: f32,
};

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > 1e-20 && len_sq < 1e30 {
        return v * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn decode_srgb(srgb: vec3<f32>) -> vec3<f32> {
    return select(pow((srgb + 0.055) / 1.055, vec3<f32>(2.4)), srgb / 12.92,
                  srgb <= vec3<f32>(0.04045));
}

fn baked_color(packed: f32) -> vec3<f32> {
    let p = u32(round(packed));
    let rgb = vec3<f32>(f32((p >> 16u) & 255u), f32((p >> 8u) & 255u), f32(p & 255u));
    return decode_srgb(rgb / 255.0);
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

fn joint_matrix(pose: u32, joint: u32) -> mat4x4<f32> {
    let at = (pose & 0xfffffu) * 32u + joint;
    let a = joint_mats[at];
    let f = f32(pose >> 20u) / 4095.0;
    if f == 0.0 {
        return a;
    }
    return a + f * (joint_mats[at + 32u] - a);
}

fn posed_normal(m: mat4x4<f32>, normal: vec3<f32>) -> vec3<f32> {
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    let det = dot(a, cross(b, c));
    if abs(det) <= 1e-6 * length(a) * length(b) * length(c) {
        return normal;
    }
    let cofactors = mat3x3<f32>(cross(b, c), cross(c, a), cross(a, b));
    return safe_normalize(cofactors * normal * sign(det));
}

fn sway_vertex(world: vec4<f32>, height: f32, strength: f32, reach: f32) -> vec4<f32> {
    let tau = 6.28318530718;
    let p = world.xz + camera.animation.yz;
    let t = camera.animation.x;
    let a = tau * (27.0 * t / 64.0 - (23.0 * p.x + 13.0 * p.y) / 256.0);
    let b = tau * (61.0 * t / 64.0 - (-31.0 * p.x + 41.0 * p.y) / 256.0);
    let h = reach * tanh(max(height, 0.0) / reach);
    let low = h * (1.0 - smoothstep(0.8, 1.2, height));
    var offset = vec3<f32>(h * strength * (sin(a) + 0.35 * sin(b)), 0.0,
                          h * strength * 0.5 * (cos(a) + 0.35 * sin(b + 1.3)));
    for (var i = 0u; i < 8u; i++) {
        let bender = camera.benders[i];
        let off = world.xz - bender.xy;
        let distance = length(off);
        if bender.z > 0.0 && distance < bender.z && distance >= 1e-4 {
            let f = 1.0 - distance / bender.z;
            let lean = bender.w * f * f * low;
            offset += vec3<f32>(off.x / distance, -0.4, off.y / distance) * lean;
        }
    }
    return world + vec4<f32>(offset, 0.0);
}

@vertex
fn vs_main(in: VertexIn, instance: InstanceIn, @builtin(instance_index) index: u32) -> VertexOut {
    let model = mat4x4<f32>(instance.m0, instance.m1, instance.m2, instance.m3);
    var position = in.position;
    var normal = in.normal;
    if instance.joint_pose != 0u {
        let joints = in.joints;
        let w = f32(joints >> 16u) / 65535.0;
        let a = joint_matrix(instance.joint_pose, joints & 255u);
        let b = joint_matrix(instance.joint_pose, (joints >> 8u) & 255u);
        let blended = w * a + (1.0 - w) * b;
        position = (blended * vec4<f32>(position, 1.0)).xyz;
        normal = posed_normal(blended, normal);
    }
    var world = model * vec4<f32>(position, 1.0);
    if instance.motion.x > 0.0 && camera.animation.w == 0.0 {
        world = sway_vertex(world, position.y, instance.motion.x, instance.motion.y);
    }
    var out: VertexOut;
    var displayed = world;
    if instance.effects.z > 0.0 {
        displayed.y = 2.0 * instance.effects.y - world.y;
    }
    out.clip_position = select(vec4<f32>(0.0, 0.0, -1.0, 1.0), camera.view_proj * displayed, instance.m3.w != 0.0);
    out.normal = transform_normal(model, normal);
    out.color = instance.color.rgb * select(vec3<f32>(1.0), in.color.xyz * instance.tint, instance.effects.w > 0.0);
    out.material_color = instance.color.rgb;
    out.baked_layer = instance.tint.x;
    if instance.effects.w == 3.0 {
        out.color = instance.color.rgb * in.color.xyz;
    }
    if i32(in.color.w + 0.5) == 12 {
        if instance.effects.w == 3.0 {
            out.color = in.color.xyz;
        } else if instance.effects.w > 0.0 {
            out.color = instance.color.rgb * baked_color(in.color.z) * instance.tint;
        }
    }
    if instance.effects.w == 2.0 {
        let color = in.color.xyz;
        out.color = select(1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055,
                           color * 12.92, color <= vec3<f32>(0.0031308));
        let slot = i32(in.color.w + 0.5) - 32;
        if slot >= 0 && slot < 8 {
            out.color *= palettes[index * 8u + u32(slot)].xyz;
        }
    }
    out.effects = instance.effects;
    out.cut = instance.cut;
    out.ripple = instance.motion.z;
    out.fade = instance.motion.w;
    out.world_position = world.xyz;
    out.uv = in.uv;
    out.style = instance.style;
    out.surface = in.color.w;
    return out;
}

fn geometry_schlick(ndot: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return ndot / (ndot * (1.0 - k) + k);
}

fn standard_light(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, albedo: vec3<f32>,
    roughness: f32, metallic: f32, radiance: vec3<f32>,
) -> vec3<f32> {
    let h = safe_normalize(v + l);
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

fn shadow_taps(layer: i32, uv: vec2<f32>, depth: f32) -> f32 {
    let radius = i32(shadows.settings.z);
    let texel = 1.0 / shadows.settings.y;
    var sum = 0.0;
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, layer, depth);
        }
    }
    let diameter = f32(2 * radius + 1);
    return sum / (diameter * diameter);
}

fn shadow_sample(layer: i32, world: vec3<f32>, bias: f32, fade_edge: bool) -> f32 {
    let clip = shadows.matrices[layer] * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if clip.w <= 0.0 || any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let shade = shadow_taps(layer, uv, ndc.z - bias);
    if !fade_edge {
        return shade;
    }
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return mix(1.0, shade, smoothstep(0.0, 0.15, edge));
}

fn cascade_shadow(layer: i32, world: vec3<f32>, n: vec3<f32>, ndl: f32) -> f32 {
    let texel = shadows.texels[layer];
    let point = world + n * texel * (2.0 - ndl);
    let bias = texel / shadows.ranges[layer] * (1.0 + 2.0 * (1.0 - ndl));
    return shadow_sample(layer, point, bias, false);
}

fn shadow_factor(caster: f32, world: vec3<f32>, n: vec3<f32>, l: vec3<f32>) -> f32 {
    if shadows.settings.x != caster {
        return 1.0;
    }
    let ndl = max(dot(n, l), 0.0);
    let count = i32(shadows.layers.x);
    if count <= 1 {
        let bias = shadows.settings.w * (1.0 + 4.0 * (1.0 - ndl));
        return shadow_sample(0, world, bias, true);
    }
    let depth = -dot(shadows.camera_depth, vec4<f32>(world, 1.0));
    if depth > shadows.splits[count - 1] {
        return 1.0;
    }
    var layer = 0;
    while layer < count - 1 && depth > shadows.splits[layer] {
        layer += 1;
    }
    let shade = cascade_shadow(layer, world, n, ndl);
    var near_depth = 0.0;
    if layer > 0 {
        near_depth = shadows.splits[layer - 1];
    }
    let far_depth = shadows.splits[layer];
    let blend = 0.1 * (far_depth - near_depth);
    let t = clamp((depth - (far_depth - blend)) / blend, 0.0, 1.0);
    if t <= 0.0 {
        return shade;
    }
    var next = 1.0;
    if layer < count - 1 {
        next = cascade_shadow(layer + 1, world, n, ndl);
    }
    return mix(shade, next, t);
}

fn phong_color(in: VertexOut, n: vec3<f32>) -> vec3<f32> {
    let view = safe_normalize(camera.position.xyz - in.world_position);
    var color = 0.03 * in.color;
    for (var i = 0u; i < 4u; i++) {
        let light = camera.lights[i];
        let offset = light.position.xyz - in.world_position;
        let direction = safe_normalize(offset);
        let incidence = dot(n, direction);
        if light.rgb_intensity.a > 0.0 && incidence > 0.0 {
            let distance_sq = max(dot(offset, offset), 0.0001);
            let radiance = light.rgb_intensity.rgb * light.rgb_intensity.a / distance_sq;
            let highlight = pow(max(dot(reflect(-direction, n), view), 0.0), in.style.x);
            color += (in.color.xyz * incidence + vec3<f32>(highlight)) * radiance;
        }
    }
    return color;
}

const BAYER: array<u32, 16> = array<u32, 16>(
    0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u,
);

fn fade_dropped(pixel: vec2<f32>, fade: f32) -> bool {
    let p = vec2<u32>(pixel) % vec2<u32>(4u);
    let threshold = (f32(BAYER[p.y * 4u + p.x]) + 0.5) / 16.0;
    return select(threshold >= fade, threshold < 1.0 + fade, fade < 0.0);
}

@fragment
fn fs_shadow(in: VertexOut) {
    if in.fade < 1.0 && fade_dropped(in.clip_position.xy, in.fade) {
        discard;
    }
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let uv_dx = dpdx(in.uv);
    let uv_dy = dpdy(in.uv);
    let pattern_normal = abs(safe_normalize(in.normal));
    let pattern_pos = in.world_position + vec3<f32>(camera.animation.y, 0.0, camera.animation.z);
    let pattern_uv = select(select(pattern_pos.xy, pattern_pos.zy, pattern_normal.x >= pattern_normal.z),
        pattern_pos.xz, pattern_normal.y >= pattern_normal.x && pattern_normal.y >= pattern_normal.z);
    let pattern_width = length(fwidth(pattern_uv));
    let detail_x = length(dpdx(pattern_pos.xz));
    let detail_y = length(dpdy(pattern_pos.xz));
    let detail_width = vec2<f32>(sqrt(detail_x * detail_y), max(detail_x, detail_y));
    if in.fade < 1.0 && fade_dropped(in.clip_position.xy, in.fade) {
        discard;
    }
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
    if mode == 5u {
        return vec4<f32>(phong_color(in, n), 1.0);
    }
    if (in.effects.z > 0.0 && in.world_position.y < in.effects.y)
        || (all(in.world_position.xz > in.cut.xy) && all(in.world_position.xz < in.cut.zw)) {
        discard;
    }
    let baked_dx = dpdx(in.color.xy);
    let baked_dy = dpdy(in.color.xy);
    var albedo = in.color;
    if in.effects.w == 2.0 {
        albedo = decode_srgb(in.color) * in.material_color;
    }
    if mode == 4u {
        let texel = textureSampleGrad(albedo_texture, albedo_sampler, in.uv, uv_dx, uv_dy);
        if texel.a < 0.5 { discard; }
        albedo = texel.rgb;
    }
    var roughness = in.style.x;
    if mode != 4u && surface_settings.options.x > 0.0 {
        let surface_id = i32(in.surface + 0.5);
        albedo = surface_albedo(albedo, surface_id, pattern_uv, pattern_width);
        if surface_settings.options.y > 0.0 {
            let details = detail_surface(albedo, surface_id, pattern_pos.xz, detail_width,
                distance(camera.position.xyz, in.world_position));
            albedo = details.xyz;
            roughness = clamp(roughness + details.w, 0.04, 1.0);
        }
    }
    if mode == 3u && in.effects.w == 3.0 && i32(in.surface + 0.5) == 12 {
        if in.baked_layer >= 0.0 {
            let layer = i32(in.baked_layer + 0.5);
            albedo = textureSampleGrad(baked_texture, baked_sampler, in.color.xy,
                                       layer, baked_dx, baked_dy).rgb * in.material_color;
        } else {
            albedo = baked_color(in.color.z) * in.material_color;
        }
    }
    let v = safe_normalize(camera.position.xyz - in.world_position);
    var color = 0.03 * albedo * in.style.z;
    if camera.environment.y > 0.0 {
        let metallic = in.style.y;
        let nv = max(dot(n, v), 0.0001);
        let f0 = mix(vec3<f32>(0.04), albedo, metallic);
        let f = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - nv, 5.0);
        let irradiance = textureSampleLevel(irradiance_map, env_sampler, n, 0.0).rgb;
        let diffuse = (1.0 - f) * (1.0 - metallic) * irradiance * albedo;
        let prefiltered = textureSampleLevel(prefiltered_map, env_sampler, reflect(-v, n), roughness * 7.0).rgb;
        let brdf = textureSampleLevel(brdf_lut, env_sampler, vec2<f32>(nv, roughness), 0.0).rg;
        color = (diffuse + prefiltered * (f * brdf.x + brdf.y)) * in.style.z * camera.environment.x;
    }
    for (var i = 0u; i < 4u; i++) {
        let light = camera.lights[i];
        if light.rgb_intensity.a > 0.0 {
            let offset = light.position.xyz - in.world_position;
            let dist_sq = max(dot(offset, offset), 0.0001);
            let radiance = light.rgb_intensity.rgb * light.rgb_intensity.a / dist_sq;
            color += standard_light(n, v, safe_normalize(offset), albedo, roughness, in.style.y, radiance);
        }
    }
    let sun = camera.sun;
    if sun.rgb_intensity.a > 0.0 {
        let l = safe_normalize(-sun.direction.xyz);
        let radiance = sun.rgb_intensity.rgb * sun.rgb_intensity.a * shadow_factor(1.0, in.world_position, n, l);
        color += standard_light(n, v, l, albedo, roughness, in.style.y, radiance);
    }
    let spot = camera.spot;
    if spot.rgb_intensity.a > 0.0 {
        let offset = spot.position_range.xyz - in.world_position;
        let dist = length(offset);
        let l = safe_normalize(offset);
        let cos_angle = dot(-l, safe_normalize(spot.direction_cos_inner.xyz));
        let cone = clamp((cos_angle - spot.cos_outer.x)
                         / max(spot.direction_cos_inner.w - spot.cos_outer.x, 0.0001), 0.0, 1.0);
        let range_fade = clamp(1.0 - pow(dist / spot.position_range.w, 4.0), 0.0, 1.0);
        let attenuation = cone * cone * range_fade * range_fade / (dist * dist + 0.0001);
        let radiance = spot.rgb_intensity.rgb * spot.rgb_intensity.a * attenuation
                     * shadow_factor(2.0, in.world_position, n, l);
        color += standard_light(n, v, l, albedo, roughness, in.style.y, radiance);
    }
    if in.ripple > 0.0 {
        let tau = 6.28318530718;
        let p = in.world_position.xz + camera.animation.yz;
        let t = camera.animation.x;
        let bend = 1.5 * sin(tau * ((96.0 * p.x - 24.0 * p.y) / 256.0 + 9.0 * t / 64.0));
        let wave = sin(tau * ((40.0 * p.x + 160.0 * p.y) / 256.0 - 16.0 * t / 64.0) + bend);
        let gap = sin(tau * ((112.0 * p.x + 8.0 * p.y) / 256.0 + 5.0 * t / 64.0));
        let band = smoothstep(0.86, 0.93, wave) * smoothstep(-0.35, 0.05, gap);
        color += in.ripple * band * (0.25 * color + vec3<f32>(0.12, 0.16, 0.18));
    }
    color = color / (color + vec3<f32>(1.0));
    if camera.fog_color.a > 0.0 {
        let distance_from_camera = distance(camera.position.xyz, in.world_position);
        let amount = clamp((distance_from_camera - camera.fog_range.x) /
            (camera.fog_range.y - camera.fog_range.x), 0.0, 1.0);
        color = mix(color, camera.fog_color.rgb, amount);
    }
    return vec4<f32>(color, in.effects.x);
}
"#,
    include_str!("surface.wgsl")
);

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

#[cfg(test)]
mod post_target_tests {
    use super::*;
    use crate::scene::{resolve_source, spawn_instances};
    use manifoldx_core::ecs::ElementType;

    #[test]
    fn post_uses_the_active_srgb_target_format() {
        let Ok(rgba_gpu) = Gpu::new(true) else {
            eprintln!("skip sRGB target test: no native adapter");
            return;
        };
        let bgra_gpu = Gpu::new(true).expect("same adapter selection");
        let mut world = World::new(2).unwrap();
        world.register("Transform", ElementType::F32, 10).unwrap();
        world.register("Mesh", ElementType::U32, 1).unwrap();
        world.register("Material", ElementType::U32, 1).unwrap();
        spawn_instances(
            &mut world,
            &[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]],
            &[1],
            &[1],
        )
        .unwrap();
        let source = resolve_source(&world, "Transform", "Mesh", Some("Material"), 2).unwrap();
        let camera = Camera::new([0.0, 0.0, 3.0], [0.0; 3]);
        let mesh = MeshData {
            positions: vec![[-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 2],
        };
        let capture = |gpu, format| {
            let mut renderer = NativeRenderer::with_gpu(gpu, format, 2);
            renderer.upload_mesh(1, &mesh, [1.0; 3]).unwrap();
            renderer
                .set_material(1, [0.85, 0.2, 0.1], Shading::Flat)
                .unwrap();
            renderer.set_post(Some(std::sync::Arc::new(PostOptions {
                ssao: None,
                bloom: None,
                shafts: None,
                grade: Some([0.75, 1.0, 0.0]),
                sky: None,
                fxaa: true,
                kernel: [[0.0; 4]; 32],
            })));
            let first = renderer
                .render(&world, source, &camera, 64, 48, false)
                .unwrap();
            let second = renderer
                .render(&world, source, &camera, 72, 56, false)
                .unwrap();
            assert_eq!(
                second.stats.post_target_generation,
                first.stats.post_target_generation + 1
            );
            assert!(second.stats.post_order.contains(&"fxaa"));
            let center = (28 * 72 + 36) * 4;
            let pixel = &second.rgba[center..center + 4];
            assert!(pixel[0] > pixel[2] + 70, "expected a red center: {pixel:?}");
            second.rgba
        };
        let rgba = capture(rgba_gpu, wgpu::TextureFormat::Rgba8UnormSrgb);
        let bgra = capture(bgra_gpu, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(rgba.iter().zip(bgra).all(|(a, b)| a.abs_diff(b) <= 1));
    }
}
