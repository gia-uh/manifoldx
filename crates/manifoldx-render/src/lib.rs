//! Native rendering on desktop surfaces and offscreen targets.

pub mod camera;
pub mod desktop;
pub mod math;
pub mod post;
pub mod rect;
pub mod renderer;
pub mod resident;
pub mod scene;
pub mod shadow;
pub mod skin;

pub use camera::Camera;
pub use post::{PostOptions, Sky};
pub use renderer::{
    json_string, srgb_to_linear, write_ppm, AdapterSummary, DirectionalLight, FrameTarget, Gpu,
    NativeRenderer, PointLight, RenderError, RenderOutput, RenderStats, RendererOptions, Shading,
    SpotLight, StandardEffects, COLOR_FORMAT, DEPTH_FORMAT, MAX_SIDE,
};
pub use scene::{
    read_instances, resolve_source, spawn_instances, used_geometry_ids, Instance, InstanceBatch,
    InstanceSource, MeshData, SceneError, DEFAULT_MAX_INSTANCES,
};
pub use wgpu;
pub use winit;

/// The name this slice reports in evidence.
pub const SLICE_NAME: &str = "manifoldx-native-render/base-offscreen";
