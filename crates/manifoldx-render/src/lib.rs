//! The first production native renderer slice.
//!
//! This crate owns one wgpu device/queue pair and a fixed offscreen opaque
//! pipeline. It uploads indexed geometry once, reads per-instance transforms
//! from the existing native [`manifoldx_core::ecs::World`] through its public
//! owned-read API, writes a depth buffer and returns an owned RGBA readback.
//!
//! It is deliberately narrow: no window/surface, textures, PBR, shadows, GUI,
//! streaming, skinning, arbitrary shaders or material graph. The window and
//! event loop are the next slice. The Python renderer and public API are not
//! touched.

pub mod camera;
pub mod math;
pub mod renderer;
pub mod scene;

pub use camera::Camera;
pub use renderer::{
    json_string, srgb_to_linear, write_ppm, AdapterSummary, Gpu, NativeRenderer, RenderError,
    RenderOutput, RenderStats, RendererOptions, COLOR_FORMAT, DEPTH_FORMAT, MAX_SIDE,
};
pub use scene::{
    read_instances, resolve_source, spawn_instances, used_geometry_ids, Instance, InstanceBatch,
    InstanceSource, MeshData, SceneError, DEFAULT_MAX_INSTANCES,
};

/// The name this slice reports in evidence.
pub const SLICE_NAME: &str = "manifoldx-native-render/base-offscreen";
