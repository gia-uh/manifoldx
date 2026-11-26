//! Python scene declarations for the native graphics owner.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use manifoldx_core::ecs::{ColumnData, EntityId, World};
use manifoldx_render::rect::RectRow;
use manifoldx_render::shadow::ShadowOptions;
use manifoldx_render::skin::{validate_matrices, JointMatrices};
use manifoldx_render::{
    resolve_source, Camera, DirectionalLight, MeshData, NativeRenderer, PointLight, PostOptions,
    RenderError, RenderStats, RendererOptions, Shading, Sky, SpotLight, StandardEffects,
    DEFAULT_MAX_INSTANCES,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::PyNativeWorld;

type CameraInput = ([f32; 3], [f32; 3], [f32; 3], f32, f32, f32);
type LightInput = ([f32; 3], [f32; 3], f32);
type SpotInput = ([f32; 3], [f32; 3], [f32; 3], f32, f32, f32, f32);
type MaterialInput = (f32, f32, f32, Option<u32>);
type VertexInput = (Vec<[f32; 3]>, Vec<u32>);
type PoseIdentity = (u32, Option<u64>);
type PoseInput = (PoseIdentity, JointMatrices);
type FrameInput = (PoseIdentity, u32, f64);
type EffectsInput = (bool, f32, Option<f32>, Option<[f32; 4]>);
type ShadowInput = (
    [f32; 3],
    f32,
    u32,
    f32,
    f32,
    f32,
    u32,
    bool,
    u32,
    Option<f32>,
);
type SkyInput = ([f32; 3], Option<[f32; 3]>, f32, f32, f32);
type PostInput = (
    Option<[f32; 5]>,
    Option<[f32; 3]>,
    Option<[f32; 4]>,
    Option<[f32; 3]>,
    Option<SkyInput>,
    bool,
    Vec<[f32; 4]>,
);

enum Upload {
    Mesh(u32, MeshData, Vec<[f32; 2]>, VertexInput),
    Material(u32, [f32; 3], Shading, StandardEffects),
    Albedo(u32, u32, u32, Vec<Vec<u8>>),
    Poses(Vec<PoseInput>),
    Bake(u32, Vec<JointMatrices>),
    Play(Vec<FrameInput>),
}

pub(crate) struct Scene {
    camera: Camera,
    background: [f32; 3],
    lights: Vec<PointLight>,
    sun: Option<DirectionalLight>,
    spot: Option<SpotLight>,
    post: Option<Arc<PostOptions>>,
    shadows: Option<ShadowOptions>,
    rects: Arc<[RectRow]>,
    rect_before_post: bool,
    clips: HashMap<u32, usize>,
    uploads: VecDeque<Upload>,
    offscreen: Option<NativeRenderer>,
}

impl Scene {
    fn new() -> Self {
        Self {
            camera: Camera::new([0.0, 1.0, 2.0], [0.0; 3]),
            background: [0.1, 0.1, 0.2],
            lights: Vec::new(),
            sun: None,
            spot: None,
            post: None,
            shadows: None,
            rects: Arc::default(),
            rect_before_post: false,
            clips: HashMap::new(),
            uploads: VecDeque::new(),
            offscreen: None,
        }
    }

    pub(crate) fn apply(
        &mut self,
        renderer: &mut NativeRenderer,
        world: &World,
    ) -> Result<Camera, RenderError> {
        renderer.set_background(self.background);
        renderer.set_point_lights(&self.lights);
        renderer.set_direct_lights(self.sun, self.spot);
        renderer.set_post(self.post.clone());
        renderer.set_shadows(self.shadows);
        renderer.set_rects(self.rects.clone(), self.rect_before_post);
        while let Some(upload) = self.uploads.pop_front() {
            match upload {
                Upload::Mesh(id, mesh, uv, (colors, joints)) => {
                    renderer.upload_mesh_attributes(id, &mesh, [1.0; 3], &uv, &colors, &joints)?
                }
                Upload::Poses(poses) => {
                    let entities = poses
                        .iter()
                        .map(|(identity, _)| skin_entity(world, renderer, *identity))
                        .collect::<Result<Vec<_>, _>>()?;
                    for (entity, (_, matrices)) in entities.into_iter().zip(poses) {
                        renderer.set_joints(entity, &matrices)?;
                    }
                }
                Upload::Bake(clip, poses) => renderer.bake_poses(clip, &poses)?,
                Upload::Play(frames) => {
                    let entities = frames
                        .iter()
                        .map(|(identity, _, _)| skin_entity(world, renderer, *identity))
                        .collect::<Result<Vec<_>, _>>()?;
                    for (entity, (_, clip, position)) in entities.into_iter().zip(frames) {
                        renderer.play_poses(entity, clip, position)?;
                    }
                }
                Upload::Material(id, color, shading, effects) => {
                    renderer.set_material_effects(id, color, shading, effects)?
                }
                Upload::Albedo(id, width, height, levels) => renderer.upload_albedo(
                    id,
                    width,
                    height,
                    &levels.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                )?,
            }
        }
        Ok(self.camera)
    }

    fn capture(
        &mut self,
        world: &World,
        width: u32,
        height: u32,
        labels: Option<&HashMap<u32, u32>>,
    ) -> Result<manifoldx_render::RenderOutput, RenderError> {
        let mut renderer = match self.offscreen.take() {
            Some(renderer) => renderer,
            None => NativeRenderer::new(RendererOptions {
                allow_software: true,
                ..Default::default()
            })?,
        };
        let result = (|| {
            let camera = self.apply(&mut renderer, world)?;
            let source = source(world)?;
            match labels {
                Some(labels) => renderer.render_ids(world, source, &camera, width, height, labels),
                None => renderer.render(world, source, &camera, width, height, false),
            }
        })();
        self.offscreen = Some(renderer);
        result
    }
}

fn skin_entity(
    world: &World,
    renderer: &NativeRenderer,
    (slot, generation): PoseIdentity,
) -> Result<EntityId, RenderError> {
    let invalid =
        || RenderError::Resource(format!("skinning needs a live skinned mesh at slot {slot}"));
    let entity = world
        .live_identity(slot)
        .filter(|id| generation.is_none_or(|version| version == id.generation()))
        .ok_or_else(invalid)?;
    let mesh = world.component_index("Mesh").ok_or_else(invalid)?;
    let rows = world
        .read_component(mesh, Some(&[entity]))
        .map_err(|error| RenderError::Resource(error.to_string()))?;
    match rows.data {
        ColumnData::U32(ids) if ids.first().is_some_and(|id| renderer.mesh_is_skinned(*id)) => {
            Ok(entity)
        }
        _ => Err(invalid()),
    }
}

pub(crate) fn source(world: &World) -> Result<manifoldx_render::InstanceSource, RenderError> {
    Ok(resolve_source(
        world,
        "Transform",
        "Mesh",
        Some("Material"),
        (world.capacity() as usize).min(DEFAULT_MAX_INSTANCES),
    )?)
}

#[pyclass(frozen, module = "manifoldx._core_native", name = "_NativeGraphics")]
pub(crate) struct NativeGraphics {
    pub(crate) scene: Arc<Mutex<Scene>>,
}

#[pymethods]
impl NativeGraphics {
    #[new]
    fn new() -> Self {
        Self {
            scene: Arc::new(Mutex::new(Scene::new())),
        }
    }

    fn camera(&self, args: CameraInput) -> PyResult<()> {
        let (position, target, up, fov_y_degrees, near, far) = args;
        if position
            .iter()
            .chain(&target)
            .chain(&up)
            .any(|value| !value.is_finite())
            || !(0.0..180.0).contains(&fov_y_degrees)
            || !near.is_finite()
            || !far.is_finite()
            || near <= 0.0
            || near >= far
        {
            return Err(PyValueError::new_err("invalid perspective camera"));
        }
        self.scene.lock().unwrap().camera = Camera {
            position,
            target,
            up,
            fov_y_degrees,
            near,
            far,
        };
        Ok(())
    }

    #[pyo3(signature = (background, lights, sun=None, spot=None))]
    fn settings(
        &self,
        background: [f32; 3],
        lights: Vec<LightInput>,
        sun: Option<LightInput>,
        spot: Option<SpotInput>,
    ) {
        let mut scene = self.scene.lock().unwrap();
        scene.sun = sun.map(|(direction, color, intensity)| DirectionalLight {
            direction,
            color,
            intensity,
        });
        scene.spot = spot.map(
            |(position, direction, color, intensity, range, cos_inner, cos_outer)| SpotLight {
                position,
                direction,
                color,
                intensity,
                range,
                cos_inner,
                cos_outer,
            },
        );
        scene.background = background;
        scene.lights = lights
            .into_iter()
            .take(4)
            .map(|(position, color, intensity)| PointLight {
                position,
                color,
                intensity,
            })
            .collect();
    }

    fn fixture_rects(&self, rows: Vec<RectRow>, before_post: bool) -> PyResult<()> {
        if rows.len() > 4096
            || rows
                .iter()
                .flatten()
                .any(|component| !component.is_finite())
        {
            return Err(PyValueError::new_err("invalid Rect instance rows"));
        }
        let mut scene = self.scene.lock().unwrap();
        scene.rects = rows.into();
        scene.rect_before_post = before_post;
        Ok(())
    }

    #[pyo3(signature = (options))]
    fn shadows(&self, options: Option<ShadowInput>) -> PyResult<()> {
        let options = options.map(
            |(
                target,
                extent,
                resolution,
                near,
                far,
                bias,
                pcf_radius,
                auto_fit,
                cascades,
                distance,
            )| ShadowOptions {
                target,
                extent,
                resolution,
                near,
                far,
                bias,
                pcf_radius,
                auto_fit,
                cascades,
                distance,
            },
        );
        if let Some(options) = options {
            options
                .validate()
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        self.scene.lock().unwrap().shadows = options;
        Ok(())
    }

    #[pyo3(signature = (settings))]
    fn post(&self, settings: Option<PostInput>) -> PyResult<()> {
        let post = if let Some((ssao, bloom, shafts, grade, sky, fxaa, kernel)) = settings {
            let kernel = kernel
                .try_into()
                .map_err(|_| PyValueError::new_err("post kernel must have 32 rows"))?;
            let sky = sky.map(
                |(zenith, horizon, sun_size_degrees, sun_intensity, glow)| Sky {
                    zenith,
                    horizon,
                    sun_size_degrees,
                    sun_intensity,
                    glow,
                },
            );
            Some(Arc::new(PostOptions {
                ssao,
                bloom,
                shafts,
                grade,
                sky,
                fxaa,
                kernel,
            }))
        } else {
            None
        };
        self.scene.lock().unwrap().post = post;
        Ok(())
    }

    #[pyo3(signature = (id, positions, normals, indices, uvs, attributes=None))]
    fn mesh(
        &self,
        id: u32,
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        indices: Vec<u32>,
        uvs: Vec<[f32; 2]>,
        attributes: Option<VertexInput>,
    ) -> PyResult<()> {
        let mesh = MeshData {
            positions,
            normals,
            indices,
        };
        mesh.validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.scene.lock().unwrap().uploads.push_back(Upload::Mesh(
            id,
            mesh,
            uvs,
            attributes.unwrap_or_default(),
        ));
        Ok(())
    }

    fn joint_poses(&self, poses: Vec<PoseInput>) -> PyResult<()> {
        for (_, matrices) in &poses {
            validate_matrices(matrices)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Poses(poses));
        Ok(())
    }

    #[pyo3(signature = (poses, into=None))]
    fn bake_poses(&self, poses: Vec<JointMatrices>, into: Option<u32>) -> PyResult<u32> {
        if poses.is_empty() {
            return Err(PyValueError::new_err(
                "a skin clip must have at least one frame",
            ));
        }
        for matrices in &poses {
            validate_matrices(matrices)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        let mut scene = self.scene.lock().unwrap();
        let id = if let Some(id) = into {
            if scene.clips.get(&id) != Some(&poses.len()) {
                return Err(PyValueError::new_err(
                    "a replacement clip must keep its frame count",
                ));
            }
            id
        } else {
            let id = u32::try_from(scene.clips.len() + 1)
                .map_err(|_| PyValueError::new_err("skin clip identifiers exhausted"))?;
            scene.clips.insert(id, poses.len());
            id
        };
        scene.uploads.push_back(Upload::Bake(id, poses));
        Ok(id)
    }

    fn pose_frames(&self, frames: Vec<FrameInput>) -> PyResult<()> {
        let mut scene = self.scene.lock().unwrap();
        for (_, clip, position) in &frames {
            let count = scene
                .clips
                .get(clip)
                .ok_or_else(|| PyValueError::new_err("unknown skin clip"))?;
            if !(0.0..=(count - 1) as f64).contains(position) {
                return Err(PyValueError::new_err(
                    "position is outside the skin clip's frames",
                ));
            }
        }
        scene.uploads.push_back(Upload::Play(frames));
        Ok(())
    }

    #[pyo3(signature = (id, color, kind, standard=None, shininess=None, effects=None))]
    fn material(
        &self,
        id: u32,
        color: [f32; 3],
        kind: &str,
        standard: Option<MaterialInput>,
        shininess: Option<f32>,
        effects: Option<EffectsInput>,
    ) -> PyResult<()> {
        let shading = match (kind, standard, shininess) {
            ("basic", None, None) => Shading::Basic,
            ("flat", None, None) => Shading::Flat,
            ("phong", None, Some(shininess)) if shininess.is_finite() && shininess > 0.0 => {
                Shading::Phong { shininess }
            }
            ("phong", None, _) => {
                return Err(PyValueError::new_err(
                    "Phong shininess must be finite and positive",
                ));
            }
            ("standard", Some((roughness, metallic, ao, albedo)), None) => Shading::Standard {
                roughness,
                metallic,
                ao,
                albedo,
            },
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported base material {kind}"
                )))
            }
        };
        let effects = effects.map_or_else(
            StandardEffects::default,
            |(vertex_colors, opacity, mirror, cut)| StandardEffects {
                vertex_colors,
                opacity,
                mirror,
                cut,
            },
        );
        effects
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Material(id, color, shading, effects));
        Ok(())
    }

    fn albedo(&self, py: Python<'_>, id: u32, width: u32, height: u32, levels: Vec<Py<PyBytes>>) {
        let levels = levels
            .iter()
            .map(|level| level.as_bytes(py).to_vec())
            .collect();
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Albedo(id, width, height, levels));
    }

    #[pyo3(signature = (world, width, height, labels=None))]
    fn capture<'py>(
        &self,
        py: Python<'py>,
        world: &PyNativeWorld,
        width: u32,
        height: u32,
        labels: Option<HashMap<u32, u32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let output = py.detach(|| {
            let world = world.lock()?;
            self.scene
                .lock()
                .unwrap()
                .capture(&world, width, height, labels.as_ref())
                .map_err(render_error)
        })?;
        let result = stats_dict(py, &output.stats)?;
        result.set_item("rgba", PyBytes::new(py, &output.rgba))?;
        let scene = self.scene.lock().unwrap();
        let renderer = scene
            .offscreen
            .as_ref()
            .expect("completed offscreen capture");
        result.set_item("adapter", renderer.adapter().to_json())?;
        result.set_item("target", "rgba8unorm-srgb")?;
        Ok(result)
    }
}

pub(crate) fn stats_dict<'py>(
    py: Python<'py>,
    stats: &RenderStats,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("instances", stats.instances)?;
    result.set_item("draw_calls", stats.draw_calls)?;
    result.set_item("triangles", stats.triangles)?;
    result.set_item("post_order", &stats.post_order)?;
    result.set_item("shadow_layers", stats.shadow_layers)?;
    result.set_item("post_target_generation", stats.post_target_generation)?;
    result.set_item("read_ns", stats.read_ns)?;
    result.set_item("upload_ns", stats.upload_ns)?;
    result.set_item("encode_submit_ns", stats.encode_submit_ns)?;
    result.set_item("readback_wait_ns", stats.wait_readback_ns)?;
    result.set_item("total_ns", stats.total_ns)?;
    Ok(result)
}

pub(crate) fn render_error(error: RenderError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeGraphics>()
}
