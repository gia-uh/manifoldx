//! Python scene declarations for the native graphics owner.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use manifoldx_core::ecs::{ColumnData, EntityId, World};
use manifoldx_render::desktop::PresentedCapture;
use manifoldx_render::environment::EnvironmentData;
use manifoldx_render::fragment::FragmentData;
use manifoldx_render::frame_compute::{ComputeBinding, ComputeProgram, FrameCompute};
use manifoldx_render::lines::{LineLayer, LineOptions, LineRow};
use manifoldx_render::manual::ManualData;
use manifoldx_render::rect::RectRow;
use manifoldx_render::renderer::{FixtureRenderFault, ViewOptions, ViewOrder, ViewRead};
use manifoldx_render::shadow::ShadowOptions;
use manifoldx_render::skin::{validate_matrices, JointMatrices};
use manifoldx_render::sprite::SpriteData;
use manifoldx_render::volume::VolumeMaterialData;
use manifoldx_render::{
    resolve_source, Camera, DirectionalLight, MeshData, NativeRenderer, PointLight, PostOptions,
    RenderError, RenderStats, RendererOptions, Shading, Sky, SpotLight, StandardEffects,
    DEFAULT_MAX_INSTANCES,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::gui::{Declaration, GuiState, Hit};
use crate::stream_cells::{CellBatch, CellCommit, CellDraw, CellKey, StreamCells};
use crate::vehicles::{tile_inputs, SampleOutput, TileInput};
use crate::PyNativeWorld;

type PresentedPixels = (u32, u32, bool, Vec<u8>);
type ComputeColumn = (Option<usize>, String, u32, bool);
type ComputeInput = (
    u32,
    String,
    u32,
    Vec<(u8, f32)>,
    (u8, u32),
    Vec<ComputeColumn>,
);

type CameraInput = ([f32; 3], [f32; 3], [f32; 3], f32, f32, f32);
type LightInput = ([f32; 3], [f32; 3], f32);
type SpotInput = ([f32; 3], [f32; 3], [f32; 3], f32, f32, f32, f32);
type MaterialInput = (f32, f32, f32, Option<u32>);
type VertexInput = (Vec<[f32; 3]>, Vec<u32>, Vec<f32>);
type PoseIdentity = (u32, Option<u64>);
type PoseInput = (PoseIdentity, JointMatrices);
type FrameInput = (PoseIdentity, u32, f64);
type EffectsInput = (
    bool,
    f32,
    Option<f32>,
    Option<[f32; 4]>,
    f32,
    f32,
    f32,
    bool,
    bool,
    bool,
    Option<u32>,
);
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
type LineInput = (String, [f32; 2], Option<Vec<LineRow>>);
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
    StreamBegin(u32, [[f32; 3]; 2], u64, u64),
    StreamWrite(u32, bool, Vec<u8>),
    StreamCommit(CellCommit, bool),
    StreamDiscardResults(Vec<u64>),
    StreamHide(CellKey, u64),
    StreamFade(CellKey, usize, f32),
    StreamRetire(u32),
    Terrain(f64),
    RetireScene(Vec<u32>, Vec<u32>),
    Mesh(u32, MeshData, Vec<[f32; 2]>, VertexInput),
    Material(u32, [f32; 3], Shading, StandardEffects, bool),
    Surface(u32, Box<[[f32; 5]; 9]>, bool),
    Albedo(u32, u32, u32, Vec<Vec<u8>>),
    TexturePatch(u32, [u32; 3], [u32; 2], Vec<u8>),
    TextureRelease(u32),
    BakedArray(u32, u32, u32),
    BakedLayer(u32, u32, Vec<Vec<u8>>),
    Poses(Vec<PoseInput>),
    Bake(u32, Vec<JointMatrices>),
    Play(Vec<FrameInput>),
    View(ViewOptions, bool),
    ViewUpdate(u32, Camera, Option<BTreeSet<i32>>, bool, bool),
    ViewDrop(u32),
    ViewRelease(u32),
    ViewRead(u32, ViewRead),
    Environment(EnvironmentData),
    Fragment(u32, FragmentData),
    Manual(u32, ManualData, bool),
    Sprite(u32, SpriteData),
    Label(u32, SpriteData),
    LabelTile(u32, Vec<u8>),
    Volume(u32, [u32; 3], Vec<u8>),
    VolumeMaterial(u32, VolumeMaterialData),
    Axis(u32, [[f32; 3]; 2]),
}

pub(crate) struct Scene {
    camera: Camera,
    background: [f32; 3],
    lights: Vec<PointLight>,
    sun: Option<DirectionalLight>,
    spot: Option<SpotLight>,
    post: Option<Arc<PostOptions>>,
    environment: Option<(f32, bool)>,
    fog: Option<([f32; 2], [f32; 3])>,
    shadows: Option<ShadowOptions>,
    rects: Arc<[RectRow]>,
    gui: GuiState,
    lines: Option<Arc<LineOptions>>,
    rect_before_post: bool,
    layers: Option<Arc<BTreeSet<i32>>>,
    time: f64,
    origin: [f64; 2],
    benders: Vec<[f64; 4]>,
    clips: HashMap<u32, usize>,
    uploads: VecDeque<Upload>,
    offscreen: Option<NativeRenderer>,
    view_materials: BTreeSet<u32>,
    view_material_revision: Option<(u64, u64)>,
    unused_view_materials: Vec<u32>,
    view_reads: HashMap<u64, ViewRead>,
    next_read: u64,
    stream_cells: StreamCells,
    stream_results: HashMap<u64, Result<(), String>>,
    fixture_fault: Option<FixtureRenderFault>,
    compute: Option<FrameCompute>,
    compute_pending: VecDeque<ComputeProgram>,
    compute_callbacks: Vec<(u32, Py<PyAny>)>,
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
            environment: None,
            fog: None,
            shadows: None,
            rects: Arc::default(),
            gui: GuiState::default(),
            lines: None,
            rect_before_post: false,
            layers: None,
            time: 0.0,
            origin: [0.0; 2],
            benders: Vec::new(),
            clips: HashMap::new(),
            uploads: VecDeque::new(),
            offscreen: None,
            view_materials: BTreeSet::new(),
            view_material_revision: None,
            unused_view_materials: Vec::new(),
            view_reads: HashMap::new(),
            next_read: 0,
            stream_cells: StreamCells::default(),
            stream_results: HashMap::new(),
            fixture_fault: None,
            compute: None,
            compute_pending: VecDeque::new(),
            compute_callbacks: Vec::new(),
        }
    }

    pub(crate) fn apply(
        &mut self,
        renderer: &mut NativeRenderer,
        world: &mut World,
    ) -> Result<Camera, RenderError> {
        renderer.set_background(self.background);
        renderer.set_point_lights(&self.lights);
        renderer.set_direct_lights(self.sun, self.spot);
        renderer.set_post(self.post.clone());
        renderer.set_shadows(self.shadows);
        renderer.set_environment(self.environment);
        renderer.set_fog(self.fog);
        renderer.set_rects(self.rects.clone(), self.rect_before_post);
        renderer.set_gui(self.gui.frame.clone(), self.gui.before_post);
        renderer.set_layers(self.layers.clone());
        renderer.set_animation(self.time, self.origin, &self.benders);
        renderer.set_lines(self.lines.clone());
        while let Some(upload) = self.uploads.pop_front() {
            match upload {
                Upload::StreamBegin(id, bounds, vertices, indices) => {
                    renderer.begin_stream_mesh(id, bounds, vertices, indices)?
                }
                Upload::StreamWrite(id, vertex, bytes) => {
                    renderer.write_stream_mesh(id, vertex, &bytes)?
                }
                Upload::StreamCommit(commit, receipt) => {
                    let version = commit.version;
                    let result = self
                        .stream_cells
                        .commit(world, renderer, commit, self.origin);
                    if receipt {
                        self.stream_results
                            .insert(version, result.map_err(|error| error.to_string()));
                    } else {
                        result?;
                    }
                }
                Upload::StreamDiscardResults(versions) => {
                    for version in versions {
                        self.stream_results.remove(&version);
                    }
                }
                Upload::StreamHide(key, version) => self.stream_cells.hide(world, key, version)?,
                Upload::StreamFade(key, part, value) => {
                    self.stream_cells.fade(world, key, part, value)?
                }
                Upload::StreamRetire(id) => renderer.retire_stream_mesh(id),
                Upload::Terrain(size) => self.stream_cells.configure(size)?,
                Upload::RetireScene(meshes, materials) => {
                    renderer.retire_scene(&meshes, &materials);
                    for id in materials {
                        self.view_materials.remove(&id);
                    }
                }
                Upload::Axis(id, points) => renderer.upload_axis(id, points)?,
                Upload::Volume(id, dims, bytes) => renderer.upload_volume(id, dims, &bytes)?,
                Upload::VolumeMaterial(id, data) => renderer.upload_volume_material(id, data)?,
                Upload::Label(id, data) => renderer.upload_label(id, data)?,
                Upload::LabelTile(layer, rgba) => renderer.upload_label_tile(layer, &rgba)?,
                Upload::Sprite(id, data) => renderer.upload_sprite(id, data)?,
                Upload::Manual(id, data, samples_view) => {
                    renderer.upload_manual(id, &data)?;
                    self.track_view_material(id, samples_view);
                }
                Upload::Fragment(id, data) => renderer.upload_fragment(id, &data)?,
                Upload::Environment(data) => renderer.upload_environment(&data)?,
                Upload::View(options, live) => renderer.create_view(options, live)?,
                Upload::ViewUpdate(id, camera, layers, live, refresh) => {
                    renderer.update_view(id, camera, layers, live, refresh)?
                }
                Upload::ViewDrop(id) => renderer.drop_view(id),
                Upload::ViewRelease(id) => renderer.release_view(id),
                Upload::ViewRead(id, result) => renderer.read_view(id, result),
                Upload::Mesh(id, mesh, uv, (colors, joints, surfaces)) => renderer
                    .upload_mesh_surfaces(
                        id,
                        &mesh,
                        [1.0; 3],
                        &uv,
                        (&colors, &surfaces),
                        &joints,
                    )?,
                Upload::Surface(id, patterns, detail) => {
                    renderer.set_surface_patterns(id, *patterns, detail)?
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
                Upload::Material(id, color, shading, effects, samples_view) => {
                    renderer.set_material_effects(id, color, shading, effects)?;
                    self.track_view_material(id, samples_view);
                }
                Upload::BakedArray(id, size, layers) => {
                    renderer.create_baked_array(id, size, layers)?
                }
                Upload::BakedLayer(id, layer, levels) => {
                    renderer.write_baked_layer(id, layer, &levels)?
                }
                Upload::TexturePatch(id, origin, size, bytes) => {
                    renderer.write_texture_patch(id, origin, size, &bytes)?
                }
                Upload::TextureRelease(id) => renderer.release_texture(id),
                Upload::Albedo(id, width, height, levels) => renderer.upload_albedo(
                    id,
                    width,
                    height,
                    &levels.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                )?,
            }
        }
        self.find_unused_view_materials(world)?;
        Ok(self.camera)
    }

    fn track_view_material(&mut self, id: u32, samples_view: bool) {
        if samples_view {
            if self.view_materials.insert(id) {
                self.view_material_revision = None;
            }
        } else {
            self.view_materials.remove(&id);
        }
    }

    // Both ordinary ECS edits and streamed commits are visible at this point.
    fn find_unused_view_materials(&mut self, world: &World) -> Result<(), RenderError> {
        if self.view_materials.is_empty() {
            return Ok(());
        }
        let column = world
            .component_index("Material")
            .ok_or_else(|| RenderError::Resource("missing Material component".into()))?;
        let revision = (
            world.entity_version(),
            world
                .column_version(column)
                .map_err(RenderError::Resource)?,
        );
        if self.view_material_revision == Some(revision) {
            return Ok(());
        }
        let rows = world
            .read_component(column, None)
            .map_err(|error| RenderError::Resource(error.to_string()))?;
        let ColumnData::U32(ids) = rows.data else {
            return Err(RenderError::Resource(
                "Material must contain u32 ids".into(),
            ));
        };
        let mut unused = self.view_materials.clone();
        for (id, member) in ids.into_iter().zip(rows.membership) {
            if member {
                unused.remove(&id);
            }
        }
        for id in unused {
            self.view_materials.remove(&id);
            self.unused_view_materials.push(id);
        }
        self.view_material_revision = Some(revision);
        Ok(())
    }

    pub(crate) fn run_compute(
        &mut self,
        renderer: &NativeRenderer,
        world: &mut World,
        dt: f32,
        dispatch: &HashMap<u32, u32>,
    ) -> Result<(), RenderError> {
        if self.compute.is_none() && self.compute_pending.is_empty() {
            return Ok(());
        }
        let passes = self.compute.get_or_insert_with(FrameCompute::default);
        while let Some(program) = self.compute_pending.pop_front() {
            passes.register(renderer, world, program)?;
        }
        passes.run(renderer, world, dt, dispatch)
    }

    fn capture(
        &mut self,
        world: &mut World,
        width: u32,
        height: u32,
        labels: Option<&HashMap<u32, u32>>,
        dt: Option<f32>,
        dispatch: &HashMap<u32, u32>,
    ) -> Result<manifoldx_render::RenderOutput, RenderError> {
        let mut renderer = match self.offscreen.take() {
            Some(renderer) => renderer,
            None => NativeRenderer::new(RendererOptions {
                allow_software: true,
                ..Default::default()
            })?,
        };
        renderer.set_fixture_fault(self.fixture_fault.take());
        let result = (|| {
            if let Some(dt) = dt.filter(|_| labels.is_none()) {
                self.run_compute(&renderer, world, dt, dispatch)?;
            }
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

pub(crate) fn compute_overrides(scene: &Mutex<Scene>) -> Result<HashMap<u32, u32>, RenderError> {
    if scene.lock().unwrap().compute_callbacks.is_empty() {
        return Ok(HashMap::new());
    }
    Python::attach(|py| {
        let callbacks: Vec<_> = scene
            .lock()
            .unwrap()
            .compute_callbacks
            .iter()
            .map(|(key, callback)| (*key, callback.clone_ref(py)))
            .collect();
        callbacks
            .into_iter()
            .map(|(key, callback)| {
                let threads: i64 = callback.call0(py)?.extract(py)?;
                let threads = u32::try_from(threads.max(0)).map_err(|_| {
                    PyValueError::new_err("Compute callable dispatch exceeds u32 threads")
                })?;
                Ok((key, threads))
            })
            .collect::<PyResult<_>>()
    })
    .map_err(|error| RenderError::Resource(error.to_string()))
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

fn camera_from_input(args: CameraInput) -> PyResult<Camera> {
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
    Ok(Camera {
        position,
        target,
        up,
        fov_y_degrees,
        near,
        far,
    })
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
    presented: Arc<PresentedCapture>,
}

impl NativeGraphics {
    pub(crate) fn presented(&self) -> Arc<PresentedCapture> {
        Arc::clone(&self.presented)
    }
}

#[pymethods]
impl NativeGraphics {
    #[new]
    fn new() -> Self {
        Self {
            scene: Arc::new(Mutex::new(Scene::new())),
            presented: Arc::new(PresentedCapture::default()),
        }
    }

    fn close(&self, py: Python<'_>) {
        let old = std::mem::replace(&mut *self.scene.lock().unwrap(), Scene::new());
        py.detach(move || drop(old));
    }

    /// Ask the desktop loop to copy the next presented frame for `manifoldx look`.
    fn request_presented(&self) -> PyResult<()> {
        if !self.presented.supported() {
            return Err(PyRuntimeError::new_err(
                "the surface does not support copying the presented frame",
            ));
        }
        self.presented.request();
        Ok(())
    }

    /// (width, height, bgra, pixels) of the frame the last request copied, if it is ready.
    fn take_presented(&self) -> PyResult<Option<PresentedPixels>> {
        match self.presented.take() {
            None => Ok(None),
            Some(Ok(image)) => Ok(Some((image.width, image.height, image.bgra, image.pixels))),
            Some(Err(error)) => Err(PyRuntimeError::new_err(error)),
        }
    }

    fn camera(&self, args: CameraInput) -> PyResult<()> {
        self.scene.lock().unwrap().camera = camera_from_input(args)?;
        Ok(())
    }

    #[pyo3(signature = (layers))]
    fn layers(&self, layers: Option<Vec<i32>>) {
        self.scene.lock().unwrap().layers =
            layers.map(|layers| Arc::new(layers.into_iter().collect()));
    }

    fn fixture_render_fault(&self, name: &str) -> PyResult<()> {
        let fault = match name {
            "late-shadow" => FixtureRenderFault::LateShadow,
            "flat-post-depth" => FixtureRenderFault::FlatPostDepth,
            "stale-offscreen-target" => FixtureRenderFault::StaleOffscreenTarget,
            _ => return Err(PyValueError::new_err("unknown render fixture fault")),
        };
        self.scene.lock().unwrap().fixture_fault = Some(fault);
        Ok(())
    }

    fn fixture_view(
        &self,
        texture_id: u32,
        camera: CameraInput,
        layers: Vec<i32>,
        size: [u32; 2],
        clear: [f32; 3],
        mode: &str,
    ) -> PyResult<()> {
        let order = match mode {
            "normal" => ViewOrder::BeforeMain,
            "omit" => ViewOrder::Omit,
            "late" => ViewOrder::AfterMain,
            _ => return Err(PyValueError::new_err("unknown fixture view mode")),
        };
        let options = ViewOptions {
            texture_id,
            size,
            layers: Some(layers.into_iter().collect()),
            camera: camera_from_input(camera)?,
            clear: [clear[0], clear[1], clear[2], 1.0],
            fog: true,
            order,
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::View(options, true));
        Ok(())
    }

    #[pyo3(signature = (texture_id, camera, layers, size, clear, live, fog))]
    #[allow(clippy::too_many_arguments)]
    fn create_view(
        &self,
        texture_id: u32,
        camera: CameraInput,
        layers: Option<Vec<i32>>,
        size: [u32; 2],
        clear: [f32; 4],
        live: bool,
        fog: bool,
    ) -> PyResult<()> {
        let options = ViewOptions {
            texture_id,
            size,
            layers: layers.map(|layers| layers.into_iter().collect()),
            camera: camera_from_input(camera)?,
            clear,
            fog,
            order: ViewOrder::BeforeMain,
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::View(options, live));
        Ok(())
    }

    #[pyo3(signature = (id, camera, layers, live, refresh))]
    fn update_view(
        &self,
        id: u32,
        camera: CameraInput,
        layers: Option<Vec<i32>>,
        live: bool,
        refresh: bool,
    ) -> PyResult<()> {
        let command = Upload::ViewUpdate(
            id,
            camera_from_input(camera)?,
            layers.map(|layers| layers.into_iter().collect()),
            live,
            refresh,
        );
        self.scene.lock().unwrap().uploads.push_back(command);
        Ok(())
    }

    fn drop_view(&self, id: u32) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::ViewDrop(id));
    }

    fn release_view(&self, id: u32) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::ViewRelease(id));
    }

    fn read_view(&self, id: u32) -> u64 {
        let mut scene = self.scene.lock().unwrap();
        let ticket = scene.next_read;
        scene.next_read += 1;
        let result = ViewRead::default();
        scene.view_reads.insert(ticket, result.clone());
        scene.uploads.push_back(Upload::ViewRead(id, result));
        ticket
    }

    fn take_view_read<'py>(
        &self,
        py: Python<'py>,
        ticket: u64,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let mut scene = self.scene.lock().unwrap();
        let value = scene
            .view_reads
            .get(&ticket)
            .and_then(|result| result.lock().unwrap().take());
        match value {
            None => Ok(None),
            Some(value) => {
                scene.view_reads.remove(&ticket);
                value
                    .map(|bytes| Some(PyBytes::new(py, &bytes)))
                    .map_err(PyRuntimeError::new_err)
            }
        }
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

    fn animation(&self, time: f64, origin: [f64; 2], benders: Vec<[f64; 4]>) -> PyResult<()> {
        if !time.is_finite()
            || origin.iter().any(|x| !x.is_finite())
            || benders.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(PyValueError::new_err("invalid scene animation settings"));
        }
        let mut scene = self.scene.lock().unwrap();
        scene.time = time;
        scene.origin = origin;
        scene.benders = benders;
        Ok(())
    }

    fn lines(
        &self,
        visible: bool,
        bias: f32,
        feather: f32,
        layers: Vec<LineInput>,
    ) -> PyResult<()> {
        if !bias.is_finite() || !feather.is_finite() || bias < 0.0 || feather < 0.0 {
            return Err(PyValueError::new_err("invalid line settings"));
        }
        let mut scene = self.scene.lock().unwrap();
        let previous: HashMap<_, _> = scene
            .lines
            .iter()
            .flat_map(|lines| &lines.layers)
            .map(|layer| (&layer.key, &layer.rows))
            .collect();
        let mut next = Vec::with_capacity(layers.len());
        for (key, look, rows) in layers {
            if look.iter().any(|x| !x.is_finite()) {
                return Err(PyValueError::new_err("invalid line layer look"));
            }
            let rows = if let Some(rows) = rows {
                if rows.len() > 1_000_000 || rows.iter().flatten().any(|x| !x.is_finite()) {
                    return Err(PyValueError::new_err("invalid line segments"));
                }
                Arc::from(rows)
            } else {
                previous
                    .get(&key)
                    .map(|rows| Arc::clone(rows))
                    .ok_or_else(|| PyValueError::new_err("unknown line layer"))?
            };
            next.push(LineLayer { key, rows, look });
        }
        scene.lines = Some(Arc::new(LineOptions {
            visible,
            bias,
            feather,
            layers: next,
        }));
        Ok(())
    }

    fn gui_font(&self, key: String, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.scene.lock().unwrap().gui.font(key, data.as_bytes())
    }

    fn gui_tree(
        &self,
        rows: Vec<Declaration>,
        viewport: [f32; 2],
        scale: f32,
        exact: bool,
    ) -> PyResult<()> {
        self.scene
            .lock()
            .unwrap()
            .gui
            .update(rows, viewport, scale, exact)
    }

    fn gui_pointer(&self, pointer: u64, phase: &str, point: [f32; 2]) -> PyResult<(bool, Hit)> {
        self.scene
            .lock()
            .unwrap()
            .gui
            .pointer(pointer, phase, point)
    }

    fn gui_focus(&self) -> Option<u64> {
        self.scene.lock().unwrap().gui.focus()
    }

    fn gui_reset(&self) {
        self.scene.lock().unwrap().gui.reset();
    }

    fn gui_boxes(&self) -> Vec<([f32; 4], [f32; 4])> {
        self.scene.lock().unwrap().gui.boxes()
    }

    fn fixture_gui_omit_capture(&self, enabled: bool) {
        self.scene.lock().unwrap().gui.fixture_omit_capture(enabled);
    }

    fn fixture_gui_before_post(&self, enabled: bool) {
        self.scene.lock().unwrap().gui.before_post = enabled;
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

    fn retire_scene(&self, meshes: Vec<u32>, materials: Vec<u32>) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::RetireScene(meshes, materials));
    }

    fn take_unused_view_materials(&self) -> Vec<u32> {
        std::mem::take(&mut self.scene.lock().unwrap().unused_view_materials)
    }

    fn stream_begin(&self, id: u32, bounds: [[f32; 3]; 2], vertices: u64, indices: u64) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::StreamBegin(id, bounds, vertices, indices));
    }

    fn stream_write(&self, id: u32, vertex: bool, data: &Bound<'_, PyBytes>) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::StreamWrite(id, vertex, data.as_bytes().to_vec()));
    }

    #[allow(clippy::too_many_arguments)]
    fn stream_commit(
        &self,
        key: CellKey,
        version: u64,
        meshes: Vec<CellDraw>,
        pos: [f64; 3],
        rows: Vec<CellBatch>,
        tiles: Vec<TileInput>,
        receipt: bool,
    ) {
        let commit = CellCommit {
            key,
            version,
            meshes,
            origin: pos,
            rows,
            tiles: tile_inputs(tiles),
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::StreamCommit(commit, receipt));
    }

    fn stream_result(&self, version: u64) -> PyResult<Option<bool>> {
        self.scene
            .lock()
            .unwrap()
            .stream_results
            .remove(&version)
            .map(|result| result.map(|()| true).map_err(PyValueError::new_err))
            .transpose()
    }

    fn stream_discard_results(&self, versions: Vec<u64>) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::StreamDiscardResults(versions));
    }

    fn stream_hide(&self, key: CellKey, version: u64) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::StreamHide(key, version));
    }

    fn stream_fade(&self, key: CellKey, part: usize, value: f32) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::StreamFade(key, part, value));
    }

    fn stream_terrain(&self, size: f64) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Terrain(size));
    }

    fn stream_sample(&self, x: f64, z: f64, near: f64) -> Option<SampleOutput> {
        let sample = self.scene.lock().unwrap().stream_cells.sample(x, z, near)?;
        Some((
            sample.height,
            sample.normal,
            sample.grip,
            sample.level,
            sample.tile,
            sample.version,
        ))
    }

    fn stream_retire(&self, id: u32) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::StreamRetire(id));
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

    #[pyo3(signature = (id, color, kind, standard=None, shininess=None, effects=None, view_albedo=false))]
    #[allow(clippy::too_many_arguments)]
    fn material(
        &self,
        id: u32,
        color: [f32; 3],
        kind: &str,
        standard: Option<MaterialInput>,
        shininess: Option<f32>,
        effects: Option<EffectsInput>,
        view_albedo: bool,
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
                )));
            }
        };
        let effects = effects.map_or_else(
            StandardEffects::default,
            |(
                vertex_colors,
                opacity,
                mirror,
                cut,
                sway,
                sway_reach,
                ripple,
                faded,
                tinted,
                paletted,
                baked,
            )| {
                StandardEffects {
                    vertex_colors,
                    opacity,
                    mirror,
                    cut,
                    sway,
                    sway_reach,
                    ripple,
                    faded,
                    tinted,
                    paletted,
                    baked,
                }
            },
        );
        effects
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Material(id, color, shading, effects, view_albedo));
        Ok(())
    }

    fn fog(&self, fog: Option<([f32; 2], [f32; 3])>) {
        self.scene.lock().unwrap().fog = fog;
    }

    fn environment(&self, environment: Option<(f32, bool)>) -> PyResult<()> {
        if environment.is_some_and(|(intensity, _)| !intensity.is_finite()) {
            return Err(PyValueError::new_err(
                "environment intensity must be finite",
            ));
        }
        self.scene.lock().unwrap().environment = environment;
        Ok(())
    }

    fn axis(&self, id: u32, points: [[f32; 3]; 2]) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Axis(id, points));
    }

    fn surface_patterns(&self, id: u32, patterns: [[f32; 5]; 9], detail: bool) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Surface(id, Box::new(patterns), detail));
    }

    fn volume(&self, id: u32, dims: [u32; 3], bytes: &Bound<'_, PyBytes>) {
        self.scene.lock().unwrap().uploads.push_back(Upload::Volume(
            id,
            dims,
            bytes.as_bytes().to_vec(),
        ));
    }

    fn volume_material(
        &self,
        id: u32,
        params: ([f32; 3], Option<f32>, u32),
        color: &Bound<'_, PyBytes>,
        opacity: Vec<f32>,
    ) {
        let (range_density, step, max_steps) = params;
        let data = VolumeMaterialData {
            range_density,
            step,
            max_steps,
            color: color.as_bytes().to_vec(),
            opacity,
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::VolumeMaterial(id, data));
    }

    fn label_material(&self, id: u32, source: String, values: [f32; 4]) {
        let data = SpriteData {
            source,
            values,
            rgba: Vec::new(),
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Label(id, data));
    }

    fn label_tile(&self, layer: u32, rgba: &Bound<'_, PyBytes>) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::LabelTile(layer, rgba.as_bytes().to_vec()));
    }

    fn sprite(&self, id: u32, source: String, values: [f32; 4], rgba: &Bound<'_, PyBytes>) {
        let data = SpriteData {
            source,
            values,
            rgba: rgba.as_bytes().to_vec(),
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Sprite(id, data));
    }

    #[pyo3(signature = (id, source, uniforms, textures, blend, axis, samples_view=false))]
    #[allow(clippy::too_many_arguments)]
    fn manual(
        &self,
        id: u32,
        source: String,
        uniforms: Vec<f32>,
        textures: Vec<(u32, u32)>,
        blend: u8,
        axis: bool,
        samples_view: bool,
    ) {
        let data = ManualData {
            source,
            uniforms,
            textures,
            blend,
            axis,
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Manual(id, data, samples_view));
    }

    fn fragment(
        &self,
        py: Python<'_>,
        id: u32,
        source: String,
        uniforms: Vec<f32>,
        textures: Vec<Py<PyBytes>>,
    ) {
        let data = FragmentData {
            source,
            uniforms,
            textures: textures
                .iter()
                .map(|tex| tex.as_bytes(py).to_vec())
                .collect(),
        };
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Fragment(id, data));
    }

    #[pyo3(signature = (irradiance, prefiltered, brdf=None))]
    fn environment_maps(
        &self,
        py: Python<'_>,
        irradiance: &Bound<'_, PyBytes>,
        prefiltered: Vec<Py<PyBytes>>,
        brdf: Option<&Bound<'_, PyBytes>>,
    ) -> PyResult<()> {
        let data = EnvironmentData {
            irradiance: irradiance.as_bytes().to_vec(),
            prefiltered: prefiltered
                .iter()
                .map(|mip| mip.as_bytes(py).to_vec())
                .collect(),
            brdf: brdf.map(|lut| lut.as_bytes().to_vec()),
        };
        data.validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::Environment(data));
        Ok(())
    }

    fn baked_array(&self, id: u32, size: u32, layers: u32) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::BakedArray(id, size, layers));
    }

    fn baked_layer(&self, py: Python<'_>, id: u32, layer: u32, levels: Vec<Py<PyBytes>>) {
        let levels = levels
            .iter()
            .map(|level| level.as_bytes(py).to_vec())
            .collect();
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::BakedLayer(id, layer, levels));
    }

    fn texture_patch(&self, id: u32, origin: [u32; 3], size: [u32; 2], data: &Bound<'_, PyBytes>) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::TexturePatch(
                id,
                origin,
                size,
                data.as_bytes().to_vec(),
            ));
    }

    fn texture_release(&self, id: u32) {
        self.scene
            .lock()
            .unwrap()
            .uploads
            .push_back(Upload::TextureRelease(id));
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

    #[pyo3(signature = (declaration, callback=None))]
    fn compute(&self, declaration: ComputeInput, callback: Option<Py<PyAny>>) {
        let (key, source, workgroup_size, uniforms, threads, columns) = declaration;
        let bindings = columns
            .into_iter()
            .map(|(column, name, width, write)| ComputeBinding {
                column,
                name,
                width,
                write,
            })
            .collect();
        let mut scene = self.scene.lock().unwrap();
        scene.compute_pending.push_back(ComputeProgram {
            key,
            source,
            workgroup_size,
            uniforms,
            threads,
            bindings,
        });
        if let Some(callback) = callback {
            scene.compute_callbacks.push((key, callback));
        }
    }

    fn compute_transfers(&self) -> [usize; 3] {
        self.scene
            .lock()
            .unwrap()
            .compute
            .as_ref()
            .map_or([0; 3], FrameCompute::transfers)
    }

    fn compute_timings(&self, py: Python<'_>) -> PyResult<Vec<Option<u64>>> {
        py.detach(|| {
            let scene = self.scene.lock().unwrap();
            match (&scene.compute, &scene.offscreen) {
                (Some(compute), Some(renderer)) => compute.timings(renderer).map_err(render_error),
                _ => Ok(Vec::new()),
            }
        })
    }

    #[pyo3(signature = (world, width, height, labels=None, dt=None))]
    fn capture<'py>(
        &self,
        py: Python<'py>,
        world: &PyNativeWorld,
        width: u32,
        height: u32,
        labels: Option<HashMap<u32, u32>>,
        dt: Option<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dispatch = if labels.is_none() && dt.is_some() {
            compute_overrides(&self.scene).map_err(render_error)?
        } else {
            HashMap::new()
        };
        let output = py.detach(|| {
            let mut world = world.lock()?;
            self.scene
                .lock()
                .unwrap()
                .capture(&mut world, width, height, labels.as_ref(), dt, &dispatch)
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
        result.set_item("view_resources", renderer.view_counts())?;
        result.set_item("stream_resources", renderer.stream_counts())?;
        result.set_item("streamed_cells", scene.stream_cells.count())?;
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
    let settings = PyDict::new(py);
    settings.set_item("environment", stats.environment_settings)?;
    settings.set_item("fog", stats.fog_settings)?;
    settings.set_item(
        "lights",
        (stats.has_sun, stats.point_lights, stats.has_spot),
    )?;
    if let Some(post) = &stats.post_settings {
        let effects = PyDict::new(py);
        effects.set_item("ssao", post.ssao)?;
        effects.set_item("bloom", post.bloom)?;
        effects.set_item("shafts", post.shafts)?;
        effects.set_item("grade", post.grade)?;
        effects.set_item(
            "sky",
            post.sky.as_ref().map(|sky| {
                (
                    sky.zenith,
                    sky.horizon,
                    sky.sun_size_degrees,
                    sky.sun_intensity,
                    sky.glow,
                )
            }),
        )?;
        effects.set_item("fxaa", post.fxaa)?;
        settings.set_item("post", effects)?;
        settings.set_item(
            "post_formats",
            ["rgba8-srgb", "depth32float", "r8unorm", "rgba16float"],
        )?;
    }
    result.set_item("settings", settings)?;
    result.set_item("post_order", &stats.post_order)?;
    result.set_item("frame_order", &stats.frame_order)?;
    result.set_item("line_draws", stats.line_draws)?;
    result.set_item("gui_quads", stats.gui_quads)?;
    result.set_item("shadow_layers", stats.shadow_layers)?;
    result.set_item("post_target_generation", stats.post_target_generation)?;
    result.set_item("read_ns", stats.read_ns)?;
    result.set_item("upload_ns", stats.upload_ns)?;
    result.set_item("stream_upload_bytes", stats.stream_upload_bytes)?;
    result.set_item("stream_upload_ns", stats.stream_upload_ns)?;
    result.set_item("stream_resources", stats.stream_meshes)?;
    result.set_item("view_resources", stats.view_resources)?;
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
