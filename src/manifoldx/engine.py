import asyncio
import argparse
import wgpu
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
from time import perf_counter_ns

# Import ECS components
import manifoldx.ecs as ecs
from manifoldx.ecs import EntityStore
from manifoldx.commands import CommandBuffer, Command, CommandType
from manifoldx.systems import SystemRegistry
from manifoldx.resources import GeometryRegistry, MaterialRegistry, VolumeRegistry
from manifoldx.renderer import RenderPipeline
from manifoldx.components import Component, Transform, Mesh, Material
from manifoldx.camera import Camera
from manifoldx.input import InputState, _InputBridge


class _TransformProxy:
    """Read/write the Transform row of a single entity (pos 0:3, rot 3:7, scale 7:10).

    `pos` is in world coordinates, float64: the row stores pos - engine.origin in
    float32, so it stays precise however far the entity is from (0, 0).
    """

    def __init__(self, store, index, engine=None):
        self._store, self._i, self._engine = store, index, engine

    def _offset(self):
        if self._engine is None:
            return np.zeros(3)
        ox, oz = self._engine._origin
        return np.array([ox, 0.0, oz])

    @property
    def pos(self):
        return (
            self._store._components["Transform"][self._i, 0:3].astype(np.float64) + self._offset()
        )

    @pos.setter
    def pos(self, v):
        rel = np.asarray(v, np.float64) - self._offset()
        self._store._components["Transform"][self._i, 0:3] = rel.astype(np.float32)

    @property
    def rot(self):
        return self._store._components["Transform"][self._i, 3:7].copy()

    @rot.setter
    def rot(self, v):
        self._store._components["Transform"][self._i, 3:7] = np.asarray(v, np.float32)

    @property
    def scale(self):
        return self._store._components["Transform"][self._i, 7:10].copy()

    @scale.setter
    def scale(self, v):
        self._store._components["Transform"][self._i, 7:10] = np.asarray(v, np.float32)


class EntityHandle:
    """A handle to a spawned entity: move its transform, swap its geometry or material."""

    def __init__(self, engine, index, geometry_id):
        self._engine = engine
        self.index = index
        self._geometry_id = geometry_id
        self.transform = _TransformProxy(engine.store, index, engine)

    def destroy(self, free_geometry: bool = True):
        """Destroy this entity and free its geometry's GPU buffers.

        The entity goes at the frame's command flush, like Engine.destroy. Its
        geometry's buffers are freed then too, unless another live entity still
        draws the same geometry, or free_geometry is False (the caller owns them,
        as a Streamer does for the cells it keeps uploaded).
        """
        self._engine.destroy(np.array([self.index]))
        if free_geometry and self._geometry_id is not None:
            self._engine._geometries_to_free.add(self._geometry_id)

    def set_geometry(self, mesh):
        """Overwrite this entity's vertex buffer (requires equal vertex count).

        `mesh` is a modeling.Mesh or an already-baked geometry dict (so results
        shipped back from `submit_process` can be applied without reconstruction).
        A skinned entity given a mesh without joints keeps its joints; a mesh with
        joints on an entity spawned without them raises ValueError.
        """
        geo = mesh.to_geometry() if hasattr(mesh, "to_geometry") else mesh
        n_new = geo["positions"].shape[0]
        reg = self._engine._geometry_registry
        orig = reg._geometries.get(self._geometry_id)
        n_old = orig["positions"].shape[0] if orig is not None else n_new
        if n_new != n_old:
            raise ValueError(
                f"set_geometry requires equal vertex count (have {n_old}, got {n_new}); "
                "the in-place update cannot resize the buffer."
            )
        if "joints" in geo and orig is not None and "joints" not in orig:
            raise ValueError(
                "set_geometry got a mesh with joints for an entity spawned without them; "
                "spawn it from a geometry with joints (Mesh(joints=...)) to skin it."
            )
        if orig is not None and "positions" in orig:
            # Mirror the rewritten vertices on the CPU side: the shadow auto-fit
            # reads them through the registry and caches a box per geometry id.
            orig["positions"] = np.array(geo["positions"], dtype=orig["positions"].dtype)
        cache = getattr(self._engine._render_pipeline, "_geom_local_bounds", None)
        if cache is not None:
            cache.pop(self._geometry_id, None)
        bufs = reg.get_gpu_buffers(self._geometry_id)
        if bufs is None:
            return  # buffers not created until the first render pass
        stride = bufs["stride"] // 4
        inter = np.zeros((n_new, stride), dtype=np.float32)
        inter[:, 0:3] = geo["positions"]
        if stride >= 6:
            inter[:, 3:6] = geo["normals"]
        if bufs.get("has_colors"):
            inter[:, 6:9] = geo["colors"]
            if "surfaces" in geo:
                inter[:, 9] = np.asarray(geo["surfaces"], dtype=np.float32)
        elif bufs.get("has_uvs"):
            inter[:, 6:8] = geo["uvs"]
        if bufs.get("has_joints"):
            joints = geo["joints"] if "joints" in geo else orig["joints"]
            inter[:, bufs["joints_offset"] // 4] = np.asarray(joints, dtype=np.uint32).view(
                np.float32
            )
        self._engine._device.queue.write_buffer(bufs["vertex_buffer"], 0, inter.tobytes())

    def set_material(self, material):
        """Draw this entity with `material` from the next frame on.

        The same material object keeps one registry id, so switching back and
        forth (a selection highlight) does not grow the registry.
        """
        mat_id = self._engine._material_registry.register(material)
        self._engine.store._components["Material"][self.index, 0] = mat_id

    def set_joints(self, matrices):
        """Pose this entity's joints: (J, 4, 4) matrices, 1 <= J <= MAX_JOINTS.

        Column-vector convention in model space, before the entity's Transform:
        a vertex of joint j moves to matrices[j] @ (x, y, z, 1); its normal by the
        inverse transpose of the 3x3 (identity when singular). Joints >= J keep
        their previous matrix (identity until first set). Uploads J*64 bytes;
        pose many entities in one call with Engine.set_joints.
        """
        from manifoldx.skinning import MAX_JOINTS

        geo = self._engine._geometry_registry.get(self._geometry_id)
        if geo is None or "joints" not in geo:
            raise ValueError(
                "set_joints needs an entity whose geometry has joints (Mesh(joints=...))"
            )
        m = np.asarray(matrices, dtype=np.float32)
        if m.ndim != 3 or m.shape[1:] != (4, 4) or not 1 <= len(m) <= MAX_JOINTS:
            raise ValueError(
                f"set_joints expects shape (J, 4, 4) with 1 <= J <= {MAX_JOINTS}, got {m.shape}"
            )
        self._engine._render_pipeline._joints.set(self.index, m)


class Task:
    """Handle to a value being computed on the engine's background worker."""

    def __init__(self, future, engine):
        self._future = future
        self._engine = engine
        self._cb = None

    @property
    def ready(self) -> bool:
        return self._future.done()

    @property
    def result(self):
        if not self._future.done():
            raise RuntimeError("Task result not ready; check .ready or use .wait()")
        return self._future.result()

    def wait(self):
        return self._future.result()

    def on_ready(self, cb):
        self._cb = cb
        self._engine._pending_tasks.append(self)
        return self


class _Loading:
    """engine.loading(label): counts its label open while the block runs (sync or async)."""

    def __init__(self, engine, label):
        self._engine, self._label = engine, label

    def __enter__(self):
        labels = self._engine._loading_labels
        labels[self._label] = labels.get(self._label, 0) + 1
        return self

    def __exit__(self, *exc):
        labels = self._engine._loading_labels
        labels[self._label] -= 1
        if labels[self._label] <= 0:
            del labels[self._label]
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *exc):
        return self.__exit__(*exc)


def _block_mode(labels: np.ndarray, ss: int) -> np.ndarray:
    """Reduce (h*ss, w*ss) labels to (h, w), each pixel the most frequent label of its block.

    A mode, not a mean: averaging ids would invent labels on every edge.
    Ties go to the block's first (top-left) label.
    """
    if ss == 1:
        return labels
    h, w = labels.shape[0] // ss, labels.shape[1] // ss
    blocks = labels.reshape(h, ss, w, ss).transpose(0, 2, 1, 3).reshape(h, w, ss * ss)
    counts = np.stack([(blocks == blocks[..., k : k + 1]).sum(-1) for k in range(ss * ss)], axis=-1)
    pick = counts.argmax(-1)
    return np.take_along_axis(blocks, pick[..., None], axis=-1)[..., 0]


def _configure_swapchain(context, device, adapter):
    """Configure the canvas with an sRGB format and return the format used.

    Shaders write linear colour and rely on the sRGB target to encode gamma
    once. A plain 8-bit preferred format is upgraded to its sRGB variant; if
    the surface rejects that, the plain format is used (colours then render
    too dark, which beats failing to start).
    """
    preferred = str(context.get_preferred_format(adapter))
    formats = [f"{preferred}-srgb", preferred] if preferred in ("rgba8unorm", "bgra8unorm") else [preferred]
    # COPY_SRC lets `manifoldx look` read a presented frame back; a surface that refuses it still draws.
    usages = [wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC, wgpu.TextureUsage.RENDER_ATTACHMENT]
    for i, (fmt, usage) in enumerate((f, u) for f in formats for u in usages):
        try:
            context.configure(device=device, format=fmt, usage=usage)
            return fmt
        except Exception:
            if i == len(formats) * len(usages) - 1:
                raise



@dataclass
class _View:
    """A camera rendered into a texture before each frame's main pass."""
    camera: Any
    handle: Any  # TextureHandle
    depth_view: Any
    live: bool
    pending: bool
    background: Any  # linear RGBA clear colour, or None for the engine's background
    fog: bool = False


def _clear_color(color):
    """An sRGB hex or RGB/RGBA tuple as the linear RGBA a clear colour takes."""
    if color is None:
        return None
    from manifoldx.resources import srgb_to_linear

    if isinstance(color, str):
        from manifoldx.gui.style import parse_color

        color = parse_color(color)
    if len(color) not in (3, 4):
        raise ValueError(f"background must be a hex string or an RGB/RGBA tuple, got {color!r}")
    r, g, b = (float(v) for v in srgb_to_linear(color[:3]))
    return (r, g, b, float(color[3]) if len(color) == 4 else 1.0)

class Engine:
    def __init__(
        self,
        title: str,
        *,
        height: int = 600,
        width: int = 800,
        fullscreen: bool = False,
        monitor: int | str | None = None,
        max_fps: float = 240.0,
        max_entities: int = 100_000,
        check_numerics: bool = True,
    ):
        self.title = title
        self.h = height
        self.w = width
        # Follow canvas resize events (interactive windows). Off for the
        # supersampled offscreen still, whose canvas is larger than (w, h).
        self._follow_resize = True
        self.fullscreen = fullscreen
        # Fullscreen target: index or name substring; None reads MANIFOLDX_MONITOR,
        # then uses the primary monitor (see backends.get_desktop_canvas).
        self.monitor = monitor
        # Interactive frame-rate cap (see backends.get_desktop_canvas); vsync still applies.
        self.max_fps = max_fps
        self.check = check_numerics  # Enable/disable validation warnings
        ecs.ENABLE_VALIDATION = check_numerics  # Set global flag for ECS validation

        # Private stuff
        self._running = False
        self._render_canvas = None
        self._wgpu_context = None
        self._adapter = None
        self._device = None
        self._event_loop = None
        self._present_mode = "fifo"
        self._texture_format = wgpu.TextureFormat.bgra8unorm
        # Event-driven system (replaces _startup/_shutdown/_update_callbacks)
        from manifoldx.events import EventBus, FrameWaiters

        self._event_bus = EventBus()
        # Lazy fallback loop: only created when no asyncio loop is currently
        # running (e.g. tests, render() to MP4). In interactive run() mode,
        # rendercanvas is itself an asyncio loop and we adopt it instead —
        # see _get_active_loop() below.
        self._aio_loop: asyncio.AbstractEventLoop | None = None
        self._frame_waiters = FrameWaiters(self._get_active_loop)
        # Task error spool — populated by add_done_callback when an async
        # handler raises (other than CancelledError). Drained by
        # _pump_aio_loop, which re-raises the first error per the v1
        # "errors crash the engine" policy.
        self._task_errors: list[BaseException] = []
        self._tasks: set = set()  # async-handler tasks still running, cancelled by _detach

        # Values that survive a hot reload under `manifoldx run` (fresh otherwise).
        from manifoldx import hot

        self.state = hot._state if hot._state is not None else hot.State()
        self._looks = []  # futures waiting for the next presented frame as PNG bytes (the dev server's look)
        self._look_copy = None  # a frame copied for them, read back on the frame after
        # Seconds the last frame spent: updating (events, async, systems), acquiring the window's
        # next image (vsync, or the GPU behind), and rendering the rest. manifoldx run --debug reads it.
        self.frame_split = {"update": 0.0, "acquire": 0.0, "render": 0.0}
        self._acquire_s = 0.0

        # Input layer — keyboard + mouse + resize. The bridge attaches to
        # the canvas in _init_canvas; until then it is dormant.
        self.input = InputState()
        self._input_bridge = _InputBridge(self, self.input)

        # GUI layer — list-like container of root Panels plus a
        # pointer_over_gui flag. The bridge (Plan 2) will toggle the flag.
        from manifoldx.gui.widgets import _GuiRoot

        self.gui = _GuiRoot()

        # GUI input bridge — routes pointer events through hit-test
        # and per-widget state. Subscribes to pointer_down/move/up at
        # construction; needs `self.gui` to already exist.
        from manifoldx.gui.bridge import _GuiBridge

        self._gui_bridge = _GuiBridge(self)

        # === ECS Infrastructure ===
        self.store = EntityStore(max_entities)
        self.commands = CommandBuffer()
        self.systems = SystemRegistry()

        # Resource registries
        self._geometry_registry = GeometryRegistry(self._device)
        self._material_registry = MaterialRegistry(self._device)
        self._volume_registry = VolumeRegistry(self._device)
        from manifoldx.textures import TextureRegistry

        self._texture_registry = TextureRegistry()
        self._views: dict = {}  # TextureHandle.id -> _View: cameras rendered into textures

        # Render pipeline
        self._render_pipeline = RenderPipeline(self.store, self._device)

        # Compute systems — declarative GPU work registered via engine.compute(cls).
        from manifoldx.compute import ComputeRunner

        self._compute_runner = ComputeRunner(self)
        self._last_dt: float = 1 / 60
        self._frame_index: int = 0

        # Configurable timestep
        self._use_fixed_dt = False
        self._fixed_dt_value = 1 / 60
        self._last_time = None
        self._start_time = None
        self.elapsed: float = 0.0
        self.background_color = (0.1, 0.1, 0.2)  # RGB clear color; set for e.g. fog horizon
        self._executor = None
        self._process_executor = None
        self._pending_tasks = []
        self.fog_enabled = False
        self.fog_start = 0.0
        # Points that swaying meshes part around, such as grass around whatever walks
        # through it: up to renderer.MAX_BENDERS (x, z, radius, strength) in world coordinates.
        # Inside the radius a vertex below 1 unit of its model's height leans away from
        # the point by strength times its sway height, fading out toward the edge.
        self.benders = []
        self.fog_end = 1.0
        self.fog_color = (0.1, 0.1, 0.2)

        # Register built-in components
        Transform.register(self.store)
        Mesh.register(self.store)
        Material.register(self.store)

        # Camera for MVP matrices
        self._camera = Camera()
        self.camera = self._camera

        # External lights (not in ECS)
        self._lights = []

        # IBL environment — set via set_environment()
        self._environment = None
        self._env_task = None  # the environment's precompute, on the background worker
        self._loading_panel = None  # the spinner shown while something is still loading
        self._loading_labels: dict[str, int] = {}  # engine.loading(label) blocks now open
        self._startup_pending = False  # run() fires startup on its first frame

        # Still-frame rendering (render_frame): the offscreen canvas is created
        # once, at the first call's size.
        self._still_size = None
        self._id_materials = {}  # label -> FlatMaterial, kept alive for the registry

        # Directional sun (separate from the point-light array). Consumed by
        # StandardMaterial as a directional term; also a shadow caster.
        self._sun = None
        # Spot light (flashlight): cone + distance falloff, and a shadow caster.
        self._spot = None
        # Floating origin (x, z), float64: entities, camera and lights are stored
        # relative to it, so the GPU's float32 stays precise far from (0, 0).
        self._origin = (0.0, 0.0)
        # Geometry ids whose entities EntityHandle.destroy() removed; freed at the
        # next command flush if no live entity still draws them.
        self._geometries_to_free = set()
        # Shadow-map config — set via enable_shadows().
        self._shadow_config = None

        # Lazily-constructed label atlas, materialized on first label render.
        self._label_atlas = None

    def get_label_atlas(self):
        """Lazily construct the label atlas on first use.

        Used by the renderer's label pass and by user code that needs to
        register strings ahead of time.
        """
        if self._label_atlas is None:
            from manifoldx.viz.text import LabelTextureAtlas

            self._label_atlas = LabelTextureAtlas()
        return self._label_atlas

    def on(self, event: str):
        """Register a sync or async handler for an event.

        Usage:
            @engine.on("startup")
            def init(payload): ...

            @engine.on("frame")
            async def each_frame(payload): ...
        """
        return self._event_bus.on(event)

    def off(self, event: str, func) -> None:
        """Unregister a handler added with `@engine.on(event)`.

        Safe during dispatch, from the handler itself included; a removed handler
        is not called again. Async tasks it already started keep running. Raises
        ValueError if `func` is not a handler for `event`.
        """
        self._event_bus.off(event, func)

    def emit(self, event: str, payload=None) -> None:
        """Queue an event for delivery at the start of the next frame."""
        self._event_bus.emit(event, payload)

    def tick(self):
        """Return a future that resolves at the next frame boundary."""
        return self._frame_waiters.add_tick()

    def delay(self, seconds: float):
        """Return a future that resolves after `seconds` of engine.elapsed."""
        return self._frame_waiters.add_delay(seconds, self.elapsed)

    def elapsed_at(self, target: float):
        """Return a future that resolves once engine.elapsed >= target."""
        return self._frame_waiters.add_elapsed_at(target)

    async def run_blocking(self, fn, *args, **kwargs):
        """Run a blocking callable in the default executor and await its result."""
        import functools

        # The loop this coroutine runs on: rendercanvas's in run(), the private one offline.
        return await asyncio.get_running_loop().run_in_executor(
            None, functools.partial(fn, *args, **kwargs)
        )

    def _get_active_loop(self) -> asyncio.AbstractEventLoop:
        """Return the asyncio loop async handlers should run on.

        Prefers the currently-running loop (e.g. rendercanvas in interactive
        run() mode). If no loop is running, lazily creates and returns a
        private fallback (used by tests and the headless render() path).
        """
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._aio_loop is None or self._aio_loop.is_closed():
                self._aio_loop = asyncio.new_event_loop()
            return self._aio_loop

    def _on_task_done(self, task) -> None:
        """Done-callback wired onto every async-handler task.

        Called synchronously when the task completes. We surface non-cancel
        exceptions into _task_errors so _pump_aio_loop can re-raise them on
        the next pump (per the v1 "errors crash the engine" policy).
        """
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._task_errors.append(exc)

    def system(self, func):
        """Decorator to register a system function.

        Parses Query type hints to determine which components to query.
        """
        import inspect
        from manifoldx.systems import Query

        # Parse type hints to extract component names
        component_names = []
        hints = func.__annotations__
        for param_name, hint in hints.items():
            if isinstance(hint, Query):
                # Query[Transform] -> Query with components tuple
                comps = hint.components
                if not isinstance(comps, tuple):
                    comps = (comps,)
                for c in comps:
                    if isinstance(c, str):
                        component_names.append(c)
                    elif hasattr(c, "__name__"):
                        component_names.append(c.__name__)

        self.systems.register(func, component_names)
        return func

    def remove_system(self, func) -> None:
        """Unregister a system added with `@engine.system`.

        Safe to call from inside any system, the one being removed included; a
        removed system does not run again, the current frame included. Raises
        ValueError if `func` is not a registered system.
        """
        self.systems.remove(func)

    def component(self, cls):
        """Decorator to register a component class with this engine."""
        from manifoldx.ecs import _make_component_class

        return _make_component_class(cls, self)

    def register_volume(
        self,
        data: "np.ndarray",
        *,
        name: str | None = None,
    ) -> int:
        """Register a 3D scalar field for volume rendering.

        Args:
            data: numpy array of shape (Nz, Ny, Nx), dtype=float32,
                  C-contiguous. data[k, j, i] is the voxel at integer
                  coordinates (i, j, k) along local-space (x, y, z).
            name: optional diagnostic label.

        Returns:
            Integer handle to pass into `Volume(volume_id=handle)`.
        """
        return self._volume_registry.register(data, name=name)

    def update_volume(self, handle: int, data: "np.ndarray") -> None:
        """Replace voxel data for a registered volume. Same shape required.

        Bumps the dirty bit; the renderer re-uploads on the next frame.
        """
        self._volume_registry.update(handle, data)

    def bind_compute_volume(self, handle: int, kernel_field: str) -> None:
        """v2: bind a Phase-2 compute kernel's `Writes[Volume3D]` field
        to this volume's storage texture. v1 raises NotImplementedError —
        the API is reserved here so v2 can land without a breaking change.
        """
        raise NotImplementedError(
            "bind_compute_volume is reserved for v2; v1 supports only "
            "CPU-uploaded volumes via register_volume()/update_volume()."
        )

    def compute(self, cls):
        """Register a Compute subclass to run as part of the per-frame loop.

        Usage:
            class Gravity(Compute):
                ...

            engine.compute(Gravity)

        Validates the kernel synchronously when the wgpu device is available:
        compiles the WGSL via `cls().compile()` and feeds it through
        `device.create_shader_module()` so transpile + validation errors
        surface at registration time, not on first frame.
        """
        if self._device is not None:
            instance = cls()
            wgsl = instance.compile()
            try:
                self._device.create_shader_module(code=wgsl)
            except Exception as e:
                from manifoldx.compute.transpile import ComputeShaderCompileError

                raise ComputeShaderCompileError(
                    category="wgpu-validation",
                    message=str(e),
                    filename=getattr(cls, "__module__", "<class>"),
                    line=0,
                    col=0,
                    source_line=None,
                )
        self._compute_runner.register(cls)
        return cls

    def set_lights(self, lights: list):
        """Set external lights (passed to renderer, not in ECS)."""
        self._lights = lights

    def add_light(self, light):
        """Add a single light to the external lights list."""
        self._lights.append(light)

    def background(self, fn):
        """Decorator: run `fn` on a background worker thread; calls return a Task.

        The decorated function must be pure (no GPU / engine-state mutation); it
        returns plain data delivered back to the main thread.
        """

        def submit(*args, **kwargs):
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1)
            return Task(self._executor.submit(fn, *args, **kwargs), self)

        return submit

    def submit_process(self, fn, *args):
        """Run `fn(*args)` in a separate process (no GIL contention); returns a Task.

        For CPU-bound work heavy enough that a thread would stall the render loop
        (numpy holds the GIL through its Python-level parts). `fn` must be a
        top-level picklable function returning picklable data; scripts using this
        must guard their setup with `if __name__ == "__main__":` (spawn context).
        """
        if self._process_executor is None:
            import multiprocessing as mp

            self._process_executor = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn")
            )
        return Task(self._process_executor.submit(fn, *args), self)

    def _drain_tasks(self):
        """Fire on_ready callbacks for finished tasks (main thread)."""
        if not self._pending_tasks:
            return
        still = []
        for t in self._pending_tasks:
            if t._future.done():
                if t._cb is not None:
                    t._cb(t._future.result())
            else:
                still.append(t)
        self._pending_tasks = still

    # === Floating origin ===

    @property
    def origin(self) -> tuple[float, float]:
        """The floating origin (x, z) in world coordinates, float64."""
        return self._origin

    def set_origin(self, x: float, z: float) -> None:
        """Move the floating origin: every entity then draws at pos - origin.

        EntityHandle.transform.pos and a spawned Transform's pos stay in world
        coordinates. What lives in the render frame is shifted here, so nothing
        moves on screen: the raw Transform rows, the camera, the spot light and
        the manual shadow target. Call it when the focus has wandered far from
        the origin, not every frame.
        """
        new = (float(x), float(z))
        dx, dz = self._origin[0] - new[0], self._origin[1] - new[1]
        self._origin = new
        if dx == 0.0 and dz == 0.0:
            return
        d = np.array([dx, 0.0, dz])
        if "Transform" in self.store._components:
            alive = self.store._alive
            rows = self.store._components["Transform"]
            rows[alive, 0:3] = (rows[alive, 0:3].astype(np.float64) + d).astype(np.float32)
        cam = self.camera
        cam.position = (cam.position.astype(np.float64) + d).astype(np.float32)
        cam.target = (cam.target.astype(np.float64) + d).astype(np.float32)
        if self._spot is not None and getattr(self._spot, "position", None) is not None:
            self._spot.position = tuple(np.asarray(self._spot.position, np.float64) + d)
        if self._shadow_config is not None:
            self._shadow_config["target"] = tuple(
                np.asarray(self._shadow_config["target"], np.float64) + d
            )

    def _free_geometries(self):
        """Free the GPU buffers of geometries EntityHandle.destroy() let go of,
        unless a live entity still draws them."""
        if not self._geometries_to_free:
            return
        pending, self._geometries_to_free = self._geometries_to_free, set()
        mesh = self.store._components.get("Mesh")
        used = set(np.unique(mesh[self.store._alive, 0]).tolist()) if mesh is not None else set()
        for gid in pending - used:
            self._geometry_registry.release(gid)

    def enable_fog(self, start, end, color=None):
        """Linear distance fog for StandardMaterial; color defaults to background_color."""
        self.fog_enabled = True
        self.fog_start = float(start)
        self.fog_end = float(end)
        self.fog_color = tuple(color) if color is not None else tuple(self.background_color)

    def set_sun(self, light):
        """Set the single directional sun (a DirectionalLight).

        The sun is engine-level state (like the IBL environment), distinct
        from the point-light array. StandardMaterial adds a directional term
        for it; enable_shadows() makes it cast a shadow.
        """
        self._sun = light

    def set_spot(self, light):
        """Set the single spot light (a SpotLight) — a flashlight cone with
        angular + distance falloff. Consumed by StandardMaterial; when shadows
        are enabled the spot is the shadow caster (takes priority over the sun).
        """
        self._spot = light

    def enable_shadows(
        self,
        target=(0.0, 0.0, 0.0),
        extent=10.0,
        resolution=2048,
        near=0.1,
        far=50.0,
        bias=0.005,
        pcf_radius=1,
        auto_fit=True,
    ):
        """Enable directional shadow mapping for the sun set via set_sun().

        pcf_radius: half-width of the PCF kernel in shadow-map texels. 0 = a
        single hard-shadow tap; 1 = 3x3 soft edges (default); 2 = 5x5, etc.
        auto_fit: when True (default), the shadow frustum is fit to the scene's
        bounding sphere every frame and target/extent/near/far are ignored. Set
        False to drive the ortho box manually via target/extent/near/far.
        """
        self._shadow_config = {
            "target": tuple(target),
            "extent": float(extent),
            "resolution": int(resolution),
            "near": float(near),
            "far": float(far),
            "bias": float(bias),
            "pcf_radius": int(pcf_radius),
            "auto_fit": bool(auto_fit),
        }

    def set_environment(self, env):
        """Set the IBL environment.

        env: EnvironmentMap instance, preset name string ("studio", "sky",
             "neutral", "dark"), or None to disable IBL.
        """
        from manifoldx.ibl import EnvironmentMap, PRESETS

        self._env_task = None  # a pending precompute belongs to the environment it replaces
        if env is None:
            self._environment = None
        elif isinstance(env, str):
            if env not in PRESETS:
                raise ValueError(f"Unknown preset {env!r}. Known: {list(PRESETS)}")
            self._environment = PRESETS[env]()
        elif isinstance(env, EnvironmentMap):
            self._environment = env
        else:
            raise TypeError(f"Expected EnvironmentMap, preset string, or None; got {type(env)}")

    @property
    def environment(self):
        """The active EnvironmentMap, or None if IBL is disabled."""
        return self._environment

    # -- loading ---------------------------------------------------------------
    # The environment's precompute takes seconds of CPU. It runs on the background
    # worker, started by the first frame that needs it; until it is done the scene
    # draws without the environment's light and a spinner says what is loading, so
    # the window keeps answering. Offline renders wait for it instead (manifoldx#25).

    def _ready_environment(self):
        """The environment the renderer may use now: None until its precompute is done."""
        env = self._environment
        return env if env is not None and env._computed else None

    def _start_environment(self):
        env = self._environment
        if env is not None and not env._computed and self._env_task is None:
            self._env_task = self.background(env._precompute)()

    def _settle_environment(self):
        """Start the precompute if needed and wait for it: for frames nobody watches load."""
        self._start_environment()
        if (
            self._env_task is not None
            and self._environment is not None
            and not self._environment._computed
        ):
            self._env_task.wait()

    def loading(self, label: str):
        """Mark work the scene is still waiting for, as `with` or `async with`:

            @engine.on("startup")
            async def load(payload):
                async with engine.loading("model"):
                    for path in paths:
                        mesh = await engine.run_blocking(load_obj, path)
                        engine.spawn(Mesh(mesh), ...)

        While a block is open, run() shows the loading spinner with its label, and
        offline frames (render_frame, render, export_gltf) wait for every block to
        close. A startup handler should open its block before its first await.
        """
        return _Loading(self, label)

    def _loading(self):
        env = (
            ["lighting"]
            if self._environment is not None and not self._environment._computed
            else []
        )
        return env + [label for label, n in self._loading_labels.items() if n > 0]

    def _fire_startup(self):
        self._event_bus.dispatch_immediate(self, "startup", {})

    def _settle(self, timeout: float = 600.0):
        """Wait for everything still loading, for frames nobody watches load: the
        environment's precompute and every engine.loading block. Advances the engine's
        asyncio loop and frame waiters meanwhile, so loaders that await tick() progress."""
        import time

        self._settle_environment()
        deadline = time.monotonic() + timeout
        for _ in range(2):  # let fresh tasks reach their first `loading`
            self._pump_aio_loop()
        while any(n > 0 for n in self._loading_labels.values()):
            if time.monotonic() > deadline:
                raise RuntimeError(f"still loading after {timeout:.0f} s: {self._loading()}")
            self._frame_waiters.resolve(self.elapsed)
            self._drain_tasks()
            if self._aio_loop is not None and not self._aio_loop.is_closed():
                self._aio_loop.run_until_complete(asyncio.sleep(0.001))
            if self._task_errors:
                raise self._task_errors.pop(0)

    def _loading_line(self):
        spin = "|/-\\"[int(perf_counter_ns() // 125_000_000) % 4]
        return f"{spin}  loading {', '.join(self._loading())}..."

    def _update_loading_overlay(self):
        from manifoldx.gui import Panel, ValueDisplay

        if self._loading():
            if self._loading_panel is None:
                self._loading_panel = Panel(
                    children=[ValueDisplay(getter=self._loading_line, min_width=200)],
                    anchor="center",
                    style_overrides={
                        "width": 230,
                        "height": 40,
                        "padding": 10,
                        "bg": "#1a1a1ad0",
                        "radius": 4,
                    },
                )
                self.gui.append(self._loading_panel)
        elif self._loading_panel is not None:
            self.gui.remove(self._loading_panel)
            self._loading_panel = None

    def _on_canvas_resize(self, ev):
        """Keep (w, h) at the canvas' logical size, so the camera aspect, GUI
        layout and hit-tests match the window rather than the construction size."""
        if self._follow_resize and ev.width > 0 and ev.height > 0:
            self.w = int(round(ev.width))
            self.h = int(round(ev.height))

    def quit(self):
        """Stop run() and close the window.

        quit closes the window; to change screens keep the engine and use
        clear_scene().
        """
        self._running = False
        self.shutdown_events()
        # Stop the rendercanvas event loop
        if hasattr(self, "_event_loop") and self._event_loop is not None:
            self._event_loop.stop()
        # Also close the canvas
        if self._render_canvas is not None:
            try:
                self._render_canvas.close()
            except Exception:
                pass

    def shutdown_events(self) -> None:
        """Fire 'shutdown', cancel async tasks, drain the loop, close it.

        Idempotent: safe to call from quit() and from finalization paths.
        Handles both interactive (rendercanvas-owned loop running) and
        headless (private fallback) modes.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        # 1. Sync 'shutdown' handlers run inline; async ones get scheduled
        #    as tasks on whichever loop is active.
        self._event_bus.dispatch_immediate(self, "shutdown", {})

        # 2. Determine which loop our tasks live on.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        target_loop = running if running is not None else self._aio_loop
        if target_loop is None:
            return  # nothing was ever async; nothing to clean up.

        # 3. Cancel every outstanding task so while-True coroutines get
        #    CancelledError on their next await.
        for task in list(asyncio.all_tasks(loop=target_loop)):
            task.cancel()

        # 4. Cancel any outstanding waiter futures.
        self._frame_waiters.cancel_all()

        # 5. Pump the loop one final time so try/finally cleanup runs.
        #    Only safe with run_until_complete when no loop is running;
        #    when one is, the running loop will dispatch the cancellations
        #    on its own and we just collect errors.
        if running is None and not self._aio_loop.is_closed():
            try:
                self._aio_loop.run_until_complete(asyncio.sleep(0))
            finally:
                pending = self._task_errors
                self._task_errors = []
                self._aio_loop.close()
        else:
            pending = self._task_errors
            self._task_errors = []

        # 6. Surface the first non-cancel exception (if any) per the v1
        #    "errors propagate" policy.
        if pending:
            raise pending[0]

    # === Timestep Configuration ===
    def set_fixed_timestep(self, dt: float):
        """Use fixed timestep for deterministic simulations."""
        self._use_fixed_dt = True
        self._fixed_dt_value = dt

    def use_wall_clock(self):
        """Use actual elapsed time (default for animations)."""
        self._use_fixed_dt = False

    def _compute_dt(self):
        """Compute delta time based on configuration."""
        if self._use_fixed_dt:
            return self._fixed_dt_value

        current_time = perf_counter_ns()
        if self._last_time is None:
            self._last_time = current_time
            self._start_time = current_time
            return self._fixed_dt_value

        dt = (current_time - self._last_time) / 1_000_000_000
        self._last_time = current_time
        self.elapsed = (current_time - self._start_time) / 1_000_000_000
        return dt

    # === Spawn & Destroy ===
    def spawn(self, *args, n: int = 1, **kwargs):
        """Spawn n entities now, with these components, and return a handle to the first (None for n=0)."""
        # Handle Mesh, Material, and Transform component objects
        processed_kwargs = {}
        mesh_geo_id = None

        # Convert positional args to kwargs by class name
        for arg in args:
            name = type(arg).__name__
            kwargs[name] = arg

        for name, value in kwargs.items():
            # Auto-register Component subclasses on first encounter so callers
            # don't have to call `engine.store.register_component(...)` by hand.
            # Classes decorated with `@engine.component` register themselves at
            # decorator time via _make_component_class, so they don't need a
            # branch here — they fall through to the get_data path below.
            if isinstance(value, Component):
                type(value).register(self.store)

            if hasattr(value, "get_data"):
                # It's a component object (Mesh, Material, Transform) - get data from it
                # Pass appropriate registry based on component type
                if name == "Material":
                    processed_kwargs[name] = value.get_data(n, self._material_registry)
                else:
                    processed_kwargs[name] = value.get_data(n, self._geometry_registry)
                    if name == "Mesh":
                        mesh_geo_id = getattr(value, "_geometry_id", None)
            elif np.isscalar(value):
                # Broadcast scalars to arrays
                processed_kwargs[name] = np.full((n,), value, dtype=np.float32)
            else:
                processed_kwargs[name] = value

        # Also register built-in Transform component if not already
        if "Transform" not in self.store._components:
            Transform.register(self.store)

        # A Transform's pos is world coordinates: store it relative to the origin,
        # subtracted in float64 so a far position keeps its precision.
        t = kwargs.get("Transform")
        if isinstance(t, Transform) and t._pos is not None and self._origin != (0.0, 0.0):
            rel = np.asarray(t._pos, np.float64) - np.array([self._origin[0], 0.0, self._origin[1]])
            processed_kwargs["Transform"][:, 0:3] = rel.astype(np.float32)

        # NOW spawn the entities immediately (not deferred)
        if n > 0:
            indices = self.store.spawn(n, **processed_kwargs)
            return EntityHandle(self, int(indices[0]), mesh_geo_id)
        return None

    def set_joints(self, indices, matrices):
        """Pose many skinned entities in one call: matrices[k] poses entity indices[k].

        `indices` is (N,) entity indices, `matrices` is (N, J, 4, 4) with
        1 <= J <= MAX_JOINTS, J the same for all. Each pose follows the rule of
        EntityHandle.set_joints; joints >= J keep their previous matrix. Writes
        every block at once and uploads the touched range of blocks once, so a
        crowd costs one call per frame instead of one per entity.
        """
        from manifoldx.skinning import MAX_JOINTS

        idx = np.asarray(indices, dtype=np.int64)
        m = np.asarray(matrices, dtype=np.float32)
        if (idx.ndim != 1 or m.ndim != 4 or len(m) != len(idx) or m.shape[2:] != (4, 4)
                or not 1 <= m.shape[1] <= MAX_JOINTS):
            raise ValueError(
                f"set_joints expects indices of shape (N,) and matrices of shape (N, J, 4, 4) "
                f"with 1 <= J <= {MAX_JOINTS}, got {idx.shape} and {m.shape}"
            )
        if len(idx) == 0:
            return
        meshes = self.store._components.get("Mesh")
        geo_ids = np.unique(meshes[idx, 0]) if meshes is not None else [None]
        for g in geo_ids:
            geo = None if g is None else self._geometry_registry.get(int(g))
            if geo is None or "joints" not in geo:
                raise ValueError(
                    "set_joints needs an entity whose geometry has joints (Mesh(joints=...))"
                )
        self._render_pipeline._joints.set_many(idx, m)

    def set_material(self, entity, material) -> None:
        """Draw `entity` (an EntityHandle or an entity index) with `material` from the
        next frame on, as EntityHandle.set_material does. Raises ValueError when the
        entity is not alive.
        """
        index = entity.index if isinstance(entity, EntityHandle) else int(entity)
        if not (0 <= index < len(self.store._alive) and self.store._alive[index]):
            raise ValueError(f"set_material: entity {index} is not alive")
        self.store._components["Material"][index, 0] = self._material_registry.register(material)

    def destroy(self, indices):
        """Destroy entities matching condition by emitting DESTROY command."""
        if indices is None:
            return

        # If boolean array, convert to indices
        indices = np.asarray(indices)
        if indices.dtype == np.bool_:
            indices = np.where(indices)[0]

        if hasattr(indices, "__len__") and len(indices) > 0:
            self.commands.append(Command(CommandType.DESTROY, {"indices": indices}))

    def clear_scene(self) -> None:
        """Destroy every live entity and remove every GUI panel: leave a screen.

        For apps that show several screens in one window: keep one Engine and one
        run(), and call clear_scene() between screens (quit() closes the window).
        Systems, event handlers, the camera, lights, sun, spot, shadows, sky,
        background, fog and environment stay; remove what the screen added with
        remove_system() and off(). The loading spinner stays while loading.

        Safe mid-frame, from a system, handler or async task: entities are
        destroyed at the frame's command flush like destroy(), so entities spawned
        after this call survive. GPU geometry and materials stay registered.
        """
        self.destroy(np.where(self.store._alive)[0])
        self.gui.clear()
        if self._loading_panel is not None:
            self.gui.append(self._loading_panel)
        self._gui_bridge._captured = {}  # a drag on a removed widget ends here

    # === Canvas Initialization ===

    def _init_canvas(self, canvas, gpu=None):
        """Initialize WebGPU context from a canvas (shared by run() and render()).

        gpu is (adapter, device, texture_format) of an engine already drawing on
        this canvas: a hot reload reuses them instead of requesting a new device.
        """
        self._render_canvas = canvas

        # Wire input event flow: rendercanvas → _InputBridge → bus + state.
        self._input_bridge.attach(canvas)
        # A window that is never resized sends no resize event; start from its size.
        w, h = canvas.get_logical_size()
        self.input._viewport_size = (int(round(w)), int(round(h)))

        # Get the wgpu context from the canvas
        self._wgpu_context = canvas.get_wgpu_context()

        if gpu is not None:
            self._adapter, self._device, self._texture_format = gpu
        else:
            # Request adapter and device (use sync API to avoid deprecation warnings)
            self._adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            self._device = self._adapter.request_device_sync()

            # Get preferred texture format from canvas context
            # Configure the swap chain (sRGB target: it encodes gamma, once)
            self._texture_format = _configure_swapchain(self._wgpu_context, self._device, self._adapter)

        # Update registries with device reference
        self._geometry_registry._device = self._device
        self._material_registry._device = self._device
        self._volume_registry._device = self._device

        # Depth texture created lazily to match canvas size
        self._depth_texture = None
        self._depth_texture_view = None
        self._depth_texture_size = (0, 0)

        # Initialize timing
        self._start_time = perf_counter_ns()
        self._last_time = self._start_time
        self.elapsed = 0.0

    # === Frame Rendering (shared by run() and render()) ===

    def _draw_frame(self):
        """Draw a single frame. Backend-agnostic - used by both run() and render()."""
        started = time.perf_counter()
        # Deliver finished background-task callbacks on the main thread.
        self._drain_tasks()
        # Check if we should stop (only for run() with event loop)
        if not self._running:
            return False  # Stop the event loop

        # Double-check canvas state
        try:
            if self._render_canvas.get_closed():
                return False
        except Exception:
            return False

        dt = self._compute_dt()
        is_fixed = self._use_fixed_dt
        self._last_dt = dt

        # Step 2: resolve frame waiters (tick / delay / elapsed_at)
        self._frame_waiters.resolve(self.elapsed)

        # Step 2.5: input bridge frame swap — finalize this-frame
        # just_pressed/just_released sets and delta accumulators before
        # any handler or system runs.
        self._input_bridge.begin_frame()

        # GUI bridge frame reset — pointer_over_gui rolls back to False
        # at the head of each frame; the per-event hit-test will set it
        # True again if the cursor is over a widget this frame.
        self._gui_bridge.begin_frame()

        # Background loading (the environment's precompute) and its spinner.
        self._start_environment()
        self._update_loading_overlay()

        # Step 3: drain pending events (frame N-1's emits + this frame's 'frame')
        frame_payload = {
            "dt": dt,
            "elapsed": self.elapsed,
            "frame": self._frame_index,
        }
        # Inject the inline 'frame' event ahead of the user-emitted queue,
        # so frame handlers see CURRENT-frame data (not last-frame's).
        self._event_bus._pending.insert(0, ("frame", frame_payload))
        self._event_bus.dispatch_pending(self)

        # Step 4: pump asyncio loop (async handlers + waiter wakers).
        self._pump_aio_loop()

        # Step 5: run user systems (may emit commands).
        self.systems.run_all(self, dt)

        # Step 6: flush command buffer (events + handlers + systems).
        # Commands queued between frames (async tasks resumed by the running loop
        # in run() mode) are still in the buffer here, so they are not lost.
        self.commands.execute(self.store)
        self.commands.clear()
        self._free_geometries()

        # Step 7: GPU compute systems.
        self._compute_runner.run_all(dt)
        self._frame_index += 1

        # Under a fixed timestep the clock is the sum of the steps simulated so
        # far: this step is done, so the next frame (and every reader after it)
        # sees one more dt. Wall-clock mode leaves elapsed to _compute_dt.
        if is_fixed:
            self.elapsed += dt

        updated = time.perf_counter()
        self._render_only(dt)
        self.frame_split = {"update": updated - started, "acquire": self._acquire_s,
                            "render": time.perf_counter() - updated - self._acquire_s}
        return True  # Continue rendering

    def _render_only(self, dt):
        """Steps 8 onward of _draw_frame: draw the current state, no simulation."""
        # Step 8: render pipeline.
        self._render_pipeline.run(self, dt)
        try:
            self._draw_passes()
        finally:
            self._render_pipeline.end_frame()

    def _draw_passes(self):

        # Ensure render pipeline is initialized
        self._render_pipeline._ensure_pipeline(self._device, self._texture_format)

        if self._look_copy is not None:  # the GPU finished last frame's copy by now
            copy, self._look_copy = self._look_copy, None
            copy.finish()

        # Get the next frame's texture and create view; waiting here is vsync or the GPU behind
        acquiring = time.perf_counter()
        texture = self._wgpu_context.get_current_texture()
        self._acquire_s = time.perf_counter() - acquiring
        texture_view = texture.create_view()

        # Recreate depth texture if canvas size changed
        tex_size = (texture.size[0], texture.size[1])
        if tex_size != self._depth_texture_size:
            self._depth_texture = self._device.create_texture(
                size=(tex_size[0], tex_size[1], 1),
                format=wgpu.TextureFormat.depth24plus,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self._depth_texture_view = self._depth_texture.create_view()
            self._depth_texture_size = tex_size

        # Second cameras first: each view is its own submit, so its globals and
        # instance uploads land before the main pass overwrites them.
        self._render_views()

        # Create command encoder
        command_encoder = self._device.create_command_encoder()

        # Shadow pass — depth-only, renders the scene from the sun's POV into
        # the shadow map, which the main mesh pass then samples. No-op unless
        # both a sun (set_sun) and enable_shadows() are set.
        self._render_pipeline.render_shadow_map(self, command_encoder)

        # Create render pass with clear color and depth attachment
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture_view,
                    "resolve_target": None,
                    "clear_value": (*self.background_color, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self._depth_texture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )

        # Issue draw calls via render pipeline
        self._render_pipeline.render(self, render_pass)

        # End render pass
        render_pass.end()

        # Submit command buffer
        self._device.queue.submit([command_encoder.finish()])

        if self._looks:  # asked by the dev server: copy the frame now, read it next frame
            looks, self._looks = self._looks, []
            from manifoldx import devchannel

            self._look_copy = devchannel.Readback(self._device, texture, self._texture_format, looks)

    def log(self, message: str) -> None:
        """A line for the dev log under `manifoldx run` (its page, /api/log, .manifoldx/log.txt); nothing otherwise.

        Write what an agent following the person needs: where they are, what opened, what they chose."""
        from manifoldx import devchannel

        devchannel.LOG.info(message)

    def render_to_texture(self, camera, size, *, live: bool = True, background=None,
                          fog: bool = False):
        """Render the scene through `camera` into a texture of `size` = (width, height)
        pixels, before each frame's main pass, and return its TextureHandle.

        The texture works as an Image's source in the GUI and as an albedo_map. The
        GUI is not drawn into it. With live=False it renders on the next frame and
        then only when refresh_texture asks, so a still view costs nothing.
        `background` is the clear colour in sRGB, like GUI colours: a hex string or an
        RGB or RGBA tuple in [0, 1] (default: the engine's background, opaque). The
        world's fog applies only with fog=True: fog grows with distance from the
        camera, and a minimap camera high above the ground would come out pale."""
        from manifoldx.textures import TextureHandle

        if self._device is None:
            raise RuntimeError(
                "engine has no device yet; initialize the canvas before render_to_texture"
            )
        w, h = int(size[0]), int(size[1])
        texture = self._device.create_texture(
            size=(w, h, 1),
            format=self._texture_format,
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_SRC),
        )
        depth = self._device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        sampler = self._device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
        registry = self._texture_registry
        handle = TextureHandle(
            id=registry.alloc_id(), texture=texture, view=texture.create_view(),
            sampler=sampler, size=(w, h),
        )
        registry.add(handle)
        self._views[handle.id] = _View(camera, handle, depth.create_view(), live, True,
                                       _clear_color(background), fog)
        return handle

    def refresh_texture(self, handle) -> None:
        """Render a render_to_texture view again on the next frame."""
        self._views[handle.id].pending = True

    def set_texture_live(self, handle, live: bool) -> None:
        """Render a render_to_texture view every frame (True) or only on request."""
        self._views[handle.id].live = bool(live)

    def drop_texture(self, handle) -> None:
        """Stop rendering a render_to_texture view; its texture keeps the last image."""
        self._views.pop(handle.id, None)

    def _render_views(self) -> None:
        views = [v for v in self._views.values() if v.live or v.pending]
        if not views:
            return
        saved = (self._camera, self.w, self.h, self.fog_enabled)
        try:
            for view in views:
                view.pending = False
                self._camera = view.camera
                self.fog_enabled = saved[3] and view.fog
                self.w, self.h = view.handle.size
                encoder = self._device.create_command_encoder()
                bg = view.background or (*self.background_color, 1.0)
                render_pass = encoder.begin_render_pass(
                    color_attachments=[{
                        "view": view.handle.view,
                        "resolve_target": None,
                        "clear_value": tuple(bg),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    }],
                    depth_stencil_attachment={
                        "view": view.depth_view,
                        "depth_clear_value": 1.0,
                        "depth_load_op": wgpu.LoadOp.clear,
                        "depth_store_op": wgpu.StoreOp.store,
                    },
                )
                self._render_pipeline._render_scene_passes(self, render_pass)
                render_pass.end()
                self._device.queue.submit([encoder.finish()])
        finally:
            self._camera, self.w, self.h, self.fog_enabled = saved

    # One frame is enough: shadows and IBL are complete on the first frame
    # (measured: identical output for 1, 2 and 3 frames, with shadows and with IBL).
    _STILL_FRAMES = 1

    def render_frame(
        self, *, supersample: int = 2, pass_: str = "beauty", groups=None, dt: float = 1 / 30
    ):
        """Render one still frame headless and return it as a numpy array.

        beauty: (h, w, 3) uint8, sRGB, area-downsampled from (h*ss, w*ss).
        Systems run once per beauty frame with a fixed `dt` (unless the engine
        already has a fixed timestep), so an animated scene renders the same
        whatever the wall clock does. The first call replaces the engine's
        canvas with an offscreen one.
        """
        if pass_ not in ("beauty", "ids"):
            raise ValueError(f"pass_ must be 'beauty' or 'ids', got {pass_!r}")
        self._ensure_offscreen(supersample)
        if pass_ == "ids":
            return _block_mode(self._render_ids(groups or []), supersample)
        if not self._use_fixed_dt:
            self.set_fixed_timestep(dt)
        self._settle()
        rgb = self._draw_still()
        if supersample == 1:
            return rgb
        ss = supersample
        small = rgb.reshape(self.h, ss, self.w, ss, 3).mean(axis=(1, 3))
        return np.round(small).astype(np.uint8)

    def export_gltf(self, path, *, names=None, extras=None, strict=False):
        """Write the scene to a glTF binary (.glb) and report what it left out.

        Meshes, materials, textures, the camera, the lights, and fog, shadow,
        background and environment settings (in the scene's extras) are
        written. Systems, event handlers, GUI, compute and materials without a
        glTF equivalent are listed in the returned report, which is also
        printed and saved next to the file. `names` and `extras` map entity
        indices to node names and JSON extras; each may be a dict or a function
        of the index, called after startup so it sees startup-spawned entities. With strict=True an incomplete
        export raises ExportIncomplete and writes nothing.
        """
        from manifoldx.gltf.engine_export import export_engine

        return export_engine(self, path, names=names, extras=extras, strict=strict)

    def _ensure_offscreen(self, supersample: int, gpu=None):
        """gpu is (adapter, device, texture_format) to reuse, as _init_canvas takes it."""
        size = (self.w * supersample, self.h * supersample)
        if self._still_size is None:
            from manifoldx.backends import get_headless_canvas

            self._follow_resize = False
            self._init_canvas(get_headless_canvas(width=size[0], height=size[1]), gpu=gpu)
            self._running = True
            self._fire_startup()
            self._settle()
            self._still_size = size
        elif self._still_size != size:
            first = self._still_size[0] // self.w
            raise ValueError(
                f"render_frame: supersample is fixed at {first} for this engine, got {supersample}"
            )

    def _draw_still(self, simulate: bool = True) -> np.ndarray:
        frame = None
        for _ in range(self._STILL_FRAMES):
            if simulate:
                self._draw_frame()
            else:
                self._render_only(getattr(self, "_last_dt", 0.0) or 0.0)
            frame = self._render_canvas.draw()
        return np.asarray(frame)[:, :, :3].copy()

    def _flat_for_label(self, label: int):
        from manifoldx.resources import FlatMaterial

        if label not in self._id_materials:
            rgb = ((label & 255) / 255, ((label >> 8) & 255) / 255, ((label >> 16) & 255) / 255)
            self._id_materials[label] = FlatMaterial(rgb)
        return self._id_materials[label]

    def _mesh_path_entities(self, alive):
        """Alive entities the mesh pass draws: real geometry, not axis/label/volume."""
        from manifoldx.viz import AxisMaterial, LabelMaterial, VolumeMaterial

        comps = self.store._components
        if "Mesh" not in comps:
            return np.zeros(len(alive), dtype=bool)
        geom = comps["Mesh"][alive, 0]
        mat_ids = comps["Material"][alive, 0]
        special = (AxisMaterial, LabelMaterial, VolumeMaterial)
        # One registry lookup per material, not per entity.
        drop = [m for m in np.unique(mat_ids) if isinstance(self._material_registry.get(int(m)), special)]
        return (geom > 0) & ~np.isin(mat_ids, drop)

    def _render_ids(self, groups):
        """Draw every mesh flat in its group's label colour; restore afterwards.

        Returns (h*ss, w*ss) int32 labels: 0 for background and entities in no
        group, i + 1 for groups[i]; render_frame reduces them to (h, w). Non-mesh
        entities (point clouds, labels, axes, volumes) and the GUI are hidden,
        and no systems run: the labels describe the same state as the last
        beauty frame.
        """
        from manifoldx.gui.widgets import _GuiRoot

        comps = self.store._components
        mats = comps["Material"]
        alive = np.where(self.store._alive)[0]
        meshes = alive[self._mesh_path_entities(alive)]
        others = np.setdiff1d(alive, meshes)
        saved_mats = mats[alive].copy()
        saved_view = (self.background_color, self._environment, self.fog_enabled, self.gui)
        try:
            mats[meshes, 0] = self._material_registry.register(self._flat_for_label(0))
            for label, indices in enumerate(groups, start=1):
                idx = np.intersect1d(np.asarray(list(indices), dtype=np.int64), meshes)
                mats[idx, 0] = self._material_registry.register(self._flat_for_label(label))
            self.store._alive[others] = False  # not drawn by any pass
            self.background_color = (0.0, 0.0, 0.0)
            self._environment = None
            self.fog_enabled = False
            self.gui = _GuiRoot()
            rgb = self._draw_still(simulate=False).astype(np.int32)
        finally:
            mats[alive] = saved_mats
            self.store._alive[others] = True
            self.background_color, self._environment, self.fog_enabled, self.gui = saved_view
        return rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)

    def _pump_aio_loop(self) -> None:
        """Drive the engine's asyncio loop until quiescent (when applicable).

        In headless mode (tests, render-to-MP4), there is no running asyncio
        loop, so we drive our private fallback loop with run_until_complete.
        In interactive run() mode, rendercanvas owns a running asyncio loop
        and our async tasks were created on it; we MUST NOT call
        run_until_complete on a different loop (RuntimeError) or recursively
        on the same one. The running loop will pump our tasks naturally
        between draw callbacks. Either way, we re-raise the first exception
        spooled on _task_errors per the v1 error policy.

        Trade-off documented in v1: in run() mode, an `await engine.tick()`
        coroutine resumes between frames (after this _draw_frame returns,
        before the next is scheduled), introducing up to one frame of
        scheduling delay vs. headless mode where the resume happens inside
        this same frame. The 1-frame difference is imperceptible at 60fps.
        """
        try:
            asyncio.get_running_loop()
            # A loop is running (rendercanvas). It pumps our tasks itself.
        except RuntimeError:
            # No running loop — drive our private fallback.
            if self._aio_loop is not None and not self._aio_loop.is_closed():
                self._aio_loop.run_until_complete(asyncio.sleep(0))

        if self._task_errors:
            exc = self._task_errors.pop(0)
            raise exc

    def _run_loop(self):
        """Called each frame by the event loop (run() mode)."""
        self._tick()
        # Re-queue for next frame if still running
        if self._running:
            self._render_canvas.request_draw(self._run_loop)

    def _tick(self):
        """One interactive frame: startup on the first one, then draw."""
        if self._startup_pending:
            # Inside the running loop, so async startup handlers live on it and load
            # between frames instead of never running (manifoldx#28).
            self._startup_pending = False
            self._fire_startup()
        self._draw_frame()

    def _detach(self):
        """Stop this engine without closing its window, for a hot reload.

        Fires 'shutdown', cancels this engine's async tasks and waiters, unhooks
        its input from the canvas and drops its background workers. The canvas,
        device and asyncio loop stay up for the engine that replaces it.
        """
        self._running = False
        self._event_bus.dispatch_immediate(self, "shutdown", {})
        for task in list(self._tasks):
            task.cancel()
        self._frame_waiters.cancel_all()
        try:
            asyncio.get_running_loop()  # a shared loop runs the cancellations itself
        except RuntimeError:
            if self._aio_loop is not None and not self._aio_loop.is_closed():
                self._aio_loop.run_until_complete(asyncio.sleep(0))  # let cancelled tasks unwind
                self._aio_loop.close()
        if self._render_canvas is not None:
            self._input_bridge.detach(self._render_canvas)
        for executor in (self._executor, self._process_executor):
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def run(self):
        """Run the engine with real-time rendering.

        Automatically uses:
        - PyodideRenderCanvas if running in browser (emscripten)
        - GlfwRenderCanvas otherwise (desktop)

        Note: Use render() for video output instead.
        """
        # Import backends module for lazy canvas creation
        from manifoldx.backends import get_desktop_canvas

        # Create canvas (GLFW on desktop, handled by backends module)
        canvas = get_desktop_canvas(
            width=self.w,
            height=self.h,
            fullscreen=self.fullscreen,
            title=self.title,
            monitor=self.monitor,
            max_fps=self.max_fps,
        )

        # Initialize WebGPU context
        self._init_canvas(canvas)

        self._running = True

        # 'startup' fires on the first frame, inside the running loop (see _run_loop).
        self._startup_pending = True

        # Register draw callback and run the canvas event loop
        from rendercanvas.glfw import loop as glfw_loop

        self._event_loop = glfw_loop
        self._render_canvas.request_draw(self._run_loop)
        try:
            glfw_loop.run()
        except Exception as e:
            print(f"Event loop error: {e}")
        finally:
            self._running = False
            self.shutdown_events()

    def render(
        self,
        output: str,
        *,
        fps: int = 30,
        duration: float | None = None,
        frame_count: int | None = None,
        codec: str = "h264",
        quality: str = "high",
        progress: bool = True,
    ):
        """Render to video file.

        Uses OffscreenRenderCanvas for headless rendering.

        Parameters
        ----------
        output : str
            Output video file path (required).
        fps : int
            Frames per second (default: 30).
        duration : float
            Duration in seconds (alternative to frame_count).
        frame_count : int
            Number of frames to render (alternative to duration).
        codec : str
            Video codec: "h264", "hevc", or "mp4v" (default: "h264").
        quality : str
            Quality preset: "low", "medium", "high" (default: "high").
        progress : bool
            Show progress bar (default: True).
        """
        # Validate parameters
        if output is None:
            raise ValueError("output path required for render()")
        if duration is None and frame_count is None:
            raise ValueError("Either duration or frame_count must be specified")

        # Calculate frame count from duration if needed
        if frame_count is None:
            frame_count = int(duration * fps)

        # Use fixed timestep for video rendering. This has to go through
        # set_fixed_timestep: otherwise _compute_dt() takes the wall-clock
        # branch and overwrites self.elapsed every frame, so scripted camera
        # paths get paced by how fast the host renders instead of by the
        # video timeline.
        dt = 1.0 / fps
        self.set_fixed_timestep(dt)

        # Import backends module for lazy canvas creation
        from manifoldx.backends import get_offscreen_canvas

        # Create offscreen canvas for headless rendering
        canvas = get_offscreen_canvas(width=self.w, height=self.h)

        # Initialize WebGPU context
        self._init_canvas(canvas)

        self._running = True

        # Set up video writer with imageio-ffmpeg
        writer = self._get_video_writer(output, fps, self.w, self.h, codec, quality)

        # Fire built-in 'startup' event before the first frame, and wait for what it loads.
        self._fire_startup()

        # Progress tracking
        pbar = None
        if progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=frame_count, desc="Rendering video", unit="frames")
            except ImportError:
                pass

        self._settle()

        # === RENDER LOOP ===
        for frame_idx in range(frame_count):
            # Draw the frame (backend-agnostic)
            self._draw_frame()

            # Capture frame from canvas
            frame = self._render_canvas.draw()

            # Convert RGBA to RGB (most video codecs don't support alpha)
            frame_rgb = frame[:, :, :3].copy()  # Make contiguous

            # Write to video
            writer.send(frame_rgb)

            # Update progress
            if pbar:
                pbar.update(1)

            # Update elapsed time
            self.elapsed = (frame_idx + 1) * dt

        # Clean up
        writer.close()
        writer = None  # type: ignore
        if pbar:
            pbar.close()

        self._running = False

        # Fire 'shutdown' and tear down the asyncio loop cleanly so
        # long-running async handlers don't leak as pending tasks.
        self.shutdown_events()

        print(f"Video saved: {output}")

    def _get_video_writer(self, output, fps, width, height, codec, quality):
        """Set up imageio-ffmpeg video writer."""
        import imageio_ffmpeg

        # Quality presets (1-10 scale for imageio-ffmpeg)
        quality_map = {
            "low": 1,
            "medium": 3,
            "high": 5,
            "ultra": 8,
        }
        quality_value = quality_map.get(quality, 5)

        # Get writer with ffmpeg
        writer = imageio_ffmpeg.write_frames(
            output,
            size=(width, height),
            fps=fps,
            codec="libx264" if codec == "h264" else codec,
            quality=quality_value,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
        )

        # Seed the generator
        writer.send(None)

        return writer

    def cli(
        self,
        *,
        fps: int = 30,
        duration: float = 60,
        output: str | None = None,
        quality: str = "high",
    ) -> None:
        """Command-line interface for the engine.

        Usage:
            python example.py           # Interactive window
            python example.py --render  # Render 60s video
            python example.py --render --fps 60 --duration 120
            python example.py --render --output custom.mp4

        Parameters
        ----------
        fps : int
            Frames per second (default: 30).
        duration : float
            Video duration in seconds (default: 60).
        output : str | None
            Output filename. If None, inferred from script name (default).
        quality : str
            Video quality: "low", "medium", "high" (default: "high").
        """
        parser = argparse.ArgumentParser(description=self.title)
        parser.add_argument(
            "--render",
            action="store_true",
            help="Render to video instead of showing window",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=fps,
            help=f"Frames per second (default: {fps})",
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=duration,
            help=f"Video duration in seconds (default: {duration})",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output filename (default: <script>.mp4)",
        )
        parser.add_argument(
            "--quality",
            type=str,
            default=quality,
            choices=["low", "medium", "high"],
            help=f"Video quality (default: {quality})",
        )

        args = parser.parse_args()

        if args.render:
            # Infer output filename from script name
            if args.output is None:
                script_path = Path(sys.argv[0]).resolve()
                output_path = script_path.with_suffix(".mp4")
            else:
                output_path = args.output

            self.render(
                output=str(output_path),
                fps=args.fps,
                duration=args.duration,
                quality=args.quality,
            )
        else:
            self.run()
