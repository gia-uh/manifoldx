"""Pipelined frames (PR D): frame N renders while frame N+1's systems run.

These tests drive `Engine._draw_frame` directly on an offscreen canvas with the
pipeline forced on (`engine._pipeline_on = True` after the canvas exists). The
window loop sets that flag in `run()` on a free-threaded build; the offscreen
canvas is only here so a test can step frames without a window. The first frame
is serial, as in a run: its systems have to run before anything can render.
"""

import threading
import time

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, cube
from manifoldx.systems import _gil_is_on
from manifoldx.types import Float

pytestmark = pytest.mark.skipif(
    _gil_is_on(), reason="pipelined frames need the free-threaded build (3.14t)"
)


# --- helpers -----------------------------------------------------------------


def _engine(name="pipelined", **kwargs):
    engine = mx.Engine(name, width=64, height=64, **kwargs)
    try:
        engine._ensure_offscreen(1)
    except Exception as error:  # pragma: no cover - no GPU on this machine
        engine.close()
        pytest.skip(f"offscreen canvas unavailable: {error}")
    engine.set_fixed_timestep(1 / 60)
    engine._pipeline_on = True  # what run() sets on a free-threaded build
    engine._pipeline_frames = 0  # the first frame is serial
    return engine


def _scene(engine, n=8):
    """n cubes, spaced out, with one shared material."""
    material = StandardMaterial("#88aaff")
    positions = np.column_stack(
        [np.arange(n, dtype=np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)]
    )
    engine.spawn(
        Mesh(cube(0.5, 0.5, 0.5)),
        Material(material),
        Transform(pos=positions),
        n=n,
    )


def _store_bytes(engine):
    """Every array the store holds, for a frame-by-frame comparison."""
    store = engine.store
    return (
        store._alive.tobytes(),
        tuple((name, array.tobytes()) for name, array in sorted(store._components.items())),
    )


def _wrap_render(engine, seen):
    """Record the store the render is about to read, then render as usual."""
    original = engine._render_only

    def render_only(dt):
        seen.append(_store_bytes(engine))
        return original(dt)

    engine._render_only = render_only


class _DeviceCalls:
    """A device proxy that records the thread of every mutating call."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def note(self, name):
        with self._lock:
            self.calls.append((threading.get_ident(), name))

    def threads(self):
        return {ident for ident, _ in self.calls}


class _QueueProxy:
    def __init__(self, queue, log):
        self._queue, self._log = queue, log

    def __getattr__(self, name):
        return getattr(self._queue, name)

    def write_buffer(self, *args, **kwargs):
        self._log.note("write_buffer")
        return self._queue.write_buffer(*args, **kwargs)

    def write_texture(self, *args, **kwargs):
        self._log.note("write_texture")
        return self._queue.write_texture(*args, **kwargs)

    def submit(self, *args, **kwargs):
        self._log.note("submit")
        return self._queue.submit(*args, **kwargs)


class _DeviceProxy:
    def __init__(self, device, log):
        self._device, self._log = device, log
        self.queue = _QueueProxy(device.queue, log)

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __getitem__(self, key):
        return self._device[key]

    def create_buffer(self, *args, **kwargs):
        self._log.note("create_buffer")
        return self._device.create_buffer(*args, **kwargs)

    def create_texture(self, *args, **kwargs):
        self._log.note("create_texture")
        return self._device.create_texture(*args, **kwargs)

    def create_shader_module(self, *args, **kwargs):
        self._log.note("create_shader_module")
        return self._device.create_shader_module(*args, **kwargs)

    def create_bind_group(self, *args, **kwargs):
        self._log.note("create_bind_group")
        return self._device.create_bind_group(*args, **kwargs)

    def create_bind_group_layout(self, *args, **kwargs):
        self._log.note("create_bind_group_layout")
        return self._device.create_bind_group_layout(*args, **kwargs)

    def create_pipeline_layout(self, *args, **kwargs):
        self._log.note("create_pipeline_layout")
        return self._device.create_pipeline_layout(*args, **kwargs)

    def create_compute_pipeline(self, *args, **kwargs):
        self._log.note("create_compute_pipeline")
        return self._device.create_compute_pipeline(*args, **kwargs)

    def create_render_pipeline(self, *args, **kwargs):
        self._log.note("create_render_pipeline")
        return self._device.create_render_pipeline(*args, **kwargs)


def _record_device_calls(engine):
    log = _DeviceCalls()
    real = engine._device
    proxy = _DeviceProxy(real, log)
    engine._device = proxy
    registries = (engine._geometry_registry, engine._material_registry, engine._volume_registry)
    for registry in registries:
        registry._device = proxy
    engine._render_pipeline._device = proxy

    def undo():
        engine._device = real
        for registry in registries:
            registry._device = real
        engine._render_pipeline._device = real

    return log, undo


# --- equivalence -------------------------------------------------------------


def test_pipelined_store_matches_serial_frame_by_frame():
    """100 pipelined frames equal 100 serial ones, and the render sees one behind."""
    serial = _engine("serial")
    serial._pipeline_on = False  # the baseline: systems then render, every frame
    pipelined = _engine("pipelined")
    try:
        seen_serial, seen_pipelined = [], []
        for engine, seen in ((serial, seen_serial), (pipelined, seen_pipelined)):
            _scene(engine)

            @engine.system
            def drift(q: mx.Query[Transform], dt: float):
                q[Transform].position += np.array([0.05, 0.0, 0.0], np.float32)

            _wrap_render(engine, seen)

        for _ in range(100):
            serial._draw_frame()
            pipelined._draw_frame()
            assert _store_bytes(pipelined) == _store_bytes(serial)

        # The pipelined render at frame k draws what serial drew at k-1: its
        # first frame is serial, so only that one is not shifted.
        assert seen_pipelined[0] == seen_serial[0]
        assert seen_pipelined[1:] == seen_serial[:-1]
    finally:
        serial.close()
        pipelined.close()


# --- read-your-writes --------------------------------------------------------


def test_a_system_sees_the_old_camera_pose_this_frame():
    """A camera write defers to the apply step: the render and the system see the old pose."""
    engine = _engine("camera")
    seen, rendered = [], []
    try:
        _scene(engine)

        @engine.system
        def aim(q: mx.Query[Transform], dt: float):
            seen.append(tuple(float(v) for v in engine.camera.position))
            engine.camera.position = np.array(
                [float(engine._frame_index), 3.0, 4.0], np.float32
            )

        original = engine._render_only

        def render_only(dt):
            rendered.append(tuple(float(v) for v in engine.camera.position))
            return original(dt)

        engine._render_only = render_only

        engine._draw_frame()  # serial first frame
        assert seen[0] == (0.0, 1.0, 2.0)
        assert tuple(float(v) for v in engine.camera.position) == (0.0, 3.0, 4.0)

        engine._draw_frame()  # pipelined
        assert seen[1] == (0.0, 3.0, 4.0)  # last frame's pose, not this frame's write
        assert rendered[1] == (0.0, 3.0, 4.0)  # the render reads the old pose too
        assert tuple(float(v) for v in engine.camera.position) == (1.0, 3.0, 4.0)

        engine._draw_frame()
        assert seen[2] == (1.0, 3.0, 4.0)  # the previous write is visible next frame
    finally:
        engine.close()


def test_frame_split_has_overlap_and_flat_floats():
    """`frame_split` gains the seconds the render and the systems ran together."""
    engine = _engine("split")
    try:
        _scene(engine)

        @engine.system
        def slow(q: mx.Query[Transform], dt: float):
            deadline = time.perf_counter() + 0.02
            while time.perf_counter() < deadline:
                pass

        engine._draw_frame()  # serial
        engine._draw_frame()  # pipelined
        split = engine.frame_split
        assert set(split) == {"update", "acquire", "render", "overlap"}
        assert all(isinstance(value, float) for value in split.values())
        assert split["overlap"] > 0.0
    finally:
        engine.close()


def test_headless_render_frame_is_serial_even_when_pipelined():
    """render_frame takes the serial path whatever the constructor says."""
    engine = mx.Engine("headless", width=32, height=32, pipelined=True)
    seen = []
    try:
        _scene(engine, n=2)

        @engine.system
        def where(q: mx.Query[Transform], dt: float):
            seen.append(threading.get_ident())

        engine._ensure_offscreen(1)
        engine._pipeline_on = True  # even forced on, the still is serial
        engine.render_frame(supersample=1)
        assert seen
        assert set(seen) == {threading.main_thread().ident}
        assert engine.frame_split["overlap"] == 0.0
    finally:
        engine.close()


# --- check mode --------------------------------------------------------------


def test_check_mode_catches_a_direct_array_write_from_a_system(monkeypatch):
    """A row written by hand while the render runs fails the frame, naming the store."""
    monkeypatch.setenv("MANIFOLDX_PIPELINE", "check")
    engine = _engine("check")
    render_started = threading.Event()
    try:
        _scene(engine)

        @engine.system
        def poke(q: mx.Query[Transform], dt: float):
            if engine._pipeline_frames:
                render_started.wait(timeout=5.0)
            engine.store._components["Transform"][0, 0] += np.float32(1.0)

        engine._draw_frame()  # serial: the write lands before any render starts

        original_run = engine._render_pipeline.run

        def slow_run(engine_, dt):
            render_started.set()
            time.sleep(0.05)  # leave the system time to write while the render runs
            return original_run(engine_, dt)

        engine._render_pipeline.run = slow_run
        with pytest.raises(
            RuntimeError,
            match="pipelined frame: the entity store changed while the render read it",
        ):
            engine._draw_frame()
    finally:
        engine.close()


def test_check_mode_allows_the_first_spawn_of_a_new_component_type(monkeypatch):
    """Registering a new component in a system is setup, not a render-state write."""
    monkeypatch.setenv("MANIFOLDX_PIPELINE", "check")
    engine = _engine("check-spawn")
    render_started = threading.Event()

    class PipelineGapMarker(Component):
        value: Float = 1.0

    try:
        _scene(engine)

        @engine.system
        def spawn_one(q: mx.Query[Transform], dt: float):
            if not engine._pipeline_frames:
                return  # only the pipelined frame spawns
            render_started.wait(timeout=5.0)
            engine.spawn(PipelineGapMarker())

        engine._draw_frame()  # serial
        original_run = engine._render_pipeline.run

        def slow_run(engine_, dt):
            render_started.set()
            time.sleep(0.05)
            return original_run(engine_, dt)

        engine._render_pipeline.run = slow_run
        engine._draw_frame()  # pipelined: no spurious check failure
        assert engine.store._components["PipelineGapMarker"].sum() == 1.0
    finally:
        engine.close()


# --- errors and the device thread -------------------------------------------


def test_a_system_exception_surfaces_on_the_main_thread_with_its_name():
    engine = _engine("raises")
    try:
        _scene(engine)

        @engine.system
        def explode(q: mx.Query[Transform], dt: float):
            if engine._pipeline_frames:
                raise ValueError("boom")

        engine._draw_frame()  # serial, no raise
        with pytest.raises(ValueError, match="boom") as caught:
            engine._draw_frame()
        assert threading.get_ident() == threading.main_thread().ident
        assert any("explode" in note for note in getattr(caught.value, "__notes__", []))
    finally:
        engine.close()


def test_no_device_call_runs_on_the_systems_thread():
    """A system that spawns and destroys defers every device call to the main thread."""
    engine = _engine("device-thread")
    try:
        _scene(engine)

        @engine.system
        def churn(q: mx.Query[Transform], dt: float):
            handle = engine.spawn(
                Mesh(cube(0.25, 0.25, 0.25)),
                Material(StandardMaterial("#ffcc88")),
                Transform(pos=(5.0, 1.0, 5.0)),
            )
            handle.transform.pos = (6.0, 1.0, 5.0)
            handle.destroy()

        engine._draw_frame()  # serial
        log, undo = _record_device_calls(engine)
        try:
            for _ in range(5):
                engine._draw_frame()
        finally:
            undo()
        assert log.calls, "the pipeline made no device calls at all"
        assert log.threads() == {threading.main_thread().ident}
    finally:
        engine.close()
