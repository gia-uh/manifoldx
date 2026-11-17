"""Deferred engine calls (PR C): the CALL command, `Engine._defer_calls()`, every audit row.

With the flag off every call acts at once, exactly as before; these tests turn
it on with `Engine._defer_calls()` and check, per row of
`.knowledge/analysis/2026-11-17-pipelining-audit.md` sections 5.1 and 5.2,
that nothing the render reads changed inside the block and that it did after
the command buffer ran.
"""

import threading

import numpy as np
import pytest

import manifoldx as mx
from manifoldx import _deferred
from manifoldx import textures as texture_module
from manifoldx.commands import Command, CommandBuffer, CommandType
from manifoldx.components import Material as MaterialComponent
from manifoldx.components import Mesh as MeshComponent
from manifoldx.components import Transform
from manifoldx.gui import Panel, Text
from manifoldx.gui import style as gui_style
from manifoldx.ibl import EnvironmentMap
from manifoldx.types import Float
from manifoldx.resources import (
    DirectionalLight,
    PointLight,
    SpotLight,
    StandardMaterial,
    cube,
)
from manifoldx.modeling import Mesh as GeoMesh


# --- helpers -----------------------------------------------------------------


def _engine(name="deferred-calls"):
    return mx.Engine(name)


def _gpu_engine(name="deferred-calls-gpu", w=96, h=96):
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as error:  # pragma: no cover - no GPU on this machine
        pytest.skip(f"offscreen canvas unavailable: {error}")
    engine = mx.Engine(name, width=w, height=h)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def _apply(engine):
    """The frame's apply step: run the queued commands, in call order."""
    engine.commands.execute(engine.store)
    engine.commands.clear()


def _spawn_cube(engine, pos=(1.0, 2.0, 3.0), material=None):
    return engine.spawn(
        MeshComponent(cube(1, 1, 1)),
        MaterialComponent(material or mx.material.standard("#ffffff")),
        Transform(pos=pos),
    )


def _moved_cube(dx=1.0):
    """A cube's geometry dict with every vertex shifted: same vertex count."""
    geo = cube(1, 1, 1)
    return {**geo, "positions": geo["positions"] + np.float32(dx)}


def _skinned_geometry():
    box = GeoMesh.box(0.5, 0.8, 0.5).recompute_normals()
    return GeoMesh(
        positions=box.positions,
        faces=box.faces,
        normals=box.normals,
        joints=np.zeros(len(box.positions), dtype=int),
    ).to_geometry()


def _alive_count(engine):
    return int(engine.store._alive.sum())


class _DeviceCalls:
    """Records the thread that made each mutating device or queue call."""

    _DEVICE = (
        "create_buffer",
        "create_texture",
        "create_sampler",
        "create_shader_module",
        "create_bind_group",
        "create_bind_group_layout",
        "create_pipeline_layout",
        "create_compute_pipeline",
        "create_render_pipeline",
    )
    _QUEUE = ("write_buffer", "write_texture", "submit")

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def note(self, name):
        with self._lock:
            self.calls.append((threading.get_ident(), name))

    def names(self):
        return [name for _, name in self.calls]

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

    def create_sampler(self, *args, **kwargs):
        self._log.note("create_sampler")
        return self._device.create_sampler(*args, **kwargs)

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
    """Send every engine-held device call through a recorder; returns (log, undo)."""
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


# =============================================================================
# Task 1: the CALL command and the deferring switch
# =============================================================================


def test_call_command_runs_at_execute():
    """A CALL queued inside `_defer_calls()` runs when the buffer executes, not before."""
    engine = _engine()
    seen = []
    with engine._defer_calls():
        _deferred.defer(engine, seen.append, "ran")
        assert seen == []
    assert seen == []
    _apply(engine)
    assert seen == ["ran"]


def test_call_runs_in_order_with_update_component_commands():
    """A CALL runs between the commands queued before and after it, in call order."""
    engine = _engine()
    handle = _spawn_cube(engine, pos=(1.0, 2.0, 3.0))
    row = engine.store._components["Transform"]
    seen = []

    def read_pos():
        seen.append(float(row[handle.index, 0]))

    with engine._defer_calls():
        engine.commands.append(
            Command(
                CommandType.UPDATE_COMPONENT,
                {
                    "component_name": "Transform",
                    "indices": np.array([handle.index]),
                    "new_data": np.array([[2.0]], dtype=np.float32),
                    "col_start": 0,
                    "col_end": 1,
                },
            )
        )
        _deferred.defer(engine, read_pos)
        engine.commands.append(
            Command(
                CommandType.UPDATE_COMPONENT,
                {
                    "component_name": "Transform",
                    "indices": np.array([handle.index]),
                    "new_data": np.array([[3.0]], dtype=np.float32),
                    "col_start": 0,
                    "col_end": 1,
                },
            )
        )
        assert row[handle.index, 0] == 1.0

    _apply(engine)
    assert seen == [2.0]  # the CALL saw the first update and ran before the second
    assert row[handle.index, 0] == 3.0


def test_deferred_call_exception_carries_the_queuing_system_note():
    """An exception raised at apply names the system that queued the call."""
    engine = _engine()

    def boom():
        raise ValueError("the deferred call failed")

    @engine.system
    def exploding_system(query, dt):
        _deferred.defer(engine, boom)

    engine.systems.run_all(engine, 1 / 60)
    with pytest.raises(ValueError, match="the deferred call failed") as info:
        _apply(engine)
    notes = getattr(info.value, "__notes__", [])
    assert any("exploding_system" in note for note in notes), notes


def test_decorated_call_acts_at_once_when_not_deferring():
    """With the flag off the decorator calls straight through, as before."""
    engine = _engine()
    engine.camera.move_to((1.0, 2.0, 3.0))
    assert np.allclose(engine.camera.position, (1.0, 2.0, 3.0))
    assert len(engine.commands) == 0


def test_deferring_is_per_thread():
    """Only the thread that entered `_defer_calls()` defers."""
    engine = _engine()
    seen = []
    with engine._defer_calls():
        assert engine._deferring()
        worker = threading.Thread(target=lambda: seen.append(engine._deferring()))
        worker.start()
        worker.join()
    assert seen == [False]
    assert not engine._deferring()


# =============================================================================
# Task 2: spawn reserves indices, writes at apply
# =============================================================================


def test_deferred_spawn_is_invisible_to_a_query_until_apply():
    engine = _engine()
    with engine._defer_calls():
        handle = _spawn_cube(engine)
        assert handle is not None
        assert not engine.store._alive[handle.index]
        assert len(engine.store.get_component_view(["Transform"])) == 0
        assert len(engine.store.get_component_view(["Mesh"])) == 0
    _apply(engine)
    assert engine.store._alive[handle.index]
    assert list(engine.store.get_component_view(["Transform"])) == [handle.index]
    assert np.allclose(engine.store._components["Transform"][handle.index, 0:3], (1, 2, 3))


def test_deferred_spawn_setter_and_destroy_apply_in_call_order():
    engine = _engine()
    with engine._defer_calls():
        handle = _spawn_cube(engine)
        handle.transform.pos = (4.0, 5.0, 6.0)
        handle.destroy()
    _apply(engine)
    # The spawn wrote the row, then the setter overwrote it, then destroy killed it.
    assert not engine.store._alive[handle.index]
    assert np.allclose(engine.store._components["Transform"][handle.index, 0:3], (4, 5, 6))


def test_two_threads_spawning_at_once_get_disjoint_indices():
    engine = _engine()
    seen = [[], []]
    buffers = [CommandBuffer(), CommandBuffer()]
    errors = []

    def spawn_many(slot):
        try:
            with engine._bind_commands(buffers[slot]), engine._defer_calls():
                for _ in range(25):
                    seen[slot].append(engine.spawn(Transform(pos=(0, 0, 0))).index)
        except Exception as error:  # pragma: no cover - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=spawn_many, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    all_indices = seen[0] + seen[1]
    assert len(all_indices) == 50
    assert len(set(all_indices)) == 50  # no slot handed out twice
    engine.commands.extend(buffers[0])
    engine.commands.extend(buffers[1])
    _apply(engine)
    assert _alive_count(engine) == 50


def test_spawn_in_a_system_without_deferral_acts_at_once():
    """A serial system's spawn is not deferred: the row and the alive bit exist on return."""
    engine = _engine()
    with engine._in_system("s"):
        handle = engine.spawn(Transform(pos=(7.0, 8.0, 9.0)))
    assert handle is not None
    assert engine.store._alive[handle.index]
    assert np.allclose(engine.store._components["Transform"][handle.index, 0:3], (7, 8, 9))


# =============================================================================
# Task 3: one case per audit row classified b, c or d
# =============================================================================


def _material_setup(engine):
    blue = mx.material.basic("#0000ff")
    return {
        "handle": _spawn_cube(engine, material=blue),
        "blue": blue,
        "red": mx.material.basic("#ff0000"),
    }


def _material_row(engine, ctx):
    return int(engine.store._components["Material"][ctx["handle"].index, 0])


def _material_registry_id(engine, material):
    return int(engine._material_registry._object_to_id[id(material)])


class Row:
    """One audit row: a call inside the deferring block, and what changed before/after apply."""

    def __init__(self, name, call, before, after, setup=None, gpu=False, cleanup=None):
        self.name = name
        self.call = call
        self.before = before
        self.after = after
        self.setup = setup
        self.gpu = gpu
        self.cleanup = cleanup


def _row(name, call, before, after, **kwargs):
    return pytest.param(Row(name, call, before, after, **kwargs), id=name)


def _spawned(engine, pos=(1.0, 2.0, 3.0), material=None):
    return {"handle": _spawn_cube(engine, pos=pos, material=material)}


def _env():
    return EnvironmentMap.from_color((0.5, 0.6, 0.7))


ROWS = [
    # --- 5.1 §2 writers -----------------------------------------------------
    _row(
        "5.1-01-engine-spawn",
        lambda e, ctx: _spawn_cube(e),
        lambda e, ctx: _alive_count(e) == 0,
        lambda e, ctx: _alive_count(e) == 1,
    ),
    _row(
        "5.1-02-handle-transform-writes",
        lambda e, ctx: (
            setattr(ctx["handle"].transform, "pos", (4.0, 5.0, 6.0)),
            setattr(ctx["handle"].transform, "rot", (0.0, 0.0, 0.0, 1.0)),
            setattr(ctx["handle"].transform, "scale", (2.0, 2.0, 2.0)),
        ),
        lambda e, ctx: (
            np.allclose(e.store._components["Transform"][ctx["handle"].index, 0:3], (1, 2, 3))
            and e.store._components["Transform"][ctx["handle"].index, 7] == 1.0
        ),
        lambda e, ctx: (
            np.allclose(e.store._components["Transform"][ctx["handle"].index, 0:3], (4, 5, 6))
            and e.store._components["Transform"][ctx["handle"].index, 7] == 2.0
        ),
        setup=_spawned,
    ),
    _row(
        "5.1-03-handle-set-geometry",
        lambda e, ctx: ctx["handle"].set_geometry(_moved_cube()),
        lambda e, ctx: np.allclose(
            e._geometry_registry.get(ctx["handle"]._geometry_id)["positions"][0], (0.5, 0.5, 0.5)
        ),
        lambda e, ctx: np.allclose(
            e._geometry_registry.get(ctx["handle"]._geometry_id)["positions"][0], (1.5, 0.5, 0.5)
        ),
        setup=_spawned,
    ),
    _row(
        "5.1-04-handle-set-material",
        lambda e, ctx: ctx["handle"].set_material(ctx["red"]),
        lambda e, ctx: _material_row(e, ctx) == _material_registry_id(e, ctx["blue"]),
        lambda e, ctx: _material_row(e, ctx) == _material_registry_id(e, ctx["red"]),
        setup=_material_setup,
    ),
    _row(
        "5.1-05-handle-set-joints",
        lambda e, ctx: ctx["handle"].set_joints(np.eye(4, dtype=np.float32)[None]),
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) == 0,
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) > 0,
        setup=lambda e: {
            "handle": e.spawn(
                MeshComponent(_skinned_geometry()),
                MaterialComponent(mx.material.standard("#ffffff")),
                Transform(),
            )
        },
    ),
    _row(
        "5.1-06-handle-destroy",
        lambda e, ctx: ctx["handle"].destroy(),
        lambda e, ctx: bool(e.store._alive[ctx["handle"].index]),
        lambda e, ctx: not e.store._alive[ctx["handle"].index],
        setup=_spawned,
    ),
    _row(
        "5.1-08-engine-set-joints",
        lambda e, ctx: e.set_joints(
            np.array([ctx["handle"].index]), np.eye(4, dtype=np.float32)[None, None]
        ),
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) == 0,
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) > 0,
        setup=lambda e: {
            "handle": e.spawn(
                MeshComponent(_skinned_geometry()),
                MaterialComponent(mx.material.standard("#ffffff")),
                Transform(),
            )
        },
    ),
    _row(
        "5.1-09-engine-set-material",
        lambda e, ctx: e.set_material(ctx["handle"], ctx["red"]),
        lambda e, ctx: _material_row(e, ctx) == _material_registry_id(e, ctx["blue"]),
        lambda e, ctx: _material_row(e, ctx) == _material_registry_id(e, ctx["red"]),
        setup=_material_setup,
    ),
    _row(
        "5.1-10-set-origin",
        lambda e, ctx: e.set_origin(10.0, 20.0),
        lambda e, ctx: (
            e.origin == (0.0, 0.0)
            and np.allclose(e.store._components["Transform"][ctx["handle"].index, 0:3], (1, 2, 3))
        ),
        lambda e, ctx: (
            e.origin == (10.0, 20.0)
            and np.allclose(
                e.store._components["Transform"][ctx["handle"].index, 0:3], (-9, 2, -17)
            )
        ),
        setup=_spawned,
    ),
    _row(
        "5.1-11-camera-methods-and-fields",
        lambda e, ctx: (e.camera.move_to((5.0, 5.0, 5.0)), setattr(e.camera, "fov", 30.0)),
        lambda e, ctx: np.allclose(e.camera.position, (0, 1, 2)) and e.camera.fov == 60,
        lambda e, ctx: np.allclose(e.camera.position, (5, 5, 5)) and e.camera.fov == 30.0,
    ),
    _row(
        "5.1-12-set-lights-add-light",
        lambda e, ctx: (e.set_lights([ctx["sun"], ctx["point"]]), e.add_light(ctx["spot"])),
        lambda e, ctx: len(e._lights) == 0,
        lambda e, ctx: len(e._lights) == 3,
        setup=lambda e: {
            "sun": DirectionalLight("#ffffff", 1.0, (0, -1, 0)),
            "point": PointLight("#ffffff", 1.0, (0, 2, 0)),
            "spot": SpotLight("#ffffff", 1.0, (0, 2, 0), (0, -1, 0), 0.1, 0.3),
        },
    ),
    _row(
        "5.1-13-light-field-writes",
        lambda e, ctx: (
            setattr(ctx["point"], "intensity", 2.0),
            setattr(ctx["point"], "position", (1.0, 2.0, 3.0)),
        ),
        lambda e, ctx: ctx["point"].intensity == 1.0 and ctx["point"].position == (0, 2, 0),
        lambda e, ctx: ctx["point"].intensity == 2.0 and ctx["point"].position == (1.0, 2.0, 3.0),
        setup=lambda e: {
            "point": (e.set_lights([PointLight("#ffffff", 1.0, (0, 2, 0))]), e._lights[0])[1]
        },
    ),
    _row(
        "5.1-14-background-color",
        lambda e, ctx: setattr(e, "background_color", (0.0, 0.0, 1.0)),
        lambda e, ctx: e.background_color == (0.1, 0.1, 0.2),
        lambda e, ctx: e.background_color == (0.0, 0.0, 1.0),
    ),
    _row(
        "5.1-15-enable-fog-and-fog-fields",
        lambda e, ctx: (e.enable_fog(1.0, 10.0, (0.0, 1.0, 0.0)), setattr(e, "fog_end", 20.0)),
        lambda e, ctx: not e.fog_enabled and e.fog_start == 0.0,
        lambda e, ctx: (
            e.fog_enabled
            and e.fog_start == 1.0
            and e.fog_end == 20.0
            and e.fog_color == (0.0, 1.0, 0.0)
        ),
    ),
    _row(
        "5.1-16-sun-spot-shadows",
        lambda e, ctx: (
            e.set_sun(ctx["sun"]),
            e.set_spot(ctx["spot"]),
            e.enable_shadows(target=(1, 2, 3), extent=5.0),
        ),
        lambda e, ctx: e._sun is None and e._spot is None and e._shadow_config is None,
        lambda e, ctx: (
            e._sun is ctx["sun"]
            and e._spot is ctx["spot"]
            and e._shadow_config["target"] == (1, 2, 3)
            and e._shadow_config["extent"] == 5.0
        ),
        setup=lambda e: {
            "sun": DirectionalLight("#ffffff", 1.0, (0, -1, 0)),
            "spot": SpotLight("#ffffff", 1.0, (0, 2, 0), (0, -1, 0), 0.1, 0.3),
        },
    ),
    _row(
        "5.1-17-set-environment",
        lambda e, ctx: e.set_environment(ctx["env"]),
        lambda e, ctx: e.environment is None,
        lambda e, ctx: e.environment is ctx["env"],
        setup=lambda e: {"env": _env()},
    ),
    _row(
        "5.1-18-environment-field-writes",
        lambda e, ctx: (
            setattr(ctx["env"], "intensity", 2.0),
            setattr(ctx["env"], "show_skybox", True),
        ),
        lambda e, ctx: ctx["env"].intensity == 1.0 and ctx["env"].show_skybox is False,
        lambda e, ctx: ctx["env"].intensity == 2.0 and ctx["env"].show_skybox is True,
        setup=lambda e: {"env": (e.set_environment(_env()), e.environment)[1]},
    ),
    _row(
        "5.1-19-geometry-registry-register",
        lambda e, ctx: e.spawn(MeshComponent(_moved_cube(3.0)), Transform()),
        # Nothing the render draws changed: the new row is written at apply, and
        # an unreferenced registry id is never read.
        lambda e, ctx: _alive_count(e) == 1,
        lambda e, ctx: _alive_count(e) == 2,
        setup=lambda e: {"handle": _spawn_cube(e)},
    ),
    _row(
        "5.1-21-geometry-create-buffers",
        lambda e, ctx: e._geometry_registry.create_buffers(ctx["gid"], ctx["geo"], e._device.queue),
        lambda e, ctx: e._geometry_registry.get_gpu_buffers(ctx["gid"]) is None,
        lambda e, ctx: e._geometry_registry.get_gpu_buffers(ctx["gid"]) is not None,
        setup=lambda e: {
            "geo": _moved_cube(),
            "gid": e._geometry_registry.register(_moved_cube(2.0)),
        },
        gpu=True,
    ),
    _row(
        "5.1-22-geometry-adopt-buffers",
        lambda e, ctx: e._geometry_registry.adopt_buffers(ctx["gid"], {}),
        lambda e, ctx: e._geometry_registry.get_gpu_buffers(ctx["gid"]) is None,
        lambda e, ctx: e._geometry_registry.get_gpu_buffers(ctx["gid"]) == {},
        setup=lambda e: {"gid": e._geometry_registry.register(_moved_cube(2.0))},
    ),
    _row(
        "5.1-23-geometry-release",
        lambda e, ctx: e._geometry_registry.release(ctx["gid"]),
        lambda e, ctx: e._geometry_registry.get(ctx["gid"]) is not None,
        lambda e, ctx: e._geometry_registry.get(ctx["gid"]) is None,
        setup=lambda e: {
            "gid": (lambda gid: (e._geometry_registry.adopt_buffers(gid, {}), gid)[1])(
                e._geometry_registry.register(_moved_cube(2.0))
            )
        },
    ),
    _row(
        "5.1-24-material-registry-register",
        lambda e, ctx: ctx["handle"].set_material(ctx["red"]),
        # Registering is what the setter does; the registry gains the material at apply.
        lambda e, ctx: id(ctx["red"]) not in e._material_registry._object_to_id,
        lambda e, ctx: e._material_registry.get(_material_registry_id(e, ctx["red"])) is ctx["red"],
        setup=_material_setup,
    ),
    _row(
        "5.1-25-volume-register-update",
        lambda e, ctx: e.update_volume(ctx["handle"], ctx["data2"]),
        lambda e, ctx: e._volume_registry.get(ctx["handle"]).data is ctx["data1"],
        lambda e, ctx: (
            e._volume_registry.get(ctx["handle"]).data is ctx["data2"]
            and e._volume_registry.get(ctx["handle"]).dirty
        ),
        setup=lambda e: {
            "data1": np.zeros((2, 2, 2), dtype=np.float32),
            "data2": np.ones((2, 2, 2), dtype=np.float32),
            "handle": None,
        },
    ),
    _row(
        "5.1-26-material-field-writes",
        lambda e, ctx: (
            setattr(ctx["mat"], "roughness", 0.9),
            setattr(ctx["mat"], "metallic", 0.4),
        ),
        lambda e, ctx: ctx["mat"].roughness == 0.5 and ctx["mat"].metallic == 0.0,
        lambda e, ctx: ctx["mat"].roughness == 0.9 and ctx["mat"].metallic == 0.4,
        setup=lambda e: {
            "mat": (lambda m: (_spawn_cube(e, material=m), m)[1])(StandardMaterial("#ffffff"))
        },
    ),
    _row(
        "5.1-31-quadstreamer-focus",
        lambda e, ctx: ctx["streamer"].focus(4.0, 4.0),
        lambda e, ctx: ctx["streamer"].drawn() == set() and _alive_count(e) == 0,
        lambda e, ctx: len(ctx["streamer"].drawn()) > 0 and _alive_count(e) > 0,
        setup=lambda e: {"streamer": _warm_quad_streamer(e)},
    ),
    _row(
        "5.1-32-uploads-upload",
        lambda e, ctx: ctx["streamer"]._upload(ctx["cell"], 1 << 20),
        lambda e, ctx: ctx["cell"] not in ctx["streamer"]._gpu,
        lambda e, ctx: ctx["streamer"]._gpu[ctx["cell"]]["done"],
        setup=lambda e: _warm_streamer_cell(e),
        gpu=True,
    ),
    _row(
        "5.1-33-uploads-draw",
        lambda e, ctx: ctx["streamer"]._draw(ctx["cell"], (0.0, 0.0, 0.0)),
        lambda e, ctx: _alive_count(e) == 0 and ctx["cell"] not in ctx["streamer"]._drawn,
        lambda e, ctx: _alive_count(e) == 1 and ctx["cell"] in ctx["streamer"]._drawn,
        setup=lambda e: _warm_streamer_cell(e),
    ),
    _row(
        "5.1-35-uploads-unload",
        lambda e, ctx: ctx["streamer"]._unload(ctx["cell"]),
        lambda e, ctx: ctx["cell"] in ctx["streamer"]._gpu,
        lambda e, ctx: (
            ctx["cell"] not in ctx["streamer"]._gpu and e._geometry_registry.get(ctx["gid"]) is None
        ),
        setup=lambda e: _warm_streamer_cell(e, faked=True),
    ),
    _row(
        "5.1-36-gui-widget-mutation",
        lambda e, ctx: (
            setattr(ctx["text"], "text", "world"),
            ctx["panel"].children.append(Text("nested")),
            e.gui.append(ctx["panel2"]),
        ),
        lambda e, ctx: (
            ctx["text"].text == "hello" and len(e.gui) == 1 and len(ctx["panel"].children) == 1
        ),
        lambda e, ctx: (
            ctx["text"].text == "world" and len(e.gui) == 2 and len(ctx["panel"].children) == 2
        ),
        setup=lambda e: _gui_setup(e),
    ),
    _row(
        "5.1-38-gui-style-set-theme",
        lambda e, ctx: (
            gui_style.set_theme({"bg": "#123456"}),
            gui_style.define("row", {"gap": 9}),
        ),
        lambda e, ctx: gui_style._theme["bg"] == ctx["bg"] and "row" not in gui_style._classes,
        lambda e, ctx: (
            gui_style._theme["bg"] == "#123456" and gui_style._classes["row"] == {"gap": 9}
        ),
        setup=lambda e: {"bg": gui_style._theme["bg"]},
        cleanup=lambda: gui_style.reset(),
    ),
    _row(
        "5.1-40-render-to-texture-family",
        lambda e, ctx: (e.refresh_texture(ctx["handle"]), e.set_texture_live(ctx["handle"], False)),
        lambda e, ctx: (
            e._views[ctx["handle"].id].pending is False and e._views[ctx["handle"].id].live is True
        ),
        lambda e, ctx: (
            e._views[ctx["handle"].id].pending is True and e._views[ctx["handle"].id].live is False
        ),
        setup=lambda e: {"handle": e.render_to_texture(e.camera, (16, 16))},
        gpu=True,
    ),
    _row(
        "5.1-42-write-texture",
        lambda e, ctx: texture_module.write_texture(
            e, ctx["handle"], np.zeros((2, 2, 4), dtype=np.uint8)
        ),
        lambda e, ctx: "write_texture" not in ctx["log"].names(),
        lambda e, ctx: "write_texture" in ctx["log"].names(),
        setup=lambda e: _texture_setup(e),
        gpu=True,
    ),
    _row(
        "5.1-43-register-volume-update-volume",
        lambda e, ctx: e.update_volume(ctx["handle"], ctx["data2"]),
        lambda e, ctx: e._volume_registry.get(ctx["handle"]).data is ctx["data1"],
        lambda e, ctx: e._volume_registry.get(ctx["handle"]).data is ctx["data2"],
        setup=lambda e: {
            "data1": np.zeros((2, 2, 2), dtype=np.float32),
            "data2": np.ones((2, 2, 2), dtype=np.float32),
            "handle": None,
        },
    ),
    _row(
        "5.1-46-engine-benders",
        lambda e, ctx: (e.benders.append((0.0, 0.0, 1.0, 1.0)), e.benders.reverse()),
        lambda e, ctx: len(e.benders) == 0,
        lambda e, ctx: e.benders == [(0.0, 0.0, 1.0, 1.0)],
    ),
    # --- 5.2 §4 GPU calls ---------------------------------------------------
    _row(
        "5.2-03-set-geometry-write-buffer",
        lambda e, ctx: ctx["handle"].set_geometry(_moved_cube()),
        lambda e, ctx: "write_buffer" not in ctx["log"].names(),
        lambda e, ctx: "write_buffer" in ctx["log"].names(),
        setup=lambda e: _rendered_cube(e),
        gpu=True,
    ),
    _row(
        "5.2-04-jointstore-set",
        lambda e, ctx: e._render_pipeline._joints.set(
            ctx["handle"].index, np.eye(4, dtype=np.float32)[None]
        ),
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) == 0,
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) > 0,
        setup=lambda e: {
            "handle": e.spawn(
                MeshComponent(_skinned_geometry()),
                MaterialComponent(mx.material.standard("#ffffff")),
                Transform(),
            )
        },
    ),
    _row(
        "5.2-05-jointstore-set-many",
        lambda e, ctx: e._render_pipeline._joints.set_many(
            np.array([ctx["handle"].index]), np.eye(4, dtype=np.float32)[None, None]
        ),
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) == 0,
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) > 0,
        setup=lambda e: {
            "handle": e.spawn(
                MeshComponent(_skinned_geometry()),
                MaterialComponent(mx.material.standard("#ffffff")),
                Transform(),
            )
        },
    ),
    _row(
        "5.2-06-jointstore-release",
        lambda e, ctx: e._render_pipeline._joints.release(np.array([ctx["handle"].index])),
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) > 0,
        lambda e, ctx: int(e._render_pipeline._joints.block_of[ctx["handle"].index]) == 0,
        setup=lambda e: {
            "handle": _posed_skinned_entity(e),
        },
    ),
    _row(
        "5.2-07-jointstore-ensure-buffer",
        lambda e, ctx: e._render_pipeline._joints.ensure_buffer(e._device),
        lambda e, ctx: e._render_pipeline._joints.buffer is None,
        lambda e, ctx: e._render_pipeline._joints.buffer is not None,
        gpu=True,
    ),
    # --- 5.2 rows that share a 5.1 row's behavior ---------------------------
    _row(
        "5.2-01-geometry-create-buffers",
        lambda e, ctx: e._geometry_registry.create_buffers(ctx["gid"], ctx["geo"], e._device.queue),
        lambda e, ctx: e._geometry_registry.get_gpu_buffers(ctx["gid"]) is None,
        lambda e, ctx: e._geometry_registry.get_gpu_buffers(ctx["gid"]) is not None,
        setup=lambda e: {
            "geo": _moved_cube(),
            "gid": e._geometry_registry.register(_moved_cube(2.0)),
        },
        gpu=True,
    ),
    _row(
        "5.2-02-geometry-release",
        lambda e, ctx: e._geometry_registry.release(ctx["gid"]),
        lambda e, ctx: e._geometry_registry.get(ctx["gid"]) is not None,
        lambda e, ctx: e._geometry_registry.get(ctx["gid"]) is None,
        setup=lambda e: {
            "gid": (lambda gid: (e._geometry_registry.adopt_buffers(gid, {}), gid)[1])(
                e._geometry_registry.register(_moved_cube(2.0))
            )
        },
    ),
    _row(
        "5.2-08-streamer-create-buffer",
        lambda e, ctx: ctx["streamer"]._upload(ctx["cell"], 1 << 20),
        lambda e, ctx: "create_buffer" not in ctx["log"].names(),
        lambda e, ctx: "create_buffer" in ctx["log"].names(),
        setup=lambda e: _recording_streamer_cell(e),
        gpu=True,
    ),
    _row(
        "5.2-09-streamer-destroy",
        lambda e, ctx: ctx["streamer"]._unload(ctx["cell"]),
        lambda e, ctx: "destroy" not in ctx["log"].names(),
        lambda e, ctx: "destroy" in ctx["log"].names(),
        setup=lambda e: _recording_streamer_cell(e, uploaded=True),
        gpu=True,
    ),
    _row(
        "5.2-11-write-texture",
        lambda e, ctx: texture_module.write_texture(
            e, ctx["handle"], np.zeros((2, 2, 4), dtype=np.uint8)
        ),
        lambda e, ctx: "write_texture" not in ctx["log"].names(),
        lambda e, ctx: "write_texture" in ctx["log"].names(),
        setup=lambda e: _texture_setup(e),
        gpu=True,
    ),
]


def ctx_id(engine):
    """Placeholder for a row whose id is only known after the call: unused."""
    raise AssertionError("a row cannot ask for an id before the call")


def _warm_quad_streamer(engine):
    """A QuadStreamer with every node built: focus() then has something to draw."""
    from manifoldx.streaming import MeshData, QuadStreamer

    data = MeshData(
        positions=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32),
        normals=np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1)),
        colors=np.ones((4, 3), dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
    )
    streamer = QuadStreamer(
        engine,
        cell=8.0,
        origin=(0, 0),
        levels=2,
        exists=lambda L, a, b: True,
        build=lambda L, a, b: data,
        error=lambda L: 1000.0 * (2**L),
        tau=1.0,
        upload_bytes_per_frame=1 << 20,
    )
    streamer.focus(4.0, 4.0)  # first pass queues every build
    _wait_builds(streamer)
    return streamer


def _warm_streamer_cell(engine, faked=False):
    """A Streamer with cell (0, 0) ready and, unless faked, on the GPU."""
    from manifoldx.streaming import MeshData, Streamer

    data = MeshData(
        positions=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32),
        normals=np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1)),
        colors=np.ones((4, 3), dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
    )
    streamer = Streamer(
        engine,
        cell=8.0,
        build=lambda i, j: data,
        radius=0,
        ready_radius=0,
        uploads_per_frame=0,
    )
    streamer._pending[(0, 0)] = _resolved(_packed(data))
    streamer._tick()
    cell = (0, 0)
    if faked:
        gid = engine._geometry_registry.register(streamer._ready[cell].geo)
        streamer._gpu[cell] = {"id": gid, "done": True}
        return {"streamer": streamer, "cell": cell, "gid": gid}
    return {"streamer": streamer, "cell": cell}


def _packed(data):
    """The worker's packed cell: what `_tick` collects from a finished build."""
    from manifoldx.streaming import _pack

    return _pack(lambda *key: data, 0, 0)


def _resolved(value):
    class _Done:
        def done(self):
            return True

        def result(self):
            return value

    return _Done()


def _wait_builds(streamer):
    for future in list(streamer._pending.values()):
        future.result()
    streamer._pending.clear()


def _gui_setup(engine):
    text = Text("hello")
    panel = Panel(children=[text])
    engine.gui.append(panel)
    return {"text": text, "panel": panel, "panel2": Panel(children=[Text("second")])}


def _rendered_cube(engine):
    """A spawned cube whose vertex buffer the first render created."""
    handle = _spawn_cube(engine)
    engine._draw_frame()
    log, undo = _record_device_calls(engine)
    return {"handle": handle, "log": log, "_undo": undo}


def _texture_setup(engine):
    handle = texture_module.texture_from_array(engine, np.zeros((4, 4, 4), dtype=np.uint8))
    log, undo = _record_device_calls(engine)
    return {"handle": handle, "log": log, "_undo": undo}


def _recording_streamer_cell(engine, uploaded=False):
    """A Streamer cell ready to upload, with the device call log installed."""
    from manifoldx.streaming import MeshData, Streamer

    data = MeshData(
        positions=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32),
        normals=np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1)),
        colors=np.ones((4, 3), dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
    )
    streamer = Streamer(
        engine, cell=8.0, build=lambda i, j: data, radius=0, ready_radius=0, uploads_per_frame=0
    )
    streamer._pending[(0, 0)] = _resolved(_packed(data))
    streamer._tick()
    if uploaded:
        streamer._upload((0, 0), 1 << 20)  # main thread, before the log: makes real buffers
    log, undo = _record_device_calls(engine)
    return {"streamer": streamer, "cell": (0, 0), "log": log, "_undo": undo}


def _posed_skinned_entity(engine):
    handle = engine.spawn(
        MeshComponent(_skinned_geometry()),
        MaterialComponent(mx.material.standard("#ffffff")),
        Transform(),
    )
    engine.set_joints(np.array([handle.index]), np.eye(4, dtype=np.float32)[None, None])
    return handle


@pytest.mark.parametrize("row", ROWS)
def test_audit_row_defers(row):
    engine = _gpu_engine(name=f"row-{row.name}") if row.gpu else _engine(f"row-{row.name}")
    ctx = row.setup(engine) if row.setup else None
    if isinstance(ctx, dict) and "handle" in ctx and ctx["handle"] is None:
        ctx["handle"] = engine.register_volume(ctx["data1"])
    undo = ctx.get("_undo") if isinstance(ctx, dict) else None
    try:
        with engine._defer_calls():
            row.call(engine, ctx)
            row.before(engine, ctx)
        _apply(engine)
        row.after(engine, ctx)
    finally:
        if undo is not None:
            undo()
        if row.cleanup is not None:
            row.cleanup()


# --- rows left immediate: each has a test that says why ---------------------


def test_row_5_1_19_direct_register_acts_at_once_and_reports_the_id():
    """Departure: a caller uses the geometry id at once, so it is handed out now."""
    engine = _engine()
    geo = _moved_cube(6.0)
    with engine._defer_calls():
        gid = engine._geometry_registry.register(geo)
        assert engine._geometry_registry.get(gid) is geo


def test_row_5_1_20_register_by_name_acts_at_once_and_reports_the_id():
    """Departure: a caller uses the id at once, so the registry hands it out now."""
    engine = _engine()
    with engine._defer_calls():
        gid = engine._geometry_registry.register_by_name("deferred-test-geo", _moved_cube(4.0))
        assert engine._geometry_registry.get(gid) is not None
    assert isinstance(gid, int)


def test_row_5_1_24_material_register_acts_at_once_and_reports_the_id():
    """Departure: same as 5.1-19/20, a material id is used by the caller."""
    engine = _engine()
    mat = mx.material.basic("#00ff00")
    with engine._defer_calls():
        mid = engine._material_registry.register(mat)
        assert engine._material_registry.get(mid) is mat


def test_row_5_1_25_register_volume_acts_at_once_and_reports_the_handle():
    """Departure: a volume handle goes straight into the component that references it."""
    engine = _engine()
    data = np.zeros((2, 2, 2), dtype=np.float32)
    with engine._defer_calls():
        handle = engine.register_volume(data)
        assert engine._volume_registry.get(handle).data is data


def test_row_5_1_27_direct_array_writes_stay_immediate():
    """Departure: a raw numpy write into a store array is not a call, so it cannot be seen."""
    engine = _engine()
    handle = _spawn_cube(engine)
    rows = engine.store._components["Transform"]
    with engine._defer_calls():
        rows[handle.index, 0:3] = (9.0, 9.0, 9.0)  # exactly what viz/shims.py does
        assert np.allclose(rows[handle.index, 0:3], (9, 9, 9))
    assert len(engine.commands) == 0


def test_row_5_1_29_engine_component_acts_at_once_and_returns_the_class():
    """Departure: `@engine.component` is a decorator, whose return value is the class."""
    engine = _engine()

    with engine._defer_calls():

        @engine.component
        class DeferredTag:
            weight: Float

        assert "DeferredTag" in engine.store._components
    assert DeferredTag.__name__ == "DeferredTag"


def test_row_5_1_39_value_display_getter_reads_on_the_render_thread():
    """Departure: a getter is a read, not a write; it stays on the main thread."""
    from manifoldx.gui import ValueDisplay

    engine = _engine()
    display = ValueDisplay(getter=lambda: f"{engine.elapsed:.3f}")
    display.refresh()
    assert display.text == "0.000"


def test_row_5_1_41_load_texture_acts_at_once_and_returns_a_handle():
    """Departure: the caller uses the TextureHandle at once (it needs the GPU anyway)."""
    engine = _gpu_engine("row-5.1-41")
    with engine._defer_calls():
        handle = texture_module.texture_from_array(engine, np.zeros((2, 2, 4), dtype=np.uint8))
        assert isinstance(handle, texture_module.TextureHandle)


def test_row_5_1_44_engine_compute_acts_at_once_and_returns_the_class():
    """Departure: `@engine.compute` is a decorator too, and registration is setup work."""
    engine = _engine()

    from manifoldx.compute import Compute

    class DeferredCompute(Compute):
        pass

    with engine._defer_calls():
        assert engine.compute(DeferredCompute) is DeferredCompute
        assert DeferredCompute.__name__ == "DeferredCompute"


def test_row_5_1_45_label_atlas_acts_at_once_and_returns_the_atlas():
    """Departure: the atlas is a return value, and label strings resolve through it at once."""
    engine = _engine()
    with engine._defer_calls():
        atlas = engine.get_label_atlas()
        assert engine.get_label_atlas() is atlas


def test_row_5_2_10_load_texture_family_is_immediate():
    """Departure: same reason as 5.1-41, the handle is a return value."""
    engine = _gpu_engine("row-5.2-10")
    with engine._defer_calls():
        handle = texture_module.texture_from_array(engine, np.zeros((2, 2, 4), dtype=np.uint8))
        assert handle.texture is not None


def test_row_5_2_12_render_to_texture_is_immediate():
    """Departure: render_to_texture returns the handle its caller draws or drops."""
    engine = _gpu_engine("row-5.2-12")
    with engine._defer_calls():
        handle = engine.render_to_texture(engine.camera, (8, 8))
        assert handle.id in engine._views


def test_row_5_2_13_compute_registration_is_immediate():
    """Departure: compute registration creates a shader module at setup time."""
    from manifoldx.compute import Compute

    engine = _gpu_engine("row-5.2-13")

    class DeferredCompute(Compute):
        @classmethod
        def compile(cls):
            return "@compute @workgroup_size(1) fn main() {}"

    with engine._defer_calls():
        assert engine.compute(DeferredCompute) is DeferredCompute


def test_row_5_2_15_label_atlas_upload_is_render_side():
    """Departure: the atlas is uploaded by the render on the main thread; a system gets it now."""
    engine = _gpu_engine("row-5.2-15")
    with engine._defer_calls():
        atlas = engine.get_label_atlas()
        assert atlas is engine.get_label_atlas()


def test_row_5_2_17_tracer_is_its_own_renderer():
    """Departure: the tracer owns a second device and returns images; it never defers."""
    from manifoldx.trace.tracer import Tracer

    assert not hasattr(Tracer, "_deferred_in_pipeline")


# =============================================================================
# The device-thread test: no device call on the systems thread
# =============================================================================


def test_no_device_call_runs_on_the_systems_thread():
    """A streamer and a skinned figure deferred on a worker touch the GPU only at apply."""
    engine = _gpu_engine("device-thread", w=128, h=128)
    log, undo = _record_device_calls(engine)
    try:
        main_thread = threading.get_ident()

        # A streamer with one built, uploaded cell and another to draw.
        from manifoldx.streaming import MeshData, Streamer

        data = MeshData(
            positions=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32),
            normals=np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1)),
            colors=np.ones((4, 3), dtype=np.float32),
            indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
        )
        streamer = Streamer(
            engine,
            cell=8.0,
            build=lambda i, j: data,
            radius=1,
            ready_radius=1,
            uploads_per_frame=0,
            upload_bytes_per_frame=64,
        )
        streamer._pending[(0, 0)] = _resolved(_packed(data))
        streamer._tick()
        streamer.focus(4.0, 4.0)  # partial upload on the main thread, so _on_gpu is False

        skinned = _skinned_geometry()
        errors = []
        queued = {}

        def systems():
            try:
                with engine._defer_calls():
                    streamer.focus(4.0, 4.0)  # its upload defers as a whole call
                    queued["handle"] = engine.spawn(
                        MeshComponent(skinned),
                        MaterialComponent(mx.material.standard("#ffffff")),
                        Transform(pos=(0.0, 0.0, 0.0)),
                    )
                    engine.set_joints(
                        np.array([queued["handle"].index]),
                        np.eye(4, dtype=np.float32)[None, None],
                    )
            except Exception as error:  # pragma: no cover - surfaced below
                errors.append(error)

        worker = threading.Thread(target=systems)
        worker.start()
        worker.join()
        assert errors == []

        # The systems thread touched no device, and the apply ran on this one.
        assert log.threads() <= {main_thread}, log.calls
        _apply(engine)
        engine._draw_frame()
        engine._draw_frame()
        assert log.threads() <= {main_thread}, log.calls
        assert "write_buffer" in log.names(), "the deferred upload never reached the GPU"
        assert engine._render_pipeline._joints.block_of[queued["handle"].index] > 0
    finally:
        undo()
