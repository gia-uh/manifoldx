"""Scene declarations and synchronous systems for the native desktop Engine."""

from __future__ import annotations

import json
import os
from itertools import count
from concurrent.futures import Future
from dataclasses import dataclass
from inspect import iscoroutinefunction
from pathlib import Path
from threading import current_thread, main_thread
from typing import get_type_hints
from weakref import WeakSet, finalize, ref

import numpy as np

from manifoldx._native_reload import _RegistrationReload, _entry, _preparing
from manifoldx._versions import _Counter
from manifoldx.camera import Camera
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.ecs import ComponentView, _make_component_class
from manifoldx.events import EventBus, FrameWaiters, _parse_query_components
from manifoldx.gui._native import _NativeGui
from manifoldx.gui.widgets import _GuiRoot
from manifoldx.input import InputState, KeyEvent, PointerEvent, ResizeEvent, WheelEvent
from manifoldx.lines import Lines
from manifoldx.shader import ShaderMaterial
from manifoldx.resources import (
    BasicMaterial,
    DirectionalLight,
    FlatMaterial,
    Material as MaterialBase,
    MaterialRegistry,
    PhongMaterial,
    PointLight,
    SpotLight,
    StandardMaterial,
    VolumeRegistry,
    SURFACE_BAKED,
    baked_columns,
)
from manifoldx.systems import Query
from manifoldx.textures import TextureHandle, _mip_chain
from manifoldx.world import World


def _color(value):
    if isinstance(value, str):
        digits = value.removeprefix("#")
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) != 6:
            raise ValueError(f"expected an opaque hexadecimal RGB color, got {value!r}")
        return tuple(int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))
    if len(value) == 4 and value[3] == 1:
        value = value[:3]
    if len(value) != 3:
        raise ValueError("the native base renderer requires opaque RGB colors")
    return tuple(value)


def _camera_input(camera):
    if camera.vertical_scale != 1:
        raise NotImplementedError("vertical scale is unavailable in the native renderer")
    return (
        camera.position.tolist(), camera.target.tolist(), camera.up.tolist(),
        camera.fov, camera.near, camera.far,
    )


def _camera_layers(camera):
    return None if camera.layers is None else sorted(camera.layers)


def _release_native_texture(owner, index, view):
    engine = owner()
    if engine is not None and not engine._closed:
        engine._texture_handles.pop(index, None)
        if view:
            engine._release_view(index)
        else:
            engine._graphics.texture_release(index)


@dataclass
class _View:
    camera: Camera
    live: bool
    pending: bool = False


class _Rows:
    def __init__(self, columns):
        self._components = columns

    def get_component_data(self, name, indices):
        return self._components[name][indices]


class _TransformHandle:
    def __init__(self, engine, entity):
        self._engine = engine
        self._entity = entity

    @property
    def pos(self):
        engine = self._engine
        row = engine._transform_rows(self._entity)[0]
        return row[:3].astype("float64") + engine._origin_vector()

    @pos.setter
    def pos(self, point):
        engine = self._engine
        rows = engine._transform_rows(self._entity)
        rows[0, :3] = np.asarray(point, "float64") - engine._origin_vector()
        if engine._commands is None:
            engine._world.replace("Transform", (self._entity,), rows)
        else:
            engine._transform_edits[self._entity] = rows[0]

    @property
    def rot(self):
        return self._engine._transform_rows(self._entity)[0, 3:7].copy()

    @rot.setter
    def rot(self, value):
        engine = self._engine
        rows = engine._transform_rows(self._entity)
        rows[0, 3:7] = np.asarray(value, "float64")
        if engine._commands is None:
            engine._world.replace("Transform", (self._entity,), rows)
        else:
            engine._transform_edits[self._entity] = rows[0]

    @property
    def scale(self):
        return self._engine._transform_rows(self._entity)[0, 7:10].copy()

    @scale.setter
    def scale(self, value):
        engine = self._engine
        rows = engine._transform_rows(self._entity)
        rows[0, 7:10] = value
        if engine._commands is None:
            engine._world.replace("Transform", (self._entity,), rows)
        else:
            engine._transform_edits[self._entity] = rows[0]

    position = pos


class _EntityHandle:
    def __init__(self, engine, entities):
        self._engine = engine
        self._entities = entities
        self.index = entities[0].slot
        self.indices = np.array([entity.slot for entity in entities])
        self.transform = _TransformHandle(engine, entities[0])

    def set_geometry(self, geometry):
        self._engine._replace_geometry(self._entities, geometry)

    def set_joints(self, matrices):
        entity = self._entities[0]
        self._engine._set_joint_poses(
            [(entity.slot, entity.generation)], np.asarray(matrices)[None]
        )


class _NativeEngine:
    def __init__(
        self,
        title,
        *,
        height=600,
        width=800,
        fullscreen=False,
        monitor=None,
        max_fps=240.0,
        max_entities=100_000,
        check_numerics=True,
        pipelined=None,
        smooth_dt=True,
    ):
        from manifoldx._core_native import FrameLoop, _NativeGraphics

        if _preparing.get():
            raise RuntimeError("reload registration modules cannot create an Engine")
        if pipelined:
            raise NotImplementedError(
                "Python graphics pipelining is unavailable with the Rust Engine"
            )
        if not 1 <= max_fps <= 1000:
            raise ValueError("max_fps must be between 1 and 1000")
        self.title, self.w, self.h = title, width, height
        self.fullscreen, self.monitor, self.max_fps = fullscreen, monitor, max_fps
        self._frame_index = 0
        self._last_dt = 1 / 60
        self.elapsed = 0.0
        # Measured by the native scheduler; the Python engine's own attrs have
        # the same names so devchannel.Meter reads both cores the same way.
        self.frame_split = {
            "update": 0.0, "acquire": 0.0, "render": 0.0,
            "submit": 0.0, "present": 0.0, "overlap": 0.0,
        }
        self.system_split = {}
        self.over_budget = {}
        self.frame_interval = None
        self._hud_sample = None
        self._debug_hud = False
        self.camera = Camera()
        self.background_color = (0.1, 0.1, 0.2)
        self._origin = (0.0, 0.0)
        self._lights = []
        self._sun = None
        self._spot = None
        self._post = None
        self._environment = None
        self.fog_enabled = False
        self.fog_start = 0.0
        self.fog_end = 1.0
        self.fog_color = self.background_color
        self._native_environment = None
        self._shadow_options = None
        self._capacity = max_entities
        from manifoldx.hot import State

        self.input = InputState()
        self.state = State()
        self.gui = _GuiRoot()
        self._gui_runtime = _NativeGui(self)
        self._event_bus = EventBus()
        self._dev_channel = None
        self._looks = []  # futures waiting for the next presented frame as PNG bytes (the dev server's look)
        self._pending_looks = []  # those whose capture is in flight
        self._look_armed = False
        self._aio_loop = None
        self._tasks = set()
        self._task_errors = []
        self._frame_waiters = FrameWaiters(self._get_active_loop)
        self._application_systems = {}
        self._systems_changed = False
        self._registration_reload = None
        self.lines = Lines()
        self.benders = []
        self._line_rows = {}
        self._world = World(max_entities)
        self._position_binding = None
        self._compute_classes = {}
        self._compute_columns = set()
        self._input_probe = None
        self._desktop_probe = (False, False, False, None)
        self._frame_loop = FrameLoop
        columns = (("Transform", "f32", 10), ("Mesh", "u32", 1), ("Material", "u32", 1))
        for name, kind, size in columns:
            self._world.register(name, kind=kind, width=size)
        # Every Component class imported before the engine existed is
        # registered now, while the native schema is still open. The first
        # spawn publishes entities, after which the native world refuses new
        # schemas; an app that spawns a mesh and then a viz component (or any
        # far-apart component) would otherwise fail on its lazy registration.
        from manifoldx.components import _COMPONENT_CLASSES

        for component in _COMPONENT_CLASSES.values():
            if not component._gpu_only:
                self._world.register(component)
        self._registries = {"Mesh": {}, "Material": {}}
        self._view_materials = {}
        self._mesh_ids = count(1)
        self._material_ids = count(1)
        self._cell_versions = count(1)
        self._streamers = WeakSet()
        self._process_executor = None
        self._pending_tasks = []
        self._manual_registry = MaterialRegistry()
        self._label_atlas = None
        self._volume_registry = VolumeRegistry()
        self._textures = 0
        self._texture_handles = {}
        self._views = {}
        self._texture_reads = {}
        self._graphics = _NativeGraphics()
        self._commands = None
        self._snapshot = None
        self._transform_edits = {}
        self._entities = ()
        self._time_base = 0.0
        self._fixed_dt = None
        self._started = False
        self._closed = False
        self._quit = False
        self._desktop_ran = False
        self.last_frame = None
        self.desktop_report = None

    def _deferring(self):
        return False

    @property
    def world(self):
        """Native World used by the desktop frame scheduler."""
        return self._world

    def bind_position(self, program, *, field, target):
        """Draw a registered kernel component's vec3 position from the GPU."""
        from manifoldx.compute.kernel import _Kernel, _VERSION

        self._world._owner()
        if self._started or self._closed or self._position_binding is not None:
            raise RuntimeError("bind one resident position field before starting the Engine")
        if target != "wgsl":
            raise ValueError("resident position binding requires target='wgsl'")
        if not isinstance(program, _Kernel) or program.version != _VERSION:
            raise TypeError("resident position binding requires a v1 @kernel declaration")
        column = self._world._column(program.component)
        _, kind, width, layout = self._world._metadata[column]
        if kind != "f32" or width != program.width or tuple(layout.items()) != program.layout:
            raise ValueError("resident position kernel does not match the registered schema")
        source = self._world.field(program.component, field)
        if source.width != 3 or field not in program.writes:
            raise ValueError("bind a registered vec3<f32> field written by the kernel")
        writes = sorted(layout[name] for name in program.writes)
        if column in self._compute_columns:
            raise ValueError("resident position cannot alias a mirrored Compute column")
        self._position_binding = (
            column,
            width,
            source.offset,
            program.version,
            writes,
            program.wgsl(),
        )

    def _desktop_key(self, name, pressed):
        if name is None:
            for key in self.input.pressed_keys:
                self.input._record_key_up(key, ())
        elif pressed:
            self.input._record_key_down(name, ())
        else:
            self.input._record_key_up(name, ())

    @property
    def origin(self):
        return self._origin

    def _origin_vector(self):
        return np.array((self._origin[0], 0, self._origin[1]), dtype="float64")

    def _transform_rows(self, entity):
        if self._snapshot is not None:
            if entity in self._transform_edits:
                return self._transform_edits[entity][None].copy()
            data = self._snapshot.read("Transform")
            try:
                index = data.entities.index(entity)
            except ValueError as exc:
                raise ValueError("stale transform handle") from exc
            if not data.membership[index]:
                raise ValueError("entity has no transform")
            row = data.numpy()[[index]]
            row[:, :3] += self._callback_origin - self._origin_vector()
            return row
        self._world._owner()
        handles = self._world._handles((entity,))
        return (
            self._world._readback(self._world._native.read_component("Transform", handles))
            .numpy()
            .copy()
        )

    def set_origin(self, x, z):
        if self._position_binding is not None:
            raise NotImplementedError(
                "changing origin with a resident position binding is unavailable"
            )
        offset = self._origin_vector() - (x, 0, z)
        if not np.any(offset):
            return
        if self._commands is None:
            data = self._world.query("Transform").read()
            rows = data.numpy().copy()
            rows[:, :3] = rows[:, :3].astype("float64") + offset
            if data.entities:
                self._world.replace("Transform", data.entities, rows)
        else:
            for row in self._transform_edits.values():
                row[:3] = row[:3].astype("float64") + offset
        self.camera.position = (self.camera.position.astype("float64") + offset).astype("float32")
        self.camera.target = (self.camera.target.astype("float64") + offset).astype("float32")
        if self._spot is not None:
            self._spot.position = tuple(np.asarray(self._spot.position, dtype="float64") + offset)
        if self._shadow_options is not None:
            target, *options = self._shadow_options
            self._shadow_options = (tuple(np.asarray(target, dtype="float64") + offset), *options)
            self._graphics.shadows(self._shadow_options)
        self._origin = (float(x), float(z))

    def on(self, event):
        def register(callback):
            if _parse_query_components(callback):
                raise NotImplementedError("native event handlers use typed payloads without World queries")
            return self._event_bus.on(event)(callback)

        return register

    def off(self, event, callback):
        self._event_bus.off(event, callback)

    def emit(self, event, payload=None):
        self._event_bus.emit(event, payload)

    def log(self, message: str) -> None:
        """A line for the dev log under `manifoldx run` (its page, /api/log, .manifoldx/log.txt)."""
        from manifoldx import devchannel

        devchannel.LOG.info(message)

    def tick(self):
        return self._frame_waiters.add_tick()

    def delay(self, seconds):
        return self._frame_waiters.add_delay(seconds, self.elapsed)

    def elapsed_at(self, target):
        return self._frame_waiters.add_elapsed_at(target)

    def _get_active_loop(self):
        from manifoldx.engine import Engine

        return Engine._get_active_loop(self)

    def _on_task_done(self, task):
        from manifoldx.engine import Engine

        Engine._on_task_done(self, task)

    def _pump_events(self):
        from manifoldx.engine import Engine

        self._event_bus.dispatch_pending(self)
        self._frame_waiters.resolve(self.elapsed)
        Engine._pump_aio_loop(self)

    def _window_events(self, events):
        self.gui.pointer_over_gui = False
        for event, key, data, button, modifiers in events:
            consumed = False
            self._gui_runtime.last_target = None
            modifiers = tuple(modifiers)
            if event == "focus_lost":
                self._desktop_key(None, False)
                for held in self.input.pressed_buttons:
                    x, y = self.input.mouse_pos
                    self.input._record_pointer_up(PointerEvent(x, y, 0, 0, held, (), (), "up"))
                self._gui_runtime.reset()
                continue
            if event in ("key_down", "key_up"):
                down = event == "key_down"
                record = self.input._record_key_down if down else self.input._record_key_up
                record(key, modifiers)
                payload = KeyEvent(key, modifiers, down)
                if down:
                    consumed = self._gui_runtime.key(payload)
            elif event == "wheel":
                x, y, dx, dy = data
                payload = WheelEvent(dx, dy, x, y, self.input.pressed_buttons, modifiers)
                self.input._record_wheel(payload)
                consumed = self._gui_runtime.pointer(payload, "wheel")
            else:
                phase = event.removeprefix("pointer_")
                buttons = set(self.input.pressed_buttons)
                if phase == "down":
                    buttons.add(button)
                elif phase == "up":
                    buttons.discard(button)
                payload = PointerEvent(*data, button, tuple(sorted(buttons)), modifiers, phase)
                getattr(self.input, f"_record_pointer_{phase}")(payload)
                consumed = self._gui_runtime.pointer(payload, phase)
            if self._input_probe is not None:
                self._input_probe(event, payload, consumed, self._gui_runtime.last_target)
            # GUI consumption is cooperative, as on the original input bus.
            self.emit(event, payload)

    def _arm_look(self):
        """Ask the desktop loop to copy this frame for every look requested since the last one."""
        if not self._looks:
            return
        looks, self._looks = self._looks, []
        self._pending_looks.extend(looks)
        if self._look_armed:
            return
        from manifoldx.devchannel import _Unavailable

        try:
            self._graphics.request_presented()
        except RuntimeError as error:
            self._pending_looks, pending = [], self._pending_looks
            reason = _Unavailable(f"look is unavailable with the Rust core: {error}")
            for shot in pending:
                if not shot.done():
                    shot.set_exception(reason)
        else:
            self._look_armed = True

    def _finish_look(self):
        """Answer the looks waiting on a frame whose copy came back, off the engine's thread."""
        if not self._pending_looks:
            return
        try:
            image = self._graphics.take_presented()
        except RuntimeError as error:
            pending, self._pending_looks = self._pending_looks, []
            self._look_armed = False
            for shot in pending:
                if not shot.done():
                    shot.set_exception(error)
            return
        if image is None:
            return
        from threading import Thread

        from manifoldx import devchannel

        self._look_armed = False
        looks, self._pending_looks = self._pending_looks, []
        width, height, bgra, data = image
        Thread(target=devchannel.answer_frame, args=(looks, data, width, height, bgra),
               name="manifoldx-look", daemon=True).start()

    def _fail_looks(self, error):
        looks, self._looks = self._looks, []
        pending, self._pending_looks = self._pending_looks, []
        self._look_armed = False
        for shot in (*looks, *pending):
            if not shot.done():
                shot.set_exception(error)

    def _record_metrics(self):
        """Read the previous presented frame's samples from the native scheduler."""
        if self._dev_channel is None or self._dev_channel.meter is None:
            return
        sample = self._world._loop.last_metrics()
        if sample is None:
            return
        self._hud_sample = sample
        ns = 1e9
        self.frame_interval = sample["interval_ns"] / ns if sample["interval_ns"] else None
        self.frame_split = {
            "update": sample["update_ns"] / ns,
            "acquire": sample["acquire_ns"] / ns,
            "render": sample["render_ns"] / ns,
            "submit": sample["submit_ns"] / ns,
            "present": sample["present_ns"] / ns,
            "overlap": 0.0,
        }
        self.system_split = {name: value / ns for name, value in sample["systems"]}
        self.over_budget = {name: value / ns for name, value in sample["over_budget"]}
        self._dev_channel.frame(self)

    def _hud_lines(self):
        """The `--debug` HUD's text, from the last presented frame's measurements."""
        sample = self._hud_sample
        if not sample or not sample.get("interval_ns"):
            return ["manifoldx --debug", "waiting for a presented frame"]
        ms = 1e6
        interval = sample["interval_ns"] / ms
        fps = 1000.0 / interval if interval else 0.0
        lines = [
            f"debug {fps:5.1f} fps  frame {interval:5.1f} ms",
            f"update {sample['update_ns'] / ms:5.2f} render {sample['render_ns'] / ms:5.2f}"
            f" acquire {sample['acquire_ns'] / ms:5.2f}",
            f"submit {sample['submit_ns'] / ms:5.2f} present {sample['present_ns'] / ms:5.2f}",
        ]
        slow = sample.get("over_budget") or []
        if slow:
            lines.append(
                "sync >1 ms: "
                + ", ".join(f"{name} {value / ms:.2f} ms" for name, value in slow)
            )
        else:
            lines.append("sync >1 ms: none")
        return lines

    def _begin_application_frame(self, dt):
        self._last_dt = dt
        payload = {"dt": dt, "elapsed": self.elapsed, "frame": self._frame_index}
        self._frame_index += 1
        self._event_bus._pending.insert(0, ("frame", payload))
        self.input._begin_frame()
        self._pump_events()

    def _desktop_begin_frame(self, size, dpi, elapsed, events, world_available):
        # Dev-channel requests run at the frame barrier, before this frame's events
        # are delivered and the Rust scheduler dispatches its systems.
        if self._dev_channel is not None:
            self._dev_channel.run_jobs()
        self._record_metrics()
        self._finish_look()
        self._arm_look()
        self.elapsed = elapsed
        self._world._running = not world_available
        self._world._started = True
        try:
            if (self.w, self.h) != tuple(size) or self._gui_runtime.dpi != dpi or not self.input.viewport_size[0]:
                self.w, self.h = size
                self._gui_runtime.dpi = dpi
                resize = ResizeEvent(round(self.w / dpi), round(self.h / dpi), dpi)
                self.input._record_resize(resize)
                self.emit("resize", resize)
                if self.gui or self._debug_hud:
                    self._gui_runtime.sync()
            self._window_events(events)
            self._begin_application_frame(self._last_dt)
            return self._frame_system_updates()
        finally:
            self._world._running = True

    def reload_systems(self, *modules):
        """Declare the source modules that supply reloadable native Engine systems."""
        if self._started or self._registration_reload is not None:
            raise RuntimeError("declare reload systems once before starting the Engine")
        if not modules or any(
            not isinstance(name, str) or not all(part.isidentifier() for part in name.split("."))
            for name in modules
        ):
            raise ValueError("expected one or more absolute registration module names")
        if len(set(modules)) != len(modules):
            raise ValueError("duplicate registration module")
        self._registration_reload = _RegistrationReload(modules)

    def _current_application_systems(self):
        reload = self._registration_reload
        return [*self._application_systems.values(), *(reload.registrations if reload else ())]

    def _frame_system_updates(self):
        reload = self._registration_reload
        if reload is not None and _entry.get() is not None:
            if reload.poll(self._prepare_reload_systems):
                self._systems_changed = True
        if self._systems_changed:
            self._systems_changed = False
            return self._current_application_systems()
        return None

    def _prepare_reload_systems(self, callbacks):
        if len(set(callbacks)) != len(callbacks) or any(
            callback in self._application_systems for callback in callbacks
        ):
            raise ValueError("system is already registered")
        registrations = tuple(
            self._bind_application_system(callback, get_type_hints(callback).values(), register=False)
            for callback in callbacks
        )
        # Match frames.rs application_frame and validate_columns before publishing.
        # Only callbacks go through this lane and every column comes from this World.
        world = self._world
        application = [*self._application_systems.values(), *registrations]
        if len(world._systems) + len(application) > 32:
            raise ValueError("too many desktop systems (maximum 32)")
        for name, _, reads, _, _ in application:
            if self._position_binding is not None and self._position_binding[0] in reads:
                raise ValueError(f"{name}: resident column cannot use CPU reads")
            if sum(self._capacity * (17 + world._metadata[col][2] * 4) for col in reads) > 16 * 1024**2:
                raise ValueError(f"{name}: declared snapshot exceeds byte limit")
        return registrations

    def _desktop_finish_frame(self):
        if self.gui or self._gui_runtime._widgets or self._debug_hud:
            self._gui_runtime.sync(refresh=True)
        self._sync_scene()

    def component(self, cls):
        cls = _make_component_class(cls, None)
        width = sum(size for _, size in cls._component_start_idx.values())
        self._world.register(cls.__name__, kind="f32", width=width)
        return cls

    def system(self, callback=None, *, parallel=False):
        if parallel:
            raise NotImplementedError(
                "parallel Engine systems require explicit World access declarations"
            )
        if callback is None:
            return self.system
        if iscoroutinefunction(callback):
            raise NotImplementedError("the Rust base Engine requires synchronous systems")
        if callback in self._application_systems:
            raise ValueError("system is already registered")
        self._application_systems[callback] = self._bind_application_system(
            callback, callback.__annotations__.values(), register=True
        )
        self._systems_changed = True
        return callback

    def _bind_application_system(self, callback, hints, *, register):
        columns = []
        for hint in hints:
            if isinstance(hint, Query):
                components = (
                    hint.components if isinstance(hint.components, tuple) else (hint.components,)
                )
                for component in components:
                    if register and isinstance(component, type) and issubclass(component, Component):
                        self._register_component(component)
                    columns.append(self._column_key(component))
        columns = tuple(dict.fromkeys(columns))
        access = tuple(dict.fromkeys((*columns, "Transform", "Mesh")))

        def invoke(snapshot, commands):
            self._finish_texture_reads()
            if self._pending_tasks:
                from manifoldx.engine import Engine

                Engine._drain_tasks(self)
            clock = snapshot.clock
            self.elapsed = self._time_base + clock.time
            data = {key: snapshot.read(key) for key in columns}
            selected = None
            for value in data.values():
                ids = {entity for entity, member in zip(value.entities, value.membership) if member}
                selected = ids if selected is None else selected & ids
            entities = (
                tuple(entity for entity in next(iter(data.values())).entities if entity in selected)
                if data
                else ()
            )
            arrays = {}
            for key, value in data.items():
                rows = {entity: index for index, entity in enumerate(value.entities)}
                arrays[self._column_name(key)] = value.numpy()[
                    [rows[entity] for entity in entities]
                ]
            originals = {name: rows.copy() for name, rows in arrays.items()}
            query = ComponentView(
                _Rows(arrays), list(arrays), np.arange(len(entities), dtype=np.intp)
            )
            self._commands, self._entities = commands, entities
            self._snapshot = snapshot
            self._callback_origin = self._origin_vector()
            self._transform_edits = {}
            self._pending_destroy = []
            try:
                callback(query, clock.dt)
                self._commit_transforms(snapshot, commands, entities, arrays, originals)
                for key in columns:
                    name = self._column_name(key)
                    if name != "Transform" and not np.array_equal(arrays[name], originals[name]):
                        self._replace_rows(commands, key, entities, arrays[name])
                if self._pending_destroy:
                    commands.destroy(tuple(dict.fromkeys(self._pending_destroy)))
            finally:
                self._commands, self._entities = None, ()
                self._snapshot = None
                self._transform_edits.clear()

        invoke.__qualname__ = callback.__qualname__
        column_ids = self._world._columns(access)
        return self._world._system_registration(
            invoke, None, column_ids, [*column_ids, self._world._structural]
        )

    def remove_system(self, callback):
        try:
            del self._application_systems[callback]
        except KeyError as error:
            raise ValueError("system is not registered") from error
        self._systems_changed = True

    def _commit_transforms(self, snapshot, commands, entities, arrays, originals):
        offset = self._callback_origin - self._origin_vector()
        edits = self._transform_edits
        if "Transform" in arrays:
            rows = arrays["Transform"]
            changed = np.flatnonzero(np.any(rows != originals["Transform"], axis=1))
            for index in changed:
                entity, row = entities[index], rows[index]
                if entity not in edits:
                    row[:3] = row[:3].astype("float64") + offset
                    edits[entity] = row
        if np.any(offset):
            data = snapshot.read("Transform")
            for entity, member, row in zip(data.entities, data.membership, data.numpy()):
                if member and entity not in edits:
                    shifted = row.copy()
                    shifted[:3] = shifted[:3].astype("float64") + offset
                    edits[entity] = shifted
        if edits:
            self._replace_rows(
                commands, "Transform", tuple(edits), np.stack(tuple(edits.values()))
            )

    def _replace_rows(self, commands, key, entities, rows):
        """Replace rows in batches no larger than one command may carry.

        A system can write every row it queried in one frame; the native
        command lease accepts `World._max_command_rows` rows per command, so a
        large query (10k sprites, for example) is split into disjoint batches
        that the barrier applies in order.
        """
        limit = self._world._max_command_rows
        for start in range(0, len(entities), limit):
            commands.replace(
                key, entities[start : start + limit], rows[start : start + limit]
            )

    @staticmethod
    def _column_name(component):
        return component if isinstance(component, str) else component.__name__

    @staticmethod
    def _column_key(component):
        return (
            component
            if isinstance(component, type) and issubclass(component, Component)
            else _NativeEngine._column_name(component)
        )

    def compute(self, cls):
        """Register a Compute class to dispatch on the native GPU."""
        import weakref

        from manifoldx.compute import Compute

        if not isinstance(cls, type) or not issubclass(cls, Compute):
            raise TypeError(f"engine.compute(...) expects a Compute subclass; got {cls!r}")
        if cls in self._compute_classes:
            return cls

        columns = []
        for bindings, write in ((cls._reads, False), (cls._writes, True)):
            for component in bindings.values():
                name = component.__name__
                if getattr(component, "_gpu_only", False):
                    index, width = None, component._shape[0]
                else:
                    key = name if name in self._world._schemas else component
                    if key not in self._world._schemas:
                        self._register_component(component)
                    index = self._world._column(key)
                    width = self._world._metadata[index][2]
                    if self._position_binding is not None and index == self._position_binding[0]:
                        raise ValueError("Compute cannot alias the resident position column")
                columns.append((index, name, width, write))

        sentinels = {"frame_dt": 1, "entity_count": 2, "frame_index": 3}
        uniforms = []
        for name in cls._uniforms:
            default = cls._uniform_defaults.get(name)
            if isinstance(default, str):
                if default not in sentinels:
                    raise ValueError(f"unknown Compute uniform source {default!r} for {name}")
                uniforms.append((sentinels[default], 0.0))
            else:
                uniforms.append((0, float(0.0 if default is None else default)))

        callback = None
        dispatch = cls.dispatch
        if isinstance(dispatch, str):
            symbols = {"entity_count": 1, "max_entities": 2}
            if dispatch not in symbols:
                raise ValueError(f"unknown Compute dispatch size {dispatch!r}")
            threads = (symbols[dispatch], 0)
        elif callable(dispatch):
            engine = weakref.ref(self)

            def callback():
                return int(dispatch(engine()))

            threads = (3, 0)
        else:
            threads = (0, max(0, int(dispatch)))

        key = len(self._compute_classes)
        self._graphics.compute(
            (key, cls().compile(), cls.workgroup_size, uniforms, threads, columns),
            callback,
        )
        self._compute_classes[cls] = key
        self._compute_columns.update(index for index, _, _, _ in columns if index is not None)
        return cls

    def _register_component(self, component):
        if component not in self._world._schemas:
            self._world.register(component)

    def _mesh(self, geometry, *, reuse=None):
        registry = self._registries["Mesh"]
        key = id(geometry)
        if key in registry:
            return registry[key][0]
        index = next(self._mesh_ids) if reuse is None else reuse
        if "vertices" in geometry and np.size(geometry.get("indices")) == 2:
            indices = np.asarray(geometry["indices"], dtype=np.intp).reshape(2)
            positions = np.asarray(geometry["vertices"], dtype="f4")[indices]
            if positions.shape != (2, 3):
                raise ValueError("axis geometry requires two three-dimensional endpoints")
            self._graphics.axis(index, positions.tolist())
            registry[key] = (index, geometry)
            return index
        joints = []
        if "joints" in geometry:
            from manifoldx.skinning import pack_joints, validate_joints, validate_weights

            indices = validate_joints(geometry["joints"], len(geometry["positions"]))
            weights = validate_weights(geometry.get("weights"), indices)
            joints = pack_joints(indices, weights).tolist()
        if "indices" not in geometry:
            raise ValueError("native meshes require triangle indices")
        surfaces = np.asarray(geometry.get("surfaces", []), dtype="f4").reshape(-1)
        colors = np.asarray(geometry.get("colors", []))
        baked = surfaces == SURFACE_BAKED
        if baked.any():
            colors = np.array(geometry.get("colors", np.ones((len(surfaces), 3))), dtype="f4")
            uv = np.asarray(geometry.get("baked_uv", np.zeros((len(surfaces), 2))))
            colors[baked] = baked_columns(colors[baked], uv[baked])
        self._graphics.mesh(
            index,
            np.asarray(geometry["positions"]).tolist(),
            np.asarray(geometry["normals"]).tolist(),
            np.asarray(geometry["indices"]).reshape(-1).tolist(),
            np.asarray(geometry.get("uvs", [])).tolist(),
            (
                colors.tolist(), joints, surfaces.tolist(),
            ),
        )
        registry[key] = (index, geometry)
        return index

    def _replace_geometry(self, entities, geometry):
        if self._commands is None:
            raise RuntimeError("geometry swaps require a native frame command lease")
        data = self._snapshot.read("Mesh")
        owners = dict(zip(data.entities, zip(data.membership, data.numpy().ravel())))
        owned = set(entities)
        if any(entity not in owners or not owners[entity][0] for entity in owned):
            raise ValueError("stale geometry handle")
        old = {int(owners[entity][1]) for entity in owned}
        others = {int(index) for entity, (has_mesh, index) in owners.items()
                  if has_mesh and entity not in owned}
        reuse = next(iter(old)) if len(old) == 1 and old.isdisjoint(others) else None
        index = self._mesh(geometry, reuse=reuse)
        if index not in old or len(old) != 1:
            self._commands.replace("Mesh", entities, np.full((len(entities), 1), index, dtype="uint32"))
        registry = self._registries["Mesh"]
        for key, (mesh, obj) in list(registry.items()):
            if mesh in old and mesh not in others and obj is not geometry:
                del registry[key]
                if mesh != index:
                    self._graphics.stream_retire(mesh)

    def clear_scene(self):
        """Release scene-owned geometry and materials between native frames."""
        self._world._owner()
        self.gui.clear()
        self._gui_runtime.reset()
        for streamer in list(self._streamers):
            streamer.close()
        snapshot = self._world._native.snapshot()
        entities = self._world._entities(snapshot["handles"])
        self._world._native.destroy(self._world._handles(entities))
        meshes = self._registries["Mesh"]
        materials = self._registries["Material"]
        self._graphics.retire_scene([entry[0] for entry in meshes.values()],
                                    [entry[0] for entry in materials.values()])
        meshes.clear()
        for key in self._view_materials.values():
            self._manual_registry._unregister(materials[key][1])
        materials.clear()
        self._view_materials.clear()
        self._graphics.take_unused_view_materials()

    def submit_process(self, fn, *args):
        """Submit CPU preparation to a spawned process without sharing GPU objects."""
        from manifoldx.engine import Engine

        return Engine.submit_process(self, fn, *args)

    def _collect_unused_view_materials(self):
        if not self._view_materials:
            return
        registry = self._registries["Material"]
        retired = []
        for index in self._graphics.take_unused_view_materials():
            key = self._view_materials.pop(index, None)
            if key is not None:
                self._manual_registry._unregister(registry[key][1])
                del registry[key]
                retired.append(index)
        if retired:
            self._graphics.retire_scene([], retired)

    def _track_view_material(self, index, key, samples_view):
        if samples_view:
            self._view_materials[index] = key
        else:
            self._view_materials.pop(index, None)

    def _material(self, material):
        # Drain before handing out an id so reuse after an empty frame registers anew.
        self._collect_unused_view_materials()
        registry = self._registries["Material"]
        key = id(material)
        if getattr(material, "_version", None) is None:
            material._version = _Counter()
        version = material._version.value
        cached = registry.get(key)
        if cached is not None and cached[2] == version:
            return cached[0]
        index = next(self._material_ids) if cached is None else cached[0]
        if isinstance(material, ShaderMaterial):
            self._graphics.fragment(
                index, material._fragment.source, material._uniforms, material._textures
            )
            registry[key] = (index, material, version)
            return index
        types = {
            BasicMaterial: "basic",
            FlatMaterial: "flat",
            PhongMaterial: "phong",
            StandardMaterial: "standard",
        }
        kind = types.get(type(material))
        if kind is None:
            from manifoldx.viz import ColormapMaterial, LabelMaterial, VolumeMaterial, colormaps

            if isinstance(material, ColormapMaterial):
                self._graphics.sprite(
                    index, material._compile(),
                    material.get_data(1, None)[0].tolist(), material.get_lut().tobytes(),
                )
            elif isinstance(material, LabelMaterial):
                self._graphics.label_material(
                    index, material._compile(), material.get_data(1, None)[0].tolist(),
                )
            elif isinstance(material, VolumeMaterial):
                self._graphics.volume_material(
                    index,
                    ((material.vmin, material.vmax, material.density_scale),
                     material.step_size, material.max_steps),
                    colormaps.get_colormap(material.cmap).tobytes(),
                    material.opacity_lut.tolist(),
                )
            else:
                samples_view = self._manual_material(index, material)
                self._track_view_material(index, key, samples_view)
            registry[key] = (index, material, version)
            return index
        params = None
        effects = None
        view_albedo = False
        if kind == "standard":
            baked = material.baked
            if baked is not None and not self._owns_texture(baked):
                raise ValueError("baked texture array belongs to a different graphics owner")
            texture = material.albedo_map
            if texture is not None and not self._owns_texture(texture):
                raise ValueError("albedo texture belongs to a different graphics owner")
            view_albedo = texture is not None and texture._native_view
            params = (
                material.roughness,
                material.metallic,
                material.ao,
                None if texture is None else texture.id,
            )
            effects = (
                material.vertex_colors,
                material.opacity,
                material.mirror,
                material.cut,
                material.sway,
                material.sway_reach,
                material.ripple,
                material.faded,
                material.tinted,
                material.paletted,
                None if baked is None else baked.id,
            )
        shininess = material.shininess if kind == "phong" else None
        self._graphics.material(
            index, _color(material.color), kind, params, shininess, effects, view_albedo
        )
        if kind == "standard" and material.vertex_colors and material.albedo_map is None:
            self._graphics.surface_patterns(
                index, [material.surface_patterns[i] for i in range(1, 10)], material.detail
            )
        registry[key] = (index, material, version)
        self._track_view_material(index, key, view_albedo)
        return index

    def _manual_material(self, index, material):
        from manifoldx.viz import AxisMaterial

        if not isinstance(material, MaterialBase):
            raise TypeError("a manual WGSL material must extend Material")
        subtype = getattr(material, "pipeline_subtype", None)
        textures = []
        samples_view = False
        slots = {0, 1, 2, 3}
        for binding, handle in sorted(material.get_texture_bindings().items()):
            if type(binding) is not int or not 4 <= binding < 999:
                raise ValueError("manual WGSL sampler indices must follow the fixed mesh bindings")
            if binding in slots or binding + 1 in slots:
                raise ValueError("manual WGSL sampler and texture binding indices overlap")
            if not self._owns_texture(handle):
                raise ValueError("manual WGSL texture belongs to a different graphics owner")
            slots.update((binding, binding + 1))
            textures.append((binding, handle.id))
            samples_view |= handle._native_view
        if not isinstance(material.uniform_type(), dict):
            raise TypeError("manual WGSL uniform_type() must return a dict")
        self._manual_registry.register(material)
        rows = np.asarray(material.get_data(1, self._manual_registry), dtype="f4")
        if rows.ndim not in (1, 2) or (rows.ndim == 2 and len(rows) != 1):
            raise ValueError("manual WGSL get_data() must return one uniform row")
        pipeline_state = getattr(material, "pipeline_state", None) or (False, False)
        blend = 2 if subtype == "scatter" else int(
            subtype in ("glow", "cloud") or pipeline_state[0]
        )
        textured = subtype == "textured" and getattr(material, "albedo_map", None) is not None
        source = material._compile(textured=True) if textured else material._compile()
        self._graphics.manual(
            index, source, rows.reshape(-1).tolist(), textures, blend,
            isinstance(material, AxisMaterial), samples_view,
        )
        return samples_view

    def render_to_texture(self, camera, size, *, live=True, background=None, fog=False):
        """Schedule a secondary camera to render into a sampled texture before main."""
        from manifoldx.engine import _clear_color

        width, height = map(int, size)
        if not 1 <= width <= 4096 or not 1 <= height <= 4096:
            raise ValueError("view dimensions must be between 1 and 4096 pixels")
        clear = _clear_color(self.background_color if background is None else background)
        self._textures += 1
        index = self._textures
        self._graphics.create_view(
            index, _camera_input(camera), _camera_layers(camera), (width, height),
            clear, bool(live), bool(fog),
        )
        self._views[index] = _View(camera, bool(live))
        handle = TextureHandle(index, None, None, None, (width, height))
        return self._track_texture(handle, view=True)

    def _track_texture(self, handle, *, view=False):
        handle._native_owner = self
        handle._native_view = view
        self._texture_handles[handle.id] = ref(handle)
        finalize(handle, _release_native_texture, ref(self), handle.id, view)
        return handle

    def _owns_texture(self, handle):
        known = self._texture_handles.get(handle.id)
        return not self._closed and known is not None and known() is handle

    def _release_view(self, index):
        self._views.pop(index, None)
        self._graphics.release_view(index)

    def _view_id(self, handle):
        if not self._owns_texture(handle):
            raise ValueError("texture belongs to a different or closed graphics owner")
        if not getattr(handle, "_native_view", False):
            raise ValueError("expected a render-to-texture handle")
        return handle.id

    def _active_view(self, handle):
        view = self._views.get(self._view_id(handle))
        if view is None:
            raise ValueError("the texture view has been dropped")
        return view

    def refresh_texture(self, handle):
        self._active_view(handle).pending = True

    def set_texture_live(self, handle, live):
        self._active_view(handle).live = bool(live)

    def drop_texture(self, handle):
        index = self._view_id(handle)
        if self._views.pop(index, None) is not None:
            self._graphics.drop_view(index)

    def read_texture(self, handle):
        """Return a future for the texture's sRGB pixels after the next draw."""
        index = self._view_id(handle)
        future = Future()
        self._texture_reads[self._graphics.read_view(index)] = (handle, future)
        return future

    def _finish_texture_reads(self):
        for ticket, (handle, future) in list(self._texture_reads.items()):
            try:
                pixels = self._graphics.take_view_read(ticket)
                if pixels is None:
                    continue
                if future.set_running_or_notify_cancel():
                    width, height = handle.size
                    future.set_result(np.frombuffer(pixels, np.uint8).copy().reshape(height, width, 3))
            except RuntimeError as exc:
                if future.set_running_or_notify_cancel():
                    future.set_exception(exc)
            del self._texture_reads[ticket]

    def _upload_native_albedo(self, image, source):
        self._textures += 1
        index = self._textures
        width, height = image.size
        self._graphics.albedo(
            index, width, height, [level.tobytes() for level in _mip_chain(image)]
        )
        texture = TextureHandle(index, None, None, None, (width, height), source)
        return self._track_texture(texture)

    def _upload_native_baked_array(self, layers, size):
        self._textures += 1
        index = self._textures
        self._graphics.baked_array(index, size, layers)
        texture = TextureHandle(index, None, None, None, (size, size), layers=layers)
        return self._track_texture(texture)

    def configure_stream_collision(self, cell_size):
        """Create the E3 terrain index that receives streamed collision tiles."""
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("streamed collision tile size must be finite and positive")
        self._graphics.stream_terrain(cell_size)

    def sample_stream_collision(self, x, z, near):
        """Read the current E3 surface from the published streaming terrain."""
        return self._graphics.stream_sample(x, z, near)

    def _write_native_texture(self, handle, rgba, x, y, layer):
        if not self._owns_texture(handle) or handle._native_view:
            raise ValueError("texture is a view or belongs to a different graphics owner")
        height, width = rgba.shape[:2]
        self._graphics.texture_patch(handle.id, (x, y, layer), (width, height), rgba.tobytes())

    def _write_native_baked_layer(self, handle, layer, levels):
        if not self._owns_texture(handle):
            raise ValueError("baked texture array belongs to a different graphics owner")
        if not 0 <= layer < handle.layers:
            raise ValueError("baked texture layer index is outside the array")
        size = handle.size[0]
        if not 1 <= len(levels) <= size.bit_length():
            raise ValueError("invalid baked texture layer mip count")
        payload = []
        for level, image in enumerate(levels):
            side = max(1, size >> level)
            if image.shape != (side, side, 4) or image.dtype != np.uint8:
                raise ValueError(f"baked texture mip {level} must be a {side}x{side} RGBA uint8 image")
            payload.append(image.tobytes())
        self._graphics.baked_layer(handle.id, layer, payload)

    def spawn(self, *components, n=1):
        if n == 0:
            return None
        geometry = next((c._geometry for c in components if isinstance(c, Mesh)), None)
        material = next((c._material for c in components if isinstance(c, Material)), None)
        if (
            isinstance(material, ShaderMaterial)
            and material._fragment.requires_uv
            and (geometry is None or "uvs" not in geometry)
        ):
            raise ValueError("the fragment requires mesh UV coordinates")
        payload = {}
        for component in components:
            key = self._column_key(type(component))
            if key == "Mesh":
                rows = np.full((n, 1), self._mesh(component._geometry), dtype="uint32")
            elif key == "Material":
                rows = np.full((n, 1), self._material(component._material), dtype="uint32")
            else:
                if isinstance(component, Component):
                    self._register_component(type(component))
                rows = component.get_data(n)
                if key == "Transform" and component._pos is not None:
                    rows[:, :3] = (
                        np.asarray(component._pos, dtype="float64") - self._origin_vector()
                    )
            if key in payload:
                raise ValueError(f"duplicate spawn component {self._column_name(key)}")
            payload[key] = rows
        if self._commands is not None:
            return self._commands.spawn(count=n, payload=payload)
        return _EntityHandle(self, self._world.spawn(count=n, payload=payload))

    @staticmethod
    def _joint_matrices(matrices):
        from manifoldx.skinning import MAX_JOINTS

        matrices = np.asarray(matrices, dtype=np.float32)
        if (
            matrices.ndim != 4
            or matrices.shape[2:] != (4, 4)
            or not 1 <= matrices.shape[1] <= MAX_JOINTS
        ):
            raise ValueError("expected joint matrices of shape (N, J, 4, 4) with 1 <= J <= 32")
        return matrices.reshape(len(matrices), matrices.shape[1], 16).tolist()

    def _set_joint_poses(self, identities, matrices):
        poses = self._joint_matrices(matrices)
        if len(identities) != len(poses):
            raise ValueError("entity and joint pose counts must match")
        self._graphics.joint_poses(list(zip(identities, poses)))

    def set_joints(self, indices, matrices):
        self._set_joint_poses([(int(index), None) for index in indices], matrices)

    def bake_poses(self, frames, into=None):
        return self._graphics.bake_poses(self._joint_matrices(frames), into)

    def set_pose_frames(self, indices, clip, positions):
        indices = list(indices)
        clips = np.broadcast_to(np.asarray(clip, dtype=np.int64), (len(indices),))
        positions = np.broadcast_to(np.asarray(positions, dtype=np.float64), (len(indices),))
        frames = []
        for index, clip_id, position in zip(indices, clips, positions):
            frames.append(((int(index), None), int(clip_id), float(position)))
        self._graphics.pose_frames(frames)

    def destroy(self, indices):
        if self._commands is None:
            raise RuntimeError("destroy requires an active native system")
        self._pending_destroy.extend(self._entities[i] for i in indices)

    def set_lights(self, lights):
        lights = list(lights)
        if any(type(light) is not PointLight for light in lights):
            raise NotImplementedError("the native base renderer supports point lights")
        self._lights = lights

    def add_light(self, light):
        self.set_lights([*self._lights, light])

    def set_sun(self, light):
        if light is not None and not isinstance(light, DirectionalLight):
            raise TypeError("sun must be a DirectionalLight or None")
        self._sun = light

    def set_spot(self, light):
        if light is not None and not isinstance(light, SpotLight):
            raise TypeError("spot must be a SpotLight or None")
        self._spot = light

    def enable_shadows(
        self,
        target=(0, 0, 0),
        extent=10.0,
        resolution=2048,
        near=0.1,
        far=50.0,
        bias=0.005,
        pcf_radius=1,
        auto_fit=True,
        cascades=1,
        shadow_distance=None,
    ):
        options = (
            tuple(target),
            extent,
            int(resolution),
            near,
            far,
            bias,
            int(pcf_radius),
            bool(auto_fit),
            int(cascades),
            shadow_distance,
        )
        self._graphics.shadows(options)
        self._shadow_options = options

    def set_post(self, **effects):
        from manifoldx.post import Post, ssao_kernel

        post = (self._post or Post()).changed(**effects)
        options = None
        if post.active():
            s, b, sh, g, sky = post.ssao, post.bloom, post.shafts, post.grade, post.sky
            samples = min(32, max(0, int(s.samples))) if s else 0
            options = (
                None if s is None else (s.radius, s.intensity, samples, s.bias, s.power),
                None if b is None else (b.threshold, b.knee, b.intensity),
                None if sh is None else (sh.intensity, sh.decay, sh.density, sh.samples),
                None if g is None else (g.saturation, g.contrast, g.vignette),
                None
                if sky is None
                else (sky.zenith, sky.horizon, sky.sun_size, sky.sun_intensity, sky.glow),
                post.fxaa,
                ssao_kernel(samples).tolist(),
            )
        self._graphics.post(options)
        self._post = post
        return post

    def enable_fog(self, start, end, color=None):
        """Apply linear distance fog to Standard materials."""
        self.fog_enabled = True
        self.fog_start, self.fog_end = float(start), float(end)
        self.fog_color = tuple(self.background_color if color is None else color)

    def set_environment(self, env):
        """Set the image-based lighting environment, preset, or None."""
        from manifoldx.ibl import EnvironmentMap, PRESETS

        if isinstance(env, str):
            if env not in PRESETS:
                raise ValueError(f"Unknown preset {env!r}. Known: {list(PRESETS)}")
            env = PRESETS[env]()
        elif env is not None and not isinstance(env, EnvironmentMap):
            raise TypeError(f"Expected EnvironmentMap, preset string, or None; got {type(env)}")
        self._environment = env

    @property
    def environment(self):
        """The active image-based lighting environment."""
        return self._environment

    def _sync_environment(self):
        from manifoldx.ibl import load_brdf_lut

        env = self._environment
        if env is not None and env is not self._native_environment:
            env._precompute()
            brdf = load_brdf_lut() if self._native_environment is None else None
            self._graphics.environment_maps(
                env._irradiance.astype("<f2", copy=False).tobytes(),
                [mip.astype("<f2", copy=False).tobytes() for mip in env._prefiltered],
                None if brdf is None else brdf.astype("<f2", copy=False).tobytes(),
            )
            self._native_environment = env
        self._graphics.environment(None if env is None else (env.intensity, env.show_skybox))

    def register_volume(self, data, *, name=None):
        """Register a contiguous float32 (z, y, x) scalar field for DVR."""
        handle = self._volume_registry.register(data, name=name)
        self._graphics.volume(handle, data.shape[::-1], data.tobytes())
        return handle

    def update_volume(self, handle, data):
        """Replace a volume's voxels without changing its dimensions."""
        self._volume_registry.update(handle, data)
        self._graphics.volume(handle, data.shape[::-1], data.tobytes())

    def get_label_atlas(self):
        """Return the atlas of world and screen labels."""
        from manifoldx.viz.text import LabelTextureAtlas

        if self._label_atlas is None:
            self._label_atlas = LabelTextureAtlas()
        return self._label_atlas

    def _sync_scene(self):
        self._collect_unused_view_materials()
        atlas = self._label_atlas
        if atlas is not None:
            for layer in sorted(atlas.dirty_slices):
                self._graphics.label_tile(layer, atlas._slices[layer].tobytes())
            atlas.clear_dirty()
        self._sync_environment()
        self._graphics.fog(
            ((self.fog_start, self.fog_end), self.fog_color) if self.fog_enabled else None
        )
        self._graphics.animation(self.elapsed, self._origin, list(self.benders)[:8])
        lines = self.lines
        if lines.layers or self._line_rows:
            layers = [
                (
                    key,
                    lines.looks.get(key, (1.0, 1.0)),
                    None if self._line_rows.get(key) is data else data.tolist(),
                )
                for key, data in lines.layers.items()
            ]
            self._graphics.lines(lines.visible, lines.bias, lines.feather, layers)
            self._line_rows = lines.layers
        for _, material, version in self._registries["Material"].values():
            if material._version.value != version:
                self._material(material)
        self._graphics.layers(_camera_layers(self.camera))
        self._graphics.camera(_camera_input(self.camera))
        for index, view in self._views.items():
            self._graphics.update_view(
                index, _camera_input(view.camera), _camera_layers(view.camera), view.live, view.pending,
            )
            view.pending = False
        lights = [
            (list(light.position), _color(light.color), light.intensity)
            for light in self._lights[:4]
        ]
        sun = self._sun
        sun_values = (
            None if sun is None else (list(sun.direction), _color(sun.color), sun.intensity)
        )
        spot = self._spot
        spot_values = None
        if spot is not None:
            spot_values = (
                list(spot.position),
                list(spot.direction),
                _color(spot.color),
                spot.intensity,
                spot.distance or 30.0,
                np.cos(spot.inner_angle),
                np.cos(spot.outer_angle),
            )
        self._graphics.settings(self.background_color, lights, sun_values, spot_values)

    def set_fixed_timestep(self, dt):
        if not 0.001 <= dt <= 1:
            raise ValueError("native frame dt must be between 0.001 and 1 second")
        self._fixed_dt = dt

    def _start(self):
        if self._closed:
            raise RuntimeError("native Engine is closed")
        if not self._started:
            self._started = True
            self._event_bus.dispatch_immediate(self, "startup", {})
            if self._registration_reload is not None:
                self._registration_reload.prepare(self._prepare_reload_systems)
                self._systems_changed = True
            if self.gui:
                self._gui_runtime.sync()
            self._sync_scene()
        return not self._quit

    def quit(self):
        self._quit = True
        if self._world._running or self._desktop_ran:
            self._world.stop()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.quit()
        self._fail_looks(RuntimeError("the native window closed before a frame was presented"))
        self._finish_texture_reads()
        for _, future in self._texture_reads.values():
            future.cancel()
        self._texture_reads.clear()
        for streamer in list(self._streamers):
            streamer.close()
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=False, cancel_futures=True)
        try:
            if self._started:
                from manifoldx.engine import Engine

                Engine.shutdown_events(self)
        finally:
            self._gui_runtime.reset()
            self.gui.clear()
            self._graphics.close()
            self._views.clear()
            self._texture_handles.clear()
            for registry in self._registries.values():
                registry.clear()

    def run(self):
        if current_thread() is not main_thread():
            raise RuntimeError("run the native desktop Engine on the main thread")
        if self.last_frame is not None or self._desktop_ran:
            raise RuntimeError("native desktop Engine requires a fresh window session")
        self._desktop_ran = True
        frames = os.environ.get("MANIFOLDX_DESKTOP_FRAMES")
        monitor = self.monitor if self.monitor is not None else os.environ.get("MANIFOLDX_MONITOR")
        hz = 1 / self._fixed_dt if self._fixed_dt is not None else min(60.0, self.max_fps)
        self._last_dt = 1 / hz
        debug = self._dev_channel is not None and self._dev_channel.meter is not None
        self._debug_hud = debug
        options = (
            self.title,
            self.w,
            self.h,
            self.fullscreen,
            None if monitor is None else str(monitor),
            os.environ.get("MANIFOLDX_PRESENT", "fifo"),
            hz,
            int(frames) if frames else None,
            os.environ.get("MANIFOLDX_GPU_TIMINGS", "1") != "0",
        )

        def startup():
            ready = self._start()
            self._world._running = ready
            return ready

        try:
            self.desktop_report = self._world._loop.run_desktop(
                self._graphics,
                options,
                self._world._systems,
                startup,
                self._desktop_key,
                self._position_binding,
                self._desktop_probe,
                (self._desktop_begin_frame, self._desktop_finish_frame),
                debug=debug,
            )
            self.elapsed = self.desktop_report["scheduler_frames"] / hz
            self.desktop_report["gui_getter_calls"] = self._gui_runtime.getter_total
            if location := os.environ.get("MANIFOLDX_DESKTOP_REPORT"):
                Path(location).write_text(json.dumps(self.desktop_report, indent=2) + "\n")
        finally:
            self._world._running = False
            if self._position_binding is not None:
                self._world._resident_closed = self._position_binding[0]
            self.close()

    def gpu_timings(self):
        """The GPU time of the newest frame the Rust core measured, in milliseconds.

        The native desktop loop measures only a bounded diagnostic session
        (`MANIFOLDX_DESKTOP_FRAMES`): one timestamp pair around each presented
        frame's main render pass, resolved and read back once after the last
        frame, so a normal run never reads timestamps. This answers after
        `run()` returns, with `main` (the measured pass) and `total` (the same
        single interval). It is None during a run, in a session without a frame
        limit, with `MANIFOLDX_GPU_TIMINGS=0`, or on an adapter without
        timestamp queries; `desktop_report["gpu_timings"]["state"]` says which.
        """
        report = self.desktop_report
        if report is None or (report.get("gpu_timings") or {}).get("state") != "measured":
            return None
        measured = [
            sample["gpu_render_ns"]
            for sample in report.get("samples", [])
            if sample.get("gpu_render_ns") is not None
        ]
        if not measured:
            return None
        milliseconds = measured[-1] / 1e6
        return {"main": milliseconds, "total": milliseconds}

    def render_frame(self, *, supersample=1, pass_="beauty", groups=None, dt=1 / 30):
        if self._desktop_ran:
            raise RuntimeError("presented and offscreen targets require separate Engine sessions")
        if self._position_binding is not None:
            raise NotImplementedError("resident position binding requires the native desktop mode")
        if pass_ not in ("beauty", "ids"):
            raise ValueError(f"pass_ must be 'beauty' or 'ids', got {pass_!r}")
        if not isinstance(supersample, (int, np.integer)) or supersample < 1:
            raise ValueError("supersample must be a positive integer")
        supersample = int(supersample)
        if pass_ == "beauty" and groups is not None:
            raise ValueError("groups are only used for the ID pass")
        if not self._start():
            raise RuntimeError("native Engine quit during startup")

        labels = None
        compute_dt = None
        if pass_ == "ids":
            labels = {}
            if groups is not None:
                for label, members in enumerate(groups, start=1):
                    if label >= 1 << 24:
                        raise ValueError("the ID pass supports at most 16,777,215 groups")
                    for member in members:
                        slot = int(member)
                        if 0 <= slot < self._capacity:
                            labels[slot] = label
        else:
            step = self._fixed_dt if self._fixed_dt is not None else dt
            compute_dt = step
            self._begin_application_frame(step)
            application = self._current_application_systems()
            updates = self._frame_system_updates()
            if updates is not None:
                application = updates
            self._world._loop = self._frame_loop(self._world._native, 64, 8, 4096)
            self._world._running = True
            self._time_base = self.elapsed
            try:
                systems = [*self._world._systems, *application]
                self._world._loop.run(
                    1, 1 / step if step else 60, False, systems, None, freeze_clock=step == 0,
                )
            finally:
                self._world._running = False
            self.elapsed = self._time_base + step

        if self.gui or self._gui_runtime._widgets:
            self._gui_runtime.sync(supersample, refresh=labels is None)
        self._sync_scene()
        width, height = self.w * supersample, self.h * supersample
        captured = self._graphics.capture(self._world._native, width, height, labels, compute_dt)
        self._finish_texture_reads()
        rgba = captured.pop("rgba")
        captured["gui_getter_calls"] = self._gui_runtime.getter_calls if labels is None else 0
        self.last_frame = captured
        rgb = np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        if labels is not None:
            from manifoldx.engine import _block_mode

            red, green, blue = np.moveaxis(rgb.astype(np.int32), -1, 0)
            return _block_mode(red | (green << 8) | (blue << 16), supersample)
        if supersample == 1:
            return rgb.copy()
        return np.round(
            rgb.reshape(self.h, supersample, self.w, supersample, 3).mean(axis=(1, 3))
        ).astype(np.uint8)

    def cli(self, **options):
        from manifoldx.engine import Engine

        return Engine.cli(self, **options)

    def render(
        self, *, output, fps=30, duration=None, frame_count=None, quality="high", codec="h264"
    ):
        from manifoldx.engine import Engine

        if (duration is None) == (frame_count is None) or fps <= 0:
            raise ValueError("provide either duration or frame_count, with a positive fps")
        count = int(duration * fps) if frame_count is None else frame_count
        if count < 1:
            raise ValueError("video requires at least one frame")
        writer = Engine._get_video_writer(self, output, fps, self.w, self.h, codec, quality)
        try:
            for _ in range(count):
                writer.send(self.render_frame(dt=1 / fps))
        except BaseException:
            writer.discard()
            raise
        else:
            writer.close()
        finally:
            self.close()
