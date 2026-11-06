"""Engine.export_gltf: the engine's scene to a .glb, and a report of everything else."""

import inspect
from pathlib import Path

import numpy as np

from manifoldx.gltf.report import ExportIncomplete, ExportReport
from manifoldx.gltf.writer import Node, build_glb


def _where(obj):
    obj = getattr(obj, "__func__", obj)
    try:
        return f"{inspect.getsourcefile(obj)}:{inspect.getsourcelines(obj)[1]}"
    except (OSError, TypeError):
        return None


def _label(obj):
    return getattr(obj, "__qualname__", None) or type(obj).__name__


def _internal(func):
    """manifoldx's own handlers (the GUI bridge) are not the app's code."""
    module = getattr(getattr(func, "__func__", func), "__module__", "") or ""
    return module == "manifoldx" or module.startswith("manifoldx.")


def _widgets(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _widgets(child)


def code_report(engine):
    r = ExportReport()
    for system in engine.systems._systems:
        r.add("system", _label(system.func), "dropped", "custom Python that runs every frame", _where(system.func))
    for event, handlers in engine._event_bus._handlers.items():
        if event == "startup":  # the export runs it: it is how most apps spawn
            continue
        for h in handlers:
            if not _internal(h.func):
                r.add("handler", f'on("{event}") {_label(h.func)}', "dropped", "event handlers are not exported",
                      _where(h.func))
    for panel in engine.gui:
        r.add("gui", type(panel).__name__, "dropped", "glTF has no GUI")
        for widget in _widgets(panel):
            for attr, value in vars(widget).items():
                if callable(value) and not isinstance(value, type):
                    r.add("gui", f"{type(widget).__name__}.{attr.lstrip('_')} = {_label(value)}", "dropped",
                          "Python callbacks cannot run outside manifoldx", _where(value))
    for cls in engine._compute_runner._registered:
        r.add("compute", _label(cls), "dropped", "GPU compute kernels are not exported", _where(cls))
    for _task in engine._pending_tasks:
        r.add("task", "background task", "dropped", "pending background work is not exported")
    return r


def scene_extras(engine):
    extras = {"background": [float(c) for c in engine.background_color[:3]]}
    if engine.fog_enabled:
        extras["fog"] = {"start": engine.fog_start, "end": engine.fog_end,
                         "color": [float(c) for c in engine.fog_color[:3]]}
    if engine._shadow_config is not None:
        extras["shadows"] = {k: list(v) if isinstance(v, tuple) else v for k, v in engine._shadow_config.items()}
    return extras


def _lookup(table):
    """A dict, or a function evaluated after startup (when startup-spawned entities exist)."""
    if table is None:
        return lambda _i: None
    return table if callable(table) else table.get


def nodes(engine, names, extras, report):
    from manifoldx.resources import BasicMaterial

    unlit = None  # the renderer draws a mesh without a Material as a white BasicMaterial
    name_of, extras_of = _lookup(names), _lookup(extras)
    comps = engine.store._components
    out = []
    for i in np.where(engine.store._alive)[0]:
        i = int(i)
        gid = int(comps["Mesh"][i, 0]) if "Mesh" in comps else 0
        mid = int(comps["Material"][i, 0]) if "Material" in comps else 0
        material = engine._material_registry.get(mid) if mid else None
        if gid == 0:
            if material is not None:
                report.add("entity", type(material).__name__, "dropped",
                           "drawn without a mesh (point cloud, label or volume); glTF has no equivalent")
            continue
        if material is None:
            unlit = unlit or BasicMaterial("#ffffff")
            material = unlit
        t = comps["Transform"][i]
        out.append(Node(engine._geometry_registry.get(gid), material,
                        pos=tuple(t[0:3]), rot=tuple(t[3:7]), scale=tuple(t[7:10]),
                        name=name_of(i) or f"entity_{i}",
                        extras={**(extras_of(i) or {}), "mx": {"entity": i, "material": mid}}))
    return out


def export_engine(engine, path, *, names=None, extras=None, strict=False):
    if engine._device is None and engine._event_bus._handlers.get("startup"):
        engine._ensure_offscreen(1)  # startup handlers may need the device (load_texture)
    report = code_report(engine)
    lights = [light for light in (engine._sun, engine._spot) if light is not None] + list(engine._lights)
    data, written = build_glb(nodes(engine, names, extras, report), camera=engine.camera, lights=lights,
                              environment=engine._environment, scene_extras=scene_extras(engine),
                              extra_materials=list(engine._material_registry._materials.items()))
    report.extend(written)
    report.print()
    if strict and len(report):
        raise ExportIncomplete(report)
    path = Path(path)
    path.write_bytes(data)
    report.write_json(path.with_suffix(".export-report.json"))
    return report
