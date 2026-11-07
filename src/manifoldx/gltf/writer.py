"""Map manifoldx geometry and materials to glTF. Returns what it could not map."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldx.gltf.glb import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, GlbBuilder
from manifoldx.gltf.report import ExportReport

_LINEAR, _LINEAR_MIPMAP_LINEAR, _REPEAT = 9729, 9987, 10497


@dataclass
class Node:
    geometry: object  # geometry dict or anything with to_geometry() (modeling.Mesh)
    material: object = None
    pos: tuple = (0.0, 0.0, 0.0)
    rot: tuple = (0.0, 0.0, 0.0, 1.0)  # quaternion x, y, z, w
    scale: tuple = (1.0, 1.0, 1.0)
    name: str | None = None
    extras: dict | None = None


def build_glb(nodes, *, camera=None, lights=(), environment=None, scene_extras=None, extra_materials=None,
              geometries=None, drawn_materials=()):
    """extra_materials: (id, material) pairs written even when no node uses them; the
    scene's extras map each id to its glTF material index under "mx_materials"
    (a material glTF cannot express is left out). geometries: (id, geometry) pairs, each
    written once as a mesh with no material and no node, mapped under "mx_geometries".
    drawn_materials: the material of each entity drawn from the library, reported as a node's is."""
    w = _Writer()
    for i, node in enumerate(nodes):
        w.node(node, i)
    for mat in drawn_materials:
        _index, note = w.material(mat)
        if note is not None:
            w.report.add("material", *note)
    mx_geometries = {str(gid): w.library(geometry) for gid, geometry in geometries or ()}
    mx_materials = {}
    for mid, mat in extra_materials or ():
        index, _note = w.material(mat)  # a note only matters for drawn materials, reported by node()
        if index is not None:
            mx_materials[str(mid)] = index
    if camera is not None:
        w.camera(camera)
    for light in lights:
        w.light(light)
    extras = dict(scene_extras or {})
    if extra_materials is not None:
        extras["mx_materials"] = mx_materials
    if geometries is not None:
        extras["mx_geometries"] = mx_geometries
    if environment is not None:
        extras["environment"] = w.environment(environment)
    return w.finish(extras), w.report


def export_gltf(path, nodes, **scene):
    data, report = build_glb(nodes, **scene)
    Path(path).write_bytes(data)
    return report


def _look_rotation(forward, up=(0.0, 1.0, 0.0)):
    """Quaternion (x, y, z, w) that turns local -Z to `forward` with local +Y towards `up`."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    if abs(f @ u) > 0.999:  # looking straight along up: any perpendicular up will do
        u = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    z = -f
    x = np.cross(u, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.column_stack([x, y, z])
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = (s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s)
    return [float(c) for c in q]


def _light_rgb(color):
    """Light colours as the renderer uploads them: hex over 255, no sRGB decode."""
    if isinstance(color, str):
        h = color.lstrip("#")
        return [int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4)]
    return [float(c) for c in color[:3]]


def _rgba(color):
    from manifoldx.resources import material_rgb

    alpha = 1.0 if isinstance(color, str) or len(color) == 3 else float(color[3])
    return [*map(float, material_rgb(color)), alpha]


def _uses_colors(mat):
    """The textured pipeline ignores vertex colours, so glTF must not multiply them in."""
    return bool(getattr(mat, "vertex_colors", False)) and getattr(mat, "albedo_map", None) is None


def _image(path):
    data = Path(path).read_bytes()
    if data[:3] == b"\xff\xd8\xff":
        return data, "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data, "image/png"
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.open(path).save(buf, format="PNG")
    return buf.getvalue(), "image/png"


class _Writer:
    def __init__(self):
        self.b = GlbBuilder()
        self.report = ExportReport()
        self._meshes = {}  # (id(geometry), material index, colours) -> mesh index
        self._geometry = {}  # (id(geometry), colours) -> (attributes, indices accessor)
        self._materials = {}  # id(material) -> (index or None, note or None)
        self._textures = {}  # TextureHandle.id -> texture index or None
        self._sampler = None
        self._roots = []

    def node(self, node, i):
        name = node.name or f"node_{i}"
        if node.extras is not None:
            try:
                json.dumps(node.extras, allow_nan=False)
            except (TypeError, ValueError) as e:
                raise type(e)(f"extras of node {name!r} are not JSON-serialisable: {e}") from None
        mat, note = self.material(node.material)
        if note is not None:
            self.report.add("material", *note)
            if note[1] == "dropped" and note[0] != "StandardMaterial.ao":
                return
        colors = _uses_colors(node.material)
        key = (id(node.geometry), mat, colors)
        if key not in self._meshes:
            attrs, indices = self.geometry(node.geometry, colors)
            prim = {"attributes": dict(attrs), "mode": 4}
            if indices is not None:
                prim["indices"] = indices
            if mat is not None:
                prim["material"] = mat
            self._meshes[key] = self.b.add("meshes", {"primitives": [prim]})
        entry = {"name": name, "mesh": self._meshes[key]}
        for key_, value, default in (("translation", node.pos, (0, 0, 0)), ("rotation", node.rot, (0, 0, 0, 1)),
                                     ("scale", node.scale, (1, 1, 1))):
            value = [float(v) for v in value]
            if value != list(default):
                entry[key_] = value
        if node.extras is not None:
            entry["extras"] = node.extras
        self._roots.append(self.b.add("nodes", entry))

    def library(self, geometry):
        """A geometry as a mesh of its own, COLOR_0 included when it has colours; the
        material that draws it says whether to use them. Returns the mesh index."""
        attrs, indices = self.geometry(geometry, True)
        prim = {"attributes": dict(attrs), "mode": 4}
        if indices is not None:
            prim["indices"] = indices
        return self.b.add("meshes", {"primitives": [prim]})

    def geometry(self, geometry, colors):
        """Accessors for a geometry, written once however many materials use it."""
        key = (id(geometry), colors)
        if key not in self._geometry:
            geo = geometry.to_geometry() if hasattr(geometry, "to_geometry") else geometry
            self._geometry[key] = self.accessors(geo, colors)
        return self._geometry[key]

    def accessors(self, geo, colors):
        pos = np.asarray(geo["positions"] if "positions" in geo else geo["vertices"], np.float32)
        attrs = {"POSITION": self.b.accessor(pos, target=ARRAY_BUFFER, bounds=True)}
        if "normals" in geo:
            attrs["NORMAL"] = self.b.accessor(np.asarray(geo["normals"], np.float32), target=ARRAY_BUFFER)
        if "uvs" in geo:
            attrs["TEXCOORD_0"] = self.b.accessor(np.asarray(geo["uvs"], np.float32), target=ARRAY_BUFFER)
        if colors and "colors" in geo:
            attrs["COLOR_0"] = self.b.accessor(np.asarray(geo["colors"], np.float32), target=ARRAY_BUFFER)
        indices = None
        if "indices" in geo:
            # glTF forbids the component type's maximum value as an index.
            dtype = np.uint16 if len(pos) < 0xFFFF else np.uint32
            idx = np.asarray(geo["indices"]).reshape(-1).astype(dtype)
            indices = self.b.accessor(idx, target=ELEMENT_ARRAY_BUFFER)
        return attrs, indices

    def material(self, mat):
        if mat is None:
            return None, None
        if id(mat) not in self._materials:
            self._materials[id(mat)] = self._material(mat)
        return self._materials[id(mat)]

    def _material(self, mat):
        from manifoldx.resources import BasicMaterial, FlatMaterial, PhongMaterial, StandardMaterial

        kind = type(mat)  # exact types: viz materials subclass BasicMaterial
        name = kind.__name__
        note = None
        if kind is StandardMaterial:
            rgba = _rgba(mat.color)
            rgba[3] *= float(mat.opacity)  # the desktop blends opacity < 1 over what is behind
            pbr = {"baseColorFactor": rgba, "metallicFactor": float(mat.metallic),
                   "roughnessFactor": float(mat.roughness)}
            if mat.albedo_map is not None:
                tex = self.texture(mat.albedo_map)
                if tex is not None:  # the textured shader ignores color=, so glTF must not tint
                    pbr["baseColorTexture"] = {"index": tex}
                    pbr["baseColorFactor"] = [1.0, 1.0, 1.0, pbr["baseColorFactor"][3]]
            if mat.ao != 1.0:
                note = ("StandardMaterial.ao", "dropped", "glTF has ambient occlusion only as a texture")
            out = {"pbrMetallicRoughness": pbr}
        elif kind is FlatMaterial:
            out = {"pbrMetallicRoughness": {"baseColorFactor": _rgba(mat.color), "metallicFactor": 0.0,
                                            "roughnessFactor": 1.0},
                   "extensions": {"KHR_materials_unlit": {}}}
            self.b.use_extension("KHR_materials_unlit")
        elif kind in (BasicMaterial, PhongMaterial):
            out = {"pbrMetallicRoughness": {"baseColorFactor": _rgba(mat.color), "metallicFactor": 0.0,
                                            "roughnessFactor": 1.0}}
            note = (name, "approximated", "written as rough PBR, lit by the scene instead of its fixed head-on light")
        else:
            return None, (name, "dropped", "no glTF equivalent; its entities are not exported")
        out["name"] = name
        out["extras"] = {"mx": {"vertex_colors": _uses_colors(mat)}}
        if out["pbrMetallicRoughness"]["baseColorFactor"][3] < 1.0:
            out["alphaMode"] = "BLEND"
        return self.b.add("materials", out), note

    def texture(self, handle):
        if handle.id not in self._textures:
            source = getattr(handle, "source", None)
            if source is None:
                self.report.add("texture", f"texture {handle.id}", "dropped",
                                "made outside load_texture, so its pixels are unknown")
                self._textures[handle.id] = None
            else:
                data, mime = _image(source)
                if self._sampler is None:
                    self._sampler = self.b.add("samplers", {"magFilter": _LINEAR, "minFilter": _LINEAR_MIPMAP_LINEAR,
                                                            "wrapS": _REPEAT, "wrapT": _REPEAT})
                image = self.b.add("images", {"bufferView": self.b.view(data), "mimeType": mime})
                self._textures[handle.id] = self.b.add("textures", {"sampler": self._sampler, "source": image})
        return self._textures[handle.id]

    def camera(self, cam):
        forward = np.asarray(cam.target, float) - np.asarray(cam.position, float)
        if not np.all(np.isfinite(forward)) or not np.linalg.norm(forward) > 0:
            self.report.add("camera", "camera", "dropped", "its position equals its target, so it has no direction")
            return
        index = self.b.add("cameras", {"type": "perspective", "perspective": {
            "yfov": float(np.radians(cam.fov)), "znear": float(cam.near), "zfar": float(cam.far)}})
        pos = [float(c) for c in cam.position]
        rot = _look_rotation(np.asarray(cam.target, float) - pos, cam.up)
        self._roots.append(self.b.add("nodes", {"name": "camera", "camera": index, "translation": pos,
                                                "rotation": rot}))

    def light(self, light):
        from manifoldx.resources import DirectionalLight, PointLight, SpotLight

        entry = {"color": _light_rgb(light.color), "intensity": float(light.intensity),
                 "extras": {"manifoldx_intensity": float(light.intensity)}}
        node = {"name": type(light).__name__}
        if isinstance(light, DirectionalLight):
            entry["type"] = "directional"
            node["rotation"] = _look_rotation(light.direction)
        elif isinstance(light, SpotLight):
            entry["type"] = "spot"
            entry["spot"] = {"innerConeAngle": float(light.inner_angle), "outerConeAngle": float(light.outer_angle)}
            node["translation"] = [float(c) for c in light.position]
            node["rotation"] = _look_rotation(light.direction)
        elif isinstance(light, PointLight):
            entry["type"] = "point"
            node["translation"] = [float(c) for c in light.position]
        else:
            self.report.add("light", type(light).__name__, "dropped", "no glTF light type")
            return
        if getattr(light, "distance", 0):
            entry["range"] = float(light.distance)
        self.b.use_extension("KHR_lights_punctual")
        ext = self.b.doc.setdefault("extensions", {}).setdefault("KHR_lights_punctual", {"lights": []})
        ext["lights"].append(entry)
        node["extensions"] = {"KHR_lights_punctual": {"light": len(ext["lights"]) - 1}}
        self._roots.append(self.b.add("nodes", node))

    def environment(self, env):
        data = np.ascontiguousarray(env.data, dtype=np.float32)
        return {"bufferView": self.b.view(data.tobytes()), "width": int(data.shape[1]),
                "height": int(data.shape[0]), "intensity": float(env.intensity),
                "layout": "equirectangular, linear RGB float32, row 0 at the zenith"}

    def finish(self, extras):
        scene = {"nodes": self._roots} if self._roots else {}
        if extras:
            scene["extras"] = {"manifoldx": extras}
        self.b.doc["scenes"] = [scene]
        self.b.doc["scene"] = 0
        return self.b.to_bytes()
