"""Read a binary glTF (.glb) into a trace Scene: every mesh instance baked to world space."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np

from manifoldx.trace.scene import Camera, Light, Material, assemble

_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_NORM = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}
MAX_TEXTURE = 2048


def parse_glb(data: bytes):
    """(json document, binary chunk) of a .glb."""
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a binary glTF file")
    if version != 2:
        raise ValueError(f"glTF version {version}; only 2 is read")
    doc, binary, off = None, b"", 12
    while off < len(data):
        size, kind = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + size]
        if kind == 0x4E4F534A:
            doc = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        off += 8 + size
    if doc is None:
        raise ValueError("glb has no JSON chunk")
    return doc, binary


def read_glb(path, overrides=None):
    doc, binary = parse_glb(Path(path).read_bytes())
    r = _Reader(doc, binary)
    return r.scene(overrides)


class _Reader:
    def __init__(self, doc, binary):
        self.doc, self.bin = doc, binary
        self.textures, self._texture_of = [], {}
        self.materials = [self.material(m) for m in doc.get("materials", [])]

    def view(self, index):
        v = self.doc["bufferViews"][index]
        if v.get("buffer", 0) != 0:
            raise ValueError("only the glb's own binary buffer is read")
        start = v.get("byteOffset", 0)
        return self.bin[start: start + v["byteLength"]], v.get("byteStride")

    def accessor(self, index):
        a = self.doc["accessors"][index]
        dtype = _DTYPES[a["componentType"]]
        width = _WIDTH[a["type"]]
        count = a["count"]
        if "bufferView" not in a:
            out = np.zeros((count, width), dtype)
        else:
            raw, stride = self.view(a["bufferView"])
            item = np.dtype(dtype).itemsize * width
            stride = stride or item
            buf = np.frombuffer(raw, np.uint8, count=stride * (count - 1) + item, offset=a.get("byteOffset", 0))
            if stride == item:
                out = buf.view(dtype).reshape(count, width)
            else:
                rows = np.lib.stride_tricks.as_strided(buf, (count, item), (stride, 1))
                out = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
        if a.get("normalized") and dtype in _NORM:
            out = np.maximum(out.astype(np.float32) / _NORM[dtype], -1.0)
        return out[:, 0] if width == 1 else out

    def texture(self, index):
        if index in self._texture_of:
            return self._texture_of[index]
        from PIL import Image

        source = self.doc["textures"][index].get("source")
        if source is None:
            self._texture_of[index] = -1
            return -1
        image = self.doc["images"][source]
        raw, _ = self.view(image["bufferView"])
        img = Image.open(io.BytesIO(bytes(raw))).convert("RGBA")
        if max(img.size) > MAX_TEXTURE:
            scale = MAX_TEXTURE / max(img.size)
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
        self.textures.append(np.asarray(img, np.uint8))
        self._texture_of[index] = len(self.textures) - 1
        return self._texture_of[index]

    def material(self, m):
        pbr = m.get("pbrMetallicRoughness", {})
        ext = m.get("extensions", {})
        base = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        strength = ext.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)
        emission = tuple(float(c) * strength for c in m.get("emissiveFactor", [0.0, 0.0, 0.0]))
        transmission = ext.get("KHR_materials_transmission", {}).get("transmissionFactor", 0.0)
        alpha = float(base[3])
        if m.get("alphaMode") == "BLEND" and alpha < 1.0:  # see-through drawn as thin glass
            transmission = max(transmission, 1.0 - alpha)
        volume = ext.get("KHR_materials_volume", {})
        tex = pbr.get("baseColorTexture")
        mx = m.get("extras", {}).get("mx", {})
        return Material(
            name=m.get("name", ""), base_color=tuple(float(c) for c in base[:3]), alpha=alpha,
            metallic=float(pbr.get("metallicFactor", 1.0)), roughness=float(pbr.get("roughnessFactor", 1.0)),
            emission=emission, transmission=float(transmission),
            ior=float(ext.get("KHR_materials_ior", {}).get("ior", 1.5)),
            thin=float(volume.get("thicknessFactor", 0.0)) <= 0.0,
            texture=self.texture(tex["index"]) if tex is not None else -1,
            vertex_colors=bool(mx.get("vertex_colors", True)),
        )

    def scene(self, overrides):
        doc = self.doc
        nodes = doc.get("nodes", [])
        roots = doc["scenes"][doc.get("scene", 0)]["nodes"] if doc.get("scenes") else \
            sorted(set(range(len(nodes))) - {c for n in nodes for c in n.get("children", [])})
        parts, camera, lights = [], None, []
        light_defs = doc.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
        stack = [(i, np.eye(4)) for i in roots]
        while stack:
            i, parent = stack.pop()
            node = nodes[i]
            world = parent @ _local(node)
            stack.extend((c, world) for c in node.get("children", []))
            if "mesh" in node and "skin" not in node:
                parts.extend(self.mesh(node, world))
            if "camera" in node and camera is None:
                camera = _camera(doc["cameras"][node["camera"]], world)
            light = node.get("extensions", {}).get("KHR_lights_punctual")
            if light is not None:
                lights.append(_light(light_defs[light["light"]], world))
        return assemble(parts, self.materials, textures=self.textures, camera=camera, lights=lights,
                        environment=self.environment(), overrides=overrides)

    def environment(self):
        """The equirect map manifoldx's writer stores in the scene's extras, if any."""
        from manifoldx.trace.environment import Environment

        scenes = self.doc.get("scenes") or [{}]
        env = scenes[self.doc.get("scene", 0)].get("extras", {}).get("manifoldx", {}).get("environment")
        if not env or "bufferView" not in env:
            return None
        raw, _ = self.view(env["bufferView"])
        data = np.frombuffer(raw, np.float32).reshape(env["height"], env["width"], 3)
        return Environment(data.copy(), float(env.get("intensity", 1.0)))

    def mesh(self, node, world):
        name = node.get("name", "")
        normal_m = np.linalg.inv(world[:3, :3]).T
        flip = np.linalg.det(world[:3, :3]) < 0
        for prim in self.doc["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            pos = self.accessor(attrs["POSITION"]).astype(np.float64)
            pos = pos @ world[:3, :3].T + world[:3, 3]
            tri = (self.accessor(prim["indices"]).astype(np.int64) if "indices" in prim
                   else np.arange(len(pos))).reshape(-1, 3)
            if flip:
                tri = tri[:, ::-1]
            out = {}
            if "NORMAL" in attrs:
                out["normals"] = self.accessor(attrs["NORMAL"]).astype(np.float64) @ normal_m.T
            if "TEXCOORD_0" in attrs:
                out["uvs"] = self.accessor(attrs["TEXCOORD_0"])
            mat = prim.get("material", -1)
            if "COLOR_0" in attrs and (mat < 0 or self.materials[mat].vertex_colors):
                out["colors"] = self.accessor(attrs["COLOR_0"])[:, :3]
            yield name, pos, tri, out, mat


def _local(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T
    m = np.eye(4)
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, :3] *= np.asarray(node.get("scale", [1.0, 1.0, 1.0]))
    m[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return m


def _camera(cam, world):
    pos = world[:3, 3]
    forward = -world[:3, 2] / np.linalg.norm(world[:3, 2])
    up = world[:3, 1] / np.linalg.norm(world[:3, 1])
    fov = np.degrees(cam.get("perspective", {}).get("yfov", np.radians(60.0)))
    return Camera(tuple(pos), tuple(pos + forward), tuple(up), float(fov))


def _light(d, world):
    forward = -world[:3, 2] / np.linalg.norm(world[:3, 2])
    spot = d.get("spot", {})
    return Light(kind=d["type"], color=tuple(d.get("color", [1.0, 1.0, 1.0])),
                 intensity=float(d.get("intensity", 1.0)), position=tuple(world[:3, 3]),
                 direction=tuple(forward), inner_cone=float(spot.get("innerConeAngle", 0.0)),
                 outer_cone=float(spot.get("outerConeAngle", np.pi / 4)))
