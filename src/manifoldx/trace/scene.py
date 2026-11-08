"""What the tracer draws: world-space triangles, materials, textures, a camera, lights and an environment."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, fields, replace

import numpy as np


@dataclass
class Material:
    """glTF metallic-roughness, colours linear. emission is radiance, strength already applied.

    A transmissive material is thin-walled (light passes straight through) unless thin=False,
    which refracts by ior, as glTF's KHR_materials_volume says."""

    name: str = ""
    base_color: tuple = (0.8, 0.8, 0.8)
    alpha: float = 1.0
    metallic: float = 0.0
    roughness: float = 1.0
    emission: tuple = (0.0, 0.0, 0.0)
    transmission: float = 0.0
    ior: float = 1.5
    thin: bool = True
    texture: int = -1  # index into Scene.textures, multiplies base_color
    vertex_colors: bool = False


@dataclass
class Camera:
    position: tuple = (0.0, 1.0, 3.0)
    target: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    fov: float = 60.0  # vertical, degrees

    def basis(self):
        """(forward, right, up) unit vectors."""
        f = np.asarray(self.target, np.float64) - np.asarray(self.position, np.float64)
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(self.up, np.float64))
        if np.linalg.norm(r) < 1e-9:  # looking straight along up
            r = np.cross(f, (0.0, 0.0, 1.0))
        r /= np.linalg.norm(r)
        return f, r, np.cross(r, f)


@dataclass
class Light:
    """A punctual light. intensity in W/sr (point, spot) or W/m² (directional), as glTF."""

    kind: str  # "directional", "point" or "spot"
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    position: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, -1.0, 0.0)  # where the light travels
    inner_cone: float = 0.0
    outer_cone: float = np.pi / 4


@dataclass
class Scene:
    """Triangles in world space, each vertex attribute shaped (T, 3, k)."""

    positions: np.ndarray  # (T, 3, 3) float32
    normals: np.ndarray  # (T, 3, 3) float32, unit
    uvs: np.ndarray  # (T, 3, 2) float32
    colors: np.ndarray  # (T, 3, 3) float32 linear, ones where a mesh has none
    material_ids: np.ndarray  # (T,) int32
    materials: list = field(default_factory=list)
    textures: list = field(default_factory=list)  # (H, W, 4) uint8, sRGB colour
    camera: Camera | None = None
    lights: list = field(default_factory=list)
    environment: object = None  # trace.Environment
    node_names: list = field(default_factory=list)  # (name, first triangle, count)

    @classmethod
    def from_glb(cls, path, overrides=None) -> "Scene":
        """Read a .glb. overrides maps a node or material name (fnmatch pattern) to Material
        fields, e.g. {"window-*": {"transmission": 1.0, "roughness": 0.0}}; node names win."""
        from manifoldx.trace.gltf_read import read_glb

        return read_glb(path, overrides)

    @classmethod
    def from_meshes(cls, meshes, *, camera=None, lights=(), environment=None) -> "Scene":
        """meshes: (geometry dict or modeling.Mesh, Material[, name]) tuples, already in world space."""
        parts, mats = [], []
        for item in meshes:
            geo, mat = item[0], item[1]
            name = item[2] if len(item) > 2 else mat.name
            geo = geo.to_geometry() if hasattr(geo, "to_geometry") else geo
            pos = np.asarray(geo["positions"] if "positions" in geo else geo["vertices"], np.float32)
            idx = np.asarray(geo["indices"], np.int64).reshape(-1, 3) if "indices" in geo else \
                np.arange(len(pos)).reshape(-1, 3)
            attrs = {"normals": geo.get("normals"), "uvs": geo.get("uvs"), "colors": geo.get("colors")}
            mats.append(mat)
            parts.append((name, pos, idx, attrs, len(mats) - 1))
        return assemble(parts, mats, camera=camera, lights=list(lights), environment=environment)

    @property
    def triangle_count(self) -> int:
        return len(self.material_ids)

    def bounds(self):
        p = self.positions.reshape(-1, 3)
        return p.min(axis=0), p.max(axis=0)


def assemble(parts, materials, *, textures=(), camera=None, lights=(), environment=None, overrides=None):
    """parts: (name, positions (V,3), triangles (T,3) int, {normals, uvs, colors}, material index)."""
    materials = list(materials)
    derived = {}
    P, N, UV, C, M, names = [], [], [], [], [], []
    first = 0
    for name, pos, tri, attrs, mat in parts:
        pos = np.asarray(pos, np.float32)
        tri = np.asarray(tri, np.int64)
        p = pos[tri]
        face_n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-20)
        if attrs.get("normals") is not None:
            n = np.asarray(attrs["normals"], np.float32)[tri]
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-20)
            bad = ~np.isfinite(n).all(axis=2) | (np.linalg.norm(n, axis=2) < 0.5)
            n[bad] = np.repeat(face_n[:, None], 3, axis=1)[bad]
        else:
            n = np.repeat(face_n[:, None], 3, axis=1)
        uv = np.asarray(attrs["uvs"], np.float32)[tri] if attrs.get("uvs") is not None else \
            np.zeros((len(tri), 3, 2), np.float32)
        col = np.asarray(attrs["colors"], np.float32)[tri][..., :3] if attrs.get("colors") is not None else \
            np.ones((len(tri), 3, 3), np.float32)
        if mat is None or mat < 0:
            materials.append(Material(name="default"))
            mat = len(materials) - 1
        mat = _override(materials, derived, mat, name, overrides)
        P.append(p), N.append(n), UV.append(uv), C.append(col)
        M.append(np.full(len(tri), mat, np.int32))
        names.append((name, first, len(tri)))
        first += len(tri)
    if not P:
        raise ValueError("the scene has no triangles")
    return Scene(np.concatenate(P), np.concatenate(N).astype(np.float32), np.concatenate(UV),
                 np.concatenate(C), np.concatenate(M), materials, list(textures), camera, list(lights),
                 environment, names)


_FIELDS = {f.name for f in fields(Material)}


def _override(materials, derived, mat, node_name, overrides):
    if not overrides:
        return mat
    changes = {}
    for pattern, values in overrides.items():  # material names first, so node names win
        if fnmatch.fnmatchcase(materials[mat].name or "", pattern):
            changes.update(values)
    for pattern, values in overrides.items():
        if node_name and fnmatch.fnmatchcase(node_name, pattern):
            changes.update(values)
    if not changes:
        return mat
    unknown = set(changes) - _FIELDS
    if unknown:
        raise ValueError(f"unknown material fields in overrides: {sorted(unknown)}")
    if "texture" in changes and changes["texture"] is None:  # {"texture": None} drops the texture
        changes["texture"] = -1
    key = (mat, tuple(sorted((k, repr(v)) for k, v in changes.items())))
    if key not in derived:
        materials.append(replace(materials[mat], **changes))
        derived[key] = len(materials) - 1
    return derived[key]
