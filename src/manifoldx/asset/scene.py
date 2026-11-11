"""A scene an asset's build stage returns: named meshes, placed, with the material each is made of."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manifoldx.modeling.mesh import Mesh


class Scene:
    """Ordered named meshes with their materials, ready to lint, paint or write to a .glb.

    Names are unique: a stage that adds one twice is a mistake, and it is raised where it happens. ``at``
    places a mesh by translating it, so a part may be built at the origin and moved into place.
    """

    def __init__(self) -> None:
        self.meshes: dict[str, Mesh] = {}
        self.materials: dict[str, object] = {}

    def add(self, name: str, mesh: Mesh, at=(0.0, 0.0, 0.0), material=None) -> Mesh:
        """Add a mesh under ``name``, translated by ``at``; returns the placed mesh."""
        if name in self.meshes:
            raise ValueError(f"the scene already has a part named {name!r}")
        placed = mesh.translate(np.asarray(at, float)) if np.any(np.asarray(at, float)) else mesh
        self.meshes[name] = placed
        self.materials[name] = material
        return placed

    def to_glb(self, path) -> object:
        """Write every mesh, with its material, to a .glb at ``path``; returns the export report."""
        from manifoldx.gltf import Node, export_gltf

        nodes = [Node(mesh, self.materials[name], name=name) for name, mesh in self.meshes.items()]
        return export_gltf(Path(path), nodes)

    def __len__(self) -> int:
        return len(self.meshes)

    def __iter__(self):
        return iter(self.meshes)

    def __contains__(self, name) -> bool:
        return name in self.meshes

    def __getitem__(self, name) -> Mesh:
        return self.meshes[name]

    def __repr__(self) -> str:
        return f"Scene({len(self.meshes)} parts: {', '.join(self.meshes)})"
