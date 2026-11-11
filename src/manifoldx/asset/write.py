"""Writing a run to disk: the model, its textures and the report.

A model with textures becomes one ``model.glb`` whose material carries the albedo as a texture; without
textures it is the same glTF with each part's own material. The textures are also written as PNG, because a
person and a paint program read them directly.
"""

from __future__ import annotations

from pathlib import Path

from manifoldx.asset.base import Result
from manifoldx.image import write_png


def save(result: Result, out_dir) -> list[Path]:
    """Write ``model.glb``, ``report.json`` and the textures as PNG into ``out_dir``; returns the paths.

    ``out_dir`` is made when it does not exist. With an ``albedo`` among the textures the model's parts share a
    white :class:`~manifoldx.resources.StandardMaterial` carrying that texture, and the meshes are the atlas's
    (a vertex split where it carries two UVs); otherwise the scene writes its own materials.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    textures = result.textures or {}
    written = []
    albedo = textures.get("albedo")
    normal = textures.get("normal")
    if albedo is not None:
        written.append(out / "albedo.png")
        write_png(written[-1], albedo)
    if normal is not None:
        written.append(out / "normal.png")
        write_png(written[-1], normal)

    from manifoldx.gltf import Node, export_gltf

    if result.scene is None:
        raise ValueError("there is no scene to save; run the build stage first")
    material = None
    meshes = textures.get("meshes") or result.scene.meshes
    if albedo is not None:
        from manifoldx.resources import StandardMaterial
        from manifoldx.textures import TextureHandle

        handle = TextureHandle(id=1, texture=None, view=None, sampler=None,
                               size=(int(albedo.shape[1]), int(albedo.shape[0])), source=out / "albedo.png")
        material = StandardMaterial(color=(255, 255, 255), albedo_map=handle)
    path = out / "model.glb"
    export_gltf(path, [Node(mesh, material, name=name) for name, mesh in meshes.items()])
    written.append(path)

    report = out / "report.json"
    report.write_text(result.report.to_json() + "\n", encoding="utf-8")
    written.append(report)
    return written
