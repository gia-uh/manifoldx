"""Read a base body GLB into a humanoid section table (see src/manifoldx/modeling/assets/data/HUMAN.md).

Usage: uv run python scripts/human_sections.py BODY.glb OUT.npz --stature 1.63 --body female
       [--landmarks-from src/manifoldx/modeling/assets/data/human_female.npz]   (for a variant)
"""

import argparse

import numpy as np

from manifoldx.modeling.assets.human_sections import load_table, read_sections, save_table
from manifoldx.trace.gltf_read import parse_glb

_TYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(doc, binary, index):
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    dtype, width = np.dtype(_TYPES[a["componentType"]]), _WIDTH[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    raw = np.frombuffer(
        binary, np.uint8, count=stride * (a["count"] - 1) + dtype.itemsize * width, offset=start
    )
    rows = np.lib.stride_tricks.as_strided(raw, (a["count"], dtype.itemsize * width), (stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(a["count"], width)


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = np.diag([*node["scale"], 1.0]) @ m
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        rm = np.eye(4)
        rm[:3, :3] = r
        m = rm @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


def read_glb_mesh(path):
    """Every triangle primitive of a GLB, world-transformed, as one (positions, faces)."""
    doc, binary = parse_glb(open(path, "rb").read())
    pos, faces, n = [], [], 0

    def visit(i, parent):
        nonlocal n
        node = doc["nodes"][i]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                p = _accessor(doc, binary, prim["attributes"]["POSITION"]).astype(float)
                p = p @ m[:3, :3].T + m[:3, 3]
                f = (
                    _accessor(doc, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
                    if "indices" in prim
                    else np.arange(len(p)).reshape(-1, 3)
                )
                pos.append(p)
                faces.append(f + n)
                n += len(p)
        for c in node.get("children", []):
            visit(c, m)

    for i in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(i, np.eye(4))
    return np.concatenate(pos), np.concatenate(faces)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--stature", type=float, required=True)
    ap.add_argument("--body", choices=("female", "male"), required=True)
    ap.add_argument(
        "--landmarks-from", help="a variant's base table: its crotch and axilla heights are kept"
    )
    a = ap.parse_args()
    positions, faces = read_glb_mesh(a.glb)
    landmarks = None
    if a.landmarks_from:
        base = load_table(a.landmarks_from)
        landmarks = (base.crotch_y, base.axilla_y)
    table = read_sections(positions, faces, a.stature, a.body, landmarks)
    save_table(table, a.out)
    print(
        f"{a.out}: crotch {table.crotch_y:.3f} m, axilla {table.axilla_y:.3f} m, acromion {table.acromion_y:.3f} m"
    )


if __name__ == "__main__":
    main()
