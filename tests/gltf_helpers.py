"""Read back .glb files written by manifoldx.gltf, through pygltflib."""
import importlib.util
import math
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

_DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_SKINNING = Path(__file__).parents[1] / "examples" / "skinning.py"


def load(path):
    return GLTF2.load_binary(str(path))


def accessor(g, index):
    acc = g.accessors[index]
    view = g.bufferViews[acc.bufferView]
    blob = g.binary_blob()
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    width = _WIDTH[acc.type]
    a = np.frombuffer(blob, dtype=_DTYPES[acc.componentType], count=acc.count * width, offset=start)
    return a if width == 1 else a.reshape(acc.count, width)


def view_bytes(g, index):
    view = g.bufferViews[index]
    start = view.byteOffset or 0
    return g.binary_blob()[start:start + view.byteLength]


def _skinning_example():
    spec = importlib.util.spec_from_file_location("mx_skinning_example", _SKINNING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def skinned_figure(posed=True):
    """examples/skinning.py's figure (four parts, joints 0 to 3) on a fresh engine, at
    pos=(1, 2, 3), turned 0.5 rad about y, scale 2, posed with P when `posed`: identity, a
    lift by 0.1, and the arms swung by 0.3 and -0.3 rad about x through the shoulders.
    Returns (engine, handle, geometry, P)."""
    import manifoldx as mx
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial

    ex = _skinning_example()
    geo = ex.figure().to_geometry()
    lift = np.eye(4)
    lift[1, 3] = 0.1
    P = np.array([np.eye(4), lift, ex.about(ex.PARTS[2][2], "x", 0.3), ex.about(ex.PARTS[3][2], "x", -0.3)],
                 dtype=np.float32)
    e = mx.Engine("skinned", width=64, height=48)
    h = e.spawn(Mesh(geo), Material(StandardMaterial("#e07a3a")),
                Transform(pos=(1, 2, 3), rot=(0.0, math.sin(0.25), 0.0, math.cos(0.25)), scale=(2, 2, 2)))
    if posed:
        h.set_joints(P)
    return e, h, geo, P


def posed_positions(engine, handle, geo, P):
    """Each vertex where the desktop draws it, TRS . P[joint] . (x, y, z, 1), in float64."""
    t = engine.store._components["Transform"][handle.index].astype(np.float64)
    x, y, z, w = t[3:7]
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    trs = np.eye(4)
    trs[:3, :3] = r * t[7:10]
    trs[:3, 3] = t[0:3]
    v = np.c_[np.asarray(geo["positions"], np.float32).astype(np.float64), np.ones(len(geo["positions"]))]
    m = trs @ P.astype(np.float64)[np.asarray(geo["joints"])]  # (n, 4, 4)
    return np.einsum("nij,nj->ni", m, v)[:, :3]
