"""Clothes, hair and face on a figure: garments over the body's rings, painted at a distance."""

import numpy as np
import pytest

from manifoldx.modeling.assets.human import HumanParams, build_human
from manifoldx.modeling.assets.human_dress import Outfit


_BUILT = {}


def _build(params=HumanParams(), outfit=None, level=0):
    key = repr((params, outfit, level))
    if key not in _BUILT:
        _BUILT[key] = build_human(params, outfit, level)
    return _BUILT[key]


def _closed(mesh):
    edges = np.sort(mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def _components(mesh, region, f=None):
    """Connected pieces of the faces whose vertices all belong to `region`."""
    part = set(f.parts[region].tolist())
    faces = [tuple(t) for t in mesh.faces.tolist() if all(v in part for v in t)]
    parent = {v: v for t in faces for v in t}

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b, c in faces:
        parent[root(b)] = root(a)
        parent[root(c)] = root(a)
    return len({root(v) for v in parent})


def _region_count(f, region):
    return len(f.parts.get(region, ()))


def _colour_share(f, hex_colour):
    rgb = np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255
    return float((np.abs(f.mesh.colors - rgb).max(1) < 0.01).mean())


def _min_signed_distance(f, region, to="body"):
    """Smallest distance of a garment vertex outside the undressed body, along the nearest body vertex's normal."""
    assert to == "body"
    body = build_human(f.params, None, f.level).mesh.recompute_normals()
    bp, bn = body.positions.astype(float), body.normals.astype(float)
    worst = np.inf
    for p in f.mesh.positions[f.parts[region]].astype(float):
        i = int(np.argmin(((bp - p) ** 2).sum(1)))
        worst = min(worst, float((p - bp[i]) @ bn[i]))
    return worst


@pytest.mark.parametrize("level", [0, 1, 2])
def test_dressed_is_closed_with_fixed_count(level):
    counts = {len(build_human(HumanParams.sample(s), Outfit(), level).mesh.faces) for s in range(5)}
    assert len(counts) == 1 and _closed(_build(HumanParams(), Outfit(), level).mesh)


@pytest.mark.parametrize(
    "outfit",
    [
        Outfit(top="shirt", sleeves=1.0, bottom="skirt", bottom_length=0.6, hair="bun", hat="brim"),
        Outfit(top="tank", bottom="shorts", shoes="boots", hair="long", hat="cap"),
        Outfit(top="blouse", bottom="trousers", shoes="sandals", hair="curly", stripe=(0.4, 0.5, "#aa2222")),
        Outfit(top=None, bottom="skirt", bottom_length=0.5, shoes="boots", hair="bun"),
    ],
)
def test_every_outfit_is_closed_and_weighted(outfit):
    f = _build(HumanParams(), outfit)
    m = f.mesh
    assert _closed(m) and m.joints.shape == (len(m.positions), 2) and len(m.weights) == len(m.positions)
    assert ((m.weights >= 0) & (m.weights <= 1)).all()


def test_covered_skin_is_dropped():
    bare = _build(HumanParams(), Outfit(top=None, bottom="shorts", shoes=None, hair=None))
    dressed = _build(HumanParams(), Outfit(top="shirt", sleeves=1.0, bottom="trousers"))
    assert len(dressed.mesh.faces) < len(bare.mesh.faces) * 1.3


def test_garment_outside_body():
    f = _build(HumanParams(weight=1.5), Outfit(top="tshirt", looseness=0.004))
    assert _min_signed_distance(f, region="top", to="body") > 0.002


def test_skirt_is_one_tube():
    f = _build(HumanParams(), Outfit(bottom="skirt", bottom_length=0.6))
    assert _components(f.mesh, "bottom", f) == 1


def test_level_one_paints_instead_of_offsets():
    a = _build(HumanParams(), Outfit(top="shirt"), level=1)
    assert _region_count(a, "top") == 0 and _colour_share(a, "#e8e2d0") > 0.2


def test_sleeves_reach_their_fraction():
    short, long = _build(HumanParams(), Outfit(sleeves=0.25)), _build(HumanParams(), Outfit(sleeves=1.0))
    assert _region_count(long, "top") > _region_count(short, "top")


def test_face_is_painted_at_level_zero():
    f = _build(HumanParams(), Outfit(hair=None, hat=None))
    head = f.joints["head"]
    front = (f.mesh.positions[:, 2] > head[2]) & (np.abs(f.mesh.positions[:, 1] - head[1]) < 0.06)
    assert len(np.unique(np.round(f.mesh.colors[front], 3), axis=0)) >= 2  # skin and the darker features


def test_no_outfit_is_the_bare_figure():
    assert np.array_equal(_build(HumanParams(), None).mesh.faces, build_human(HumanParams()).mesh.faces)
