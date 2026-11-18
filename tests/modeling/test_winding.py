"""Every vehicle piece is wound so the renderer, which culls back faces, draws its outside.

A closed piece seen from outside must show its outer faces: within each closed connected component every shared edge
is crossed once in each direction (no face flipped against its neighbours) and the signed volume is positive (the
faces point out, not in). Open pieces, such as glass drawn from both sides, are not checked.
"""

from collections import Counter, defaultdict

import numpy as np
import pytest

from manifoldx.modeling.assets import bus, car, cycle, truck
from manifoldx.modeling.assets.vehicle import orient_outward

VEHICLES = {
    "car": lambda: car.build_car(car.CarParams()),
    "car_hubcap": lambda: car.build_car(car.CarParams(wheel="hubcap", whitewall=0.04)),
    "bus": lambda: bus.build_bus(bus.BusParams()),
    **{n: (lambda n=n: truck.build_truck(getattr(truck, n)()))
       for n in ("cargo_truck_1980s", "box_truck", "tanker", "dump_truck", "fire_engine")},
    **{n: (lambda n=n: cycle.build_cycle(getattr(cycle, n)()))
       for n in ("bicycle", "cycle_rickshaw", "cargo_tricycle", "motorcycle", "electric_moto")},
}


def _closed_components(faces):
    """Lists of face indices, one per connected component whose every edge is shared by exactly two faces."""
    by_edge = defaultdict(list)
    for i, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            by_edge[(min(a, b), max(a, b))].append(i)
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fs in by_edge.values():
        for j in fs[1:]:
            parent[find(j)] = find(fs[0])
    groups = defaultdict(list)
    for i in range(len(faces)):
        groups[find(i)].append(i)
    open_faces = {i for fs in by_edge.values() if len(fs) != 2 for i in fs}
    return [g for g in groups.values() if not open_faces.intersection(g)]


def _problems(mesh):
    P = np.asarray(mesh.positions, float)
    F = np.asarray(mesh.faces, np.int64)
    out = []
    for g in _closed_components(F):
        directed = Counter()
        for a, b, c in F[g]:
            directed.update([(a, b), (b, c), (c, a)])
        if any(n > 1 for n in directed.values()):
            out.append("flipped neighbours")
            continue
        a, b, c = P[F[g, 0]], P[F[g, 1]], P[F[g, 2]]
        if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() <= 0:
            out.append("inside out")
    return out


@pytest.mark.parametrize("name", list(VEHICLES))
def test_every_closed_piece_faces_out(name):
    v = VEHICLES[name]()
    bad = [f"{part.name}/{mat}: {p}" for part in v.parts for mat, m in part.pieces.items() for p in _problems(m)]
    assert not bad, bad


def test_orient_outward_fixes_a_box_with_one_flipped_face_and_one_inside_out():
    v = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = np.array([t for a, b, c, d in q for t in ((a, b, c), (a, c, d))])
    from manifoldx.modeling.mesh import Mesh

    flipped = f.copy()
    flipped[:2] = flipped[:2, ::-1]  # one side turned against the rest
    for faces in (flipped, f[:, ::-1]):  # and the whole box inside out
        assert _problems(Mesh(v, faces))
        assert not _problems(Mesh(v, orient_outward(v, faces)))


def test_orient_outward_leaves_an_open_sheet_alone():
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    assert np.array_equal(orient_outward(v, f[:, ::-1]), f[:, ::-1])
