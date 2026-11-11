import numpy as np
import pytest

from manifoldx.modeling.lint import (
    check_seen,
    closest_offset,
    gap,
    mesh_report,
    part_groups,
    winding,
)
from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.views import extrude, silhouette

SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def box(x=0.0, size=1.0):
    m = extrude(SQUARE * size / 2, size)
    return Mesh(positions=np.asarray(m.positions) + [x, 0, 0], faces=np.asarray(m.faces))


def test_a_closed_box_has_no_open_edges_and_a_positive_volume():
    r = mesh_report(box())
    assert r["open_edges"] == 0 and r["nonmanifold_edges"] == 0 and abs(r["volume"] - 1.0) < 1e-6


def test_a_box_without_a_face_has_open_edges():
    m = box()
    r = mesh_report(Mesh(positions=np.asarray(m.positions), faces=np.asarray(m.faces)[2:]))
    assert r["open_edges"] > 0


def test_winding_is_one_inside_and_zero_outside():
    w = winding([[0, 0, 0], [3, 0, 0]], box())
    assert abs(w[0] - 1) < 1e-6 and abs(w[1]) < 1e-6


def test_overlapping_parts_have_no_gap_and_a_floating_part_is_its_own_group():
    parts = {"a": box(0.0), "b": box(0.8), "c": box(3.0)}
    assert gap(parts["a"], parts["b"]) == 0.0
    r = part_groups(parts, tol=0.05)
    assert sorted(r["groups"][0]) == ["a", "b"] and r["groups"][1] == ["c"]
    assert r["nearest"]["c"][0] == "b" and abs(r["nearest"]["c"][1] - 1.2) < 1e-6


def test_vertices_split_at_a_seam_are_not_holes():
    m = box()
    p, f = np.asarray(m.positions), np.asarray(m.faces)
    split = np.vstack([p, p])                        # the second half of the faces uses copies of the vertices
    faces = f.copy()
    faces[len(f) // 2:] += len(p)
    assert mesh_report(Mesh(positions=split, faces=faces))["open_edges"] == 0


def test_closest_offset_moves_a_floating_part_onto_its_neighbour():
    a, b = box(0.0), box(2.2)
    offset = closest_offset(b, a)
    assert offset[0] == pytest.approx(-1.2) and abs(offset[1]) < 1e-9 and abs(offset[2]) < 1e-9
    assert gap(b.translate(offset), a) < 1e-6
    assert np.allclose(closest_offset(a, b), 0.0) or closest_offset(a, b)[0] > 0.0
    assert np.allclose(closest_offset(box(0.9), box(0.0)), 0.0)  # already touching: no move


def test_seen_counts_the_instances_of_one_part_together():
    frame = (-2.2, 2.2, -1.1, 1.1)
    parts = {"leg": [box(-0.6, 0.3), box(0.6, 0.3)]}
    drawn = np.zeros((64, 64), bool)
    for x in (-0.6, 0.6):
        drawn |= silhouette(box(x, 0.3), "front", (64, 64), frame)
    assert check_seen(parts, "front", {"leg": drawn}, frame) == []
    half = drawn.copy()
    half[:, 32:] = False
    (finding,) = check_seen(parts, "front", {"leg": half}, frame)
    assert finding.code == "A-SEEN" and finding.part == "leg"
