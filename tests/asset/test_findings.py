"""Each lint finding raised by a crafted case, and silent on a clean one (modeling.lint)."""

import numpy as np
import pytest

from manifoldx.modeling.lint import (
    Finding,
    check_contact,
    check_meshes,
    check_seen,
    check_triangles,
    mesh_report,
)
from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.views import extrude, silhouette

SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def box(x=0.0, size=1.0):
    m = extrude(SQUARE * size / 2, size)
    return Mesh(positions=np.asarray(m.positions) + [x, 0, 0], faces=np.asarray(m.faces))


def codes(findings):
    return [f.code for f in findings]


def test_finding_is_frozen_and_defaults_its_optional_parts():
    f = Finding("A-OPEN", "error")
    assert f.part is None and f.view is None and f.value is None
    with pytest.raises(Exception):
        f.code = "A-FLIP"


def test_a_clean_box_raises_nothing():
    parts = {"box": box()}
    assert check_meshes(parts) == []
    assert check_contact(parts, tol=0.05) == []
    assert check_triangles(parts, max_tris=100) == []


def test_a_box_with_a_face_removed_is_open():
    m = box()
    holed = Mesh(positions=np.asarray(m.positions), faces=np.asarray(m.faces)[2:])
    assert mesh_report(holed)["open_edges"] > 0
    (finding,) = check_meshes({"body": holed})
    assert finding.code == "A-OPEN" and finding.level == "error" and finding.part == "body"
    assert finding.value == mesh_report(holed)["open_edges"]
    assert "open edges" in finding.message and finding.hint == "a builder bug: report it"


def test_reversed_faces_are_inside_out():
    m = box()
    flipped = Mesh(positions=np.asarray(m.positions), faces=np.asarray(m.faces)[:, ::-1].copy())
    (finding,) = check_meshes({"shell": flipped})
    assert finding.code == "A-FLIP" and finding.level == "error"
    assert finding.value < 0 and "inside out" in finding.message


def test_a_part_apart_floats_and_names_its_nearest_neighbour():
    parts = {"base": box(0.0), "cap": box(2.2)}
    (finding,) = check_contact(parts, tol=0.05)
    assert finding.code == "A-FLOAT" and finding.level == "error" and finding.part == "cap"
    assert finding.value == pytest.approx(1.2, abs=1e-5)
    assert "nearest part base" in finding.message
    assert "attach" in finding.hint


def test_touching_parts_do_not_float():
    assert check_contact({"base": box(0.0), "cap": box(0.9)}, tol=0.05) == []


def test_too_many_triangles_warn():
    (finding,) = check_triangles({"ball": box()}, max_tris=11)
    assert finding.code == "A-TRIS" and finding.level == "warn" and finding.value == 12
    assert "fewer rings" in finding.hint


def _seen_fixture(mask):
    mesh = box()
    frame = (-1.1, 1.1, -1.1, 1.1)
    return {"box": mesh}, "front", {"box": np.asarray(mask, bool)}, frame


def test_a_drawing_that_agrees_is_silent():
    frame = (-1.1, 1.1, -1.1, 1.1)
    drawn = silhouette(box(), "front", (64, 64), frame)
    assert check_seen(*_seen_fixture(drawn)) == []


def test_a_mask_that_disagrees_is_seen_the_worse_the_further_it_is():
    frame = (-1.1, 1.1, -1.1, 1.1)
    drawn = silhouette(box(), "front", (64, 64), frame)
    part = drawn.copy()
    part[:, :30] = False                             # the box's left 40% missing: above 0.5, below 0.75
    (warn,) = check_seen(*_seen_fixture(part))
    assert warn.code == "A-SEEN" and warn.level == "warn" and warn.view == "front"
    assert 0.5 <= warn.value < 0.75 and "px drawn but not built" in warn.message
    assert "rows" in warn.message
    corner = np.zeros_like(drawn)
    corner.fill(False)
    corner[0:8, 0:8] = True
    (error,) = check_seen(*_seen_fixture(corner))
    assert error.level == "error" and error.value < 0.5
    assert "px built but not drawn" in error.message


def test_a_part_with_no_or_empty_drawn_mask_is_left_alone():
    frame = (-1.1, 1.1, -1.1, 1.1)
    assert check_seen({"box": box()}, "front", {"box": np.zeros((32, 32), bool)}, frame) == []
    assert check_seen({"box": box()}, "front", {}, frame) == []
