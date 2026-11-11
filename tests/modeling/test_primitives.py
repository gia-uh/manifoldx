import numpy as np
from manifoldx.modeling import Mesh


def test_icosphere_face_count_grows_by_four():
    base = Mesh.icosphere(subdivisions=0)
    assert base.faces.shape[0] == 20            # icosahedron
    one = Mesh.icosphere(subdivisions=1)
    assert one.faces.shape[0] == 80             # x4 per subdivision


def test_icosphere_vertices_on_radius():
    m = Mesh.icosphere(subdivisions=2, radius=2.0)
    r = np.linalg.norm(m.positions, axis=1)
    assert np.allclose(r, 2.0, atol=1e-5)


def test_box_is_closed_triangle_soup():
    m = Mesh.box(2, 1, 1)
    assert m.faces.shape[1] == 3
    assert m.faces.shape[0] == 12           # 6 faces x 2 tris
    assert m.positions[:, 0].max() == 1.0   # width 2 -> [-1, 1]


def test_box_between_spans_two_opposite_corners():
    from manifoldx.modeling import primitives
    from manifoldx.modeling.lint import mesh_report

    m = primitives.box_between((1, 0, -2), (-1, 3, 0))
    p = np.asarray(m.positions)
    assert p[:, 0].min() == -1 and p[:, 0].max() == 1
    assert p[:, 1].min() == 0 and p[:, 1].max() == 3
    assert p[:, 2].min() == -2 and p[:, 2].max() == 0
    assert mesh_report(m)["open_edges"] == 0 and mesh_report(m)["volume"] > 0


def test_plane_segments_grid_counts():
    m = Mesh.plane(width=1.0, depth=1.0, segments=2)
    assert m.positions.shape[0] == 9        # (segments+1)^2
    assert m.faces.shape[0] == 8            # segments^2 * 2


def test_cylinder_and_torus_watertight_vertex_counts():
    cyl = Mesh.cylinder(radius=1.0, height=2.0, segments=8)
    assert cyl.positions.shape[0] > 0 and cyl.faces.shape[1] == 3
    tor = Mesh.torus(major=1.0, minor=0.3, major_segments=8, minor_segments=6)
    assert tor.faces.shape[0] == 8 * 6 * 2  # quad per (i,j) -> 2 tris
