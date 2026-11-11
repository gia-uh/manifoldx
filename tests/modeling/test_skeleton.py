import numpy as np

from manifoldx.modeling.lint import mesh_report, part_groups
from manifoldx.modeling.skeleton import fit_radii, signed_distance, skin, surface_nets, taubin


def test_surface_nets_of_a_sphere_is_closed_and_has_its_volume():
    step = 0.05
    axis = np.arange(-1.0, 1.0001, step)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    mesh = surface_nets(np.sqrt(x * x + y * y + z * z) - 0.7, (-1.0, -1.0, -1.0), step)
    report = mesh_report(mesh)
    faces = np.asarray(mesh.faces, np.int64)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all()                      # every edge between two faces (before any welding)
    assert len(np.unique(directed, axis=0)) == len(directed)  # and wound the same way on both
    assert report["open_edges"] == 0
    assert abs(report["volume"] - 4 / 3 * np.pi * 0.7 ** 3) < 0.05 * 4 / 3 * np.pi * 0.7 ** 3


def test_a_branching_skeleton_skins_into_one_closed_piece():
    joints = [(0, 0, 0), (0, 1, 0), (0.5, 1.2, 0), (-0.5, 1.2, 0), (0, 1.45, 0)]
    mesh = skin(joints, [0.15, 0.12, 0.05, 0.05, 0.1], [(0, 1), (1, 2), (1, 3), (1, 4)], step=0.03)
    report = mesh_report(mesh)
    assert report["open_edges"] == 0 and report["volume"] > 0
    p = np.asarray(mesh.positions)
    assert p[:, 0].max() > 0.5 and p[:, 0].min() < -0.5 and p[:, 1].max() > 1.5
    faces = np.asarray(mesh.faces)
    pieces = {}                                     # the mesh's pieces, as sets of faces sharing a vertex
    parent = list(range(len(p)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in faces:
        for u, v in ((a, b), (b, c)):
            parent[root(u)] = root(v)
    for a, _, _ in faces:
        pieces.setdefault(root(a), 0)
    assert len(pieces) == 1


def test_signed_distance_is_negative_inside_and_grows_outward():
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    d = signed_distance(mask)
    assert d[20, 20] < -5 and d[20, 5] > 3 and abs(d[20, 10]) <= 1.5


def test_fit_radii_reads_a_bars_half_width_across_its_bone():
    mask = np.zeros((100, 100), bool)
    mask[:, 40:60] = True                           # a vertical bar twenty pixels wide, 0.01 m a pixel
    view = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), mask, (-0.5, 0.5, 0.0, 1.0))
    radii = fit_radii([(0.0, 0.2, 0.0), (0.0, 0.8, 0.0)], [(0, 1)], [view])
    assert np.allclose(radii, 0.1, atol=0.015)


def test_the_hull_cuts_the_skin_to_the_drawings():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 30:70] = True                       # a box 0.4 m wide and 0.6 m tall in both views
    front = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), mask, (-0.5, 0.5, 0.0, 1.0))
    side = ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), mask, (-0.5, 0.5, 0.0, 1.0))
    mesh = skin([(0, 0.2, 0), (0, 0.8, 0)], [0.4, 0.4], [(0, 1)], step=0.02, hull=[front, side])
    p = np.asarray(mesh.positions)
    assert p[:, 0].max() < 0.23 and p[:, 2].max() < 0.23 and p[:, 1].max() < 0.83 and p[:, 1].min() > 0.17


def test_taubin_smooths_without_shrinking_much():
    joints = [(0, 0, 0), (0, 1, 0)]
    mesh = skin(joints, [0.2, 0.2], [(0, 1)], step=0.05)
    smooth = taubin(mesh, 10)
    assert abs(mesh_report(smooth)["volume"] / mesh_report(mesh)["volume"] - 1) < 0.05
