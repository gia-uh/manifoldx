"""The shared vehicle shape, its merged instancing mesh, liveries and a lightbar."""

import numpy as np

from manifoldx.modeling.assets.car import PALETTE, CarParams, build_car
from manifoldx.modeling.assets.vehicle import Vehicle
from manifoldx.modeling.mesh import Mesh

INSIDE = {"interior", "carpet", "dash", "seat", "liner"}


def _rgb(colour):
    """A palette entry's sRGB hex as floats in 0..1."""
    s = colour.lstrip("#")
    return np.array([int(s[k:k + 2], 16) for k in (0, 2, 4)], float) / 255.0


def _pieces(car, material):
    return [mesh for part in car.parts for mat, mesh in part.pieces.items() if mat == material]


def test_a_car_is_a_vehicle():
    car = build_car()
    assert isinstance(car, Vehicle)
    assert car.part("body").kind == "body"
    assert set(car.pose()) == {part.name for part in car.parts}
    assert {"nose", "tail", "roof"} <= set(car.landmarks)


def test_merged_keeps_the_default_car_on_the_ground_and_centred():
    car = build_car()
    mesh = car.merged()
    assert isinstance(mesh, Mesh)
    assert mesh.colors is not None and mesh.colors.shape == (len(mesh.positions), 3)
    assert mesh.colors.min() >= 0.0 and mesh.colors.max() <= 1.0
    assert mesh.normals is not None and np.isfinite(mesh.normals).all()
    p = np.asarray(mesh.positions, float)
    assert abs(p[:, 1].min()) < 0.01  # wheels on the ground
    assert abs(p[:, 0].max() + p[:, 0].min()) < 0.01  # centred along the car
    assert abs(p[:, 2].max() + p[:, 2].min()) < 0.01  # centred across it


def test_merged_leaves_the_interior_out_unless_asked():
    car = build_car()
    outside = car.merged()
    with_inside = car.merged(inside=True)
    expected = sum(len(mesh.positions) for part in car.parts for mat, mesh in part.pieces.items() if mat not in INSIDE)
    assert len(outside.positions) == expected
    assert len(outside.positions) < len(with_inside.positions)
    assert len(outside.faces) < len(with_inside.faces)


def test_merged_poses_every_part_so_a_vehicle_rolled_off_its_rest_pose_still_merges():
    car = build_car()
    rest, rolled = car.merged(), car.merged(pose=car.pose(wheel_turn=1.1, steer=0.3))
    assert len(rest.positions) == len(rolled.positions)
    assert not np.allclose(np.asarray(rest.positions), np.asarray(rolled.positions), atol=1e-3)


def test_merged_on_a_grid_keeps_the_palette_colours():
    car = build_car()
    whole, small = car.merged(), car.merged(grid=28)
    assert len(small.positions) < len(whole.positions)
    assert small.colors is not None and small.normals is not None
    palette = np.array([_rgb(entry[0]) for entry in PALETTE.values()])
    colours = np.asarray(small.colors, float)
    assert np.isclose(colours[:, None, :], palette[None, :, :], atol=1e-6).all(2).any(1).all()


def test_a_palette_override_repaints_the_merged_mesh():
    car = build_car()
    red = np.array([1.0, 0.0, 0.0])
    repainted = np.asarray(car.merged(palette={"paint": ("#ff0000", 0.3, 0.1)}).colors, float)
    plain = np.asarray(car.merged().colors, float)
    assert (np.abs(repainted - red).max(1) < 1e-6).any()
    assert not (np.abs(plain - red).max(1) < 1e-6).any()


POLICE = CarParams(
    livery=(("#1f3f8f", [(0.0, 0.3), (1.0, 0.3), (1.0, 0.75), (0.0, 0.75)]),),
    lightbar="bar",
)


def test_a_police_sedan_wears_a_livery_band_on_both_flanks():
    car = build_car(POLICE)
    livery = _pieces(car, "livery0")
    assert livery
    v = np.concatenate([mesh.positions for mesh in livery]).astype(float)
    assert v[:, 2].min() < -0.5 and v[:, 2].max() > 0.5  # both sides
    assert 0.25 < v[:, 1].min() and v[:, 1].max() < 0.80  # at door height, not over the roof
    # The colour reaches the palette the merged mesh and spawn read.
    assert np.allclose(_rgb(car.palette["livery0"][0]), _rgb("#1f3f8f"))


def test_a_police_sedan_carries_a_two_colour_lightbar_over_the_roof_peak():
    car = build_car(POLICE)
    for name in ("beacon_a", "beacon_b"):
        v = np.concatenate([mesh.positions for mesh in _pieces(car, name)]).astype(float)
        assert v[:, 1].max() > car.params.height + 0.01  # standing over the roof
    a = np.concatenate([m.positions for m in _pieces(car, "beacon_a")]).astype(float)
    b = np.concatenate([m.positions for m in _pieces(car, "beacon_b")]).astype(float)
    assert a[:, 2].mean() * b[:, 2].mean() < 0  # one colour each side of the roof
    peak = car.params.length / 2 - car.params.roof_peak_u * car.params.length
    assert abs(np.concatenate([a, b])[:, 0].mean() - peak) < 0.25
    # spawn_car finds a colour for every piece it is handed.
    materials = {mat for part in car.parts for mat in part.pieces}
    assert materials <= set(car.palette)


def test_a_dome_is_a_single_small_lamp_at_the_roof_peak():
    car = build_car(CarParams(lightbar="dome"))
    v = np.concatenate([mesh.positions for mesh in _pieces(car, "beacon_a")]).astype(float)
    assert 0 < len(v)
    assert v[:, 1].max() > car.params.height and np.ptp(v[:, 2]) < 0.4
    assert not _pieces(car, "beacon_b")  # a dome is one lamp
