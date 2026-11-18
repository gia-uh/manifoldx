"""A parametric city bus: measurements, parts, glass, livery, detail and the merged mesh."""

import hashlib

import numpy as np
import pytest

from manifoldx.modeling.assets.bus import Bus, BusParams, build_bus, coach
from manifoldx.modeling.assets.vehicle import Vehicle


def _verts(part):
    return np.concatenate([mesh.positions for mesh in part.pieces.values()]).astype(float)


def _posed(part, pose):
    pos, q = pose[part.name]
    v = _verts(part)
    u, w = q[:3], q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v) + pos


def _triangles(bus):
    return sum(len(mesh.faces) for part in bus.parts for mesh in part.pieces.values())


def _pieces(bus, material):
    return [mesh for part in bus.parts for mat, mesh in part.pieces.items() if mat == material]


def test_the_defaults_are_a_twelve_metre_low_floor_city_bus():
    p = BusParams()
    assert (p.length, p.width, p.height) == (12.0, 2.55, 3.1)
    assert (p.floor, p.corner_radius) == (0.35, 0.25)
    assert (p.front_overhang, p.wheelbase) == (2.6, 6.0)
    assert (p.wheel_radius, p.tyre_width, p.track) == (0.5, 0.3, 2.1)
    assert p.doors == ((0.04, 1.2, 2), (0.47, 1.2, 2), (0.86, 1.2, 2))


def test_the_body_measures_its_length_width_and_height():
    p = BusParams()
    bus = build_bus(p)
    body = _verts(bus.part("body"))
    lo, hi = body.min(0), body.max(0)
    assert abs((hi[0] - lo[0]) - p.length) <= 0.02 * p.length
    assert abs((hi[2] - lo[2]) - p.width) <= 0.02 * p.width
    assert abs(hi[1] - p.height) <= 0.02 * p.height
    assert abs(lo[1] - (p.floor - 0.09)) < 0.25  # a low floor, not a car's sill


def test_part_names_include_the_door_leaves_and_the_four_wheels():
    bus = build_bus()
    doors = [part for part in bus.parts if part.kind == "door"]
    assert len(doors) == sum(leaves for _, _, leaves in BusParams().doors)
    assert {part.name for part in doors} == {
        f"door_{d + 1}{leaf}" for d, (_, _, leaves) in enumerate(BusParams().doors) for leaf in "ab"[:leaves]
    }
    wheels = {part.name for part in bus.parts if part.kind == "wheel"}
    assert wheels == {"wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"}
    assert sum(part.steer for part in bus.parts if part.kind == "wheel") == 2
    assert bus.part("body").kind == "body"
    assert isinstance(bus, Bus) and isinstance(bus, Vehicle)
    assert set(bus.pose()) == {part.name for part in bus.parts}


def test_a_door_opens_outward_on_its_vertical_hinge():
    bus = build_bus()
    shut, open_ = bus.pose(), bus.pose(doors=1.0)
    for part in bus.parts:
        if part.kind != "door":
            continue
        assert np.allclose(part.axis, [0.0, 1.0, 0.0])  # a vertical hinge
        before, after = _posed(part, shut), _posed(part, open_)
        assert np.abs(after - before).max() > 0.2
        assert after[:, 2].max() > before[:, 2].max() + 0.1  # both leaves swing out of the right flank
        # the hinge edge itself stays put
        assert np.allclose(before[:, 2].min(), part.pivot[2] - 0.0, atol=0.01)


def test_wheels_turn_about_their_hubs_and_only_the_front_ones_steer():
    bus = build_bus()
    pose = bus.pose(wheel_turn=1.3, steer=0.4)
    for part in bus.parts:
        if part.kind != "wheel":
            continue
        centre = (_verts(part).min(0) + _verts(part).max(0)) / 2
        pos, q = pose[part.name]
        u, w = q[:3], q[3]
        moved = centre + 2 * np.cross(u, np.cross(u, centre) + w * centre) + pos
        assert np.allclose(moved, part.pivot, atol=0.02)
        assert (abs(q[1]) > 0.1) == part.steer


def test_glass_is_on_both_sides_and_on_the_front():
    p = BusParams()
    bus = build_bus(p)
    glass = np.concatenate([mesh.positions for mesh in _pieces(bus, "glass")]).astype(float)
    sides = glass[(glass[:, 0] > -p.length / 2 + 1.0) & (glass[:, 0] < p.length / 2 - 1.0)]
    assert sides[:, 2].min() < -0.8 * p.width / 2
    assert sides[:, 2].max() > 0.8 * p.width / 2
    front = glass[glass[:, 0] > p.length / 2 - 0.4]
    assert len(front) > 20  # the windscreen, on the front face
    assert front[:, 1].min() >= p.windscreen_bottom - 0.15  # a triangle's corner may cross the frame
    assert front[:, 1].max() <= p.windscreen_top + 0.15


def test_livery_adds_triangles_on_both_flanks():
    p = BusParams(livery=(("#2f9e44", [(0.02, 0.5), (0.98, 0.5), (0.98, 1.2), (0.02, 1.2)]),))
    plain, painted = build_bus(), build_bus(p)
    v = np.concatenate([mesh.positions for mesh in _pieces(painted, "livery0")]).astype(float)
    assert len(v) and v[:, 2].min() < -0.9 and v[:, 2].max() > 0.9
    assert _triangles(painted) > _triangles(plain)
    assert "livery0" in painted.palette


def test_the_merged_mesh_sits_on_the_ground_and_works_on_a_grid():
    bus = build_bus()
    whole = bus.merged()
    small = bus.merged(grid=72)
    assert len(small.positions) < len(whole.positions)
    assert small.colors is not None and small.normals is not None
    p = np.asarray(small.positions, float)
    assert abs(p[:, 1].min()) < 0.01  # the tyres touch the ground
    assert abs(p[:, 0].max() + p[:, 0].min()) < 0.02
    assert abs(p[:, 2].max() + p[:, 2].min()) < 0.02


def test_detail_scales_the_resolution():
    assert _triangles(build_bus(BusParams(detail=0.5))) < _triangles(build_bus())


def test_the_city_bus_builds_exactly_as_before():
    p = np.round(np.asarray(build_bus().merged().positions, np.float64), 4)
    assert hashlib.sha256(p.tobytes()).hexdigest()[:16] == "31a7f9a26a74097a"


def test_a_coach_has_one_front_door_small_tinted_windows_and_luggage_bays():
    p = coach()
    bus = build_bus(p)
    doors = [part for part in bus.parts if part.kind == "door"]
    assert len(doors) == 1
    tinted = np.concatenate([mesh.positions for mesh in _pieces(bus, "tinted")]).astype(float)
    assert tinted[:, 2].min() < -0.9 and tinted[:, 2].max() > 0.9  # a band down each flank
    assert np.ptp(tinted[:, 1]) < 0.9  # a shallow band, not the city bus's tall windows
    bays = np.concatenate([mesh.positions for mesh in _pieces(bus, "bay")]).astype(float)
    assert bays[:, 1].max() < p.floor  # below the passenger floor
    assert bays[:, 2].min() < -0.9 and bays[:, 2].max() > 0.9
    assert {"bay", "tinted"} <= set(bus.palette)
    assert p.windscreen_rake > BusParams().windscreen_rake
    assert p.window_top - p.window_bottom < BusParams().window_top - BusParams().window_bottom


def test_a_coach_measures_its_package():
    p = coach()
    bus = build_bus(p)
    body = _verts(bus.part("body"))
    lo, hi = body.min(0), body.max(0)
    assert abs((hi[0] - lo[0]) - p.length) <= 0.02 * p.length
    assert abs((hi[2] - lo[2]) - p.width) <= 0.03 * p.width
    assert abs(hi[1] - p.height) <= 0.02 * p.height


def test_a_luggage_bay_at_or_above_the_floor_is_refused():
    with pytest.raises(ValueError, match="floor"):
        BusParams(luggage_bays=4, floor=0.8, bay_top=0.9).validate()
