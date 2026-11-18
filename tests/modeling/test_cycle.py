"""Bicycles, cargo tricycles, cycle rickshaws and motorcycles: their measurements, parts, steering and wheels."""

from dataclasses import replace

import numpy as np
import pytest

from manifoldx.modeling.assets.cycle import (
    Cycle,
    CycleParams,
    bicycle,
    build_cycle,
    cargo_tricycle,
    cycle_rickshaw,
    electric_moto,
    motorcycle,
)
from manifoldx.modeling.assets.vehicle import Vehicle
from manifoldx.modeling.mesh import Mesh

PRESETS = (bicycle, cycle_rickshaw, cargo_tricycle, motorcycle, electric_moto)


def _verts(part):
    return np.concatenate([m.positions for m in part.pieces.values()]).astype(float)


def _posed(cycle, name, **pose):
    at, q = cycle.pose(**pose)[name]
    v = _verts(cycle.part(name))
    u, w = np.asarray(q, float)[:3], np.asarray(q, float)[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v) + np.asarray(at, float)


def _bounds(cycle):
    p = np.asarray(cycle.merged().positions, float)
    return p.min(0), p.max(0)


def _material(cycle, name):
    return np.concatenate([mesh.positions for part in cycle.parts for mat, mesh in part.pieces.items()
                           if mat == name]).astype(float)


def test_every_preset_is_a_vehicle():
    for preset in PRESETS:
        cycle = build_cycle(preset())
        assert isinstance(cycle, Cycle) and isinstance(cycle, Vehicle)
        assert cycle.part("frame").kind == "body"
        assert cycle.part("steer").kind == "steer"
        assert set(cycle.pose()) == {part.name for part in cycle.parts}
        assert cycle.joints["front_axle"][1] == pytest.approx(cycle.params.wheel_radius)


def test_the_bicycle_is_about_eighteen_hands_long():
    lo, hi = _bounds(build_cycle(bicycle()))
    assert 1.7 < hi[0] - lo[0] < 1.9
    assert 1.0 < hi[1] - lo[1] < 1.2
    assert abs(lo[1]) < 0.01


def test_the_rickshaw_is_long_and_its_canopy_is_head_high():
    lo, hi = _bounds(build_cycle(cycle_rickshaw()))
    assert 2.6 < hi[0] - lo[0] < 3.2
    assert 1.9 < hi[1] - lo[1] < 2.2


def test_the_motorcycle_keeps_its_1970s_length():
    lo, hi = _bounds(build_cycle(motorcycle()))
    assert 1.9 < hi[0] - lo[0] < 2.1


def test_wheel_count_follows_the_layout():
    expected = {
        "bicycle": {"wheel_front", "wheel_rear"},
        "cycle_rickshaw": {"wheel_front", "wheel_rear_left", "wheel_rear_right"},
        "cargo_tricycle": {"wheel_front", "wheel_rear_left", "wheel_rear_right"},
        "motorcycle": {"wheel_front", "wheel_rear"},
        "electric_moto": {"wheel_front", "wheel_rear"},
    }
    for preset in PRESETS:
        cycle = build_cycle(preset())
        wheels = {part.name for part in cycle.parts if part.kind == "wheel"}
        assert wheels == expected[preset.__name__]
        assert len(wheels) == cycle.params.wheels


def test_a_tadpole_puts_the_two_wheels_at_the_front():
    cycle = build_cycle(CycleParams(wheels=3, layout="tadpole"))
    wheels = {part.name for part in cycle.parts if part.kind == "wheel"}
    assert wheels == {"wheel_front_left", "wheel_front_right", "wheel_rear"}
    left, right = cycle.part("wheel_front_left"), cycle.part("wheel_front_right")
    assert left.pivot[2] * right.pivot[2] < 0
    assert cycle.part("wheel_rear").pivot[2] == pytest.approx(0.0)


def test_steering_turns_the_bars_and_the_front_wheels_but_not_the_rear():
    for preset in PRESETS:
        cycle = build_cycle(preset())
        still = cycle.pose(steer=0.0)
        turned = cycle.pose(steer=0.3)
        assert not np.allclose(turned["steer"][1], still["steer"][1])
        for part in cycle.parts:
            moved = not np.allclose(_posed(cycle, part.name, steer=0.3), _posed(cycle, part.name))
            assert moved == (part.kind == "steer" or (part.kind == "wheel" and part.steer)), part.name
        assert cycle.part("steer").axis @ (cycle.joints["head_top"] - cycle.joints["head"]) > 0


def test_the_wheels_turn_about_their_axles():
    cycle = build_cycle(bicycle())
    for name in ("wheel_front", "wheel_rear"):
        assert not np.allclose(_posed(cycle, name, wheel_turn=0.6), _posed(cycle, name))


def test_merged_is_one_grounded_coloured_mesh():
    for preset in PRESETS:
        cycle = build_cycle(preset())
        mesh = cycle.merged()
        assert isinstance(mesh, Mesh)
        assert mesh.colors is not None and mesh.colors.shape == (len(mesh.positions), 3)
        assert mesh.normals is not None and np.isfinite(mesh.normals).all()
        p = np.asarray(mesh.positions, float)
        assert abs(p[:, 1].min()) < 0.01
        assert abs(p[:, 0].max() + p[:, 0].min()) < 0.01


def test_a_coarser_detail_finds_fewer_triangles():
    for preset in PRESETS:
        coarse = build_cycle(replace(preset(), detail=0.5)).merged()
        fine = build_cycle(preset()).merged()
        assert len(coarse.faces) < len(fine.faces)


def test_the_presets_carry_what_their_names_promise():
    rickshaw = cycle_rickshaw()
    assert (rickshaw.layout, rickshaw.payload, rickshaw.canopy, rickshaw.power) == ("delta", "bench", True, "pedal")
    cargo = cargo_tricycle()
    assert (cargo.layout, cargo.payload, cargo.power) == ("delta", "cargo_box", "hub")
    moto = motorcycle()
    assert (moto.layout, moto.power) == ("inline", "engine")
    assert 0.28 < moto.wheel_radius < 0.33
    scooter = electric_moto()
    assert (scooter.layout, scooter.power, scooter.step_through) == ("inline", "hub", True)


def test_the_frame_is_tubes_between_the_named_joints():
    cycle = build_cycle(bicycle())
    assert {"head", "seat", "bottom_bracket", "rear_axle", "front_axle"} <= set(cycle.joints)
    frame = _verts(cycle.part("frame"))
    assert frame[:, 1].max() > cycle.joints["seat"][1]  # the saddle stands over the seat joint
    assert frame[:, 1].min() < cycle.params.wheel_radius  # the frame hangs down to the chain stays


def test_a_cargo_box_sits_over_the_rear_axle_and_a_canopy_over_it():
    cycle = build_cycle(cargo_tricycle())
    box = _material(cycle, "box")
    rear = cycle.joints["rear_axle"][0]
    assert box[:, 0].min() < rear < box[:, 0].max()  # the box spans the rear axle
    assert box[:, 1].max() < cycle.params.canopy_height
    canopy = _material(cycle, "canopy")
    assert canopy[:, 1].max() == pytest.approx(cycle.params.canopy_height, abs=0.05)
    assert np.ptp(canopy[:, 0]) > 0.5  # a hood, not a point


def test_a_bench_seats_two_behind_the_rider():
    cycle = build_cycle(cycle_rickshaw())
    bench = _material(cycle, "box")
    assert np.ptp(bench[:, 2]) > 0.6  # wide enough for two
    assert bench[:, 0].mean() < cycle.joints["seat"][0]  # behind the rider
