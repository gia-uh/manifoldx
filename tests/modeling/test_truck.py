"""Parametric trucks: the presets' measures, their parts, the dump pose and the merged mesh."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from manifoldx.modeling.assets.truck import (
    Truck,
    TruckParams,
    build_truck,
    box_truck,
    cargo_truck_1980s,
    dump_truck,
    fire_engine,
    passenger_truck,
    tanker,
)
from manifoldx.modeling.assets.vehicle import Vehicle

PRESETS = {
    "cargo": cargo_truck_1980s,
    "box": box_truck,
    "tanker": tanker,
    "dump": dump_truck,
    "fire": fire_engine,
    "passenger": passenger_truck,
}


def _verts(part):
    return np.concatenate([mesh.positions for mesh in part.pieces.values()]).astype(float)


def _posed(part, pose):
    pos, q = pose[part.name]
    v = _verts(part)
    u, w = q[:3], q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v) + pos


def _triangles(truck):
    return sum(len(mesh.faces) for part in truck.parts for mesh in part.pieces.values())


def _pieces(truck, material):
    return [mesh for part in truck.parts for mat, mesh in part.pieces.items() if mat == material]


def test_a_truck_is_a_vehicle():
    truck = build_truck()
    assert isinstance(truck, Truck) and isinstance(truck, Vehicle)
    assert truck.part("body").kind == "body"
    assert set(truck.pose()) == {part.name for part in truck.parts}
    assert {"body", "paint"} <= set(truck.palette)


@pytest.mark.parametrize("name", PRESETS)
def test_each_preset_builds_to_its_measures(name):
    p = PRESETS[name]()
    truck = build_truck(p)
    v = np.concatenate([_verts(part) for part in truck.parts])
    lo, hi = v.min(0), v.max(0)
    assert abs((hi[0] - lo[0]) - p.length) <= 0.08 * p.length
    assert abs((hi[2] - lo[2]) - p.width) <= 0.18 * p.width
    assert abs(hi[1] - p.height) <= 0.15
    assert abs(lo[1]) < 0.02  # the tyres touch the ground
    assert hi[1] > p.cab_height  # something rises above the cab


def test_the_presets_have_the_expected_package():
    assert cargo_truck_1980s().cab == "forward" and cargo_truck_1980s().body == "flatbed"
    assert cargo_truck_1980s().axles == 2
    assert box_truck().cab == "over" and box_truck().body == "box"
    assert tanker().body == "tanker" and tanker().axles == 3
    assert dump_truck().body == "dump" and dump_truck().axles == 3
    assert fire_engine().cab == "over" and fire_engine().body == "fire"
    passenger = passenger_truck()
    assert (passenger.body, passenger.bench_facing, passenger.bench_rows) == ("passenger", "forward", 3)
    assert passenger.rear_ladder and passenger.hoops == 4


@pytest.mark.parametrize("name", PRESETS)
def test_axles_and_twin_rear_wheels_set_the_wheel_count(name):
    p = PRESETS[name]()
    truck = build_truck(p)
    wheels = [part for part in truck.parts if part.kind == "wheel"]
    assert len(wheels) == 2 + 4 * (p.axles - 1)
    assert len({part.name for part in wheels}) == len(wheels)
    assert sum(part.steer for part in wheels) == 2


@pytest.mark.parametrize("name", PRESETS)
def test_every_door_opens_and_swings_out_of_the_body(name):
    truck = build_truck(PRESETS[name]())
    doors = [part for part in truck.parts if part.kind == "door"]
    assert len(doors) >= 2  # a cab door on each side, at least
    shut, opened = truck.pose(), truck.pose(doors=1.0)
    for door in doors:
        moved = np.abs(_posed(door, opened) - _posed(door, shut)).max()
        assert moved > 0.15


def test_the_box_trucks_rear_doors_swing_out_behind_it():
    truck = build_truck(box_truck())
    rear = [part for part in truck.parts if part.kind == "door" and part.name.startswith("door_rear")]
    assert len(rear) == 2
    shut, opened = truck.pose(), truck.pose(doors=1.0)
    for door in rear:
        before, after = _posed(door, shut), _posed(door, opened)
        assert after[:, 0].min() < before[:, 0].min() - 0.15  # out behind the box


def test_the_dump_bin_pivots_at_its_rear_edge_and_raises_its_front():
    truck = build_truck(dump_truck())
    bin_ = truck.part("bin")
    assert bin_.kind == "bin"
    rest = _posed(bin_, truck.pose())
    tipped = _posed(bin_, truck.pose(dump=1.0))
    front = rest[:, 0] > np.median(rest[:, 0])
    assert tipped[front, 1].mean() > rest[front, 1].mean() + 0.5
    assert tipped[:, 1].max() > rest[:, 1].max() + 0.5
    assert np.allclose(tipped[:, 2].min(), rest[:, 2].min(), atol=0.01)  # it turns about a lateral axis


@pytest.mark.parametrize("name", PRESETS)
def test_wheels_turn_about_their_hubs_and_only_the_front_ones_steer(name):
    truck = build_truck(PRESETS[name]())
    pose = truck.pose(wheel_turn=1.2, steer=0.35)
    for part in truck.parts:
        if part.kind != "wheel":
            continue
        v = _verts(part)
        centre = (v.min(0) + v.max(0)) / 2
        pos, q = pose[part.name]
        u, w = q[:3], q[3]
        moved = centre + 2 * np.cross(u, np.cross(u, centre) + w * centre) + pos
        assert np.allclose(moved, part.pivot, atol=0.03)
        assert (abs(q[1]) > 0.1) == part.steer


@pytest.mark.parametrize("name", PRESETS)
def test_the_merged_mesh_sits_on_the_ground_and_is_centred(name):
    truck = build_truck(PRESETS[name]())
    whole = truck.merged()
    small = truck.merged(grid=72)
    assert len(small.positions) < len(whole.positions)
    assert small.colors is not None and small.normals is not None
    assert np.isfinite(np.asarray(small.positions)).all()
    p = np.asarray(small.positions, float)
    assert abs(p[:, 1].min()) < 0.02  # the tyres touch the ground
    assert abs(p[:, 0].max() + p[:, 0].min()) < 0.03
    assert abs(p[:, 2].max() + p[:, 2].min()) < 0.03


def test_the_body_has_a_paint_slot_of_its_own():
    truck = build_truck(box_truck())
    assert truck.palette["body"][0] != truck.palette["paint"][0]
    assert _pieces(truck, "body") and _pieces(truck, "paint")


def test_a_fire_engine_carries_beacons_a_ladder_and_a_hose_reel():
    p = fire_engine()
    truck = build_truck(p)
    beacons = np.concatenate([mesh.positions for mesh in _pieces(truck, "beacon_a") + _pieces(truck, "beacon_b")])
    assert len(beacons) and beacons[:, 1].max() > p.cab_height + 0.01
    ladder = np.concatenate([mesh.positions for mesh in _pieces(truck, "ladder")]).astype(float)
    assert len(ladder)
    assert ladder[:, 1].min() > p.cab_height  # stowed over the body's roof
    assert np.ptp(ladder[:, 0]) > 2.0  # two long rails
    assert _pieces(truck, "hose") and _pieces(truck, "locker")
    # the red paint and its white stripe reach the palette the merged mesh reads
    assert {"body", "paint", "livery0"} <= set(truck.palette)


def test_detail_scales_the_resolution():
    plain = build_truck(box_truck())
    coarse = build_truck(replace(box_truck(), detail=0.5))
    assert _triangles(coarse) < _triangles(plain)


def _signature(truck):
    """A short hash of the merged mesh's positions, rounded so a no-op change stays bit for bit equal."""
    p = np.round(np.asarray(truck.merged().positions, np.float64), 4)
    return hashlib.sha256(p.tobytes()).hexdigest()[:16]


def test_the_untouched_truck_presets_build_exactly_as_before():
    golden = {
        "cargo_truck_1980s": "7181623d9182a1e8",
        "box_truck": "eed8be7ffbecd7e7",
        "tanker": "b5c43ac8ff314531",
        "dump_truck": "e485b4f4ad9968b5",
        "fire_engine": "228f5e95c7b39b2d",
    }
    for name, digest in golden.items():
        preset = {"cargo_truck_1980s": cargo_truck_1980s, "box_truck": box_truck, "tanker": tanker,
                  "dump_truck": dump_truck, "fire_engine": fire_engine}[name]
        assert _signature(build_truck(preset())) == digest, name


def test_a_passenger_truck_carries_benches_hoops_a_canvas_roof_and_a_rear_ladder():
    p = passenger_truck()
    truck = build_truck(p)
    canvas = np.concatenate([mesh.positions for mesh in _pieces(truck, "canvas")]).astype(float)
    assert canvas[:, 1].max() == pytest.approx(p.canvas_height, abs=0.03)
    assert np.ptp(canvas[:, 2]) > 0.9 * p.width  # a roof over the whole body
    ladder = np.concatenate([mesh.positions for mesh in _pieces(truck, "ladder")]).astype(float)
    rear = ladder[ladder[:, 0] < -p.length / 2 + 0.4]
    assert len(rear) and np.ptp(rear[:, 1]) > 1.5  # rails down the rear face
    deck = np.concatenate([mesh.positions for mesh in _pieces(truck, "deck")]).astype(float)
    seats = deck[(deck[:, 1] > p.deck_height + p.bench_height - 0.06)]
    assert np.ptp(seats[:, 0]) > 1.5  # rows spread along the body


def test_a_side_facing_passenger_body_leaves_a_central_aisle():
    p = replace(passenger_truck(), bench_facing="side", bench_rows=2)
    truck = build_truck(p)
    deck = np.concatenate([mesh.positions for mesh in _pieces(truck, "deck")]).astype(float)
    seats = deck[deck[:, 1] > p.deck_height + p.bench_height - 0.06]
    assert seats[:, 2].min() < -0.6 and seats[:, 2].max() > 0.6  # a bench down each side
    assert np.abs(seats[:, 2]).min() > 0.25  # and nothing in the aisle


def test_a_canvas_that_does_not_clear_the_deck_is_refused():
    with pytest.raises(ValueError, match="deck_height"):
        TruckParams(body="passenger", canvas_height=2.0, deck_height=2.0).validate()
    with pytest.raises(ValueError, match="canvas_height"):
        TruckParams(body="passenger", bench_height=0.8, canvas_height=2.0, deck_height=1.3).validate()
