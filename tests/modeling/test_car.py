import numpy as np

from manifoldx.modeling.assets.car import CarParams, build_car


def _verts(part):
    return np.concatenate([m.positions for m in part.pieces.values()])


def _posed(part, pose):
    pos, q = pose[part.name]
    v = _verts(part).astype(float)
    u, w = q[:3], q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v) + pos


def test_measurements_reach_the_mesh():
    p = CarParams(length=4.2, width=1.8, height=1.5)
    car = build_car(p)
    body = np.concatenate([_verts(car.part(n)) for n in ("body", "hood", "hatch", "door_fl", "door_fr")])
    lo, hi = body.min(0), body.max(0)
    assert abs((hi[0] - lo[0]) - 4.2) < 0.03
    assert abs(hi[1] - 1.5) < 0.01
    # Mirrors stick out past the body; without them the body is the width.
    assert abs(_verts(car.part("body"))[:, 2].max() - 0.9) < 0.01
    wheel = _verts(car.part("wheel_fl"))
    assert abs(wheel[:, 1].max() - 2 * p.wheel_radius) < 0.01


def test_wheelbase_moves_the_rear_axle():
    a, b = build_car(CarParams(wheelbase=2.5)), build_car(CarParams(wheelbase=2.8))
    def gap(car):
        return car.part("wheel_fl").pivot[0] - car.part("wheel_rl").pivot[0]
    assert abs(gap(a) - 2.5) < 1e-9 and abs(gap(b) - 2.8) < 1e-9


def test_regions_and_parts():
    car = build_car()
    names = {p.name for p in car.parts}
    assert {"body", "hood", "hatch", "door_fl", "door_fr", "door_rl", "door_rr"} <= names
    assert {f"wheel_{a}{s}" for a in "fr" for s in "lr"} <= names
    materials = {m for p in car.parts for m in p.pieces}
    assert {"paint", "glass", "lamp", "tail", "plate", "liner", "tyre"} <= materials


def test_doors_hood_and_hatch_open_outward_and_up():
    car = build_car()
    shut, open_ = car.pose(), car.pose(doors=1.0, hood=1.0, hatch=1.0)
    right, left = car.part("door_fr"), car.part("door_fl")
    assert _posed(right, open_)[:, 2].max() > _posed(right, shut)[:, 2].max() + 0.3
    assert _posed(left, open_)[:, 2].min() < _posed(left, shut)[:, 2].min() - 0.3
    for name in ("hood", "hatch"):
        part = car.part(name)
        assert _posed(part, open_)[:, 1].max() > _posed(part, shut)[:, 1].max() + 0.2


def test_wheels_turn_about_their_hubs_and_only_the_front_steers():
    car = build_car()
    pose = car.pose(wheel_turn=1.3, steer=0.4)
    for part in car.parts:
        if part.kind == "wheel":
            centre = (_verts(part).min(0) + _verts(part).max(0)) / 2
            pos, q = pose[part.name]
            u, w = q[:3], q[3]
            moved = centre + 2 * np.cross(u, np.cross(u, centre) + w * centre) + pos
            assert np.allclose(moved, part.pivot, atol=0.02)
            assert (abs(q[1]) > 0.1) == part.steer  # a turn about Y only on the steered wheels


def test_same_parameters_same_car():
    a, b = build_car(), build_car()
    for pa, pb in zip(a.parts, b.parts):
        for m in pa.pieces:
            assert np.array_equal(pa.pieces[m].positions, pb.pieces[m].positions)


def test_every_face_sits_in_one_island_of_the_atlas():
    from manifoldx.modeling.assets.car import ATLAS

    car = build_car()
    boxes = np.array(list(ATLAS.values()))
    for part in car.parts:
        for mesh in part.pieces.values():
            uv = mesh.uvs[mesh.faces]  # (faces, 3, 2)
            lo, hi = uv.min(1), uv.max(1)
            inside = ((lo[:, None, :] >= boxes[None, :, :2] - 1e-5) & (hi[:, None, :] <= boxes[None, :, 2:] + 1e-5)).all(2)
            assert inside.any(1).all()
    # The left flank is drawn as a side view: further back on the car is further right in its island.
    door = car.part("door_fl").pieces["paint"]
    order = np.argsort(-door.positions[:, 0])
    s0, _, s1, _ = ATLAS["left"]
    flank = (door.uvs[:, 0] >= s0) & (door.uvs[:, 0] <= s1) & (door.uvs[:, 1] <= ATLAS["left"][3])
    x, s = door.positions[order, 0][flank[order]], door.uvs[order, 0][flank[order]]
    assert np.corrcoef(x, s)[0, 1] < -0.99


def test_a_wheel_shape_stands_inside_the_rim_and_both_sides_wear_one_picture():
    p = CarParams()
    car = build_car(p)
    left, right = car.part("wheel_fl"), car.part("wheel_fr")
    face = left.pieces["alloy"].positions
    assert np.hypot(*(face[:, :2] - left.pivot[:2]).T).max() <= p.wheel_radius * p.rim_ratio * 1.01
    assert abs(face[:, 2] - left.pivot[2]).max() < p.tyre_width / 2  # behind the tyre's sidewall
    assert np.allclose(left.pieces["alloy"].uvs, right.pieces["alloy"].uvs, atol=1e-6)
    plain = build_car(CarParams(wheel="spokes", mirror="box")).part("wheel_fl")
    assert "spoke" in plain.pieces and "alloy" not in plain.pieces


def test_a_mirror_shape_faces_outward_and_fills_its_size():
    from manifoldx.modeling.assets.car import _mirror
    from manifoldx.modeling.assets.mirrors import MIRRORS

    size = (0.20, 0.165)
    for side in (1, -1):
        pieces = _mirror(MIRRORS["pod"], size, side)
        shell = [(v, t) for m, v, t, _, _ in pieces if m in ("paint", "black")]
        volume = sum(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6 for v, t in shell)
        assert volume > 0  # the faces look out of the shell, not into it
        v = shell[0][0]
        assert np.allclose([np.ptp(v[:, 2]), np.ptp(v[:, 1])], size, rtol=0.06)
        glass = next(v for m, v, _, _, _ in pieces if m == "glass")
        assert glass[:, 0].mean() < v[:, 0].mean()  # the glass is on the rear side
        assert (side * v[:, 2]).max() > 0.1  # and the mirror stands out from the door
