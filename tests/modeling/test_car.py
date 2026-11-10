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
    plain = build_car(CarParams(wheel="spokes", mirror="box", grille_style="bar", grille=(0.60, 0.27, 0.43), dash="box",
                                steering="ring"))
    assert "spoke" in plain.part("wheel_fl").pieces and "alloy" not in plain.part("wheel_fl").pieces
    assert len(_low_lamps(plain)) > 0
    assert "dash" in plain.part("cabin").pieces and "chrome" not in plain.part("cabin").pieces


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


def _low_lamps(car):
    v = car.part("body").pieces["lamp"].positions
    return v[v[:, 1] < 0.6]  # the lamps under the number plate: the grille's


def test_a_grille_shape_fills_its_box_and_keeps_its_ends():
    from manifoldx.modeling.assets.car import _slabs

    p = CarParams()
    half, bottom, top = p.grille
    rings = _low_lamps(build_car(p))
    assert abs(abs(rings[:, 2]).max() - half) < 0.01
    assert bottom - 0.01 < rings[:, 1].min() and rings[:, 1].max() < top + 0.01
    # A shorter grille moves its rings together; it does not squeeze them.
    shorter = _low_lamps(build_car(CarParams(grille=(half - 0.1, bottom, top))))
    assert abs(abs(shorter[:, 2]).max() - (half - 0.1)) < 0.01
    assert abs(np.ptp(abs(shorter[:, 2])) - np.ptp(abs(rings[:, 2]))) < 0.005
    assert len(_low_lamps(build_car(CarParams(fog_rings=False)))) == 0
    bar = np.array([(0, 0), (1, 0), (1, 0.1), (0.5, 0.2), (0, 0.1)])
    cut = _slabs(bar, 0.3)
    assert np.isclose(sum(abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])) / 2 for q in cut), 0.15)


def _volume(pieces):
    return sum(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6 for v, t in pieces)


def test_a_dashboard_shape_spans_the_cabin_and_its_screens_stand_over_the_deck():
    from manifoldx.modeling.assets.car import _dash, _steering
    from manifoldx.modeling.assets.dashboards import DASHBOARDS
    from manifoldx.modeling.assets.steering import STEERING

    width, deck, rear, front = 1.5, 1.06, 0.40, 0.84
    pieces = _dash(DASHBOARDS["deck"], width, deck, rear, front, 40.0)
    solid = np.concatenate([v for m, v, _, _, _ in pieces if m == "dash"])
    assert np.allclose([solid[:, 2].min(), solid[:, 2].max()], [-width / 2, width / 2], atol=0.01)
    assert solid[:, 1].max() <= deck + 1e-6 and abs(solid[:, 0].min() - rear) < 0.01 and solid[:, 0].max() <= front + 1e-6
    assert _volume([(v, t) for m, v, t, _, _ in pieces if m == "dash"]) > 0  # its faces look outward
    screens = np.concatenate([v for m, v, _, _, shown in pieces if m == "black" and shown])
    assert screens[:, 1].max() > deck + 0.05 and screens[:, 0].max() < rear  # over the deck, on the cabin's side
    assert {"seat", "chrome"} <= {m for m, *_ in pieces}  # the leather panel and the vent's frame are laid on it
    wheel = _steering(STEERING["tri"], 0.19)
    rim = [(v, t) for m, v, t, _ in wheel if m == "dash"]
    assert _volume(rim) > 0 and max(np.hypot(v[:, 0], v[:, 1]).max() for v, _ in rim) < 0.19 * 1.1  # wider than tall
