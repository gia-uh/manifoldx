import importlib.util
from pathlib import Path

import numpy as np
import pytest

from manifoldx.modeling.assets.car import CarParams, _body_grid, _lamp_centers, build_car

_spec = importlib.util.spec_from_file_location("car_shapes", Path(__file__).parents[2] / "examples/models/car_shapes.py")
shapes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shapes)  # one traced shape of each kind


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
    p = CarParams(wheel=shapes.SWIRL)
    car = build_car(p)
    left, right = car.part("wheel_fl"), car.part("wheel_fr")
    face = left.pieces["alloy"].positions
    assert np.hypot(*(face[:, :2] - left.pivot[:2]).T).max() <= p.wheel_radius * p.rim_ratio * 1.01
    assert abs(face[:, 2] - left.pivot[2]).max() < p.tyre_width / 2  # behind the tyre's sidewall
    assert np.allclose(left.pieces["alloy"].uvs, right.pieces["alloy"].uvs, atol=1e-6)
    plain = build_car(CarParams(fog_rings=True))  # the defaults are the plain parts
    assert "seat" not in plain.part("door_fl").pieces  # no leather on a bare door
    assert "spoke" in plain.part("wheel_fl").pieces and "alloy" not in plain.part("wheel_fl").pieces
    assert len(_low_lamps(plain)) > 0
    assert "dash" in plain.part("cabin").pieces and "chrome" not in plain.part("cabin").pieces


def test_a_mirror_shape_faces_outward_and_fills_its_size():
    from manifoldx.modeling.assets.car import _mirror

    size = (0.20, 0.165)
    for side in (1, -1):
        pieces = _mirror(shapes.POD, size, side)
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

    p = CarParams(grille_style=shapes.SMILE, grille=(0.83, 0.37, 0.50), fog_rings=True)
    half, bottom, top = p.grille
    rings = _low_lamps(build_car(p))
    assert abs(abs(rings[:, 2]).max() - half) < 0.01
    assert bottom - 0.01 < rings[:, 1].min() and rings[:, 1].max() < top + 0.01
    # A shorter grille moves its rings together; it does not squeeze them.
    shorter = _low_lamps(build_car(CarParams(grille_style=shapes.SMILE, grille=(half - 0.1, bottom, top), fog_rings=True)))
    assert abs(abs(shorter[:, 2]).max() - (half - 0.1)) < 0.01
    assert abs(np.ptp(abs(shorter[:, 2])) - np.ptp(abs(rings[:, 2]))) < 0.005
    assert len(_low_lamps(build_car(CarParams(grille_style=shapes.SMILE, grille=p.grille)))) == 0  # no fog rings by default
    bar = np.array([(0, 0), (1, 0), (1, 0.1), (0.5, 0.2), (0, 0.1)])
    cut = _slabs(bar, 0.3)
    assert np.isclose(sum(abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])) / 2 for q in cut), 0.15)


def _volume(pieces):
    return sum(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6 for v, t in pieces)


def test_a_dashboard_shape_spans_the_cabin_and_its_screens_stand_over_the_deck():
    from manifoldx.modeling.assets.car import _dash, _steering

    width, deck, rear, front = 1.5, 1.06, 0.40, 0.84
    pieces = _dash(shapes.DECK, width, deck, rear, front, 40.0)
    solid = np.concatenate([v for m, v, _, _, _ in pieces if m == "dash"])
    assert np.allclose([solid[:, 2].min(), solid[:, 2].max()], [-width / 2, width / 2], atol=0.01)
    assert solid[:, 1].max() <= deck + 1e-6 and abs(solid[:, 0].min() - rear) < 0.01 and solid[:, 0].max() <= front + 1e-6
    assert _volume([(v, t) for m, v, t, _, _ in pieces if m == "dash"]) > 0  # its faces look outward
    screens = np.concatenate([v for m, v, _, _, shown in pieces if m == "black" and shown])
    assert screens[:, 1].max() > deck + 0.05 and screens[:, 0].max() < rear  # over the deck, on the cabin's side
    assert {"seat", "chrome"} <= {m for m, *_ in pieces}  # the leather panel and the vent's frame are laid on it
    wheel = _steering(shapes.TRI, 0.19)
    rim = [(v, t) for m, v, t, _ in wheel if m == "dash"]
    assert _volume(rim) > 0 and max(np.hypot(v[:, 0], v[:, 1]).max() for v, _ in rim) < 0.19 * 1.1  # wider than tall


def test_door_cards_line_the_doors_from_inside_and_a_console_stands_between_the_seats():
    from manifoldx.modeling.assets.car import ATLAS

    p = CarParams(door_card=shapes.QUILT, console=shapes.TRAY)
    car = build_car(p)
    left, right = car.part("door_fl"), car.part("door_fr")
    leather, skin = right.pieces["seat"], right.pieces["paint"]
    assert 0.5 < leather.positions[:, 2].max() < skin.positions[:, 2].max()  # on the door, on the cabin's side of its skin
    assert leather.positions[:, 1].max() < p.belt_front and (leather.normals[:, 2] < 0).all()  # under the glass, facing in
    assert np.allclose(left.pieces["seat"].uvs, leather.uvs, atol=1e-6)  # the left door wears the same picture, mirrored
    assert np.allclose(left.pieces["seat"].positions * [1, 1, -1], leather.positions, atol=1e-5)
    s0, t0, s1, t1 = ATLAS["console"]
    rim = car.part("cabin").pieces["chrome"]
    on_console = (rim.uvs[:, 0] >= s0) & (rim.uvs[:, 0] <= s1) & (rim.uvs[:, 1] >= t0) & (rim.uvs[:, 1] <= t1)
    assert on_console.any() and abs(rim.positions[on_console][:, 2]).max() < 0.15  # its rim, between the seats


def test_a_grille_shape_can_carry_chrome():
    def chrome(mat):
        bars = {"aspect": 4.0, "ends": 0.6, "layers": [
            ("black", [(-1, -1), (1, -1), (1, 1), (-1, 1)]),
            (mat, [(-0.9, -0.1), (0.9, -0.1), (0.9, 0.1), (-0.9, 0.1)]),
        ]}
        pieces = build_car(CarParams(grille_style=bars)).part("body").pieces
        return len(pieces["chrome"].positions) if "chrome" in pieces else 0
    assert chrome("chrome") > chrome("black")  # the bar is drawn in chrome, not dark trim


def test_a_bed_is_open_above_its_floor_and_its_tailgate_folds_down_behind():
    p = CarParams(bed_u=0.7, bed_floor=0.85, roof_end_u=0.66, hatch_u=0.68, hatch_height=1.1, tail_top=1.1)
    car = build_car(p)
    x0 = p.length / 2 - p.bed_u * p.length  # where the bed starts
    skin = car.part("body").pieces["paint"].positions
    over = skin[(skin[:, 0] < x0 - 0.1) & (abs(skin[:, 2]) < 0.5)]
    assert (over[:, 1] < 1.0).all()  # nothing spans the bed between the rails
    floor = car.part("body").pieces["liner"].positions
    at = floor[(floor[:, 0] < x0 - 0.1) & np.isclose(floor[:, 1], p.bed_floor)]
    assert np.ptp(at[:, 2]) > 0.8 * p.width  # the bed's floor, in liner, from rail to rail
    gate = car.part("hatch")
    shut, open_ = _posed(gate, car.pose()), _posed(gate, car.pose(hatch=1.0))
    assert open_[:, 0].min() < shut[:, 0].min() - 0.2 and open_[:, 1].max() < shut[:, 1].max() - 0.1
    assert "glass" not in gate.pieces  # the cab's rear window stays on the body



def _hubs(**kw):
    """World positions of the four wheel hubs after place_car, read through each wheel part's transform."""
    from types import SimpleNamespace

    from manifoldx.modeling.assets.car import _rotate, place_car

    car = build_car(CarParams())
    handles = {name: [SimpleNamespace(transform=SimpleNamespace(pos=None, rot=None))] for name in car.pose()}
    place_car(handles, car, at=(5.0, 1.0, -2.0), **kw)
    out = {}
    for part, (end, side) in {"wheel_fl": ("front", "left"), "wheel_fr": ("front", "right"),
                              "wheel_rl": ("rear", "left"), "wheel_rr": ("rear", "right")}.items():
        t = handles[part][0].transform
        out[part] = np.array(t.pos) + _rotate(np.array(t.rot), car.landmarks[f"wheel_{end}_{side}"])
    return out


def test_pitch_raises_the_nose_and_roll_lowers_the_right_side():
    p = CarParams()
    up = _hubs(yaw=0.7, pitch=0.1)
    assert np.isclose(up["wheel_fl"][1] - up["wheel_rl"][1], p.wheelbase * np.sin(0.1), atol=1e-4)
    rolled = _hubs(yaw=0.7, roll=0.1)
    assert np.isclose(rolled["wheel_fl"][1] - rolled["wheel_fr"][1], p.track * np.sin(0.1), atol=1e-4)
    flat = _hubs(yaw=0.7)
    assert np.isclose(flat["wheel_fl"][1], flat["wheel_rr"][1])


# --- step 1 of the vehicle family: the parts that tell the decades apart ---------------------------------------


def _triangles(car):
    return sum(len(mesh.faces) for part in car.parts for mesh in part.pieces.values())


def test_the_default_car_mesh_is_unchanged():
    """The values recorded from origin/main before the new options existed."""
    car = build_car(CarParams())
    assert _triangles(car) == 15760
    body = np.concatenate([_verts(car.part(n)) for n in ("body", "hood", "hatch", "door_fl", "door_fr", "door_rl", "door_rr")])
    assert np.allclose(body.min(0), [-2.130746, 0.2, -1.025525], atol=1e-4)
    assert np.allclose(body.max(0), [2.130708, 1.479951, 1.025525], atol=1e-4)


@pytest.mark.parametrize("style,mat", [("chrome_bar", "chrome"), ("blade", "chrome"), ("rubber", "trim")])
def test_a_bumper_is_a_piece_on_the_body_at_both_ends(style, mat):
    p = CarParams(bumper=style, bumper_height=0.42, bumper_proud=0.06)
    car = build_car(p)
    v = car.part("body").pieces[mat].positions
    front = v[(v[:, 0] > p.length / 2 - 0.25) & (np.abs(v[:, 1] - p.bumper_height) < 0.25)]
    rear = v[(v[:, 0] < -p.length / 2 + 0.25) & (np.abs(v[:, 1] - p.bumper_height) < 0.25)]
    assert len(front) and len(rear)
    # It stands proud of the body but no further, and wraps no wider than the car.
    assert v[:, 0].max() <= p.length / 2 + p.bumper_proud + 0.01
    assert v[:, 0].min() >= -p.length / 2 - p.bumper_proud - 0.01
    assert np.abs(v[:, 2]).max() <= p.width / 2 + p.bumper_proud
    plain = build_car()
    assert len(v) > len(plain.part("body").pieces[mat].positions)


def test_bumper_overriders_add_two_vertical_guards():
    plain = build_car(CarParams(bumper="chrome_bar", bumper_proud=0.06))
    guarded = build_car(CarParams(bumper="chrome_bar", bumper_proud=0.06, overriders=True))
    assert _triangles(guarded) > _triangles(plain)
    v = guarded.part("body").pieces["chrome"].positions
    for sign in (1, -1):
        guards = v[(sign * v[:, 0] > 1.9) & (v[:, 1] > 0.49) & (v[:, 1] < 0.62)]
        assert len(guards)  # the bar itself stops at 0.485; these stand above it


def test_a_mesh_grille_frames_a_dark_panel_and_full_width_spans_the_nose():
    base = build_car()
    narrow = build_car(CarParams(grille_style="mesh"))
    wide = build_car(CarParams(grille_style="mesh", grille_span="full_width"))

    def front_frame(car):
        v = car.part("body").pieces["chrome"].positions
        return v[(v[:, 0] > 1.6) & (v[:, 1] < 0.55)]

    assert not len(front_frame(base))  # no chrome low on the nose until a mesh grille puts its frame there
    assert len(front_frame(narrow)) and len(front_frame(wide))
    assert np.abs(front_frame(wide)[:, 2]).max() > np.abs(front_frame(narrow)[:, 2]).max() + 0.05
    assert {"chrome", "trim"} <= set(narrow.part("body").pieces)  # the frame and its dark panel


def test_a_mesh_grille_is_an_egg_crate_of_chrome_bars():
    def grille_chrome(car):
        mesh = car.part("body").pieces["chrome"]
        mid = mesh.positions[mesh.faces].mean(1)
        return int(((mid[:, 0] > 1.6) & (mid[:, 1] < 0.55)).sum())

    one = build_car(CarParams(grille_style="mesh"))
    low = build_car(CarParams(grille_style="mesh", detail=0.5))
    high = build_car(CarParams(grille_style="mesh", detail=2.0))
    assert grille_chrome(one) > 3688  # the frame and three bars, as recorded before the egg-crate
    assert grille_chrome(low) < grille_chrome(one) < grille_chrome(high)  # the bars follow detail


def test_a_round_pod_lands_on_the_wing_tip_at_both_sizes():
    for size in (0.18, 0.24):
        p = CarParams(headlamp="round_pod", headlamp_center=None, headlamp_size=(size, size), fender_crown=0.12)
        car = build_car(p)
        mesh = car.part("body").pieces["lamp"]
        tri = mesh.positions[mesh.faces]
        tri = tri[tri.mean(1)[:, 0] > p.length / 2 - 0.4]  # the headlamps, not the tail
        one = tri[tri.mean(1)[:, 2] > 0]  # one side of the car
        assert len(one) > 20
        head = np.array(_lamp_centers(p)[0])
        assert np.allclose(one.mean(1)[:, [2, 1]].mean(0), head, atol=0.05)  # it lands where it was derived
        n = np.cross(one[:, 1] - one[:, 0], one[:, 2] - one[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        assert n[:, 0].mean() > 0.5  # the ring looks forward


def test_new_lamp_styles_build_on_the_nose_and_the_tail():
    for style in ("round_pod", "rect"):
        car = build_car(CarParams(headlamp=style))
        v = car.part("body").pieces["lamp"].positions
        front = v[v[:, 0] > 1.6]
        assert len(front) and front[:, 1].max() > 0.6
        if style == "round_pod":
            assert "chrome" in car.part("body").pieces  # the ring
    for style in ("fin", "block"):
        car = build_car(CarParams(taillamp=style, taillamp_center=(0.66, 0.86)))
        v = car.part("body").pieces["tail"].positions
        rear = v[v[:, 0] < -1.6]
        one = rear[rear[:, 2] > 0]  # one lamp, not both sides of the car
        assert len(one)
        if style == "fin":
            assert np.ptp(one[:, 1]) > np.ptp(one[:, 2])  # tall and narrow
        else:
            assert np.ptp(one[:, 2]) > 0.1  # a large rectangle


def test_a_hubcap_and_a_whitewall_dress_the_wheels():
    p = CarParams(wheel="hubcap", whitewall=0.05)
    car = build_car(p)
    wheel = car.part("wheel_fl")
    assert "chrome" in wheel.pieces and "whitewall" in wheel.pieces
    assert "spoke" not in wheel.pieces
    hub = wheel.pivot
    cap = wheel.pieces["chrome"].positions
    assert np.hypot(cap[:, 0] - hub[0], cap[:, 1] - hub[1]).max() <= p.wheel_radius * p.rim_ratio * 1.02
    wall = wheel.pieces["whitewall"].positions
    radius = np.hypot(wall[:, 0] - hub[0], wall[:, 1] - hub[1])
    assert p.wheel_radius * p.rim_ratio < radius.mean() < p.wheel_radius
    assert abs(np.abs(wall[:, 2] - hub[2]).mean() - p.tyre_width / 2) < 0.01


def test_a_painted_roof_is_its_own_material_and_the_pillars_may_take_it():
    from manifoldx.modeling.assets.car import PALETTE

    assert PALETTE["roof"][0] == PALETTE["paint"][0]
    assert "roof" not in build_car().part("body").pieces
    with_pillars = build_car(CarParams(roof_paint=True))
    without = build_car(CarParams(roof_paint=True, roof_pillars=False))
    assert "roof" in with_pillars.part("body").pieces
    assert len(without.part("body").pieces["roof"].positions) < len(with_pillars.part("body").pieces["roof"].positions)
    black = build_car(CarParams(roof="black"))
    assert "roof" not in black.part("body").pieces and "black" in black.part("body").pieces


def test_none_positions_follow_a_tall_body():
    tall = dict(height=1.75, hood_front=1.15)
    low = build_car(CarParams(**tall))
    auto = build_car(CarParams(**tall, headlamp_center=None, taillamp_center=None, grille=None))

    def lamps(car):
        v = car.part("body").pieces["lamp"].positions
        return v[v[:, 0] > 1.7]

    def tails(car):
        v = car.part("body").pieces["tail"].positions
        return v[v[:, 0] < -1.7]

    assert lamps(low)[:, 1].max() < 0.9  # the numeric default stays down at the old nose
    assert lamps(auto)[:, 1].max() > 1.0  # the derived lamp follows hood_front
    assert tails(low)[:, 1].max() > 1.0  # the numeric default rides up the rear glass
    assert tails(auto)[:, 1].max() < 0.9  # the derived lamp sits on the tail
    assert "trim" in auto.part("body").pieces


def test_a_decal_that_misses_its_face_raises_naming_the_part():
    with pytest.raises(ValueError, match="headlamp"):
        build_car(CarParams(headlamp="round", headlamp_size=(0.06, 0.06), headlamp_center=(0.90, 0.40)))
    with pytest.raises(ValueError, match="taillamp"):
        build_car(CarParams(taillamp_center=(0.0, 3.0)))


def test_detail_scales_the_resolution_but_one_is_today():
    low, one, high = build_car(CarParams(detail=0.5)), build_car(), build_car(CarParams(detail=2.0))
    assert _triangles(low) < _triangles(one) < _triangles(high)
    assert _triangles(one) == 15760
    build_car(CarParams(sections=10, detail=0.5))  # never fewer than 8 sections


# --- step 2 of the vehicle family: wings, fins, a wrapping screen and side trim -------------------------------


def _top(car, x, below=False, zmin=None, centre=False):
    """The highest skin point in a stretch of the car, on one flank or on the centre line."""
    v = car.part("body").pieces["paint"].positions
    select = (v[:, 0] < x) if below else (v[:, 0] > x)
    select &= (np.abs(v[:, 2]) < 0.06) if centre else (v[:, 2] > (zmin if zmin is not None else 0.0))
    return v[select, 1].max()


def test_the_new_options_default_to_todays_car():
    p = CarParams()
    assert (p.fender_crown, p.fender_crown_u, p.fin_height, p.fin_start_u, p.fin_width) == (0.0, 0.25, 0.0, 0.7, 0.22)
    assert (p.windscreen_wrap, p.side_trim) == (0.0, ())
    assert _triangles(build_car(CarParams())) == 15760


def test_fender_crown_raises_the_wings_at_the_nose_but_not_the_centre():
    def nose(p):
        _, P, _, _ = _body_grid(p, (p.front_overhang, p.front_overhang - p.wheelbase))
        return P[0]

    base, crowned = nose(CarParams()), nose(CarParams(fender_crown=0.10, fender_crown_u=0.25))

    def wing(s):
        return s[:, 1].max()

    def centre(s):
        return s[np.abs(s[:, 2]) < 1e-9, 1].max()

    assert 0.07 < wing(crowned) - wing(base) < 0.12  # about the crown
    assert np.isclose(centre(crowned), centre(base))  # the hood's centre keeps its line


def test_fender_crown_makes_two_ridges_over_a_low_hood():
    p = CarParams(fender_crown=0.12)
    _, P, _, _ = _body_grid(p, (p.front_overhang, p.front_overhang - p.wheelbase))
    nose = P[0]  # the section at the very front of the car
    centre = nose[np.abs(nose[:, 2]) < 1e-9, 1].max()
    wing = nose[:, 1].max()
    assert wing - centre >= 0.8 * p.fender_crown
    assert nose[7:11, 1].min() < centre - 0.02  # the hood dips between the crests, a step not a slope


def test_fin_height_raises_the_tail_wings_but_not_the_centre():
    base = build_car(CarParams())
    finned = build_car(CarParams(fin_height=0.15, fin_start_u=0.7, fin_width=0.22))
    lift = _top(finned, -1.9, below=True, zmin=0.6) - _top(base, -1.9, below=True, zmin=0.6)
    assert lift > 0.04
    assert np.isclose(_top(finned, -1.7, below=True, centre=True), _top(base, -1.7, below=True, centre=True), atol=1e-6)


def test_derived_pod_and_fin_lamps_follow_the_wings():
    p = CarParams(headlamp="round_pod", headlamp_center=None, headlamp_size=(0.16, 0.16), hood_crown=0.03,
                  fender_crown=0.10, fender_crown_u=0.25,
                  taillamp="fin", taillamp_center=None, fin_height=0.15, fin_start_u=0.7)
    car = build_car(p)  # both derived lamps land: no ValueError
    plain = build_car(CarParams(headlamp="round_pod", headlamp_center=None, headlamp_size=(0.16, 0.16),
                                hood_crown=0.03, taillamp="fin", taillamp_center=None))
    lamps = car.part("body").pieces["lamp"].positions
    lamps = lamps[lamps[:, 0] > 1.6]
    one = lamps[lamps[:, 2] > 0]
    assert len(one) and np.allclose(one[:, [2, 1]].mean(0), _lamp_centers(p)[0], atol=0.05)
    # The pod sits under the raised wing, whose crest now stands over it.
    paint = car.part("body").pieces["paint"].positions
    crest = paint[(paint[:, 0] > p.length / 2 - 0.25) & (np.abs(paint[:, 2]) > 0.55), 1].max()
    assert crest > lamps[:, 1].max() + 0.03
    tails = car.part("body").pieces["tail"].positions
    tails = tails[tails[:, 0] < -1.6]
    plain_tails = plain.part("body").pieces["tail"].positions
    plain_tails = plain_tails[plain_tails[:, 0] < -1.6]
    assert len(tails) and tails[:, 1].max() > plain_tails[:, 1].max() + 0.03


def test_windscreen_wrap_adds_glass_round_the_a_pillars():
    plain = build_car(CarParams())
    wrapped = build_car(CarParams(windscreen_wrap=0.6))
    def glass(car):
        return sum(len(mesh.faces) for part in car.parts for mat, mesh in part.pieces.items() if mat == "glass")
    assert glass(wrapped) > glass(plain)


def test_side_trim_lays_a_chrome_strip_along_both_flanks():
    plain = build_car(CarParams())
    p = CarParams(side_trim=((0.78, 0.5, 0.9, "chrome"),))
    car = build_car(p)
    trim = car.part("body").pieces["chrome"].positions
    strip = trim[(trim[:, 1] > 0.76) & (trim[:, 1] < 0.80)]
    assert len(strip) and strip[:, 2].min() < -0.5 and strip[:, 2].max() > 0.5  # both sides
    assert len(trim) > len(plain.part("body").pieces["chrome"].positions)
