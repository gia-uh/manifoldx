"""Tests for baking an atlas by projecting orthographic views."""

from functools import lru_cache

import numpy as np
import pytest

from manifoldx.modeling.assets.car import CarParams, build_car
from manifoldx.modeling.bake import (
    _bounds,
    _fill_gutters,
    _keypoint_warp,
    _material_names,
    _prepare_image,
    _rasterise,
    _view_box,
    _view_uv,
    bake_projection,
)

SIZE = 256
RED = np.array([220.0, 40.0, 40.0], np.float32)
BLUE = np.array([40.0, 60.0, 220.0], np.float32)
ORANGE = np.array([230.0, 120.0, 20.0], np.float32)


def _split_view(size=SIZE):
    """A left view with the front half red and the rear half blue on a white field."""
    image = np.full((size, size, 3), 255, np.uint8)
    image[24 : size - 24, 20 : size // 2] = RED
    image[24 : size - 24, size // 2 : size - 20] = BLUE
    return image


def _flat_view(colour, size=SIZE):
    image = np.full((size, size, 3), 255, np.uint8)
    image[24 : size - 24, 20 : size - 20] = colour
    return image


@lru_cache(maxsize=1)
def _default_car():
    """One default car and its raster, shared by the tests that only read them."""
    car = build_car(CarParams())
    names = _material_names(car)
    pos, nrm, mat = _rasterise(car, SIZE, names)
    return car, pos, nrm, mat, names.index("paint")


def _mean(atlas, selection):
    return atlas[selection].astype(np.float32).mean(axis=0)


def test_left_view_and_its_mirror_land_front_and_rear_on_their_sides():
    car, pos, nrm, mat, paint_id = _default_car()
    atlas = bake_projection(car, {"left": _split_view()}, size=SIZE, materials={"paint"})
    paint = mat == paint_id
    front, rear = pos[..., 0] > 0.6, pos[..., 0] < -0.6
    left, right = nrm[..., 2] < -0.7, nrm[..., 2] > 0.7

    left_front = paint & front & left
    right_front = paint & front & right
    left_rear = paint & rear & left
    right_rear = paint & rear & right
    for selection in (left_front, right_front, left_rear, right_rear):
        assert selection.sum() > 100

    # The left drawing has the front at its left, so the front is red there.
    assert np.allclose(_mean(atlas, left_front), RED, atol=12)
    assert np.allclose(_mean(atlas, left_rear), BLUE, atol=12)
    # The right drawing is the left one mirrored, and its front is at its right.
    assert np.allclose(_mean(atlas, right_front), RED, atol=12)
    assert np.allclose(_mean(atlas, right_rear), BLUE, atol=12)


def test_texels_no_view_faces_keep_the_fallback():
    car, pos, nrm, mat, paint_id = _default_car()
    fallback = np.array([17, 19, 23], np.float32)
    atlas = bake_projection(
        car, {"left": _split_view()}, size=SIZE, materials={"paint"}, fallback=fallback
    )
    paint = mat == paint_id
    roof = paint & (nrm[..., 1] > 0.95)
    assert roof.sum() > 100
    # The left view is edge-on to the roof, so no view faces those texels.
    assert np.allclose(atlas[roof].astype(np.float32), fallback, atol=1)

    front = paint & (pos[..., 0] > 0.6) & (nrm[..., 2] < -0.7)
    assert front.sum() > 100
    assert not np.allclose(atlas[front].astype(np.float32), fallback, atol=1)


def test_gutters_are_filled_from_the_nearest_covered_texel():
    car, _, _, mat, _ = _default_car()
    atlas = bake_projection(car, {"left": _flat_view(ORANGE)}, size=SIZE, materials={"paint"})

    checked = 0
    for row in (40, 50, 60):
        covered = np.nonzero(mat[row, : int(0.71 * SIZE)] >= 0)[0]
        assert len(covered), "the left island should cover this row"
        x = int(covered.min())
        assert x > 4
        # A few pixels outside the island are still background in the raster;
        # the gutter fill must have taken the island's colour there.
        assert np.allclose(
            atlas[row, x - 4].astype(np.float32), atlas[row, x].astype(np.float32), atol=1
        )
        assert not np.allclose(atlas[row, x - 4], 255, atol=1)
        checked += 1
    assert checked == 3


def test_a_light_subject_on_a_white_background_is_foreground():
    # A white car on a white field must not be mistaken for background just
    # because its paint is light grey: the field is the colour to subtract.
    image = np.full((64, 64, 3), 255, np.uint8)
    image[16:48, 16:48] = (228, 228, 231)
    _, mask = _prepare_image(image)
    assert mask[32, 32]
    assert not mask[0, 0]


def test_the_view_box_ignores_a_stray_speck():
    image = np.full((120, 200, 3), 255, np.uint8)
    image[20:100, 30:170] = (30, 40, 50)
    image[5, 7] = (0, 0, 0)  # one stray dark pixel, far from the subject
    assert _view_box(_prepare_image(image)) == (30, 169, 20, 99)


def test_gutters_reach_the_far_corner_of_the_atlas():
    colour = np.full((80, 120, 3), 255.0, np.float32)
    colour[20, 30] = ORANGE
    cover = np.zeros((80, 120), bool)
    cover[20, 30] = True
    out = _fill_gutters(colour, cover)
    assert np.allclose(out, ORANGE)


def test_the_top_view_has_the_assets_right_side_up():
    car, pos, nrm, mat, paint_id = _default_car()
    image = np.full((SIZE, SIZE, 3), 255, np.uint8)
    image[24 : SIZE // 2, 20 : SIZE - 20] = RED
    image[SIZE // 2 : SIZE - 24, 20 : SIZE - 20] = BLUE
    atlas = bake_projection(car, {"top": image}, size=SIZE, materials={"paint"})
    paint = mat == paint_id
    facing_up = paint & (nrm[..., 1] > 0.7)
    left_side = facing_up & (pos[..., 2] < -0.3)
    right_side = facing_up & (pos[..., 2] > 0.3)
    assert left_side.sum() > 100
    assert right_side.sum() > 100
    assert np.allclose(_mean(atlas, right_side), RED, atol=12)
    assert np.allclose(_mean(atlas, left_side), BLUE, atol=12)


# --- registration by keypoints -------------------------------------------------------------------------------

BODY = np.array([40, 60, 220], np.uint8)
PLACED = np.array([240, 200, 20], np.uint8)
MARGIN, SCALE = 16.0, 224.0

# The landmarks a plan view shows; the wheels are distinct here, unlike in a side view.
TOP_LANDMARKS = (
    "nose", "tail", "roof", "mirror_left", "mirror_right",
    "wheel_front_left", "wheel_front_right", "wheel_rear_left", "wheel_rear_right",
)


def _drawn(car, names, view="top"):
    """Where each landmark is drawn in our synthetic view: a fixed scale and offset over its own projection."""
    bounds = _bounds(car)
    uv = {name: _view_uv(view, np.asarray(car.landmarks[name], float), bounds) for name in names}
    return uv, {name: (MARGIN + SCALE * float(q[0]), MARGIN + SCALE * float(q[1])) for name, q in uv.items()}


def test_landmarks_name_the_wheels_glass_and_mirrors():
    car = build_car(CarParams())
    p = car.params
    marks = car.landmarks
    for name, part, x in (
        ("wheel_front_left", "wheel_fl", p.length / 2 - p.front_overhang),
        ("wheel_front_right", "wheel_fr", p.length / 2 - p.front_overhang),
        ("wheel_rear_left", "wheel_rl", p.length / 2 - p.front_overhang - p.wheelbase),
        ("wheel_rear_right", "wheel_rr", p.length / 2 - p.front_overhang - p.wheelbase),
    ):
        assert np.allclose(marks[name], car.part(part).pivot, atol=1e-3)
        assert abs(marks[name][0] - x) < 1e-6
    for name in ("nose", "tail", "windscreen_foot", "windscreen_top", "rear_window_top",
                 "rear_window_foot", "roof", "mirror_left", "mirror_right"):
        assert marks[name].shape == (3,)
    assert marks["roof"][1] >= max(v[1] for v in marks.values())
    assert marks["windscreen_top"][1] > marks["windscreen_foot"][1]
    assert marks["rear_window_top"][1] > marks["rear_window_foot"][1]
    assert marks["nose"][0] > 0 > marks["tail"][0]
    assert marks["mirror_left"][2] < 0 < marks["mirror_right"][2]
    assert "mirror_left" not in build_car(CarParams(mirrors=False)).landmarks


def test_keypoints_pull_a_part_drawn_out_of_place_onto_its_texels():
    car, pos, nrm, mat, paint_id = _default_car()
    bounds = _bounds(car)
    _, marks = _drawn(car, TOP_LANDMARKS)
    # The roof's apex is drawn 8% of the image width toward the rear, as a drawing drifting from the asset
    # would place it.  A mark at that spot is the only colour in the drawing near the roof.
    roof_x, roof_y = marks["roof"]
    marks["roof"] = (roof_x + 0.08 * SIZE, roof_y)
    drawing = np.full((SIZE, SIZE, 3), 255, np.uint8)
    drawing[10:200, 10:246] = BODY
    x, y = int(round(marks["roof"][0])), int(round(marks["roof"][1]))
    drawing[y - 12 : y + 12, x - 12 : x + 12] = PLACED

    u, v = _view_uv("top", pos, bounds)
    roof_u, roof_v = _view_uv("top", np.asarray(car.landmarks["roof"], float), bounds)
    near = (np.hypot(u - roof_u, v - roof_v) < 0.03) & (mat == paint_id) & (nrm[..., 1] > 0.6)
    assert near.sum() > 10

    with_keypoints = bake_projection(
        car, {"top": drawing}, size=SIZE, materials={"paint"}, keypoints={"top": marks}
    )
    without = bake_projection(car, {"top": drawing}, size=SIZE, materials={"paint"})
    assert np.allclose(with_keypoints[near].astype(float), PLACED, atol=8)
    assert np.allclose(without[near].astype(float), BODY, atol=8)


def test_three_and_five_keypoints_fit_a_pure_scale_and_offset_exactly():
    car = build_car(CarParams())
    bounds = _bounds(car)
    uv, marks = _drawn(car, TOP_LANDMARKS)
    for chosen in (
        ("nose", "tail", "wheel_front_left"),
        ("nose", "tail", "wheel_front_left", "wheel_rear_right", "mirror_left"),
    ):
        warp = _keypoint_warp("top", {name: marks[name] for name in chosen}, car.landmarks, bounds)
        for name in TOP_LANDMARKS:
            x, y = warp(np.float32(uv[name][0]), np.float32(uv[name][1]))
            assert abs(float(x) - marks[name][0]) < 0.05
            assert abs(float(y) - marks[name][1]) < 0.05


def test_six_keypoints_bend_the_drawing_where_the_keypoint_moved():
    car = build_car(CarParams())
    bounds = _bounds(car)
    uv, marks = _drawn(car, TOP_LANDMARKS)
    marks = dict(marks)
    marks["roof"] = (marks["roof"][0] + 0.08 * SIZE, marks["roof"][1])
    warp = _keypoint_warp("top", marks, car.landmarks, bounds)

    x, y = warp(np.float32(uv["roof"][0]), np.float32(uv["roof"][1]))
    assert np.hypot(float(x) - marks["roof"][0], float(y) - marks["roof"][1]) < 0.5
    # An affine fit would move the other landmarks by pixels; a thin-plate spline leaves them where they are.
    for name in ("nose", "tail", "wheel_front_left", "wheel_rear_right", "mirror_right"):
        x, y = warp(np.float32(uv[name][0]), np.float32(uv[name][1]))
        assert np.hypot(float(x) - marks[name][0], float(y) - marks[name][1]) < 0.5


def test_a_view_without_keypoints_keeps_the_box_registration():
    car, _, nrm, mat, paint_id = _default_car()
    left = np.full((SIZE, SIZE, 3), 255, np.uint8)
    stripe = np.arange(SIZE) // 16 % 2
    left[24 : SIZE - 24, 20 : SIZE - 20] = np.stack([RED, BLUE])[stripe[20 : SIZE - 20]][None, :]
    top = _flat_view(BLUE)
    top[100:160, 60:120] = ORANGE
    _, marks = _drawn(car, TOP_LANDMARKS)
    marks = {name: (x + 0.08 * SIZE, y) for name, (x, y) in marks.items()}
    fallback = (17, 19, 23)

    box = bake_projection(car, {"left": left, "top": top}, size=SIZE, materials={"paint"}, fallback=fallback)
    moved = bake_projection(
        car, {"left": left, "top": top}, size=SIZE, materials={"paint"}, fallback=fallback,
        keypoints={"top": marks},
    )
    changed = np.any(box != moved, axis=2)
    assert changed.any(), "the top view should have moved"
    side = (mat == paint_id) & (nrm[..., 2] < -0.7) & (nrm[..., 1] <= 0.0)
    assert side.sum() > 100
    assert not changed[side].any(), "keypoints on the top view moved the left view too"


def test_keypoints_must_name_a_view_that_was_supplied():
    # The right view made by mirroring the left one keeps the box registration: it has no drawing of its
    # own, so it cannot carry keypoints of its own either.
    car = build_car(CarParams())
    _, marks = _drawn(car, ("nose", "tail", "roof"), view="left")
    with pytest.raises(ValueError, match="has no image"):
        bake_projection(car, {"left": _flat_view(ORANGE)}, size=SIZE, keypoints={"right": marks})


def test_keypoints_are_checked_before_a_view_is_baked():
    car = build_car(CarParams())
    left = _flat_view(ORANGE)
    three = {"nose": (20.0, 20.0), "tail": (200.0, 30.0), "roof": (110.0, 40.0)}
    for images, keypoints, message in (
        ({"left": left}, {"side": three}, "unknown view"),
        ({"left": left}, {"rear": three}, "has no image"),
        ({"left": left}, {"left": {}}, "non-empty"),
        ({"left": left}, {"left": {**three, "bonnet": (10.0, 10.0)}}, "unknown landmark"),
        ({"left": left}, {"left": {**three, "nose": (10.0,)}}, "pair"),
        ({"left": left}, {"left": {**three, "nose": (float("nan"), 10.0)}}, "pair"),
        ({"left": left}, {"left": {"nose": (10.0, 10.0), "tail": (20.0, 20.0)}}, "at least 3"),
        # The left view shows both front wheels at the same place, so these three source points are degenerate.
        ({"left": left}, {"left": {"wheel_front_left": (10.0, 10.0), "wheel_front_right": (20.0, 20.0),
                                   "mirror_left": (30.0, 30.0)}}, "degenerate"),
    ):
        with pytest.raises(ValueError, match=message):
            bake_projection(car, images, size=32, materials={"paint"}, keypoints=keypoints)


def test_an_empty_keypoints_mapping_bakes_as_if_it_were_absent():
    car = build_car(CarParams())
    drawing = _split_view()
    absent = bake_projection(car, {"left": drawing}, size=SIZE, materials={"paint"})
    empty = bake_projection(car, {"left": drawing}, size=SIZE, materials={"paint"}, keypoints={})
    assert np.array_equal(absent, empty)
