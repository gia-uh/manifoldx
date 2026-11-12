"""Tests for baking an atlas by projecting orthographic views."""

from functools import lru_cache

import numpy as np

from manifoldx.modeling.assets.car import CarParams, build_car
from manifoldx.modeling.bake import (
    _fill_gutters,
    _material_names,
    _prepare_image,
    _rasterise,
    _view_box,
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


def test_the_top_view_has_the_assets_left_side_up():
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
    assert np.allclose(_mean(atlas, left_side), RED, atol=12)
    assert np.allclose(_mean(atlas, right_side), BLUE, atol=12)
