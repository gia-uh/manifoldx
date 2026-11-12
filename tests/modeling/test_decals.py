"""Tests for decals laid over a flat base paint."""

from functools import lru_cache

import numpy as np

from manifoldx.modeling.assets.car import CarParams, build_car
from manifoldx.modeling.bake import (
    Decal,
    _VIEW_AXES,
    _VIEW_DIRECTIONS,
    _material_names,
    _rasterise,
    apply_decals,
    bake_projection,
)

SIZE = 256
BASE = np.array([90, 110, 130], np.uint8)
MARK = np.array([220, 40, 40], np.uint8)
DOOR = np.array([0.0, 0.70, -0.885])


def _square(rgb=MARK, alpha=255, pixels=32):
    """An opaque or translucent square decal."""
    image = np.zeros((pixels, pixels, 4), np.uint8)
    image[..., :3] = rgb
    image[..., 3] = alpha
    return image


@lru_cache(maxsize=1)
def _default():
    """One default car, its raster and its flat base paint, shared by the tests."""
    car = build_car(CarParams())
    names = _material_names(car)
    pos, nrm, mat = _rasterise(car, SIZE, names)
    base = bake_projection(car, {}, size=SIZE, materials={"paint"}, fallback={"paint": BASE})
    return car, pos, nrm, mat, names.index("paint"), names.index("glass"), base


def _view_plane(pos, view, centre):
    """A grid of 3D points as metres across and down that view's image plane, from a centre."""
    across, down = _VIEW_AXES[view]
    relative = pos - np.asarray(centre, float)
    return relative @ across, relative @ down


def _spacing(values):
    """The texel spacing of one view-plane coordinate, in metres."""
    unique = np.unique(np.round(values, 5))
    return float(np.median(np.diff(unique))) if unique.size > 1 else 0.0


def test_the_modeling_package_reexports_the_decal_api():
    from manifoldx.modeling import Decal as PublicDecal
    from manifoldx.modeling import apply_decals as public_apply_decals

    assert PublicDecal is Decal
    assert public_apply_decals is apply_decals


def test_empty_views_lay_the_flat_base_paint():
    car, _, _, mat, paint_id, glass_id, _ = _default()
    fallback = {"paint": (31, 47, 59)}
    atlas = bake_projection(car, {}, size=SIZE, materials={"paint"}, fallback=fallback)

    paint = mat == paint_id
    glass = mat == glass_id
    assert paint.sum() > 100
    assert glass.sum() > 100
    assert np.allclose(atlas[paint].astype(float), fallback["paint"], atol=0)
    # Materials the caller did not bake keep their palette colours.
    assert not np.allclose(atlas[glass].astype(float), fallback["paint"], atol=1)


def test_a_square_decal_lands_centred_and_sized_on_the_left_door():
    car, pos, nrm, mat, paint_id, _, base = _default()
    width, height = 0.5, 0.35
    decal = Decal(_square(), "left", DOOR, (width, height))
    atlas = apply_decals(base, car, [decal])

    changed = np.any(atlas != base, axis=2)
    allowed = (mat == paint_id) & (nrm @ _VIEW_DIRECTIONS["left"] >= 0.3)
    across, down = _view_plane(pos, "left", DOOR)
    expected = allowed & (np.abs(across) <= width / 2) & (np.abs(down) <= height / 2)

    assert expected.sum() > 200
    assert changed.sum() > 0
    assert not (changed & ~expected).any(), "the decal painted outside its rectangle"
    assert changed.sum() >= 0.95 * expected.sum(), "the decal left part of its rectangle unpainted"

    spacing_x = _spacing(across[expected])
    spacing_y = _spacing(down[expected])
    assert abs(float(across[changed].mean())) <= 2 * spacing_x
    assert abs(float(down[changed].mean())) <= 2 * spacing_y
    assert abs(float(np.ptp(across[changed])) - width) <= 2 * spacing_x
    assert abs(float(np.ptp(down[changed])) - height) <= 2 * spacing_y
    assert np.allclose(atlas[changed].astype(float), MARK, atol=1)


def test_a_decal_does_not_paint_glass_or_texels_facing_away():
    car, pos, nrm, mat, _, glass_id, base = _default()
    decal = Decal(_square(), "left", (0.0, 0.70, -0.9), (4.5, 1.6))
    atlas = apply_decals(base, car, [decal])

    changed = np.any(atlas != base, axis=2)
    facing = nrm @ _VIEW_DIRECTIONS["left"]
    across, down = _view_plane(pos, "left", (0.0, 0.70, -0.9))
    inside = (np.abs(across) <= 2.25) & (np.abs(down) <= 0.8)

    assert changed.sum() > 500
    assert (inside & (mat == glass_id)).sum() > 0, "the rectangle should cover side glass"
    assert not changed[mat == glass_id].any()
    assert not changed[facing < 0.3].any()


def test_alpha_zero_leaves_the_atlas_untouched_and_half_alpha_blends_straight():
    car, _, _, _, _, _, base = _default()

    clear = apply_decals(base, car, [Decal(_square(alpha=0), "left", DOOR, (0.5, 0.35))])
    assert np.array_equal(clear, base)

    half = apply_decals(base, car, [Decal(_square(alpha=128), "left", DOOR, (0.5, 0.35))])
    changed = np.any(half != base, axis=2)
    assert changed.sum() > 200
    blended = (128.0 / 255.0) * MARK + (1.0 - 128.0 / 255.0) * BASE
    assert np.allclose(half[changed].astype(float), blended, atol=1)


def test_a_decal_placed_by_landmark_follows_the_cars_measurements():
    car_a, pos_a, nrm_a, mat_a, _, _, base_a = _default()
    paint_a = mat_a == _material_names(car_a).index("paint")

    car_b = build_car(CarParams(length=4.8, wheelbase=2.9))
    names_b = _material_names(car_b)
    pos_b, nrm_b, mat_b = _rasterise(car_b, SIZE, names_b)
    paint_b = mat_b == names_b.index("paint")
    base_b = bake_projection(car_b, {}, size=SIZE, materials={"paint"}, fallback={"paint": BASE})

    moved = car_b.landmarks["mirror_left"] - car_a.landmarks["mirror_left"]
    assert abs(float(moved[0])) > 0.05

    offset = np.array([0.0, -0.12, 0.0])
    decal = Decal(_square(), "left", "mirror_left", (0.25, 0.2), offset=offset)
    for car, pos, nrm, paint, base in (
        (car_a, pos_a, nrm_a, paint_a, base_a),
        (car_b, pos_b, nrm_b, paint_b, base_b),
    ):
        atlas = apply_decals(base, car, [decal])
        changed = np.any(atlas != base, axis=2)
        centre = np.asarray(car.landmarks["mirror_left"], float) + offset
        across, down = _view_plane(pos, "left", centre)
        allowed = paint & (nrm @ _VIEW_DIRECTIONS["left"] >= 0.3)
        expected = allowed & (np.abs(across) <= 0.125) & (np.abs(down) <= 0.1)

        assert expected.sum() > 40
        assert changed.sum() > 0
        assert not (changed & ~expected).any()
        assert abs(float(across[changed].mean())) <= 2 * _spacing(across[expected])
        assert abs(float(down[changed].mean())) <= 2 * _spacing(down[expected])
