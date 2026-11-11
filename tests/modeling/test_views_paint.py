"""Tests for solids and atlases from orthographic views (modeling.views) and for
painting a model's answers back onto an atlas (modeling.paint)."""

import numpy as np
import pytest

from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.paint import (
    bleed,
    clean,
    flip_normal_x,
    kept_outline,
    normal_from_albedo,
    paste_island,
)
from manifoldx.modeling.views import (
    Atlas,
    Island,
    extrude,
    silhouette,
    template,
    two_view_solid,
    view_atlas,
)


def _disc_mask(size, centre, radius):
    height, width = size
    yy, xx = np.indices((height, width))
    return (xx + 0.5 - centre[0]) ** 2 + (yy + 0.5 - centre[1]) ** 2 <= radius * radius


def _volume(mesh):
    p = mesh.positions.astype(float)
    f = mesh.faces.astype(np.intp)
    v0, v1, v2 = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


# --- Task 4: solids from views ---------------------------------------------------------------


def test_extrude_is_a_closed_prism_with_outward_faces():
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    mesh = extrude(square, depth=2.0, centre=0.5)
    assert mesh.positions.dtype == np.float32 and mesh.faces.dtype == np.uint32
    assert abs(mesh.positions[:, 2].min() + 0.5) < 1e-6
    assert abs(mesh.positions[:, 2].max() - 1.5) < 1e-6
    assert len(mesh.faces) == 12  # two caps of two triangles plus four walls of two
    assert _volume(mesh) == pytest.approx(8.0)  # area 4 times depth 2, wound outward


def test_extrude_clips_a_concave_outline():
    l_shape = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    mesh = extrude(l_shape, depth=1.0)
    assert _volume(mesh) == pytest.approx(3.0)
    assert len(mesh.faces) == 2 * 4 + 6 * 2  # 4 cap triangles each, 6 wall quads


def test_two_view_solid_of_two_discs_is_a_sphere():
    front = _disc_mask((201, 201), (100.5, 100.5), 100.0)
    side = front.copy()
    mesh = two_view_solid(front, side, height_range=(-1.0, 1.0),
                          x_range=(-1.0, 1.0), z_range=(-1.0, 1.0), n=64, segments=32)
    p = mesh.positions.astype(float)
    radius = np.linalg.norm(p, axis=1)
    equator = np.abs(p[:, 1]) < 0.1
    assert np.all(np.abs(radius[equator] - 1.0) < 0.03)
    assert radius.max() < 1.03
    assert _volume(mesh) == pytest.approx(4.0 / 3.0 * np.pi, rel=0.03)


def test_two_view_solid_uses_the_whole_span_of_a_mask_with_holes():
    front = np.zeros((41, 41), dtype=bool)
    front[10:31, 5:36] = True
    front[17:24, 15:26] = False  # a part drawn on top leaves a hole in the silhouette
    side = np.zeros((41, 41), dtype=bool)
    side[10:31, 10:31] = True
    mesh = two_view_solid(front, side, height_range=(-1.0, 1.0),
                          x_range=(-2.0, 2.0), z_range=(-1.0, 1.0), n=16)
    assert mesh.positions[:, 0].max() > 1.3  # the hole does not pinch the section
    assert mesh.positions[:, 0].min() < -1.3


def test_two_view_solid_needs_a_height_where_both_masks_are_drawn():
    empty = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        two_view_solid(empty, empty, height_range=(-1.0, 1.0))


def test_silhouette_covers_the_projected_cube():
    mask = silhouette(Mesh.box(1.0, 1.0, 1.0), "front", size=(64, 64),
                      frame=(-0.6, 0.6, -0.6, 0.6))
    assert mask.dtype == bool and mask.shape == (64, 64)
    assert mask[32, 32]
    assert not mask[0, 0] and not mask[-1, -1]
    assert mask.mean() == pytest.approx((1.0 / 1.2) ** 2, abs=0.02)


def test_view_atlas_sends_a_cubes_six_faces_to_six_islands():
    cube = Mesh.box(1.0, 1.0, 1.0)
    atlas = view_atlas({"cube": cube}, ["front", "back", "left", "right", "top", "bottom"],
                       size=(512, 512))
    assert isinstance(atlas, Atlas)
    assert len(atlas.islands) == 6
    assert all(isinstance(island, Island) for island in atlas.islands)
    island_of = atlas.island_of["cube"]
    assert len(set(island_of.tolist())) == 6
    for index in range(6):
        assert int(np.sum(island_of == index)) == 2  # two triangles per cube face

    split = atlas.meshes["cube"]
    uvs = atlas.uvs["cube"]
    assert split.uvs is not None and np.array_equal(split.uvs, uvs)
    assert uvs.shape == (len(split.positions), 2) and uvs.dtype == np.float32
    assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0)
    # every face's UVs stay inside the rectangle of the island it was sent to
    for face, index in zip(split.faces, atlas.island_of["cube"]):
        island = atlas.islands[index]
        x0, y0, x1, y1 = island.rect
        px = uvs[face, 0] * 512
        py = uvs[face, 1] * 512
        assert np.all(px >= x0 - 1e-3) and np.all(px <= x1 + 1e-3)
        assert np.all(py >= y0 - 1e-3) and np.all(py <= y1 + 1e-3)


def test_template_paints_the_nearest_part():
    parts = {
        "red": Mesh.box(1.0, 1.0, 1.0).translate((0.0, 0.0, 0.25)),
        "blue": Mesh.box(2.0, 2.0, 1.0).translate((0.0, 0.0, -0.25)),
    }
    colors = {"red": (255, 0, 0), "blue": (0, 0, 255)}
    atlas = view_atlas(parts, ["front"], size=(128, 128))
    img, regions = template(parts, atlas, colors, "front")
    assert img.dtype == np.uint8 and img.shape == (128, 128, 3)
    assert regions.dtype.kind == "i" and regions.shape == (128, 128)

    x0, y0, x1, y1 = atlas.islands[0].rect
    cy = (y0 + y1) // 2
    assert tuple(img[cy, (x0 + x1) // 2]) == (255, 0, 0)  # red is nearer the front view
    assert regions[cy, (x0 + x1) // 2] == 1
    assert tuple(img[cy, x0 + (x1 - x0) // 10]) == (0, 0, 255)  # the wider blue cube shows
    assert regions[cy, x0 + (x1 - x0) // 10] == 2
    assert tuple(img[0, 0]) == (255, 255, 255) and regions[0, 0] == 0


# --- Task 5: painting the atlas --------------------------------------------------------------


def test_paste_island_resizes_the_answer_and_crops_the_padding():
    atlas_img = np.full((40, 40, 3), 255, dtype=np.uint8)
    picture = np.full((30, 30, 3), 200, dtype=np.uint8)
    paste_island(atlas_img, (10, 12, 20, 22), picture, (5, 3))
    assert np.all(atlas_img[12:22, 10:20] == 200)
    assert np.all(atlas_img[0, 0] == 255) and np.all(atlas_img[22, 10] == 255)


def test_kept_outline_is_one_for_an_image_with_itself():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    img[5:15, 6:12] = (10, 20, 30)
    assert kept_outline(img, img) == 1.0
    other = np.full_like(img, 255)
    assert kept_outline(img, other) == 0.0


def test_clean_fills_white_holes_inside_a_region():
    picture = np.full((10, 10, 3), 100, dtype=np.uint8)
    regions = np.zeros((10, 10), dtype=np.int32)
    regions[:, :5] = 1
    picture[2:4, 2:4] = 255  # the model left the background white inside the part
    out = clean(picture, regions, {"steel": (40, 60, 80)})
    assert np.all(out[2:4, 2:4] == (40, 60, 80))  # refilled with its part's colour
    assert np.all(out[0, 7] == 100)  # outside every region the picture is untouched


def test_clean_scales_the_island_to_a_region_tone():
    picture = np.full((10, 10, 3), 100, dtype=np.uint8)
    regions = np.zeros((10, 10), dtype=np.int32)
    regions[:, :5] = 1
    out = clean(picture, regions, {"steel": (40, 60, 80)}, tone=(1, (150, 150, 150)))
    assert abs(int(np.median(out[:, :5])) - 150) <= 2
    assert abs(int(np.median(out[:, 5:])) - 150) <= 2  # the whole island is scaled


def test_bleed_grows_a_regions_colour_outwards():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    img[mask] = (10, 20, 30)
    out = bleed(img, mask, steps=3)
    assert tuple(out[12, 8]) == (10, 20, 30)
    assert tuple(out[13, 8]) == (10, 20, 30)
    assert tuple(out[0, 0]) == (255, 255, 255)


def test_normal_from_albedo_of_a_flat_image_is_flat():
    albedo = np.full((32, 32, 3), 180, dtype=np.uint8)
    normal = normal_from_albedo(albedo)
    assert normal.dtype == np.uint8 and normal.shape == (32, 32, 3)
    assert np.all(normal[..., 0] == 128)
    assert np.all(normal[..., 1] == 128)
    assert np.all(normal[..., 2] == 255)


def test_normal_from_albedo_turns_a_dark_line_into_a_groove():
    albedo = np.full((64, 64, 3), 255, dtype=np.uint8)
    albedo[:, 30:34] = 40
    normal = normal_from_albedo(albedo)
    red = normal[32, :, 0].astype(int) - 128
    left, right = red[26:30], red[34:38]
    assert np.any(left < 0) and np.any(right > 0)


def test_flip_normal_x_inverts_red():
    normal = np.array([[[0, 128, 255], [64, 10, 20]]], dtype=np.uint8)
    out = flip_normal_x(normal)
    assert out.tolist() == [[[255, 128, 255], [191, 10, 20]]]


def test_two_view_solid_takes_masks_of_different_widths():
    front = np.zeros((40, 40), bool)
    front[5:35, 10:30] = True
    side = np.zeros((40, 80), bool)
    side[5:35, 20:60] = True
    mesh = two_view_solid(front, side, (0.0, 1.0), x_range=(-1.0, 1.0), z_range=(-2.0, 2.0))
    p = np.asarray(mesh.positions)
    assert abs(p[:, 0].max() - 0.5) < 0.05 and abs(p[:, 2].max() - 1.0) < 0.05
