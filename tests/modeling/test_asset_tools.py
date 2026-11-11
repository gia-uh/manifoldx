"""Tests for tracing a flat-colour blueprint into Shape layers (modeling.trace)."""

import numpy as np
import pytest

from manifoldx.modeling.shape import Shape
from manifoldx.modeling.trace import (
    classify,
    close_mask,
    components,
    contours,
    cut_by_rays,
    dilate,
    erode,
    iou,
    open_mask,
    redraw,
    simplify,
    trace_layers,
)

PALETTE = {"white": (255, 255, 255), "steel": (70, 100, 140), "brass": (190, 150, 60)}


def _disc(size, centre, radius):
    height, width = size
    yy, xx = np.indices((height, width))
    return (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius * radius


def _paint(mask, colour, size, background=(255, 255, 255)):
    img = np.empty((size[0], size[1], 3), dtype=np.uint8)
    img[...] = background
    img[mask] = colour
    return img


def _nonwhite(img):
    return np.any(img != 255, axis=-1)


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_classify_picks_the_nearest_palette_colour():
    img = np.array([[[255, 255, 255], [69, 99, 141]],
                    [[191, 149, 61], [10, 10, 10]]], dtype=np.uint8)
    labels = classify(img, PALETTE)
    assert labels.tolist() == [[0, 1], [2, 1]]
    assert labels.dtype.kind == "i"


def test_morphology_square_and_disc():
    dot = np.zeros((9, 9), dtype=bool)
    dot[4, 4] = True

    grown = dilate(dot, 1)
    assert grown.sum() == 9 and grown[3:6, 3:6].all()

    disc = dilate(dot, 1, shape="disc")
    assert disc.sum() == 5 and disc[4, 4] and disc[3, 4] and disc[5, 4]

    assert erode(dilate(dot, 2), 2).sum() == 1
    assert open_mask(dot, 1).sum() == 0
    assert close_mask(dot, 1).sum() == 1


def test_components_respect_connectivity():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    mask[4:6, 4:6] = True  # touches the first block only at a corner
    labels4, count4 = components(mask, connectivity=4)
    labels8, count8 = components(mask, connectivity=8)
    assert count4 == 2 and count8 == 1
    assert labels4.max() == 2 and labels4[0, 0] == 0
    assert set(np.unique(labels8)) == {0, 1}


def test_contours_square_is_closed_and_ccw():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 8:20] = True
    outer, holes = contours(mask, holes=True)
    assert len(outer) == 1 and holes == []
    poly = outer[0]
    assert not np.allclose(poly[0], poly[-1])  # closed by ordering, not a repeated point
    assert _signed_area(poly) == pytest.approx(120, rel=0.05)
    assert _signed_area(poly) > 0  # counter-clockwise in the y-up frame
    assert poly[:, 1].max() <= 0  # row 0 of the picture is y = 0, lower rows are negative


def test_contours_reports_a_hole():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[15:25, 15:25] = False
    outer, holes = contours(mask, holes=True)
    assert len(outer) == 1 and len(holes) == 1
    assert _signed_area(outer[0]) == pytest.approx(900, rel=0.05)
    assert _signed_area(holes[0]) == pytest.approx(-100, rel=0.1)
    assert _signed_area(holes[0]) < 0


def test_simplify_keeps_a_square_four_corners():
    points = []
    for i in range(10):
        points.append((i, 0))
    for i in range(10):
        points.append((10, i))
    for i in range(10, 0, -1):
        points.append((i, 10))
    for i in range(10, 0, -1):
        points.append((0, i))
    poly = np.array(points, dtype=float)
    simple = simplify(poly, 0.1)
    assert len(simple) == 4
    assert {(round(x), round(y)) for x, y in simple} == {(0, 0), (10, 0), (10, 10), (0, 10)}


def test_iou_of_masks():
    a = np.zeros((5, 5), dtype=bool)
    a[:3, :3] = True
    assert iou(a, a) == 1.0
    b = ~a
    assert iou(a, b) == 0.0
    c = np.zeros_like(a)
    c[:2, :3] = True
    assert iou(a, c) == pytest.approx(6 / 9)


def _wheel(size=(260, 260), radius=110, centre=(130.0, 130.0), windows=5,
           window_radius=12, window_distance=55):
    angles = np.deg2rad(np.arange(windows) * 360 / windows + 90)
    steel = _disc(size, centre, radius)
    for angle in angles:
        at = (centre[0] + window_distance * np.cos(angle),
              centre[1] + window_distance * np.sin(angle))
        steel &= ~_disc(size, at, window_radius)
    return steel, centre, angles


def test_cut_by_rays_opens_every_window():
    steel, centre, angles = _wheel()
    assert components(steel)[1] == 1
    cut = cut_by_rays(steel, centre, angles, width=1)
    labels, count = components(cut, connectivity=4)
    assert count == 5
    _, holes = contours(cut, holes=True)
    assert holes == []
    assert cut.sum() < steel.sum()


def test_trace_layers_redraws_a_cut_wheel():
    steel, centre, angles = _wheel()
    drawing = _paint(steel, PALETTE["steel"], (260, 260))
    cut_image = _paint(cut_by_rays(steel, centre, angles, 1), PALETTE["steel"], (260, 260))

    shape = trace_layers(cut_image, PALETTE, ["steel"], background="white", tol=1.0, min_area=20)
    assert isinstance(shape, Shape)
    steel_layer = shape.layers[0]
    assert steel_layer.material == "steel" and len(steel_layer.polygons) == 5

    redrawn = redraw(shape, (260, 260), PALETTE)
    assert iou(_nonwhite(redrawn), _nonwhite(drawing)) > 0.97


def test_trace_layers_turns_a_hole_into_a_layer_below():
    size = (60, 80)
    steel = np.zeros(size, dtype=bool)
    steel[10:50, 20:60] = True
    steel[25:35, 35:45] = False
    drawing = _paint(steel, PALETTE["steel"], size)

    shape = trace_layers(drawing, PALETTE, ["steel"], background="white", tol=0.5, min_area=10)
    assert [layer.material for layer in shape.layers] == ["steel", "white"]
    assert len(shape.layers[0].polygons) == 1 and len(shape.layers[1].polygons) == 1
    # The outer polygon has no hole: a region's holes are separate layers.
    outer = shape.layers[0].polygons[0]
    assert _signed_area(outer) > 0

    redrawn = redraw(shape, (80, 60), PALETTE)
    assert iou(_nonwhite(redrawn), _nonwhite(drawing)) > 0.99


def test_trace_layers_maps_the_frame_to_the_unit_square():
    size = (60, 80)
    steel = np.zeros(size, dtype=bool)
    steel[10:50, 20:60] = True
    shape = trace_layers(_paint(steel, PALETTE["steel"], size), PALETTE, ["steel"],
                         background="white", tol=0.5)
    poly = shape.layers[0].polygons[0]
    assert np.allclose(poly.min(axis=0), (-1, -1), atol=0.02)
    assert np.allclose(poly.max(axis=0), (1, 1), atol=0.02)
