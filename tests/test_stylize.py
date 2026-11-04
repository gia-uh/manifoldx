"""manifoldx.stylize.ink_and_wash: passes to a drawing, without a GPU."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from manifoldx.stylize import Label, ink_and_wash  # noqa: E402

STYLE = {
    "ink": {"color": "#20160f", "object": 2.3, "part": 1.2, "crease": 0.9, "wobble": 0.8, "far": 100},
    "sky": {"gradient": ["#3060c0", "#f0e0c0"], "blotch": 0.2, "blotch_scale": 40},
    "materials": {
        "red": {"color": "#c03020", "shadow": "#601810", "blotch": 0.2, "blotch_scale": 10},
        "sand": {"color": "#e0c890", "shadow": "#8a7040", "blotch": 0.3, "blotch_scale": 20, "streak": "x"},
    },
}


def _buffers(h=80, w=120):
    """Ground (label 1) on the lower half, a box (labels 2 and 3, one object) in front."""
    ids = np.zeros((h, w), np.int32)
    ids[h // 2 :] = 1
    ids[20:60, 40:60] = 2
    ids[20:60, 60:80] = 3
    normals = np.zeros((h, w, 3), np.float32)
    normals[ids == 1] = (0, 1, 0)
    normals[ids == 2] = (0, 0, 1)
    normals[ids == 3] = (1, 0, 0)
    depth = np.full((h, w), np.inf, np.float32)
    depth[ids > 0] = 10
    light = np.where(ids == 3, 0.2, 0.9).astype(np.float32)
    light[ids == 0] = 0
    return dict(ids=ids, normals=normals, depth=depth, light=light)


LABELS = [Label(1, "sand"), Label(2, "red"), Label(2, "red")]


def test_same_inputs_same_image():
    a = ink_and_wash(_buffers(), LABELS, STYLE, seed=3)
    b = ink_and_wash(_buffers(), LABELS, STYLE, seed=3)
    assert a.shape == (80, 120, 3) and np.array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 1


def test_seed_changes_the_grain():
    assert not np.array_equal(ink_and_wash(_buffers(), LABELS, STYLE, seed=1), ink_and_wash(_buffers(), LABELS, STYLE, seed=2))


def test_outline_is_darker_than_the_inside():
    img = ink_and_wash(_buffers(), LABELS, STYLE).mean(-1)
    assert img[40, 40] < img[40, 50] - 0.2  # the box's left outline against its red face


def test_shadow_side_is_darker_than_the_lit_side():
    img = ink_and_wash(_buffers(), LABELS, STYLE).mean(-1)
    assert img[30:50, 64:76].mean() < img[30:50, 44:56].mean()


def test_no_ink_object_has_no_outline():
    plain = ink_and_wash(_buffers(), LABELS, STYLE).mean(-1)
    soft = ink_and_wash(_buffers(), [LABELS[0], Label(2, "red", no_ink=True), Label(2, "red", no_ink=True)], STYLE).mean(-1)
    assert soft[40, 40] > plain[40, 40] + 0.2


def test_sky_follows_its_gradient_top_to_bottom():
    img = ink_and_wash(_buffers(), LABELS, STYLE)
    top, low = img[2, 5:35].mean(0), img[36, 5:35].mean(0)
    assert top[2] > top[0] and low[0] > top[0]  # blue at the top, warmer lower down


def test_unknown_material_is_named():
    with pytest.raises(ValueError, match="stone"):
        ink_and_wash(_buffers(), [Label(1, "stone")] + LABELS[1:], STYLE)
