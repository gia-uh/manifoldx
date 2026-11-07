"""examples/web_crowd.py in headless Chromium: tests/web/test_web_crowd.py's script driven
through Playwright's keyboard and mouse runs as Python runs it on the frames the page built,
the page draws one call per non-empty (geometry, material) batch, and a picked box is drawn by
the batch of its material. tests/web/browser/test_visual.py checks the batched scene against
the desktop's: lit_yard exports as a library and draws through batches."""

import json

from manifoldx.web.replay import compare, replay_python

from ..test_web_crowd import CLICKS, PATHS, SPACE, script
from .conftest import BATCHES_OF, WEB_CROWD


def drive(page, n: int | None = None) -> list[dict]:
    """The script's first n frames (all by default): the pointer moves to each frame's pos, a
    click frame clicks there, a Space frame presses Space, the resize frame resizes the view,
    then one __mx.tick. Returns what each tick returned."""
    page.evaluate(f"() => {{ __mx.paths = {json.dumps(PATHS)}; }}")
    ran, view = [], None
    for f in script()[:n]:
        if f["vp"] != view:
            view = f["vp"]
            page.set_viewport_size({"width": view[0], "height": view[1]})
        if f["clicks"]:
            page.mouse.click(*f["pos"])
        else:
            page.mouse.move(*f["pos"])
        if f["just"]:
            page.keyboard.press("Space")
        ran.append(page.evaluate("__mx.tick(1 / 60)"))
    return ran


def test_chromium_matches_python(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    ran = drive(page)
    frames = page.evaluate("__mx.frames")
    trace = page.evaluate("__mx.trace")
    assert len(frames) == len(script())
    # the frames the page built hold the script's input
    assert [i for i, f in enumerate(frames) if f["just"] == ["Space"]] == list(SPACE)
    assert [i for i, f in enumerate(frames) if f["clicks"] == [1]] == list(CLICKS)
    assert frames[300]["vp"] == [640, 400]

    python = replay_python(None, str(WEB_CROWD), frames, PATHS)
    mismatch = compare(python, trace)
    assert mismatch is None, str(mismatch)

    # what the script does happened in the page: waves, destroys, a pick in gold
    assert trace[-1]["state"]["waves"] == 4
    alive = [len(e["store"]["alive"]) for e in trace]
    assert alive[7] == 49 and alive[199] == 1
    gold = page.evaluate("__mx.app.modules.__main__.GOLD.id")
    assert ran[40]["entry"]["paths"]["picked"] is not None
    assert ["material", ran[40]["entry"]["paths"]["picked"], gold] in ran[40]["entry"]["log"]


def test_one_draw_call_per_batch(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    drive(page, 42)  # two waves, and frame 41 painted one box gold
    calls = page.evaluate("__mx.renderer.info.render.calls")
    drawn = page.evaluate("[...__mx.scene.batches.values()].filter((b) => b.count > 0).length")
    assert page.evaluate("__mx.renderer.info.autoReset") is True
    assert calls == drawn == 4  # the ground, two colours of 23 and 24 boxes, the gold one


def test_an_entitys_material_is_read_from_its_batch(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    ran = drive(page, 41)  # frame 40 picks the front box of the first ring
    picked = ran[40]["entry"]["paths"]["picked"]
    assert picked is not None
    gold = page.evaluate("__mx.app.modules.__main__.GOLD.id")
    keys = page.evaluate(BATCHES_OF, picked)
    assert len(keys) == 1, f"batches drawing entity {picked}: {keys}"
    assert page.evaluate(
        "([key, id]) => __mx.scene.batches.get(key).material === __mx.scene.materials.get(id)",
        [keys[0], gold],
    ), f"entity {picked} is drawn by batch {keys[0]}, not with GOLD"
    # its neighbour keeps its wave's colour
    red = page.evaluate("__mx.app.modules.__main__.COLORS[0].id")
    others = [e for e in ran[40]["entry"]["store"]["alive"] if e not in (0, picked)]
    [key] = page.evaluate(BATCHES_OF, others[0])
    assert page.evaluate(
        "([key, id]) => __mx.scene.batches.get(key).material === __mx.scene.materials.get(id)",
        [key, red],
    )
