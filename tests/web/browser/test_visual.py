"""Visual parity: the page's first frame against the desktop's, by the mean colour of each
channel. The desktop side needs a wgpu adapter (lavapipe is enough) for Engine.render_frame."""

import io

import numpy as np
import pytest

from manifoldx.web.export import _as_main, capture_engine

from .conftest import APPS, WEB_WALK, _export

W, H = 320, 200
# For each channel, abs(web - py) <= max(THRESHOLD * py, FLOOR), in 0-255. The plan started at
# 0.08, but lit_yard with the sky uncalibrated stayed within it (4.0, 5.5, 7.3% off), so it is
# the measured worst case plus 2 points: 0.3% (lit_yard's blue at CALIBRATION.environment 1.45;
# web_walk is within 0.1%) + 2 = 2.3%. With means near 130, FLOOR (about 3%) is the bound.
THRESHOLD = 0.023
FLOOR = 4.0
SCENES = {"web_walk": WEB_WALK, "lit_yard": APPS / "lit_yard.py"}


def mean_rgb(img: np.ndarray) -> tuple[float, float, float]:
    """The mean of each of the first three channels, in 0-255."""
    return tuple(float(c) for c in img[:, :, :3].reshape(-1, 3).mean(axis=0))


def within(web, py) -> list[bool]:
    return [abs(w - p) <= max(THRESHOLD * p, FLOOR) for w, p in zip(web, py)]


def show(name, web, py) -> None:
    ratio = ", ".join(f"{w / p:.3f}" if p else "-" for w, p in zip(web, py))
    print(f"\n{name}: desktop {_fmt(py)}  browser {_fmt(web)}  ratio {ratio}")


def _fmt(rgb) -> str:
    return "(" + ", ".join(f"{c:.1f}" for c in rgb) + ")"


@pytest.fixture(scope="module")
def desktop():
    """The desktop's mean colour of each scene: the engine captured as replay_python captures
    it, its GUI removed, one frame at W x H once nothing is loading."""
    means = {}

    def mean(name: str):
        if name not in means:
            engine, main = capture_engine(None, str(SCENES[name]), [])
            with _as_main(main):
                engine.gui.clear()
                engine.w, engine.h = W, H
                img = engine.render_frame()
                while engine._loading():
                    img = engine.render_frame()
            means[name] = mean_rgb(img)
        return means[name]

    return mean


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    done = {}

    def export(name: str):
        if name not in done:
            done[name] = _export(SCENES[name], tmp_path_factory.mktemp(name))
        return done[name]

    return export


def browser_mean(open_page, url, before_tick=None):
    """The page at W x H, device scale 1, its GUI hidden, after one tick of an empty frame."""
    from PIL import Image

    page = open_page(url, width=W, height=H, scale=1)
    page.evaluate("document.getElementById('mx-gui').hidden = true")
    if before_tick is not None:
        page.evaluate(before_tick)
    assert page.evaluate("__mx.tick(1 / 60).frame.vp") == [W, H]
    png = page.locator("#mx-canvas").screenshot()
    return mean_rgb(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")))


@pytest.mark.parametrize("name", list(SCENES))
def test_the_page_draws_as_the_desktop(name, desktop, exports, served, open_page):
    py = desktop(name)
    web = browser_mean(open_page, served(exports(name)))
    show(name, web, py)
    assert all(within(web, py)), f"{name}: browser {_fmt(web)} against desktop {_fmt(py)}"


def test_break_sky_uncalibrated(desktop, exports, served, open_page):
    """lit_yard with the environment's factor at 1.0: out of the threshold."""
    py = desktop("lit_yard")
    web = browser_mean(
        open_page, served(exports("lit_yard")), "__mx.scene.setCalibration({ environment: 1.0 })"
    )
    show("lit_yard, environment 1.0", web, py)
    assert not all(within(web, py))


# The batches are built at the first tick's sync from the library, so the swap is there.
SWAP_BASIC = """async () => {
  const THREE = await import("three");
  for (const [id, m] of __mx.scene.materials) {
    if (m.userData.mxBasic) {
      __mx.scene.materials.set(id, new THREE.MeshStandardMaterial({ color: m.color, roughness: 1 }));
    }
  }
}"""


def test_break_plain_standard_for_basic(desktop, exports, served, open_page):
    """web_walk with three.js's MeshStandardMaterial in place of the mx basic material: out."""
    py = desktop("web_walk")
    web = browser_mean(open_page, served(exports("web_walk")), SWAP_BASIC)
    show("web_walk, MeshStandardMaterial", web, py)
    assert not all(within(web, py))
