"""trace.js in headless Chromium (SwiftShader's WebGPU) against manifoldx.trace in Python: the
same scene, camera and seed give the same radiance, denoised radiance and image. Skipped where
Chromium or Python has no WebGPU adapter."""

import base64
import io
import json
import struct

import numpy as np
import pytest

from manifoldx.trace import Environment, Scene, scenes
from manifoldx.trace.tracer import Tracer

SIZE = 32
WEBGPU = ["--enable-unsafe-webgpu", "--enable-unsafe-swiftshader"]

PAGE = """<!doctype html><html><body><script type="module">
import * as trace from './trace/web/trace.js';
import * as glb from './trace/web/glb.js';
const f32 = (b) => new Float32Array(Uint8Array.from(atob(b), (c) => c.charCodeAt(0)).buffer);
window.sceneOf = (s) => ({ ...s, positions: f32(s.positions), normals: f32(s.normals), uvs: f32(s.uvs),
  colors: f32(s.colors), materialIds: Int32Array.from(s.materialIds), textures: [],
  environment: s.environment && { ...s.environment, radiance: f32(s.environment.radiance) } });
window.render = async (s, { spp, seed, width, height }) => {
  const t = await trace.createTracer(sceneOf(s), { width, height, seed });
  t.step(spp);
  const out = { radiance: Array.from(await t.radiance()), denoised: Array.from(await t.denoised()),
    image: Array.from((await t.image()).data) };
  t.destroy();
  return out;
};
window.trace = trace;
window.glb = glb;
window.ready = true;
</script></body></html>"""


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def _env(env: Environment) -> dict:
    h, w = env.radiance.shape[:2]
    return {"radiance": _b64(env.radiance), "width": w, "height": h, "intensity": env.intensity,
            "sun_direction": None if env.sun_direction is None else list(env.sun_direction),
            "sun_color": list(env.sun_color), "sun_illuminance": env.sun_illuminance,
            "sun_angular_radius": env.sun_angular_radius}


def _scene(s: Scene) -> dict:
    """A Python Scene as the page's scene object (textures left out)."""
    mats = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in m.__dict__.items()} for m in s.materials]
    cam = s.camera
    lights = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in light.__dict__.items()} for light in s.lights]
    return {"positions": _b64(s.positions), "normals": _b64(s.normals), "uvs": _b64(s.uvs), "colors": _b64(s.colors),
            "materialIds": s.material_ids.tolist(), "materials": mats, "lights": lights,
            "camera": {"position": list(cam.position), "target": list(cam.target), "up": list(cam.up), "fov": cam.fov},
            "environment": _env(s.environment), "triangleCount": s.triangle_count}


@pytest.fixture(scope="module")
def page(served, tmp_path_factory):
    from playwright.sync_api import Error, sync_playwright

    from manifoldx.trace.web import copy_to

    root = tmp_path_factory.mktemp("trace_web")
    copy_to(root / "trace")
    (root / "index.html").write_text(PAGE)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, args=WEBGPU)
        except Error as e:
            pytest.skip(f"chromium not installed: {str(e).splitlines()[0]}")
        pg = browser.new_page()
        pg.goto(served(root) + "index.html")
        pg.wait_for_function("window.ready === true")
        if not pg.evaluate("async () => Boolean(navigator.gpu && await navigator.gpu.requestAdapter())"):
            pytest.skip("chromium has no WebGPU adapter")
        pg.root = root
        yield pg
        browser.close()


@pytest.fixture(scope="module")
def python_device():
    import wgpu

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as e:  # no Vulkan, Metal or D3D driver at all
        pytest.skip(f"no wgpu adapter: {e}")
    if adapter is None:
        pytest.skip("no wgpu adapter")
    from manifoldx.trace.tracer import _device

    return _device()


def _page(page, scene, spp, seed=1):
    out = page.evaluate("([s, o]) => render(s, o)",
                        [_scene(scene), {"spp": spp, "seed": seed, "width": SIZE, "height": SIZE}])
    return {k: np.asarray(v, np.float32).reshape(SIZE, SIZE, -1) for k, v in out.items()}


def _python(device, scene, spp, seed):
    t = Tracer(scene, SIZE, SIZE, seed=seed, device=device)
    t.step(spp)
    return t


def _blocks(img, block=4):
    k = SIZE // block
    return img[..., :3].reshape(k, block, k, block, 3).mean(axis=(1, 3))


def _error(a, b) -> float:
    """Mean difference of 4x4 block averages, relative to b's mean."""
    return float(np.abs(_blocks(a) - _blocks(b)).mean() / max(np.abs(b).mean(), 1e-9))


# The page runs on another device than Python (SwiftShader against lavapipe here), whose sin, cos
# and exp differ in the last bits; a path that grazes an edge then goes the other way, so pages and
# Python agree as two seeds do, not bit for bit. Same seed, same first hits, same mean.
@pytest.mark.parametrize("name", ["cornell_box", "glass_sphere", "window_room", "lit_plane"])
def test_the_page_traces_what_python_traces(page, python_device, name):
    scene = getattr(scenes, name)()
    js = _page(page, scene, 16)["radiance"]
    py = _python(python_device, scene, 16, seed=1).radiance()
    spread = _error(_python(python_device, scene, 16, seed=2).radiance(), py)
    assert _error(js, py) <= 1.5 * spread + 1e-4, f"{name}: page {_error(js, py):.4f}, seeds {spread:.4f}"
    assert abs(js.mean() / py.mean() - 1) < 0.03


def test_the_page_denoises_and_tonemaps_as_python(page, python_device):
    scene = scenes.window_room()
    js = _page(page, scene, 16)
    a, b = _python(python_device, scene, 16, seed=1), _python(python_device, scene, 16, seed=2)
    spread = _error(b.denoised(), a.denoised())
    assert _error(js["denoised"], a.denoised()) <= 1.5 * spread + 1e-4
    img_spread = np.abs(b.image().astype(np.float32) - a.image()).mean()
    assert np.abs(js["image"][..., :3] - a.image()).mean() <= 1.5 * img_spread + 0.5


def test_a_different_seed_gives_a_different_image(page):
    a = _page(page, scenes.cornell_box(), 4, seed=1)["radiance"]
    b = _page(page, scenes.cornell_box(), 4, seed=2)["radiance"]
    assert np.abs(a - b).mean() / a.mean() > 0.05


def test_a_lambert_plane_matches_the_analytic_answer(page):
    """As tests/trace's: the page's own kernels, no Python in the loop."""
    want = 0.5 * np.sin(np.radians(60)) / np.pi
    out = page.evaluate("""async (s) => { const t = await trace.createTracer(sceneOf(s), { width: 8, height: 8, diffuseOnly: true });
        t.step(4); const r = Array.from(await t.radiance()); t.destroy(); return r; }""", _scene(scenes.lit_plane(60, 0.5)))
    assert np.allclose(out, want, rtol=1e-3)


# ------------------------------------------------------------------ the glb reader


def _png(w=4, h=2) -> bytes:
    from PIL import Image

    px = (np.arange(w * h * 4, dtype=np.uint32) * 37 % 256).astype(np.uint8).reshape(h, w, 4)
    px[..., 3] = 255
    buf = io.BytesIO()
    Image.fromarray(px, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _glb_with_texture_and_groups(tmp_path):
    """Three boxes, two of them under a rotated, mirrored group, one textured; written with
    manifoldx's writer and then extended by hand."""
    from manifoldx.gltf import Node, build_glb
    from manifoldx.resources import StandardMaterial
    from manifoldx.trace.gltf_read import parse_glb

    geo = scenes.box((0, 0, 0), (1, 2, 3))
    p = geo["positions"]
    geo = dict(geo, uvs=(p[:, :2] + 1.0).astype(np.float32),
               normals=(p / np.linalg.norm(p, axis=1, keepdims=True)).astype(np.float32))
    nodes = [Node(geo, StandardMaterial(color=(1.0, 0.5, 0.25), roughness=0.3),
                  name=f"part-{i}", pos=(i, 0.5 * i, 0.0)) for i in range(3)]
    data, _ = build_glb(nodes)
    doc, binary = parse_glb(data)
    png = _png()
    pad = b"\0" * (-len(binary) % 4)
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary) + len(pad), "byteLength": len(png)})
    binary = binary + pad + png + b"\0" * (-len(png) % 4)
    doc["buffers"][0]["byteLength"] = len(binary)
    doc["images"] = [{"bufferView": len(doc["bufferViews"]) - 1, "mimeType": "image/png"}]
    doc["textures"] = [{"source": 0}]
    textured = dict(doc["materials"][0], name="painted")
    textured["pbrMetallicRoughness"] = dict(textured["pbrMetallicRoughness"], baseColorTexture={"index": 0})
    doc["materials"].append(textured)
    for prim in doc["meshes"][doc["nodes"][1]["mesh"]]["primitives"]:
        prim["material"] = len(doc["materials"]) - 1
    q = (0.0, float(np.sin(np.pi / 8)), 0.0, float(np.cos(np.pi / 8)))
    doc["nodes"].append({"name": "group", "children": [1, 2], "rotation": list(q), "scale": [-1.0, 1.0, 2.0],
                         "translation": [0.0, 1.0, 0.0]})
    doc["scenes"][doc.get("scene", 0)]["nodes"] = [0, len(doc["nodes"]) - 1]
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(binary))
    out += struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    path = tmp_path / "groups.glb"
    path.write_bytes(out)
    return path


def test_the_page_reads_a_glb_as_python_does(page, tmp_path):
    src = _glb_with_texture_and_groups(tmp_path)
    (page.root / "groups.glb").write_bytes(src.read_bytes())
    overrides = {"painted": {"roughness": 0.8}, "part-2": {"metallic": 1.0, "texture": None}}
    want = Scene.from_glb(src, overrides)
    got = page.evaluate("""async (o) => {
        const s = await glb.loadGlb('groups.glb', { overrides: o });
        return { positions: Array.from(s.positions), normals: Array.from(s.normals), uvs: Array.from(s.uvs),
                 materialIds: Array.from(s.materialIds), materials: s.materials, nodeNames: s.nodeNames,
                 textures: s.textures.map((t) => [t.width, t.height, Array.from(t.data)]) };
    }""", overrides)
    assert got["nodeNames"] == [list(n) for n in want.node_names]
    assert np.allclose(np.reshape(got["positions"], (-1, 3, 3)), want.positions, atol=1e-5)
    assert np.allclose(np.reshape(got["normals"], (-1, 3, 3)), want.normals, atol=1e-5)
    assert np.allclose(np.reshape(got["uvs"], (-1, 3, 2)), want.uvs, atol=1e-6)
    assert got["materialIds"] == want.material_ids.tolist()
    for m, w in zip(got["materials"], want.materials):
        assert m["name"] == w.name
        assert np.allclose(m["base_color"], w.base_color) and m["texture"] == w.texture
        assert (m["roughness"], m["metallic"]) == pytest.approx((w.roughness, w.metallic))
    assert len(got["textures"]) == len(want.textures) == 1
    w, h, data = got["textures"][0]
    assert np.array_equal(np.asarray(data, np.uint8).reshape(h, w, 4), want.textures[0])


def test_hide_drops_a_node_and_its_subtree(page, tmp_path):
    src = _glb_with_texture_and_groups(tmp_path)
    (page.root / "groups.glb").write_bytes(src.read_bytes())
    names = page.evaluate("async () => (await glb.loadGlb('groups.glb', { hide: ['gr*'] })).nodeNames.map((n) => n[0])")
    assert names == ["part-0"]
