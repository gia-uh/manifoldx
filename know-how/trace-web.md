---
when: running manifoldx.trace in a web page (trace.js), wiring it into a site's viewer, changing a trace kernel or its buffers, or testing the page's tracer in headless Chromium
triggers: ["trace.js", "webgpu", "browser tracer", "path tracer in the page", "copy_to", "storage buffers per stage", "swiftshader", "headless canvas"]
---

# The path tracer in a web page

## When to use this

`manifoldx.trace` runs the same WGSL kernels in Python (wgpu-py) and in a page (WebGPU, `src/manifoldx/trace/web/trace.js`). Read this before you put the tracer in a site, before you change a kernel or the buffers it binds, and before you test the page.

## Shipping it to a site

`manifoldx.trace.web.copy_to(dest)` writes `dest/web/*.js` and `dest/wgsl/*.wgsl`. `trace.js` fetches the kernels from `../wgsl/` beside itself, and builds the BVH in `bvh-worker.js` (a module worker) beside itself. The site serves `dest` as static files; the modules import nothing outside it, three.js included.

```js
import { loadGlb, createTracer, cameraFromThree, supported } from '/static/trace/web/trace.js';

if (await supported()) {
  // settings: what a Python service would hand Scene.from_glb and Environment.sky
  const scene = await loadGlb('scene-hq.glb', { overrides: settings.overrides, hide: settings.hide });
  const tracer = await createTracer(scene, { width: 960, height: 600, canvas, environment: settings.environment });
  // when the camera stops:
  tracer.setCamera(cameraFromThree(threeCamera));  // true when it moved: the image restarts
  tracer.start({ maxSamples: 512 });              // a few samples per frame within a time budget, drawn denoised
  // when it moves again:
  tracer.stop();
}
```

- `overrides` maps a node or material name (fnmatch glob) to Material fields named as in Python (`base_color` linear, `roughness`, `metallic`, `transmission`, `ior`, `thin`, `texture: null`); node names win over material names. `hide` is a list of node-name globs; a match drops the node and its subtree.
- `environment` is `{sun_direction, turbidity, ground}` (any of `Environment.sky`'s keyword arguments) or an environment object from `sky()`. Without it the glb's stored map, else the default sky.
- Show the raster view while the camera moves and the canvas once `tracer.samples > 0`. The canvas is the trace's size, stretched to its CSS size; trace at a fraction of the screen's pixels on a weak GPU.
- `tracer.image()` returns `{width, height, data}` (RGBA sRGB) without a canvas; `radiance()` and `denoised()` return linear floats, as their Python namesakes.
- A large scene costs before the first sample: on a loaded server CPU, 777k triangles read in 0.9 s, build in about 4 s in the worker, and pack in 1 s on the main thread; 22k triangles take 0.2 s each.

## Changing a kernel or its buffers

- WebGPU's default device allows 8 storage buffers per shader stage, and a page should not ask for more. The scene uses four (`nodes`, `tris`, `shade`, `aux`); everything else read-only lives in `aux` at offsets passed in `Params` (`tracer.pack_aux`, `packAux` in trace.js). The heaviest kernel, `prep.wgsl`, binds 8. Add a table to `aux`, not a binding; `tests/trace/test_tracer.py::test_every_kernel_fits_webgpus_default_limits` fails otherwise.
- Python's `Tracer._upload` and trace.js's `upload()` pack the same layout, and `_write_params` and `writeParams` write the same `Params`. Change both, then run both test folders.
- Anything sampled by index (the emissive CDF) is listed in the scene's triangle order, not the BVH's: the page's BVH has another shape than numpy's.

## Testing

`tests/web/browser/test_trace_web.py` traces the reference scenes in headless Chromium and compares with Python. Two things you will meet:

- Page and Python run on different devices (SwiftShader against lavapipe here, two GPUs in general) whose `sin`, `cos` and `exp` differ in the last bits. Primary hits match exactly; a path that grazes an edge later goes the other way. Compare by block averages against Python's own spread between two seeds, never bit for bit.
- With Chromium's default flags a WebGPU canvas destroys the device at its first present in headless mode; `--enable-features=Vulkan,UseSkiaRenderer --use-angle=vulkan` keeps it alive, but the canvas is still not composited, so screenshots of it are blank. Read pixels with `tracer.image()`, which shares `render()` with the canvas.
- Compute alone works with `--enable-unsafe-webgpu`; `navigator.gpu` exists only on a secure origin, so serve the page from `127.0.0.1`, not `file://` or `about:blank`.
