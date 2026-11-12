# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `manifoldx asset FILE COMMAND`: run one asset file's stages from the shell (`draw`, `build`, `lint`, `inspect`,
  `paint`, `render`), with `--json`, `--budget`, `--model`, `--again`, `--max-tris`, `--lod low` and `--out`; exit
  0 on success, 1 on a finding and 2 on a usage error. `Parts.paint` asks the image model for one albedo per view
  from a whole-view template, re-asks an answer that keeps under 0.85 of the outline (at most three times), strips
  the code colours, bleeds the atlas and computes a normal map, recording A-CODE and A-OUTLINE. `save(result,
  out_dir)` writes `model.glb` (textured when an albedo exists), `albedo.png`, `normal.png` and `report.json`, and
  `inspect(asset, c, scene, out_dir)` writes `overlay-<view>.png`, `groups.png` and `summary.txt`. `render` runs a
  toon or photo still through Blender when a `blender` executable is on PATH, from scripts shipped as package
  data.

- `manifoldx.modeling.bake.bake_projection(car, views, size=2048, materials=None, fallback=None)`: texture a
  car's atlas by projecting orthographic `left`, `right`, `front`, `rear`, `top` and `bottom` drawings. Every
  UV-mapped texel is rasterised to its 3D point, normal and material (barycentric fill per triangle, vectorised
  over texels); each view is registered by fitting the asset's projected silhouette box to the drawing's
  non-background box, sampled bilinearly, and blended by how squarely the normal faces the view. Only the named
  materials (by default paint, black, trim, chrome, plate and underbody) are baked, others keep their palette
  colour, texels no view faces keep a fallback, and the whole atlas is filled from each texel's nearest covered
  neighbour so mip levels never sample the background. numpy only, no new dependency; a missing `right`
  mirrors `left`.

- `bake_projection(..., keypoints=None)` registers a view by named parts instead of by its silhouette box:
  `{view: {landmark: (px, py)}}`, with `px` from the image's left and `py` from its top. `Car.landmarks`
  names the four wheel centres, the nose and the tail, the middle of the windscreen's and rear window's
  edges, the roof's highest point and where each door mirror meets the door, as 3D points. Three to five
  keypoints fit an affine transform by least squares; six or more a regularised thin-plate spline, so a
  drawing whose parts sit in the wrong place lands each one on the part of the asset it shows. A view
  without keypoints keeps the box registration. numpy only.

- `manifoldx.modeling.assets`, procedural assets from named parameters. First asset: `build_car(CarParams(...))`, a low-poly car lofted from spec-sheet measurements and profile curves, with glass, pillars and doors as face tags, lamps and plates as projected outlines, and doors, hood, hatch and wheels as hinged parts (`Car.pose`, `spawn_car`, `place_car`). Its texture atlas (`ATLAS`) has one island per view of the car (left, right, top, front, rear, wheel, dashboard, seats), so an island is a picture a paint program or an image model reads at a glance; `spawn_car(..., texture=path)` wraps an albedo image laid out that way. A lamp is a shape (layers of flat outlines: housing, lit guides, painted parts, lens), placed by an anchor and turned to face its corner of the car. The grille is a shape (the opening and the lights at its ends; its middle stretches to the grille's length and its ends keep their proportions). A wheel's face is a shape (hub, spokes and ring as solid pieces standing on the dish of the rim, bolts and cap on them), and a door mirror a shape (front view, glass and plan, built as a row of rounded sections and swept back by `mirror_yaw`); The dashboard is a shape (front view and side section: the section is swept across the cabin and cut to the front view's height, leather, trim and vent are laid on it and each screen stands off it as a slab), and the steering wheel a shape, solid, with a picture of its own in the dash island. The defaults are a neutral compact hatchback with the plain parts (`grille_style="bar"`, `wheel="spokes"`, `mirror="box"`, `dash="box"`, `steering="ring"`). A shape is a dict of traced outlines passed in its field; manifoldx ships none. One traced shape of each kind is in `examples/models/car_shapes.py`, which also says how each kind is drawn. A door's inner panel is a shape, fitted to each door and laid on its skin from inside, and the console between the front seats a shape, made solid with its layers on top; each has an island of its own in the atlas (`door`, `console`), and by default they are left out. The steering wheel now leans with its top away from the driver. numpy only. A grille shape may carry `chrome` layers (bars, a frame) besides its dark opening and lit ends. `bed_u` makes a pickup: behind the cab the body is an open bed, a liner tub between rails at `bed_floor`, and the rear end is a tailgate hinged at the floor that folds down (`hatch`).

## [0.10.0] - 2026-11-08

### Added

- **The guide.** `docs/` is a literate book in scriptorium that follows one project, a town
  plaza, over fourteen chapters. Every block runs against the package when the guide renders,
  each chapter tangles its program into `docs/examples/`, and rift checks that every public
  class outside `manifoldx.web` is named in it. `make docs` renders it and `make docs-check`
  is the gate CI runs on every PR. The README points to it instead of keeping a second guide.
- **Release pipeline.** A tag publishes only after the whole suite passes on lavapipe from the
  lockfile, the docs gate passes and the tag matches `pyproject.toml`; the wheel is imported in
  a clean environment before upload, and the GitHub release carries this changelog's section.
  CI runs the whole suite on a software Vulkan adapter instead of skipping every render test,
  and lints the tests.

- **The path tracer in a web page.** `src/manifoldx/trace/web/trace.js` runs `manifoldx.trace`'s
  WGSL kernels on the page's WebGPU device: `loadGlb(url, {overrides, hide})` reads a .glb as
  `Scene.from_glb` does, the BVH is built in a worker, `createTracer(scene, {canvas, environment})`
  refines on a canvas with `start()` and restarts when `setCamera` moves the camera; `image()`,
  `radiance()` and `denoised()` read the result back. `manifoldx.trace.web.copy_to(dest)` writes
  the files a site serves. The kernels now bind at most 8 storage buffers per stage (WebGPU's
  default): the read-only tables share one buffer, and the denoiser's preparation runs in WGSL
  (`prep.wgsl`) in Python too (#178).

- **A path tracer, `manifoldx.trace`.** `Scene.from_glb(path, overrides)` reads any .glb
  (node trees, PBR factors, emission, transmission and volume, alpha blend as thin glass,
  base-colour textures, vertex colours, cameras, `KHR_lights_punctual`, the environment map
  manifoldx writes). `Tracer(scene, w, h)` builds a binned-SAH BVH in numpy and traces it in
  WGSL compute: Lambert + GGX, thin and solid dielectrics, emissive triangles, a Preetham sky
  with a sun disk (`Environment.sky`) or an HDR map, next-event estimation with MIS, Russian
  roulette. `tracer.step(n)` accumulates samples; `tracer.image()` denoises with an
  edge-aware à-trous filter and tonemaps to sRGB. `manifoldx trace scene.glb -o out.png`
  renders from the command line; `python -m benchmarks trace` measures it (#175).

- **export --web: skinned entities and matrices in frame code.** Transpiled code may call
  `set_joints` on the engine and on entity handles, use `@` on matrices and stacks of them
  (numpy's broadcasting; float32 summed in numpy's order, so the page matches bit for bit),
  `np.eye`, nested `np.array`, `np.stack` of any rank and `Camera.project`; the page draws
  each skinned entity posed, a three.js SkinnedMesh of its own, one draw call each.
  `examples/web_crowd.py` gains a walking figure (#140).

- **A Joystick widget.** `manifoldx.gui.Joystick(name=...)`: a drag gives a value in the unit
  disc, y up, emitting `gui:joystick:<name>:change`; release returns it to (0, 0) (#140).

- **export --web: the entity store in the page.** The export writes the store's arrays as
  `ecs.bin` and their layouts in `state.json`; the page keeps them in a `Store` with the
  desktop's slot rules and its flush after the systems. Transpiled systems use their
  component queries (`q[Velocity].v += GRAVITY * dt`, `len(q)`, `for i in q`), with the
  desktop's field-view semantics; `engine.spawn(..., n=)` from geometries and materials the
  export saw and `engine.destroy(...)`; entity handles (`h.transform.pos`, `.rot`, `.scale`,
  `h.set_material`, `h.destroy`) and `Transform.rotation(x, y, z)`. The array subset covers
  rank-1 and rank-2 float32, float64, int32 and bool arrays with numpy's dtype, broadcasting
  and rounding rules: `+ - * /`, comparisons, `& |`, `**` by 2, 0.5 and -1, masks and index
  arrays, `np.sin cos exp sqrt floor clip minimum maximum where sum mean min max argmin
  argmax stack column_stack linalg.norm`. The trace carries the store, and the Python replay
  flushes after the systems (#140). Know-how: `know-how/web-export.md`, "Entities in the
  page".
- **export --web: one instanced mesh per batch.** Under `--web`, `scene.glb` is a library of
  geometries and materials by id with no entity nodes (`export_engine(..., library=True)`);
  the page draws one `InstancedMesh` per (geometry, material) from the store and rebuilds the
  batches when the store's membership moves, so a frame draws one call per non-empty batch.
- **`_FieldView[:, j]`**, a one-column view of a query field on the desktop:
  `engine.destroy(q[Transform].pos[:, 1] < -10.0)` destroys the entities below -10.
- **export --web: events in the page.** `engine.emit` and `@engine.on` handlers run in the page
  and in both replays in the desktop's order: the `frame` event first, then what was emitted
  since the last delivery, then the frame's own events; an emit waits for the next frame. The
  seven input events (`key_down`, `key_up`, `pointer_down`, `pointer_up`, `pointer_move`,
  `wheel`, `resize`) reach handlers with the desktop's payload fields, `pointer_id` included.
  `engine.elapsed` and the `frame` event's `{"dt", "elapsed", "frame"}`. A handler of an app's
  own event annotates its payload (`def on_ping(p: dict)`). Replay frames gain `events` and
  `clock`, and the trace the names delivered each frame (#140).
- **export --web: `time.time()` in frame code** reads the frame's clock, which the page takes
  once per frame and never lets run backwards (#140).
- **export --web: `random.Random` in frame code**, made and seeded at setup: `random`,
  `uniform`, `randrange`, `randint`, `choice` and `getrandbits` (up to 53 bits) draw bit for bit
  as CPython's MT19937, and the module functions of the same names use the module's generator
  (#140).
- **export --web: `Button`, `Slider`, `Toggle`, `Joystick` and `Image` in the page.** HTML
  elements drawn from the widgets' styles, with ARIA roles and `touch-action: none`; the page
  records which widget each pointer event went to, and the desktop's widget code runs in the
  Python replay and its port in the page, so values, latches and events agree. A Slider and a
  Joystick keep their pointer, one per finger. A pointer down on the GUI is also a click, and
  `engine.gui.pointer_over_gui` comes from the widgets an event hit, as on the desktop (#140).
- **export --web: a host page talks to the export.** A page that frames it posts
  `{mx: "emit", name, payload}` into the app's next frame, and gets every event the app or its
  widgets emit back as the same message and as the `mx:emit` DOM event. An agent's
  `manifoldx emit` under `run --web` is recorded in the tab's frames too (#140).
- **`examples/web_crowd.py`**: boxes spawned by Space, drifting by a query system, destroyed
  when they fall, picked and painted gold; a controls panel with a Button that throws a wave,
  a gravity Slider, a spin Toggle, a wind Joystick and a compass Image, a handler per event,
  `random.Random` and the clock. It replays with no mismatch in Node and in Chromium, under the
  mouse and under two fingers.
- **`benchmarks` suite `web`**, case `web-crowd`: the exported page's work per frame in
  headless Chromium for 1,000 and 10,000 boxes, and its draw calls.

- **`manifoldx run --web [-m module | app.py] [--port N]`: the dev loop in a browser tab.**
  It serves the app's page on 127.0.0.1 (the app at `/`, the dev page at `/dev`), and on
  every save reruns setup in Python, exports again and pushes the new version over
  Server-Sent Events; the tab loads the new modules and snapshot in place, without a page
  reload. `engine.state` survives as plain data sent back from the tab (keys holding app
  objects, handles or functions keep Python's value), the three.js scene is kept when its
  hash is unchanged, and the pointer lock and held keys survive. The tab freezes while the
  reload builds, and a tab opened later gets a rebuild with the last state any tab sent.
  Versions live under `.manifoldx/web/` beside static files written once; the glb is written
  only when its hash is new. Each reload logs its time by phase. `--debug` is refused.
  `manifoldx.web.export.build_export` builds an export in memory, which `export --web` and
  `run --web` both write (#140).

- **The dev overlay of `run --web`.** A status line at the bottom right says `v3 · live`,
  `v4 · reloading`, `v3 · reload failed, running v3`, `v3 · paused until the next edit` or
  `v3 · dev server gone, retrying`; an edit that fails to load or transpile keeps the
  running version and shows its Python file and line, and an exception in a frame pauses
  the tab until the next save, as on the desktop (#140).

- **`manifoldx look`, `state`, `set`, `emit`, `events` and `log` against the tab.** Under
  `run --web` the dev server forwards them to the tab that connected last: `look` is the
  WebGL frame with the HTML GUI painted over it, `emit` fires the app's handlers at the
  start of the tab's next frame, and the log takes the tab's console warnings and errors as
  `tab: ` lines. With no tab open they answer 503; `events` answers from Python and
  `metrics` does not exist under `--web` (#140).

- **`engine.log(text)` in the web subset.** Frame code may log: under `run --web` the line
  reaches the dev server's log, an exported page drops it, and a replay in Node collects it
  in `engine.lines`. The runtime's record of engine calls is now `engine.calls`; the trace's
  JSON key stays `log` (#140).

- **`manifoldx export --web` writes a browser page.** Serve the folder over HTTP and open
  `index.html`: the app's systems run every animation frame on the browser's input, built
  into the replay's frame form (releases, pointer position, wheel and modifiers included),
  three.js draws `scene.glb` with the desktop's calibration (the sky environment at 1.45,
  BasicMaterial and PhongMaterial under the desktop's fixed light), and `engine.gui` is
  HTML (`Panel`, `Text`, `ValueDisplay`, nine anchors; `Button`, `Slider`, `Toggle` and
  `Image` fail the export). `capture_pointer` is pointer lock, requested at a user gesture,
  refused after a touch; losing the lock gives the app an Escape. A JS error stops the page
  and names the Python file and line from the source maps; a progress bar follows the
  scene's download. Transpiled code may read `engine.gui.pointer_over_gui` and
  `engine.camera.fov`. Tests in headless Chromium replay `examples/web_walk.py` against
  Python and compare the page's frame with the desktop's by mean colour; they need the new
  `browser` dependency group (Playwright) and run in their own CI job (#140).

- **three.js 0.186.1, vendored in `manifoldx/web/vendor/three/`.** The published ES
  modules and the GLTFLoader with its two utils, unchanged, under the npm integrity hash
  that `SOURCE` records with the upgrade steps (#140).

- **`manifoldx export --web [-m module | app.py] OUTDIR`: the app's per-frame Python,
  transpiled to JavaScript, with the state its setup left.** The app runs its setup in
  Python (startup handlers included); its systems, event handlers and ValueDisplay getters,
  and what they reach through types, become ES modules with source maps to the `.py` lines.
  The folder also holds the runtime, `state.json` (the reached objects and `engine.state`),
  `scene.glb` and `report.json` (roots, transpiled and setup functions, state left out).
  Code outside the subset fails with its file, line and root, and nothing is written. The
  folder also runs in Node. `know-how/web-export.md` says how (#140).

- **`manifoldx.web`: transpiler, snapshot, headless runtime and replay.** `transpile_app`
  and its typing from annotations, live objects and call sites; `snapshot` of reached
  state; `js/pyrt.js` and `js/manifoldx.js` (input, camera, `set_material` and
  `capture_pointer` without a renderer); and `manifoldx.web.replay`, which runs one script
  of input frames through the app in Python and through its export in Node and reports the
  first frame where they differ (`replay_python`, `replay_node`, `compare`) (#140).

- **`examples/web_walk.py`: walk, pick boxes and name them in a GUI panel.** The app the
  replay test exports and replays for 600 scripted frames (#140).

- **`engine.set_material(entity, material)`: switch an entity's material by handle or
  index.** Raises ValueError for a dead entity. With `engine.input.capture_pointer`, it is
  the engine API the web runtime implements for picking and mouse look (#140).

- **glTF export: entity and material ids, and unused materials.** Each node's extras carry
  `"mx": {"entity", "material"}` next to the app's own extras; the scene's extras map every
  registered material id to its glTF material under `"mx_materials"`, and materials no
  entity uses yet (a selection highlight) are written too (#140).

- **`engine.set_joints(indices, matrices)`: pose a crowd of skinned entities in one
  call.** `matrices` is (N, J, 4, 4), one pose per index, with the same rule and checks as
  `EntityHandle.set_joints`. It allocates every missing joint block at once, writes the
  CPU mirror in one assignment and uploads the touched range of blocks once, instead of
  one `write_buffer` per entity: 5000 entities of 20 joints take about 30-38 ms instead
  of about 106 ms of CPU time per frame (#115).

- **`benchmarks/`: suites that time manifoldx as N grows, and before/after tables
  for pull requests.** `uv run python -m benchmarks <suite> [--case ...] [--sizes ...]`
  runs the `frame`, `ecs`, `compute`, `skinning`, `modeling` and `loaders` suites;
  `--against main` runs the same cases on another ref's `src/` and prints a Markdown
  speed-up table. A smoke test runs every case at a tiny size; `know-how/benchmarking.md`
  says how to prove a performance change (#109).

- **`engine.input.capture_pointer(on)` and `engine.input.pointer_captured`: mouse look
  without the window edge.** In a glfw window it hides and locks the cursor (raw motion
  where the system has it); `mouse_delta` keeps counting, and the GUI ignores the pointer
  while it is captured. Offscreen it returns False and does nothing. `examples/input_fly.py`
  captures on a left click and releases on Escape (#16).

- **`EntityHandle.set_material(material)`.** Swaps an entity's material from the next
  frame, to any material type; the same material object keeps one registry id, so a
  selection highlight that switches back and forth does not grow the registry (#15).

- **`manifoldx.workers.ProcessStream`: heavy pure-Python work in a child process.** It
  runs a generator function in a spawned process and hands back what it yields, one
  item per `take()`, so a frame loop unpickles one piece per frame. A thread running the
  same work shares the GIL with the loop, and frames then stall for up to a second.

- **`manifoldx run app.py` (or `-m package.module`) with hot reload, and
  `engine.state`.** Saving any of the app's source files reruns it in the same window,
  on the same GPU device. The scene is rebuilt from the new code; `engine.state`, a
  dict that also reads as attributes, carries over, so an app that keeps its camera
  angle or game progress there continues where it was. A broken edit keeps the running
  version; an error while running pauses the app until the next save (#92).

- **Surface patterns in the fragment shader.** A vertex-coloured geometry's `surfaces`
  key gives each vertex a pattern id (0 flat, 1 grass, 2 cobbles, 3 veins, 4 brick,
  5 blocks, 6 planks, 7 tiles), and `StandardMaterial(vertex_colors=True)` draws that
  pattern from world position round the vertex colour: two or three flat tones and
  antialiased ink joints, fading to the flat colour once a cell is a few pixels wide.
  No texture memory. Parameters per pattern (scale, contrast, tones, ink width, ink
  darkness) default to `resources.SURFACE_PATTERNS`; `surface_patterns=` overrides
  them per material. Costs 0.5 ms on a laptop GPU and 5.3 ms on llvmpipe for a
  1280x720 frame mostly covered by patterns. Example: `examples/surface_patterns.py`.
  Streamed cells carry them too: `streaming.MeshData(..., surfaces=ids)`.

- **`Image` style key `uv` = (u0, v0, u1, v1) and `textures.write_texture(engine,
  handle, rgba, (x, y))`.** An Image draws that part of its source into its box, and
  the window can change every frame without an upload, so a big map texture pans under
  a round minimap. `write_texture` patches a region of a texture in place. The GUI
  samples a texture's full-size level only, so a patch shows at once.

- **Render layers: `Layer(layer=n)` on entities, `camera.layers = {n, ...}`.** A camera
  draws only entities on its layers (`None`, the default, draws all; an entity without
  `Layer` is on layer 0), and drops the others before any per-entity work. A
  render_to_texture view of a stage on its own layer then costs what the stage costs:
  1.3 to 1.6 ms of CPU a frame for a one-entity portrait with a game town loaded, against
  2 to 3 ms without layers (laptop GPU and llvmpipe, host under load).

- **`engine.render_to_texture(camera, (w, h), live=True, background=None)`: a second
  camera into a texture.** Rendered before each frame's main pass, without the GUI, and
  usable as an Image's source or an albedo_map. `live=False` renders once and then only
  on `refresh_texture(handle)`, so a still view costs nothing; `set_texture_live` and
  `drop_texture` switch and stop it. A live view costs 2.5 to 3.5 ms of CPU a frame in a
  20-entity scene and 4 to 7 ms with 2000 entities, nearly independent of its size from
  150 to 300 px, on a laptop GPU and on llvmpipe alike. Example:
  `examples/render_to_texture.py`.
- **`Image` draws a `TextureHandle`, and rounds its corners with `radius`.** Half the
  size makes a circle, antialiased in the shader.
- **`Text(style_overrides={"font": path})`: text in a font file.** Drawn by Pillow at
  its pixel size through the image path, with kerning and no 256-pixel width limit, and
  tinted by `fg`. Text without a `font` key is unchanged.

- **SVG images in the GUI: `manifoldx.gui.Image`.** Takes an SVG as a path, markup or
  bytes and draws it at the pixel size it lands at on screen (layout size times the GUI
  scale times the display's pixel ratio or a still's supersampling), so it stays crisp
  through resizes and scale changes. Style keys: `width`, `height` (one alone keeps the
  aspect), `tint`, `opacity`. While its size animates it draws from rasters bucketed in
  steps of 2^(1/8) and goes back to an exact raster when the size holds. Needs the new
  `svg` extra (`thorvg-python`, no system libraries): rasterizing takes 0.13 ms for a
  48 px icon and 1.9 ms at 512 px. `manifoldx.svg` exposes `rasterize_svg`, `svg_size`,
  `load_font` and `clear_cache`; rasters are cached by content and size.
- **`textures.texture_from_svg` and `textures.texture_from_array`.** An SVG or an RGBA
  array as a mipmapped texture for `StandardMaterial(albedo_map=...)`; `load_texture`
  now decodes and calls `texture_from_array`. Example: `examples/svg_icons.py`.

- **Frustum culling.** A mesh whose bounding sphere lies out of the camera's view is not
  drawn, and, with a fixed shadow box (`auto_fit=False`), one outside it casts nothing. A
  skinned mesh's sphere is padded for its moving joints, and a mirrored one is tested where
  its reflection shows. What a game parks out of sight costs nothing, and a geometry is
  first uploaded when first in view. `RenderPipeline.drawn` counts the instances drawn.
- **`StandardMaterial(mirror=..., opacity=...)`: a cheap reflection.** `mirror=y` draws a
  mesh reflected in the plane y, lit where it really stands, with culling flipped and
  whatever lies below the plane dropped. `opacity < 1` blends a surface over what is
  behind it; the renderer draws translucent meshes after every opaque one, whatever the
  spawn order. Scenery mirrored under translucent water reflects in it, with no second
  camera and no render target. Neither kind casts a shadow.
  `cut=(x0, z0, x1, z1)` drops whatever of a mesh lies inside an xz rectangle, such as
  far land under near water, so it hides no reflection.
- **`StandardMaterial(ripple=...)`: water in the fragment shader.** Dashed bands of light
  drift across the surface, one wave train bent by a second and broken by a third. Like
  the sway, every wave repeats over 64 s and 256 world units, so the clock and
  `set_origin` never make it jump. A material's uniform buffer now grows to its data.
- **`StandardMaterial(sway=...)`: wind in the vertex shader.** A vertex moves in proportion
  to its height above its model's origin, with a phase from its world position, so a field
  of grass or a wood sways as waves cross it and no two trees move alike. It costs a few
  vertex instructions and no CPU work. The clock and the floating origin reach the shader
  in the globals' spare slots (time mod 64 s, origin mod 256 units, the periods every wave
  repeats over), so neither a long session nor `set_origin` makes the sway jump. Shadows
  stay still. `sway_reach` levels the sway off past a height (`reach * tanh(h / reach)`),
  so short grass can bend as much as a tall tree's crown.
- **`manifoldx.streaming`: a cell streamer.** `Streamer(engine, cell, build)` keeps the
  cells around a focus point built in a worker (a thread, or a process with
  `workers="process"`) and draws the ring under it, uploading at most
  `uploads_per_frame` cells a frame and dropping and freeing cells that leave
  `ready_radius`. `build(i, j)` returns a `MeshData` relative to the cell's corner (#85).
- **A floating origin.** `Engine.origin` and `Engine.set_origin(x, z)`: every entity draws
  at `pos - origin`. `EntityHandle.transform.pos` reads and writes world positions in
  float64 and stores them relative to the origin in float32, so an entity 50 km out
  draws as precisely as one at the start. `set_origin` shifts the raw Transform rows,
  the camera, the spot light and the manual shadow target, so nothing moves on screen (#85).
- **`EntityHandle.destroy()` frees GPU memory.** It destroys the entity and, at the
  frame's command flush, frees its geometry's vertex and index buffers unless another
  live entity draws the same geometry (#85).
- **`examples/stream_walk.py`** walks a camera across 50 km of streamed noise terrain and
  fails if any frame takes more than twice the median (#85).

- **Rigid GPU skinning in the mesh pass.** `modeling.Mesh(joints=)` gives each vertex a
  joint index (`(N,)` non-negative ints below `manifoldx.MAX_JOINTS = 32`), and
  `to_geometry()["joints"]` carries it as uint32. On an entity spawned from such a
  geometry, `handle.set_joints(matrices)` takes `(J, 4, 4)` model-space matrices
  (column vectors, before the entity's Transform) and uploads only those J*64 bytes; the
  vertex shader moves each vertex by its joint's matrix and its normal by the inverse
  transpose (identity when the 3x3 is singular). Joints not given keep their previous
  matrix, identity until first set, so an unposed entity draws its rest pose. Two
  entities spawned from one geometry pose independently, `set_geometry` keeps the
  joints, and unskinned geometry draws through the same pipelines and buffers as
  before. Every StandardMaterial variant (plain, vertex colours, textured) skins, as do
  BasicMaterial, FlatMaterial and PhongMaterial. Modeling ops that keep vertex count and
  order keep the joints. Example: `examples/skinning.py`, now over a ground plane with a
  sun shadow (#79).
- **Skinned shadows and picking.** The shadow pass poses skinned entities with the same
  joint matrices, so a turned part casts its turned shadow, and the shadow auto-fit
  bounds follow the posed parts. `render_frame(pass_="ids")` labels a posed part where it
  is drawn. Unskinned shadows draw through the same shader and pipelines as before (#79).
- **`set_geometry` refuses joints the entity cannot use.** Giving a mesh with joints to an
  entity spawned from a geometry without them raises `ValueError` instead of dropping the
  joints. A skinned entity given a mesh without joints keeps its joints, as before (#79).
- **Fullscreen on a chosen monitor.** `Engine(fullscreen=True, monitor=...)` takes an index
  into the monitor list or a case-insensitive substring of a monitor's name (`"HDMI"`, a
  model name). With `monitor=None` the `MANIFOLDX_MONITOR` environment variable is read
  with the same rules, then the primary monitor is used, as before. No match raises
  `ValueError` listing the monitors. The window takes the monitor's current video mode.
  `manifoldx.backends.list_monitors()` returns `(index, name, (w, h), refresh, is_primary)`;
  `examples/ui_scale.py --list-monitors` prints it.
  A fullscreen window no longer minimizes when focus moves to another monitor
  (GLFW's `AUTO_ICONIFY` is turned off for it).
- **`engine.gui.scale` and `engine.gui.design_height`: scale the whole GUI.** Widget
  sizes, paddings, radii, borders, font sizes and offsets are UI pixels; on screen they
  are multiplied by `engine.gui.effective_scale(engine.h)`, which is `scale` (default
  1.0) or, when `design_height` is set, `engine.h / design_height`. Layout runs against
  a viewport of `(w / scale, h / scale)`, so a panel placed at the UI viewport's edge
  still touches the screen's edge. Text is rasterized at the scaled font size, so it
  stays sharp; a string too wide for one label-atlas tile at that size is stretched
  instead of cut off. Hit-tests map the pointer by the same scale, so clicks land on
  what is drawn. Example:
  `examples/ui_scale.py` (`--scale`, `+` / `-` live).
- **One engine, many screens: `Engine.remove_system`, `Engine.off`, `Engine.clear_scene`.**
  An app that shows several screens in sequence keeps one `Engine` and one `run()` and
  swaps screens inside it, instead of `quit()` and a new engine per screen, which closes
  and reopens the window. `remove_system(func)` and `off(event, func)` unregister what
  `@engine.system` and `@engine.on(event)` registered; `clear_scene()` destroys every
  entity and GUI panel and keeps systems, handlers, camera, lights, sky, fog and
  environment. All three are safe mid-frame, from a system, handler or async task, and
  raise `ValueError` for something never registered. `examples/screens.py` shows two
  screens as coroutines that wait for Space and tear down.

- **`engine.loading(label)`: load without freezing the window.** A context manager, `with`
  or `async with`, marking work the scene is still waiting for. While a block is open,
  `run()` shows the loading spinner with its label; `render_frame`, `render` and
  `export_gltf` wait for every block to close. `run_blocking` now works under `run()`'s
  loop. uh-twin, loading 24 OBJs and its textures this way: 866 frames drawn during
  startup, the longest gap 1.7 s (#28).

- **glTF export.** `manifoldx.gltf.export_gltf(path, nodes)`, `modeling.Mesh.to_gltf(path)`
  and `Engine.export_gltf(path)` write `.glb` files that Blender, Godot, Unity and three.js
  open: geometry, PBR materials, embedded textures, punctual lights, the camera, and fog,
  shadow, background and environment settings in the scene's extras. Everything glTF cannot
  carry (systems, event handlers, GUI, compute, volumes, point clouds, labels, materials
  without an equivalent) is reported by name, printed, and saved as
  `<name>.export-report.json`; `strict=True` raises `ExportIncomplete` and writes nothing.
  `manifoldx export app.py out.glb` exports any app that ends in `engine.run()` without
  changing it. The output passes the Khronos glTF validator with no errors. Example:
  `examples/gltf_export.py` (#23).

- **`TextureHandle.source`**: the image file `load_texture` read, so exporters can embed it.

- **Mipmapped textures.** `load_texture` builds the full mip chain (box-filtered halvings
  down to 1x1) and samples it trilinearly with 8x anisotropic filtering, so a texture
  tiled across a large mesh no longer turns to crawling noise at a distance (#21).

- **`manifoldx.humanoid`**: posable mannequins. A 15-joint skeleton with anatomical
  angles (`flex`, `abduct`, `twist`, `lean`), a pose library with `base` plus
  overrides, ANSUR II proportions with style presets (heroic, disney, anime,
  fashion, child, toddler, chibi), a grounding forward-kinematics `solve`, and
  `spawn_mannequin`. Example: `examples/humanoid.py`.
- **`Engine.render_frame`**: one still frame headless, as a numpy array, with
  supersampling; `pass_="ids"` returns a per-group label map drawn with the new
  `FlatMaterial`.
- **`Camera.project`**: world points to pixel coordinates and depth.
- **GPU fields — transpile the `Field` algebra to WGSL.** A composed field (`ridged`, `fbm`, `warp`, `remap`, arithmetic, …) now optionally carries a symbolic AST (`Field._ast`, populated additively by the sources/combinators — the numpy CPU path is untouched, 699 tests pass). `manifoldx.modeling.field_to_wgsl(field, name)` walks it and emits `fn <name>(P: vec3<f32>) -> f32` plus a shared value-noise prelude, so the **same developer-composed field** can drive a shader instead of only baking on the CPU. `examples/gpu_field.py` displaces a flat 256² plane into detailed terrain live in the vertex shader (normals from finite differences of the same field) — no CPU displacement, no baked heightmap. The WGSL noise is a decorative value-noise approximation (not a bit-for-bit CPU Perlin match); `worley`/AST-less fields raise `FieldNotTranspilable`. Design: `.knowledge/analysis/2026-07-28-gpu-fields-design.md`.
- **Procedural cloud deck (`CloudMaterial`) + sun-aware starfield.** A new alpha-blended `"cloud"` pipeline subtype (reuses the glow alpha-over target) renders a cloud shell whose coverage comes from in-shader fbm value-noise sampled in object space — so slowly rotating the shell drifts the clouds over a static planet. Sunlit tops, cool shadowed undersides at the terminator, dark at night; gaps `discard` to transparent. `manifoldx.sky` gains `StarfieldMaterial`, a point-sprite star material that reads the lighting globals and fades stars out under a daylit sky — altitude-gated so space stays starry and the night side keeps its stars (`starfield(..., ground_radius, atmo_top)` opts in). To let point/sprite shaders read the lighting block, the sprite pass now binds the full globals buffer (shaders using only the compact prefix are unaffected). `demos/planet_dive.py` gains a drifting cloud deck above the peaks and the day-faded starfield.
- **Physically-based atmospheric scattering (`AtmosphereScatteringMaterial`).** A single-scattering Rayleigh + Mie sky rendered on a shell around the planet: the fragment shader raymarches the view ray (16 primary + 8 secondary sun samples), accumulating optical depth against an analytic ray-sphere atmosphere and combining a Rayleigh phase (`1+cos²θ`) with a forward-peaked Henyey-Greenstein Mie phase, then tonemaps with an exposure control. The **whole sky is now physically derived from the sun angle** — vivid blue by day, warm sunset limb near the terminator, and it goes genuinely **dark at night so the stars reappear** (the secondary march returns full shadow once the sun drops below the local horizon). Additive `"scatter"` pipeline subtype (the renderer gained a `_scatter_target`, one/one blend, no cull, depth-write off — following the existing glow/label blended-pass precedent). 16-byte `vec4` uniform (`[Rg, Ra, intensity, exposure]`). `demos/planet_dive.py` now uses it in place of the old fresnel-glow shell + fake altitude-driven day/night fog: one shell provides the limb from space *and* the full sky at the surface, and the equatorial orbit carries the camera through day into a starry night for free. Design: `.knowledge/analysis/2026-07-28-planet-dive-slice-design.md`; next is aerial perspective (scattering applied to distant terrain) and GPU fields for near-surface detail.
- **Reusable space/planet look: sun-aware materials, water, stars.** All react to the sun via the existing `Globals.sun_direction`/`sun_color`, so a moving sun gives a cohesive **day/night** cycle for free.
  - **`AtmosphereMaterial` is now sun-aware** — the halo glows on the **day limb**, fades on the **night side**, and warms to a **sunset tint** near the terminator.
  - **`WaterMaterial`** (new) — day/night water shading, fresnel sky-reflection at grazing, and a sharp sun **glint**; unlit (16-byte `vec4` uniform: deep colour + fresnel power).
  - **`manifoldx.sky.starfield(engine, count, radius)`** — a reusable point-sprite star backdrop for any space scene.
  - `demos/planet_dive.py` now uses all three plus a swept sun (day→dusk→night), and a `starfield` backdrop.
- **`AtmosphereMaterial` + planet-dive demo.** A new unlit **fresnel rim-glow** material (`pipeline_subtype == "glow"`) renders an atmosphere-halo shell; the renderer gained a small "glow" pipeline branch (alpha blend, depth-write off, no cull) following the label/volume/gui blended-pass precedent. `demos/planet_dive.py`: fly from orbit to the surface of a procedural planet — signed terrain displaces an icosphere into continents + ocean basins (biome-colored), a glossy ocean sphere fills the basins, the atmosphere shell gives the limb halo from space, and a frame-paced descent drives fog + background from space-black to sky-blue. The planet mesh is generated once in a worker process (`submit_process`). First Tier-1 slice of the planet arc (design: `.knowledge/analysis/2026-07-28-planet-dive-slice-design.md`); next is GPU field evaluation for near-surface detail.

- **Geometric modeling v1 (foundation + deformers)** — new `manifoldx.modeling` subpackage: a host-side, numpy-first procedural geometry layer that builds and transforms meshes in code and bakes to the existing geometry dict — no renderer or ECS changes. Core is the immutable `modeling.Mesh` value type (`positions (N,3)` / `faces (M,3)` triangles / lazy `normals` / optional `uvs`) with area-weighted `recompute_normals()`, `with_positions()`, and the single integration seam `to_geometry()` ↔ `from_geometry()` (`modeling.Mesh` is namespaced and never re-exported at top level, so the ECS `Mesh` component keeps that name). Primitives `icosphere` (subdivided icosahedron) / `box` / `plane` / `cylinder` / `torus` generate a `Mesh`. Seeded, deterministic noise fields `noise.perlin` and `noise.fbm` (permutation-table gradient noise; `_resolve_rng`-compatible seeding, same seed ⇒ identical output) drive displacement. Four deformers compose fluently as `Mesh -> Mesh` methods: `displace(field, amount, along="normal"|vector)`, `twist(angle, axis)`, `bend(angle, axis, along)`, `taper(factor, axis)`, each recomputing normals. **Batch 2 — programmatic sculpt brushes:** a radial `Falloff` region model (`smooth`/`linear`/`constant` profiles), cached one-ring vertex `adjacency` (CSR), and the brushes `draw` (along region normal), `inflate` (per-vertex normal), `pinch` (toward center), `flatten` (project region to its best-fit plane), and `smooth` (Laplacian relaxation, global or local via a falloff gate) — all `Mesh -> Mesh`, no viewport. Demo `examples/modeling_asteroid.py` builds a procedural asteroid — `icosphere(4).displace(fbm).twist(...).taper(...).smooth(...)` — PBR-lit and rotating (renders to MP4 via `--render`). **Batch 3 — topology:** `subdivide(iterations, scheme="midpoint"|"loop")` (midpoint linear + Loop smooth subdivision with shared edge vertices), `extrude(face_mask, distance)` (raise a face region along vertex normals with stitched side walls), and `decimate(grid)` (grid vertex-clustering; QEM deferred). `recompute_normals` now gives degenerate/fold vertices a fallback direction so every normal is unit-length. Second demo `examples/modeling_topology.py`: a shadow-casting sun over a gallery of three orbiting, spinning procedural forms — a Loop-subdivided noise blob, an icosphere with an extruded crystal facet, and a decimated low-poly asteroid. **Batch 4 — booleans (CSG):** `union`/`difference`/`intersection` on `Mesh` values via a pure-numpy BSP-tree algorithm (faithful port of the classic Naylor-Amanatides-Thibault / csg.js approach — reimplemented, no vendored `manifold3d`), plus a `translate` convenience. Latent inward winding in the `box` and `torus` primitives was fixed (now CCW-from-outside / outward normals), required for correct CSG orientation. Third demo `examples/modeling_boolean.py`: a shadow-casting sun over three orbiting, spinning CSG solids — a rounded die (box ∩ sphere), a square-drilled ball (sphere − bar), and a fused two-sphere bond (∪). The Manifold-style exact-predicate kernel remains the deferred robustness tier. Design: `.knowledge/analysis/2026-07-28-modeling-booleans-v1-design.md`. This completes geometric-modeling v1 (batches 0-4).
- **Free-form deformation (FFD).** New `manifoldx.modeling.FFD` (and `Mesh.ffd(resolution)`): a Bezier control lattice embedding a mesh over its bounding box. Move the `(I, J, K, 3)` `points` array, call `apply()` to get the warped `Mesh`; an unmoved lattice reproduces the original exactly. Demo `examples/modeling_ffd.py` oscillates the lattice per frame and writes the deformed vertices straight into the live GPU vertex buffer (FFD preserves vertex count), producing a real-time breathing mesh whose cast shadow warps with it — the first example of animated *geometry* (not just transforms) in the repo.
- **Composable field algebra (`manifoldx.modeling.fields` + `Field`).** A scalar field is a callable `(points (K,3)) -> (K,)`; the new `Field` type wraps it with operator overloading (`+ - * /`, unary `-`, scalar coercion) and combinators (`mix`, `minimum`/`maximum`, `clamp`, `remap`, `abs`, `power`, `scale`/`bias`, and domain-warp `warp(amount, fx, fy, fz)`) so developers compose their own terrain/deformation fields from primitives — no opinionated terrain generator. Sources: `perlin`/`fbm` (moved here from `noise`, now returning `Field`; `noise` is a back-compat shim), `ridged` (mountain ridges), `billow` (dunes), `worley` (cellular/Voronoi, `f1` or `f2f1` for cracks), `constant`, `coord` (axis ramps/masks), `distance` (radial/island masks). Because a `Field` is callable, `Mesh.displace(field)` consumes it unchanged. Demo `examples/modeling_fields.py`: a developer-composed terrain (ridged + fbm, domain-warped, island-masked) displacing a plane, lit + shadowed. First of four composable-primitive sub-projects for PCG (fields → colour → curves/sweep → scatter). Design: `.knowledge/analysis/2026-07-28-modeling-fields-v1-design.md`.
- **Per-vertex colour (`color_by` + `Gradient`).** Second PCG primitive: shade a mesh by any field. New `Gradient([(pos, "#rrggbb"), ...])` maps a scalar to RGB (piecewise-linear, clamped; accepts `#rgb`/`#rrggbb`/`(r,g,b)`). `Mesh` gains a `colors` attribute, `with_colors(colors)`, and `color_by(field, gradient)` (samples the field at each vertex → gradient → per-vertex colour); colours ride through deformers and are emitted by `to_geometry()`. On the GPU, `StandardMaterial(vertex_colors=True)` selects a new `"vcolor"` pipeline subtype: geometry interleaves `[pos, normal, colour]` (stride 36) and a WGSL variant tints the PBR albedo by the vertex colour across all light paths (point/sun/spot/IBL/ambient) — full lighting + shadows preserved. Additive behind the new subtype; the scalar and textured paths are untouched. Demo `examples/modeling_colored_terrain.py`: the composed terrain coloured by a developer-defined height gradient (water→sand→grass→rock→snow), lit + shadowed. Design: `.knowledge/analysis/2026-07-28-modeling-color-by-field-design.md`.
- **`engine.background_color`** — the render-pass clear color is now a settable `(r, g, b)` attribute (default unchanged), so scenes can match a fog/horizon color.
- **Real-time terrain streaming + supporting primitives.** Amortizes CPU terrain generation off the render loop (~1.7 → ~30 fps): world-aligned patches are generated on a background thread and swapped into a fixed pool of recycled mesh slots. New reusable pieces:
  - **`@engine.background` + `Task`** — decorate a pure function to run it on the engine's worker thread; the call returns a `Task` (`.ready`, `.result`, `.wait()`, `.on_ready(cb)` with callbacks fired on the main thread at the frame boundary).
  - **`engine.submit_process(fn, *args) -> Task`** — run a heavy CPU-bound `fn` in a separate *process* (spawn context) so it shares no GIL with the render loop. A patch takes ~1 s of numpy, which holds the GIL through its Python-level parts and would freeze a threaded render loop for ~0.9 s; the process worker drops the worst render-thread stall to ~5 ms. The streaming demo uses this (its patch builder is a top-level function; setup is under `if __name__ == "__main__":`).
  - **`engine.spawn(...)` returns an `EntityHandle`** (was `None`; backward-compatible) with a `.transform` proxy (read/write pos/rot/scale) and **`.set_geometry(mesh)`** — overwrites the entity's vertex buffer in place (requires equal vertex count), hiding the raw buffer poke. `examples/modeling_ffd.py` and `modeling_ridge_flyby.py` were refactored onto it.
  - **`engine.enable_fog(start, end, color=None)`** — linear distance fog for `StandardMaterial`, added to the base fragment shader (inherited by scalar/textured/vcolor variants) and applied after tonemap so full fog exactly matches `background_color` for a seamless horizon (`Globals` grew 416→432).
  - **`Field.shift(offset)`** — sample a field at translated coordinates (turns "the patch at world Z" into one line).
  - Demo `examples/modeling_terrain_stream.py`. Design: `.knowledge/analysis/2026-07-28-terrain-streaming-and-background-tasks-design.md`.
- **Infinite ridge flyby demo** — `examples/modeling_ridge_flyby.py`: a real-time procedural mountain range with distance fog, built entirely from the composable primitives. The terrain field is sampled through a forward-scrolling window each frame and poked into the live vertex buffer (constant vertex count), so an endless ridge system flows toward a fixed camera; per-vertex color is a height gradient mixed toward the horizon color by camera distance (CPU-composed fog — no fog shader), with `engine.background_color` set to match so distant peaks dissolve into the sky. Design: `.knowledge/analysis/2026-07-28-geometric-modeling-v1-design.md`. Plans: `.knowledge/plans/2026-07-28-geometric-modeling-v1-plan-1-foundation-deformers.md`, `.knowledge/plans/2026-07-28-geometric-modeling-v1-plan-2-sculpt-brushes.md`.

- **`examples/labyrinth_fps.py`** — a first-person flashlight demo: walk a procedurally-generated maze in the dark. Showcases `engine.input` (WASD grounded movement + right-drag mouse look) and the spot-light + spot-shadow subsystem together. Recursive-backtracker maze rendered as one shared cube mesh per wall cell (two draw batches), grid-based axis-separated wall collision (slides along walls), a `SpotLight` mounted on the camera that casts shadows down the corridors, and a wide FPS FOV. Under `--render` a scripted forward-only flythrough (BFS solution path) walks the corridors so the headless MP4 needs no live input.

- **Spot lights + spot shadows (flashlight)** — `engine.set_spot(SpotLight(...))` gives `StandardMaterial` a flashlight: a cone with smooth inner→outer angular falloff, inverse-square distance attenuation, and a range cutoff. Like `DirectionalLight` before it, `SpotLight` gains its first real consumer (it was previously unused). When shadows are enabled the spot is the caster (priority over the sun): `light_view_proj` becomes a **perspective** projection (vertical FOV = 2·`outer_angle`) from the spot's position/direction (`manifoldx.shadow.compute_spot_light_view_proj`), the depth-only pass renders from the spot's POV, and the fragment samples it to shadow the spot term. A `shadow_caster` field (0=none/1=sun/2=spot) in `Globals` routes the slope-scaled bias + shadow factor to the correct light. `Globals` grew 352 → 416 bytes (spot position/range, direction/cos-inner, color/intensity, cos-outer, caster). Demo: `examples/spot_demo.py` — a flashlight sweeping a dark scene, every object's shadow swinging around the floor as the light orbits.

- **Shadow mapping v1 (directional)** — `StandardMaterial` now casts and receives hard shadows from a single directional sun. New `engine.set_sun(DirectionalLight(...))` adds a real directional term to the PBR shader — previously `DirectionalLight` was silently packed into the point-light array and never worked as a directional light, so this gives the class its first correct consumer. `engine.enable_shadows(target, extent, resolution, near, far, bias)` makes the sun cast: a new depth-only shadow pass (`src/manifoldx/render/passes/shadow.py`, encoded before the main pass in `Engine._draw_frame`) rasterizes all mesh geometry from the sun's POV into a `depth24plus` shadow map, which the `StandardMaterial` fragment shader samples through a comparison sampler (`textureSampleCompareLevel`, hardware 2×2 PCF) to attenuate the sun term. Fragments outside the sun's ortho frustum stay lit; a constant `bias` kills shadow acne. Light-space matrix math is the pure-numpy `manifoldx.shadow.compute_light_view_proj` (ortho × look-at, wgpu z∈[0,1]). The `Globals` uniform grew 240 → 352 bytes (`light_view_proj` mat4 + sun direction/color/intensity + `shadow_enabled`/`shadow_bias`/`shadow_map_size`); the shadow map binds as `@group(2)` on the `StandardMaterial` pipeline, with a 1×1 placeholder depth texture when shadows are off so the pipeline layout is always satisfied. Point-light and IBL contributions are unshadowed in v1 (ambient does not cast a directional shadow). Only `StandardMaterial` casts/receives; sprite/label/volume/axis/GUI paths are untouched. Demo at `examples/shadow_demo.py` (four objects — three spheres + a cube in metallic/matte/polished materials — orbiting and bobbing at different heights over a central pillar and floor, their cast shadows sweeping the ground, climbing the pillar, and crossing each other). **PCF soft shadows (VS2):** `enable_shadows(pcf_radius=...)` averages a `(2r+1)²` grid of comparison taps around each shadow lookup for soft edges — `0` = a single hard tap (VS1 behaviour), `1` = 3×3 (default), `2` = 5×5, etc.; the radius rides in the previously-unused `Globals` pad slot, so the uniform size is unchanged. **Auto-fit + slope-scaled bias:** `enable_shadows(auto_fit=True)` (the default) fits the shadow frustum to the scene's bounding sphere every frame (computed from each geometry's local AABB transformed by its instances), so `target`/`extent`/`near`/`far` no longer need hand-tuning — tighter frustum, sharper shadows; pass `auto_fit=False` to drive the box manually. The fragment bias is now slope-scaled (`bias·(1 + 4·(1−N·L))`), so grazing surfaces stop self-shadowing (acne) without peter-panning on faces toward the sun. Spot/point-light shadows and cascaded maps remain follow-ups. Design: `.knowledge/analysis/2026-07-17-shadow-mapping-v1-design.md`. Plan: `.knowledge/plans/2026-07-17-shadow-mapping-v1-plan.md`.

- **IBL v1 (Image-Based Lighting)** — split-sum physically-based environment lighting for `StandardMaterial`. New `manifoldx.ibl` module: `EnvironmentMap` dataclass with builders `from_color(rgb)`, `from_sky(zenith, horizon, ground)`, `from_image(path, exposure)`, and `from_hdr(path)` (Radiance RGBE decoder included). `engine.set_environment(env_or_preset)` activates an environment; built-in presets: `"studio"`, `"sky"`, `"neutral"`, `"dark"`. On first use, `_precompute()` runs CPU-side: equirectangular → 128 px cubemap, 64 px irradiance map (256 cosine-weighted samples), 8-mip prefiltered specular chain (512 GGX-importance samples per mip), all stored as float16. Pre-baked 512×512 Smith-GGX BRDF LUT ships as a binary asset (`src/manifoldx/assets/ibl/brdf_lut.npy`; regenerable via `scripts/gen_brdf_lut.py`). The `StandardMaterial` WGSL shader was extended to add a correct `proj: mat4x4<f32>` field (fixing a latent `camera_pos` offset bug), `@group(1)` IBL texture bindings, `fresnelSchlickRoughness`, and a split-sum IBL block that additively combines diffuse irradiance + specular split-sum with existing directional/point lights. The `Globals` uniform was extended from 224 → 240 bytes to add `ibl_intensity f32`, `ibl_enabled u32`, and 8 bytes of pad. Optional skybox: new `src/manifoldx/render/passes/skybox.py` draws a fullscreen triangle at depth=1.0 (far plane) sampling the prefiltered map with Reinhard tone map + sRGB gamma; activated when `env.show_skybox = True`. Demo at `examples/ibl_demo.py` (4×4 metallic-roughness grid, press 1/2/3 to cycle environments, S to toggle skybox). Design: `.knowledge/analysis/2026-07-05-ibl-v1-design.md`. Plan: `.knowledge/plans/2026-07-05-ibl-v1-plan.md`.

- **Textured PBR v1** — first slice of texture support in the core mesh-PBR path. New `manifoldx.textures` module exposes `load_texture(engine, path) -> TextureHandle`, backed by a per-engine `TextureRegistry`; images decode via Pillow and upload as `Rgba8UnormSrgb` (so sampling auto-decodes sRGB to linear in the shader). `StandardMaterial` accepts a new `albedo_map: TextureHandle | None` kwarg; when set, the fragment shader samples `textureSample(albedo_tex, albedo_sampler, uv).rgb` instead of using the scalar `material.albedo`. Geometry dicts now carry an optional `"uvs"` key; `sphere()` and `plane()` emit UVs by default. `GeometryRegistry.create_buffers` interleaves `[pos, normal, uv]` (stride 32B) when UVs are present, else falls back to `[pos, normal]` (stride 24B); the mesh-pass pipeline now reads the geometry's actual buffer stride so a scalar `StandardMaterial` bound to a UV-bearing geometry still renders correctly. The pipeline cache key includes `material_subtype` so scalar and textured `StandardMaterial` coexist in one scene without collision. Trying to attach a textured `StandardMaterial` to a geometry without UVs raises `MaterialGeometryMismatchError` at pipeline-cache time. New `manifoldx.assets.obj.load_obj` (re-exported as `manifoldx.load_obj`) parses Wavefront OBJ files with all four face-line forms (`f v`, `f v/vt`, `f v/vt/vn`, `f v//vn`); fan-triangulates polygon faces; rejects mixed forms and negative indices with line-numbered errors. First bundled binary assets: `examples/assets/teapot/teapot.obj` (UV-mapped Newell teapot, ~93 KB) and a procedural `teapot_albedo.png` regenerable via `scripts/gen_teapot_albedo.py`; and `examples/assets/sun_earth_moon/{earth,moon}.jpg` (public-domain NASA imagery). Two new demos: `examples/teapot_demo.py` and `examples/sun_earth_moon_demo.py`. Plan: `.knowledge/plans/2026-06-09-textured-pbr-v1.md`. Design: `.knowledge/analysis/2026-06-09-textured-pbr-v1-design.md`.

- **GUI v1** — in-engine retained-mode GUI for sim controls + HUD readouts. New `manifoldx.gui` package: `Panel`, `Text`, `ValueDisplay`, `Button`, `Slider`, `Toggle`; stack + flex layout; theme + named-class styling with per-widget overrides. Rendering via a new `gui` render pass at the end of the order, using `RectMaterial` (WGSL signed-distance rounded rect) for backgrounds and the existing `LabelTextureAtlas` for glyphs (no new font dependency). Interaction via `_GuiBridge` — subscribes to pointer events on the bus, hit-tests the widget tree top-down, dispatches per-widget state mutations, and exposes `engine.gui.pointer_over_gui` for cooperative consume by user systems. Slider supports drag capture and emits `:change` continuously plus `:commit` on pointer-up. Example at `examples/gui_demo.py`.

### Changed

- **glTF export writes a skinned entity posed.** `Engine.export_gltf` writes a skinned entity
  with its joints and its current pose: `JOINTS_0` and `WEIGHTS_0` on its mesh, a skin with
  identity inverse binds, and one joint node per joint under a node carrying the entity's
  `Transform`. Viewers that ignored the joints now pose the figure, and the file holds one
  more node per joint and per skinned entity. Lifts the 0.10.0 known limitation (#82).

- **A pointer capture per pointer.** `PointerEvent.pointer_id` is new, always 0 on the desktop;
  the GUI keeps one pointer capture per pointer, so a pointer going down no longer ends another
  pointer's drag on a Slider or a Joystick (#140).
- **The arrow keys step a clicked Slider.** After a pointer goes down on a Slider, the arrow keys
  step it by a twentieth of its range (Left and Down lower it, Right and Up raise it), emitting
  change and commit, until a pointer goes down outside every widget. An app that steers with the
  arrow keys and has a Slider sees the Slider move too once the user has clicked it (#140).

- **A query returns only the entities that have its components.** A system's query returns
  only the alive entities that have every component it names, as its docstring said; before,
  it returned every alive entity. An app whose query names a component some entities lack no
  longer writes their rows, and `len(query)` counts only the matching entities (#140).
  `examples/point_cloud.py` kept its velocities per store slot, star included; they are now
  per query row.

- **Key names no longer depend on Shift, and Space is `"Space"`.** A letter pressed with
  Shift held arrived as `"W"`, so `is_pressed("w")` was false while sprinting. Letters are
  now lowercase on `KeyEvent.key` and in the polling state, and the space bar, which
  rendercanvas reports as `" "`, is `"Space"`; `is_pressed(" ")` still works. Code that
  compared `KeyEvent.key` with an uppercase letter or with `" "` needs the new name (#18, #73).

- **`render_to_texture` views skip the world's fog unless `fog=True`, and `background`
  is sRGB** (a hex string or an RGB/RGBA tuple, like GUI colours; an RGB tuple used to
  fail inside wgpu). Fog grows with distance from the camera, so a minimap camera high
  over the ground came out pale.

- **GUI colours are sRGB, so a hex shows as itself.** `bg`, `border_color`, `fg` and
  `tint` are decoded from sRGB like material colours and SVG fills; before, a hex was
  drawn as linear light and showed lighter (`#808080` came out as about `#bcbcbc`).
  Every GUI looks darker and truer. `engine.gui.exact_colors = False` keeps the old look.

- **The GUI draws in the widget tree's order.** Rects, images and text interleave by
  root panel and child order, so a later panel covers an earlier one's text and images.
  Text used to be drawn above every panel.
- **A textured `StandardMaterial` honours the albedo map's alpha.** Texels under 0.5
  alpha are cut out, and with `opacity < 1` the rest blends by its alpha. A texture with
  an alpha channel that rendered opaque before now shows its holes; opaque textures are
  unchanged. Shadows are still cast by the whole mesh.

- **Shadows fade out over the outer 15% of the shadow map** instead of stopping at its edge,
  so the edge of what the map covers never shows on screen as a line.

- **`startup` fires on `run()`'s first frame, inside the running loop**, instead of before
  the loop starts. Sync handlers still run before anything is drawn; async handlers now
  run on the loop that `run()` drives, between frames. Before, they were scheduled on the
  engine's private loop, which nothing pumps while the window is open, so they never ran
  (#28).
- **The environment's precompute no longer blocks the window.** It runs on the background
  worker, started by the first frame that needs it; until it is done the scene draws
  without the environment's light and a centred spinner reads "loading lighting...", so
  the window keeps answering instead of showing "not responding". `render_frame` and
  `render` wait for it, so offline frames always carry the full lighting. Measured with a
  gradient sky: 5,589 frames drawn during a 26.5 s precompute, median 4 ms (#25).

- **Examples reorganized into `examples/` (focused) + `demos/` (show-off).** `examples/` files each demonstrate one feature with minimal code (documentation), named by the feature (`pbr.py`, `shadows.py`, `fields.py`, `booleans.py`, `ffd.py`, `color_by.py`, `topology.py`, `deformers.py`, `textures.py`, `gui.py`, `volume.py`, `point_cloud.py`, `axes.py`, `spotlight.py`, `ibl.py` — dropping the `*_demo` suffix). Multi-system, game-like/sim-like showcases moved to a new `demos/` folder (`terrain_stream.py`, `labyrinth_fps.py`, `sun_earth_moon.py`, `smoke.py`). Each folder has a `README.md`; the long-term demo roadmap (Sebastian Lague-inspired) is at `.knowledge/analysis/2026-07-28-demo-roadmap-lague-inspired.md`. Removed the superseded naive `modeling_ridge_flyby.py` (replaced by `demos/terrain_stream.py`).

### Performance

- **`load_obj` and the HDR decoder no longer run Python per line or per byte** (#117):
  a file whose faces are all triangles of one form is parsed with numpy in one pass over
  its bytes and deduplicated on one int64 key; quads, n-gons, mixed forms and indented
  lines still go to the line parser, with the same output and error messages. The RGBE
  decoder appends runs to one bytearray per scanline and scales by a lookup table, with
  the same floats. A 300k-triangle v/vt/vn OBJ loads in 0.99 s instead of 4.05 s, and a
  2048x1024 HDR decodes in 136 ms instead of 829 ms.

- **The mesh pass reuses its bind groups and uploads a material's uniform only when it
  changed** (#113): bind groups are cached per material and rebuilt when a bound buffer
  grows or a texture is swapped, the shadow-map one when its view changes, and each
  material's uniform row is built once per frame (`get_data(1)`, not one row per
  instance) and written only when its bytes differ. The sprite, label and axis passes
  also build one row. A frame of 5000 entities with a geometry each drops from 2040 ms
  to 421 ms; 100k vertex-coloured instances from 212 ms to 153 ms.

- **Compute dispatches move only the slots ever used, not `max_entities` rows** (#114):
  `EntityStore` tracks a high-water mark, one past the highest slot ever spawned into, and
  `ComputeRunner` uploads and reads back only the rows below it. Kernel results are
  unchanged. With 1000 live entities and one kernel reading and writing `Transform` and a
  3-float component, a dispatch at `max_entities=1_000_000` drops from 189 ms to 3.2 ms.

- **`modeling` operators on large meshes avoid `np.unique(axis=0)` and `np.add.at`**
  (#116): `Mesh.adjacency()`, `subdivide` and `decimate` dedup edge and cell rows through
  one int64 key and a 1-D `np.unique`; `smooth`, Loop subdivision and `decimate` sum with
  one `np.bincount` per axis. Outputs are unchanged. On a 327 680-face icosphere,
  `adjacency()` drops from about 4.0 s to 1.2 s, `subdivide(1)` from 2.2 s to 0.26 s,
  `decimate(64)` from 0.30 s to 0.06 s and one `smooth` iteration from 189 ms to 97 ms.

- **System field access no longer copies the component or indexes with `np.ix_`**
  (#118): `q[Comp].field` skips a fancy-indexed copy of the whole component that was
  never used, and field reads, writes and the deferred write in `CommandBuffer` index
  with `arr[indices, a:b]`. A system doing `q[Transform].position += c` plus the command
  flush drops from about 13-16 ms to 8.5 ms at 100k entities.

- **Model matrices are computed once per frame, and the sun's auto-fit shadow bounds
  once per frame from the matrix columns** (#112). Every pass marked all transforms dirty
  before reading them, so each frame recomputed every matrix 2 times, 4 with sun shadows;
  now `RenderPipeline.run` recomputes them and the passes until `end_frame` reuse them.
  The bounds take a mesh's box under its affine model matrix as `centre ± |R|·half`
  instead of multiplying 8 corners per instance (skinned geometry keeps its per-joint
  boxes). Reads outside a frame still recompute. On a 4-CPU machine without a GPU, a
  frame of 100k static meshes with shadows drops from about 450 ms to 190-230 ms.

- **`EntityStore.spawn` assigns slots in one vectorised step** (#107): free list first,
  then the lowest dead slots via `np.flatnonzero`, instead of scanning the alive mask
  once per entity. `spawn(200_000)` drops from about 60 s to 0.03 s. Running out of
  slots now raises before any slot is taken, and the error names
  `Engine(max_entities=...)`.

- **The scene pass routes entities without a per-entity Python loop** (#111). Axis,
  label, volume, sprite or mesh depends only on the material, so each distinct material is
  classified once and entities are grouped by sorting; the passes take index arrays.
  Frustum culling now runs only on entities with geometry (it tested every point sprite
  and kept them all), and the id pass of `render_frame` builds its mesh mask per material.
  On a 4-CPU machine without a GPU, a frame of 100k point sprites drops from about 450 ms
  to 110 ms, of 100k static meshes from about 550-730 ms to 170-240 ms, and an id pass
  over 100k meshes from about 1.1 s to 0.65 s.

- **`Camera.get_view_matrix` uses plain float arithmetic** instead of `np.cross` on
  3-vectors, which cost more than the math; the renderer asks for it several times a
  frame.

- **Frustum culling looks up bounds once per geometry and material, not per instance,
  and the scene pass skips culled meshes before its per-entity loop** (when no axis,
  label or volume entities exist). A camera that sees a corner of a large scene, such
  as a render_to_texture view, no longer walks every entity in Python.

- **The Streamer uploads the ready ring hidden, a few MB a frame.** Its worker also packs
  each cell's vertices (`resources.pack_vertices`, shared with `create_buffers`), and the
  frame copies at most `upload_bytes_per_frame` (2 MB by default) of them to the GPU,
  nearest cell first, so a cell that comes into view is already on the GPU and drawing it
  builds nothing on the frame. Cells keep their buffers until they leave `ready_radius`
  (`EntityHandle.destroy(free_geometry=False)`). Cells drawn before the engine has a device
  still get their buffers from the render pass. A 24 MB cell no longer stalls the frame
  where it first draws.
- **Mesh pipelines are keyed by vertex layout, not by geometry.** Every new geometry
  compiled its own pipeline, about 50 ms and 2.9 MB each on a software renderer. Meshes
  with the same layout and material now share one, so 1,000 new meshes went from about
  50 s to 0.7 s, with flat memory (#85).

- **Interactive windows are no longer held to 30 fps.** rendercanvas caps a canvas at
  `max_fps=30` unless told otherwise, and the desktop canvas never told it. `Engine` and
  `backends.get_desktop_canvas` take `max_fps` (default 240), so vsync sets the pace: a
  light scene went from 29 fps to the display's 60 (#75).
- **GUI text is measured once per string and size.** The painter and the gui pass measure
  every `Text` widget every frame, and `LabelTextureAtlas.measure_string` opened the font
  file each time. The measurement and the font are now cached. A scene with ~70 labels
  spent ~19 ms a frame there and went from ~25 fps to ~56 fps (#75).
- **`Mesh.recompute_normals` is 2-5x faster.** It scatters face normals with one
  `np.bincount` per axis instead of `np.add.at`, gathers each corner once and takes the
  cross product by columns. The normals match the old ones to 1e-6. A mesh posed on the
  CPU every frame pays this cost once per frame.
- **IBL precompute: about 3.5x faster, and cached on disk.** The prefiltered chain's mip 0
  is roughness 0, a mirror, so it is now the radiance cube itself instead of 50 M GGX
  samples (about 70% of the work). The result is also cached in
  `$MANIFOLDX_CACHE_DIR/ibl/` (default `~/.cache/manifoldx/ibl/`), keyed by a hash of the
  radiance data and the precompute parameters, so a second run with the same environment
  skips it. A gradient sky: about 23 s to 6.5 s cold; uh-twin's first frame 26.6 s to
  11.7 s cold, 4.1 s warm. Tests point the cache at a temporary directory (#25).

### Fixed

- **`demos/labyrinth_fps.py` exports to the web.** The flashlight is a `SpotLight` whose
  position and aim read the camera rather than frame-code writes the page cannot apply, and
  the controller keeps the wall grid as a flat list, so the whole demo is inside the subset.
  Lights stay setup-only in the page, as `know-how/web-export.md` now says (#155).
- **The test job runs the suite on a software GPU, and lint covers the tests.** CI's `test`
  job installs lavapipe (`mesa-vulkan-drivers`), so the render tests run instead of
  skipping: it runs `tests/ --ignore=tests/web`, prints skip reasons with `-rs` and fails
  on a skip that says no GPU adapter. `ruff check` now covers `tests/` too, and the old
  test files' lint errors are fixed.

- **export --web: a member of an engine object outside the page's runtime surface fails the
  export.** Frame code that called an `Engine` method `RUNTIME_SURFACE` lacks — a bare
  `engine.enable_shadows()`, or the same call inside an expression — transpiled to a call the
  page has no member for and threw only there; the export now stops at its file and line, as
  it does for other constructs outside the subset (#171).
- **`set_geometry` moves the shadow auto-fit bounds.** The renderer cached each geometry's
  local bounding box; after `EntityHandle.set_geometry` wrote vertices outside it, a
  `enable_shadows(auto_fit=True)` frustum was still fit to the old box and clipped the new
  shape's shadow. `set_geometry` now mirrors the new positions into the registry and drops
  the cached box (#83).
- **`engine.elapsed` advances under a fixed timestep.** `_compute_dt` returns the fixed
  step without touching the clock, so `engine.elapsed` stayed at its start value and
  `engine.delay()` and `elapsed_at()` never resolved. Each frame now adds its fixed dt
  once its systems have run, keeping the page's rule: 0 before the first step, then one
  dt per simulated step (#168).
- **export --web: instanced meshes are lit with their instance rotation.** The page's fixed
  light for `BasicMaterial` and `PhongMaterial` built the normal from `modelMatrix` alone, so
  every instance of a batch was shaded as if unrotated; it now takes `instanceMatrix` too,
  under `USE_INSTANCING`, as three.js's own chunks do, and rotated boxes match the desktop
  (#172).
- **GUI panels honour all nine anchors on the desktop.** A root panel anchored anywhere but
  top-left painted and hit-tested at the top-left. Paint and hit-test now place it through
  one function, `gui.layout.anchored_slot`, by the page's rule: the offset measured inward
  from the anchored edges and, on a centred axis, rightward or downward; a panel without a
  width or height fits its content, except a top-left one, which still fills the window.
  The loading overlay is a `center` panel instead of an offset recomputed every frame (#13).
- **A query reads the fields of a `Component` subclass.** `q[Velocity].v` raised
  `AttributeError` for a component declared with the `Component` base, the form AGENTS.md
  recommends; the accessor now reads the class's layout.
- **export --web: frame code reads and writes `engine.camera.up`** (#153), which the page
  copies to the three.js camera each frame.
- **export --web: writing a material's field points at `set_material`** (#154) instead of
  asking for an annotation no annotation could satisfy.

- **export --web: handlers and systems registered with decorators export.**
  `@engine.on("name")` and `@engine.system` on a module function, `engine` a module global
  holding the Engine, export as the call forms `engine.on("name")(fn)` and
  `engine.system(fn)` do, where they failed with "the decorator ... is outside the subset".
  Any other decorator still fails the export (#157).

- **export --web: a widget field that reached code writes reaches the page.** A Panel's
  `offset`, a Text's `text` and any widget's `style_overrides` written every frame move,
  rewrite or restyle the HTML GUI as they do the desktop's; the page writes only the CSS
  that changed. A write of any other field of a widget or of `engine.gui` (`visible`,
  `anchor`, `style`, `scale`) fails the export with its file and line instead of differing
  in silence (#147).

- **export --web: every attribute write the page cannot apply fails the export.** A write
  through a value of unknown type (`x.attr = v`, `x.attr += v`) names what to annotate,
  and a write to a field of the engine, `engine.input`, `engine.camera` or any other
  manifoldx object fails as outside the subset unless the page applies it each frame: the
  camera's `position`, `target`, `fov`, `near` and `far`, and the widget fields above.
  `position` and `target` take a float32 vector, `fov`, `near` and `far` a number (#151).

- **`replay_python` takes `app_args`**, the app's `sys.argv[1:]` as `export_web` takes
  them, so an app that branches on its arguments replays the branch its export took (#148).

- **Sets in transpiled code have `add`, `discard`, `remove`, `clear` and `update`.** Each
  returns None, `remove` of a missing item throws KeyError, and an item a JS Set compares
  by identity (a tuple, a list, an object) fails the export (#149).

- **`export --web` types what it used to leave unknown.** An attribute read on a value of
  unknown type fails the export, naming what to annotate, instead of leaving the field out
  of the snapshot; an annotation the transpiler cannot read (a bare `tuple`) gives way to
  the live type (#142). Unpacking a call result takes the callee's return type, and a
  return annotation is read (#143). `tuple[X, ...]` types every item as X (#144). A starred
  item in a list, set or tuple display keeps its items' types (#145).

- **`export --web` treats a global's initializer as setup.** Reach stops at a reached
  global, so a function only its initializer calls is not transpiled, and the globals that
  initializer read stay out of the snapshot (#140).

- **Startup at export, `render_frame`, the glTF export and the replay need a wgpu device,
  not the `[offline]` extra.** The headless canvas no longer goes through the video
  renderer's imageio-ffmpeg check, so an app with a startup handler exports on plain
  `manifold-gfx` (#140).

- **glTF export: `StandardMaterial` opacity.** The base colour's alpha is the colour's
  alpha times `opacity`, and `opacity` below 1 writes alphaMode BLEND, so a transparent
  material is transparent in the browser too (#140).

- **`engine.input.viewport_size` is right from the first frame.** It read (0, 0) until a
  resize event, which a window that is never resized does not send (#17).

- **`Engine.w` / `Engine.h` follow window resizes.** They were fixed at construction, so
  a resized or fullscreen window drew the construction size stretched: the camera aspect,
  the GUI layout and the GUI hit-tests all used the old size. They now track the canvas'
  logical size. A resized window now keeps GUI panels at their pixel size instead of
  stretching them with the window; set `engine.gui.design_height` to get proportional
  scaling. `render_frame` keeps the requested size, since its supersampled canvas is
  larger than the frame.
- **Commands queued between frames are no longer dropped.** The command buffer was
  cleared at the head of each frame, so a `destroy()` or component write from an async
  task resumed by `run()`'s loop between frames never applied. It is now cleared after
  the flush.
- **An entity destroyed twice no longer shares its slot.** `EntityStore.destroy` skips
  dead and repeated indices; before, the slot entered the free list twice and two later
  spawns got the same index.

- **Textured `StandardMaterial` under the sun and the spot.** The textured shader swapped
  in the sampled albedo for point lights, IBL and ambient but not for the sun and spot
  terms, so a textured mesh rendered its plain `color` (usually white, so grey) under
  them (#5). Pixel tests now compare a `#cc2222` texture with `color="#cc2222"` under each.

- **The wheel builds again.** `pyproject.toml` force-included `src/manifoldx/viz/assets`,
  which `packages` already ships, so hatchling added `DejaVuSansMono.ttf` twice and
  refused to build. That broke `uv add` / `pip install` of manifoldx from git.

- Colours were gamma-encoded twice: `StandardMaterial` (and its textured and
  vertex-colour variants), the water and atmosphere-scattering materials and
  the skybox applied `pow(color, 1/2.2)` and then wrote to an sRGB target that
  encoded again. Those shaders now write linear colour and the sRGB target
  encodes once (forced when the preferred format is plain 8-bit).
- Material colours meant two things: an albedo map is uploaded as sRGB and
  decoded, while `color=` was used as linear, so `#cc2222` rendered (111, 50, 50)
  as a colour and (98, 19, 19) as a texture under the same light. Every
  material colour is now sRGB and decoded (`resources.material_rgb`), as in
  three.js and Blender. Together with the gamma fix, scenes render darker and
  more saturated than before. Re-tuning the demos' lighting is tracked in #6.
- `BasicMaterial`'s docstring claimed it was unlit; it applies a fixed light.

- **Offline render was paced by the wall clock, not the video clock.** `render()` computed `dt = 1.0 / fps` under a "use fixed timestep for video rendering" comment but never called `set_fixed_timestep`, so `_compute_dt()` kept taking the wall-clock branch and overwrote `self.elapsed` with real elapsed seconds on every frame — clobbering the `self.elapsed = (frame_idx + 1) * dt` the render loop assigned. Systems reading `engine.elapsed` therefore saw *how long the render had been running*, so a scripted camera path advanced at whatever rate the host GPU happened to achieve. Concretely, a `--render --duration 26 --fps 12` pass on an Intel iGPU produced 312 correct frames containing only ~12 s of animation (the flythrough never left its first phase), and the same script on faster hardware would produce different motion. Every `demos/` flythrough was affected. Fixed by routing the computed `dt` through `set_fixed_timestep(dt)`; regression test in `tests/test_render_clock.py` asserts frame `i` observes `elapsed == i / fps` and that systems receive `dt == 1 / fps`.
- **Mesh-pass per-material uniform buffers** — `mesh_batches` previously keyed only on `(geom_id, mat_type, mat_subtype)`, so two material instances of the same class on the same geometry collapsed into one batch and the first instance's uniform data won. The bug had been silently affecting `pbr_demo.py` for a while (every cube rendered with the first cube's color; every sphere with the first sphere's). Made visible by the new textured-PBR work: Earth and Moon — both textured `StandardMaterial` on `sphere()` — initially rendered with the same texture. Fixed by extending the batching key to include `mat_id` and allocating one material-uniform buffer per `mat_id` in the mesh pass.

- **Fixes found while writing the guide.** Indices given as `(M, 3)` triangles draw every
  triangle (they drew only the first M indices); a camera looking straight down its up axis has
  a view matrix (it divided by zero); `mxv.points` after `mxv.mesh` syncs its own rows (it
  overwrote the mesh's transform) and `AtlasOverflowError` is exported from `manifoldx.viz`; an
  OBJ without normals gets vertex normals and renders (its first frame failed); a pointer event
  after `clear_scene` reaches the GUI (it raised `AttributeError`); `engine.delay` resolves on
  its own frame when a step sum lands one ulp short; a `Color` object works as a material colour.

### Known limitations

- **glTF export writes a skinned entity in its rest pose.** `Engine.export_gltf` writes the
  geometry's rest positions and no joints or skin, and its report does not mention it (#79).
- **IBL skybox: first-frame device lockup on NVIDIA Quadro M2000M / Vulkan** — enabling `env.show_skybox = True` on the *very first* presented frame deadlocks the device (the frame's submit never completes) on this GPU/driver (580.159.03). It is a driver first-frame-init lockup, not a code defect: depth format (`depth24plus` everywhere), depth config, bind group, and pipeline layout are all correct, and the pass renders perfectly once *any* prior frame has been presented — a constant-colour skybox renders, and the mesh IBL path samples the same cube fine. The skybox therefore stays **default-OFF**; `examples/ibl_demo.py` is unaffected because it only toggles the skybox via the `S` key after startup (post-first-frame). The only broken usage is scripting `show_skybox = True` before the first present. No warm-up-frame workaround was added (would mask an old-GPU driver quirk in the shared render loop). Investigation + evidence table: `.knowledge/analysis/2026-07-05-ibl-v1-design.md` (§ "Known limitation: first-frame device lockup").

## [0.9.0] - 2026-05-10

### Features

- **Input layer v1** — keyboard, mouse, and resize events ride the event bus from event-driven-system v1. New `manifoldx.input` module exposes `KeyEvent`, `PointerEvent`, `WheelEvent`, and `ResizeEvent` dataclasses (re-exported from the package root). Handlers register the same way as any other event: `@engine.on("key_down")`, `@engine.on("pointer_move")`, etc. Alongside discrete events, a polling state at `engine.input` exposes Bevy-style `is_pressed(key)` / `just_pressed(key)` / `just_released(key)` for keys and mouse buttons, plus `mouse_pos`, `mouse_delta`, `wheel_delta`, and `viewport_size`. Polling state is updated immediately on every event (so `is_pressed` reads are live), while `just_*` sets and `*_delta` accumulators are this-frame-bound and reset at the new step 2.5 of `_draw_frame`. New `examples/input_orbit.py` (drag-to-orbit + wheel-zoom via events) and `examples/input_fly.py` (WASD fly cam via polling). Design: `.knowledge/analysis/2026-05-08-input-events-design.md`.
- **Event-driven system v1** — new `engine.emit(name, payload)` / `@engine.on(name)` event bus running parallel to the frame loop. Sync handlers run inline at the head of the next frame; async handlers are scheduled on the active asyncio loop (rendercanvas's in interactive `run()` mode, a private fallback in headless `render()` / tests) and progress between draw callbacks. `await engine.tick()`, `await engine.delay(seconds)`, and `await engine.elapsed_at(target)` are the async waiter primitives, plus `await engine.run_blocking(fn, ...)` as the escape hatch for genuinely blocking work. Globals (camera, lights) are written immediately; ECS reads come from a `ReadOnlyView` (mutations route through `engine.commands` / `engine.spawn` / `engine.destroy`). The legacy `@engine.startup` / `@engine.shutdown` / `@engine.update` decorators have been replaced by `@engine.on("startup" | "shutdown" | "frame")`; the `'frame'` event payload carries `dt`, `elapsed`, and `frame`. `engine.quit()` cancels all in-flight async tasks, allowing `try/finally` cleanup to run before the loop closes. Errors from handlers propagate and crash the frame loop in v1 by design. New `examples/event_dolly.py` (async camera dolly with `await engine.delay(...)`) and `examples/event_pulse.py` (sync handler emitting `engine.spawn`). Design: `.knowledge/analysis/2026-05-08-event-driven-system-design.md`.
- **Volume rendering v1** — direct volume rendering (DVR) of a 3D scalar field via fragment-shader raymarching. New `Volume` ECS component (handle into a per-engine `VolumeRegistry`), new `VolumeMaterial` with reusable colormap LUT + per-material 256-sample opacity LUT (built from piecewise stops or a numpy array). `engine.register_volume(numpy_array)` / `engine.update_volume(handle, new_array)` for CPU upload; `engine.bind_compute_volume(...)` is reserved as a v2 stub. The volume pass renders between sprite and label with depth-test on / depth-write off and pre-multiplied alpha blending; a fullscreen-quad fragment shader does ray/box-AABB intersection in entity-local space, then a fixed-step front-to-back composite sampling two LUTs per step. New `examples/volume_demo.py` shows a 64³ Gaussian blob. v1 is unlit and CPU-uploaded; lighting, isosurface/MIP modes, compute-written volumes (`Writes[Volume3D]`), editable transfer-function widgets, and slicing are explicit non-goals deferred to follow-up plans. Design: `.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md`.
- **Compute systems Phase 2 — Python → WGSL shader compiler** — `Compute.compile()`'s default body now traces a typed-Python `def main(self, i)` body to WGSL via `manifoldx.compute.transpile`. Kernel authors write plain Python with PEP-526 annotations on every local; the transpiler emits the bind-group header, helper functions (`fn _<ClassName>_<name>`), and the `@compute @workgroup_size(W) fn main(...)` wrapper. `examples/nbody_compute.py` is now a Compute subclass with a `pair_accel(...) -> vec3` helper method instead of an inlined WGSL string; numerics agree with the Phase-1 hand-written kernel within `rtol=1e-5`.
- **`engine.compute(cls)` validates synchronously** — WGSL is compiled via `device.create_shader_module(...)` at registration time. Errors surface as `ComputeShaderCompileError` (category `wgpu-validation`) before any frame runs, alongside Python-source-level transpiler errors (`unsupported-construct`, `missing-annotation`, `unknown-name`, `type-mismatch`, `implicit-promotion`, `recursion`).
- **Per-Component `_layout` table** — `Component.__init_subclass__` now derives `{field_name: (offset_in_floats, length_in_floats)}` from the existing field annotations. Pre-base-class built-ins (`Transform`, `Mesh`, `Material`) ship explicit `_layout` class attrs. The transpiler reads this table to emit storage-buffer offset arithmetic without component-specific knowledge.
- **`examples/point_cloud_compute.py`** — protoplanetary-disk demo (~10000 colormapped dust particles + central star) ported to a Phase-2 compute kernel. The kernel has no inner loop — per-particle Keplerian gravity around the origin — and writes a *scalar* component (`ScalarValue.value = current speed`), so the GPU-driven colormap pipeline picks up speeds without any CPU round-trip.
- **vec3/vec4 swizzle reads in compute kernels** — `accel.x`, `next_pos.y`, `self.transforms[i].pos.z` and similar `.x/.y/.z[/w]` attribute access on vec3/vec4-typed expressions now lower to WGSL component access, returning f32. Compound forms (swizzle on top of an indexed binding read) work too.
- **`examples/gas_compute.py`** — ideal-gas box-bounce demo ported to a Phase-2 compute kernel. First example to use vec3 swizzle inside a kernel: per-axis bounce conditions read individual components (`if next_pos.x < -self.half_size`). Pairwise elastic collisions are now also on GPU, reformulated to be race-free: each thread `i` surveys all neighbours and accumulates *its own* impulse from approaching overlaps; thread `j` independently computes the equal-and-opposite impulse for itself, so no thread ever writes another thread's velocity. Also the first example to exercise `vec3 / f32` broadcast (collision normal `diff / dist`).
- **Phase-2 DSL is now mypy-clean in kernel bodies** — `vec3`/`vec4` are real classes with arithmetic dunders; math builtins (`dot`, `cross`, `length`, `sqrt`, `pow`, `abs`, etc.) carry proper PEP-484 signatures; `Reads` / `Writes` / `ReadsWrites` are `Generic[T]` with `__getitem__(int) -> T` under TYPE_CHECKING; `Uniform[T]` is a PEP-695 type alias for `T`; `Float` is a TYPE_CHECKING alias for `float`. Both compute examples now type-check at zero errors against `mypy`. Auto-bound sentinel-string uniforms (e.g. `dt: Uniform[float] = "frame_dt"`) keep needing a `# type: ignore[assignment]` per uniform.
- **`examples/smoke_demo.py`** — tileable Perlin-FBM volumetric smoke, exercising the volume pass with a moving camera and a wrap-padded source for seamless looping.

### Refactors

- **`compile()` override is the escape hatch, not the default.** Phase-1 user kernels that override `compile()` continue to work unchanged; only the default body changed.
- **`renderer.py` split by render pass** — `_render_mesh_batches`, `_render_sprite_batches`, `_render_label_pass`, `_render_axis_pass`, `_render_volume_pass` and their pass-private helpers move out of `RenderPipeline` into `src/manifoldx/render/passes/{mesh,sprite,label,axis,volume}.py` as module-level functions taking the renderer as their first argument. `RenderPipeline.render` becomes a thin dispatcher; cross-pass shared state (pipeline cache, `_BatchBuffers`, `TransformCache`, `_get_or_create_pipeline`) stays in `renderer.py`. `renderer.py` shrinks from 1991 → 964 lines; full test suite (392) and `examples/volume_demo.py` smoke-render unchanged.

## [0.8.0] - 2026-05-06

### Features
- **Sci-viz Plan 1 (foundation)** — `manifoldx.viz` subpackage for scientific-visualization primitives. Six built-in 1D RGBA8 colormap LUTs (viridis, magma, plasma, inferno, turbo, gray) precomputed from matplotlib. New ECS components: `PointCloud` (marker), `ScalarValue` (per-particle scalar attribute), `Radius` (per-particle world-space radius). New `ColormapMaterial` maps the per-instance scalar through a 1D LUT in the fragment shader; default unlit, optional `lit=True` Lambert against a fixed view-space light direction.
- **Sprite render path** — Camera-facing point sprites with sphere-imposter fragment shading, scaled by per-instance `Radius`. New `SPRITE_QUAD` built-in geometry. `RenderPipeline` splits batches into mesh and sprite groups; sprite path uploads parallel storage buffers (`transforms`, `scalar_values`, `radii`) per frame and binds a per-cmap LUT texture.
- **`[viz]` extra** — Optional `pillow>=10.0` dependency group, staged for Plan 2 text rendering.
- **Sci-viz Plan 2 (text rendering)** — `manifoldx.viz` adds the `TextLabel` ECS component, the `LabelMaterial` camera-facing-billboard material with depth-test on / depth-write off / alpha-blend on, and the `LabelTextureAtlas` host-side cache that rasterizes strings via PIL (DejaVu Sans Mono bundled, 256×64 RGBA8 tiles, sRGB-correct) and uploads them lazily to a `texture_2d_array` (256-slice cap in v1).
- **Label render pass** — `RenderPipeline.run` now batches `TextLabel + LabelMaterial` entities into a third draw group dispatched after the 3D opaque pass. Pipeline cache key extended with a `"label"` fourth element so the world-anchored label pipeline never collides with sprite or mesh pipelines.
- **`engine.get_label_atlas()`** — lazy accessor for the per-engine atlas, used by the renderer and by user code that wants to register strings up front.
- **Sci-viz Plan 3 (axes, screen anchoring, scale bars, colormap legends)** — `LabelMaterial(anchor_mode="screen")` actually renders in NDC (was a silent fallback in Plan 2); `AxisFrame` component + `AxisMaterial` line material with `anchor_mode` (`"world"` | `"screen"`) + native LineList rendering pipeline. Built-in `axis_line_x` / `axis_line_y` / `axis_line_z` geometries auto-registered alongside `sprite_quad`. Screen-anchored axes give scale bars without a dedicated component.
- **`LabelTextureAtlas.register_colormap_legend(cmap, orientation=...)`** — rasterizes a colormap LUT into one of the 256×64 atlas slices and returns the slot index. Renders via the existing screen-anchored LabelMaterial path — no new material or pipeline needed for legends.
- **`Globals` uniform extended** — `viewport_size: vec2<f32>` (208 → 224 bytes) replaces the v1 `1024.0` calibration constant; labels and screen-anchored primitives scale to actual viewport pixel dimensions on any canvas.
- **`examples/axes_demo.py`** — small standalone demo that exercises every Plan 3 primitive end-to-end: world-anchored axes with billboard end-cap labels, screen-anchored HUD overlay, screen-anchored scale bar, screen-anchored colormap legend. Camera orbits to highlight the world-vs-screen-anchored contrast.

### Refactors (Plan 3)
- **Pipeline factory `_get_or_create_pipeline`** — accepts `line=True` for a LineList-topology, opaque, depth-write-on path. Three bind slots (globals, transforms, material uniform); buffer is 32 bytes to fit `AxisMaterial`'s rgba + anchor_mode + 3 pad uniform.
- **`RenderPipeline.run` routing** — priority `axis > label > sprite > mesh`; new `_render_axis_pass` and dedicated `_axis_batch_buffers` parallel the existing sprite/label paths.
- **`AxisMaterial.pipeline_subtype = anchor_mode`** — world and screen modes get separate pipelines in the cache (parallels Plan 2's `LabelMaterial`).
- **Component base unchanged** — the new `AxisFrame` (2 floats: `extent`, `thickness`) auto-registers via the `Component` annotation pattern, no boilerplate in user code.

- **Sci-viz Plan 4 (Altair-style declarative shim)** — `manifoldx.viz` adds a grammar-of-graphics API on top of the imperative ECS primitives Plans 1-3 shipped:

  ```python
  import manifoldx.viz as mxv

  chart = (
      mxv.points(positions=p, color=mxv.color(s, cmap="viridis"), size=r)
      + mxv.mesh(geometry=sphere(0.3, 16), material=...)
      + mxv.axes(extent=4)
      + mxv.legend(cmap="viridis", title="Speed")
      + mxv.scale_bar(label="2 units")
      + mxv.lights([...])
  )

  @chart.simulate
  def step(dt):
      p[:] += v * dt   # mutate live arrays in place
      s[:] = np.linalg.norm(v, axis=1)

  chart.cli()
  ```

  Six marks (`points`, `mesh`, `axes`, `legend`, `scale_bar`, `lights`), three Channel wrappers (`color`, `size`, `position`) for per-channel config, `+` operator for layered single-scene composition, `@chart.simulate(dt)` decorator for per-frame mutation. The shim holds references to the user's numpy arrays — a per-frame sync system copies them into ECS storage so live mutation propagates to the GPU. Escape hatch: `chart.engine` returns the underlying Engine for users who need to drop down to imperative ECS ops mid-design (e.g. camera config, custom systems).

- **`examples/scatter_plot.py`** — the "hello world" of the declarative API: 500-particle 3D scatter with axes, colormap legend, scale bar, two PBR lights, and live physics in ~30 lines (vs ~120 for the equivalent imperative-ECS demo).

- **`manifoldx.random`** — initial-condition generators for demos. Six position generators (`positions_uniform`, `_in_box`, `_in_sphere`, `_on_sphere`, `_in_disk`, `_gaussian`), five velocity generators (`velocities_gaussian`, `_uniform`, `_on_sphere`, `_tangent`, `_orbit`), two scalar generators (`scalars_uniform`, `scalars_gaussian`). All return float32 arrays. Optional `rng` kwarg accepts None (entropy seed), int (deterministic), or an existing `numpy.random.Generator`.

- **`manifoldx.physics`** — vectorized simulation primitives. `all_pairs(positions)` returns a dataclass with `diff` / `dist` / `dist_safe`; `gravity(positions, masses, G, softening)` returns N-body acceleration; `central_gravity(positions, GM, softening, center)` for single-source cases; `box_boundary(positions, velocities, half_size, dt, mode)` and `sphere_boundary(positions, velocities, radius, mode, strength, dt)` for in-place velocity reflection; `elastic_collisions(positions, velocities, radius, restitution)` for equal-mass pair collisions.

- **Examples rewritten** to use `mx.random.*` and `mx.physics.*`. Net effect on the four physics-heavy demos (nbody, gas, boids, point_cloud_demo): 99 lines of inline physics → 20 lines of helper calls; same visual behavior. Initial-condition setup compressed similarly: ~50 lines saved across five demos.

### Deferred from Plan 4 spec
- **Multi-viewport composition** (`|`, `&`). Renderer rewrite required; Plan 5+.
- **Generic `lines` mark.** Would need a generalized line material beyond axes; defer until concrete demand.
- **Standalone `text` mark.** Today's `TextLabel` + `LabelMaterial` covers this imperatively via the escape hatch; reconsider when usage shows it's wanted.
- **Reactive transforms** (Altair's `transform_calculate` etc.). The `@chart.simulate` callback covers everything with a fraction of the surface area.
- **Per-frame domain auto-scale.** `mxv.color(arr)` infers `(vmin, vmax)` from data once at build time; per-frame re-derivation would mean re-uploading material uniforms each tick.
- **Camera channel.** `chart.engine.camera.fit(...)` via the escape hatch suffices for now; promote to a `mxv.camera(...)` mark when ergonomics warrant.

### Deferred from Plan 3 spec
- **`ScaleBar` tag component** — the spec called for `ScaleBar(length, label_id)` as a routing tag, but on review the renderer doesn't need a distinct identity for scale bars: a screen-anchored axis line + a screen-anchored label compose into one. Plan 4's `scale_bar(...)` shim will package the composition.
- **Dedicated `ColormapLegendMaterial` + pipeline** — same simplification: legends route through the existing label pipeline by stashing the LUT as an atlas slice. A future `vertical` colormap orientation would benefit from a dedicated material when the atlas tile aspect (4:1) becomes a real constraint, but v1 horizontal legends work fine.
- **Line thickness honoring** — `AxisFrame.thickness` field is reserved; Plan 3 ships native LineList (1px on Vulkan/Metal). Quad-extrusion deferred until thicker lines have a concrete requirement.

### Refactors
- **`_BatchBuffers` helper** — Per-batch GPU buffer management lifted out of `RenderPipeline` into a dedicated helper, supporting capacity-tracking lazy allocation for `transforms`, `scalar_values`, and `radii`.
- **Globals uniform extended** — `vp + view + proj + camera_pos + pad` (208 bytes). The sprite vertex shader projects view-space billboards directly through `globals.proj`, removing a brittle `transpose(view)` approximation. Existing material shaders (Basic, Phong, Standard) updated to match the new layout.
- **Pipeline cache key** — Now `(geom_id, mat_type, mat_subtype, sprite)` so different colormaps share a sprite pipeline but rebind only the LUT texture.
- **`GeometryRegistry`** — Added name-based lookup APIs (`register_by_name`, `get_id`, `get_by_name`, `__contains__`) and auto-registers `SPRITE_QUAD` on init.
- **`_BatchBuffers` extended** — additional lazily-allocated `label_indices` buffer parallels the existing `transforms` / `scalar_values` / `radii` storage paths.
- **Pipeline factory `_get_or_create_pipeline`** — now accepts `label=True` for an alpha-blended, depth-write-off path with bind slots for a `texture_2d_array` + sampler.

### Fixes
- **Colormap LUT sRGB** — LUT textures now use `rgba8unorm_srgb` so the GPU sRGB-decodes on sample. Previously the matplotlib-encoded LUT bytes were sampled as linear and the framebuffer applied sRGB encoding on write, producing colors brighter than matplotlib's intended swatch.

## [0.2.0] - 2026-04-06

### Features
- **N-body simulation** — 500-body gravitational simulation with pure-numpy vectorized physics (250k force pairs/frame)
- **Ideal gas simulation** — 500 particles with elastic collisions and virtual bounding box walls
- **Boids flocking** — 300 boids with separation, alignment, cohesion rules + 4 wandering predators they flee from
- **Camera far plane** — Added near/far clipping planes (0.1/1000) for distant object support
- **Engine quit fix** — Fixed event loop termination for proper window closing

### Documentation
- Expanded motivation section with scientific simulation use cases
- Added PyGfx comparison explaining why ECS is better for data-driven simulations
- Added detailed showcase of three complex demos with vectorization patterns
- Simplified install instructions (pip + uv)

### Refactors
- Split nbody.py into gravity-only demo; gas.py for collision demo
- Tuned boids/gas parameters for better visual balance

### Fixes
- Boids now significantly scared of predators (20x fear force, 10-unit detection radius)
- Camera test now uses correct near/far values for VP matrix validation

## [0.1.0] - 2026-04-06

### Initial Release
- ECS with Structure-of-Arrays (SoA) layout
- Instanced GPU rendering with material-specific pipelines
- PBR (Physically Based Rendering) with GGX BRDF
- Basic materials (Phong, Standard)
- Geometries: cube, sphere, plane
- Camera with orbit controls and fit/bounds helpers
- Examples: hello_world, cube, pbr_demo, spheres