# Asset tools: from image-model drawings to a textured mesh (#226)

Generic tools taken from the parametric car (#224, #225). An agent with manifoldx and an image model
must be able to model and texture a new object with them. The prototype scripts they come from live
outside the repo; this plan states what each tool does, so it does not depend on them.

Rules for every task:

- numpy and the Python standard library only. No Pillow, OpenCV, scipy, trimesh. Tests too.
- Images are numpy arrays: `uint8`, shape `(H, W, 3)` (RGB) or `(H, W, 4)`; masks are `bool (H, W)`.
  Row 0 is the top of the picture. Image coordinates `(x, y)` = (column, row).
- Meshes are `manifoldx.modeling.Mesh` (`Mesh(positions=float32 (N,3), faces=uint32 (M,3))`,
  `.recompute_normals()`), counter-clockwise faces seen from outside.
- Every new public function is named in the book (`docs/chapters/`), in the modeling chapter's new
  section "Assets from an image model" (one sentence or a short example each), so `rift check`
  stays at 0 errors and adds no warnings.
- Tests in `tests/modeling/test_asset_tools.py` (and `tests/test_image.py`), synthetic inputs only:
  draw shapes with numpy, never ship image files. `uv run pytest tests/modeling tests/test_image.py -q`.
- Conventional commits, one per logical change, `Refs #226`. Never `git add -A`.

## Task 1: `manifoldx.image` (src/manifoldx/image.py)

- `read_png(path) -> ndarray`: 8-bit RGB, RGBA, grey and grey+alpha, palette (PLTE, tRNS optional),
  non-interlaced; all five filter types. Grey becomes RGB. Raise `ValueError` on anything else.
- `write_png(path, img)`: RGB or RGBA uint8 (filter 0 is enough). `devchannel.py` has a writer
  (`png`, line ~575): share or reuse it.
- `resize(img, size)` with `size=(w, h)`: bilinear for colour, `nearest=True` for label maps.
- `pad_to_aspect(img, aspect, fill=255) -> (img, (x0, y0))`: pad with `fill` (centred) to the nearest
  aspect ratio of `ASPECTS = {"1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"}` or the one
  given; return the offset of the original inside.
- Tests: round-trip each colour type; a PNG written by `write_png` read back equal; resize of a
  constant image stays constant; pad offset correct.

## Task 2: `modeling.trace` (src/manifoldx/modeling/trace.py)

A blueprint is a flat-colour drawing with a fixed colour code (`palette`: name -> RGB). Functions:

- `classify(img, palette) -> labels`: int array of the index of the nearest palette colour
  (Euclidean in RGB). Order of `palette` is the index order.
- `open_mask(mask, r)`, `close_mask(mask, r)`, `dilate(mask, r)`, `erode(mask, r)`: square or disc
  structuring element of radius `r`, numpy only (shifted ORs, or a summed-area table).
- `components(mask, connectivity=4) -> (labels, count)`: connected components.
- `contours(mask) -> list[ndarray (K,2)]`: outer boundary of every component, as closed polygons in
  pixel coordinates, counter-clockwise in a y-up frame; holes are returned with `holes=True` as a
  second list. Marching squares or Moore boundary tracing.
- `simplify(poly, tol)`: Douglas-Peucker on a closed polygon.
- `cut_by_rays(mask, centre, angles, width)`: remove thin wedges along rays from `centre`, so a disc
  with windows becomes pieces without holes (a wheel face). Rays must run past the mask.
- `trace_layers(img, palette, order, background="white", tol=1.5, min_area=30, frame=None) -> Shape`:
  classify, then for each name in `order` (larger regions first) its contours (holes of a region
  become a layer of the region under it), simplified, in units of the drawing's half-size: the
  frame (default: bounding box of everything that is not background) maps to `[-1, 1]` on its
  longer side, centred, y up.
- `redraw(shape, size, palette, background="white") -> img`: rasterise a Shape back (even-odd fill
  of each polygon, layers in order).
- `iou(a, b)` of two masks.
- Tests: a drawn disc with five windows traced with `cut_by_rays` gives pieces whose union redraws
  at IoU > 0.97 against the drawing; a rectangle with a hole; `simplify` keeps a square's 4 corners.

## Task 3: `modeling.shape` (src/manifoldx/modeling/shape.py)

`Shape`: a dataclass of `layers: list[Layer]`, `Layer(material: str, polygons: list[ndarray (K,2)])`,
plus `anchor: (u, v)` and `meta: dict`. Units: the part's half-size; outer side at +u (document this:
a shape traced from a drawing seen from behind must be mirrored first).

- `Shape.to_json() / Shape.from_json(text)`, `save(path) / load(path)`.
- `mirror(shape)` (u -> -u, orientation kept CCW), `mirror_half(shape, keep="left")`: keep one half
  of a symmetric drawing and reflect it.
- `stretch_middle(shape, ratio, ends=0.25)`: make the shape `ratio` times longer along u keeping the
  outer `ends` fraction at each end as drawn and stretching only the middle.
- `fit_box(shape, half_w, half_h)`: scale to fill a box.
- `bounds(shape)`.
- The car's `assets/*.py` dictionaries can be converted later (task 7); do not touch them now.

## Task 4: `modeling.views` (src/manifoldx/modeling/views.py)

Solids and atlases built from orthographic views. Axes: x right, y up, z toward the viewer of the
front view (the engine's convention). Views: `"front"` (looks along -z), `"back"` (+z), `"left"`
(the object's left side, looks along +x), `"right"`, `"top"` (looks along -y).

- `extrude(polygon, depth, centre=0.0)`: a closed prism from a 2-D outline (ear clipping, reuse
  `_triangulate` from `assets/car.py` by moving it to a shared helper).
- `two_view_solid(front, side, height_range, n=48, power=2.0, segments=24) -> Mesh`: `front` and
  `side` are masks of the same part in the front view and the left view, already scaled to the
  same metres per pixel and aligned on y. For each of `n` heights, the front mask gives the x
  interval and the side mask the z interval; the section is a superellipse (exponent `power`)
  inscribed in that rectangle; consecutive sections are joined, ends capped. Heights where either
  mask is empty are skipped; a part split into several intervals uses the widest one.
- `silhouette(mesh, view, size, frame) -> mask`: rasterise the mesh's silhouette (for checking
  against the drawing).
- `view_atlas(parts, views, size, texel=None, margin=8) -> Atlas`: every face of every part goes
  to the view among `views` whose direction is closest to the face normal (largest dot product);
  its UV is the orthographic projection in that view. Each view is one island laid out in a grid
  in a square atlas of `size` px; islands keep the same metres per pixel unless `texel` is given.
  `Atlas` holds per part the `uv` (float32 (N,2), the mesh is split so a vertex has one UV) and
  per island its rectangle in pixels and the view's frame (metres).
- `template(parts, atlas, colors, island) -> (img, regions)`: draw the island's faces in flat
  colours (`colors`: part or material name -> RGB) with a depth buffer whose depth is the position
  along the view direction (linear across a triangle), white background; `regions` is an int map
  of the material index (0 = background).
- Tests: `two_view_solid` of two disc masks is close to a sphere (radius within 3 %); a cube's six
  faces land in six different islands of a six-view atlas; the template of a red cube in front of
  a blue one (front view) is red where they overlap.

## Task 5: `modeling.paint` (src/manifoldx/modeling/paint.py)

- `paste_island(atlas_img, island_rect, picture, pad_offset)`: resize the model's answer to the padded
  canvas, crop the island back out and paste it in place.
- `kept_outline(template, picture, white=235) -> float`: IoU of the non-white pixels.
- `clean(picture, regions, colors, tone=None)`: background left white inside a region becomes that
  region's flat colour; optional scaling of the island so the median of one region equals `tone`.
- `bleed(img, mask, steps=8)`: grow the colours of `mask` into the rest, so filtering at seams does
  not pick up white.
- `normal_from_albedo(albedo, strength=6.0, blur=6, line=4.0) -> img`: a tangent-space normal map
  (OpenGL, flat = 128,128,255) computed from the picture: thin lines darker than their surroundings
  become grooves (local mean minus pixel), smoothed, differentiated. The image-model normal map is
  the optional alternative (task 6).
- `flip_normal_x(normal)`: for a mirrored island, invert red.
- Tests: `normal_from_albedo` of a flat image is flat; a dark vertical line gives red values on
  both sides of opposite sign; `kept_outline` of an image with itself is 1.

## Task 6: `manifoldx.imagegen` (src/manifoldx/imagegen.py)

- `ImageModel` protocol: `generate(prompt, images=(), aspect="1:1", size="1K") -> ndarray`, with
  `name`, and a `cost` attribute (dollars of the last call, or `None`).
- `OpenRouterImage(model, key=None, key_file=None, log=None, budget=None)`: `urllib` POST to
  `https://openrouter.ai/api/v1/chat/completions` with `modalities: ["image","text"]`,
  `image_config: {aspect_ratio, image_size}` (only for models that take it; ignore errors from
  others by retrying without it), `usage: {include: true}`; images as PNG data URLs. Appends
  `time, model, out_name, cost` to the `log` file and refuses a call (raises `BudgetExceeded`)
  when the log's total for this budget would pass `budget`.
- `HttpImage(url, token_file)`: a local service with `POST /generate {prompt, refs, width, height, seed}`
  returning `{image: data-url}` (the FLUX.2 klein service's API).
- `prompts`: functions returning prompt text: `blueprint(part, view, palette, notes="")`,
  `albedo(view, palette_meaning, notes="")`, `normal_map()`, `roughness_map(materials)`,
  `emission_map(lit)`, `edit(change, keep)`. Each states the rules learnt on the car: one view per
  call, orthographic, flat colours with hard edges, no text, keep the canvas and every outline pixel
  for pixel, flat even light, white background. Keep them short and in English.
- Tests: no network. A fake server (`http.server` on localhost) checks the request shape; budget
  refusal; prompts mention the palette colours.

## Task 7 (later): the car's shapes as data

`assets/lamps.py`, `grilles.py`, `wheels.py`, `mirrors.py`, `dashboards.py`, `steering.py`,
`doors.py`, `consoles.py` hold traced shapes of one real car. Move the traced data to JSON files under
`docs/examples/data/` (or the example that uses them), keep the plain styles in code, and make the
car take a `Shape` or a path. The 12 car tests keep passing.
