---
when: making an asset with manifoldx asset, alone or as an agent: turning an image model's drawings into a mesh, reading a lint and changing one option at a time, drawing and painting without paying twice, or choosing between Parts and an open Asset
triggers: ["manifoldx asset", "asset", "Parts", "two_view", "swept_tube", "A-SEEN", "A-FLOAT", "A-KEEP", "A-SHEET", "A-CODE", "A-OUTLINE", "A-TRIS", "blueprint", "blueprints", "image model", "albedo", "kept_outline", "LOD", "lint findings"]
---

# Making an asset

An asset is one Python file whose class inherits from `Asset` and overrides the
stages of a pipeline: `draw` asks an image model for the pictures, `build`
returns a `Scene`, `paint` returns the textures, `check` adds findings of its
own. `manifoldx asset FILE draw|build|lint|inspect|paint|render` runs them. The
cache is `NAME.asset/` next to the file, holding every drawing and the sidecar
that records the prompt and key that produced it.

`Parts` is the ready subclass for one object of named parts. It asks for a
reference (unless the file names photos), one view drawing and one blueprint
per view, traces the blueprints, aligns the views and builds each part with a
builder: `two_view` for a rounded or boxy part, `tube` for a limb or frame,
`plate` for a flat part, `lathe` for a turned one. Anything a set of three
views cannot describe — a house from a plan, a scene made of sub-assets — is an
open `Asset` whose `build` writes ordinary `manifoldx` code.

The worked examples are `examples/models/hydrant.py` and
`examples/models/armchair.py` (`Parts`) and `examples/models/house.py` (an open
`Asset`). Their drawings are committed, so build and lint run offline; the
chapter on models from drawings (`docs/chapters/15-models-from-drawings.md`)
reads the three.

## The loop

1. Write the file, or change one thing in it. One thing: one part's text, one
   builder, one option.
2. Draw what the change needs: `manifoldx asset FILE draw --budget 1.00`. A
   second run with the same prompts sends nothing. `--again VIEW` redraws one
   view, its blueprint and its albedos; `--again PART` redraws the views that
   part's builder trusts.
3. Build and lint: `manifoldx asset FILE lint --json`. Exit 1 means an
   error-level finding, 2 a usage error; read the JSON when the lines are long.
   Read `A-SHEET` and `A-KEEP` first: a bad drawing is redrawn, not built
   around.
4. Inspect: `manifoldx asset FILE inspect`. It writes `overlay-<view>.png`,
   `groups.png` and `summary.txt` into `NAME.asset/`, and prints their paths.
5. Read `summary.txt`: parts and triangles, the groups of touching parts, each
   part's IoU per view, then every finding. Look at the overlay: green is both,
   grey is drawn but not built, red is built but not drawn. In `groups.png`, a
   part in a group of its own is a part that floats.
6. Change one option, and go back to 3. The next report then shows what that
   change did, not two changes at once.
7. Paint only when the geometry has no error: `manifoldx asset FILE paint`. It
   is the second paid step. Repaint a view only on `A-CODE` or `A-OUTLINE`.
8. When the model is good, `--lod low` writes a coarse version, and `render`
   writes Blender stills when Blender is installed.

The same order from Python:

```python
result = Hydrant().run(stages=("build", "lint"))
print(result.report.text(), result.exit_code)
```

`run(stages=("build", "lint"))` never reaches the network, `cache=` points it at
another directory, and `again=("left",)` does what `--again left` does.

## Writing the file

- **Name the parts so that none hides another in a view.** A part hidden behind
  another in the front drawing gets no front mask; either give it a view that
  shows it or leave that view out with `views=("front", "left")`.
- **One code colour per part, from `CODE_COLOURS`.** A blueprint pixel belongs
  to the nearest palette colour, so a drawing made with other values is read
  wrong.
- **Say what each view shows when the object has no obvious front.** The
  hydrant's `notes` say that the pumper points at the viewer in the front view
  and toward the left edge in the side view, and that one side outlet faces the
  viewer. Without that line the model turns the object and invents the outlet
  in the back blueprint.
- **Leave thin attachments out of the blueprint.** A chain, a wire or a thin
  handle drawn in a part's colour becomes a fin in the mesh. Describe the part
  without it, or give it a part and a builder of its own.
- **Describe the part, not the whole object, in its `p.add` text.** The back
  frame's uprights were painted as arms by one view; saying "the back,
  including its two upright side posts" put them in the part that owns them.
- **A repeated part gets `grid=True`.** Four legs are four pieces in the front
  drawing and two in the side; the option pairs the outer front pair with the
  frontmost side leg and the inner pair with the other.
- **A part seen only on one side of the object gets `mirror=True`**, so one
  instance is built and reflected instead of two half-drawn ones.

The cache lives beside the file because `Asset.cache_dir()` finds the file that
defines the class. Load the class from its own file in a test or a notebook
(`importlib` with `sys.modules[name] = module`); a class retyped in another
file looks for its drawings in the wrong place.

## Drawing

The order matters more than the prompt:

1. **A reference first**, unless the file names photos. It fixes the style and
   the proportions for every later call.
2. **One view per call, asked from the reference.** The model draws a sheet of
   views when asked for two; the view prompt says "one single view of one
   object, not a sheet". It also names the direction: seen exactly from the
   front, from behind, or from the left in strict profile.
3. **The view drawing first, then its blueprint.** The blueprint call takes the
   view drawing as its reference and asks for the same outline, pixel for
   pixel, painted in the code colours. Asked for a blueprint straight from a
   three-quarter reference, every model copies the three-quarter pose: the
   "side view" is not a profile.
4. **Say what the view shows** in the view's note, as above.

Other rules that save calls:

- The **back view is the weakest**. Models often draw it as a mirror of the
  front composition — a bag on the wrong shoulder, a tail on the wrong side.
  Build from the front and the side, and use the back for texture, or leave it
  out of a part's `views` when it disagrees.
- **Align the views on the main part.** If the front drawing cuts a pedestal at
  the frame and the side does not, scaling each view by its whole figure puts
  the object at three scales. `align` names the parts the views are scaled by.
- **Check `A-SHEET` and `A-KEEP` before building.** They compare each drawing
  with its own view and blueprint; one says the drawing holds several views,
  the other that the blueprint lost the drawing's outline. Both are answered by
  drawing again, not by a builder.
- **A drawing the build never reads is a paid call with no effect.** The house
  once asked for four facades and a door close-up and read only the plan and
  the front; the other four left `draw`, and their pictures left the cache.

## Reading the lint

Every finding has a code, a level, the part and the view it is about, a number
and a hint. `info` is a fact worth knowing, `warn` a shape to improve, `error`
a model the command exits 1 on.

- **`A-SEEN`** — what the camera sees of a part in one view against that
  drawing's mask. The number is the IoU: both pixels over either pixel. Below
  0.75 it warns, below 0.5 it is an error. The message counts pixels built but
  not drawn, pixels drawn but not built, and the rows they fall on. Try another
  builder, a different `power`, or `views` to leave out the drawing that
  disagrees. A low value in every view usually means the part's text and the
  drawings describe different parts.
- **`A-FLOAT`** — a part touches no other. The message gives the gap to the
  nearest part. Answer it with `attach="nearest part"`; for a limb or an arm,
  give the tube an `anchor`. Check the part's text if the drawings disagree on
  where it starts.
- **`A-ATTACH`** — info: the build moved a part the distance it names to reach
  the part it attaches to. A move of a centimetre or two is normal; a large one
  says the drawing and the part's text disagree.
- **`A-ROWS`** — warn: more than 20 % of the part's rows were filled from the
  other view. The two views disagree about the shape, or one hides the part's
  end behind another. Say it in the part's text, or set `fill="off"` when a
  filled row is wrong.
- **`A-STRAY`** — info: pieces `keep` dropped, with their size. Nothing is
  wrong if the big piece is the part; `keep="all"` builds them, a smaller
  threshold drops fewer.
- **`A-KEEP`** — warn: the blueprint lost too much of its view drawing's
  outline. Draw again for that view, or write a note for it.
- **`A-SHEET`** — error: a drawing holds more than one figure of comparable
  size, a sheet of views. `draw --again VIEW`.
- **`A-OPEN`, `A-FLIP`** — error: a mesh with holes, or inside out. These are
  builder bugs: report them with the masks that produce them, not a change to
  the part's text.
- **`A-TRIS`** — warn: a part has more triangles than `max_tris` (20,000 by
  default, or `--max-tris`). Fewer `rings` and `segments`, or `--lod low`.
- **`A-CODE`** — warn: after painting, part of a view's albedo is still near a
  part's code colour. `paint --again VIEW`.
- **`A-OUTLINE`** — warn: the painted answer kept under 0.85 of its template's
  outline after the retries, and the best answer was kept anyway. A note for
  that view, a plainer template, or another model.

An open `Asset` writes codes of its own in `check`, such as the house's
`A-COVER` (the roof must cover every wall) and `A-DOOR` (the door must stand
inside them). Follow the same shape: a code, a level, the part, a number and a
hint that names the option or the drawing that answers it.

## Painting

`paint` builds a view atlas and one template per view, then makes one albedo
call per view. A rounded object's own island is a ragged band the model cannot
read, so the template is the whole view (`whole=True`, which `Parts` uses); the
reference is the view drawing and the first photos, when there are any.

- **Gate every answer on `kept_outline`.** `Parts` asks again up to three times
  and keeps the best answer; a view under 0.85 after that is `A-OUTLINE`.
  Models sometimes return the figure smaller or moved.
- **Strip the code colours.** Pixels near a part's colour become the asset's
  `base`; what is left near one is `A-CODE`. Repaint that view, do not ship it.
- **The right view comes from the left, mirrored.** No call is needed for it.
- **A normal map is computed from the finished albedo** by default; it is
  registered by construction and costs nothing. Ask the model for one only when
  its sculpted relief is worth a call.
- **Ask for one view per call here too.** Two views on one answer share a
  canvas, and a view painted at another view's scale does not fit its island.

## Which model

The measured runs behind these lines used OpenRouter. Prices move; these are
the shapes to expect.

| model | what it is good for | cost and caveats |
| --- | --- | --- |
| `google/gemini-3-pro-image` | the best drawings and albedos, the most literal about the template | about $0.14 for a 2K image, roughly three times the flash models |
| `google/gemini-3.1-flash-image` | the balance: good blueprints and albedos, quick | a few cents an image; the default choice for a new object |
| `google/gemini-nano-banana-2.1` | cheap, good with the 0.85 gate and one retry | cheaper than flash, a little less consistent |
| `google/gemini-3.1-flash-lite-image` | geometry drawings | the side albedo failed three times on a rounded object |
| `google/gemini-2.5-flash-image` | albedos from photos | it shades blueprints made from a generated reference; from photos it drew flat ones |

An object takes 9 to 14 calls: a reference, one view and one blueprint per
view, and one albedo per view, with a retry or two. At these prices that is
$0.35 to $1.40 an object. The cache makes a rebuild free, so spend the calls on
drawings and keep them: commit `NAME.asset/` beside the file, as the examples
do, and build and lint run offline.

## LOD, size and the bill

- `--lod low` builds with 10 rings and 8 segments instead of 30 and 16: about a
  sixth of the triangles, textured from the same answers, so no extra calls. A
  7,680-triangle armchair becomes about 1,280. The command writes it to
  `NAME.asset/low/` unless `--out` names another directory.
- `A-TRIS` is the warning line, not a hard limit. A background object can be
  coarse; the object the camera walks up to cannot.
- One picture is one call: `draw` once, `paint` once, and change one prompt
  before asking again. The sidecar under `NAME.asset/images/` says which prompt
  and references produced each picture, so a redraw is a change you can see.
