# Models from drawings

The plaza is nearly finished, but three objects are missing and nobody wants to
model them by hand: a hydrant at the kerb, an armchair in the town hall, and a
small house past the square. Their shapes are not a stack of parameters the way
the car's body was. A hydrant's bonnet, a chair's arms and a house's front wall
are drawings before they are meshes.

`manifoldx.asset` is the layer that turns such drawings into a scene. It sits on
the functions of chapter 6's last section — trace, two-view solids, paint, lint —
and on `manifoldx.imagegen`, which asks an image model for a picture. The
command `manifoldx asset` sits on top of it. Each layer can be used without the
one above: a script can call the tracing functions, Python can run an asset
class, and a shell can run the command.

The drawings in this chapter were made once, with an image model, and committed
next to the models. Every block below builds from those files: no network, no
image model, no Blender.

## An asset is a class with stages

An asset is one Python file with a class that inherits from `Asset` and
overrides the stages it needs:

- `draw(c)` asks for the pictures the build will read. It returns nothing; the
  answers stay in the cache.
- `build(c)` returns a `Scene`: named meshes, placed, with their materials.
- `paint(c, scene)` returns a texture per mesh. Nothing by default.
- `check(c, scene)` returns findings of its own, on top of the standard lint.

`Asset.run(cache=None, stages=..., model=None, budget=None, max_tris=None,
again=())` runs the stages in order — `build` when `lint` or `paint` needs a
scene — and returns a `Result` holding the scene, the textures, the `Report` of
findings, and the `Context` the run used. The `c` every stage receives is that
`Context`: `c.image` for a paid call, `c.load` and `c.mask` for a cached
picture, `c.metres` for the drawing's scale, `c.scene` for a new scene,
`c.expect` for what a drawing says the camera sees, and `c.asset` to build
another asset and place what it returns.

`Parts` is the ready subclass for the commonest case: one object, named parts,
drawn in the same flat colours from the front, the back and the left. A `Parts`
subclass says what the object is (`desc`, `style`), how tall it is (`height`),
which part centres the views (`core`), and then makes one `p.add` call per part.
Each part names a `Builder`:

| builder | for | options |
| --- | --- | --- |
| `two_view` | rounded or boxy parts seen whole from the front and the side: heads, seats, a barrel | `power`, `rings`, `segments`, `fill` |
| `tube` | limbs and frames that reach across and forward: arms, armrests on a post, arches | `anchor`, `radius`, `samples`, `section` (`"drape"` for a robed arm) |
| `plate` | flat parts seen whole from one side: fins, signs, a tail | `thickness`, `lean` |
| `lathe` | turned parts: vases, columns, a lamp | `segments`, `rings` |

Every builder also takes `views` (which drawings say something about the part),
`attach` (a part it must touch), `keep` (what to do with stray pieces), `grid`
(repeated instances seen two from the front and two from the side) and `mirror`
(build the near instance and reflect it). The build traces the blueprints,
aligns the views to the same rows, and hands each part's masks and the shared
`Frame` to its builder. A builder returns a `Built`: one mesh per instance, the
rows it filled from the other view, and the area of every piece `keep` dropped.

## The hydrant at the kerb

The hydrant is a short file: a description, a height, the drawings' notes, and
five parts with their colours and builders.

```python
from manifoldx.asset import Parts, two_view


class Hydrant(Parts):
    desc = ("a classic red cartoon fire hydrant: a round barrel on a wide base flange, a domed bonnet on top "
            "with a pentagon nut, one short hose outlet on each side with a round cap, and one big pumper outlet "
            "with a round cap facing the front")
    style = "Clean cel-shaded cartoon style with flat colours and thin dark outlines, even light."
    height = 0.8
    model = "google/gemini-3.1-flash-image"
    core = "barrel"
    base = (200, 40, 35)

    def parts(self, p):
        p.add("barrel", "orange", "the round barrel, the main body")
        p.add("bonnet", "yellow", "the domed bonnet on top, with its nut and the ring under it")
        p.add("base", "magenta", "the wide base flange at the bottom")
        p.add("side", "blue", "each side hose outlet with its cap",
              build=two_view(power=6.0, views=("front", "left")))
        p.add("pumper", "green", "the big front pumper outlet with its cap",
              build=two_view(power=6.0, keep="largest"))
```

`notes` and `details` are left out here; the file next to the blueprints has a
line per view for the drawing prompt and a line per view for the paint prompt.
The code colour is the part's name inside the blueprint, so no two parts may
share one. The palette is `CODE_COLOURS`, ten flat colours with white for the
paper; a pixel belongs to the nearest one, which is why the drawings must use
exactly those values.

Load the class from its own file rather than retyping it here. The cache
directory is `NAME.asset/` next to the file that defines the class:
`Asset.cache_dir()` returns it, and it finds the file through
`inspect.getfile`. A class defined in a chapter would look for `hydrant.asset`
in the chapter's directory instead.

```python {run}
import importlib.util
import sys
from pathlib import Path

ROOT = next(path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (path / "pyproject.toml").exists())


def example(path, name):
    """Load an example from its own file, so its cache stays beside it."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module     # inspect.getfile() looks the class's module up here
    spec.loader.exec_module(module)
    return module


def show(label, result):
    """One run's size and findings, the way a person reads a report."""
    print(f"{label}: {len(result.scene)} parts, {len(result.report)} findings, exit {result.exit_code}")
    print(result.report.text())


Hydrant = example("examples/models/hydrant.py", "hydrant_example").Hydrant
hydrant = Hydrant()
result = hydrant.run(stages=("build", "lint"))
show("Hydrant", result)
```

Six findings, and the command would still exit 0, because none of them is an
error. The report has three levels: `info`, `warn` and `error`. A `Report`
prints one finding per line, `Report.to_json` gives the same as JSON, and
`Report.exit_code` is 1 when any finding is an error.

Five lines are `A-SEEN`. In one view, the check projects the built part back
into the drawing and compares what the camera sees with the mask the blueprint
gives that part. The number in parentheses is the intersection over union: both
pixels over either pixel. Below 0.75 the finding warns; below 0.5 it is an
error. The message counts the two mistakes separately — pixels built but not
drawn, pixels drawn but not built — and the rows they fall on, as a fraction of
the drawing's height.

The barrel warns in the front (0.736) and in the back (0.683): the blueprint
draws the body a little wider than the two-view solid makes it. The base is the
worst at 0.58 in all three views, because the drawing rounds the flange into a
skirt and the section built from the front and side spans cannot follow it. The
hint names the options that could answer this: another builder, `power` (2 is an
ellipse, 6 a rounded box), or `views` to leave out a drawing that disagrees.
These are shapes to improve, not a broken model.

The sixth line is `A-STRAY`: the pumper's blueprint holds two small pieces
beside the outlet, 0.20 % of the drawing each. Its builder says
`keep="largest"`, so the build keeps the big piece, drops the two, and reports
their size. The default threshold is a thousandth of a 480 by 480 drawing;
`keep="all"` builds every piece.

The other codes say what a blueprint set can get wrong, and each is raised by
the stage that can see it: `A-FLOAT` (a part touches no other), `A-OPEN` and
`A-FLIP` (a mesh with holes, or inside out), `A-ROWS` (too many rows filled from
the other view), `A-SHEET` (a drawing holds several views), `A-KEEP` (a
blueprint lost the outline of its drawing), `A-CODE` and `A-OUTLINE` (the paint
kept a code colour or lost the template's outline), and `A-TRIS` (more triangles
than `max_tris`).

## What inspect shows

An agent writing an asset cannot look at the mesh, so `inspect(asset, c, scene,
out_dir, report=None)` writes pictures and a summary instead. For every
expectation it draws the scene's silhouette over the blueprint: green where both
agree, grey where the drawing shows something the build did not make, red where
the build made something the drawing does not show. It also writes `groups.png`,
the front, left and top silhouettes with each group of touching parts in its own
colour, and `summary.txt`, one fact per line.

```python {run continue}
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from manifoldx.asset import inspect

with TemporaryDirectory() as scratch:
    files = inspect(hydrant, result.context, result.scene, scratch, report=result.report)
    print(Path(scratch, "summary.txt").read_text())
    views = {path.stem: np.asarray(Image.open(path).convert("RGB"))
             for path in files if path.suffix == ".png"}
Image.fromarray(views["overlay-front"])
```

The summary counts six parts and 5,760 triangles, and one group of touching
parts: the hydrant is one object, which is what an `A-FLOAT` would have denied.
The IoUs are the same numbers the report gave, plus the ones for parts that
passed. The overlay's grey is at the side outlets' caps, drawn but not built;
the red is a thin fringe where the barrel is a little too wide.

```python {run continue}
Image.fromarray(views["groups"])
```

The top view shows the two side outlets and the front pumper as one cross: the
parts touch, so the group has one colour.

## The armchair, one option at a time

The armchair's six committed blueprints produce a lint with an `A-SEEN` error,
and the way to read it is the way an agent works: read the finding, change one
option, run again, and compare. A change that is not one option makes the next
report unreadable, because two effects arrive at once.

The first draft trusts every drawing and attaches nothing. It is the committed
class with those two options taken away, so the four `p.add` lines appear again
here.

```python {run continue}
from manifoldx.asset import Parts, two_view

Armchair = example("examples/models/armchair.py", "armchair_example").Armchair
CACHE = ROOT / "examples/models/armchair.asset"     # the drawings are committed here


class FirstTry(Armchair):
    """Every part trusts all three drawings, and nothing is attached."""

    def parts(self, p):
        p.add("seat", "orange", "the seat cushion with the wooden seat frame under it",
              build=two_view(power=5.0))
        p.add("back", "yellow", "the back: the back cushion and the whole wooden back frame, including its two "
                                "upright side posts from the seat up to the top", build=two_view(power=5.0))
        p.add("arms", "blue", "each flat wooden armrest with the single short post under its front end, and "
                              "nothing above the armrest", build=two_view(power=6.0))
        p.add("legs", "magenta", "each tapered wooden leg", build=two_view(power=2.0, grid=True))


show("first try", FirstTry().run(cache=CACHE, stages=("build", "lint")))
```

Five `A-FLOAT` errors and three `A-SEEN` errors. `A-FLOAT` says a part touches
no other and gives the gap to the nearest part: the arms miss the back by 2 cm,
the legs miss the seat by 12 to 14 mm. The tolerance is one percent of the
asset's height, 8.5 mm here, so a part that nearly touches still counts as
floating.

The three `A-SEEN` errors are all in the back view, and they have one cause: the
back drawing paints the back cushion over the arms and the legs, so a part that
trusts the back blueprint gets a silhouette that is mostly the cushion. The hint
says so: leave out a view that disagrees.

```python {run continue}
class ViewsOnly(Armchair):
    """The second run: the parts the back drawing hides trust the front and the left only."""

    def parts(self, p):
        p.add("seat", "orange", "the seat cushion with the wooden seat frame under it",
              build=two_view(power=5.0, views=self.FRONT_LEFT))
        p.add("back", "yellow", "the back: the back cushion and the whole wooden back frame, including its two "
                                "upright side posts from the seat up to the top", build=two_view(power=5.0))
        p.add("arms", "blue", "each flat wooden armrest with the single short post under its front end, and "
                              "nothing above the armrest", build=two_view(power=6.0, views=self.FRONT_LEFT))
        p.add("legs", "magenta", "each tapered wooden leg",
              build=two_view(power=2.0, grid=True, views=self.FRONT_LEFT))


show("with views", ViewsOnly().run(cache=CACHE, stages=("build", "lint")))
```

The three back-view errors are gone, and two warnings remain where the drawings
still differ a little. The `A-FLOAT` errors are untouched, because `views` says
nothing about contact. The next change is the one that answers them: `attach`
names the part the piece must touch, and the build moves it the shortest way
until it does.

```python {run continue}
show("with views and attach", Armchair().run(cache=CACHE, stages=("build", "lint")))
```

The `A-FLOAT` errors are gone. Two `A-ATTACH` info lines say how far each part
moved — 2.0 cm for the arms, 1.4 cm for the legs — so a move that quietly
distorts the model is visible, not hidden. One `A-SEEN` error is left: the arms
in the front view at 0.43. The report names its options — another builder,
`power`, `views` — and that is the next run. The committed example keeps this
error: the lint reports, and choosing the next change is the agent's work.

`check` is where a `Parts` subclass adds its own findings, as the house does in
a moment. The `Parts` version reports `A-ROWS`, `A-STRAY`, `A-ATTACH`, `A-KEEP`
and `A-SHEET`; the standard lint under it reports the rest.

## An open asset: the house

The house is not one object seen from three sides. It is a floor plan, the
front facade of the same house, and a set of parts placed by measurement. No
single `desc` and three blueprints describe it, so `House` inherits `Asset`, not
`Parts`, and writes its own stages.

```python
class House(Asset):
    height = 2.75            # walls plus roof, in metres
    wall_height = 2.6
    model = "google/gemini-2.5-flash-image"

    def draw(self, c):
        c.image("plan", _PLAN)
        c.image("facade-front", _FACADE, refs=["plan"])
```

Two drawings, not six. The file once asked for four facades and a close-up of
the door, and the build read only the plan and the front. A drawing the build
never reads is a paid call with no effect, so the other four left `draw` and
their pictures left the cache with them.

The build is ordinary Python with `manifoldx` calls in it. It measures the
facade to get metres per pixel, traces the plan's walls, stands a box around
every enclosed room instead of cutting the plan with a boolean, fills the gap
between two rooms with one wall computed by `_wall_between`, splits the front
wall around the door, and places the windows and the roof from the colours of
the front elevation.

```python
    def build(self, c):
        scene = c.scene()
        facade = c.load("facade-front")
        masks = {colour: c.mask("facade-front", colour)
                 for colour in ("orange", "yellow", "red", "blue")}
        wall = _main(masks["orange"])
        scale = self.wall_height / (wall[3] - wall[1])

        plan = c.mask("plan", "black") | c.mask("plan", "red")   # the door gap, closed for tracing
        outer, rooms = _bounds(plan), _rooms(plan)

        walls = [ ... ]                     # three outer walls, then one between each pair of rooms
        for index, mesh in enumerate(walls, start=1):
            scene.add(f"wall{index}", mesh)
        scene.add("door", ...)               # the red region of the facade, as a panel
        scene.add("roof", ...)               # the yellow band, as a slab over the footprint
        c.expect("front", {name: mask for name, mask in shown.items() if mask.any()},
                 frame=(x_of(0), x_of(image_w), y_of(image_h), y_of(0)))
        return scene
```

`c.expect(view, masks, frame=...)` records an `Expectation`: for each named
part, the mask the drawing gives it and the rectangle of metres that mask
covers. The standard lint's `standard_findings` then checks each part in that
view part by part, not the whole silhouette as one mask. The frame matters
here: the facade's drawing has margins and its own scale, so the build passes
the rectangle its own measurement produced.

The house also checks what no drawing can say. `check` adds `A-COVER` when the
roof does not reach past every wall, and `A-DOOR` when the door stands outside
them. Both are codes of the asset's own, and they follow the same `Finding`
shape as the rest.

```python {run continue}
House = example("examples/models/house.py", "house_example").House
house = House()
result = house.run(stages=("build", "lint"))
print(result.report.text() or "no findings")
print(len(result.scene), "parts:", ", ".join(result.scene))
expectation = result.context.expectations[0]
print("expected in", expectation.view, ":", ", ".join(sorted(expectation.names)))
```

No findings: the eleven boxes pass the standard checks, and the front
expectation agrees with the drawing for the walls, the roof, the door and the
windows. A `Scene` is an ordered mapping of names to meshes; `Scene.to_glb`
writes one glTF, and `save(result, out_dir)` writes `model.glb`, any textures as
PNG and `report.json` — the same files the command writes.

## Paid calls and the cache

Everything above ran offline. The one call that costs money is `c.image`:

```python
from manifoldx.asset import Context

c = Context("hydrant.asset", model="google/gemini-3.1-flash-image", budget=0.50)
front = c.image("view-front", "the hydrant, seen exactly from the front, orthographic, white background",
                refs=["ref"], aspect="1:1")
```

The picture is cached under `hydrant.asset/images/view-front.png`, and a sidecar
`view-front.json` records the prompt, the references, the model, the aspect and
a key. The key is the SHA-256 of the model, the prompt, the aspect and the
pixels of every reference. A later run with the same key reads the PNG and
sends nothing. Change one part's text and only the pictures whose prompt
changed are asked again; the others keep their answers.

`c.load(name)` reads a cached picture and never reaches the network. `c.forget`
deletes a picture and its sidecar; a name is a prefix. A `Parts` subclass's
`forget` widens it first: forgetting a view forgets its drawing, its blueprint
and its albedos, and forgetting a part forgets the views its builder trusts.

```bash
uv run manifoldx asset examples/models/hydrant.py draw --budget 0.50
uv run manifoldx asset examples/models/hydrant.py draw --again left
uv run manifoldx asset examples/models/hydrant.py paint --again front
```

`--budget` caps the spend: each answer's cost is added to the context, and the
next call raises `BudgetExceeded` before anything is sent. `--again` forgets
first, so `draw --again left` redraws the left view and its blueprint, and
`draw --again side` forgets the views that the `side` part's builder trusts.
`--model` overrides the class's model for one run.

## The command

`manifoldx asset FILE draw|build|lint|inspect|paint|render` runs layer 2 from a
shell:

| command | what it does |
| --- | --- |
| `draw` | the reference (unless there are photos), one view drawing and one blueprint per view; cached |
| `build` | traces and builds every part; writes `model.glb` and `report.json` |
| `lint` | builds and prints the findings; `--json` for an array, exit 1 on an error |
| `inspect` | the overlays, `groups.png` and `summary.txt`, into `NAME.asset/` |
| `paint` | a template and one albedo call per view, the maps computed from it |
| `render` | Blender stills, toon or photo, when Blender is installed |

A usage error exits 2, an error finding exits 1, and a clean run exits 0. The
command's `load` reads the one `Asset` subclass a file defines; `run` is what the
shell command calls; `add_parser` is its argparse wiring. Beyond that it has no
logic: `save` and `inspect` are the same functions a Python caller uses.
`render` is the one command with a dependency: it asks `blender.available()`,
calls `blender.render`, and without a `blender` on PATH it raises
`BlenderMissing` and exits 2.

```python {run continue}
import subprocess

lint = subprocess.run(
    ["uv", "run", "manifoldx", "asset", str(ROOT / "examples/models/hydrant.py"), "lint", "--json"],
    capture_output=True, text=True,
)
print("exit", lint.returncode)
print(lint.stdout)
```

The JSON has one object per finding, with the code, the level, the part, the
view, the message, the value and the hint. An agent reads it without parsing
prose. The same command on the armchair exits 1: its `A-SEEN` error is still an
error.

## Three objects on the square

The three models go where the square needs them, at the sizes their files
declare: the hydrant at the kerb, the armchair where the hall's portico stands,
the house past the paving. Their scenes are `Scene` values of `Mesh` values; the
engine only sees the geometry dicts, as in chapter 6.

```python {run export=examples/models_from_drawings.py}
import importlib.util
import sys
from pathlib import Path

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial
from PIL import Image

ROOT = next(path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (path / "pyproject.toml").exists())


def example(path, name):
    """Load an example from its own file, so its cache stays beside it."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hydrant = example("examples/models/hydrant.py", "hydrant_example").Hydrant()
armchair = example("examples/models/armchair.py", "armchair_example").Armchair()
house = example("examples/models/house.py", "house_example").House()

# Where each object stands, and the colour of its clay until it is painted.
placed = [(hydrant, (-2.0, 0.0, 2.8), "#b03028"),
          (armchair, (-0.4, 0.0, -1.2), "#c4962c"),
          (house, (5.2, 0.0, -7.6), "#ded2b8")]

engine = mx.Engine("plaza", width=640, height=360)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)

engine.spawn(Mesh(GeoMesh.plane(width=40, depth=40).to_geometry()),
             Material(StandardMaterial(color="#b5a98f", roughness=0.95)), Transform())

counts = []
for asset, at, colour in placed:
    scene = asset.run(stages=("build",)).scene      # the lint was read in this chapter
    counts.append((len(scene), asset.name))
    material = StandardMaterial(color=colour, roughness=0.85)
    for mesh in scene.meshes.values():
        engine.spawn(Mesh(mesh.to_geometry()), Material(material), Transform(pos=at))

engine.camera.set_pose(position=(0.5, 1.6, 6.4), target=(1.0, 0.9, -3.0))
frame = engine.render_frame(supersample=1)
print(counts)
Image.fromarray(frame)
```

The hydrant and the armchair are 5,760 and 7,680 triangles, and the house is
132: a rounded part is thirty rings of sixteen segments, while a wall is a box.
`--lod low` asks a `Parts` asset for ten rings and eight segments, and `--max-tris`
lowers the threshold at which `A-TRIS` warns. For a background object, that is
the dial to turn; for the object the camera walks up to, it is not.
