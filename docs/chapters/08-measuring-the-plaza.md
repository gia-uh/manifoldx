# Measuring the plaza

By nine in the morning the square is full, and an urban researcher has a
morning of counts to show for it: five hundred observations of where people
stood, how long they stayed, and a noise survey that rose and fell with the
crowd. She does not want a plot beside a photograph of the plaza. She wants one
figure in which the counts sit where they happened, the names are on the
places, the distances are in metres, and the noise hangs in the air above the
stones.

## A chart of the morning's counts

The scene and its data are one object here, so we describe the figure as a
composition of marks. `Chart` is the recipe, `Mark` is one layer of it, and the
`+` operator stacks layers. `mxv.points` returns a `Chart` holding a single
`PointsMark`, so a one-layer figure is already a chart.

```python {run export=examples/plaza_measures.py}
import manifoldx as mx
import manifoldx.viz as mxv
import numpy as np
from manifoldx.components import Material, Transform
from manifoldx.resources import PointLight, StandardMaterial, cube
from PIL import Image

HALF = 20.0
rng = np.random.default_rng(7)
N = 500
spots = np.column_stack(
    [
        rng.uniform(-19.0, 19.0, N),
        np.full(N, 0.10),
        rng.uniform(-19.0, 19.0, N),
    ]
).astype(np.float32)


def hotspot(cx, cz, sx, sz):
    """A smooth bump over the square, in metres."""
    dx = (spots[:, 0] - cx) / sx
    dz = (spots[:, 2] - cz) / sz
    return np.exp(-(dx**2 + dz**2))


visits = 0.4 + 4.5 * hotspot(-3, 2, 4, 3) + 2.2 * hotspot(10, -8, 3, 3) + 0.8 * hotspot(0, 14, 8, 2)
visits = np.clip(visits + rng.normal(0.0, 0.15, N), 0.0, None).astype(np.float32)

dwell = 30.0 + 90.0 * hotspot(-3, 2, 5, 4) + rng.normal(0.0, 5.0, N)
radius = (0.30 + np.clip(dwell, 10.0, 150.0) / 350.0).astype(np.float32)

visits_color = mxv.color(visits, cmap="turbo", domain=(0.0, 6.0), title="Visits / min")

chart = (
    mxv.points(
        positions=mxv.position(spots),
        color=visits_color,
        size=mxv.size(radius),
    )
    + mxv.mesh(
        geometry=cube(40, 0.2, 40),
        material=StandardMaterial("#5c5242", roughness=0.9),
        position=(0.0, -0.12, 0.0),
    )
    + mxv.mesh(
        geometry=cube(4, 1, 4),
        material=StandardMaterial("#9aa8b8", roughness=0.6),
        position=(0.0, 0.5, 0.0),
    )
    + mxv.lights(
        [
            PointLight(color="#fff3dd", intensity=1300.0, position=(0.0, 16.0, 4.0)),
            PointLight(color="#bcd4ff", intensity=800.0, position=(-14.0, 10.0, -12.0)),
        ]
    )
)

engine = chart.engine
engine.background_color = (0.32, 0.38, 0.48)
engine.camera.fit(25.0, center=(0.0, 1.5, 0.0), azimuth=45, elevation=32, margin=0.95)
print(type(chart).__name__, type(chart.marks[0]).__name__)
print(
    type(mxv.position(spots)).__name__,
    type(visits_color).__name__,
    type(mxv.size(radius)).__name__,
)
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
```

The three encodings are channel objects. A bare array is wrapped in a
`PositionChannel`, a `ColorChannel` or a `SizeChannel` with defaults; all three
inherit `Channel`, and `mxv.position`, `mxv.color` and `mxv.size` are the
explicit constructors. The colour channel is worth keeping in a variable: its
colormap, domain and title are what the legend will need later.

The mark does the ECS work that would otherwise be a long `engine.spawn`
call: it tags a batch of entities with `PointCloud` so the renderer
draws camera-facing sprites, stores one value per point in `ScalarValue`, one
world-space size in `Radius`, and gives the whole batch a single
`ColormapMaterial` that maps values between `vmin` and `vmax` through the
256-entry lookup table. All five hundred points are therefore one draw call,
and the arrays stay yours to change. The two `MeshMark`s stand for the plaza
itself. The floor is a thin box rather than a `plane` because `mxv.mesh`
places a geometry but does not rotate it; the box needs no rotation, and it is
dropped two centimetres so that the axis lines in the next section do not lie
exactly on its surface. `LightsMark` is a thin wrapper over
`engine.set_lights`, and the list is of point lights: a `DirectionalLight` in
this list does not light the meshes, because the sun is engine state set with
`set_sun`, not part of the light array. Without a light a `StandardMaterial`
is black, which is why the lights join the chart before the first frame. A
`Chart` has no size argument and draws at 800×600 in 0.10; when the frame size
matters, use a plain `Engine` as in chapter 1.

## A frame you can measure

A cloud of colour says nothing about distance. The figure needs three axes in
metres, a legend that decodes the colours, and a scale bar for a reader who
prints it. Each is another mark.

```python {run continue export=examples/plaza_measures.py}
cam = engine.camera
right = cam.get_right()
pixels, _ = cam.project([cam.target, cam.target + 5.0 * right], engine.w, engine.h)
five_metres = abs(pixels[1, 0] - pixels[0, 0]) * 2.0 / engine.w

chart = (
    chart
    + mxv.axes(extent=10.0)
    + mxv.legend(visits_color, position="top-right")
    + mxv.scale_bar(ndc_length=five_metres, label="5 m", position="bottom-left")
)
engine = chart.engine
engine.background_color = (0.32, 0.38, 0.48)
engine.camera.fit(25.0, center=(0.0, 1.5, 0.0), azimuth=45, elevation=32, margin=0.95)
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
```

`AxesMark` spawns three line entities, each carrying one of the unit
`axis_line_x`, `axis_line_y`, `axis_line_z` geometries, an `AxisFrame` tag and
an `AxisMaterial` with its colour. `Transform.scale` stretches each unit line
to the requested half-length, so `extent=10.0` gives ten metres either way
from the origin. The lines use wgpu's native line list and are one pixel wide
at any distance; the three end caps are ordinary labels.

`LegendMark` rasterizes the colormap into the label atlas with
`register_colormap_legend` and draws it with a screen-anchored `LabelMaterial`,
with a second screen-anchored label for the caption. Passing the colour channel
rather than a colormap name is what puts `0 → 6` under the title, so the
legend cannot drift from the points.

`ScaleBarMark` is a screen-anchored `AxisMaterial` line plus a caption. It
takes a label, not a distance: the mark cannot know how many metres a fraction
of the frame is worth. We compute the NDC length of five metres at the point
the camera looks at, using `Camera.get_right` and `Camera.project`, and divide
by the frame width. The bar is then correct at the centre of the square and a
little short at the near corner; it is a report aid, not a world object. If the
camera moves, rebuild the chart with a new length.

## Names for the places

The cloud shows where people stopped. It does not say that the bright
gathering on the right is the market, or that the quiet corner on the left is
the bench row. Text is not a mark in the shim, so this is where the figure
drops to the engine the chart built.

```python {run continue export=examples/plaza_measures.py}
atlas = engine.get_label_atlas()
places = [
    (0.0, 2.4, 0.0, "Fountain"),
    (10.0, 2.8, -9.0, "Market"),
    (-11.0, 2.2, 9.0, "Bench row"),
]
slots = [atlas.get_or_create(place[3]) for place in places]
print("slots", slots, "of", atlas.slice_count)

for (x, y, z, _), slot in zip(places, slots):
    engine.spawn(
        Material(mxv.LabelMaterial(pixel_width=240, pixel_height=60)),
        Transform(pos=(x, y, z)),
        mxv.TextLabel(index=slot),
        n=1,
    )

frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
```

`LabelTextureAtlas` is a texture array of 256 tiles, each 256×64 pixels, one
per unique string. `get_or_create` rasterizes a string once and returns its
slice index; asking for the same string again returns the same slot, and
`slice_count` reports how many seats are taken. The three names land in slots 6, 7 and 8,
after the axes and the two overlay labels. `TextLabel` stores the slot index
per entity, `LabelMaterial` says how large the label is on screen, and
`Transform.pos` anchors it in the world. The label is a billboard, so it
always faces the camera; the quad is 240 by 60 pixels, close to the 4:1 tile,
so the glyphs are crisp. `engine.get_label_atlas` is the same atlas the marks
used.

## The atlas has 256 seats

One morning of names fits easily. A month of changing labels does not, so it
is worth knowing where the wall is. In 0.10 the atlas refuses a 257th unique
string:

```python {run continue}
from manifoldx.viz import LabelTextureAtlas
from manifoldx.viz import AtlasOverflowError

full = LabelTextureAtlas()
for i in range(256):
    full.get_or_create(f"stop {i}")
print("tiles", full.slice_count)

try:
    full.get_or_create("one stop too many")
except AtlasOverflowError as error:
    print(error)
```

The cap is per atlas, and the engine owns one. Colormap legends take a seat
too. A live readout that prints a fresh number every frame can exhaust it in
seconds, which is why an instrumented scene buckets its strings ("47 %",
"48 %") or rewrites one slot instead of allocating a new one. `AtlasOverflowError`
is importable from `manifoldx.viz.text`; the package does not re-export it.

## Noise in the air

The noise survey is a value at every place and height, not a value at five
hundred points, so it wants a volume. The engine keeps the voxels; the entity
carries only the handle.

```python {run continue export=examples/plaza_measures.py}
NZ, NY, NX = 32, 12, 32
zs = np.linspace(-HALF, HALF, NZ, dtype=np.float32)
ys = np.linspace(0.0, 10.0, NY, dtype=np.float32)
xs = np.linspace(-HALF, HALF, NX, dtype=np.float32)
Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")

noise = (
    44.0
    + 30.0 * np.exp(-(((X - 9.0) ** 2 + (Z + 6.0) ** 2) / 25.0))
    + 24.0 * np.exp(-(((X + 4.0) ** 2 + (Z - 2.0) ** 2) / 16.0))
) * np.exp(-Y / 5.0) + 44.0 * (1.0 - np.exp(-Y / 5.0))
noise = np.ascontiguousarray(noise.astype(np.float32))

handle = engine.register_volume(noise, name="noise")
engine.spawn(
    mxv.Volume(volume_id=handle),
    Material(
        mxv.VolumeMaterial(
            cmap="inferno",
            vmin=49.0,
            vmax=82.0,
            opacity_stops=[(0.0, 0.0), (0.35, 0.02), (0.7, 0.08), (1.0, 0.22)],
            density_scale=5.0,
            max_steps=128,
        )
    ),
    Transform(pos=(0.0, 5.0, 0.0), scale=(40.0, 10.0, 40.0)),
    n=1,
)

frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
```

`engine.register_volume` takes a float32, C-contiguous array of shape
`(Nz, Ny, Nx)`, where `data[k, j, i]` is the voxel at local `(x, y, z) =
(i, j, k)`, and returns an integer handle. Behind it, a `VolumeRegistry` keeps
the array, marks it dirty and uploads it to a 3D texture on the frame that
needs it; `engine.update_volume` replaces the field with another array of the
same shape, which is how the next hour's survey arrives. The `Volume`
component is only the handle per entity. `VolumeMaterial` is the transfer
function: `vmin` and `vmax` map the values through the colormap, and
`opacity_stops` say how opaque each part of that range is, so the quiet
forty-four decibels at the edges disappear while the two sources glow. The
shader marches through a unit cube, so `Transform.scale` is the full size of
the box: forty metres across, ten high, centred five metres above the ground.
The field is raymarched per pixel rather than drawn per voxel, and `max_steps`
bounds the cost.

The result is the whole job in one frame: colour is visits per minute, size is
how long people stayed, the axes and the bar are in metres, the names are on
the places, and the noise hangs where it was measured.

## The finished program

Everything above is one chain, tangled by the guide into
`docs/examples/plaza_measures.py`. Run it without a window: the last
`render_frame` returns the finished figure, and the arrays at the top are
where the researcher's CSV goes. The classes are worth keeping straight by
what they answer. `Chart`, `Mark` and the six marks compose the figure;
`Channel`, `ColorChannel`, `SizeChannel` and `PositionChannel` say what the
data means; `PointCloud`, `ScalarValue`, `Radius` and `ColormapMaterial` are
the sprite layer; `TextLabel`, `LabelMaterial` and `LabelTextureAtlas` are the
text layer, with `AtlasOverflowError` as its limit; `AxisFrame` and
`AxisMaterial` draw the metre frame; `Volume`, `VolumeMaterial` and
`VolumeRegistry` carry the field. `MeshMark` puts the plaza itself under all of
it.
