# A still worth printing

The plaza is going on the cover of the year's report: one image, printed at A4,
looked at for a year. A frame from the window is enough while someone walks
through it, but print is less forgiving. The sky has to fill what the sun leaves
in shadow, the benches have to throw the sun's shadows on the stones, and the
water and the bronze have to hold what is around them. The raster engine
approximates all three every frame. `manifoldx.trace` follows the light instead.

## The square, as triangles

The tracer does not draw the engine's entities; it traces rays through
triangles. So the plaza is built the way the earlier chapters built it, with an
`Engine`, and then handed over as a `.glb`, the format chapter 11 reads and
writes. Meshes, transforms, `StandardMaterial`s, the camera and the lights
cross; the sky does not, because the sky is a choice for the still.

```python {run export=examples/plaza_still.py}
import tempfile
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane, sphere
from manifoldx.trace import Camera, Environment, Scene, Tracer
from PIL import Image

FLOOR = Transform.rotation(x=-np.pi / 2)

engine = mx.Engine("plaza-cover", width=480, height=270)
engine.set_sun(DirectionalLight(color="#ffe6c4", intensity=3.0, direction=(-0.45, -0.30, 0.45)))

paving = StandardMaterial(color="#b9ad98", roughness=0.9)
stone = StandardMaterial(color="#a8a296", roughness=0.7)
dark = StandardMaterial(color="#5d5850", roughness=0.8)
wood = StandardMaterial(color="#7b5a3a", roughness=0.6)
water = StandardMaterial(color="#4f7385", roughness=0.02, metallic=1.0)
bronze = StandardMaterial(color="#7a5a2e", roughness=0.3, metallic=1.0)
glass = StandardMaterial(color="#2b3440", roughness=0.12, metallic=0.5)
plaster = StandardMaterial(color="#c2b49c", roughness=0.95)

engine.spawn(Mesh(plane(40, 40)), Material(paving), Transform(rot=FLOOR))

for half in (9.0, 13.0):
    for side in (-1, 1):
        engine.spawn(
            Mesh(cube(2 * half + 0.25, 0.02, 0.25)),
            Material(dark),
            Transform(pos=(0, 0.01, side * half)),
        )
        engine.spawn(
            Mesh(cube(0.25, 0.02, 2 * half + 0.25)),
            Material(dark),
            Transform(pos=(side * half, 0.01, 0)),
        )

engine.spawn(Mesh(cube(4.0, 1.0, 0.3)), Material(stone), Transform(pos=(0, 0.5, -1.85)))
engine.spawn(Mesh(cube(4.0, 1.0, 0.3)), Material(stone), Transform(pos=(0, 0.5, 1.85)))
engine.spawn(Mesh(cube(0.3, 1.0, 3.4)), Material(stone), Transform(pos=(-1.85, 0.5, 0)))
engine.spawn(Mesh(cube(0.3, 1.0, 3.4)), Material(stone), Transform(pos=(1.85, 0.5, 0)))
engine.spawn(Mesh(cube(3.4, 0.12, 3.4)), Material(dark), Transform(pos=(0, 0.06, 0)))
engine.spawn(
    Mesh(plane(3.4, 3.4)),
    Material(water),
    Transform(pos=(0, 0.995, 0), rot=FLOOR),
)
engine.spawn(Mesh(cube(0.8, 1.6, 0.8)), Material(stone), Transform(pos=(0, 0.9, 0)))
engine.spawn(Mesh(sphere(0.55)), Material(bronze), Transform(pos=(0, 2.25, 0)))
```

The sun's `direction` is where its light travels, not where the sun is:
`(-0.45, -0.30, 0.45)` sends light down, towards -x and +z. The fountain is four
wall boxes and a bottom, with the water a plane at 0.995 m, just under the rim,
so it is visible from standing height and the rim still reads as a basin. Two
inlaid bands at 9 and 13 m give the paving a scale.

Two helpers keep the square from turning into a page of `engine.spawn` calls.
`bench` places a seat, a back and two legs and turns them to face the fountain;
`building` places a block with a plinth, a cornice and a grid of windows on its
plaza face. Neither of them knows anything about the tracer.

```python {run continue export=examples/plaza_still.py}


def bench(engine, at, yaw, stone, wood):
    """A seat, a back and two legs, turned to face the fountain."""
    x, z = at
    rot = Transform.rotation(y=yaw)
    along_x, along_z = np.cos(yaw), -np.sin(yaw)
    out_x, out_z = np.sin(yaw), np.cos(yaw)
    engine.spawn(Mesh(cube(1.9, 0.1, 0.55)), Material(wood), Transform(pos=(x, 0.46, z), rot=rot))
    engine.spawn(
        Mesh(cube(1.9, 0.55, 0.09)),
        Material(wood),
        Transform(pos=(x + 0.24 * out_x, 0.74, z + 0.24 * out_z), rot=rot),
    )
    for side in (-0.8, 0.8):
        engine.spawn(
            Mesh(cube(0.12, 0.46, 0.5)),
            Material(stone),
            Transform(pos=(x + side * along_x, 0.23, z + side * along_z), rot=rot),
        )


def building(engine, center, size, normal, plaster, stone, glass, window):
    """A block with a plinth, a cornice and a grid of windows on its plaza face."""
    x, _, z = center
    width, height, depth = size
    normal_x, normal_z = normal
    engine.spawn(
        Mesh(cube(width, height, depth)), Material(plaster), Transform(pos=(x, height / 2, z))
    )
    engine.spawn(
        Mesh(cube(width + 0.4, 0.3, depth + 0.4)),
        Material(stone),
        Transform(pos=(x, height + 0.15, z)),
    )
    engine.spawn(
        Mesh(cube(width + 0.3, 0.5, depth + 0.3)),
        Material(stone),
        Transform(pos=(x, 0.25, z)),
    )
    length = depth if normal_z else width
    columns = max(3, int(length // 2.6))
    for column in range(columns):
        offset = (column - (columns - 1) / 2) * (length / columns)
        for row in (3.4, 5.4, 7.4):
            if row > height - 1.5:
                continue
            if normal_z:
                engine.spawn(
                    Mesh(window),
                    Material(glass),
                    Transform(pos=(x + offset, row, z + normal_z * depth / 2)),
                )
            else:
                engine.spawn(
                    Mesh(window),
                    Material(glass),
                    Transform(
                        pos=(x + normal_x * width / 2, row, z + offset),
                        rot=Transform.rotation(y=np.pi / 2),
                    ),
                )
    for door in range(3):
        offset = (door - 1) * (length / 3.4)
        if normal_z:
            engine.spawn(
                Mesh(cube(1.5, 2.7, 0.2)),
                Material(glass),
                Transform(pos=(x + offset, 1.35, z + normal_z * depth / 2)),
            )
        else:
            engine.spawn(
                Mesh(cube(0.2, 2.7, 1.5)),
                Material(glass),
                Transform(pos=(x + normal_x * width / 2, 1.35, z + offset)),
            )


for corner in range(4):
    angle = corner * np.pi / 2
    bench(engine, (4.8 * np.sin(angle), 4.8 * np.cos(angle)), angle, stone, wood)

window = cube(1.2, 1.9, 0.16)
building(engine, (-13.5, 0, -22.5), (12, 9, 5), (0, 1), plaster, stone, glass, window)
building(engine, (0, 0, -22.5), (12, 10, 5), (0, 1), plaster, stone, glass, window)
building(engine, (13.5, 0, -22.5), (12, 8, 5), (0, 1), plaster, stone, glass, window)
building(engine, (-22.5, 0, -5), (5, 8, 18), (1, 0), plaster, stone, glass, window)
building(engine, (22.5, 0, -6), (5, 9, 16), (-1, 0), plaster, stone, glass, window)
engine.spawn(Mesh(cube(40, 1.0, 0.4)), Material(stone), Transform(pos=(0, 0.5, 20)))

path = Path(tempfile.gettempdir()) / "plaza-cover.glb"
report = engine.export_gltf(path)
scene = Scene.from_glb(path)

print(len(report), "things left out")
print(scene.triangle_count, len(scene.materials), len(scene.lights))
print(tuple(round(v, 1) for v in scene.bounds()[1]))
print([light.kind for light in scene.lights])
water_surface = next(m for m in scene.materials if m.roughness < 0.05)
print(
    water_surface.name, water_surface.metallic, tuple(round(c, 2) for c in water_surface.base_color)
)
```

`engine.export_gltf` writes the file and returns a report of anything it had to
leave out; for this scene the report is empty. `Scene.from_glb` reads it back as
a `Scene`: triangles in world space, one `material_ids` row per triangle, a list
of the tracer's `Material`s, a `Camera` and a list of `Light`s. The five prints
are the report's length, the triangle count, the far corner of the bounding box
(`Scene.bounds()`), the kinds of light, and one material.

All eight engine surfaces arrive named `StandardMaterial`, because the export
names a material after its class, so the values are what tell them apart.
`base_color` is linear: the water's `#4f7385` is `(0.08, 0.17, 0.23)` here.
`metallic` 1.0 with `roughness` 0.02 is the mirror the fountain needs.
`transmission` and `ior` turn a surface into glass, and `emission` makes it a
lamp, which is how you light a night still that the engine's `StandardMaterial`
cannot describe.

## The light

The engine's sun crossed the glb as a `Light`, and it is still the sun: a
directional light with a colour, an intensity and a direction. The raster
engine's 3.0 was a shading-unit number. The tracer reads `intensity` as
irradiance in W/m², so the still sets 20.0, a few times the sky's own horizontal
illuminance at this turbidity. Raise it for a harder sun, lower it for an
overcast afternoon.

The sky is an `Environment`: an equirectangular radiance map, and optionally a
sun disk. `Environment.sky` builds one with Preetham's clear-sky model. Its
arguments are the ones a photographer would reach for: `sun_direction`, towards
the sun; `turbidity`, where 2 is a clear sky and 10 is haze; and `ground`, the
reflectance of the lower half of the dome, so the horizon is not black.
`sun_ratio=0.0` is the important one. It leaves the sun out of the map, because
the `Light` above is already the sun; with the default ratio the sky carries a
second sun disk in the same direction and the plaza is lit twice.
`Environment.from_hdr` reads a photographed sky, `from_map` takes one from
`manifoldx.ibl`, and `uniform` is a flat studio dome, when the cover needs one of
those instead.

The `Camera` is the tracer's, not the window's: a `position`, a `target`, an
`up` and a vertical `fov` in degrees. At (6.2, 2.3, 7.6) looking at the fountain,
the basin fills the lower half and the north buildings and the sky fill the
rest. A `Tracer` takes the scene, a width and a height; it keeps six bounces and
seed 0 unless told otherwise. On construction it packs the scene into GPU
buffers and builds the acceleration structure.

That structure is a `BVH`, built by `bvh.build` in numpy: every triangle goes
into a tree of nested boxes, and a leaf lists a short run of triangles. The
shader walks the boxes and tests only the few triangles inside; without it every
ray would test all 5,300. The package says why the walk is in software: wgpu
exposes no ray queries, so it is a compute kernel. `node_count` and `depth()`
say how the tree came out, and `build_seconds` says what it cost. Assigning
`tracer.camera` later calls `reset()`, so the next `step` starts a fresh
accumulation.

```python {run continue export=examples/plaza_still.py}
sun = scene.lights[0]
sun.color = (1.0, 0.86, 0.68)
sun.intensity = 20.0

sun_direction = tuple(-v for v in sun.direction)
sky = Environment.sky(sun_direction, turbidity=2.4, sun_ratio=0.0, ground=(0.34, 0.30, 0.24))

camera = Camera(position=(6.2, 2.3, 7.6), target=(0, 1.2, -1.5), fov=48.0)
tracer = Tracer(scene, 480, 270, camera=camera, environment=sky)
print(type(tracer.bvh).__name__, tracer.bvh.node_count, tracer.bvh.depth())
print(f"built in {tracer.build_seconds:.3f} s, {tracer.width}x{tracer.height}")
```

## Samples

`step(n)` adds n path samples per pixel and returns the seconds it took;
`samples` counts them. One sample is one path through the pixel: the camera ray,
a bounce off the stone, a bounce off the water, a direction sampled from the sky.
At 8 samples the picture is already the right picture, and visibly made of
particles.

`image(denoise=False)` turns the accumulated linear radiance into 8-bit sRGB.
`auto_exposure` maps the log-average luminance to 0.18 and the ACES curve rolls
the highlights off. `radiance()` is the same buffer in linear floats, before
exposure and tone mapping, for a still that has to go into a composite.

```python {run continue export=examples/plaza_still.py}
seconds = tracer.step(8)
print(f"{tracer.samples} samples per pixel in {seconds:.1f} s")
Image.fromarray(tracer.image(denoise=False))
```

The long, soft shading under the benches is the sky; the crisp dark shadows are
the `Light`. The water is a mirror and the bronze sphere is another, and neither
is a screen-space trick.

## The denoiser

Sixteen samples is enough for the shapes and not for the surfaces.
`denoised(passes=5)` closes the gap. It divides the radiance by the first-hit
albedo, estimates a per-pixel variance, and runs five à-trous passes whose step
doubles each time, stopping at edges in normal, depth and luminance. It is a
reconstruction, not a blur: the fountain's silhouette and the shadow line stay
where they are. `image()` uses it unless you pass `denoise=False`.

```python {run continue export=examples/plaza_still.py}
seconds = tracer.step(8)
print(f"{tracer.samples} samples per pixel in {seconds:.1f} s")
noisy = tracer.image(denoise=False)
smooth = tracer.image()
Image.fromarray(np.concatenate([noisy, smooth], axis=1))
```

The left half is 16 samples without the filter, the right half with it. The
grain in the paving is gone; the edges are not.

## The still

Sixteen more samples, 32 in all, are enough for the cover. The block prints the
seconds it took; on the software Vulkan device that builds this guide it was
under ten. A print run is the same program with a larger `Tracer` and a longer
`step`, or
`manifoldx trace <file>.glb -o cover.png --size 1920x1080 --spp 512` on the
exported file.

```python {run continue export=examples/plaza_still.py}
seconds = tracer.step(16)
still = tracer.image()
print(f"{tracer.samples} samples per pixel in {seconds:.1f} s")
Image.fromarray(still)
```

The still has what the window approximated. The facades turned away from the
sun are blue with sky light instead of black, the benches and the basin throw the
sun's shadows across the stones, and the water and the bronze hold the sky and
the buildings. Save it for the printer with Pillow:

```python
Image.fromarray(still).save("plaza-cover.png")
```

The same kernels run in a web page. `manifoldx.trace.web.copy_to` writes
`trace.js` and the WGSL beside it; a page builds a scene with `loadGlb`, creates
a tracer with `createTracer`, and calls `start` when the camera stops, drawing
the denoised image as it converges. Python and the page share the kernels, the
BVH layout and the parameter buffer, so they agree by statistics rather than bit
for bit.

The next chapter puts the plaza in front of an agent, with the dev channel and
the web page.
