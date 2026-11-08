# Light and surface

The plaza is paved and the fountain is in place, but at dusk it still looks
like painted cardboard: one grey floor, one grey basin, and no shadows. The
caretaker wants the square to read after sunset, with stone that looks like
stone, the brass handrail of a bench catching the last light, a street lamp and
a spotlight on the fountain, and long shadows from the low sun.

## A material is a shader

In chapter 1 an entity got a `Material` component. That component is only a
handle: it writes a number into the entity store, and the number points at a
surface. The surface itself is a subclass of `Material` in
`manifoldx.resources`, which is an abstract base class. The base class asks
each subclass for the WGSL it draws with (`_compile`), the uniform fields it
uses (`uniform_type`), and the bytes of one instance (`get_data`); each family
also carries a `binding_slot` tag. The engine compiles a pipeline once per
vertex layout and material type, and a `MaterialRegistry` keeps every material
object by the id the component stored.

Four families ship with the library. The same red cube, left to right, is
drawn with `BasicMaterial`, `FlatMaterial`, `PhongMaterial` and
`StandardMaterial` under one sun.

```python {run}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import (
    BasicMaterial,
    DirectionalLight,
    FlatMaterial,
    PhongMaterial,
    StandardMaterial,
    cube,
)

engine = mx.Engine("materials", width=480, height=270)
engine.background_color = (0.10, 0.12, 0.18)
engine.camera.set_pose(position=(0.0, 1.5, 5.5), target=(0.0, 0.5, 0.0))
engine.set_sun(DirectionalLight(color="#fff1d6", intensity=3.0, direction=(-0.5, -1.0, -0.4)))

swatch = cube(1, 1, 1)
for i, material in enumerate(
    [
        BasicMaterial(color="#c05a4a"),
        FlatMaterial(color="#c05a4a"),
        PhongMaterial(color="#c05a4a", shininess=128.0),
        StandardMaterial(color="#c05a4a", roughness=0.55),
    ]
):
    engine.spawn(Mesh(swatch), Material(material), Transform(pos=(i * 1.4 - 2.1, 0.5, 0.0)))
Image.fromarray(engine.render_frame())
```

The first cube has a fixed key light from (1, 1, 1) hard-coded in the shader:
`BasicMaterial` ignores the sun, the sun's colour and the shadows, and is the
cheap surface for something that only has to read as a shape. The second,
`FlatMaterial`, writes its colour out with no lighting at all; the engine uses
it for id passes, where each pixel must carry an object's exact colour, and it
also makes a lamp bulb read as a source rather than a lit object. The third,
`PhongMaterial`, is meant to add a specular lobe, but in 0.10 it compiles
`BasicMaterial`'s shader and its `shininess` does nothing: the two draw the
same pixels. The fourth, `StandardMaterial`, is the physically based one. It
reads the sun, the point lights and the spot, and its `roughness` and
`metallic` decide how the light leaves the surface. That is the material the
plaza needs.

A material can also require geometry attributes it does not get. `cube`
carries positions, normals and indices but no UVs, so a `StandardMaterial`
with an `albedo_map` cannot sample it:

```python {run continue}
from manifoldx.renderer import MaterialGeometryMismatchError
from manifoldx.textures import texture_from_array

uv_free = cube(1, 1, 1)
grey = texture_from_array(engine, np.full((2, 2, 4), 180, dtype=np.uint8))
engine.spawn(
    Mesh(uv_free),
    Material(StandardMaterial(color="#ffffff", albedo_map=grey)),
    Transform(pos=(0.0, 2.5, 0.0)),
)
try:
    engine.render_frame()
except MaterialGeometryMismatchError as error:
    print(error)
```

The engine raises `MaterialGeometryMismatchError`, a `ValueError`, from the
render pipeline in `manifoldx.renderer`, not from `spawn`: the mesh is
accepted, and the first frame that tries to build its pipeline fails with the
geometry's id. Chapter 5 adds UVs and textures.

## A basin in the evening sun

The plaza starts with the fountain and one plane for the ground. The plane is
dark, and will become the joint between paving stones; for now it is only
something for the sun to throw a shadow onto.

```python {run export=examples/evening_light.py}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import (
    DirectionalLight,
    FlatMaterial,
    PointLight,
    SpotLight,
    StandardMaterial,
    cube,
    plane,
    sphere,
)

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.06, 0.08, 0.14)
FLOOR = Transform.rotation(x=-np.pi / 2)

engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#5c5850", roughness=0.95)),
    Transform(pos=(0, -0.02, 0), rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#a8a196", roughness=0.72)),
    Transform(pos=(0, 0.5, 0)),
)
engine.spawn(
    Mesh(plane(3.4, 3.4)),
    Material(StandardMaterial(color="#3e6a72", roughness=0.5)),
    Transform(pos=(0, 1.005, 0), rot=FLOOR),
)

engine.camera.set_pose(position=(6.0, 4.2, 7.5), target=(0.5, 0.9, 0.0))
engine.set_sun(DirectionalLight(color="#ffd0a0", intensity=3.2, direction=(-0.7, -0.3, 0.35)))
engine.enable_shadows(resolution=1024)
Image.fromarray(engine.render_frame())
```

A material colour such as `"#a8a196"` is sRGB, the same convention as a CSS
colour or a texture texel. `material_rgb` decodes it with `srgb_to_linear`, the
shader works in linear light, and the frame is encoded back to sRGB once, so
the hex you write is the colour you see on a lit, neutral surface.

`DirectionalLight` is the sun; its `direction` is the direction the light
travels, not the place it sits, so `(-0.7, -0.3, 0.35)` is a low sun off to the
camera's right. `set_sun` stores it on the engine rather than in the
entity store. `enable_shadows` records a depth map of everything the sun sees.
Its `resolution` is the map's side in texels, `bias` lifts a surface off its
own depth to stop acne, and `pcf_radius` widens the small blur that softens the
edge. With `auto_fit=True`, the default, the engine fits the light's box to the
scene's bounding sphere every frame, so the whole forty-metre floor is covered
without a manual `target` and `extent`; pass `auto_fit=False` when you want a
tighter box around one part of the square. The basin now throws a long shadow
to the left; the water is a dark teal because that is the colour we gave it,
and its roughness catches a band of sun along the far rim.

One convention to keep straight: material colours are decoded from sRGB, but
light colours are not. `PointLight.get_data` and `DirectionalLight.get_data`
divide the hex by 255 and use it as linear light, so a warm lamp is a little
more saturated than the same hex on a stone. Choose light colours by eye.

## Stone

A single plane is not paving. Real paving is many slabs with joints, each one
a slightly different age, and the joints are what make the grid read at a
distance. The square is forty metres on a side, so a two-metre grid gives
nineteen slabs across: 361 of them, each 1.9 m wide, sitting on the dark plane
that now shows through as the joint.

All 361 share one `plane` geometry and two material objects. The
`GeometryRegistry` uploads a geometry once and hands back a numeric id;
`register` is idempotent for the same object, and `get` returns the geometry
behind an id. The `MaterialRegistry` does the same for materials. The
`RenderPipeline` collects entities that share a geometry and a material object
and draws each group with one instanced call, so the number of entities is not
the number of draw calls. When the frame is built, it reads the matrix of every
entity from a `TransformCache`; `mark_dirty` is the seam for saying which rows
changed and `get_transforms` returns the cached mat4 for the rows you ask for.
In 0.10 `RenderPipeline.run` marks every row dirty at the start of each frame,
so the cache still rebuilds all of them; that is a detail to know before
expecting a still scene to cost less than a moving one. When a frame is drawn,
`run` refreshes the transforms, `render_shadow_map` draws the depth pass,
`render` draws the batches, and `end_frame` closes the frame.

```python {run continue export=examples/evening_light.py}
stone_a = StandardMaterial(color="#a49d90", roughness=0.92)
stone_b = StandardMaterial(color="#969084", roughness=0.88)
slab = plane(1.9, 1.9)
for ix in range(-9, 10):
    for iz in range(-9, 10):
        engine.spawn(
            Mesh(slab),
            Material(stone_a if (ix + iz) % 2 == 0 else stone_b),
            Transform(pos=(ix * 2.0, 0.0, iz * 2.0), rot=FLOOR),
        )

frame = engine.render_frame()
geometry_id = engine._geometry_registry.register(slab)
print("slab geometry:", geometry_id, engine._geometry_registry.get(geometry_id) is slab)
material_id = engine._material_registry.register(stone_a)
print("stone material:", material_id, engine._material_registry.get(material_id) is stone_a)
print("entities:", len(engine.store.get_component_view(["Transform"])))
print("instances drawn:", engine._render_pipeline.drawn)
cache = engine._render_pipeline._transform_cache
print("cached matrices:", cache.get_transforms(engine.store, np.arange(4)).shape)
Image.fromarray(frame)
```

The print is the point: there is one geometry id for every slab, one material
id for the lighter stone, and `drawn` counts the instances that survived
frustum culling, not the entities in the store. The slabs go to the GPU as two
instanced batches, one per stone material. The ground, the water and the slabs
share one pipeline because they share a vertex layout; the fountain's cube,
which has no UVs, gets another. The next time a chapter spawns a thousand
things, none of this changes; that is why the engine has registries instead of
a scene graph to walk.

## Metal

The bench is where the surface has to do real work. `StandardMaterial` takes
`metallic` between 0 and 1. At 1 there is no diffuse colour left: the albedo
tints the reflection, and a smooth surface shows a tight highlight where the
sun or a lamp lands. The brass rails here are `metallic=1` and
`roughness=0.05`; the wooden seat is a rough dielectric, `metallic=0` and
`roughness=0.7`. Without an environment map a metal has almost nothing to
reflect except the lights, so most of the rail is dark and only its edge and
its top catch the sun. That is what polished metal looks like under a clear
sky; chapter 5 gives it a sky and the surrounding buildings to reflect.

The camera moves in for this frame because the rail is small, then goes back
to the wide view for the lamp.

```python {run continue export=examples/evening_light.py}
wood = StandardMaterial(color="#6b4a2f", roughness=0.7)
brass = StandardMaterial(color="#e8b860", metallic=1.0, roughness=0.05)
engine.spawn(
    Mesh(cube(0.7, 0.12, 2.4)),
    Material(wood),
    Transform(pos=(4.2, 0.48, 2.6)),
)
for dz in (-1.0, 1.0):
    engine.spawn(
        Mesh(cube(0.6, 0.48, 0.12)),
        Material(brass),
        Transform(pos=(4.2, 0.24, 2.6 + dz)),
    )
engine.spawn(
    Mesh(cube(0.08, 0.08, 2.4)),
    Material(brass),
    Transform(pos=(4.55, 0.9, 2.6)),
)
for dz in (-1.0, 1.0):
    engine.spawn(
        Mesh(cube(0.08, 0.5, 0.08)),
        Material(brass),
        Transform(pos=(4.55, 0.65, 2.6 + dz)),
    )

engine.camera.set_pose(position=(7.2, 1.8, 5.2), target=(4.2, 0.6, 2.6))
frame = engine.render_frame()
engine.camera.set_pose(position=(6.0, 4.2, 7.5), target=(0.5, 0.9, 0.0))
Image.fromarray(frame)
```

## The street lamp

The lamp is a `PointLight`: a position, a colour and an intensity that falls
with the square of the distance. The shader divides the intensity by
`distance * distance`, so 110 is a reasonable lamp four metres above the
paving and 10 would hardly show. `set_lights` replaces the whole list and
`add_light` appends one; the renderer reads the first four, and a fifth is
ignored. The constructor also accepts `distance` and `decay`, but in 0.10
`PointLight.get_data` packs only position, colour and intensity, so those two
arguments change nothing. And a point light casts no shadow: the only shadow
casters are the sun and the spot.

The globe of the lamp is a `FlatMaterial`. It is not lit by the lamp, it is
the lamp: its colour goes out unchanged, which is exactly what a bright bulb
should do.

```python {run continue export=examples/evening_light.py}
dark = StandardMaterial(color="#2a2c30", roughness=0.4, metallic=0.6)
engine.spawn(
    Mesh(cube(0.14, 4.2, 0.14)),
    Material(dark),
    Transform(pos=(-4.5, 2.1, 3.4)),
)
engine.spawn(
    Mesh(cube(0.9, 0.1, 0.1)),
    Material(dark),
    Transform(pos=(-4.1, 4.2, 3.4)),
)
engine.spawn(
    Mesh(sphere(0.2, 20)),
    Material(FlatMaterial("#ffe2b0")),
    Transform(pos=(-3.7, 4.02, 3.4)),
)

lamp = PointLight(color="#ffb464", intensity=110.0, position=(-3.7, 4.02, 3.4))
engine.set_lights([lamp])
Image.fromarray(engine.render_frame())
```

## One shadow, one caster

The fountain also gets a `SpotLight`: a cone with an inner and an outer angle,
in radians, and a distance cutoff. `set_spot` stores the single spot on the
engine. Inside `inner_angle` the cone is at full strength, it falls smoothly to
nothing at `outer_angle`, and past `distance` it is cut off; a `distance` of 0
means thirty metres. The street lamp's globe is the fixture, and the spot
points from it across the square at the water.

There is a catch worth knowing before the first frame. `enable_shadows` keeps
one shadow map, not one per light, and the renderer gives it to the spot if a
spot is set and to the sun otherwise. Adding the spot therefore takes the
sun's shadows away even while the sun is still up, and it does so even for a
spot with intensity 0. Below, the sun is still at 3.2 and lights the stone,
but the fountain's long shadow has vanished: the map now belongs to the spot,
and the pool of light at the fountain is the spot's.

```python {run continue export=examples/evening_light.py}
spot = SpotLight(
    color="#ffe6c0",
    intensity=350.0,
    position=(-3.7, 4.02, 3.4),
    direction=(3.7, -3.1, -3.4),
    inner_angle=0.12,
    outer_angle=0.32,
    distance=12,
)
engine.set_spot(spot)
Image.fromarray(engine.render_frame())
```

After dark the calculation changes. The sun is turned down to a cold sliver,
the lamp and the spot carry the square, and the spot's shadow is the one that
matters. The camera comes in to the fountain, where the cone lands.

```python {run continue export=examples/evening_light.py}
engine.set_sun(DirectionalLight(color="#8fa6d8", intensity=0.08, direction=(-0.7, -0.3, 0.35)))
lamp.intensity = 60.0
spot.intensity = 700.0
engine.camera.set_pose(position=(5.5, 3.2, 7.5), target=(0.0, 0.8, 0.0))
Image.fromarray(engine.render_frame())
```

The finished program is `docs/examples/evening_light.py`: it builds the stone,
the metal, the lamp and the spot, and draws the plaza at dusk and after dark.
The next chapter adds the sky that the metal has been waiting to reflect, the
water in the basin, and textures for the stone.
