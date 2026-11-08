# Sky, water and textures

The plaza is paved in one flat colour, the fountain basin is a dry stone box, and
behind it the engine clears the screen to a wall of blue. A visitor would not
mistake any of it for a place. The paving needs a surface, the basin needs water,
the bronze needs something to reflect, and the sky needs an atmosphere, clouds by
day and stars by night.

## A texture is an array

A surface's colour changes faster than geometry can: at the scale of a paving
stone, the shape is flat, but the stone is not one colour. A *texture* is an image
the fragment shader samples through the geometry's UV coordinates. Upload a
`(height, width, 3)` or `(height, width, 4)` uint8 array with
`texture_from_array` and it returns a `TextureHandle` — the reference you hand to
a material as `albedo_map`. The handle holds the GPU texture, its view and sampler,
its pixel size, and the file it came from, if any; the engine's `TextureRegistry`
owns those resources for as long as the engine lives, so the handle stays cheap to
copy.

A texture needs the GPU device, and the device is opened by the engine's first
frame. In an interactive program you would upload textures inside a `startup`
handler; for a still, one throwaway `render_frame` is enough.

The paving is thirty-two flags on a side over the forty-metre square: about 1.25
metres each. Each flag gets a tint, the joints are darker, and a little noise
keeps the stones from looking printed.

```python {run}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, plane
from manifoldx.textures import texture_from_array, write_texture

FLOOR = Transform.rotation(x=-np.pi / 2)


def paving(n=1024, stones=32, seed=7):
    """A 40 m square of stone flags, as an sRGB uint8 image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    gx, gy = xx * stones, yy * stones
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - ix, gy - iy
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    grout = edge < 0.045
    tint = rng.random((stones, stones, 1))
    stone = np.array([0.72, 0.66, 0.55], np.float32)
    img = stone * (0.82 + 0.30 * tint[ix % stones, iy % stones])
    img = np.clip(img + rng.normal(0, 0.035, (n, n, 1)), 0, 1)
    img[grout] *= 0.55
    return np.round(img * 255).astype(np.uint8)


engine = mx.Engine("paving", width=480, height=270)
engine.set_sun(DirectionalLight(color="#fff4e8", intensity=3.2, direction=(-0.25, -0.72, -0.45)))
engine.render_frame(supersample=1)  # open the device for the upload

flags = texture_from_array(engine, paving())
print(flags.size, flags.source)

engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#ffffff", roughness=0.85, albedo_map=flags)),
    Transform(rot=FLOOR),
)
engine.camera.set_pose((10, 9, -10), (10.01, 0, -10))
Image.fromarray(engine.render_frame(supersample=1))
```

`flags.size` is `(1024, 1024)` and `flags.source` is `None`, because the image
never touched a file. The material's `color` stays white: the map supplies the
albedo and the roughness stays in the material. The upload builds a full mip
chain, samples trilinearly with anisotropic filtering, and repeats outside the
UV square.

An image from disk goes through `load_texture`, which decodes any format Pillow
knows into the same sRGB upload. A texture larger than the device's
`max_texture_dimension_2d` is refused with `TextureSizeError` before anything is
sent to the GPU. To change part of a texture in place, `write_texture` overwrites
a rectangle of pixels: here a contractor replaces four cracked flags with fresh
ones.

```python {run continue}
patch = np.full((64, 64, 4), 0, np.uint8)
patch[..., :3] = (214, 205, 186)
patch[..., 3] = 255
patch[0:3, :, :3] = (95, 88, 76)
patch[:, 0:3, :3] = (95, 88, 76)
patch[30:34, :, :3] = (95, 88, 76)
patch[:, 30:34, :3] = (95, 88, 76)

before = Image.fromarray(engine.render_frame(supersample=1)).resize((240, 270))
write_texture(engine, flags, patch, origin=(768, 768))
after = Image.fromarray(engine.render_frame(supersample=1)).resize((240, 270))
Image.fromarray(np.concatenate([np.asarray(before), np.asarray(after)], axis=1))
```

The right half is the repair. In 0.10, `write_texture` changes only the largest
mip level; a material samples the smaller levels when the texture is far away, so
the patch is crisp up close and fades out at a distance.

Not every surface starts as pixels. A plaza medallion is a drawing, and vector
art stays sharp at any size: `texture_from_svg` rasterises an SVG path, markup or
bytes with thorvg (the `svg` extra) and uploads the result like any other texture.
Transparent pixels stay transparent, and a `StandardMaterial` discards texels with
alpha below one half, so the paving shows through around the disc.

```python {run}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, plane
from manifoldx.textures import texture_from_svg

FLOOR = Transform.rotation(x=-np.pi / 2)

MEDALLION = """
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <circle cx="256" cy="256" r="250" fill="#3c4a50"/>
  <circle cx="256" cy="256" r="226" fill="none" stroke="#c9a25a" stroke-width="12"/>
  <circle cx="256" cy="256" r="160" fill="none" stroke="#c9a25a" stroke-width="6"/>
  <g stroke="#c9a25a" stroke-width="5">
    <line x1="256" y1="30" x2="256" y2="482"/>
    <line x1="30" y1="256" x2="482" y2="256"/>
    <line x1="96" y1="96" x2="416" y2="416"/>
    <line x1="416" y1="96" x2="96" y2="416"/>
  </g>
  <path d="M256 140 C300 200 330 232 330 272 A74 74 0 1 1 182 272 C182 232 212 200 256 140 Z"
        fill="#7fa8bf" stroke="#c9a25a" stroke-width="8"/>
</svg>
"""

engine = mx.Engine("inlay", width=480, height=270)
engine.set_sun(DirectionalLight(color="#fff4e8", intensity=3.2, direction=(-0.25, -0.72, -0.45)))
engine.render_frame(supersample=1)

inlay = texture_from_svg(engine, MEDALLION, 512)
print(inlay.size)

engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#9a958c", roughness=0.9)),
    Transform(rot=FLOOR),
)
engine.spawn(
    Mesh(plane(5, 5)),
    Material(StandardMaterial(color="#ffffff", roughness=0.55, albedo_map=inlay)),
    Transform(pos=(0, 0.02, 0), rot=FLOOR),
)
engine.camera.set_pose((3, 5, 9), (0, 0, 0))
Image.fromarray(engine.render_frame(supersample=1))
```

Rasterising is not free, but a GUI that rebuilds the same icon every frame should
not pay for it every frame. The SVG module keeps the last 64 MB of rasterised
images in an `LruCache`, keyed by the document's content hash and the requested
size: `put` adds an entry, `get` refreshes its age, and the least recently used
entry is dropped when the budget is full. You meet it indirectly:
`texture_from_svg` and the GUI's images both go through it.

```python {run}
from manifoldx.svg import LruCache

cache = LruCache(budget=8)
cache.put("first", 1, nbytes=4)
cache.put("second", 2, nbytes=4)
cache.put("third", 3, nbytes=4)  # 12 bytes requested, budget 8: "first" goes
print(len(cache), cache.get("first"), cache.get("third"))
```

That prints `2 None 3`: the oldest entry was evicted, the newest is still there.

## Water

A fountain that is a stone box is a plinth. The water is its own material,
`WaterMaterial`, because water is not a rough dielectric: it is dark in the deep,
bright at a grazing angle, and it throws a small, hard sun glint. The shader takes
a deep colour and a `fresnel_power`; it adds the sun's diffuse term by day, fades
to black at night, mixes in a fixed sky tint as the view turns grazing, and paints
a sharp highlight where the half-vector meets the surface.

```python {run}
import numpy as np

import manifoldx as mx
from PIL import Image
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, WaterMaterial, cube, plane

FLOOR = Transform.rotation(x=-np.pi / 2)
GROUND = 50_000.0

engine = mx.Engine("water", width=480, height=270)
engine.background_color = (0.01, 0.03, 0.10)
engine.camera.set_pose((4.5, GROUND + 3.2, 4.5), (0, GROUND + 0.9, 0))
engine.camera.near = 0.2
engine.camera.far = 500_000.0
engine.set_sun(DirectionalLight(color="#fff4e8", intensity=3.2, direction=(-0.25, -0.72, -0.45)))

engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#8a8170", roughness=0.9)),
    Transform(pos=(0, GROUND, 0), rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#9a958c", roughness=0.8)),
    Transform(pos=(0, GROUND + 0.5, 0)),
)
engine.spawn(
    Mesh(plane(3.6, 3.6)),
    Material(WaterMaterial(color="#0a2352", fresnel_power=4.0)),
    Transform(pos=(0, GROUND + 1.01, 0), rot=FLOOR),
)
Image.fromarray(engine.render_frame(supersample=1))
```

The basin here is a solid block with the water plane laid on top; in the finished
program the walls are four boxes, so the water sits inside them. The reflection in
the water is the shader's own sky tint, not the environment map that comes next;
water has no roughness parameter to blur an environment with.

## Reflections from an environment

A polished surface needs to know what surrounds it. Image-based lighting (IBL)
does that with one equirectangular image of the whole sphere of directions,
linear colour, stored as a float array in an `EnvironmentMap`. Build one
procedurally with `from_sky` (a zenith, horizon and ground colour), `from_color`
(a uniform ambient), `from_image` (any Pillow image, linearised) or `from_hdr` (a
Radiance file). `engine.set_environment` takes the map, a preset name such as
`"studio"`, `"sky"`, `"neutral"` or `"dark"`, or `None` to switch it off.

The map is not sampled directly. On first use the engine filters it into an
irradiance cubemap for diffuse light and a prefiltered mip chain for specular
reflections, and caches the result on disk keyed by the data (under
`~/.cache/manifoldx/ibl`, or `$MANIFOLDX_CACHE_DIR`). The precompute is seconds of
CPU: a window shows a loading spinner, and `render_frame` waits for it. Changing
`intensity` on the map reuses the same precomputation.

Here is the same bronze sphere twice: on the left under a sky gradient, on the
right with the environment switched off. The sun still makes its highlight, but
the sphere has nothing to reflect.

```python {run}
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.ibl import EnvironmentMap
from manifoldx.resources import DirectionalLight, StandardMaterial, plane, sphere
from manifoldx.textures import load_texture

FLOOR = Transform.rotation(x=-np.pi / 2)


def patina(n=256, seed=3):
    """A weathered bronze map, as an sRGB uint8 image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    noise = np.sin(12 * xx + 2 * np.sin(9 * yy)) * np.sin(11 * yy + 3 * np.cos(7 * xx))
    m = (noise + 1) * 0.5
    bronze = np.array([0.72, 0.45, 0.20], np.float32)
    green = np.array([0.32, 0.55, 0.42], np.float32)
    img = bronze * (1 - 0.55 * m[..., None]) + green * (0.55 * m[..., None])
    img = np.clip(img + rng.normal(0, 0.03, (n, n, 1)), 0, 1)
    return np.round(img * 255).astype(np.uint8)


path = Path(tempfile.gettempdir()) / "plaza_patina.png"
Image.fromarray(patina()).save(path)

engine = mx.Engine("ibl", width=240, height=270)
engine.background_color = (0.02, 0.02, 0.03)
engine.camera.set_pose((0, 1.4, 3.8), (0, 1.0, 0))
engine.camera.near = 0.1
engine.camera.far = 200.0
engine.set_sun(DirectionalLight(color="#ffffff", intensity=2.0, direction=(-0.5, -0.8, -0.4)))
engine.render_frame(supersample=1)

weathered = load_texture(engine, path)
print(weathered.size, weathered.source.name)

engine.spawn(
    Mesh(plane(8, 8)),
    Material(StandardMaterial(color="#6f6a62", roughness=0.9)),
    Transform(rot=FLOOR),
)
engine.spawn(
    Mesh(sphere(0.9, 48)),
    Material(
        StandardMaterial(
            color="#ffffff", metallic=1.0, roughness=0.12, albedo_map=weathered
        )
    ),
    Transform(pos=(0, 1.0, 0)),
)

engine.set_environment(
    EnvironmentMap.from_sky(
        zenith=(0.08, 0.20, 0.55),
        horizon=(0.65, 0.78, 0.95),
        ground=(0.16, 0.14, 0.11),
        height=16,
        width=32,
    )
)
lit = engine.render_frame(supersample=1)
engine.set_environment(None)
unlit = engine.render_frame(supersample=1)
Image.fromarray(np.concatenate([lit, unlit], axis=1))
```

`load_texture` printed the size and the file name it remembered, `(256, 256)
plaza_patina.png`; the handle is the same one a material takes. Metallic at one
means the albedo tints the reflection rather than supplying a diffuse colour;
roughness 0.12 is what keeps the sky's reflection readable instead of a blur.

## The sky

The sky is a sphere around the plaza with `AtmosphereScatteringMaterial` on it.
For every pixel the shader marches a ray through a Rayleigh and Mie atmosphere,
sampling the sunlight that reaches each point, so the blue, the pale horizon and
the warm band at sunset all come from the same sun direction. It is written for a
planet: it measures altitude above a sphere of `ground_radius` centred on the
world origin. To give the plaza a horizon, the program puts the whole plaza at
`GROUND` metres above that centre, on a small world of radius fifty kilometres;
that is why every position below carries the offset.

```python {run}
import numpy as np

import manifoldx as mx
from PIL import Image
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import (
    AtmosphereMaterial,
    AtmosphereScatteringMaterial,
    BasicMaterial,
    CloudMaterial,
    DirectionalLight,
    StandardMaterial,
    cube,
    plane,
    sphere,
)
from manifoldx.sky import starfield

FLOOR = Transform.rotation(x=-np.pi / 2)
GROUND = 50_000.0
SKY_TOP = GROUND + 4_000.0

engine = mx.Engine("sky", width=480, height=270)
engine.background_color = (0.01, 0.03, 0.10)
engine.camera.set_pose((7, GROUND + 3.5, 7), (0, GROUND + 4.2, 0))
engine.camera.near = 0.3
engine.camera.far = 500_000.0
engine.set_sun(
    DirectionalLight(color="#fff4e8", intensity=3.2, direction=(-0.25, -0.72, -0.45))
)

engine.spawn(
    Mesh(plane(4000, 4000)),
    Material(StandardMaterial(color="#7d7564", roughness=1.0)),
    Transform(pos=(0, GROUND - 0.2, 0), rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#9a958c", roughness=0.8)),
    Transform(pos=(0, GROUND + 0.5, 0)),
)
engine.spawn(
    Mesh(GeoMesh.icosphere(4, SKY_TOP).to_geometry()),
    Material(AtmosphereScatteringMaterial(GROUND, SKY_TOP, intensity=0.5)),
    Transform(),
)
cloud = engine.spawn(
    Mesh(GeoMesh.icosphere(4, 150).to_geometry()),
    Material(CloudMaterial(coverage=0.52, softness=0.12, freq=14.0, opacity=0.8)),
    Transform(pos=(0, GROUND, 0)),
)

# Spawned now, invisible by day: the material fades the stars under a daylit sky.
starfield(
    engine,
    count=1400,
    radius=300_000.0,
    seed=42,
    ground_radius=GROUND,
    atmo_top=SKY_TOP - GROUND,
)
Image.fromarray(engine.render_frame(supersample=1))
```

The clouds are a second sphere, this one 150 metres above the plaza, with
`CloudMaterial`: a five-octave fbm pattern in object space, thresholded into
coverage, alpha-blended and lit by the sun. Because the pattern lives in the
shell's own space, rotating the cloud entity drifts the clouds without rebuilding
the texture. `StarfieldMaterial` is the material behind `starfield`: point sprites
on a sphere, with a per-star brightness and size, faded by the sun's angle and the
camera's altitude. The stars were spawned in the day frame and discarded there;
they need no code to turn on.

At night `set_sun` puts the sun below the horizon, the scattering goes black, the
cloud entity's `destroy` removes its shell and the same starfield comes out. The
moon is a plain `BasicMaterial` sphere
with a slightly larger shell of `AtmosphereMaterial` around it: an unlit fresnel
rim glow, alpha-blended, for a body seen from outside. It is a cheap halo, not a
sky — the physical sky is the scattering shell.

```python {run continue}
engine.background_color = (0.004, 0.006, 0.02)
engine.set_sun(
    DirectionalLight(color="#aab8ff", intensity=0.35, direction=(0.4, 0.8, 0.3))
)
cloud.destroy()

moon = (-120_000, GROUND + 100_000, -120_000)
engine.spawn(
    Mesh(sphere(5000, 24)),
    Material(BasicMaterial("#f2ead0")),
    Transform(pos=moon),
)
engine.spawn(
    Mesh(sphere(5600, 24)),
    Material(AtmosphereMaterial("#9fc4ff", intensity=1.0)),
    Transform(pos=moon),
)
Image.fromarray(engine.render_frame(supersample=1))
```

## The finished program

The finished program puts the pieces in the order they need: one frame to open
the device, the paving and the medallion, the basin and its water, the bronze with
its environment, the sky shell and the clouds. The night branch, the starfield and
the moon are the same code path with the sun moved; the day frame below is what
the default call draws.

```python {run export=examples/sky_and_water.py}
"""The plaza's paving, fountain, sky and reflections, rendered offscreen."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.ibl import EnvironmentMap
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import (
    AtmosphereMaterial,
    AtmosphereScatteringMaterial,
    BasicMaterial,
    CloudMaterial,
    DirectionalLight,
    StandardMaterial,
    WaterMaterial,
    cube,
    plane,
    sphere,
)
from manifoldx.sky import starfield
from manifoldx.textures import load_texture, texture_from_array, texture_from_svg

FLOOR = Transform.rotation(x=-np.pi / 2)
GROUND = 50_000.0
SKY_TOP = GROUND + 4_000.0

MEDALLION = """
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <circle cx="256" cy="256" r="250" fill="#3c4a50"/>
  <circle cx="256" cy="256" r="226" fill="none" stroke="#c9a25a" stroke-width="12"/>
  <circle cx="256" cy="256" r="160" fill="none" stroke="#c9a25a" stroke-width="6"/>
  <g stroke="#c9a25a" stroke-width="5">
    <line x1="256" y1="30" x2="256" y2="482"/>
    <line x1="30" y1="256" x2="482" y2="256"/>
    <line x1="96" y1="96" x2="416" y2="416"/>
    <line x1="416" y1="96" x2="96" y2="416"/>
  </g>
  <path d="M256 140 C300 200 330 232 330 272 A74 74 0 1 1 182 272 C182 232 212 200 256 140 Z"
        fill="#7fa8bf" stroke="#c9a25a" stroke-width="8"/>
</svg>
"""


def paving(n=1024, stones=32, seed=7):
    """A 40 m square of stone flags, as an sRGB uint8 image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    gx, gy = xx * stones, yy * stones
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - ix, gy - iy
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    grout = edge < 0.045
    tint = rng.random((stones, stones, 1))
    stone = np.array([0.72, 0.66, 0.55], np.float32)
    img = stone * (0.82 + 0.30 * tint[ix % stones, iy % stones])
    img = np.clip(img + rng.normal(0, 0.035, (n, n, 1)), 0, 1)
    img[grout] *= 0.55
    return np.round(img * 255).astype(np.uint8)


def patina(n=256, seed=3):
    """A weathered bronze map, as an sRGB uint8 image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    noise = np.sin(12 * xx + 2 * np.sin(9 * yy)) * np.sin(11 * yy + 3 * np.cos(7 * xx))
    m = (noise + 1) * 0.5
    bronze = np.array([0.72, 0.45, 0.20], np.float32)
    green = np.array([0.32, 0.55, 0.42], np.float32)
    img = bronze * (1 - 0.55 * m[..., None]) + green * (0.55 * m[..., None])
    img = np.clip(img + rng.normal(0, 0.03, (n, n, 1)), 0, 1)
    return np.round(img * 255).astype(np.uint8)


def build_plaza(*, night=False):
    """Build the plaza and return its engine; night moves the sun below the horizon."""
    engine = mx.Engine("plaza", width=480, height=270)
    engine.background_color = (0.004, 0.006, 0.02) if night else (0.01, 0.03, 0.10)
    engine.camera.set_pose((7, GROUND + 3.5, 7), (0, GROUND + 4.2, 0))
    engine.camera.near = 0.3
    engine.camera.far = 500_000.0
    engine.set_sun(
        DirectionalLight(
            color="#aab8ff" if night else "#fff4e8",
            intensity=0.35 if night else 3.2,
            direction=(0.4, 0.8, 0.3) if night else (-0.25, -0.72, -0.45),
        )
    )
    engine.render_frame(supersample=1)  # open the device for the texture uploads

    flags = texture_from_array(engine, paving())
    engine.spawn(
        Mesh(plane(4000, 4000)),
        Material(StandardMaterial(color="#7d7564", roughness=1.0)),
        Transform(pos=(0, GROUND - 0.2, 0), rot=FLOOR),
    )
    engine.spawn(
        Mesh(plane(40, 40)),
        Material(StandardMaterial(color="#ffffff", roughness=0.85, albedo_map=flags)),
        Transform(pos=(0, GROUND, 0), rot=FLOOR),
    )

    inlay = texture_from_svg(engine, MEDALLION, 512)
    engine.spawn(
        Mesh(plane(5, 5)),
        Material(StandardMaterial(color="#ffffff", roughness=0.55, albedo_map=inlay)),
        Transform(pos=(-2.5, GROUND + 0.02, 4.5), rot=FLOOR),
    )

    stone = StandardMaterial(color="#9a958c", roughness=0.8)
    engine.spawn(Mesh(cube(4, 0.2, 4)), Material(stone), Transform(pos=(0, GROUND + 0.1, 0)))
    engine.spawn(Mesh(cube(4, 0.9, 0.2)), Material(stone), Transform(pos=(0, GROUND + 0.55, 1.9)))
    engine.spawn(Mesh(cube(4, 0.9, 0.2)), Material(stone), Transform(pos=(0, GROUND + 0.55, -1.9)))
    engine.spawn(Mesh(cube(0.2, 0.9, 3.6)), Material(stone), Transform(pos=(1.9, GROUND + 0.55, 0)))
    engine.spawn(
        Mesh(cube(0.2, 0.9, 3.6)), Material(stone), Transform(pos=(-1.9, GROUND + 0.55, 0))
    )

    bronze = StandardMaterial(color="#c08a4a", metallic=1.0, roughness=0.2)
    engine.spawn(
        Mesh(cube(4.2, 0.12, 0.3)), Material(bronze), Transform(pos=(0, GROUND + 1.02, 1.9))
    )
    engine.spawn(
        Mesh(cube(4.2, 0.12, 0.3)), Material(bronze), Transform(pos=(0, GROUND + 1.02, -1.9))
    )
    engine.spawn(
        Mesh(cube(0.3, 0.12, 3.6)), Material(bronze), Transform(pos=(1.9, GROUND + 1.02, 0))
    )
    engine.spawn(
        Mesh(cube(0.3, 0.12, 3.6)), Material(bronze), Transform(pos=(-1.9, GROUND + 1.02, 0))
    )

    engine.spawn(
        Mesh(plane(3.6, 3.6)),
        Material(WaterMaterial(color="#0a2352", fresnel_power=4.0)),
        Transform(pos=(0, GROUND + 0.88, 0), rot=FLOOR),
    )

    patina_path = Path(tempfile.gettempdir()) / "plaza_patina.png"
    Image.fromarray(patina()).save(patina_path)
    weathered = load_texture(engine, patina_path)
    engine.spawn(
        Mesh(sphere(0.4, 32)),
        Material(
            StandardMaterial(color="#ffffff", metallic=1.0, roughness=0.18, albedo_map=weathered)
        ),
        Transform(pos=(0, GROUND + 1.15, 0)),
    )

    env = EnvironmentMap.from_sky(
        zenith=(0.08, 0.20, 0.55),
        horizon=(0.65, 0.78, 0.95),
        ground=(0.16, 0.14, 0.11),
        height=16,
        width=32,
    )
    env.intensity = 0.02 if night else 1.0
    engine.set_environment(env)

    engine.spawn(
        Mesh(GeoMesh.icosphere(4, SKY_TOP).to_geometry()),
        Material(AtmosphereScatteringMaterial(GROUND, SKY_TOP, intensity=0.5)),
        Transform(),
    )
    if night:
        starfield(
            engine,
            count=1400,
            radius=300_000.0,
            seed=42,
            ground_radius=GROUND,
            atmo_top=SKY_TOP - GROUND,
        )
        moon = (-120_000, GROUND + 100_000, -120_000)
        engine.spawn(
            Mesh(sphere(5000, 24)), Material(BasicMaterial("#f2ead0")), Transform(pos=moon)
        )
        engine.spawn(
            Mesh(sphere(5600, 24)),
            Material(AtmosphereMaterial("#9fc4ff", intensity=1.0)),
            Transform(pos=moon),
        )
    else:
        engine.spawn(
            Mesh(GeoMesh.icosphere(4, 150).to_geometry()),
            Material(CloudMaterial(coverage=0.52, softness=0.12, freq=14.0, opacity=0.8)),
            Transform(pos=(0, GROUND, 0)),
        )
    return engine


```

The builder is the chapter in one place. The order matters: the first
`render_frame` exists only to open the device, the environment is set after the
textures so its precompute starts on the frame that needs it, and the sky shell is
spawned last so it costs nothing until the frame is drawn.

```python {run continue export=examples/sky_and_water.py}
engine = build_plaza()
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
```

The plaza is a place now: the paving has joints and variation, the water sits in
a bronze-rimmed basin and the bronze carries the sky, the day sky has clouds, and
the night sky has stars and a haloed moon. The next chapter builds the buildings
around it.
