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


engine = build_plaza()
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
