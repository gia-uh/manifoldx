"""IBL demo — 4×4 grid of metallic spheres at varying roughness and metallic value,
rendered under three environment presets.

Controls:
  1 / 2 / 3  — cycle through studio / sky / neutral environments
  S          — toggle skybox background on/off
"""
import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import PointLight, StandardMaterial, sphere

ENVS = ["studio", "sky", "neutral"]

engine = mx.Engine("IBL Demo", max_entities=256)

sphere_geo = sphere(0.4, 32)

# 4×4 grid: column = roughness (0→1), row = metallic (0→1)
for row in range(4):
    for col in range(4):
        roughness = col / 3.0
        metallic = row / 3.0
        pos = ((col - 1.5) * 1.2, (row - 1.5) * 1.2, 0.0)
        engine.spawn(
            Mesh(sphere_geo),
            Material(StandardMaterial(color=(0.7, 0.7, 0.8), roughness=roughness, metallic=metallic)),
            Transform(pos=pos),
            n=1,
        )

engine.set_lights([PointLight(position=(3.0, 3.0, 5.0), color="#ffffff", intensity=20)])
engine.set_environment("studio")
engine.camera.position = (0, 0, 8)

env_idx = [0]


@engine.on("key_down")
def on_key(ev):
    key = getattr(ev, "key", None)
    if key in ("1", "2", "3"):
        env_idx[0] = int(key) - 1
        engine.set_environment(ENVS[env_idx[0]])
        print(f"Environment: {ENVS[env_idx[0]]}")
    elif key in ("s", "S"):
        if engine.environment:
            engine.environment.show_skybox = not engine.environment.show_skybox
            print(f"Skybox: {engine.environment.show_skybox}")


if __name__ == "__main__":
    engine.cli()
