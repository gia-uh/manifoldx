"""A lit yard for the browser's visual parity: StandardMaterial boxes of several colours and
roughnesses, a sun with shadows, the sky environment and fog, seen from a fixed camera."""

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight

engine = mx.Engine("Lit yard", width=320, height=200)
engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.6, -0.7, -0.4)))
engine.enable_shadows()
engine.set_environment("sky")
engine.enable_fog(20, 60)

BOXES = [  # centre, side, colour, roughness
    ((-3.0, 0.5, -2.0), 1.0, "#c0392b", 0.3),
    ((0.0, 0.75, -4.0), 1.5, "#2e86c1", 0.6),
    ((3.0, 1.0, -1.0), 2.0, "#27ae60", 0.9),
    ((-1.5, 0.4, 1.5), 0.8, "#f2c14e", 0.2),
    ((2.0, 0.3, 2.5), 0.6, "#8e44ad", 0.5),
]


@engine.on("startup")
def build(_):
    cube = mx.geometry.cube(1, 1, 1)
    ground = mx.material.standard("#9a9a8a", roughness=0.95)
    engine.spawn(Mesh(cube), Material(ground), Transform(pos=(0, -0.05, 0), scale=(40, 0.1, 40)))
    for centre, side, colour, roughness in BOXES:
        material = mx.material.standard(colour, roughness=roughness)
        engine.spawn(Mesh(cube), Material(material), Transform(pos=centre, scale=(side,) * 3))


for i, v in enumerate((0.0, 5.0, 9.0)):
    engine.camera.position[i] = v
for i, v in enumerate((0.0, 0.5, -1.0)):
    engine.camera.target[i] = v

if __name__ == "__main__":
    engine.cli()
