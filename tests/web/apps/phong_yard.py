"""Rotated PhongMaterial boxes for the browser's visual parity.

One cube geometry and one Phong material, so the page draws every box as one InstancedMesh
batch, and each box is turned differently. The desktop rotates each box's normals with its
model matrix; the page must apply the instance matrix too, or every box is lit as if
unrotated. See #172.
"""

import math

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform

engine = mx.Engine("Phong yard", width=320, height=200)

BOXES = [  # centre, side, rotation (x, y) in degrees
    ((-2.8, 0.0, 0.0), 2.7, (0.0, -25.0)),
    ((0.0, 0.0, 0.0), 2.7, (0.0, -25.0)),
    ((2.8, 0.0, 0.0), 2.7, (0.0, -25.0)),
]


@engine.on("startup")
def build(_):
    cube = mx.geometry.cube(1, 1, 1)
    paint = mx.material.phong("#d9d2c0")
    for centre, side, (rx, ry) in BOXES:
        engine.spawn(
            Mesh(cube),
            Material(paint),
            Transform(
                pos=centre,
                rot=Transform.rotation(x=math.radians(rx), y=math.radians(ry)),
                scale=(side,) * 3,
            ),
        )


engine.camera.position[2] = 5.2

if __name__ == "__main__":
    engine.cli()
