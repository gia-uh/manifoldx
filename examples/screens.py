"""Two screens in one window: one Engine, one run(), screens swap inside it.

Each screen is a coroutine: it spawns its entities, panels and systems, waits
frames until Space is pressed, then tears down with clear_scene() and
remove_system(). quit() would close the window, so it is not used to switch.
"""

import math

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.gui import Panel, Text
from manifoldx.systems import Query

engine = mx.Engine("Screens")

# Built once: the registries keep every geometry and material for the engine's
# lifetime, so building them inside a screen would add new ones on each visit.
CUBE = mx.geometry.cube(1, 1, 1)
SPHERE = mx.geometry.sphere(0.8)
RED = mx.material.phong(mx.colors.RED)
BLUE = mx.material.phong(mx.colors.BLUE)
CAPTION = {"width": 310, "height": 30, "bg": "#171b24cc"}



async def wait_for_space():
    while True:
        await engine.tick()
        if engine.input.just_pressed("Space"):
            return


async def screen_a():
    engine.spawn(Mesh(CUBE), Material(RED), Transform(pos=(0, 0, 0)))
    engine.gui.append(Panel([Text("Screen A: press Space")], offset=(10, 10), style_overrides=CAPTION))

    @engine.system
    def spin(query: Query[Transform], dt: float):
        query[Transform].rot += Transform.rotation(x=0, y=dt * math.pi, z=0)

    await wait_for_space()
    engine.remove_system(spin)
    engine.clear_scene()


async def screen_b():
    engine.spawn(Mesh(SPHERE), Material(BLUE), Transform(pos=(0, 0, 0)))
    engine.gui.append(Panel([Text("Screen B: press Space to go back")], offset=(10, 10), style_overrides=CAPTION))
    await wait_for_space()
    engine.clear_scene()


@engine.on("startup")
async def main(_payload):
    while True:
        await screen_a()
        await screen_b()


if __name__ == "__main__":
    engine.cli()
