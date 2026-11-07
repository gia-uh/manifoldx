"""Spawning and destroying from frame code: Space spawns a ring of 20 boxes that live two
seconds, b one ball from a geometry no entity drew at setup, x destroys the ball through its
handle. age's query destroys the boxes whose life ran out, and a spawn of zero entities gives
None."""

import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.engine import EntityHandle
from manifoldx.resources import PhongMaterial, cube, sphere
from manifoldx.types import Float


class Life(Component):
    t: Float = 1.0
    born: Float = 7.0


class Tag(Component):
    k: Float = 3.0


class Game:
    # At class level: __init__ runs at setup, so the transpiler does not read its annotations.
    ball: EntityHandle | None = None
    zero: bool = False


engine = mx.Engine("Spawner", width=320, height=200)
engine.set_origin(1000.5, -2000.25)  # spawned positions are world coordinates
CUBE = cube(1.0, 1.0, 1.0)
BALL = sphere(0.5)
BLUE = PhongMaterial(color="#3366ff")
RED = PhongMaterial(color="#ff3333")
engine.spawn(Mesh(CUBE), Material(BLUE), Transform(pos=(0.0, -1.0, 0.0), scale=(10.0, 0.1, 10.0)))
Life.register(engine.store)  # age's query reads Life before the first wave
game = Game()


def spawn(query, dt: float):
    if not game.zero:
        engine.state["none"] = engine.spawn(Mesh(CUBE), Material(BLUE), Life(t=1.0), n=0)
        game.zero = True
    if engine.input.just_pressed("Space"):
        a = np.arange(20) * (2.0 * np.pi / 20)
        ring = np.column_stack([np.cos(a) * 3.0, np.cos(a) * 0.0 + 1.0, np.sin(a) * 3.0])
        engine.spawn(
            Mesh(CUBE),
            Material(BLUE),
            Transform(pos=ring, scale=(0.4, 0.4, 0.4)),
            Life(t=2.0),
            n=20,
        )
    if engine.input.just_pressed("b"):
        game.ball = engine.spawn(Mesh(BALL), Material(RED), Transform(pos=(0.0, 2.0, 0.0)), Tag())
    if engine.input.just_pressed("x") and game.ball is not None:
        game.ball.destroy()
        game.ball = None


def age(q: mx.Query[Life], dt: float):
    q[Life].t -= dt
    engine.destroy(q[Life].t <= 0.0)


def count(q: mx.Query[Life], dt: float):
    engine.state["lives"] = len(q)  # still there in the frame age destroys them: it queues


engine.system(spawn)
engine.system(age)
engine.system(count)
engine.run()
