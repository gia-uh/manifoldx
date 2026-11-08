"""First-person flashlight demo — walk a procedurally-generated maze in the dark.

Showcases three engine features at once:
  * keyboard movement    — WASD walks (grounded, no flying), collision with walls
  * mouse look           — hold the RIGHT mouse button and drag to look around
  * a single spot light  — a flashlight mounted on the camera, casting shadows
                           down the corridors. The scene is dark except its cone.

    uv run python examples/labyrinth_fps.py            # play it
    uv run python examples/labyrinth_fps.py --render --duration 12 --output /tmp/maze.mp4

Under --render, live input can't be captured, so a scripted flythrough walks the
corridors automatically for the video.
"""

import math
import random
import sys

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import cube, StandardMaterial, SpotLight

# --- Tunables ------------------------------------------------------------
COLS, ROWS = 8, 8  # maze cells per side -> (2*n+1) grid
SEED = 7
CELL = 4.0  # world size of one grid cell (corridor width)
WALL_H = 2.4  # wall height (< corridor width so hallways read)
EYE = 1.5  # camera eye height
FOV = 85  # wide FPS field of view (default camera is 60)
MOVE_SPEED = 6.0  # units / second
LOOK_SENS = 0.2  # degrees per mouse pixel
PLAYER_R = 1.1  # collision radius (corridor is CELL wide)
FLY_SPEED = 2.2  # flythrough: cells / second
SPOT_INTENSITY = 120.0
SPOT_RANGE = 18.0

FLYTHROUGH = "--render" in sys.argv  # scripted walkthrough for headless video


# --- Maze generation (recursive backtracker -> perfect maze) -------------
def generate_maze(cols, rows, seed):
    """Return an (H, W) bool grid, True = wall. Fully connected corridors."""
    W, H = 2 * cols + 1, 2 * rows + 1
    wall = np.ones((H, W), dtype=bool)
    rng = random.Random(seed)
    wall[1, 1] = False
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < H - 1 and 1 <= nc < W - 1 and wall[nr, nc]:
                nbrs.append((nr, nc, dr, dc))
        if nbrs:
            nr, nc, dr, dc = rng.choice(nbrs)
            wall[r + dr // 2, c + dc // 2] = False  # knock out the wall between
            wall[nr, nc] = False
            stack.append((nr, nc))
        else:
            stack.pop()
    return wall


def solution_path(wall, start=(1, 1)):
    """Forward-only path from `start` to the farthest open cell (BFS). No
    reversals — the flythrough walks straight through corridors, turning only at
    junctions, so the camera always looks *down* the hallway it's entering."""
    from collections import deque

    H, W = wall.shape
    parent = {start: None}
    q = deque([start])
    far = start
    while q:
        r, c = far = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not wall[nr, nc] and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    path = []
    node = far
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


maze = generate_maze(COLS, ROWS, SEED)
_H, _W = maze.shape
# Frame code reads the walls as a flat list: the web snapshot carries 1-d arrays and lists,
# not a 2-d bool grid (know-how/web-export.md).
WALLS = maze.reshape(-1).tolist()


def cell_to_world(r, c):
    """Grid cell centre -> world (x, z), maze centred on the origin."""
    return (c - _W // 2) * CELL, (r - _H // 2) * CELL


def _hits_wall(x, z):
    """True if the player's circle at world (x, z) overlaps any wall cell."""
    for sx in (x - PLAYER_R, x + PLAYER_R):
        for sz in (z - PLAYER_R, z + PLAYER_R):
            c = int(math.floor(sx / CELL + 0.5)) + _W // 2
            r = int(math.floor(sz / CELL + 0.5)) + _H // 2
            if not (0 <= r < _H and 0 <= c < _W) or WALLS[r * _W + c]:
                return True
    return False


_walk = solution_path(maze)
_path = np.array([[*cell_to_world(r, c)] for r, c in _walk], dtype=np.float32)  # (N, 2) x,z


# --- Engine + scene ------------------------------------------------------
engine = mx.Engine("Labyrinth FPS", width=1024, height=768)

_cube = cube(1, 1, 1)
_look = {"yaw": 0.0, "pitch": 0.0}


def _forward():
    a, e = math.radians(_look["yaw"]), math.radians(_look["pitch"])
    return np.array(
        [math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)], dtype=np.float32
    )


class _CameraSpot(SpotLight):
    """The flashlight: a cone whose position and aim are read from the camera.

    The renderer samples the light every frame, so the beam follows the view without
    frame code writing the light. A write to a light is outside the web subset: the
    page keeps the lights scene.glb was written with (know-how/web-export.md).
    """

    def __init__(self, engine):
        super().__init__(
            color="#ffe9c0",
            intensity=SPOT_INTENSITY,
            position=(0, EYE, 0),
            direction=(1, 0, 0),
            inner_angle=0.16,
            outer_angle=0.32,
            distance=SPOT_RANGE,
        )
        self._engine = engine

    @property
    def position(self):
        return tuple(self._engine.camera.position)

    @position.setter
    def position(self, value):
        pass  # set_origin moves a fixed light; this one is wherever the camera is

    @property
    def direction(self):
        return tuple(_forward())

    @direction.setter
    def direction(self, value):
        pass  # the camera's aim is the light's only direction


# Flashlight: a warm cone mounted on the camera; it is the shadow caster.
engine.set_spot(_CameraSpot(engine))
engine.enable_shadows(resolution=2048, near=0.15, bias=0.003, pcf_radius=2)


@engine.on("startup")
def build(_payload):
    # Floor: one wide thin cube under the whole maze.
    engine.spawn(
        Mesh(_cube),
        Material(StandardMaterial(color="#3a3a42", roughness=0.95, metallic=0.0)),
        Transform(pos=(0, -0.05, 0), scale=(_W * CELL, 0.1, _H * CELL)),
        n=1,
    )
    # Walls: one cube per wall cell (shared mesh -> a single batch).
    wall_mat = StandardMaterial(color="#5a5f6a", roughness=0.85, metallic=0.0)
    for r in range(_H):
        for c in range(_W):
            if maze[r, c]:
                x, z = cell_to_world(r, c)
                engine.spawn(
                    Mesh(_cube),
                    Material(wall_mat),
                    Transform(pos=(x, WALL_H * 0.5, z), scale=(CELL, WALL_H, CELL)),
                    n=1,
                )

    engine.camera.fov = FOV
    # Start at the first corridor cell, facing along the first step of the walk.
    sx, sz = cell_to_world(*_walk[0])
    engine.camera.position = np.array([sx, EYE, sz], dtype=np.float32)
    d = _path[1] - _path[0]
    _look["yaw"] = math.degrees(math.atan2(d[1], d[0]))
    engine.camera.target = engine.camera.position + _forward()


def fps_controls(query: mx.Query[Transform], dt: float):
    """Interactive: WASD move + right-drag look, with wall collision."""
    if engine.input.is_mouse_pressed(2):
        dx, dy = engine.input.mouse_delta
        _look["yaw"] += dx * LOOK_SENS
        _look["pitch"] = max(-89.0, min(89.0, _look["pitch"] - dy * LOOK_SENS))

    fwd = _forward()
    hf = np.array([fwd[0], 0.0, fwd[2]], dtype=np.float32)
    n = np.linalg.norm(hf)
    if n > 0:
        hf /= n
    right = np.array([hf[2], 0.0, -hf[0]], dtype=np.float32)  # world up x hf
    nr = np.linalg.norm(right)
    if nr > 0:
        right /= nr

    move = np.zeros(3, np.float32)
    if engine.input.is_pressed("w"):
        move += hf
    if engine.input.is_pressed("s"):
        move -= hf
    if engine.input.is_pressed("d"):
        move += right
    if engine.input.is_pressed("a"):
        move -= right
    nm = np.linalg.norm(move)
    if nm > 0:
        move = move / nm * MOVE_SPEED * dt

    cam = engine.camera.position
    # Axis-separated collision so we slide along walls instead of sticking.
    px = cam[0] + move[0]
    if _hits_wall(px, cam[2]):
        px = cam[0]
    pz = cam[2] + move[2]
    if _hits_wall(px, pz):
        pz = cam[2]
    pos = np.array([px, EYE, pz], dtype=np.float32)
    engine.camera.position = pos
    engine.camera.target = pos + fwd


def flythrough(query: mx.Query[Transform], dt: float):
    """Headless video: walk the precomputed corridor path automatically."""
    t = engine.elapsed * FLY_SPEED
    seg = int(t)
    if seg >= len(_path) - 1:
        seg, frac = len(_path) - 2, 1.0
    else:
        frac = t - seg
    a, b = _path[seg], _path[seg + 1]
    xz = a + (b - a) * frac
    pos = np.array([xz[0], EYE, xz[1]], dtype=np.float32)
    # Snap yaw to the axis-aligned travel direction (the next cell is always
    # open) so the camera is always looking *down* a corridor, never diagonally
    # into a corner. A slight downward pitch shows the flashlight pooling on the
    # floor ahead rather than a flat wall face.
    d = b - a
    if np.linalg.norm(d) > 1e-5:
        _look["yaw"] = math.degrees(math.atan2(d[1], d[0]))
    _look["pitch"] = -15.0
    engine.camera.position = pos
    engine.camera.target = pos + _forward()


# Register exactly one controller for the active mode.
engine.system(flythrough if FLYTHROUGH else fps_controls)


if __name__ == "__main__":
    engine.cli()
