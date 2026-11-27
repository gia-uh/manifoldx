"""Real-time terrain streaming — CPU patches generated in a worker *process*.

World-aligned terrain patches tile the ground ahead of a forward-flying camera.
Each patch is built in a separate process via `engine.submit_process` and swapped
into a fixed pool of recycled mesh slots (`handle.set_geometry`) when ready.

A separate *process* (not a thread) is essential: a patch takes ~1 s of numpy,
which holds the GIL through its Python-level parts and would freeze the render
thread for ~0.9 s. A process shares no GIL, so the render loop never stalls; the
patch's arrays ship back over IPC in a couple of milliseconds. (Because it uses
a spawn-context process pool, all engine/GPU setup lives under
`if __name__ == "__main__":`, and the patch builder is a top-level pure function
the worker can import cleanly.)

    uv run python demos/terrain_stream.py
    uv run python demos/terrain_stream.py --render --duration 10 --output /tmp/stream.mp4
"""

import json
import os
from time import perf_counter

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.modeling import Mesh as GeoMesh, fields, Gradient
from manifoldx.resources import DirectionalLight, StandardMaterial

# ---- pure module-level world (imported cleanly by the worker process) --------
HEIGHT, DEPTH, SPEED, K = 9.0, 52.0, 9.0, 3
CAM_Y, LOOK_Y, LOOK_AHEAD = 12.0, 2.5, 28.0        # above the peaks, tilted down, forward
TEMPLATE = GeoMesh.plane(width=44, depth=DEPTH, segments=180)
terrain = (
    fields.ridged(seed=5, freq=0.11) * 0.74        # big ridges
    + fields.ridged(seed=11, freq=0.28) * 0.14     # secondary ridges
    + fields.fbm(seed=7, freq=0.9) * 0.09          # fine surface texture
).warp(1.3, fx=fields.fbm(2, 0.14), fz=fields.fbm(9, 0.14))
palette = Gradient([(0.00, "#2f5788"), (0.05, "#c6b884"), (0.11, "#4f7a2c"),
                    (0.40, "#59702e"), (0.58, "#7c6440"), (0.76, "#8f857a"),
                    (0.90, "#c2bcb4"), (0.98, "#ffffff")])

# Boundary vertices — pinned across the smooth pass so adjacent patches keep
# meeting exactly (smoothing edges would break the seamless join).
_X, _Z = TEMPLATE.positions[:, 0], TEMPLATE.positions[:, 2]
_EDGE = (np.isclose(_X, _X.min()) | np.isclose(_X, _X.max())
         | np.isclose(_Z, _Z.min()) | np.isclose(_Z, _Z.max()))


def build_patch(world_z):
    """Pure, picklable: the terrain patch at `world_z` as a baked geometry dict."""
    raw = TEMPLATE.displace(terrain.shift((0, 0, world_z)), amount=HEIGHT)
    smoothed = raw.smooth(iterations=1, strength=0.5)   # tame the spikiest peaks
    pos = smoothed.positions.copy()
    pos[_EDGE] = raw.positions[_EDGE]                    # keep patch borders seamless
    return (smoothed.with_positions(pos)
            .color_by(fields.coord("y").remap(0.0, HEIGHT, 0.0, 1.0), palette)
            .to_geometry())


def main():
    engine = mx.Engine("Terrain Stream", width=1024, height=768)
    engine.background_color = (0.72, 0.78, 0.86)
    engine.set_sun(DirectionalLight(color="#fff0dc", intensity=3.1, direction=(-0.5, -0.55, -0.65)))
    engine.enable_fog(start=30, end=110)

    # A ring of K recycled slots, built synchronously at startup.
    slots = [
        engine.spawn(
            Mesh(build_patch(i * DEPTH)),
            Material(StandardMaterial("#ffffff", roughness=0.86, vertex_colors=True)),
            Transform(pos=(0, 0, i * DEPTH)),
        )
        for i in range(K)
    ]
    engine.camera.set_pose((0, CAM_Y, 0), (0, LOOK_Y, LOOK_AHEAD))

    st = {"next_z": K * DEPTH, "pending": None, "rear": 0}
    trace_path = os.environ.get("MANIFOLDX_TERRAIN_TRACE")

    @engine.system
    def stream(query, dt):
        # Fly forward: the look-at target advances with the camera.
        z = SPEED * engine.elapsed
        engine.camera.set_pose((0, CAM_Y, z), (0, LOOK_Y, z + LOOK_AHEAD))
        if st["pending"] is None and z > (st["rear"] + 1) * DEPTH:
            st["submitted"] = perf_counter()
            st["pending"] = engine.submit_process(build_patch, st["next_z"])  # generate off-process
        if st["pending"] is not None and st["pending"].ready:
            index = st["rear"] % K
            slot = slots[index]
            started_swap = perf_counter()
            slot.set_geometry(st["pending"].result)      # main-thread swap-in (in place)
            slot.transform.pos = (0, 0, st["next_z"])
            if trace_path:
                with open(trace_path, "a") as trace:
                    observation = {"slot": index, "world_z": st["next_z"], "elapsed_s": engine.elapsed,
                                   "worker_ms": (started_swap - st["submitted"]) * 1000,
                                   "swap_ms": (perf_counter() - started_swap) * 1000}
                    trace.write(json.dumps(observation) + "\n")
            st["next_z"] += DEPTH
            st["rear"] += 1
            st["pending"] = None

    engine.cli()


if __name__ == "__main__":
    main()
