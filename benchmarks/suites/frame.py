"""Per-frame CPU cost of whole scenes: render prep, batching, culling, uploads, encoding."""

from benchmarks import scenes
from benchmarks.harness import case, count_calls, frame_ms, measure, start

SIZES = [1_000, 10_000, 100_000]


def _frame(engine) -> dict:
    from manifoldx.renderer import TransformCache

    start(engine)
    frame_ms(engine, frames=1, warmup=2)  # pipelines, buffers
    with count_calls(TransformCache, "_recompute_matrices") as c:
        engine._draw_frame()
        engine._render_canvas.draw()
    return {"ms": frame_ms(engine, warmup=0), "matrix_recomputes": c["calls"]}


@case("frame", "sprites", sizes=SIZES, smoke=10)
def sprites(n):
    """n static point sprites."""
    return _frame(scenes.sprites(n))


@case("frame", "sprites_mixed", sizes=[10_000, 100_000, 300_000], smoke=200)
def sprites_mixed(n):
    """n point sprites, 1% of them moved by a system every frame."""
    return _frame(scenes.sprites_mixed(n))


@case("frame", "meshes", sizes=SIZES, smoke=10)
def meshes(n):
    """n static instances of one mesh."""
    return _frame(scenes.meshes(n))


@case("frame", "meshes_vcolor", sizes=SIZES, smoke=10)
def meshes_vcolor(n):
    """n static instances of one vertex-coloured mesh (StandardMaterial(vertex_colors=True))."""
    return _frame(scenes.meshes(n, vertex_colors=True))


@case("frame", "meshes_shadow", sizes=SIZES, smoke=10)
def meshes_shadow(n):
    """n static mesh instances under a sun with shadows (auto-fit)."""
    return _frame(scenes.meshes(n, shadows=True))


@case("frame", "meshes_moving", sizes=SIZES, smoke=10)
def meshes_moving(n):
    """n mesh instances moved by a system every frame."""
    return _frame(scenes.meshes(n, moving=True))


@case("frame", "unique_meshes", sizes=[100, 1_000, 5_000], smoke=5)
def unique_meshes(n):
    """n entities with a geometry each: one draw batch per entity."""
    return _frame(scenes.unique_meshes(n))


@case("frame", "ids_pass", sizes=SIZES, smoke=10)
def ids_pass(n):
    """render_frame(pass_='ids') over n mesh instances, one group."""
    engine = scenes.meshes(n)
    start(engine)
    group = [range(0, n, 2)]
    engine.render_frame(supersample=1, pass_="ids", groups=group)
    return measure(lambda: engine.render_frame(supersample=1, pass_="ids", groups=group), number=1)


@case("frame", "quadstream", sizes=[6, 8, 10], smoke=3)
def quadstream(levels):
    """A QuadStreamer over a full tree of `levels` levels of 100 m cells, camera 20 m up: focus() plus the frame."""
    import time

    import numpy as np

    import manifoldx as mx
    from manifoldx.streaming import MeshData, QuadStreamer

    def build(L, a, b):
        p = np.array([[0, 0, 0], [8, 0, 0], [0, 0, 8]], np.float32)
        return MeshData(p, np.tile([0, 1, 0], (3, 1)).astype(np.float32), np.full((3, 3), 0.5, np.float32),
                        np.array([0, 2, 1], np.uint32))

    engine = mx.Engine("quadstream", width=64, height=64)
    engine.camera.far = 15000.0
    s = QuadStreamer(engine, 100.0, (0, 0), levels, lambda L, a, b: True, build, error=lambda L: 100.0 * 2**L / 16)
    start(engine)
    mid = 100.0 * 2 ** (levels - 1)
    engine.set_origin(mid, mid)
    for _ in range(400):  # until everything picked is built and drawn
        s.focus(mid, mid, 20.0)
        engine._draw_frame()
        if not s._pending and len(s.drawn()) == len(s._selected):
            break
        time.sleep(0.005)

    def frame():
        s.focus(mid, mid, 20.0)
        engine._draw_frame()

    ms = measure(frame)
    out = {"ms": ms, "nodes": len(s.drawn())}
    s.close()
    return out
