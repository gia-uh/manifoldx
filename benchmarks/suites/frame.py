"""Per-frame CPU cost of whole scenes: render prep, batching, culling, uploads, encoding."""

import numpy as np

from benchmarks import harness, scenes
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


@case("frame", "meshes_tinted", sizes=SIZES, smoke=10)
def meshes_tinted(n):
    """n static instances of one vertex-coloured mesh, each with its own Tint (StandardMaterial(tinted=True))."""
    return _frame(scenes.meshes(n, tinted=True))


@case("frame", "meshes_shadow", sizes=SIZES, smoke=10)
def meshes_shadow(n):
    """n static mesh instances under a sun with shadows (auto-fit)."""
    return _frame(scenes.meshes(n, shadows=True))


@case("frame", "meshes_moving", sizes=SIZES, smoke=10)
def meshes_moving(n):
    """n mesh instances moved by a system every frame."""
    return _frame(scenes.meshes(n, moving=True))


@case("frame", "parallel_systems", sizes=[1, 4, 8], smoke=1)
def parallel_systems(k):
    """k independent systems of 2,000 entities each, registered as one parallel group when the
    engine supports it (serially on an engine without @engine.system(parallel=True))."""
    return _frame(scenes.independent_systems(k))


@case("frame", "pipelined", sizes=[2, 6, 12], smoke=2)
def pipelined(system_ms):
    """A ~6 ms render against a system of `system_ms` ms, serial then pipelined; drives the window
    path's _draw_frame on an offscreen canvas with the pipeline forced on."""
    import time

    import manifoldx as mx
    from manifoldx.components import Transform

    # The real scene renders in about 6 ms on the GPU desktop; the smoke size
    # keeps tests quick (the harness lowers REPEAT to 1 there).
    n = 22_000 if harness.REPEAT > 1 else 200

    def build(pipeline):
        engine = scenes.meshes(n)

        @engine.system
        def spin(q: mx.Query[Transform], dt: float):
            data = q[Transform].position.data
            deadline = time.perf_counter() + system_ms / 1000.0
            while time.perf_counter() < deadline:
                data = np.sin(data) * 0.5 + 0.5
            q[Transform].position = data

        start(engine)
        if pipeline:
            # run() turns this on for a free-threaded window; a benchmark pins
            # it here because _draw_frame is driven directly.
            engine._pipeline_on = True
            engine._pipeline_frames = 0
        return engine

    serial = build(False)
    piped = build(True)
    try:
        serial_ms = frame_ms(serial, frames=5, warmup=2)
        pipelined_ms = frame_ms(piped, frames=5, warmup=2)
        # One frame can stall on the OS scheduler; report the median overlap of
        # a few more frames, not whichever frame happened to be last.
        overlaps = []
        for _ in range(7):
            piped._draw_frame()
            piped._render_canvas.draw()
            overlaps.append(piped.frame_split["overlap"])
        return {
            "ms": serial_ms,
            "pipelined_ms": pipelined_ms,
            "overlap_ms": 1000.0 * float(np.median(overlaps)),
        }
    finally:
        serial.close()
        piped.close()


@case("frame", "unique_meshes", sizes=[100, 1_000, 5_000], smoke=5)
def unique_meshes(n):
    """n entities with a geometry each: one draw batch per entity."""
    return _frame(scenes.unique_meshes(n))


@case("frame", "still_frame", sizes=[256, 1024, 2048], smoke=32)
def still_frame(side):
    """render_frame(supersample=1) of an empty side × side scene: the readback and its copy."""
    import manifoldx as mx

    engine = mx.Engine("still", width=side, height=side)
    engine.render_frame(supersample=1)
    return measure(lambda: engine.render_frame(supersample=1))


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


def _settled_streamer(levels):
    """A QuadStreamer over a full tree, settled with the camera 20 m up over the middle; returns it and mid."""
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
    for _ in range(400):
        s.focus(mid, mid, 20.0)
        engine._draw_frame()
        if not s._pending and len(s.drawn()) == len(s._selected):
            break
        time.sleep(0.005)
    for _ in range(20):  # let the refinement passes finish
        s.focus(mid, mid, 20.0)
    return s, mid


@case("frame", "quadstream_settled", sizes=[6, 8, 10], smoke=3)
def quadstream_settled(levels):
    """QuadStreamer.focus alone with the camera still over a settled tree of `levels` levels."""
    s, mid = _settled_streamer(levels)
    ms = measure(lambda: s.focus(mid, mid, 20.0))
    out = {"ms": ms, "nodes": len(s.drawn())}
    s.close()
    return out


@case("frame", "quadstream_moving", sizes=[6, 8, 10], smoke=3)
def quadstream_moving(levels):
    """QuadStreamer.focus alone with the camera moving 0.5 m a call, back and forth within 25 m of the
    spot a tree of `levels` levels settled at, so every node it needs is built: selection, not building."""
    s, mid = _settled_streamer(levels)
    k = [0]

    def step():
        k[0] += 1
        s.focus(mid + 25.0 - abs(k[0] % 100 - 50) * 0.5, mid, 20.0)

    ms = measure(step)
    out = {"ms": ms, "nodes": len(s.drawn())}
    s.close()
    return out


@case("frame", "gui_still", sizes=[10, 40, 160], smoke=2)
def gui_still(n):
    """n HUD panels over an empty scene, none of them changing: the GUI pass alone."""
    engine, _texts = scenes.gui_hud(n)
    return _frame(engine)


@case("frame", "gui_one_text", sizes=[10, 40, 160], smoke=2)
def gui_one_text(n):
    """n HUD panels over an empty scene with one panel's number changing every frame, like a speedometer."""
    engine, texts = scenes.gui_hud(n)
    k = [0]

    @engine.system
    def tick(q, dt):
        k[0] += 1
        texts[0].text = f"{k[0] % 200:03d}"

    return _frame(engine)
