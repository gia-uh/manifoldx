"""Cell streaming: keep meshes built and drawn around a moving focus point.

A Streamer knows nothing of what it streams. It cuts the ground plane into
square cells of side `cell`; cell (i, j) covers x in [i*cell, (i+1)*cell) and
z in [j*cell, (j+1)*cell). `build(i, j)` returns the cell's MeshData (or None for
a cell with nothing to draw) and runs in a worker, never on the frame.

Two rings sit around the focus cell. The cells within `ready_radius` are built
in the background and uploaded to the GPU hidden; the cells within `radius` are
drawn. The worker also packs each cell's vertices the way its vertex buffer
holds them, and the frame copies at most `upload_bytes_per_frame` of them to
the GPU, nearest cell first, so a big cell takes several frames and none of
them stalls. Stepping into a new cell then draws cells that are already on the
GPU, at most `uploads_per_frame` a frame, and costs the frame next to nothing.
Several streamers that cross the same border together can take turns:
`turn=(k, n)` lets this one draw only on every n-th focus() call, the k-th, so
n streamers with k = 0..n-1 never draw new cells in the same frame.
A cell that leaves `radius` stops drawing and keeps its buffers; one that
leaves `ready_radius` is dropped, its buffers freed and its data forgotten.

    streamer = Streamer(engine, cell=32.0, build=build_cell)

    @engine.system
    def stream(query, dt):
        streamer.focus(player_x, player_z)   # float64 world position, once a frame

`workers="thread"` builds in one background thread; `"process"` builds in one
worker process, which shares no GIL with the frame but needs a picklable,
importable `build` (a top-level function, with the engine set up under
`if __name__ == "__main__":`) and ships each MeshData back over IPC.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import time
import itertools
import threading
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter
from typing import Callable, NamedTuple

import numpy as np

from manifoldx import _deferred
from manifoldx.renderer import vertex_bounds
from manifoldx.resources import pack_vertices
from manifoldx import counters


class MeshData(NamedTuple):
    """A cell's mesh: positions relative to the cell's corner (i*cell, 0, j*cell)."""

    positions: np.ndarray  # f32[N, 3]
    normals: np.ndarray  # f32[N, 3]
    colors: np.ndarray  # f32[N, 3], linear
    indices: np.ndarray  # u32[M]
    surfaces: np.ndarray | None = None  # i[N]: surface pattern ids (resources.SURFACE_PATTERNS)
    facade: np.ndarray | None = None  # f32[N, 4]: bays along, floors up, building type, seed (facade_albedo)


class _Cell:
    """A built cell: its geometry dict, and its packed bytes until they are on the GPU."""

    __slots__ = ("geo", "info", "vbytes", "ibytes")

    def __init__(self, geo, info, vbytes, ibytes):
        self.geo, self.info, self.vbytes, self.ibytes = geo, info, vbytes, ibytes


def _started() -> None:
    """A no-op job: submitted when a streamer is made, so its workers start (and warm up) before any cell."""


def _worker_start(parent: int, warm) -> None:
    """A process worker's start: exit as soon as the app that made it is gone, then warm up.

    An app killed by a signal (an out-of-memory killer, `timeout`) never shuts its pool down, and its
    workers, each holding what `warm` loaded, would live on as orphans; a thread here checks the parent
    once a second and ends the worker when it changed."""

    def watch():
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="parent-watch").start()
    if warm is not None:
        warm()


def _pack(build, *key):
    """The worker's job: build cell (i, j), or node (L, a, b), and pack it for its GPU buffers."""
    with counters.worker_span("stream", key):
        return _pack_now(build, key)


def _pack_now(build, key):
    data = build(*key)
    if data is None:
        return None
    geo = {
        "positions": np.asarray(data.positions, np.float32),
        "normals": np.asarray(data.normals, np.float32),
        "colors": np.asarray(data.colors, np.float32),
        "indices": np.asarray(data.indices, np.uint32),
    }
    if data.surfaces is not None:
        geo["surfaces"] = np.asarray(data.surfaces, np.int32)
    if data.facade is not None:
        geo["facade"] = np.asarray(data.facade, np.float32)
    vbytes, info = pack_vertices(geo)
    if len(geo["positions"]):
        geo["bounds"] = vertex_bounds(geo["positions"])  # the frustum test reads it, not the frame
    return _Cell(geo, info, vbytes, geo["indices"].tobytes())


_VECTOR_FROM = 256  # QuadStreamer computes distances with numpy from this many nodes up


_LIVE: weakref.WeakSet = weakref.WeakSet()  # every streamer made and not closed
_SERIAL = itertools.count()


def live() -> list:
    """The streamers made and not closed yet, in the order they were made."""
    return sorted(_LIVE, key=lambda s: s._serial)


def _streamer_engine(args, kwargs):
    """The engine a streamer method belongs to; the object is not the engine itself."""
    return args[0].engine


class _Uploads:
    """What both streamers share: the worker, the GPU uploads in slices, drawing and freeing by key.

    A key is a cell (i, j) for Streamer and a node (L, a, b) for QuadStreamer; `_draw(key, pos)` spawns
    the key's mesh at world position pos.
    """

    def _start(self, engine, build, uploads_per_frame, upload_bytes_per_frame, workers, material, builders=1,
               warm=None):
        if workers == "thread":
            self._pool = ThreadPoolExecutor(max_workers=builders, initializer=warm)
        elif workers == "process":
            self._pool = ProcessPoolExecutor(
                max_workers=builders, mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_start, initargs=(os.getpid(), warm),
            )
        else:
            raise ValueError(f'workers must be "thread" or "process", got {workers!r}')
        for _ in range(builders):  # start the workers now: they warm up while the app loads, not at the first cell
            self._pool.submit(_started)
        if material is None:
            from manifoldx.resources import StandardMaterial

            material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True)
        self.engine, self.build = engine, build
        self.uploads_per_frame = uploads_per_frame
        self.material = material
        self.upload_bytes_per_frame = max(4, int(upload_bytes_per_frame) // 4 * 4)
        self._pending = {}  # key -> Future
        self._lock = threading.RLock()  # _pending, also changed by QuadStreamer's done callbacks
        self._ready = {}  # key -> _Cell | None, built and kept
        self._gpu = {}  # key -> its upload: geometry id, buffers, bytes written, done
        self._drawn = {}  # key -> EntityHandle | None (None: an empty cell)
        self._stats = {"built": 0, "uploaded": 0, "dropped": 0, "max_frame_ms": 0.0}
        self._last = None
        self._landed = 0  # uploads finished so far
        self._serial = next(_SERIAL)
        _LIVE.add(self)

    def _note(self, ev, key):
        """Tell a recording bench run what happened to a cell or node."""
        if counters.recording():
            counters.event("node", ev=ev, key=list(key), level=key[0] if len(key) == 3 else 0,
                           streamer=self._serial)

    def _is_deferred(self) -> bool:
        """Whether this thread's calls to this streamer's engine are deferred."""
        return self.engine._deferring()

    def _tick(self) -> int:
        """Note the time since the last focus() and collect the finished builds; returns how many."""
        now = perf_counter()
        if self._last is not None:
            self._stats["max_frame_ms"] = max(self._stats["max_frame_ms"], (now - self._last) * 1000.0)
        self._last = now
        got = 0
        with self._lock:
            done = [(c, fut) for c, fut in self._pending.items() if fut.done()]
            for c, _fut in done:
                del self._pending[c]
        for c, fut in done:
            self._ready[c] = fut.result()
            self._stats["built"] += 1
            self._note("built", c)
            got += 1
        return got

    def _on_gpu(self, c) -> bool:
        """Whether cell c can be drawn without building a buffer on the frame."""
        if self._ready.get(c) is None or self.engine._device is None:
            return True  # nothing to upload, or no device yet: the render pass builds the buffers
        return self._gpu.get(c, {}).get("done", False)

    @_deferred.deferred_in_pipeline(find=_streamer_engine)
    def _upload(self, c, room: int) -> int:
        """Copy up to `room` bytes of cell c to its GPU buffers; returns the bytes copied."""
        import wgpu

        cell, reg, dev = self._ready[c], self.engine._geometry_registry, self.engine._device
        if dev is None:
            return 0
        up = self._gpu.get(c)
        if up is None:
            gid = reg.register(cell.geo)
            up = self._gpu[c] = {
                "id": gid,
                "done": False,
                "at": [0, 0],
                "bufs": [
                    dev.create_buffer(
                        size=len(cell.vbytes),
                        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
                    ),
                    dev.create_buffer(
                        size=len(cell.ibytes),
                        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
                    ),
                ],
            }
        spent = 0
        for k, data in enumerate((cell.vbytes, cell.ibytes)):
            at = up["at"][k]
            n = min(len(data) - at, room - spent)
            if n > 0:
                counters.write_buffer(dev.queue, up["bufs"][k], at, memoryview(data)[at : at + n])
                up["at"][k] += n
                spent += n
        if up["at"] == [len(cell.vbytes), len(cell.ibytes)]:
            vb, ib = up.pop("bufs")
            reg.adopt_buffers(
                up["id"],
                {
                    **cell.info,
                    "vertex_buffer": vb,
                    "index_buffer": ib,
                    "index_count": len(cell.geo["indices"]),
                },
            )
            up["done"] = True
            self._landed += 1
            self._note("landed", c)
            cell.vbytes = cell.ibytes = None
        return spent

    @_deferred.deferred_in_pipeline(find=_streamer_engine)
    def _unload(self, c):
        """Free cell c's GPU buffers, whole or half written."""
        up = self._gpu.pop(c, None)
        if up is None:
            return
        for buf in up.get("bufs", ()):
            buf.destroy()
        self.engine._geometry_registry.release(up["id"])

    @_deferred.deferred_in_pipeline(find=_streamer_engine)
    def _draw(self, c, pos):
        cell = self._ready[c]
        if cell is None:
            self._drawn[c] = None
            return
        from manifoldx.components import Fade, Material, Mesh, Transform

        self._drawn[c] = self.engine.spawn(
            Mesh(cell.geo),
            Material(self.material),
            Transform(pos=pos),
            *([Fade(value=0.0)] if getattr(self, "crossfade", None) else []),
        )
        # Drawn before any device: its buffers come from the render pass, and are still ours to free.
        if c not in self._gpu:
            self._gpu[c] = {"id": self._drawn[c]._geometry_id, "done": True}
            self._note("landed", c)
        self._stats["uploaded"] += 1
        self._note("drawn", c)

    def _undraw(self, c):
        handle = self._drawn.pop(c)
        self._note("undrawn", c)
        if handle is not None:
            handle.destroy(free_geometry=False)

    def close(self) -> None:
        """Stop the worker and destroy every drawn cell."""
        _LIVE.discard(self)
        self._pool.shutdown(wait=False, cancel_futures=True)
        for c in list(self._drawn):
            self._undraw(c)
        for c in list(self._gpu):
            self._unload(c)
        self._pending.clear()
        self._ready.clear()



class Streamer(_Uploads):
    """Builds, uploads and frees cell meshes around a focus point.

    Its workers start when it is made, and each runs `warm()` first when given: what a build needs loaded once
    per worker (the app's data) loads while the app is still starting, not when the first cell is asked for."""

    def __init__(
        self,
        engine,
        cell: float,
        build: Callable[[int, int], MeshData | None],
        radius: int = 1,
        ready_radius: int = 2,
        uploads_per_frame: int = 2,
        upload_bytes_per_frame: int = 2 << 20,
        workers: str = "thread",
        material=None,
        turn: tuple[int, int] = (0, 1),
        warm: Callable[[], None] | None = None,
    ):
        if ready_radius < radius:
            raise ValueError(f"ready_radius ({ready_radius}) must be at least radius ({radius})")
        self._start(engine, build, uploads_per_frame, upload_bytes_per_frame, workers, material, warm=warm)
        self.cell = float(cell)
        self.radius, self.ready_radius = radius, ready_radius
        self.turn = turn
        self._calls = 0

    def focus(self, x: float, z: float) -> None:
        """Move the focus to world position (x, z). Call it once a frame."""
        self._tick()
        fi, fj = math.floor(x / self.cell), math.floor(z / self.cell)

        def ring(cell):
            return max(abs(cell[0] - fi), abs(cell[1] - fj))

        for c in [c for c in self._drawn if ring(c) > self.radius]:
            self._undraw(c)
        for c in [c for c in self._ready if ring(c) > self.ready_radius]:
            del self._ready[c]
            self._unload(c)
            self._stats["dropped"] += 1
            self._note("dropped", c)
        for c in [c for c in self._pending if ring(c) > self.ready_radius]:
            if self._pending.pop(c).cancel():
                self._note("cancelled", c)

        r = self.ready_radius
        wanted = sorted(
            ((fi + di, fj + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)),
            key=lambda c: (c[0] - fi) ** 2 + (c[1] - fj) ** 2,
        )
        self._want = {c for c in wanted if ring(c) <= self.radius}
        for c in wanted:
            if c not in self._ready and c not in self._pending:
                self._pending[c] = self._pool.submit(_pack, self.build, *c)
                self._note("submitted", c)
        room = self.upload_bytes_per_frame
        for c in wanted:
            if room <= 0:
                break
            if self._ready.get(c) is not None and not self._on_gpu(c):
                if self._is_deferred():
                    self._upload(c, room)  # queued whole at apply: it spends the frame's budget
                    room = 0
                else:
                    room -= self._upload(c, room)
        budget = self.uploads_per_frame if self._calls % self.turn[1] == self.turn[0] else 0
        self._calls += 1
        for c in wanted:
            if budget == 0:
                break
            if (
                ring(c) <= self.radius
                and c not in self._drawn
                and c in self._ready
                and self._on_gpu(c)
            ):
                self._draw(c, (c[0] * self.cell, 0.0, c[1] * self.cell))
                budget -= 1

    def drawn(self) -> set[tuple[int, int]]:
        """The cells drawn now, empty cells (build returned None) included."""
        return set(self._drawn)

    @property
    def settled(self) -> bool:
        """Nothing building or uploading, and every cell the camera needs is drawn."""
        want = getattr(self, "_want", None)
        return want is not None and not self._pending and want <= self._drawn.keys()

    def stats(self) -> dict:
        """built, uploaded and dropped cell counts, and the longest gap between
        two focus() calls in ms (one frame, when focus runs once a frame)."""
        return dict(self._stats)


class QuadStreamer(_Uploads):
    """Streams a quadtree of cells: near nodes fine, far nodes coarse, chosen by their error on screen.

    Node (L, a, b) covers cells origin[0] + a*2^L to origin[0] + (a+1)*2^L - 1 in i, and the same in j;
    the root is (levels, 0, 0) and level 0 is one cell. `exists(L, a, b)` says whether a node holds
    anything; a node's children are the existing (L-1, 2a+da, 2b+db). `build(L, a, b)` returns its
    MeshData relative to the node's corner, in a worker, and `error(L)` its geometric error in metres.

    The nodes to draw are a cut of the tree that each focus() refines: a node splits into its children
    while its error seen from the camera, error * focal / distance, is above `tau` pixels; children merge
    back only once their parent's falls under tau / merge, so a camera that hovers does not make nodes
    blink. Nodes wholly beyond the camera's far plane are left out. The refinement spends at most
    `refine_ms` a frame, nearest nodes first, and carries the rest to the next frames, so a big world or
    a teleport costs a bounded slice of each frame; `refine_ms=None` refines to the end every frame. Once a
    pass settles, the next starts only after the camera moves `move` metres (cell / 8 by default) or its fov,
    far plane or viewport changes, or tau does, and the nodes keep their order till then. A game may
    change `tau` while it runs, to trade detail for frame time. The nodes picked are built by `builders`
    workers, with `in_flight` (twice builders) jobs at most submitted: each time one finishes, the node with
    the most error on screen still to build goes next. They are uploaded within `upload_bytes_per_frame`,
    in the same order. The workers start when the streamer is made, each running `warm()` first when given,
    so what a build loads once per worker loads while the app starts.

    The view sharpens a level at a time: a drawn node gives way to its children as soon as every one of
    them is built and uploaded (or covered by its own children), and those children are built before the
    finer nodes under them, so each level shows on the way down. A selected node not built yet leaves the
    nodes drawn under it drawn, and each spot is drawn once. Built nodes stay in a cache: past `cache` nodes
    out of use, the least recently used are freed, so zooming back out finds the coarse nodes still built.
    A frame where the selection, the builds and the uploads did not change only carries on the uploads
    in flight.

    With `crossfade` seconds (about 0.3), a node that gives way stays drawn that long, fading out while
    the nodes that replace it fade in, over the complementary pixels of a 4x4 pattern (see Fade), so a
    level of detail never pops. It needs a StandardMaterial made with faded=True; without a material the
    streamer makes one.

        streamer = QuadStreamer(engine, cell=200.0, origin=(-2048, -1024), levels=13,
                                exists=tree.exists, build=build_node, error=lambda L: 200.0 * 2**L / 16)
        streamer.focus(cam_x, cam_z, height=cam_y)   # world float64, once a frame
    """

    def __init__(
        self,
        engine,
        cell: float,
        origin: tuple[int, int],
        levels: int,
        exists: Callable[[int, int, int], bool],
        build: Callable[[int, int, int], MeshData | None],
        error: Callable[[int], float],
        tau: float = 6.0,
        merge: float = 1.5,
        upload_bytes_per_frame: int = 2 << 20,
        workers: str = "thread",
        material=None,
        builders: int = 1,
        refine_ms: float | None = 2.0,
        move: float | None = None,
        warm: Callable[[], None] | None = None,
        cache: int = 256,
        crossfade: float | None = None,
    ):
        if crossfade and material is None:
            from manifoldx.resources import StandardMaterial

            material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True, faded=True)
        if crossfade and not getattr(material, "faded", False):
            raise ValueError("crossfade needs a StandardMaterial made with faded=True")
        self._start(engine, build, None, upload_bytes_per_frame, workers, material, builders, warm)
        self.in_flight, self.cache = 2 * builders, int(cache)
        self._todo: list = []  # the nodes to build next, most error on screen last
        self.crossfade = float(crossfade) if crossfade else None
        self._fading: dict = {}  # node -> (handle, started, out): drawn while it fades in, or out once undrawn
        self.cell, self.origin, self.levels = float(cell), (int(origin[0]), int(origin[1])), int(levels)
        self.exists, self.error, self.tau, self.merge = exists, error, float(tau), float(merge)
        self.refine_ms = refine_ms
        self.move = self.cell / 8 if move is None else float(move)
        self._pass_at = None  # the camera the last pass started at
        self._sel, self._sel_at = None, None  # the nodes to draw, nearest first, and the camera they were for
        self._shown_for, self._shown_landed = None, -1  # the selection and uploads the drawn set was last made for
        self._to_upload: list = []  # selected nodes built and not on the GPU yet, nearest first
        self._waiting = False  # some node wanted is not built and uploaded yet
        self._selected: set = set()
        self._kids: dict = {}  # node -> its existing children
        self._cut: set | None = None  # the nodes that tile the tree now; None before the first focus
        self._queue: list = []  # this pass's nodes still to check, nearest last
        self._pass_quiet, self._pass_started = True, False
        self._quiet, self._still = False, None  # focus() returns early while quiet at the same camera
        self._order: list | None = None  # the cut as a list, rebuilt when it changes
        self._cut_changed = False

    def _box(self, node):
        L, a, b = node
        side = (1 << L) * self.cell
        x0 = (self.origin[0] + a * (1 << L)) * self.cell
        z0 = (self.origin[1] + b * (1 << L)) * self.cell
        return x0, z0, side

    def _distance(self, node, x, z, height):
        x0, z0, side = self._box(node)
        dx = max(x0 - x, 0.0, x - (x0 + side))
        dz = max(z0 - z, 0.0, z - (z0 + side))
        return max(math.sqrt(dx * dx + dz * dz + height * height), 1e-3)

    def _boxes(self, nodes):
        """(x0, z0, side) arrays of these nodes' squares."""
        L, a, b = np.asarray(nodes, np.int64).reshape(-1, 3).T
        side = (1 << L) * self.cell
        return (self.origin[0] + a * (1 << L)) * self.cell, (self.origin[1] + b * (1 << L)) * self.cell, side

    def _distances(self, nodes, x, z, height, boxes=None) -> np.ndarray:
        """_distance for many nodes at once, as one array; `boxes` is their _boxes, when kept."""
        if len(nodes) < _VECTOR_FROM and boxes is None:  # building the arrays costs more than it saves
            return np.array([self._distance(n, x, z, height) for n in nodes], dtype=float)
        x0, z0, side = boxes if boxes is not None else self._boxes(nodes)
        dx = np.maximum(np.maximum(x0 - x, 0.0), x - (x0 + side))
        dz = np.maximum(np.maximum(z0 - z, 0.0), z - (z0 + side))
        return np.maximum(np.sqrt(dx * dx + dz * dz + height * height), 1e-3)

    def _children(self, node):
        """The node's existing children, asked of `exists` once and kept: the tree does not change."""
        kids = self._kids.get(node)
        if kids is None:
            L, a, b = node
            kids = [] if L == 0 else [k for k in ((L - 1, 2 * a + da, 2 * b + db) for da in (0, 1) for db in (0, 1))
                                      if self.exists(*k)]
            self._kids[node] = kids
        return kids

    def _select(self, x, z, height) -> list:
        """The nodes to draw this frame, nearest first: the kept cut, refined within `refine_ms`.

        The cut tiles the tree. Its nodes are checked nearest first, a pass over all of them at a time,
        carried over to the next frame when the budget runs out: a node whose error is over `tau` splits
        into its children, and four siblings merge into their parent once its error is under tau / merge.
        Without a budget, passes run until one changes nothing, which is the full walk's answer: a
        parent's error is at least twice its child's, so a fixed point of these two rules is the walk's."""
        cam = self.engine.camera
        focal = self.engine.h / (2.0 * math.tan(math.radians(cam.fov) / 2.0))
        here = (x, z, height, cam.fov, cam.far, self.engine.h, self.tau)
        first = self._cut is None
        if first:  # refined to the end: the first frame asks for the nodes it will draw, not for coarser ones
            root = (self.levels, 0, 0)
            self._cut = {root} if self.exists(*root) else set()
        stop = None if self.refine_ms is None or first else perf_counter() + self.refine_ms / 1000.0
        checked = 0
        while True:
            if not self._queue:
                if self._pass_quiet and self._pass_started:  # the last pass changed nothing: settled
                    self._pass_started = False
                    break
                if not self._pass_started and not self._moved(self._pass_at, here):
                    break  # settled for a camera this close: nothing new to check
                self._pass_at = here
                cut = list(self._cut)
                self._queue = [cut[k] for k in np.argsort(-self._distances(cut, x, z, height), kind="stable")]
                self._pass_quiet, self._pass_started = True, True
                if not self._queue:
                    break
            if stop is not None and checked >= 16 and perf_counter() > stop:
                break
            self._check(self._queue.pop(), x, z, height, focal, cam.far)
            checked += 1
        if self._cut_changed or self._order is None:
            self._order = list(self._cut)
            self._order_boxes = self._boxes(self._order) if len(self._order) >= _VECTOR_FROM else None
            self._cut_changed, self._sel = False, None
        if self._sel is None or self._moved(self._sel_at, here):
            d = self._distances(self._order, x, z, height, self._order_boxes)
            near = np.flatnonzero(d <= cam.far)
            self._sel = [self._order[k] for k in near[np.argsort(d[near], kind="stable")]]
            self._sel_at = here
        return self._sel

    def _moved(self, before, here) -> bool:
        """Whether the camera at `here` is `move` metres or more from `before`, or its fov, far plane,
        viewport or tau changed."""
        if before is None or before[3:] != here[3:]:
            return True
        return math.dist(before[:3], here[:3]) >= self.move

    def _check(self, node, x, z, height, focal, far):
        """Split `node` or merge it with its siblings, if its error says so; queue what changed."""
        if node not in self._cut:  # merged away earlier in this pass
            return
        d = self._distance(node, x, z, height)
        kids = self._children(node)
        if kids and d <= far and self.error(node[0]) * focal / d > self.tau:
            self._cut.discard(node)
            self._cut.update(kids)
            self._queue.extend(sorted(kids, key=lambda n: -self._distance(n, x, z, height)))
            self._pass_quiet, self._cut_changed = False, True
            return
        L, a, b = node
        if L >= self.levels:
            return
        parent = (L + 1, a >> 1, b >> 1)
        sibs = self._children(parent)
        if all(s in self._cut for s in sibs) and \
                self.error(L + 1) * focal / self._distance(parent, x, z, height) <= self.tau / self.merge:
            self._cut.difference_update(sibs)
            self._cut.add(parent)
            self._queue.append(parent)  # it may merge on up
            self._pass_quiet, self._cut_changed = False, True

    def _ancestors(self, node):
        L, a, b = node
        for up in range(1, self.levels - L + 1):
            yield (L + up, a >> up, b >> up)

    def focus(self, x: float, z: float, height: float = 0.0) -> None:
        """Move the camera to world position (x, height, z) and stream for it. Call it once a frame.

        A camera that has not moved, with every selected node drawn and nothing building, costs
        nothing: the last frame's answer stands."""
        got = self._tick()
        if self._fading:
            self._fade()
        cam = self.engine.camera
        still = (float(x), float(z), float(height), cam.fov, cam.far, self.engine.h, self.tau)
        last = self._still
        if (self._quiet and not got and not self._fading and last is not None and last[3:] == still[3:]
                and max(abs(last[0] - still[0]), abs(last[1] - still[1]), abs(last[2] - still[2])) < self.cell * 1e-3):
            return
        self._still = still
        selected = self._select(float(x), float(z), float(height))
        if not got and selected is self._shown_for and self._landed == self._shown_landed:
            # Nothing that decides what to draw changed: carry on the uploads in flight, nearest first.
            room = self.upload_bytes_per_frame
            for n in self._to_upload:
                if room <= 0:
                    break
                if n in self._ready and not self._on_gpu(n):
                    if self._is_deferred():
                        self._upload(n, room)
                        room = 0
                    else:
                        room -= self._upload(n, room)
            self._quiet = not (self._queue or self._pass_started or self._pending or self._waiting)
            return
        self._selected = set(selected)
        landed = self._landed  # what the draw set below sees; an upload finished after it calls for another
        show, wanted, parked = self._plan(selected, x, z, height)

        # Build what is wanted, by its error on screen, `in_flight` jobs at a time: each job that finishes
        # submits the next (_next), so no backlog of jobs picked for an old camera builds first, and the
        # workers never wait for a frame.
        keep = self._selected | show | set(wanted) | set(parked) | self._fading.keys()
        with self._lock:
            for n in [n for n in self._pending if n not in keep]:
                if self._pending[n].cancel():  # one already building lands in the cache instead
                    del self._pending[n]
                    self._note("cancelled", n)
            todo = [n for n in wanted if n not in self._ready and n not in self._pending]
            self._todo = todo[::-1]  # most error last, for pop()
            while self._todo and len(self._pending) < self.in_flight:
                self._submit(self._todo.pop())
        room = self.upload_bytes_per_frame
        self._to_upload = [n for n in wanted if self._ready.get(n) is not None and not self._on_gpu(n)]
        for n in self._to_upload:
            if room <= 0:
                break
            if self._is_deferred():
                self._upload(n, room)  # queued whole at apply: it spends the frame's budget
                room = 0
            else:
                room -= self._upload(n, room)

        for n in [n for n in self._drawn if n not in show]:
            self._undraw(n)
        for n in sorted(show - set(self._drawn), key=lambda n: self._distance(n, x, z, height)):
            x0, z0, _ = self._box(n)
            self._draw(n, (x0, 0.0, z0))

        # The built nodes in use move to the end of _ready; past `cache` others, the least recently used go.
        for n in keep:
            if n in self._ready:
                self._ready[n] = self._ready.pop(n)
        spare = [n for n in self._ready if n not in keep]
        for n in spare[: max(0, len(spare) - self.cache)]:
            del self._ready[n]
            self._unload(n)
            self._stats["dropped"] += 1
            self._note("dropped", n)
        # Quiet: the refinement settled, nothing builds, and every selected node is drawn.
        self._shown_for, self._shown_landed, self._waiting = selected, landed, bool(wanted)
        self._quiet = not (self._queue or self._pass_started or self._pending or self._waiting)

    def _submit(self, n):
        """Start building node n; call it holding the lock."""
        self._pending[n] = fut = self._pool.submit(_pack, self.build, *n)
        self._note("submitted", n)
        fut.add_done_callback(self._next)

    def _next(self, fut):
        """A build finished, in a worker's thread or the pool's: submit the next node to build."""
        if fut.cancelled():
            return
        with self._lock:
            while self._todo:
                n = self._todo.pop()
                if n not in self._pending:
                    self._submit(n)
                    return

    def close(self) -> None:
        with self._lock:
            self._todo = []  # no done callback submits to a pool shut down
        for c, (handle, _, out) in list(self._fading.items()):
            if out:
                handle.destroy(free_geometry=False)
        self._fading.clear()
        super().close()

    def _plan(self, selected, x, z, height) -> tuple[set, list, list]:
        """What to draw for the selected nodes, the nodes to build for it, most error on screen first, and
        the built nodes waiting under a node drawn in their place, to keep till their siblings land.

        A selected node is drawn once it is built and on the GPU; till then the nodes drawn under it stay.
        Above them, a node is drawn in place of its children until every one of them is covered, so each
        level shows as soon as it is whole, and its children are the ones to build next. A spot nothing
        covers yet asks for the parents of its picked nodes, not the root: after a teleport the first
        detail comes in one build, and no node drawn is much larger than the view."""
        sel = self._selected
        inner = set()  # the ancestors of the selected nodes
        for n in selected:
            for u in self._ancestors(n):
                if u in inner:
                    break
                inner.add(u)
        below: dict | None = None  # node -> the drawn nodes under it, built on need
        wanted, parked = [], []

        def ready(n):
            return n in self._ready and self._on_gpu(n)

        def cover(n):
            """The nodes that draw n's square now, and whether they cover all of what is picked in it."""
            nonlocal below
            if n in sel:
                if ready(n):
                    return [n], True
                wanted.append(n)
                if below is None:
                    below = {}
                    for d in self._drawn:
                        for u in self._ancestors(d):
                            below.setdefault(u, []).append(d)
                kept = below.get(n, [])
                return kept, bool(kept)
            if n not in inner:
                return [], True  # nothing in it is picked: beyond the far plane
            parts, short = [], []
            for k in self._children(n):
                p, whole = cover(k)
                parts += p
                if not whole:
                    short.append(k)
            if not short:
                return parts, True
            if ready(n):
                wanted.extend(k for k in short if k in inner and not ready(k))
                parked.extend(parts)
                return [n], True
            if (not parts and any(k in sel for k in self._children(n))
                    and not any(u in self._drawn for u in self._ancestors(n))):
                wanted.append(n)  # nothing drawn here: the level just above the picked nodes shows first
            return parts, False

        show = set()
        if selected:
            root = (self.levels, 0, 0)
            parts, _ = cover(root)
            show.update(parts)
        if wanted:
            cam = self.engine.camera
            focal = self.engine.h / (2.0 * math.tan(math.radians(cam.fov) / 2.0))
            d = self._distances(wanted, x, z, height)
            err = np.array([self.error(n[0]) for n in wanted]) * focal / d
            wanted = [wanted[k] for k in np.argsort(-err, kind="stable")]
        return show, wanted, parked

    @_deferred.deferred_in_pipeline(find=_streamer_engine)
    def _draw(self, c, pos):
        if not self.crossfade:
            return super()._draw(c, pos)
        old = self._fading.pop(c, None)
        if old is not None and old[2]:  # given way and back before it faded out: it fades in from what shows
            handle, started, _ = old
            self._drawn[c] = handle
            self._note("drawn", c)
            self._fading[c] = (handle, 2 * perf_counter() - started - self.crossfade, False)
            return
        super()._draw(c, pos)
        if self._drawn[c] is not None:
            self._fading[c] = (self._drawn[c], perf_counter(), False)

    def _undraw(self, c):
        if not self.crossfade or self._drawn.get(c) is None:
            self._fading.pop(c, None)
            return super()._undraw(c)
        handle = self._drawn.pop(c)
        self._note("undrawn", c)
        old = self._fading.get(c)
        now = perf_counter()
        # Fading out from what shows now: a node still fading in starts out where its fade in stands.
        started = now if old is None else 2 * now - old[1] - self.crossfade
        self._fading[c] = (handle, started, True)

    @_deferred.deferred_in_pipeline(find=_streamer_engine)
    def _set_fade(self, handle, value):
        self.engine.store._components["Fade"][handle.index] = value

    def _fade(self):
        """Step every fade; a node faded out is destroyed and stays built, for the cache to keep or free."""
        now = perf_counter()
        for c, (handle, started, out) in list(self._fading.items()):
            t = min((now - started) / self.crossfade, 1.0)
            if t < 1.0:
                self._set_fade(handle, -(1.0 - t) if out else t)
                continue
            del self._fading[c]
            if not out:
                self._set_fade(handle, 1.0)
                continue
            handle.destroy(free_geometry=False)  # still built: the cache frees it when it must

    def drawn(self) -> set[tuple[int, int, int]]:
        """The nodes drawn now, empty ones (build returned None) included; not those fading out."""
        return set(self._drawn)

    @property
    def settled(self) -> bool:
        """Nothing building or uploading, and every node the camera needs is drawn."""
        return self._quiet and not self._fading

    def stats(self) -> dict:
        """held (built or building), drawn nodes by level, built, dropped, and the longest frame in ms."""
        by_level: dict[int, int] = {}
        for L, _a, _b in self._drawn:
            by_level[L] = by_level.get(L, 0) + 1
        return {**self._stats, "held": len(self._ready) + len(self._pending), "by_level": by_level}


__all__ = ["MeshData", "QuadStreamer", "Streamer"]
