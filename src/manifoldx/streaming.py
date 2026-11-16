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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter
from typing import Callable, NamedTuple

import numpy as np

from manifoldx.renderer import vertex_bounds
from manifoldx.resources import pack_vertices


class MeshData(NamedTuple):
    """A cell's mesh: positions relative to the cell's corner (i*cell, 0, j*cell)."""

    positions: np.ndarray  # f32[N, 3]
    normals: np.ndarray  # f32[N, 3]
    colors: np.ndarray  # f32[N, 3], linear
    indices: np.ndarray  # u32[M]
    surfaces: np.ndarray | None = None  # i[N]: surface pattern ids (resources.SURFACE_PATTERNS)


class _Cell:
    """A built cell: its geometry dict, and its packed bytes until they are on the GPU."""

    __slots__ = ("geo", "info", "vbytes", "ibytes")

    def __init__(self, geo, info, vbytes, ibytes):
        self.geo, self.info, self.vbytes, self.ibytes = geo, info, vbytes, ibytes


def _pack(build, *key):
    """The worker's job: build cell (i, j), or node (L, a, b), and pack it for its GPU buffers."""
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
    vbytes, info = pack_vertices(geo)
    if len(geo["positions"]):
        geo["bounds"] = vertex_bounds(geo["positions"])  # the frustum test reads it, not the frame
    return _Cell(geo, info, vbytes, geo["indices"].tobytes())


class _Uploads:
    """What both streamers share: the worker, the GPU uploads in slices, drawing and freeing by key.

    A key is a cell (i, j) for Streamer and a node (L, a, b) for QuadStreamer; `_draw(key, pos)` spawns
    the key's mesh at world position pos.
    """

    def _start(self, engine, build, uploads_per_frame, upload_bytes_per_frame, workers, material, builders=1):
        if workers == "thread":
            self._pool = ThreadPoolExecutor(max_workers=builders)
        elif workers == "process":
            self._pool = ProcessPoolExecutor(
                max_workers=builders, mp_context=multiprocessing.get_context("spawn")
            )
        else:
            raise ValueError(f'workers must be "thread" or "process", got {workers!r}')
        if material is None:
            from manifoldx.resources import StandardMaterial

            material = StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True)
        self.engine, self.build = engine, build
        self.uploads_per_frame = uploads_per_frame
        self.material = material
        self.upload_bytes_per_frame = max(4, int(upload_bytes_per_frame) // 4 * 4)
        self._pending = {}  # key -> Future
        self._ready = {}  # key -> _Cell | None, built and kept
        self._gpu = {}  # key -> its upload: geometry id, buffers, bytes written, done
        self._drawn = {}  # key -> EntityHandle | None (None: an empty cell)
        self._stats = {"built": 0, "uploaded": 0, "dropped": 0, "max_frame_ms": 0.0}
        self._last = None

    def _tick(self) -> int:
        """Note the time since the last focus() and collect the finished builds; returns how many."""
        now = perf_counter()
        if self._last is not None:
            self._stats["max_frame_ms"] = max(self._stats["max_frame_ms"], (now - self._last) * 1000.0)
        self._last = now
        got = 0
        for c, fut in list(self._pending.items()):
            if fut.done():
                del self._pending[c]
                self._ready[c] = fut.result()
                self._stats["built"] += 1
                got += 1
        return got

    def _on_gpu(self, c) -> bool:
        """Whether cell c can be drawn without building a buffer on the frame."""
        if self._ready.get(c) is None or self.engine._device is None:
            return True  # nothing to upload, or no device yet: the render pass builds the buffers
        return self._gpu.get(c, {}).get("done", False)

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
                dev.queue.write_buffer(up["bufs"][k], at, memoryview(data)[at : at + n])
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
            cell.vbytes = cell.ibytes = None
        return spent

    def _unload(self, c):
        """Free cell c's GPU buffers, whole or half written."""
        up = self._gpu.pop(c, None)
        if up is None:
            return
        for buf in up.get("bufs", ()):
            buf.destroy()
        self.engine._geometry_registry.release(up["id"])

    def _draw(self, c, pos):
        cell = self._ready[c]
        if cell is None:
            self._drawn[c] = None
            return
        from manifoldx.components import Material, Mesh, Transform

        self._drawn[c] = self.engine.spawn(
            Mesh(cell.geo),
            Material(self.material),
            Transform(pos=pos),
        )
        # Drawn before any device: its buffers come from the render pass, and are still ours to free.
        self._gpu.setdefault(c, {"id": self._drawn[c]._geometry_id, "done": True})
        self._stats["uploaded"] += 1

    def _undraw(self, c):
        handle = self._drawn.pop(c)
        if handle is not None:
            handle.destroy(free_geometry=False)

    def close(self) -> None:
        """Stop the worker and destroy every drawn cell."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        for c in list(self._drawn):
            self._undraw(c)
        for c in list(self._gpu):
            self._unload(c)
        self._pending.clear()
        self._ready.clear()



class Streamer(_Uploads):
    """Builds, uploads and frees cell meshes around a focus point."""

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
    ):
        if ready_radius < radius:
            raise ValueError(f"ready_radius ({ready_radius}) must be at least radius ({radius})")
        self._start(engine, build, uploads_per_frame, upload_bytes_per_frame, workers, material)
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
        for c in [c for c in self._pending if ring(c) > self.ready_radius]:
            self._pending.pop(c).cancel()

        r = self.ready_radius
        wanted = sorted(
            ((fi + di, fj + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)),
            key=lambda c: (c[0] - fi) ** 2 + (c[1] - fj) ** 2,
        )
        for c in wanted:
            if c not in self._ready and c not in self._pending:
                self._pending[c] = self._pool.submit(_pack, self.build, *c)
        room = self.upload_bytes_per_frame
        for c in wanted:
            if room <= 0:
                break
            if self._ready.get(c) is not None and not self._on_gpu(c):
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
    a teleport costs a bounded slice of each frame; `refine_ms=None` refines to the end every frame. The nodes picked are built
    nearest first, by `builders` workers at once, and uploaded within `upload_bytes_per_frame`. One not
    built yet leaves its nearest drawn ancestor drawn, and a drawn node hides everything under it, so
    each spot is drawn once.

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
    ):
        self._start(engine, build, None, upload_bytes_per_frame, workers, material, builders)
        self.cell, self.origin, self.levels = float(cell), (int(origin[0]), int(origin[1])), int(levels)
        self.exists, self.error, self.tau, self.merge = exists, error, float(tau), float(merge)
        self.refine_ms = refine_ms
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

    def _distances(self, nodes, x, z, height) -> np.ndarray:
        """_distance for many nodes at once, as one array."""
        if not len(nodes):
            return np.zeros(0)
        L, a, b = np.asarray(nodes, np.int64).T
        side = (1 << L) * self.cell
        x0 = (self.origin[0] + a * (1 << L)) * self.cell
        z0 = (self.origin[1] + b * (1 << L)) * self.cell
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
        first = self._cut is None
        if first:  # refined to the end: coarse nodes drawn first would hide everything under them till all is built
            root = (self.levels, 0, 0)
            self._cut = {root} if self.exists(*root) else set()
        stop = None if self.refine_ms is None or first else perf_counter() + self.refine_ms / 1000.0
        checked = 0
        while True:
            if not self._queue:
                if self._pass_quiet and self._pass_started:  # the last pass changed nothing: settled
                    self._pass_started = False
                    break
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
            self._cut_changed = False
        d = self._distances(self._order, x, z, height)
        near = np.flatnonzero(d <= cam.far)
        return [self._order[k] for k in near[np.argsort(d[near], kind="stable")]]

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
        cam = self.engine.camera
        still = (float(x), float(z), float(height), cam.fov, cam.far, self.engine.h)
        last = self._still
        if (self._quiet and not got and last is not None and last[3:] == still[3:]
                and max(abs(last[0] - still[0]), abs(last[1] - still[1]), abs(last[2] - still[2])) < self.cell * 1e-3):
            return
        self._still = still
        selected = self._select(float(x), float(z), float(height))
        self._selected = set(selected)
        for n in selected:
            if n not in self._ready and n not in self._pending:
                self._pending[n] = self._pool.submit(_pack, self.build, *n)
        room = self.upload_bytes_per_frame
        for n in selected:
            if room <= 0:
                break
            if self._ready.get(n) is not None and not self._on_gpu(n):
                room -= self._upload(n, room)

        # What to draw: each selected node once it is built and on the GPU, else what covers its spot now;
        # then nothing under a node that is drawn, so four children replace their parent in one frame.
        # The selection is disjoint, so the ancestor map and filter are only needed once a node falls back.
        under = None  # node -> the drawn nodes below it, built on the first fallback
        show = set()
        fell_back = False
        for n in selected:
            if n in self._drawn or (n in self._ready and self._on_gpu(n)):
                show.add(n)
                continue
            fell_back = True  # not ready: what covers its spot now stays, a coarser node above or finer ones below
            up = next((u for u in self._ancestors(n) if u in self._drawn), None)
            if up is not None:
                show.add(up)
                continue
            if under is None:
                under = {}
                for d in self._drawn:
                    for u in self._ancestors(d):
                        under.setdefault(u, []).append(d)
            show.update(under.get(n, ()))
        if fell_back:
            show = {n for n in show if not any(u in show for u in self._ancestors(n))}
        for n in [n for n in self._drawn if n not in show]:
            self._undraw(n)
        for n in sorted(show - set(self._drawn), key=lambda n: self._distance(n, x, z, height)):
            x0, z0, _ = self._box(n)
            self._draw(n, (x0, 0.0, z0))

        keep = self._selected | show
        for n in [n for n in self._ready if n not in keep]:
            del self._ready[n]
            self._unload(n)
            self._stats["dropped"] += 1
        for n in [n for n in self._pending if n not in keep]:
            self._pending.pop(n).cancel()
        # Quiet: the refinement settled, nothing builds, and every selected node is drawn.
        self._quiet = not (self._queue or self._pass_started or self._pending or fell_back
                           or any(n not in self._drawn for n in selected))

    def drawn(self) -> set[tuple[int, int, int]]:
        """The nodes drawn now, empty ones (build returned None) included."""
        return set(self._drawn)

    def stats(self) -> dict:
        """held (built or building), drawn nodes by level, built, dropped, and the longest frame in ms."""
        by_level: dict[int, int] = {}
        for L, _a, _b in self._drawn:
            by_level[L] = by_level.get(L, 0) + 1
        return {**self._stats, "held": len(self._ready) + len(self._pending), "by_level": by_level}


__all__ = ["MeshData", "QuadStreamer", "Streamer"]
