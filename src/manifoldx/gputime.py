"""GPU time per pass, measured with timestamp queries and read back two frames late.

One frame in EVERY is measured: each pass the engine draws in it (the second cameras' views, the shadow map,
the main pass) writes a GPU timestamp where it begins and where it ends, and the frame's last encoder resolves
them into a buffer. Two frames later the GPU has long finished that frame, so mapping the buffer does not
stall, and the differences become milliseconds. Measuring every frame cost about 0.4 ms of CPU a frame. An adapter without the `timestamp-query` feature (a software rasterizer such
as llvmpipe) gets no timer, and `Engine.gpu_timings()` returns None; so does `MANIFOLDX_GPU_TIMINGS=0`.

Timestamps are taken to be nanoseconds, which they are on the drivers wgpu reports a one-nanosecond period for
(NVIDIA and AMD on Vulkan); wgpu-py does not expose the period to correct other drivers.
"""

from __future__ import annotations

import os

PASSES = ("views", "shadow", "main")
EVERY = 8  # frames between two measured ones
LATE = 2  # frames after a measured frame before its timestamps are read


def wanted(adapter) -> bool:
    """Whether to time this adapter's passes: it has timestamps and the environment does not turn them off."""
    return os.environ.get("MANIFOLDX_GPU_TIMINGS", "1") != "0" and "timestamp-query" in adapter.features


class GpuTimer:
    """A query set with a begin and an end slot per pass, and a buffer to read one measured frame back through."""

    def __init__(self, device):
        import wgpu

        n = 2 * len(PASSES)
        self.size = 8 * n
        self.queries = device.create_query_set(type=wgpu.QueryType.timestamp, count=n)
        # a resolve lands on a 256-byte boundary (QUERY_RESOLVE_BUFFER_ALIGNMENT): one block per pass
        self.resolved = device.create_buffer(size=256 * len(PASSES),
                                             usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
        self.readback = device.create_buffer(size=self.size,
                                             usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        self.frame = 0
        self.pending: tuple[int, set] | None = None  # (frame, passes) resolved into readback, not read yet
        self.current: set[str] = set()  # the passes that wrote both their timestamps this frame
        self.last: dict | None = None  # the newest frame read back, in milliseconds

    def measuring(self) -> bool:
        """Whether this frame is one of those measured."""
        return self.pending is None and self.frame % EVERY == 0

    def writes(self, name: str, begin: bool = True, end: bool = True) -> dict | None:
        """The `timestamp_writes` of a pass, or None on a frame not measured; several passes may share one
        name (begin on the first, end on the last), as the views do."""
        if not self.measuring():
            return None
        k = PASSES.index(name)
        writes = {"query_set": self.queries}
        # an index left out is undefined to wgpu; given as None it would become 0
        if begin:
            writes["beginning_of_pass_write_index"] = 2 * k
        if end:
            writes["end_of_pass_write_index"] = 2 * k + 1
            self.current.add(name)
        return writes if begin or end else None

    def resolve(self, encoder) -> None:
        """In the frame's last encoder, before it is submitted: copy a measured frame's timestamps out."""
        if self.measuring() and self.current:
            # Only the queries written this frame: on Vulkan, resolving one never written waits for it forever.
            for name in self.current:
                k = PASSES.index(name)
                encoder.resolve_query_set(self.queries, 2 * k, 2, self.resolved, 256 * k)
                encoder.copy_buffer_to_buffer(self.resolved, 256 * k, self.readback, 16 * k, 16)
            self.pending = (self.frame, self.current)
        self.current = set()
        self.frame += 1

    def read(self) -> dict | None:
        """Read the measured frame once LATE frames have passed, into `last`; return `last`."""
        import numpy as np
        import wgpu

        if self.pending is None or self.frame - self.pending[0] < LATE:
            return self.last
        _, names = self.pending
        self.pending = None
        self.readback.map_sync(wgpu.MapMode.READ)
        ticks = np.frombuffer(bytes(self.readback.read_mapped()), np.uint64).astype(np.int64)
        self.readback.unmap()
        out, spans = {}, []
        for k, name in enumerate(PASSES):
            if name in names:
                begin, end = ticks[2 * k], ticks[2 * k + 1]
                out[name] = max(0, int(end - begin)) / 1e6
                spans.append((begin, end))
        out["total"] = max(0, int(max(e for _, e in spans) - min(b for b, _ in spans))) / 1e6
        self.last = out
        return self.last
