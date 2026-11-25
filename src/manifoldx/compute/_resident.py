"""The isolated E2 WGSL lane on an explicitly supplied graphics device."""

import math
import struct
from dataclasses import dataclass
from operator import index
from time import perf_counter_ns

import wgpu

from manifoldx.compute.kernel import _VERSION, _Kernel

_WORKGROUP = 64
_MAX_PIPELINES = 32
_MAX_PENDING = 64
_MAX_TRANSFER = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _GpuReadback:
    version: int
    data: bytes
    frame: int
    dt: float | None


class GpuKernelGroup:
    """Execute v1 WGSL kernels over one resident component column.

    The transition boundary accepts packed little-endian bytes and a slot
    mask. It does not borrow a World or claim a connection to its renderer.
    Use one group per explicit graphics owner, on its owning thread.
    """

    backend = "wgsl"

    def __init__(self, device, program, *, rows, target):
        if target != self.backend:
            raise ValueError("GpuKernelGroup requires target='wgsl'")
        if device is None:
            raise RuntimeError("WGSL kernel backend requires a graphics device")
        if not isinstance(program, _Kernel) or program.version != _VERSION:
            raise ValueError("WGSL backend requires a v1 @kernel declaration")
        rows = index(rows)
        width = program.width
        size = rows * width * 4
        max_size = min(
            _MAX_TRANSFER,
            device.limits["max-buffer-size"],
            device.limits["max-storage-buffer-binding-size"],
        )
        if (
            not 0 < size <= max_size
            or (rows + _WORKGROUP - 1) // _WORKGROUP
            > device.limits["max-compute-workgroups-per-dimension"]
        ):
            raise ValueError("WGSL kernel row count exceeds the device or transfer limit")
        self._device = device
        self._program = program
        self._rows = rows
        self._size = size
        self._loaded = False
        self._pending = 0
        self._pending_bytes = 0
        self._version = 0
        self._dt = None
        self._snapshot = None
        self._pipelines = {}
        self._counts = dict.fromkeys(
            (
                "dispatches",
                "upload_calls",
                "upload_bytes",
                "upload_rows",
                "readback_calls",
                "readback_bytes",
                "readback_rows",
                "readback_wait_ns",
                "readback_consumers",
                "uniform_calls",
                "uniform_bytes",
                "queue_waits",
                "queue_wait_ns",
            ),
            0,
        )
        self._data = device.create_buffer(
            size=size,
            usage=(
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            ),
        )
        self._selected = device.create_buffer(
            size=rows * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._uniform = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._pipeline(program)

    def _pipeline(self, program):
        if not isinstance(program, _Kernel) or program.version != _VERSION:
            raise ValueError("WGSL backend requires a v1 @kernel declaration")
        if (
            program.component is not self._program.component
            or program.layout != self._program.layout
            or program.width != self._program.width
        ):
            raise ValueError("WGSL kernel column differs from the compiled schema")
        found = self._pipelines.get(program)
        if found is not None:
            return found
        if len(self._pipelines) >= _MAX_PIPELINES:
            raise ValueError("WGSL kernel pipeline limit reached")
        visibility = wgpu.ShaderStage.COMPUTE
        entries = [
            {
                "binding": binding,
                "visibility": visibility,
                "buffer": {"type": kind},
            }
            for binding, kind in enumerate(
                (
                    "storage" if program.writes else "read-only-storage",
                    "read-only-storage",
                    "uniform",
                )
            )
        ]
        layout = self._device.create_bind_group_layout(entries=entries)
        bindings = self._device.create_bind_group(
            layout=layout,
            entries=[
                {"binding": binding, "resource": {"buffer": buffer}}
                for binding, buffer in enumerate((self._data, self._selected, self._uniform))
            ],
        )
        pipeline = self._device.create_compute_pipeline(
            layout=self._device.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={
                "module": self._device.create_shader_module(code=program.wgsl()),
                "entry_point": "main",
            },
        )
        self._pipelines[program] = (pipeline, bindings)
        return pipeline, bindings

    def _admit(self, size):
        if self._pending >= _MAX_PENDING or self._pending_bytes + size > 2 * _MAX_TRANSFER:
            raise RuntimeError("WGSL submission credit exhausted; wait before submitting")

    def _submitted(self, size):
        self._pending += 1
        self._pending_bytes += size

    def wait(self):
        """Wait for outstanding compute and release submission credit without a readback."""
        if self._pending:
            start = perf_counter_ns()
            self._device.queue.on_submitted_work_done_sync()
            self._counts["queue_waits"] += 1
            self._counts["queue_wait_ns"] += perf_counter_ns() - start
            self._pending = self._pending_bytes = 0

    def load(self, data, selected):
        """Publish packed component rows and a u32 zero/one mask from CPU."""
        if (
            not isinstance(data, bytes)
            or len(data) != self._size
            or not isinstance(selected, bytes)
            or len(selected) != self._rows * 4
        ):
            raise ValueError("WGSL input bytes must match the component and slot mask extents")
        size = len(data) + len(selected)
        self._admit(size)
        queue = self._device.queue
        queue.write_buffer(self._data, 0, data)
        queue.write_buffer(self._selected, 0, selected)
        queue.submit([])
        self._submitted(size)
        self._counts["upload_calls"] += 2
        self._counts["upload_bytes"] += size
        self._counts["upload_rows"] += self._rows
        self._loaded = True
        self._version += 1
        self._snapshot = None
        self._dt = None

    def dispatch(self, program, *, dt, timestamp_writes=None):
        """Dispatch WGSL; keep its writes on the device for the next consumer."""
        if not self._loaded:
            raise RuntimeError("upload the component and slot mask before dispatch")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("kernel dt must be finite and nonnegative")
        try:
            uniforms = struct.pack("<IIfI", _VERSION, self._rows, dt, 0)
        except (OverflowError, struct.error) as error:
            raise ValueError("kernel dt must fit in f32") from error
        self._admit(len(uniforms))
        pipeline, bindings = self._pipeline(program)
        queue = self._device.queue
        encoder = self._device.create_command_encoder()
        compute = encoder.begin_compute_pass(timestamp_writes=timestamp_writes)
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, bindings)
        compute.dispatch_workgroups((self._rows + _WORKGROUP - 1) // _WORKGROUP)
        compute.end()
        queue.write_buffer(self._uniform, 0, uniforms)
        queue.submit([encoder.finish()])
        self._submitted(len(uniforms))
        self._counts["dispatches"] += 1
        self._counts["uniform_calls"] += 1
        self._counts["uniform_bytes"] += len(uniforms)
        if program.writes:
            self._version += 1
            self._snapshot = None
            self._dt = struct.unpack_from("<f", uniforms, 8)[0]

    def readback(self):
        """Wait for and return immutable bytes for the current version."""
        if not self._loaded:
            raise RuntimeError("upload the component and slot mask before readback")
        if self._snapshot is None:
            start = perf_counter_ns()
            data = bytes(self._device.queue.read_buffer(self._data, 0, self._size))
            self._pending = self._pending_bytes = 0
            self._counts["readback_wait_ns"] += perf_counter_ns() - start
            self._counts["readback_calls"] += 1
            self._counts["readback_bytes"] += len(data)
            self._counts["readback_rows"] += self._rows
            self._snapshot = _GpuReadback(self._version, data, self._counts["dispatches"], self._dt)
        self._counts["readback_consumers"] += 1
        return self._snapshot

    def counters(self):
        """Report GPU dispatch and transfer counts, in bytes and rows."""
        return {**self._counts, "version": self._version}
