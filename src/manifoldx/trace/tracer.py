"""The GPU side: scene buffers, compute pipelines, progressive accumulation."""

from __future__ import annotations

import time
from importlib import resources

import numpy as np
import wgpu

from manifoldx.trace import bvh as _bvh
from manifoldx.trace.scene import Camera, Scene

_WG = 8
_PARAMS_SIZE = 144


def _wgsl(name: str) -> str:
    return resources.files("manifoldx.trace").joinpath("wgsl", f"{name}.wgsl").read_text()


def _device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("no wgpu adapter: the tracer needs a GPU or a software Vulkan driver")
    limits = adapter.limits
    want = {k: limits[k] for k in ("max-storage-buffer-binding-size", "max-buffer-size",
                                   "max-storage-buffers-per-shader-stage") if k in limits}
    return adapter.request_device_sync(required_limits=want)


def pack_colors(c: np.ndarray) -> np.ndarray:
    """(..., 3) linear [0, 1] -> u32 as 11, 11 and 10 bit unorm."""
    c = np.clip(c, 0.0, 1.0)
    r = np.round(c[..., 0] * 2047).astype(np.uint32)
    g = np.round(c[..., 1] * 2047).astype(np.uint32)
    b = np.round(c[..., 2] * 1023).astype(np.uint32)
    return r | (g << 11) | (b << 22)


class Tracer:
    """Progressive path tracer over a Scene.

    tracer.step(n) adds n samples per pixel; tracer.image() returns the sRGB result.
    Changing the camera restarts the accumulation."""

    def __init__(self, scene: Scene, width: int, height: int, *, camera: Camera | None = None,
                 max_bounces: int = 6, seed: int = 0, device=None):
        self.scene = scene
        self.width, self.height = int(width), int(height)
        self.max_bounces = int(max_bounces)
        self.seed = int(seed)
        self.device = device or _device()
        self._camera = camera or scene.camera or _default_camera(scene)
        t = time.perf_counter()
        self.bvh = _bvh.build(scene.positions)
        self.build_seconds = time.perf_counter() - t
        self._upload()
        self._pipelines = {}
        self.samples = 0

    # ------------------------------------------------------------ camera

    @property
    def camera(self) -> Camera:
        return self._camera

    @camera.setter
    def camera(self, cam: Camera):
        self._camera = cam
        self.reset()

    def reset(self):
        self.samples = 0

    # ------------------------------------------------------------ upload

    def _buffer(self, data: np.ndarray, usage=wgpu.BufferUsage.STORAGE):
        data = np.ascontiguousarray(data)
        if data.nbytes == 0:
            data = np.zeros(4, np.uint32)
        return self.device.create_buffer_with_data(data=data.tobytes(), usage=usage)

    def _upload(self):
        s, order = self.scene, self.bvh.order
        p = s.positions[order]
        tris = np.zeros((len(p), 3, 4), np.float32)
        tris[:, 0, :3] = p[:, 0]
        tris[:, 1, :3] = p[:, 1] - p[:, 0]
        tris[:, 2, :3] = p[:, 2] - p[:, 0]

        n, uv, col = s.normals[order], s.uvs[order], s.colors[order]
        shade = np.zeros((len(p), 5, 4), np.float32)
        shade[:, 0:3, :3] = n
        shade[:, 0, 3] = s.material_ids[order].view(np.float32)
        shade[:, 1, 3], shade[:, 2, 3] = uv[:, 0, 0], uv[:, 0, 1]
        shade[:, 3, :2], shade[:, 3, 2:] = uv[:, 1], uv[:, 2]
        packed = shade.view(np.uint32)
        packed[:, 4, :3] = pack_colors(col)

        mats = np.zeros((max(len(s.materials), 1), 3, 4), np.float32)
        for i, m in enumerate(s.materials):
            mats[i, 0] = (*m.base_color, m.metallic)
            mats[i, 1] = (*m.emission, m.roughness)
            mats[i, 2] = (m.transmission, m.ior, 1.0 if m.thin else 0.0, m.texture)

        table, chunks, offset = [], [], 0
        for tex in s.textures:
            h, w = tex.shape[:2]
            table.append((offset, w, h, 0))
            chunks.append(np.ascontiguousarray(tex, np.uint8).view(np.uint32).reshape(-1))
            offset += w * h
        texels = np.concatenate(chunks) if chunks else np.zeros(1, np.uint32)
        table = np.asarray(table or [(0, 1, 1, 0)], np.uint32)

        lo, hi = s.bounds()
        self.scene_radius = float(np.linalg.norm(hi - lo)) * 0.5 + 1e-3
        self._params = self.device.create_buffer(size=_PARAMS_SIZE,
                                                 usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._scene_buffers = [self._params, self._buffer(self.bvh.packed()), self._buffer(tris),
                               self._buffer(shade), self._buffer(mats), self._buffer(texels),
                               self._buffer(table)]
        self.gpu_bytes = sum(b.size for b in self._scene_buffers)

    def _out(self, nbytes):
        return self.device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                                         | wgpu.BufferUsage.COPY_DST)

    # ------------------------------------------------------------ pipelines

    def _pipeline(self, name: str, extra: int):
        if name not in self._pipelines:
            source = _wgsl("common") + "\n" + _wgsl(name)
            module = self.device.create_shader_module(code=source)
            C = wgpu.ShaderStage.COMPUTE
            entries = [{"binding": 0, "visibility": C, "buffer": {"type": wgpu.BufferBindingType.uniform}}]
            entries += [{"binding": b, "visibility": C,
                         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}} for b in range(1, 7)]
            entries += [{"binding": 7 + k, "visibility": C,
                         "buffer": {"type": wgpu.BufferBindingType.storage}} for k in range(extra)]
            layout = self.device.create_bind_group_layout(entries=entries)
            pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": module, "entry_point": "main"})
            self._pipelines[name] = (pipeline, layout)
        return self._pipelines[name]

    def _bind(self, name, outputs):
        pipeline, layout = self._pipeline(name, len(outputs))
        buffers = self._scene_buffers + list(outputs)
        group = self.device.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(buffers)])
        return pipeline, group

    def _write_params(self, frame=0):
        cam = self._camera
        f, r, u = cam.basis()
        buf = np.zeros(_PARAMS_SIZE // 4, np.float32)
        ints = buf.view(np.uint32)
        buf[0:3], buf[3] = cam.position, np.tan(np.radians(cam.fov) / 2)
        buf[4:7], buf[7] = f, self.width / self.height
        buf[8:11], ints[11] = r, self.width
        buf[12:15], ints[15] = u, self.height
        ints[16:20] = (frame, self.seed, self.max_bounces, 0)
        buf[33] = self.scene_radius
        self.device.queue.write_buffer(self._params, 0, buf.tobytes())

    def _dispatch(self, encoder, pipeline, group):
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((self.width + _WG - 1) // _WG, (self.height + _WG - 1) // _WG)
        cp.end()

    def _read(self, buf, channels=4):
        data = np.frombuffer(self.device.queue.read_buffer(buf), np.float32)
        return data.reshape(self.height, self.width, channels)

    # ------------------------------------------------------------ primary rays

    def aovs(self) -> dict:
        """First hit through each pixel centre: albedo (linear), normal and distance (-1 on a miss)."""
        n = self.width * self.height * 16
        albedo, normal_depth = self._out(n), self._out(n)
        pipeline, group = self._bind("primary", [albedo, normal_depth])
        self._write_params()
        enc = self.device.create_command_encoder()
        self._dispatch(enc, pipeline, group)
        self.device.queue.submit([enc.finish()])
        nd = self._read(normal_depth)
        return {"albedo": self._read(albedo)[..., :3], "normal": nd[..., :3], "depth": nd[..., 3]}


def _default_camera(scene: Scene) -> Camera:
    lo, hi = scene.bounds()
    c = (lo + hi) / 2
    r = float(np.linalg.norm(hi - lo)) / 2
    return Camera(position=tuple(c + np.array([0.6, 0.5, 1.0]) * r * 1.6), target=tuple(c), fov=50.0)
