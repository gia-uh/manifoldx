"""The GPU side: scene buffers, compute pipelines, progressive accumulation."""

from __future__ import annotations

import time
from importlib import resources

import numpy as np
import wgpu

from manifoldx.trace import bvh as _bvh
from manifoldx.trace.environment import Environment
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
                 environment: Environment | None = None, max_bounces: int = 6, seed: int = 0,
                 clamp: float = 0.0, diffuse_only: bool = False, device=None):
        """environment: the scene's, else a clear sky with the sun at 40° (Environment.sky()).
        clamp > 0 caps each indirect sample's radiance (less noise, a little bias)."""
        self.scene = scene
        self.environment = environment or scene.environment or Environment.sky()
        self.clamp = float(clamp)
        self.diffuse_only = diffuse_only
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
        self._upload_lights(mats)
        self.gpu_bytes = sum(b.size for b in self._scene_buffers + self._light_buffers)
        n = self.width * self.height * 16
        self._accum, self._aov_albedo, self._aov_nd = self._out(n), self._out(n), self._out(n)

    def _upload_lights(self, mats):
        s, order = self.scene, self.bvh.order
        env = self.environment
        rad, marginal, cond, total = env.tables()
        self._env_on = bool(np.asarray(env.radiance).max() * env.intensity > 0)
        self._env_shape = env.radiance.shape[:2] if self._env_on else (0, 0)

        lights = np.zeros((max(len(s.lights), 1), 3, 4), np.float32)
        for i, light in enumerate(s.lights):
            kind = {"directional": 0.0, "point": 1.0, "spot": 2.0}[light.kind]
            power = np.asarray(light.color, np.float32) * light.intensity
            lights[i, 0] = (*light.position, kind)
            lights[i, 1] = (*(np.asarray(light.direction) / np.linalg.norm(light.direction)),
                            np.cos(light.outer_cone))
            lights[i, 2] = (*power, np.cos(light.inner_cone))

        # emissive triangles, in BVH order, with a CDF by power (luminance × area)
        emission = mats[:, 1, :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        tri_mats = s.material_ids[order]
        hot = np.flatnonzero(emission[tri_mats] > 0)
        self.emissive_total = 0.0
        table = np.zeros((max(len(hot), 1), 4), np.float32)
        if len(hot):
            p = s.positions[order][hot].astype(np.float64)
            area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
            w = emission[tri_mats[hot]] * area
            self.emissive_total = float(w.sum())
            table.view(np.uint32)[:, 0] = hot.astype(np.uint32)
            table[:, 1] = np.cumsum(w) / w.sum()
            table[-1, 1] = 1.0
        self._emissive_count = len(hot) if self.emissive_total > 0 else 0
        self._env_total = total
        self._light_buffers = [self._buffer(rad), self._buffer(marginal), self._buffer(cond),
                               self._buffer(lights), self._buffer(table)]

    def _out(self, nbytes):
        return self.device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                                         | wgpu.BufferUsage.COPY_DST)

    # ------------------------------------------------------------ pipelines

    def _pipeline(self, name: str, extra):
        """extra: one read_only flag per binding after the scene's."""
        if name not in self._pipelines:
            source = _wgsl("common") + "\n" + _wgsl(name)
            module = self.device.create_shader_module(code=source)
            C = wgpu.ShaderStage.COMPUTE
            entries = [{"binding": 0, "visibility": C, "buffer": {"type": wgpu.BufferBindingType.uniform}}]
            entries += [{"binding": b, "visibility": C,
                         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}} for b in range(1, 7)]
            T = wgpu.BufferBindingType
            entries += [{"binding": 7 + k, "visibility": C,
                         "buffer": {"type": T.read_only_storage if ro else T.storage}} for k, ro in enumerate(extra)]
            layout = self.device.create_bind_group_layout(entries=entries)
            pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": module, "entry_point": "main"})
            self._pipelines[name] = (pipeline, layout)
        return self._pipelines[name]

    def _bind(self, name, outputs, read_only=None):
        pipeline, layout = self._pipeline(name, tuple(read_only or [False] * len(outputs)))
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
        ints[16:20] = (frame, self.seed, self.max_bounces, len(self.scene.lights))
        ints[20], ints[21] = self._env_shape[1], self._env_shape[0]
        buf[22], ints[23] = self.environment.intensity, self._emissive_count
        env = self.environment
        if env.sun_direction is not None and env.sun_illuminance > 0:
            d = np.asarray(env.sun_direction, np.float64)
            cos = np.cos(env.sun_angular_radius)
            buf[24:27], buf[27] = d / np.linalg.norm(d), cos
            buf[28:31] = np.asarray(env.sun_color) * env.sun_illuminance / (2 * np.pi * (1 - cos))
        else:
            buf[27] = 1.0
        buf[31] = self._env_total
        buf[32], buf[33], buf[34] = self.clamp, self.scene_radius, self.emissive_total
        ints[35] = 1 if self.diffuse_only else 0
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

    # ------------------------------------------------------------ path tracing

    def reset(self):
        self.samples = 0
        if hasattr(self, "_accum"):
            enc = self.device.create_command_encoder()
            for b in (self._accum, self._aov_albedo, self._aov_nd):
                enc.clear_buffer(b)
            self.device.queue.submit([enc.finish()])

    def step(self, n: int = 1) -> float:
        """Add n samples per pixel; returns the seconds it took, GPU work included."""
        pipeline, group = self._bind("path", [self._accum, self._aov_albedo, self._aov_nd] + self._light_buffers,
                                     [False] * 3 + [True] * 5)
        t = time.perf_counter()
        for _ in range(n):
            self._write_params(frame=self.samples)
            enc = self.device.create_command_encoder()
            self._dispatch(enc, pipeline, group)
            self.device.queue.submit([enc.finish()])
            self.samples += 1
        self.device.queue.read_buffer(self._accum, 0, 16)  # wait for the GPU
        return time.perf_counter() - t

    def radiance(self) -> np.ndarray:
        """Mean linear radiance per pixel, (h, w, 3) float32."""
        acc = self._read(self._accum)
        return acc[..., :3] / np.maximum(acc[..., 3:4], 1.0)

    def guides(self) -> dict:
        """Averaged first-hit albedo, normal and distance, and the second moment of luminance."""
        n = np.maximum(self._read(self._accum)[..., 3], 1.0)
        a = self._read(self._aov_albedo)
        nd = self._read(self._aov_nd)
        return {"albedo": a[..., :3] / n[..., None], "moment2": a[..., 3] / n,
                "normal": nd[..., :3] / n[..., None], "depth": nd[..., 3] / n, "samples": n}

    def denoised(self, passes: int = 5) -> np.ndarray:
        """Linear radiance after the à-trous filter, (h, w, 3) float32."""
        rad = self.radiance()
        g = self.guides()
        w = np.array([0.2126, 0.7152, 0.0722], np.float32)
        albedo = np.maximum(g["albedo"], 0.02)
        lum_a = np.maximum(albedo @ w, 0.02)
        var = np.maximum(g["moment2"] - (rad @ w) ** 2, 0.0) / g["samples"] / lum_a ** 2
        var = _blur3(var)
        src = np.concatenate([rad / albedo, var[..., None]], axis=2).astype(np.float32)
        normal = g["normal"] / np.maximum(np.linalg.norm(g["normal"], axis=2, keepdims=True), 1e-6)
        guide = np.concatenate([normal, g["depth"][..., None]], axis=2).astype(np.float32)

        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        a, b = self._buffer(src, usage), self._out(src.nbytes)
        gbuf = self._buffer(guide)
        flags = [True, False, True]
        groups = [self._bind("atrous", [a, b, gbuf], flags), self._bind("atrous", [b, a, gbuf], flags)]
        for k in range(passes):
            self._write_params(frame=1 << k)
            enc = self.device.create_command_encoder()
            self._dispatch(enc, *groups[k % 2])
            self.device.queue.submit([enc.finish()])
        out = self._read(b if passes % 2 else a)
        return out[..., :3] * albedo

    def image(self, denoise: bool = True, exposure="auto", tonemap: str = "aces") -> np.ndarray:
        """(h, w, 3) uint8 sRGB. exposure: a multiplier, or "auto" (log-average luminance to 0.18)."""
        rgb = self.denoised() if denoise else self.radiance()
        return to_srgb8(rgb, exposure, tonemap)

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


def _blur3(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, 1, mode="edge")
    k = (1.0, 2.0, 1.0)
    rows = sum(k[i] * p[i: i + x.shape[0], :] for i in range(3)) / 4
    return sum(k[j] * rows[:, j: j + x.shape[1]] for j in range(3)) / 4


def auto_exposure(rgb: np.ndarray) -> float:
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return float(0.18 / np.exp(np.mean(np.log(np.maximum(lum, 1e-4)))))


def to_srgb8(rgb: np.ndarray, exposure="auto", tonemap: str = "aces") -> np.ndarray:
    k = auto_exposure(rgb) if exposure == "auto" else float(exposure)
    x = np.maximum(rgb * k, 0.0)
    if tonemap == "aces":  # Narkowicz's fit of the ACES filmic curve
        x = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    elif tonemap == "reinhard":
        x = x / (1.0 + x)
    elif tonemap != "clamp":
        raise ValueError(f"tonemap must be aces, reinhard or clamp, got {tonemap!r}")
    x = np.clip(x, 0.0, 1.0)
    x = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.round(x * 255).astype(np.uint8)


def _default_camera(scene: Scene) -> Camera:
    lo, hi = scene.bounds()
    c = (lo + hi) / 2
    r = float(np.linalg.norm(hi - lo)) / 2
    return Camera(position=tuple(c + np.array([0.6, 0.5, 1.0]) * r * 1.6), target=tuple(c), fov=50.0)
