"""A progressive path tracer for manifoldx scenes, as WGSL compute kernels over a numpy-built BVH.

    from manifoldx.trace import Scene, Tracer, Environment
    scene = Scene.from_glb("scene.glb")
    tracer = Tracer(scene, 1024, 640, environment=Environment.sky((0.5, 0.6, 0.4)))
    tracer.step(64)
    rgb = tracer.image()  # (640, 1024, 3) uint8, sRGB, denoised

No hardware ray tracing: wgpu-py exposes no ray queries, so the BVH walk is in the shader.
"""

from manifoldx.trace.environment import Environment
from manifoldx.trace.scene import Camera, Light, Material, Scene
from manifoldx.trace.tracer import Tracer

__all__ = ["Camera", "Environment", "Light", "Material", "Scene", "Tracer"]
