"""Compute systems — first-class GPU work as ECS extension.

Re-exports the Phase-1 surface from the underlying core module + the
Phase-2 shader-DSL module. See:
- `.knowledge/analysis/2026-05-06-compute-systems-design.md` (Phase 1).
- The Phase-2 design lives inline in this module's docstrings.
"""
from manifoldx.compute._core import (
    Compute,
    ComputeRunner,
    Reads,
    ReadsWrites,
    Uniform,
    Writes,
    _AUTO_BOUND_UNIFORMS,
    _DISPATCH_SYMBOLS,
)
from manifoldx.compute import shader
from manifoldx.compute._resident import GpuKernelGroup
from manifoldx.compute.transpile import (
    ComputeShaderCompileError,
    transpile_compute,
)

__all__ = [
    "Compute",
    "GpuKernelGroup",
    "ComputeRunner",
    "Reads",
    "Writes",
    "ReadsWrites",
    "Uniform",
    "shader",
    "ComputeShaderCompileError",
    "transpile_compute",
]
