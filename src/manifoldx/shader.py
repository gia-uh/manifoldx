"""Restricted Python fragment shaders for the native GPU renderer.

Vec2 inputs receive the mesh's interpolated UV. Scalar bindings are finite f32
material uniforms. Texture1D bindings are immutable (width, 4) RGBA uint8
arrays. The GPU samples normalized coordinates with linear filtering and
clamp-to-edge. RGB texels decode from sRGB on sample, alpha stays linear, and
the fragment returns linear RGBA for the renderer's single output encode.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from numbers import Real

import numpy as np

from manifoldx._versions import _Counter
from manifoldx.compute.shader import clamp as clamp, f32 as f32, vec4 as Vec4


class Vec2:
    """Interpolated mesh texture coordinates."""


class Texture1D:
    """An immutable 1D RGBA sRGB texture declared from texels."""


def sample(texture: Texture1D, coordinate: f32) -> Vec4:
    """Sample a Texture1D at its normalized coordinate in a fragment."""
    raise TypeError("sample is only available inside a GPU fragment")


class ShaderCompileError(SyntaxError):
    """A source-positioned error in a restricted fragment declaration."""


@dataclass(frozen=True)
class _Fragment:
    source: str
    bindings: tuple[tuple[str, str], ...]
    requires_uv: bool

    def __call__(self, *args, **kwargs):
        raise TypeError("a compiled fragment runs only on the GPU")


@dataclass(frozen=True, init=False, eq=False)
class ShaderMaterial:
    """Attach a compiled fragment and its named material bindings to a mesh."""

    _fragment: _Fragment
    _uniforms: tuple[float, ...]
    _textures: tuple[bytes, ...]
    _version: _Counter

    def __init__(self, fragment, **bindings):
        if not isinstance(fragment, _Fragment):
            raise TypeError("ShaderMaterial requires a @shader.fragment declaration")
        expected = {name for name, _ in fragment.bindings}
        missing, extra = expected - bindings.keys(), bindings.keys() - expected
        if missing or extra:
            raise TypeError(f"fragment bindings: missing={sorted(missing)}, extra={sorted(extra)}")
        uniforms, textures = [], []
        for name, kind in fragment.bindings:
            value = bindings[name]
            if kind == "f32":
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError(f"fragment binding {name!r} requires an f32 scalar")
                if not -np.finfo("f4").max <= value <= np.finfo("f4").max:
                    raise ValueError(f"fragment binding {name!r} requires a finite f32 scalar")
                uniforms.append(float(value))
            else:
                array = np.asarray(value)
                if array.dtype != np.uint8:
                    raise TypeError(f"fragment binding {name!r} requires uint8 texels")
                if array.ndim != 2 or array.shape[1] != 4 or not 1 <= len(array) <= 8192:
                    raise ValueError(f"fragment binding {name!r} requires (width, 4) RGBA texels")
                textures.append(array.tobytes())
        object.__setattr__(self, "_fragment", fragment)
        object.__setattr__(self, "_uniforms", tuple(uniforms))
        object.__setattr__(self, "_textures", tuple(textures))
        object.__setattr__(self, "_version", _Counter())


class _Compiler:
    _ARITHMETIC = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def __init__(self, fn):
        lines, self.line = inspect.getsourcelines(fn)
        self.filename = inspect.getsourcefile(fn) or fn.__code__.co_filename
        self.lines = lines
        self.indent = len(lines[0]) - len(lines[0].lstrip())
        self.node = ast.parse(textwrap.dedent("".join(lines))).body[0]
        self.values = {}

    def fail(self, node, reason):
        offset = node.lineno - 1
        column = node.col_offset + self.indent + 1
        raise ShaderCompileError(
            reason, (self.filename, self.line + offset, column, self.lines[offset])
        )

    def expression(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                number = float(node.value)
            except OverflowError:
                self.fail(node, "fragment literal exceeds f32")
            if not -np.finfo("f4").max <= number <= np.finfo("f4").max:
                self.fail(node, "fragment literal exceeds f32")
            return repr(number), "f32"
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.values[node.id]
        if isinstance(node, ast.Attribute):
            value, kind = self.expression(node.value)
            if kind == "Vec2" and node.attr in ("x", "y"):
                return f"{value}.{node.attr}", "f32"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value, kind = self.expression(node.operand)
            if kind == "f32":
                return f"(-{value})", kind
        if isinstance(node, ast.BinOp) and type(node.op) in self._ARITHMETIC:
            left, left_type = self.expression(node.left)
            right, right_type = self.expression(node.right)
            if left_type == right_type == "f32":
                return f"({left} {self._ARITHMETIC[type(node.op)]} {right})", "f32"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name not in ("clamp", "sample", "Vec4", "vec4", "f32"):
                self.fail(node, f"unsupported fragment call {name!r}")
            args = [self.expression(arg) for arg in node.args]
            values, types = zip(*args) if args else ((), ())
            if name == "sample" and types == ("Texture1D", "f32"):
                texture, coord = values
                return f"textureSampleLevel({texture}, {texture}_sampler, {coord}, 0.0)", "Vec4"
            if name == "clamp" and types == ("f32",) * 3:
                return f"clamp({', '.join(values)})", "f32"
            if name in ("Vec4", "vec4") and types == ("f32",) * 4:
                return f"vec4<f32>({', '.join(values)})", "Vec4"
            if name == "f32" and types == ("f32",):
                return values[0], "f32"
            self.fail(node, f"invalid arguments to fragment builtin {name!r}")
        self.fail(node, f"unsupported fragment expression {type(node).__name__}")

    def compile(self):
        fn = self.node
        if not isinstance(fn, ast.FunctionDef):
            self.fail(fn, "a fragment must be a synchronous function")
        args = fn.args
        if args.defaults or args.kwonlyargs or args.vararg or args.kwarg:
            self.fail(fn, "fragment inputs require positional annotations without defaults")
        if not isinstance(fn.returns, ast.Name) or fn.returns.id != "Vec4":
            self.fail(fn, "a fragment must return Vec4")
        bindings, uniforms, textures = [], [], []
        uv_used = False
        for arg in [*args.posonlyargs, *args.args]:
            kind = arg.annotation.id if isinstance(arg.annotation, ast.Name) else None
            if kind == "Vec2" and not uv_used:
                uv_used = True
                value = "in.uv"
            elif kind == "f32":
                value = f"fragment_uniforms.v{len(uniforms)}"
                uniforms.append(f"    v{len(uniforms)}: f32,")
            elif kind == "Texture1D":
                value = f"fragment_texture_{len(textures)}"
                textures.append(value)
            else:
                self.fail(arg, "use one Vec2 mesh input and f32 or Texture1D material bindings")
            self.values[arg.arg] = (value, kind)
            if kind != "Vec2":
                bindings.append((arg.arg, kind))

        declarations = []
        if uniforms:
            declarations += [
                "struct FragmentUniforms {",
                *uniforms,
                "};",
                "@group(1) @binding(2) var<uniform> fragment_uniforms: FragmentUniforms;",
            ]
        for i, name in enumerate(textures):
            declarations += [
                f"@group(1) @binding({3 + i * 2}) var {name}_sampler: sampler;",
                f"@group(1) @binding({4 + i * 2}) var {name}: texture_1d<f32>;",
            ]
        code = ["@fragment fn fs_custom(in: VertexOut) -> @location(0) vec4<f32> {"]
        for i, node in enumerate(fn.body):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name = node.targets[0].id
                if name in self.values:
                    self.fail(node, f"fragment local {name!r} is already defined")
                value, kind = self.expression(node.value)
                if kind == "Texture1D":
                    self.fail(node, "sample textures directly from their declared binding")
                local = f"fragment_local_{i}"
                self.values[name] = (local, kind)
                code.append(f"    let {local} = {value};")
            elif isinstance(node, ast.Return) and i == len(fn.body) - 1:
                value, kind = self.expression(node.value)
                if kind != "Vec4":
                    self.fail(node, "a fragment must return a Vec4")
                code.append(f"    return {value};")
            else:
                self.fail(node, f"unsupported fragment statement {type(node).__name__}")
        if not isinstance(fn.body[-1], ast.Return):
            self.fail(fn, "a fragment must end with return")
        return _Fragment("\n".join([*declarations, *code, "}"]), tuple(bindings), uv_used)


def fragment(fn):
    """Validate and lower a restricted Python fragment when it is declared."""
    try:
        return _Compiler(fn).compile()
    except OSError as error:
        raise ShaderCompileError("fragment source must be available in a Python file") from error
