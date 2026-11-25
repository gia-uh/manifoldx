"""Import-time compiler for the restricted, per-entity kernel IR v1."""

import ast
import inspect
import math
import struct
import textwrap
from dataclasses import dataclass
from sys import _getframe

from manifoldx.compute import shader
from manifoldx.compute.transpile import (
    _BUILTIN_RETURNS,
    _TYPE_MAP,
    ComputeShaderCompileError,
)

_VERSION = 1
_TYPES = {1: _TYPE_MAP[float], 3: _TYPE_MAP[shader.vec3]}
_OPERATIONS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}
_COMPARISONS = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
_SYMBOLS = dict(
    zip(
        ("add", "sub", "mul", "eq", "ne", "lt", "le", "gt", "ge"),
        ("+", "-", "*", "==", "!=", "<", "<=", ">", ">="),
        strict=True,
    )
)


@dataclass(frozen=True, slots=True)
class _Expr:
    op: str
    width: int  # 0 = bool; 1 = f32; 3 = vec3<f32>
    span: tuple
    args: tuple


@dataclass(frozen=True, slots=True)
class _Statement:
    op: str
    span: tuple
    args: tuple


@dataclass(frozen=True, slots=True)
class _Kernel:
    version: int
    name: str
    component: type
    width: int
    layout: tuple
    uniforms: tuple
    reads: frozenset
    writes: frozenset
    filename: str
    body: tuple
    locals: int

    def wgsl(self):
        """Compile the validated IR to a WGSL compute shader."""
        access = "read_write" if self.writes else "read"
        lines = [
            f"@group(0) @binding(0) var<storage, {access}> rows: array<f32>;",
            "@group(0) @binding(1) var<storage, read> selected: array<u32>;",
            "struct Args { version: u32, count: u32, dt: f32, padding: u32 }",
            "@group(0) @binding(2) var<uniform> args: Args;",
            "@compute @workgroup_size(64)",
            "fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
            f"  if (args.version != {self.version}u || id.x >= args.count) {{ return; }}",
            "  if (selected[id.x] == 0u) { return; }",
            f"  let base = id.x * {self.width}u;",
        ]
        _wgsl_block(self.body, lines, "  ")
        return "\n".join([*lines, "}"])


class _Compiler:
    def __init__(self, function, namespace):
        self.function = function
        self.filename = function.__code__.co_filename
        self.namespace = namespace
        self.lines = []
        self.start = function.__code__.co_firstlineno
        self.reads = set()
        self.writes = set()
        self.locals = 0

    def span(self, node):
        def location(line, offset):
            original = self.lines[line - 1].rstrip("\n\r")
            parsed = self.source[line - 1]
            indentation = len(original) - len(parsed)
            column = len(parsed.encode()[:offset].decode()) + indentation + 1
            return (self.start + line - 1, column)

        return (
            *location(node.lineno, node.col_offset),
            *location(node.end_lineno, node.end_col_offset),
        )

    def fail(self, node, message, category="unsupported-construct"):
        line, col = self.span(node)[:2] if node is not None else (self.start, 1)
        raise ComputeShaderCompileError(
            category=category,
            message=message,
            filename=self.filename,
            line=line,
            col=col,
            source_line=self.lines[line - self.start].rstrip() if self.lines else None,
        )

    def resolve(self, name):
        return self.namespace.get(name, self.function.__globals__.get(name))

    def compile(self):
        from manifoldx.components import Component

        try:
            self.lines, self.start = inspect.getsourcelines(self.function)
            source = textwrap.dedent("".join(self.lines))
            self.source = source.splitlines()
            (fn,) = ast.parse(source).body
        except (OSError, TypeError, ValueError, SyntaxError) as error:
            self.fail(None, f"cannot parse kernel source: {error}")
        if not isinstance(fn, ast.FunctionDef) or len(fn.decorator_list) != 1:
            self.fail(fn, "expected a plain function with a single @kernel decorator")
        args = fn.args
        if (
            len(args.args) != 2
            or args.posonlyargs
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or args.defaults
        ):
            self.fail(fn, "kernel requires a component argument and dt: f32")
        component_arg, uniform = args.args
        annotation = component_arg.annotation
        component = self.resolve(annotation.id) if isinstance(annotation, ast.Name) else None
        if not isinstance(component, type) or not issubclass(component, Component):
            self.fail(component_arg, "expected a declared Component class", "type-error")
        self.component = component
        self.component_name = component_arg.arg
        if (
            uniform.arg != "dt"
            or not isinstance(uniform.annotation, ast.Name)
            or self.resolve(uniform.annotation.id) is not shader.f32
        ):
            self.fail(uniform, "uniform must be declared as dt: f32", "type-error")
        if not isinstance(fn.returns, ast.Constant) or fn.returns.value is not None:
            self.fail(fn, "kernel return annotation must be None", "type-error")
        if component._dtype.str[1:] != "f4" or not component._dtype.isnative or component._gpu_only:
            self.fail(component_arg, "expected a native float32 component layout", "type-error")
        width = component._shape[0]
        if not 0 < width <= 256:
            self.fail(component_arg, "kernel component width must be between 1 and 256")
        body = self.block(fn.body, {})
        return _Kernel(
            _VERSION,
            fn.name,
            component,
            width,
            tuple(component._layout.items()),
            (("dt", "f32"),),
            frozenset(self.reads),
            frozenset(self.writes),
            self.filename,
            body,
            self.locals,
        )

    def field(self, node, *, write=False):
        if not (isinstance(node.value, ast.Name) and node.value.id == self.component_name):
            self.fail(node, "field must belong to this invocation's component")
        layout = self.component._layout.get(node.attr)
        if layout is None or layout[1] not in _TYPES:
            self.fail(node, f"field {node.attr!r} must have type f32 or vec3", "type-error")
        offset, width = layout
        (self.writes if write else self.reads).add(node.attr)
        return _Expr("field", width, self.span(node), (self.component.__name__, node.attr, offset))

    def arithmetic(self, node, operator, left, right):
        op = _OPERATIONS.get(type(operator))
        if op is None:
            self.fail(node, "only addition, subtraction, and multiplication are supported")
        if not left.width or not right.width or (left.width != right.width and op != "mul"):
            self.fail(node, f"invalid operand types for {op}", "type-error")
        return _Expr(op, max(left.width, right.width), self.span(node), (left, right))

    def expr(self, node, locals_):
        span = self.span(node)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                (value,) = struct.unpack("<f", struct.pack("<f", node.value))
            except OverflowError, struct.error:
                self.fail(node, "literal must fit in f32", "type-error")
            if not math.isfinite(value):
                self.fail(node, "literal must be finite", "type-error")
            return _Expr("literal", 1, span, (value,))
        if isinstance(node, ast.Name):
            if node.id == "dt":
                return _Expr("uniform", 1, span, ("dt",))
            if node.id in locals_:
                index, width = locals_[node.id]
                return _Expr("local", width, span, (index,))
            self.fail(node, f"unknown local {node.id!r}")
        if isinstance(node, ast.Attribute):
            return self.field(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.expr(node.operand, locals_)
            if not value.width:
                self.fail(node, "negation requires a numeric operand", "type-error")
            return _Expr("neg", value.width, span, (value,))
        if isinstance(node, ast.BinOp):
            return self.arithmetic(
                node, node.op, self.expr(node.left, locals_), self.expr(node.right, locals_)
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id in locals_
                or self.resolve(node.func.id) is not shader.clamp
            ):
                self.fail(node, "only the clamp intrinsic can be called")
            if node.keywords or len(node.args) != _BUILTIN_RETURNS["clamp"][0]:
                self.fail(node, "clamp requires three positional operands", "type-error")
            operands = tuple(self.expr(arg, locals_) for arg in node.args)
            if any(operand.width != 1 for operand in operands):
                self.fail(node, "clamp requires f32 operands", "type-error")
            return _Expr("clamp", 1, span, operands)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.expr(node.left, locals_), self.expr(node.comparators[0], locals_)
            operator = _COMPARISONS.get(type(node.ops[0]))
            if operator is None or left.width != 1 or right.width != 1:
                self.fail(node, "comparison requires two f32 operands", "type-error")
            return _Expr("compare", 0, span, (operator, left, right))
        self.fail(node, f"unsupported expression: {type(node).__name__}")

    def target(self, node, locals_):
        if isinstance(node, ast.Attribute):
            return self.field(node, write=True)
        if isinstance(node, ast.Name) and node.id in locals_:
            return self.expr(node, locals_)
        self.fail(node, "assignment requires a declared local or a component field")

    def block(self, nodes, locals_):
        result = []
        for node in nodes:
            span = self.span(node)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if name in locals_ or name in (self.component_name, "dt") or node.value is None:
                    self.fail(node, "local declarations require a fresh name and a value")
                tag = (
                    self.resolve(node.annotation.id)
                    if isinstance(node.annotation, ast.Name)
                    else None
                )
                width = 1 if tag is shader.f32 else 3 if tag is shader.vec3 else None
                value = self.expr(node.value, locals_)
                if width != value.width:
                    self.fail(node, "local declaration requires the exact type", "type-error")
                target = _Expr("local", width, self.span(node.target), (self.locals,))
                locals_[name] = (self.locals, width)
                self.locals += 1
                result.append(_Statement("let", span, (target, value)))
            elif isinstance(node, ast.AugAssign):
                target = self.target(node.target, locals_)
                value = self.arithmetic(
                    node, node.op, self.expr(node.target, locals_), self.expr(node.value, locals_)
                )
                result.append(self.store(node, target, value))
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = self.target(node.targets[0], locals_)
                result.append(self.store(node, target, self.expr(node.value, locals_)))
            elif isinstance(node, ast.If):
                condition = self.expr(node.test, locals_)
                if condition.width != 0:
                    self.fail(node.test, "branch requires a comparison", "type-error")
                result.append(
                    _Statement(
                        "if",
                        span,
                        (
                            condition,
                            self.block(node.body, locals_.copy()),
                            self.block(node.orelse, locals_.copy()),
                        ),
                    )
                )
            else:
                self.fail(node, f"unsupported statement: {type(node).__name__}")
        return tuple(result)

    def store(self, node, target, value):
        if target.width != value.width:
            self.fail(node, "store requires the exact target type", "type-error")
        return _Statement("store", self.span(node), (target, value))


def _field_cells(expr):
    offset = expr.args[2]
    return [f"rows[base + {offset + i}u]" for i in range(expr.width)]


def _wgsl_expr(expr):
    op, args = expr.op, expr.args
    if op == "literal":
        (bits,) = struct.unpack("<I", struct.pack("<f", args[0]))
        return f"bitcast<f32>({bits}u)"
    if op == "uniform":
        return "args.dt"
    if op == "local":
        return f"v{args[0]}"
    if op == "field":
        cells = _field_cells(expr)
        return cells[0] if expr.width == 1 else f"vec3<f32>({', '.join(cells)})"
    if op == "neg":
        return f"(-{_wgsl_expr(args[0])})"
    if op == "clamp":
        return f"clamp({', '.join(map(_wgsl_expr, args))})"
    if op == "compare":
        op, *args = args
    left, right = map(_wgsl_expr, args)
    return f"({left} {_SYMBOLS[op]} {right})"


def _wgsl_block(statements, lines, indent):
    for statement in statements:
        if statement.op == "if":
            condition, yes, no = statement.args
            lines.append(f"{indent}if ({_wgsl_expr(condition)}) {{")
            _wgsl_block(yes, lines, indent + "  ")
            lines.append(f"{indent}}} else {{")
            _wgsl_block(no, lines, indent + "  ")
            lines.append(f"{indent}}}")
            continue
        target, value = statement.args
        expression = _wgsl_expr(value)
        if statement.op == "let":
            lines.append(
                f"{indent}var {_wgsl_expr(target)}: {_TYPES[target.width]} = {expression};"
            )
        elif target.op == "local":
            lines.append(f"{indent}{_wgsl_expr(target)} = {expression};")
        elif target.width == 1:
            lines.append(f"{indent}{_field_cells(target)[0]} = {expression};")
        else:
            lines.append(f"{indent}{{")
            lines.append(f"{indent}  let result = {expression};")
            for cell, coordinate in zip(_field_cells(target), "xyz", strict=True):
                lines.append(f"{indent}  {cell} = result.{coordinate};")
            lines.append(f"{indent}}}")


def kernel(function):
    """Validate a file-backed Python kernel at import and retain only its typed IR."""
    if not inspect.isfunction(function):
        raise TypeError("@kernel requires a Python function")
    return _Compiler(function, _getframe(1).f_locals).compile()


def _native_words(ir):
    words = []
    next_value = ir.locals + 1

    def emit(op, out=0, a=0, b=0, c=0, literal=0.0):
        words.append((op, out, a, b, c, literal))

    def address(expr):
        nonlocal next_value
        op, args = expr.op, expr.args
        if op == "uniform":
            return 0
        if op == "local":
            return args[0] + 1
        out = next_value
        next_value += 1
        if op == "literal":
            emit(op, out, literal=args[0])
        elif op == "field":
            emit("load", out, args[2], expr.width)
        elif op == "neg":
            emit(op, out, address(args[0]), expr.width)
        elif op == "clamp":
            emit(op, out, *map(address, args))
        elif op == "compare":
            emit(args[0], out, address(args[1]), address(args[2]))
        else:
            left, right = args
            vector_flags = int(left.width == 3) * 2 + int(right.width == 3)
            emit(op, out, address(left), address(right), vector_flags)
        return out

    def block(statements):
        for statement in statements:
            if statement.op == "if":
                condition, yes, no = statement.args
                value = address(condition)
                branch = len(words)
                emit("branch")
                block(yes)
                jump = len(words)
                emit("jump")
                words[branch] = ("branch", 0, value, len(words), 0, 0.0)
                block(no)
                words[jump] = ("jump", 0, len(words), 0, 0, 0.0)
                continue
            target, value = statement.args
            src = address(value)
            if target.op == "local":
                emit("copy", address(target), src)
            else:
                emit("store", src, target.args[2], target.width)

    block(ir.body)
    return words
