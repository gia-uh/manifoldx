"""Phase-2 Python → WGSL transpiler for Compute kernels.

See `.knowledge/analysis/2026-05-08-compute-phase-2-shader-compiler-design.md`.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Dict, List

from manifoldx.compute import shader as _shader


class ComputeShaderCompileError(Exception):
    """Raised when a Compute kernel cannot be transpiled to valid WGSL.

    Carries structured fields for the IDE/REPL to surface clearly:
    file, line, column, error category, the offending source line, and
    a human-readable message.
    """

    def __init__(
        self,
        *,
        category: str,
        message: str,
        filename: str,
        line: int,
        col: int,
        source_line: str | None = None,
    ):
        self.category = category
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        self.source_line = source_line
        super().__init__(self._render())

    def _render(self) -> str:
        head = f"{self.filename}:{self.line}:{self.col}: {self.category}: {self.message}"
        if self.source_line is None:
            return head
        caret = " " * self.col + "^"
        return f"{head}\n  {self.source_line}\n  {caret}"


def _collect_method_asts(cls: type) -> Dict[str, ast.FunctionDef]:
    """Parse every method defined on `cls` into a {name: FunctionDef} dict.

    Methods inherited from object/Compute are ignored. Decorators are
    rejected — Phase-2 supports only plain methods.
    """
    methods: Dict[str, ast.FunctionDef] = {}
    for name, member in cls.__dict__.items():
        # Dunders are not kernel methods; Python 3.14 also adds a generated
        # __annotate_func__ to every class with annotations.
        if not inspect.isfunction(member) or (name.startswith("__") and name.endswith("__")):
            continue
        try:
            src = textwrap.dedent(inspect.getsource(member))
        except OSError as e:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"cannot read source for method {name!r}: {e}",
                filename=getattr(member, "__code__").co_filename,
                line=0, col=0, source_line=None,
            )
        module = ast.parse(src)
        fn = module.body[0]
        if not isinstance(fn, ast.FunctionDef):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"{name!r} is not a plain function (lambdas / async / decorators rejected)",
                filename=member.__code__.co_filename,
                line=getattr(fn, "lineno", 0),
                col=getattr(fn, "col_offset", 0),
                source_line=None,
            )
        if fn.decorator_list:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"decorators are not supported on Compute methods (found on {name!r})",
                filename=member.__code__.co_filename,
                line=fn.lineno, col=fn.col_offset, source_line=None,
            )
        methods[name] = fn

    if "main" not in methods:
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"Compute class {cls.__name__!r} must define a `main(self, i: int)` method",
            filename=getattr(cls, "__module__", "<class>"),
            line=0, col=0, source_line=None,
        )

    main = methods["main"]
    arg_names = [a.arg for a in main.args.args]
    if arg_names != ["self", "i"]:
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"`main` must have signature (self, i: int); got {arg_names!r}",
            filename="<class>", line=main.lineno, col=main.col_offset, source_line=None,
        )
    return methods


def _method_calls_in(fn: ast.FunctionDef) -> List[str]:
    """Return the names of sibling-method calls (`self.<name>(...)`) in fn."""
    names: List[str] = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            names.append(node.func.attr)
    return names


def _check_no_recursion(methods: Dict[str, ast.FunctionDef]) -> None:
    """DFS the call graph; raise ComputeShaderCompileError if any cycle is found."""
    graph = {name: _method_calls_in(fn) for name, fn in methods.items()}

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}

    def dfs(node: str, fn: ast.FunctionDef) -> None:
        color[node] = GRAY
        for callee in graph.get(node, []):
            if callee not in graph:
                continue  # ignored — bound binding/uniform attrs aren't methods
            if color[callee] == GRAY:
                raise ComputeShaderCompileError(
                    category="recursion",
                    message=f"recursion detected: {node!r} (in)directly calls itself via {callee!r}",
                    filename="<class>", line=fn.lineno, col=fn.col_offset, source_line=None,
                )
            if color[callee] == WHITE:
                dfs(callee, methods[callee])
        color[node] = BLACK

    for name, fn in methods.items():
        if color[name] == WHITE:
            dfs(name, fn)


_TYPE_MAP = {
    int:           "i32",
    float:         "f32",
    bool:          "bool",
    _shader.vec3:  "vec3<f32>",
    _shader.vec4:  "vec4<f32>",
}


def _python_type_to_wgsl(tp) -> str:
    """Map a Python annotation tag to a WGSL type string."""
    if tp in _TYPE_MAP:
        return _TYPE_MAP[tp]
    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported annotation type: {getattr(tp, '__name__', tp)!r}; "
                f"use int, float, bool, vec3, or vec4",
        filename="<annotation>", line=0, col=0, source_line=None,
    )


class TypeEnv:
    """Lookup table for the WGSL type of each name in scope."""

    def __init__(self) -> None:
        self._params: Dict[str, str] = {}
        self._locals: Dict[str, str] = {}
        self._uniforms: Dict[str, str] = {}
        self._bindings: Dict[str, str] = {}  # binding name → Component class name
        self._method_sigs: Dict[str, Dict[str, str]] = {}
        self._class_name: str = ""

    def set_param(self, name: str, wgsl: str) -> None:
        self._params[name] = wgsl

    def set_local(self, name: str, wgsl: str) -> None:
        self._locals[name] = wgsl

    def set_uniform(self, name: str, wgsl: str) -> None:
        self._uniforms[name] = wgsl

    def set_binding(self, name: str, component_name: str) -> None:
        self._bindings[name] = component_name

    def lookup(self, name: str) -> str:
        if name in self._locals:
            return self._locals[name]
        if name in self._params:
            return self._params[name]
        raise ComputeShaderCompileError(
            category="unknown-name",
            message=f"name {name!r} not in scope",
            filename="<expr>", line=0, col=0, source_line=None,
        )

    def lookup_uniform(self, name: str) -> str:
        if name not in self._uniforms:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"uniform {name!r} not declared on Compute class",
                filename="<expr>", line=0, col=0, source_line=None,
            )
        return self._uniforms[name]

    def lookup_binding(self, name: str) -> str:
        if name not in self._bindings:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"binding {name!r} not declared on Compute class",
                filename="<expr>", line=0, col=0, source_line=None,
            )
        return self._bindings[name]

    def set_method_sigs(self, sigs: Dict[str, Dict[str, str]], *, class_name: str) -> None:
        self._method_sigs = sigs
        self._class_name = class_name

    def lookup_method(self, name: str) -> Dict[str, str]:
        if name not in self._method_sigs:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"method {name!r} not declared on Compute class {self._class_name!r}",
                filename="<expr>", line=0, col=0, source_line=None,
            )
        return self._method_sigs[name]

    @property
    def class_name(self) -> str:
        return self._class_name


def _resolve_annotation(node, cls: type):
    """Look up a Name/Attribute annotation in the class's defining module.

    Fallback path used when only the AST is available; prefer reading the
    live `func.__annotations__` dict instead, since Python has already
    resolved names against the right namespace at function-def time.
    """
    import sys
    mod = sys.modules.get(cls.__module__)
    src = ast.unparse(node)
    builtins = {"int": int, "float": float, "bool": bool}
    if src in builtins:
        return builtins[src]
    if mod is not None and hasattr(mod, src):
        return getattr(mod, src)
    raise ComputeShaderCompileError(
        category="unknown-name",
        message=f"could not resolve annotation {src!r}; not a builtin or module-level name",
        filename="<class>", line=getattr(node, "lineno", 0),
        col=getattr(node, "col_offset", 0), source_line=None,
    )


def _build_method_signatures(cls: type) -> Dict[str, Dict[str, str]]:
    """For each method on `cls`, derive {param_name → wgsl_type} + 'return'.

    Reads the live `func.__annotations__` dict so closure-defined classes
    work without a module-globals lookup.
    """
    methods = _collect_method_asts(cls)
    sigs: Dict[str, Dict[str, str]] = {}
    for name, fn in methods.items():
        member = cls.__dict__[name]
        live_annotations = getattr(member, "__annotations__", {})
        sig: Dict[str, str] = {}
        for arg in fn.args.args:
            if arg.arg == "self":
                continue
            if arg.arg not in live_annotations and arg.annotation is None:
                raise ComputeShaderCompileError(
                    category="missing-annotation",
                    message=f"parameter {arg.arg!r} of method {name!r} requires a type annotation",
                    filename="<class>", line=arg.lineno, col=arg.col_offset,
                    source_line=None,
                )
            tp = (
                live_annotations[arg.arg]
                if arg.arg in live_annotations
                else _resolve_annotation(arg.annotation, cls)
            )
            sig[arg.arg] = _python_type_to_wgsl(tp)
        if name != "main":
            if "return" not in live_annotations and fn.returns is None:
                raise ComputeShaderCompileError(
                    category="missing-annotation",
                    message=f"method {name!r} requires a return-type annotation",
                    filename="<class>", line=fn.lineno, col=fn.col_offset,
                    source_line=None,
                )
            ret_tp = (
                live_annotations["return"]
                if "return" in live_annotations
                else _resolve_annotation(fn.returns, cls)
            )
            sig["return"] = _python_type_to_wgsl(ret_tp)
        sigs[name] = sig
    return sigs


# Builtin signature table. Each entry: name -> (arg_count or None, return_type).
# return_type may be a string (concrete WGSL type) or a callable
# `arg_types -> wgsl_type` for polymorphic builtins.
_BUILTIN_RETURNS = {
    "vec3":      (3, "vec3<f32>"),
    "vec4":      (4, "vec4<f32>"),
    "length":    (1, "f32"),
    "dot":       (2, "f32"),
    "cross":     (2, "vec3<f32>"),
    "normalize": (1, lambda ts: ts[0]),
    "sqrt":      (1, "f32"),
    "pow":       (2, "f32"),
    "floor":     (1, lambda ts: ts[0]),
    "ceil":      (1, lambda ts: ts[0]),
    "abs":       (1, lambda ts: ts[0]),
    "min":       (2, lambda ts: ts[0]),
    "max":       (2, lambda ts: ts[0]),
    "clamp":     (3, lambda ts: ts[0]),
}

_CASTS = {"f32", "i32", "u32", "bool"}


def _emit_expr(node, env: TypeEnv, src: str) -> tuple[str, str]:
    """Emit (wgsl_text, wgsl_type) for an expression AST node."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return ("true" if v else "false"), "bool"
        if isinstance(v, int):
            return str(v), "i32"
        if isinstance(v, float):
            return repr(v), "f32"
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported literal {v!r}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    if isinstance(node, ast.Name):
        return node.id, env.lookup(node.id)

    if isinstance(node, ast.Attribute):
        # self.<binding>[idx].<field> — indexed component field access.
        if (
            isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Attribute)
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id == "self"
        ):
            binding_name = node.value.value.attr
            component_name = env.lookup_binding(binding_name)
            from manifoldx.components import _COMPONENT_CLASSES, Material, Mesh, Transform
            cls_lookup = {
                "Transform": Transform, "Mesh": Mesh, "Material": Material,
                **_COMPONENT_CLASSES,
            }
            comp_cls = cls_lookup[component_name]
            layout = comp_cls._layout
            if node.attr not in layout:
                raise ComputeShaderCompileError(
                    category="unknown-name",
                    message=f"field {node.attr!r} not declared on component {component_name!r}",
                    filename="<expr>", line=node.lineno, col=node.col_offset,
                    source_line=None,
                )
            offset, length = layout[node.attr]
            stride = sum(L for _, L in layout.values())
            idx_text, _ = _emit_expr(node.value.slice, env, src)
            if length == 1:
                wgsl = f"{binding_name}[{idx_text} * {stride}u + {offset}u]"
                return wgsl, "f32"
            if length == 3:
                parts = ", ".join(
                    f"{binding_name}[{idx_text} * {stride}u + {offset + k}u]"
                    for k in range(3)
                )
                return f"vec3<f32>({parts})", "vec3<f32>"
            if length == 4:
                parts = ", ".join(
                    f"{binding_name}[{idx_text} * {stride}u + {offset + k}u]"
                    for k in range(4)
                )
                return f"vec4<f32>({parts})", "vec4<f32>"
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"field {node.attr!r} has length {length}; only 1/3/4 supported",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )

        # self.<uniform> — fall through to uniform branch.
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"uniforms.{node.attr}", env.lookup_uniform(node.attr)

        # Swizzle read on a vec3/vec4-typed expression: `accel.x`,
        # `next_pos.y`, `self.transforms[i].pos.z`, etc. Returns f32.
        if node.attr in {"x", "y", "z", "w"}:
            inner_text, inner_type = _emit_expr(node.value, env, src)
            if inner_type == "vec3<f32>":
                if node.attr == "w":
                    raise ComputeShaderCompileError(
                        category="unsupported-construct",
                        message="cannot access .w on vec3<f32>; use vec4 if a 4th component is needed",
                        filename="<expr>", line=node.lineno, col=node.col_offset,
                        source_line=None,
                    )
                return f"{inner_text}.{node.attr}", "f32"
            if inner_type == "vec4<f32>":
                return f"{inner_text}.{node.attr}", "f32"
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"swizzle .{node.attr} requires a vec3/vec4 expression; got {inner_type}",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )

        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported attribute access: {ast.unparse(node)!r}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    if isinstance(node, ast.Call):
        # self.<method>(args) → free WGSL function `_<ClassName>_<method>`.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            method = node.func.attr
            sig = env.lookup_method(method)
            arg_emits = [_emit_expr(a, env, src) for a in node.args]
            arg_texts = [t for t, _ in arg_emits]
            return f"_{env.class_name}_{method}({', '.join(arg_texts)})", sig["return"]

        if isinstance(node.func, ast.Name):
            fname = node.func.id
            arg_emits = [_emit_expr(a, env, src) for a in node.args]
            arg_texts = [t for t, _ in arg_emits]
            arg_types = [ty for _, ty in arg_emits]

            if fname in _CASTS:
                if len(node.args) != 1:
                    raise ComputeShaderCompileError(
                        category="unsupported-construct",
                        message=f"{fname}() cast takes exactly one argument",
                        filename="<expr>", line=node.lineno, col=node.col_offset,
                        source_line=None,
                    )
                return f"{fname}({arg_texts[0]})", fname

            if fname in _BUILTIN_RETURNS:
                expected_arity, ret = _BUILTIN_RETURNS[fname]
                if expected_arity is not None and len(node.args) != expected_arity:
                    raise ComputeShaderCompileError(
                        category="unsupported-construct",
                        message=f"{fname} expects {expected_arity} args, got {len(node.args)}",
                        filename="<expr>", line=node.lineno, col=node.col_offset,
                        source_line=None,
                    )
                ret_type = ret(arg_types) if callable(ret) else ret
                emit_name = ret_type if fname in {"vec3", "vec4"} else fname
                return f"{emit_name}({', '.join(arg_texts)})", ret_type

            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"unknown function {fname!r}; not a recognized builtin or cast",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )

        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported call form: {ast.unparse(node)!r}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="chained comparisons not supported; use explicit and/or",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        l_text, _ = _emit_expr(node.left, env, src)
        r_text, _ = _emit_expr(node.comparators[0], env, src)
        op_str = {
            ast.Eq: "==", ast.NotEq: "!=",
            ast.Lt: "<",  ast.LtE:   "<=",
            ast.Gt: ">",  ast.GtE:   ">=",
        }.get(type(node.ops[0]))
        if op_str is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"unsupported comparison {type(node.ops[0]).__name__}",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        return f"({l_text} {op_str} {r_text})", "bool"

    if isinstance(node, ast.BoolOp):
        op_str = "&&" if isinstance(node.op, ast.And) else "||"
        emits = [_emit_expr(v, env, src)[0] for v in node.values]
        joined = f" {op_str} ".join(emits)
        return f"({joined})", "bool"

    if isinstance(node, ast.UnaryOp):
        v_text, v_type = _emit_expr(node.operand, env, src)
        if isinstance(node.op, ast.USub):
            return f"(-{v_text})", v_type
        if isinstance(node.op, ast.Not):
            return f"(!{v_text})", "bool"
        raise ComputeShaderCompileError(
            category="unsupported-construct",
            message=f"unsupported unary op {type(node.op).__name__}",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    if isinstance(node, ast.BinOp):
        l_text, l_type = _emit_expr(node.left, env, src)
        r_text, r_type = _emit_expr(node.right, env, src)
        op = node.op

        if isinstance(op, ast.Pow):
            return f"pow({l_text}, {r_text})", l_type

        op_str = {
            ast.Add:  "+",
            ast.Sub:  "-",
            ast.Mult: "*",
            ast.Div:  "/",
            ast.Mod:  "%",
        }.get(type(op))
        if op_str is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"unsupported binary op {type(op).__name__}",
                filename="<expr>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )

        # Strict promotion. Same-type → fine.
        if l_type == r_type:
            return f"({l_text} {op_str} {r_text})", l_type

        # vec * scalar / scalar * vec native broadcast (WGSL).
        VEC = {"vec3<f32>", "vec4<f32>"}
        if l_type in VEC and r_type == "f32":
            return f"({l_text} {op_str} {r_text})", l_type
        if r_type in VEC and l_type == "f32":
            return f"({l_text} {op_str} {r_text})", r_type

        raise ComputeShaderCompileError(
            category="implicit-promotion",
            message=f"mixed types in binary op: {l_type} {op_str} {r_type}; "
                    f"insert an explicit cast (e.g. f32(...))",
            filename="<expr>", line=node.lineno, col=node.col_offset,
            source_line=None,
        )

    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported expression: {ast.unparse(node)!r}",
        filename="<expr>", line=node.lineno, col=node.col_offset,
        source_line=None,
    )


def _scan_mutability(body: List[ast.stmt]) -> Dict[str, int]:
    """Count assignments per local name. >1 OR any AugAssign → 'var' later."""
    counts: Dict[str, int] = {}
    aug_targets: set[str] = set()

    def visit(stmts):
        for s in stmts:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                counts[s.target.id] = counts.get(s.target.id, 0) + 1
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        counts[t.id] = counts.get(t.id, 0) + 1
            elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
                aug_targets.add(s.target.id)
                counts[s.target.id] = counts.get(s.target.id, 0) + 1
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.stmt):
                    visit([child])
                elif hasattr(child, "body") and isinstance(getattr(child, "body"), list):
                    visit(child.body)

    visit(body)
    return {**counts, **{k: 999 for k in aug_targets}}


def _resolve_annotation_simple(node) -> type:
    """Annotation→type for a Name annotation. Vec types resolve via shader module."""
    if isinstance(node, ast.Name):
        builtins = {"int": int, "float": float, "bool": bool}
        if node.id in builtins:
            return builtins[node.id]
        if node.id in {"vec3", "Vector3"}:
            return _shader.vec3
        if node.id in {"vec4", "Vector4"}:
            return _shader.vec4
    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported annotation: {ast.unparse(node)!r}",
        filename="<stmt>", line=getattr(node, "lineno", 0),
        col=getattr(node, "col_offset", 0), source_line=None,
    )


def _emit_assign(target, value, *, op: str | None, env: TypeEnv, src: str,
                 line: int, col: int) -> str:
    """Emit a single assignment or augmented-assignment statement."""
    rhs_text, rhs_type = _emit_expr(value, env, src)

    # Storage-buffer field LHS: self.<binding>[idx].<field>.
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Subscript)
        and isinstance(target.value.value, ast.Attribute)
        and isinstance(target.value.value.value, ast.Name)
        and target.value.value.value.id == "self"
    ):
        binding_name = target.value.value.attr
        component_name = env.lookup_binding(binding_name)
        from manifoldx.components import _COMPONENT_CLASSES, Material, Mesh, Transform
        cls_lookup = {
            "Transform": Transform, "Mesh": Mesh, "Material": Material,
            **_COMPONENT_CLASSES,
        }
        comp_cls = cls_lookup[component_name]
        layout = comp_cls._layout
        if target.attr not in layout:
            raise ComputeShaderCompileError(
                category="unknown-name",
                message=f"field {target.attr!r} not declared on component {component_name!r}",
                filename="<stmt>", line=line, col=col, source_line=None,
            )
        offset, length = layout[target.attr]
        stride = sum(L for _, L in layout.values())
        idx_text, _ = _emit_expr(target.value.slice, env, src)

        def slot(k): return f"{binding_name}[{idx_text} * {stride}u + {offset + k}u]"
        components = ["x", "y", "z", "w"]

        lines = []
        for k in range(length):
            lhs = slot(k)
            rhs_part = rhs_text if length == 1 else f"{rhs_text}.{components[k]}"
            if op is None:
                lines.append(f"{lhs} = {rhs_part};")
            else:
                lines.append(f"{lhs} = ({lhs} {op} {rhs_part});")
        return "\n".join(lines)

    # Plain name target (reassignment).
    if isinstance(target, ast.Name):
        env.lookup(target.id)
        if op is None:
            return f"{target.id} = {rhs_text};"
        return f"{target.id} = ({target.id} {op} {rhs_text});"

    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported assignment target: {ast.unparse(target)!r}",
        filename="<stmt>", line=line, col=col, source_line=None,
    )


def _emit_block(stmts: List[ast.stmt], env: TypeEnv, mut: Dict[str, int], src: str) -> str:
    """Emit a sequence of statements, indented two spaces each."""
    lines = []
    for s in stmts:
        text = _emit_stmt(s, env, mut, src)
        for ln in text.splitlines():
            lines.append(f"  {ln}")
    return "\n".join(lines)


def _emit_stmt(node, env: TypeEnv, mut: Dict[str, int], src: str) -> str:
    """Emit a WGSL statement string for a single AST stmt."""
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="annotated assignment target must be a plain name",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        target_name = node.target.id
        wgsl_type = _python_type_to_wgsl(_resolve_annotation_simple(node.annotation))
        if node.value is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="annotated declarations require an initializer",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        rhs_text, _ = _emit_expr(node.value, env, src)
        env.set_local(target_name, wgsl_type)
        keyword = "var" if mut.get(target_name, 0) > 1 else "let"
        return f"{keyword} {target_name}: {wgsl_type} = {rhs_text};"

    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="multi-target assignment not supported",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        return _emit_assign(node.targets[0], node.value, op=None, env=env, src=src,
                             line=node.lineno, col=node.col_offset)

    if isinstance(node, ast.AugAssign):
        op_str = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
        }.get(type(node.op))
        if op_str is None:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"unsupported augmented op {type(node.op).__name__}",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        return _emit_assign(node.target, node.value, op=op_str, env=env, src=src,
                             line=node.lineno, col=node.col_offset)

    if isinstance(node, ast.If):
        cond_text, _ = _emit_expr(node.test, env, src)
        body_text = _emit_block(node.body, env, mut, src)
        out = f"if ({cond_text}) {{\n{body_text}\n}}"
        if node.orelse:
            else_text = _emit_block(node.orelse, env, mut, src)
            out += f" else {{\n{else_text}\n}}"
        return out

    if isinstance(node, ast.While):
        cond_text, _ = _emit_expr(node.test, env, src)
        body_text = _emit_block(node.body, env, mut, src)
        return f"while ({cond_text}) {{\n{body_text}\n}}"

    if isinstance(node, ast.For):
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="for-loops must iterate over range(...) only",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        if not isinstance(node.target, ast.Name):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="for-loop target must be a plain name",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        args = node.iter.args
        if len(args) == 1:
            start_text, stop_text = "0u", _emit_expr(args[0], env, src)[0]
        elif len(args) == 2:
            start_text = _emit_expr(args[0], env, src)[0]
            stop_text  = _emit_expr(args[1], env, src)[0]
        else:
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message="range(start, stop, step) — step not supported in v1",
                filename="<stmt>", line=node.lineno, col=node.col_offset,
                source_line=None,
            )
        env.set_local(node.target.id, "u32")
        body_text = _emit_block(node.body, env, mut, src)
        return (
            f"for (var {node.target.id}: u32 = {start_text}; "
            f"{node.target.id} < {stop_text}; "
            f"{node.target.id} = {node.target.id} + 1u) {{\n"
            f"{body_text}\n}}"
        )

    if isinstance(node, ast.Return):
        if node.value is not None:
            v_text, _ = _emit_expr(node.value, env, src)
            return f"return {v_text};"
        return "return;"

    if isinstance(node, ast.Continue):
        return "continue;"

    if isinstance(node, ast.Break):
        return "break;"

    raise ComputeShaderCompileError(
        category="unsupported-construct",
        message=f"unsupported statement: {ast.unparse(node)!r}",
        filename="<stmt>", line=node.lineno, col=node.col_offset,
        source_line=None,
    )


def _resolve_class_bindings(cls: type) -> Dict[str, str]:
    """{binding_name: ComponentClassName} from cls._reads + cls._writes."""
    out: Dict[str, str] = {}
    for name, comp in {**cls._reads, **cls._writes}.items():
        if not isinstance(comp, type):
            raise ComputeShaderCompileError(
                category="unsupported-construct",
                message=f"binding {name!r} parameter {comp!r} is not a class",
                filename="<class>", line=0, col=0, source_line=None,
            )
        out[name] = comp.__name__
    return out


def _resolve_class_uniforms(cls: type) -> Dict[str, str]:
    """{uniform_name: wgsl_type} from cls._uniforms.

    v1: every Uniform[T] emits as f32 (matches Phase-1 packing exactly).
    """
    out: Dict[str, str] = {}
    for name, _param in cls._uniforms.items():
        out[name] = "f32"
    return out


def _emit_header(
    uniforms: Dict[str, str],
    reads: List[str],
    writes: List[str],
) -> str:
    lines: List[str] = []
    slot = 0
    if uniforms:
        struct_fields = ", ".join(f"{n}: {t}" for n, t in uniforms.items())
        lines.append(f"struct Uniforms {{ {struct_fields}, }};")
        lines.append(f"@group(0) @binding({slot}) var<uniform> uniforms: Uniforms;")
        slot += 1
    for name in reads:
        lines.append(
            f"@group(0) @binding({slot}) var<storage, read> {name}: array<f32>;"
        )
        slot += 1
    for name in writes:
        lines.append(
            f"@group(0) @binding({slot}) var<storage, read_write> {name}: array<f32>;"
        )
        slot += 1
    return "\n".join(lines)


def _build_env(cls: type, sigs, bindings, uniforms, *, fn_params) -> TypeEnv:
    env = TypeEnv()
    env.set_method_sigs(sigs, class_name=cls.__name__)
    for n, t in uniforms.items():
        env.set_uniform(n, t)
    for n, comp_name in bindings.items():
        env.set_binding(n, component_name=comp_name)
    for n, t in fn_params.items():
        env.set_param(n, t)
    return env


def _emit_helper(cls, name, fn, sigs, bindings, uniforms) -> str:
    sig = sigs[name]
    params = {k: v for k, v in sig.items() if k != "return"}
    env = _build_env(cls, sigs, bindings, uniforms, fn_params=params)

    mut = _scan_mutability(fn.body)
    body_lines = []
    for s in fn.body:
        body_lines.append(_emit_stmt(s, env, mut, ast.unparse(s)))
    body = "\n".join(f"  {ln}" for chunk in body_lines for ln in chunk.splitlines())

    param_list = ", ".join(f"{k}: {v}" for k, v in params.items())
    return (
        f"fn _{cls.__name__}_{name}({param_list}) -> {sig['return']} {{\n"
        f"{body}\n}}"
    )


def _emit_main(cls, fn, sigs, bindings, uniforms) -> str:
    env = _build_env(cls, sigs, bindings, uniforms, fn_params={"i": "u32"})
    mut = _scan_mutability(fn.body)
    body_lines = []
    for s in fn.body:
        body_lines.append(_emit_stmt(s, env, mut, ast.unparse(s)))
    body = "\n".join(f"  {ln}" for chunk in body_lines for ln in chunk.splitlines())

    return (
        f"@compute @workgroup_size({cls.workgroup_size})\n"
        f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
        f"  let i: u32 = gid.x;\n"
        f"{body}\n}}"
    )


def transpile_compute(cls: type) -> str:
    """Walk the class's methods and emit a complete WGSL shader source.

    Single entry point. The Phase-1 Compute base class's default
    `compile()` calls this. Raises `ComputeShaderCompileError` for any
    Python-source-level issue.
    """
    methods = _collect_method_asts(cls)
    _check_no_recursion(methods)

    sigs = _build_method_signatures(cls)
    bindings = _resolve_class_bindings(cls)
    uniforms = _resolve_class_uniforms(cls)

    header = _emit_header(uniforms, list(cls._reads), list(cls._writes))

    helper_chunks: List[str] = []
    for name, fn in methods.items():
        if name == "main":
            continue
        helper_chunks.append(_emit_helper(cls, name, fn, sigs, bindings, uniforms))

    main_chunk = _emit_main(cls, methods["main"], sigs, bindings, uniforms)

    parts = [header, *helper_chunks, main_chunk]
    return "\n\n".join(p for p in parts if p)
