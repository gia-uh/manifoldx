"""Python to JavaScript for the code an app runs in the browser.

One ES module per Python module, holding what the roots reach. Every construct outside
the subset raises TranspileError with its file and line; nothing is dropped. Types
(types.py) decide how an operator, a test or a call is written: `[] or x` needs the
runtime's `or_`, `a + b` on lists concatenates, `%` takes the divisor's sign.

The generated code imports the runtime as `$rt` (js/pyrt.js). Names the transpiler makes
start with `$`, which a Python name cannot contain, so they never collide with the app's.
"""

from __future__ import annotations

import ast
import builtins
import json
import math
from dataclasses import dataclass

from .types import (
    LIB_CONSTANTS,
    MATH_FUNCTIONS,
    Ctx,
    FuncInfo,
    FunctionKey,
    Ty,
    TypeEnv,
    UNKNOWN,
    arith_dtype,
    elem,
    key_label,
    own_statements,
    precision,
    target_names,
)


class TranspileError(Exception):
    """A reached construct outside the subset, at path:line, and the root that reached it."""

    def __init__(self, path: str, line: int, what: str, root: str | None = None):
        super().__init__(path, line, what, root)
        self.path, self.line, self.what, self.root = path, line, what, root

    def __str__(self) -> str:
        where = f" (reached from {self.root})" if self.root else ""
        return f"{self.path}:{self.line}: {self.what} is outside the subset{where}"


@dataclass(frozen=True)
class JsModule:
    name: str
    code: str
    source_map: str | None = None


# JS reserved words and the globals the generated code uses: a Python name equal to one
# gets a "_$" suffix.
RESERVED = frozenset(
    """break case catch class const continue debugger default delete do else enum export
    extends false finally for function if implements import in instanceof interface let new
    null package private protected public return static super switch this throw true try
    typeof var void while with yield await arguments eval undefined NaN Infinity Math Object
    Array Number String Set Map JSON Error TypeError Float32Array ArrayBuffer""".split()
)

FRIENDLY = {
    "Try": "try/except",
    "TryStar": "try/except*",
    "Raise": "raise",
    "With": "a with block",
    "AsyncWith": "an async with block",
    "Global": "global",
    "Nonlocal": "nonlocal",
    "Delete": "del",
    "Assert": "assert",
    "AsyncFunctionDef": "an async function",
    "ClassDef": "a class",
    "FunctionDef": "a nested function",
    "Yield": "yield",
    "YieldFrom": "yield from",
    "Await": "await",
    "NamedExpr": "the := operator",
    "JoinedStr": "an f-string",
    "DictComp": "a dict comprehension",
    "Match": "match",
    "AsyncFor": "async for",
    "Starred": "a starred expression",
    "Slice": "a slice",
    "Import": "an import inside a function",
    "ImportFrom": "an import inside a function",
    "TypeAlias": "a type alias",
}

BUILTINS = frozenset(dir(builtins))


def js_name(name: str) -> str:
    return name + "_$" if name in RESERVED else name


def module_path(name: str) -> str:
    return name.replace(".", "/") + ".py"


# numpy's arithmetic as runtime helpers; a "32" suffix gives a Float32Array.
VECTOR_OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
SCALAR_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
MATH_JS = {"fabs": "Math.abs", "radians": "$rt.radians", "degrees": "$rt.degrees"}


def _simple(n: ast.expr) -> bool:
    """n evaluates to the same value twice and without side effects."""
    if isinstance(n, (ast.Name, ast.Constant)):
        return True
    return isinstance(n, ast.Attribute) and _simple(n.value)


def _const_int(n: ast.expr) -> int | None:
    if isinstance(n, ast.Constant) and type(n.value) is int:
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = _const_int(n.operand)
        return None if v is None else -v
    return None


def _constant(n: ast.expr) -> bool:
    """A literal whose value cannot change between calls: a safe parameter default."""
    if isinstance(n, ast.Constant):
        return True
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        return _constant(n.operand)
    return isinstance(n, ast.Tuple) and all(_constant(e) for e in n.elts)


# ---------------------------------------------------------------- reach


class _Reach:
    """Which functions and module globals are emitted, and the root that reached each.

    With no roots, everything is. With roots ("module.function"), what their bodies name
    through module-level names and app-module attributes, transitively."""

    def __init__(self, env: TypeEnv, roots: list[str] | None):
        self.env = env
        self.everything = roots is None
        self.functions: dict[FunctionKey, str] = {}
        self.globals: dict[tuple[str, str], str] = {}
        if roots is None:
            return
        todo = []
        for label in roots:
            key = self._root_key(label)
            self.functions[key] = label
            todo.append((key[0], self.env.function(key).node, label, self.env.function(key)))
        while todo:
            module, node, label, info = todo.pop()
            for n in ast.walk(node):
                sym = None
                if isinstance(n, ast.Name) and not (
                    info and (n.id in info.locals or n.id in info.param_types)
                ):
                    sym = self.env.resolve(module, n.id)
                elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                    outer = self.env.resolve(module, n.value.id)
                    if outer is not None and outer.kind == "module":
                        sym = self.env.resolve(outer.target, n.attr)
                if sym is None:
                    continue
                if sym.kind == "func" and sym.target not in self.functions:
                    self.functions[sym.target] = label
                    f = self.env.function(sym.target)
                    todo.append((sym.target[0], f.node, label, f))
                elif sym.kind == "const" and sym.target not in self.globals:
                    self.globals[sym.target] = label
                    value = self._global_value(*sym.target)
                    if value is not None:
                        todo.append((sym.target[0], value, label, None))

    def _root_key(self, label: str) -> FunctionKey:
        module, _, name = label.rpartition(".")
        while module and module not in self.env.modules:
            module, _, head = module.rpartition(".")
            name = f"{head}.{name}"
        if (module, name) not in self.env.functions:
            raise ValueError(f"no module-level function {label} in the sources")
        return (module, name)

    def _global_value(self, module: str, name: str) -> ast.expr | None:
        for n in self.env.modules[module].body:
            if isinstance(n, ast.Assign) and any(name in target_names(t) for t in n.targets):
                return n.value
            if isinstance(n, ast.AnnAssign) and name in target_names(n.target):
                return n.value
        return None

    def function(self, key: FunctionKey) -> bool:
        return self.everything or key in self.functions

    def root(self, key: FunctionKey) -> str | None:
        return self.functions.get(key)

    def global_(self, module: str, name: str) -> bool:
        return self.everything or (module, name) in self.globals


def transpile_app(
    sources: dict[str, str], roots: list[str] | None = None, live=None
) -> dict[str, JsModule]:
    """Transpile the app's modules ({module name: source}); with roots ("module.function"),
    only what they reach. A module "pkg.sub" is reported as pkg/sub.py and written as
    pkg.sub.js; its imports of other app modules become ES imports of those files."""
    trees = {name: ast.parse(src, filename=module_path(name)) for name, src in sources.items()}
    env = TypeEnv(trees, live)
    env.solve()
    reach = _Reach(env, roots)
    return {name: _Module(env, name, reach).write() for name in trees}


# ---------------------------------------------------------------- one module

Lines = list[tuple[str, int]]


class _Module:
    def __init__(self, env: TypeEnv, name: str, reach: _Reach):
        self.env, self.name, self.reach = env, name, reach
        self.path = module_path(name)
        self.ctx = Ctx(name)
        self.root: str | None = None
        self.uses_rt = False

    # ---------------------------------------------------------- helpers
    def fail(self, node: ast.AST, what: str):
        raise TranspileError(self.path, getattr(node, "lineno", 0), what, self.root)

    def rt(self, name: str) -> str:
        self.uses_rt = True
        return f"$rt.{name}"

    def ty(self, n: ast.expr) -> Ty:
        return self.env.expr(n, self.ctx)

    def unknown(self, n: ast.expr, operand: ast.expr):
        self.fail(
            n,
            f"`{ast.unparse(n)}` with an operand of unknown type "
            f"(annotate `{ast.unparse(operand)}`)",
        )

    def numeric(self, call: ast.expr, x: ast.expr, what: str) -> None:
        """Fail unless x is known to be a number: `what` is the call that needs one."""
        t = self.ty(x)
        if not t.is_numeric:
            self.fail(
                call,
                f"{what} of a {t}"
                if t.known
                else f"{what} of `{ast.unparse(x)}`, of unknown type (annotate it),",
            )

    def test(self, n: ast.expr) -> str:
        """n in a test position: plain JS only for a known bool."""
        return self.e(n) if self.ty(n).is_bool else f"{self.rt('truthy')}({self.e(n)})"

    def is_local(self, name: str) -> bool:
        f = self.ctx.func
        return any(k == name for k, _ in self.ctx.extra) or (
            f is not None and (name in f.locals or name in f.param_types)
        )

    # ---------------------------------------------------------- module
    def write(self) -> JsModule:
        body: Lines = []
        imports: list[str] = []
        for n in self.env.modules[self.name].body:
            self.root = None
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                imports += self.import_(n)
            elif isinstance(n, ast.If) and ast.unparse(n.test) in (
                "__name__ == '__main__'",
                '__name__ == "__main__"',
            ):
                continue  # the script entry point is setup
            elif isinstance(n, ast.FunctionDef):
                key = (self.name, n.name)
                if self.reach.function(key):
                    self.root = self.reach.root(key)
                    body += self.function(n, self.env.function(key)) + [("", n.end_lineno)]
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                names = [x for t in targets for x in target_names(t)]
                reached = [x for x in names if self.reach.global_(self.name, x)]
                if reached and n.value is not None:
                    self.root = self.reach.globals.get((self.name, reached[0]))
                    if len(targets) != 1:
                        self.fail(n, "a chained assignment at module level")
                    body.append(
                        (f"export let {self.target(targets[0])} = {self.e(n.value)};", n.lineno)
                    )
            elif isinstance(n, ast.ClassDef):
                if self.reach.everything:
                    self.fail(n, f"the class {n.name}")
            elif self.reach.everything:
                self.fail(n, FRIENDLY.get(type(n).__name__, type(n).__name__))
        head = [f"// Generated from {self.path} by manifoldx.web; do not edit."]
        if self.uses_rt:
            head.append('import * as $rt from "./pyrt.js";')
        lines = [(h, 0) for h in head + imports] + [("", 0)] + body
        return JsModule(self.name, "\n".join(text for text, _ in lines) + "\n")

    def exported(self, module: str, name: str) -> bool:
        sym = self.env.symbols[module].get(name)
        if sym is None:
            return False
        if sym.kind == "func":
            return self.reach.function(sym.target)
        if sym.kind == "const":
            return self.reach.global_(*sym.target)
        return False

    def import_(self, n: ast.Import | ast.ImportFrom) -> list[str]:
        """ES imports for the app modules n names; other imports bind setup-only names."""
        out = []
        for a in n.names:
            bound = a.asname or a.name.split(".")[0]
            sym = self.env.symbols[self.name].get(bound)
            if sym is None:
                continue
            if sym.kind == "module":
                out.append(f'import * as {js_name(bound)} from "./{sym.target}.js";')
            elif sym.kind == "import":
                final = self.env.resolve(self.name, bound)
                if final is not None and final.kind == "module":
                    out.append(f'import * as {js_name(bound)} from "./{final.target}.js";')
                elif final is not None and final.kind in ("func", "const"):
                    module, name = final.target
                    if self.exported(module, name):
                        alias = "" if name == bound else f" as {js_name(bound)}"
                        out.append(f'import {{ {js_name(name)}{alias} }} from "./{module}.js";')
        return out

    # ---------------------------------------------------------- functions
    def function(self, n: ast.FunctionDef, info: FuncInfo) -> Lines:
        a = n.args
        if a.vararg or a.kwarg:
            self.fail(n, "*args or **kwargs")
        if n.decorator_list:
            self.fail(n, f"the decorator @{ast.unparse(n.decorator_list[0])}")
        params = []
        for p in info.params:
            d = info.defaults.get(p)
            if d is not None and not _constant(d):
                self.fail(d, f"the default `{ast.unparse(d)}`, which is not a constant,")
            params.append(js_name(p) if d is None else f"{js_name(p)} = {self.e(d)}")
        outer, self.ctx = self.ctx, Ctx(self.name, info)
        try:
            local = [js_name(v) for v in info.locals if v not in info.param_types]
            out = [(f"export function {js_name(n.name)}({', '.join(params)}) {{", n.lineno)]
            if local:
                out.append((f"  let {', '.join(local)};", n.lineno))
            out += self.block(n.body, 1)
            return out + [("}", n.end_lineno)]
        finally:
            self.ctx = outer

    def block(self, body: list[ast.stmt], ind: int) -> Lines:
        out: Lines = []
        for st in body:
            out += self.stmt(st, ind)
        return out

    # ---------------------------------------------------------- statements
    def stmt(self, n: ast.stmt, ind: int) -> Lines:
        p = "  " * ind
        line = n.lineno
        if isinstance(n, ast.Expr):
            if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                return []  # a docstring
            return [(f"{p}{self.e(n.value)};", line)]
        if isinstance(n, ast.Pass):
            return []
        if isinstance(n, ast.Return):
            return [(f"{p}return {self.e(n.value) if n.value else 'null'};", line)]
        if isinstance(n, ast.Assign):
            return [(f"{p}{self.assign(n.targets, n.value, n)};", line)]
        if isinstance(n, ast.AnnAssign):
            return [] if n.value is None else [(f"{p}{self.assign([n.target], n.value, n)};", line)]
        if isinstance(n, ast.AugAssign):
            t = n.target
            if isinstance(t, ast.Subscript) and not (_simple(t.value) and _simple(t.slice)):
                self.fail(n, f"`{ast.unparse(n)}` on a computed target")
            if not isinstance(t, (ast.Name, ast.Attribute, ast.Subscript)):
                self.fail(n, "this augmented assignment")
            if self.ty(t).kind == "vec":
                return [(f"{p}{self.vector_inplace(n)};", line)]
            value = ast.copy_location(ast.BinOp(t, n.op, n.value), n)
            load = ast.copy_location(type(t)(**{**vars(t), "ctx": ast.Load()}), t)
            value.left = load
            return [(f"{p}{self.assign([t], value, n)};", line)]
        if isinstance(n, ast.If):
            out = [(f"{p}if ({self.test(n.test)}) {{", line), *self.block(n.body, ind + 1)]
            while len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If):
                n = n.orelse[0]
                out += [
                    (f"{p}}} else if ({self.test(n.test)}) {{", n.lineno),
                    *self.block(n.body, ind + 1),
                ]
            if n.orelse:
                out += [(f"{p}}} else {{", n.orelse[0].lineno), *self.block(n.orelse, ind + 1)]
            return out + [(f"{p}}}", line)]
        if isinstance(n, ast.For):
            if n.orelse:
                self.fail(n, "for/else")
            head, first = self.for_head(n.target, n.iter, declare=False)
            body = [(f"{p}  {first}", line)] if first else []
            return [(f"{p}{head} {{", line), *body, *self.block(n.body, ind + 1), (f"{p}}}", line)]
        if isinstance(n, ast.While):
            if n.orelse:
                self.fail(n, "while/else")
            return [
                (f"{p}while ({self.test(n.test)}) {{", line),
                *self.block(n.body, ind + 1),
                (f"{p}}}", line),
            ]
        if isinstance(n, ast.Break):
            return [(f"{p}break;", line)]
        if isinstance(n, ast.Continue):
            return [(f"{p}continue;", line)]
        self.fail(n, FRIENDLY.get(type(n).__name__, type(n).__name__))

    def vector_inplace(self, n: ast.AugAssign) -> str:
        """`a += b` on a numpy array writes into a, which every alias of a sees."""
        if type(n.op) not in VECTOR_OPS:
            self.fail(n, f"`{ast.unparse(n)}` on a numpy array")
        a_t, b_t = self.ty(n.target), self.ty(n.value)
        if not (b_t.kind == "vec" or b_t.is_numeric) or precision(b_t) == "mixed":
            self.fail(n, f"`{ast.unparse(n)}` with a {b_t} on the right")
        # Into a float32 array, a float32 or Python right side is rounded to float32 first.
        f32 = a_t.dtype == "f32" and precision(b_t) in (None, "f32")
        helper = f"vi{VECTOR_OPS[type(n.op)]}{'32' if f32 else ''}"
        return f"{self.rt(helper)}({self.e(n.target)}, {self.e(n.value)})"

    def assign(self, targets: list[ast.expr], value: ast.expr, n: ast.stmt) -> str:
        if len(targets) > 1:
            if not all(isinstance(t, ast.Name) for t in targets):
                self.fail(n, "a chained assignment to anything but names")
            return " = ".join(js_name(t.id) for t in targets) + f" = {self.e(value)}"
        t = targets[0]
        if isinstance(t, ast.Subscript):
            return self.setitem(t, self.e(value))
        if isinstance(t, (ast.Tuple, ast.List)):
            return f"{self.target(t)} = {self.iterable(value, destructure=True)}"
        return f"{self.target(t)} = {self.e(value)}"

    def target(self, t: ast.expr) -> str:
        if isinstance(t, ast.Name):
            return js_name(t.id)
        if isinstance(t, (ast.Tuple, ast.List)):
            parts = []
            for i, e in enumerate(t.elts):
                if isinstance(e, ast.Starred):
                    if i != len(t.elts) - 1 or not isinstance(e.value, ast.Name):
                        self.fail(t, "a starred target before the last position")
                    parts.append(f"...{js_name(e.value.id)}")
                elif isinstance(e, (ast.Name, ast.Tuple, ast.List)):
                    parts.append(self.target(e))
                else:
                    self.fail(e, f"unpacking into `{ast.unparse(e)}`")
            return "[" + ", ".join(parts) + "]"
        if isinstance(t, ast.Attribute):
            return self.e(t)
        self.fail(t, f"assignment to `{ast.unparse(t)}`")

    def setitem(self, t: ast.Subscript, value: str) -> str:
        if isinstance(t.slice, ast.Slice):
            self.fail(t, "assignment to a slice")
        c = self.ty(t.value)
        obj, key = self.e(t.value), self.e(t.slice)
        if c.all("dict") or (
            c.all("list", "vec")
            and (_const_int(t.slice) or 0) >= 0
            and _const_int(t.slice) is not None
        ):
            return f"{obj}[{key}] = {value}"
        return f"{self.rt('setitem')}({obj}, {key}, {value})"

    def iterable(self, it: ast.expr, destructure: bool = False) -> str:
        """`it` as something a JS for-of (or a destructuring) walks the way Python does."""
        t = self.ty(it)
        if t.all("list", "tuple", "set", "str", "vec"):
            return self.e(it)
        if t.all("dict"):
            return f"Object.keys({self.e(it)})"
        return f"{self.rt('iter')}({self.e(it)})"

    def is_builtin_call(self, n: ast.expr, name: str) -> bool:
        return (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == name
            and not self.is_local(name)
            and self.env.resolve(self.ctx.module, name) is None
        )

    def for_head(self, target: ast.expr, it: ast.expr, declare: bool) -> tuple[str, str]:
        """A for loop's head, and the statement that opens its body (or "")."""
        if self.is_builtin_call(it, "range"):
            if it.keywords or not 1 <= len(it.args) <= 3:
                self.fail(it, "this call to range()")
            for x in it.args:
                if not self.ty(x).all("int", "bool"):
                    self.fail(x, f"range() over `{ast.unparse(x)}`, which is not an int,")
            a = [self.e(x) for x in it.args]
            start, stop, step = (
                ("0", a[0], "1") if len(a) == 1 else ((a[0], a[1], "1") if len(a) == 2 else a)
            )
            k = _const_int(it.args[2]) if len(a) == 3 else 1
            if k is not None and k != 0 and isinstance(target, ast.Name):
                cmp = "<" if k > 0 else ">"
                v = js_name(target.id)
                if declare:  # a comprehension's own binding
                    return f"for (let {v} = {start}, $n = {stop}; {v} {cmp} $n; {v} += {step})", ""
                # Python's loop variable keeps its last value; a counter does the stepping.
                return (
                    f"for (let $i = {start}, $n = {stop}; $i {cmp} $n; $i += {step})",
                    f"{v} = $i;",
                )
            return (
                f"for ({'const ' if declare else ''}{self.target(target)} of "
                f"{self.rt('range')}({start}, {stop}, {step}))",
                "",
            )
        return (
            f"for ({'const ' if declare else ''}{self.target(target)} of {self.iterable(it)})",
            "",
        )

    # ---------------------------------------------------------- expressions
    def e(self, n: ast.expr) -> str:
        m = getattr(self, "e_" + type(n).__name__, None)
        if m is None:
            self.fail(n, FRIENDLY.get(type(n).__name__, type(n).__name__))
        return m(n)

    def e_Constant(self, n: ast.Constant) -> str:
        v = n.value
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            if abs(v) > 2**53:
                self.fail(n, f"the integer {v}, wider than a double,")
            return str(v)
        if isinstance(v, float):
            return (
                "Infinity"
                if v == math.inf
                else "-Infinity"
                if v == -math.inf
                else ("NaN" if v != v else repr(v))
            )
        if isinstance(v, str):
            return json.dumps(v)
        self.fail(n, f"the constant {v!r}")

    def e_Name(self, n: ast.Name) -> str:
        name = n.id
        if self.is_local(name):
            return js_name(name)
        sym = self.env.symbols[self.ctx.module].get(name)
        if sym is not None:
            if sym.kind == "setup":
                self.fail(n, f"`{name}` (bound by `{sym.target}`, which runs only at setup)")
            if sym.kind == "lib":
                self.fail(n, f"the module {name} as a value")
            if sym.kind == "class":
                self.fail(n, f"the class {name}")
            final = self.env.resolve(self.ctx.module, name)
            if final is None:
                self.fail(n, f"`{name}`, which its module does not define,")
            return js_name(name)
        if name == "len":
            return self.rt("len")
        if name in BUILTINS:
            self.fail(n, f"the builtin {name} as a value")
        self.fail(n, f"the undefined name {name}")

    def e_Attribute(self, n: ast.Attribute) -> str:
        lib = self.env.lib_name(n, self.ctx)
        if lib in LIB_CONSTANTS:
            return LIB_CONSTANTS[lib]
        if lib is not None:
            self.fail(n, ast.unparse(n))
        if isinstance(n.value, ast.Name) and not self.is_local(n.value.id):
            sym = self.env.symbols[self.ctx.module].get(n.value.id)
            final = self.env.resolve(self.ctx.module, n.value.id) if sym else None
            if final is not None and final.kind == "module":
                if not self.exported(final.target, n.attr):
                    self.fail(n, f"`{ast.unparse(n)}`, which is not a function or constant,")
                return f"{js_name(n.value.id)}.{js_name(n.attr)}"
            if sym is not None and sym.kind == "lib":
                self.fail(n, f"{ast.unparse(n)}")
        t = self.ty(n.value)
        if t.known and not t.all("obj"):
            self.fail(n, f"the attribute .{n.attr} of a {t}")
        return f"{self.e(n.value)}.{n.attr}"

    def items(self, elts: list[ast.expr]) -> str:
        parts = []
        for x in elts:
            if isinstance(x, ast.Starred):
                parts.append(f"...{self.iterable(x.value)}")
            else:
                parts.append(self.e(x))
        return "[" + ", ".join(parts) + "]"

    def e_List(self, n: ast.List) -> str:
        return self.items(n.elts)

    e_Tuple = e_List

    def e_Set(self, n: ast.Set) -> str:
        return f"new Set({self.items(n.elts)})"

    def e_Dict(self, n: ast.Dict) -> str:
        pairs = []
        for k, v in zip(n.keys, n.values):
            if k is None:
                self.fail(n, "a ** in a dict literal")
            if not self.ty(k).all("str"):
                self.fail(k, f"the dict key `{ast.unparse(k)}`, which is not a str,")
            pairs.append(f"[{self.e(k)}, {self.e(v)}]")
        return f"{self.rt('dict')}([{', '.join(pairs)}])"

    def e_BinOp(self, n: ast.BinOp) -> str:
        a_t, b_t = self.ty(n.left), self.ty(n.right)
        op = n.op
        if isinstance(op, (ast.Add, ast.Mult, ast.Mod)):
            for side, t in ((n.left, a_t), (n.right, b_t)):
                if not t.known:
                    self.unknown(n, side)
        if any(m.kind == "vec" for m in a_t.members + b_t.members):
            return self.vector_op(n, a_t, b_t)
        a, b = self.e(n.left), self.e(n.right)
        num = a_t.is_numeric and b_t.is_numeric
        if num and arith_dtype(a_t, b_t) is not None:
            return self.numpy_scalar_op(n, a_t, b_t, a, b)
        loose = num or not a_t.known or not b_t.known
        if isinstance(op, ast.Add):
            if num or (a_t.all("str") and b_t.all("str")):
                return f"({a} + {b})"
            if a_t.all("list", "tuple") and b_t.all("list", "tuple"):
                return f"[...{a}, ...{b}]"
        elif isinstance(op, ast.Mult):
            if num:
                return f"({a} * {b})"
            for xs, k, k_t, xs_t in ((a, b, b_t, a_t), (b, a, a_t, b_t)):
                if k_t.all("int", "bool") and xs_t.all("list", "tuple"):
                    return f"{self.rt('lrep')}({xs}, {k})"
                if k_t.all("int", "bool") and xs_t.all("str"):
                    return f"{xs}.repeat(Math.max(0, {k}))"
        elif isinstance(op, ast.Mod):
            if num:
                return f"{self.rt('mod')}({a}, {b})"
        elif isinstance(op, ast.FloorDiv):
            if loose:
                return f"{self.rt('floordiv')}({a}, {b})"
        elif isinstance(op, (ast.Sub, ast.Div, ast.Pow)):
            if loose:
                return f"({a} {({ast.Sub: '-', ast.Div: '/', ast.Pow: '**'})[type(op)]} {b})"
        else:
            self.fail(n, f"the operator {type(op).__name__}")
        self.fail(n, f"`{ast.unparse(n)}` on {a_t} and {b_t}")

    def vector_op(self, n: ast.BinOp, a_t: Ty, b_t: Ty) -> str:
        """Arithmetic with a numpy array on a side: a number broadcasts."""
        if type(n.op) not in VECTOR_OPS:
            self.fail(n, f"the operator {type(n.op).__name__} on numpy arrays")
        for side, t in ((n.left, a_t), (n.right, b_t)):
            if not (t.kind == "vec" or t.is_numeric):
                self.fail(n, f"`{ast.unparse(n)}` on a {t} (annotate `{ast.unparse(side)}`)")
        d = arith_dtype(a_t, b_t)
        if d == "mixed":
            self.fail(n, f"`{ast.unparse(n)}`, whose float32 or float64 result the run decides,")
        helper = f"v{VECTOR_OPS[type(n.op)]}{'32' if d == 'f32' else ''}"
        return f"{self.rt(helper)}({self.e(n.left)}, {self.e(n.right)})"

    def numpy_scalar_op(self, n: ast.BinOp, a_t: Ty, b_t: Ty, a: str, b: str) -> str:
        """Arithmetic on a numpy scalar: float64 is JS's own; float32 rounds each operand
        and the result, which gives float32's result for + - * /."""
        d = arith_dtype(a_t, b_t)
        if d == "mixed":
            self.fail(n, f"`{ast.unparse(n)}`, whose float32 or float64 result the run decides,")
        if type(n.op) not in SCALAR_OPS:
            self.fail(n, f"the operator {type(n.op).__name__} on a numpy {d} scalar")
        if d == "f64":
            return f"({a} {SCALAR_OPS[type(n.op)]} {b})"

        def f32(t: Ty, code: str) -> str:
            return code if all(m.dtype == "f32" for m in t.members) else f"Math.fround({code})"

        return f"Math.fround({f32(a_t, a)} {SCALAR_OPS[type(n.op)]} {f32(b_t, b)})"

    def e_UnaryOp(self, n: ast.UnaryOp) -> str:
        if isinstance(n.op, ast.Not):
            return f"!{self.test(n.operand)}"
        t = self.ty(n.operand)
        if t.kind == "vec" and isinstance(n.op, ast.USub):
            return f"{self.rt('vneg')}({self.e(n.operand)})"
        if t.known and not t.is_numeric:
            self.fail(n, f"`{ast.unparse(n)}` on a {t}")
        if isinstance(n.op, ast.USub):
            return f"(-{self.e(n.operand)})"
        if isinstance(n.op, ast.UAdd):
            return f"(+{self.e(n.operand)})"
        self.fail(n, "the operator ~")

    def e_BoolOp(self, n: ast.BoolOp) -> str:
        if all(self.ty(v).is_bool for v in n.values):
            op = " && " if isinstance(n.op, ast.And) else " || "
            return "(" + op.join(self.e(v) for v in n.values) + ")"
        # Python's and/or return an operand; the operands after the first stay lazy.
        f = self.rt("or_" if isinstance(n.op, ast.Or) else "and_")
        rest = ", ".join(f"() => {self.e(v)}" for v in n.values[1:])
        return f"{f}({self.e(n.values[0])}, {rest})"

    def compare(self, op: ast.cmpop, ln: ast.expr, rn: ast.expr, a: str, b: str, n) -> str:
        lt, rt_ = self.ty(ln), self.ty(rn)
        if not isinstance(op, (ast.In, ast.NotIn)) and any(
            m.kind == "vec" for m in lt.members + rt_.members
        ):
            self.fail(n, "comparing numpy arrays, which numpy does item by item,")
        if isinstance(op, (ast.Is, ast.IsNot)):
            if not (lt.all("none") or rt_.all("none")):
                self.fail(n, "`is` against anything but None")
            x = b if lt.all("none") else a
            return f"{x} {'==' if isinstance(op, ast.Is) else '!='} null"
        if isinstance(op, (ast.In, ast.NotIn)):
            return f"{'!' if isinstance(op, ast.NotIn) else ''}{self.rt('contains')}({b}, {a})"
        if isinstance(op, (ast.Eq, ast.NotEq)):
            neg = isinstance(op, ast.NotEq)
            if lt.all("none") or rt_.all("none"):
                return f"{b if lt.all('none') else a} {'!=' if neg else '=='} null"
            same = any(lt.all(*k) and rt_.all(*k) for k in (("num", "int"), ("str",), ("bool",)))
            if same:
                return f"{a} {'!==' if neg else '==='} {b}"
            return f"{'!' if neg else ''}{self.rt('eq')}({a}, {b})"
        sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}[type(op)]
        if (
            (lt.is_numeric and rt_.is_numeric)
            or (lt.all("str") and rt_.all("str"))
            or not (lt.known and rt_.known)
        ):
            return f"{a} {sym} {b}"
        self.fail(n, f"ordering a {lt} against a {rt_}")

    def e_Compare(self, n: ast.Compare) -> str:
        nodes = [n.left, *n.comparators]
        if len(n.ops) == 1 or all(_simple(x) for x in nodes[1:-1]):
            parts = [
                self.compare(op, nodes[i], nodes[i + 1], self.e(nodes[i]), self.e(nodes[i + 1]), n)
                for i, op in enumerate(n.ops)
            ]
            return "(" + " && ".join(parts) + ")"
        # A computed middle operand is evaluated once, in Python's order, and the chain stops
        # at the first false comparison.
        body = [f"const $c0 = {self.e(nodes[0])};"]
        for i, op in enumerate(n.ops):
            body.append(f"const $c{i + 1} = {self.e(nodes[i + 1])};")
            cmp = self.compare(op, nodes[i], nodes[i + 1], f"$c{i}", f"$c{i + 1}", n)
            body.append(f"if (!({cmp})) return false;")
        return f"(() => {{ {' '.join(body)} return true; }})()"

    def e_IfExp(self, n: ast.IfExp) -> str:
        return f"({self.test(n.test)} ? {self.e(n.body)} : {self.e(n.orelse)})"

    def e_Lambda(self, n: ast.Lambda) -> str:
        a = n.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults:
            self.fail(n, "a lambda with defaults, *args or keyword-only parameters")
        names = [x.arg for x in a.args]
        outer, self.ctx = self.ctx, self.ctx.bind({x: UNKNOWN for x in names})
        try:
            return f"(({', '.join(map(js_name, names))}) => {self.e(n.body)})"
        finally:
            self.ctx = outer

    def e_Subscript(self, n: ast.Subscript) -> str:
        t = self.ty(n.value)
        s = n.slice
        if isinstance(s, ast.Slice):
            if s.step is not None:
                self.fail(n, "a slice with a step")
            if t.kind == "vec":  # a view, as numpy's
                lo = self.e(s.lower) if s.lower else "0"
                hi = f", {self.e(s.upper)}" if s.upper else ""
                return f"{self.e(n.value)}.subarray({lo}{hi})"
            if not t.all("list", "tuple"):
                self.fail(
                    n,
                    f"slicing a {t}"
                    if t.known
                    else f"slicing `{ast.unparse(n.value)}`, of unknown type (annotate it),",
                )
            lo = self.e(s.lower) if s.lower else "0"
            hi = f", {self.e(s.upper)}" if s.upper else ""
            return f"{self.e(n.value)}.slice({lo}{hi})"
        obj = self.e(n.value)
        if t.all("dict"):
            return f"{obj}[{self.e(s)}]"
        if t.all("list", "tuple", "vec"):
            k = _const_int(s)
            return f"{obj}[{k}]" if k is not None and k >= 0 else f"{obj}.at({self.e(s)})"
        return f"{self.rt('getitem')}({obj}, {self.e(s)})"

    def comprehension(self, n, wrap: str = "") -> str:
        """A comprehension as an IIFE that fills an array, eagerly; a generator expression
        is one too (the subset has no lazy generators)."""
        heads = []
        outer = self.ctx
        try:
            for g in n.generators:
                if g.is_async:
                    self.fail(n, "an async comprehension")
                head, _ = self.for_head(g.target, g.iter, declare=True)
                self.ctx = self.env.comprehension([g], self.ctx)
                heads.append((head, [self.test(c) for c in g.ifs]))
            body = f"$r.push({self.e(n.elt)});"
        finally:
            self.ctx = outer
        for head, conds in reversed(heads):
            for c in reversed(conds):
                body = f"if ({c}) {{ {body} }}"
            body = f"{head} {{ {body} }}"
        iife = f"(() => {{ const $r = []; {body} return $r; }})()"
        return f"new Set({iife})" if wrap == "set" else iife

    def e_ListComp(self, n: ast.ListComp) -> str:
        return self.comprehension(n)

    e_GeneratorExp = e_ListComp

    def e_SetComp(self, n: ast.SetComp) -> str:
        return self.comprehension(n, "set")

    # ---------------------------------------------------------- calls
    def e_Call(self, n: ast.Call) -> str:
        if any(isinstance(a, ast.Starred) for a in n.args) or any(
            k.arg is None for k in n.keywords
        ):
            self.fail(n, "*args or **kwargs in a call")
        f = n.func
        if isinstance(f, ast.Name) and not self.is_local(f.id):
            sym = self.env.symbols[self.ctx.module].get(f.id)
            if sym is None:
                if f.id in BUILTINS:
                    return self.builtin(f.id, n)
                self.fail(n, f"the undefined name {f.id}")
            final = self.env.resolve(self.ctx.module, f.id)
            if final is not None and final.kind == "func":
                return self.place(n, self.env.function(final.target), js_name(f.id))
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and not self.is_local(f.value.id)
        ):
            outer = self.env.resolve(self.ctx.module, f.value.id)
            if outer is not None and outer.kind == "module":
                target = self.env.resolve(outer.target, f.attr)
                if target is not None and target.kind == "func":
                    return self.place(n, self.env.function(target.target), self.e(f))
        if isinstance(f, ast.Attribute):
            lib = self.env.lib_name(f, self.ctx)
            if lib is not None:
                return self.lib_call(lib, n)
            method = self.method(n, f)
            if method is not None:
                return method
        if n.keywords:
            self.fail(n, f"keyword arguments to `{ast.unparse(f)}`, whose signature is unknown,")
        return f"{self.e(f)}({', '.join(self.e(a) for a in n.args)})"

    def lib_call(self, name: str, n: ast.Call) -> str:
        """A call into math or numpy."""
        what = f"{ast.unparse(n.func)}()"
        args, kw = n.args, {k.arg for k in n.keywords}
        if name in ("numpy.array", "numpy.zeros"):
            if kw - {"dtype"} or not 1 <= len(args) + len(kw) <= 2:
                self.fail(n, f"this call to {what}")
            d = self.env.np_dtype(n, self.ctx)
            if d is None:
                self.fail(n, f"the dtype in `{ast.unparse(n)}` (float32 and float64 are in)")
            cls = "Float32Array" if d == "f32" else "Float64Array"
            x, t = args[0], self.ty(args[0])
            if name == "numpy.zeros":
                if not t.all("int", "bool"):
                    self.fail(n, f"{what} of a shape that is not an int")
                return f"new {cls}({self.e(x)})"
            if not (t.kind == "vec" or (t.all("list", "tuple") and elem(t).is_numeric)):
                self.fail(n, f"{what} of a {t}" if t.known else f"{what} of an unknown type")
            return f"{cls}.from({self.iterable(x)})"
        if kw:
            self.fail(n, f"keyword arguments to {what}")
        if name == "numpy.arange" and 1 <= len(args) <= 3:
            for x in args:
                self.numeric(n, x, what)
            return f"{self.rt('arange')}({', '.join(self.e(x) for x in args)})"
        if name == "numpy.linalg.norm" and len(args) == 1:
            t = self.ty(args[0])
            if t.kind != "vec":
                self.fail(n, f"{what} of a {t}, not a numpy array,")
            return f"{self.rt('vnorm32' if t.dtype == 'f32' else 'vnorm')}({self.e(args[0])})"
        fname = name.removeprefix("math.")
        arity = 2 if fname in ("atan2", "hypot") else 1
        if fname in MATH_FUNCTIONS and name.startswith("math.") and len(args) == arity:
            for x in args:
                self.numeric(n, x, what)
            js = MATH_JS.get(fname, f"Math.{fname}")
            if js.startswith("$rt."):
                js = self.rt(js[4:])
            return f"{js}({', '.join(self.e(x) for x in args)})"
        self.fail(n, what)

    def method(self, n: ast.Call, f: ast.Attribute) -> str | None:
        """A method call on a typed builtin value; None for an app object's method."""
        t = self.ty(f.value)
        if t.all("obj"):
            return None
        if (
            isinstance(f.value, ast.Name)
            and self.env.symbols[self.ctx.module].get(f.value.id, None) is not None
            and not self.is_local(f.value.id)
        ):
            sym = self.env.symbols[self.ctx.module][f.value.id]
            if sym.kind == "setup":
                self.fail(
                    n,
                    f"`{ast.unparse(f)}()` (`{f.value.id}` is bound by "
                    f"`{sym.target}`, which runs only at setup)",
                )
            if sym.kind == "lib":
                self.fail(n, f"{ast.unparse(f)}()")
        if not t.known:
            self.fail(
                n,
                f"`{ast.unparse(n)}`, a method call on a value of unknown type "
                f"(annotate `{ast.unparse(f.value)}`),",
            )
        if n.keywords:
            self.fail(n, f"keyword arguments to .{f.attr}()")
        obj, args = self.e(f.value), [self.e(a) for a in n.args]
        if t.all("list"):
            if f.attr == "append" and len(args) == 1:
                return f"{obj}.push({args[0]})"
            if f.attr == "pop" and (not args or _const_int(n.args[0]) == -1):
                return f"{obj}.pop()"
            if f.attr == "pop" and _const_int(n.args[0]) == 0:
                return f"{obj}.shift()"
            if f.attr == "extend" and len(args) == 1:
                return f"{obj}.push(...{self.iterable(n.args[0])})"
            if f.attr == "insert" and len(args) == 2:
                return f"{obj}.splice({args[0]}, 0, {args[1]})"
        self.fail(n, f"the method {t}.{f.attr}()")

    def place(self, n: ast.Call, info: FuncInfo, callee: str) -> str:
        """Arguments placed by the callee's signature; keywords become positions."""
        label = key_label(info.key)
        a = info.node.args
        positional = len(a.posonlyargs) + len(a.args)
        if len(n.args) > positional:
            self.fail(n, f"a call to {label} with {len(n.args)} positional arguments")
        slots: list[ast.expr | None] = [*n.args] + [None] * (len(info.params) - len(n.args))
        for k in n.keywords:
            if (
                k.arg not in info.params
                or info.params.index(k.arg) < len(n.args)
                or (k.arg in [x.arg for x in a.posonlyargs])
            ):
                self.fail(n, f"the keyword {k.arg} to {label}")
            slots[info.params.index(k.arg)] = k.value
        for p, s in zip(info.params, slots):
            if s is None and p not in info.defaults:
                self.fail(n, f"a call to {label} without the argument {p}")
        while slots and slots[-1] is None:
            slots.pop()
        given = [*n.args, *(k.value for k in n.keywords)]
        in_order = [s for s in slots if s is not None] == given
        if in_order or all(_simple(x) for x in given):
            args = ["undefined" if s is None else self.e(s) for s in slots]
            return f"{callee}({', '.join(args)})"
        # Keywords out of the parameters' order: evaluate every argument in source order.
        names = {id(x): f"$a{i}" for i, x in enumerate(given)}
        args = ["undefined" if s is None else names[id(s)] for s in slots]
        return (
            f"(({', '.join(names.values())}) => {callee}({', '.join(args)}))"
            f"({', '.join(self.e(x) for x in given)})"
        )

    def builtin(self, name: str, n: ast.Call) -> str:
        kw = {k.arg: k.value for k in n.keywords}
        args = n.args
        if name in ("min", "max"):
            if set(kw) - {"key"} or not args:
                self.fail(n, f"this call to {name}()")
            key = self.e(kw["key"]) if "key" in kw else "null"
            if len(args) == 1:
                return f"{self.rt(name + '_')}({self.e(args[0])}, {key})"
            if "key" not in kw and all(self.ty(x).is_numeric for x in args):
                return f"Math.{name}({', '.join(self.e(x) for x in args)})"
            return f"{self.rt(name + '_')}({self.items(args)}, {key})"
        if kw:
            self.fail(n, f"keyword arguments to {name}()")
        if name == "range" and 1 <= len(args) <= 3:
            return f"{self.rt('range')}({', '.join(self.e(x) for x in args)})"
        if name in ("list", "tuple") and len(args) <= 1:
            return "[]" if not args else f"Array.from({self.iterable(args[0])})"
        if name == "set" and len(args) <= 1:
            return f"new Set({self.iterable(args[0]) if args else ''})"
        if len(args) != 1:
            self.fail(n, f"this call to {name}()")
        x, t = args[0], self.ty(args[0])
        if name == "len":
            if t.all("list", "tuple", "vec"):
                return f"{self.e(x)}.length"
            if t.all("set"):
                return f"{self.e(x)}.size"
            return f"{self.rt('len')}({self.e(x)})"
        if name == "bool":
            return self.test(x)
        if name in ("any", "all"):
            return f"{self.rt(name + '_')}({self.e(x)})"
        if name in ("abs", "int", "float"):
            if not t.is_numeric:
                self.fail(
                    n,
                    f"{name}() of a {t}"
                    if t.known
                    else f"{name}() of `{ast.unparse(x)}`, of unknown type (annotate it),",
                )
            js = {"abs": "Math.abs", "int": "Math.trunc", "float": "Number"}[name]
            return f"{js}({self.e(x)})"
        self.fail(n, f"{name}()")
