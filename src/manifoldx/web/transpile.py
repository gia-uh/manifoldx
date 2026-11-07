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
import re
from dataclasses import dataclass

from .reach import Reach, closure_site
from .types import (
    GEOMETRY,
    LIB_CONSTANTS,
    MATH_FUNCTIONS,
    ND_FUNCTIONS,
    ND_ONLY_OPS,
    RANDOM_FUNCTIONS,
    RUNTIME_KEYWORDS,
    RUNTIME_SURFACE,
    STATE,
    STR_METHODS,
    TIME_HINT,
    ClassInfo,
    Ctx,
    FuncInfo,
    FunctionKey,
    LiveTypes,
    Ty,
    TypeEnv,
    TranspileError,
    arith_dtype,
    component_fields,
    array_rank,
    elem,
    field_compare_dtype,
    is_array,
    is_material,
    nd_compare_dtype,
    readonly,
    key_label,
    module_path,
    own_statements,
    precision,
    target_names,
)


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


# numpy's arithmetic as runtime helpers; a "32" suffix gives a Float32Array.
VECTOR_OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
# Operators on numpy arrays of any rank, as $rt.nd's functions.
ND_OPS = {**VECTOR_OPS, ast.BitAnd: "and", ast.BitOr: "or", ast.MatMult: "matmul"}
ND_COMPARE = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}
SCALAR_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
# A field view's operators with the field on the right (a number on the left), as _FieldView's
# reflected methods compute them; and its in-place forms, which queue a write.
FIELD_ROPS = {"add": "radd", "sub": "rsub", "mul": "mul", "div": "rtruediv"}
FIELD_IOPS = {ast.Add: "iadd", ast.Sub: "isub", ast.Mult: "imul", ast.Div: "idiv"}
OP_TEXT = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
MATH_JS = {"fabs": "Math.abs", "radians": "$rt.radians", "degrees": "$rt.degrees"}


# What a dict's attributes name; on engine.state these win over its keys, as in Python.
DICT_ATTRS = frozenset(dir(dict))


def _simple(n: ast.expr) -> bool:
    """n evaluates to the same value twice and without side effects."""
    if isinstance(n, (ast.Name, ast.Constant)):
        return True
    return isinstance(n, ast.Attribute) and _simple(n.value)


def _const_number(n: ast.expr) -> float | None:
    """The number a constant (or a negated constant) spells, else None."""
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = _const_number(n.operand)
        return None if v is None else -v
    return None


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


def reach_app(
    sources: dict[str, str], roots: list | None = None, live: LiveTypes | None = None
) -> Reach:
    """What roots reach among the app's modules ({module name: source}), typed."""
    trees = {name: ast.parse(src, filename=module_path(name)) for name, src in sources.items()}
    return Reach(roots, TypeEnv(trees, live))


def transpile_app(
    sources: dict[str, str],
    roots: list | None = None,
    live: LiveTypes | None = None,
    reach: Reach | None = None,
    runtime: str = "./pyrt.js",
) -> dict[str, JsModule]:
    """Transpile the app's modules ({module name: source}); with roots (reach.Root objects,
    or "module.function" strings), only what they reach. A module "pkg.sub" is reported as
    pkg/sub.py and written as pkg.sub.js; its imports of other app modules become ES
    imports of those files, and the runtime's import reads `runtime`. A reach_app result
    given as reach is used instead of a new one (the snapshot's, so both see one reach)."""
    if reach is None:
        reach = reach_app(sources, roots, live)
    return {name: _Module(reach.env, name, reach, runtime).write() for name in reach.env.modules}


_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _vlq(n: int) -> str:
    """n as a source map's base64 VLQ: the sign in the lowest bit, then 5 bits per digit."""
    n = (-n << 1) | 1 if n < 0 else n << 1
    out = ""
    while True:
        digit, n = n & 31, n >> 5
        out += _B64[digit | (32 if n else 0)]
        if not n:
            return out


def source_map(file: str, source: str, lines: list[int]) -> str:
    """A source map v3 at line granularity: generated line i maps from its column 0 to the
    start of the Python line lines[i] (1-based); a 0 maps nothing."""
    groups, prev = [], 0
    for line in lines:
        if line <= 0:
            groups.append("")
            continue
        groups.append("A" + "A" + _vlq(line - 1 - prev) + "A")
        prev = line - 1
    return json.dumps(
        {"version": 3, "file": file, "sources": [source], "names": [], "mappings": ";".join(groups)}
    )


# ---------------------------------------------------------------- one module

Lines = list[tuple[str, int]]


class _Module:
    def __init__(self, env: TypeEnv, name: str, reach: Reach, runtime: str = "./pyrt.js"):
        self.env, self.name, self.reach, self.runtime = env, name, reach, runtime
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
        return self.ctx.binder(name) is not None

    def is_self(self, name: str) -> bool:
        """name is the instance parameter of the method it is read in (or of a method a
        closure or lambda sits in): it emits as `this`, which arrows keep."""
        c = self.ctx.binder(name)
        return c is not None and not any(k == name for k, _ in c.extra) and c.func.self_name == name

    # ---------------------------------------------------------- module
    def write(self) -> JsModule:
        body: Lines = []
        imports: list[str] = []
        from_state: list[str] = []
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
                if reached and not self.reach.everything:
                    # Setup's value arrives with the snapshot (__set_globals).
                    for x in reached:
                        if x not in from_state:
                            from_state.append(x)
                            body.append((f"export let {js_name(x)} = null;", n.lineno))
                elif reached and n.value is not None:
                    self.root = self.reach.root(("global", self.name, reached[0]))
                    if len(targets) != 1:
                        self.fail(n, "a chained assignment at module level")
                    body.append(
                        (f"export let {self.target(targets[0])} = {self.e(n.value)};", n.lineno)
                    )
            elif isinstance(n, ast.ClassDef):
                qual = f"{self.name}.{n.name}"
                if self.reach.class_(qual):
                    self.root = self.reach.root(("class", qual))
                    body += self.klass(n, self.env.classes[qual]) + [("", n.end_lineno)]
            elif self.reach.everything:
                self.fail(n, FRIENDLY.get(type(n).__name__, type(n).__name__))
        for key in self.reach.closures(self.name):
            body += self.closure_factory(key) + [("", 0)]
        if from_state:
            sets = " ".join(
                f'if (Object.hasOwn($g, "{x}")) {js_name(x)} = $g["{x}"];' for x in from_state
            )
            body.append((f"export function __set_globals($g) {{ {sets} }}", 0))
        head = [f"// Generated from {self.path} by manifoldx.web; do not edit."]
        if self.uses_rt:
            head.append(f'import * as $rt from "{self.runtime}";')
        lines = [(h, 0) for h in head + imports] + [("", 0)] + body
        # One generated line per entry: an entry whose text holds newlines maps each.
        lines = [(t, line) for text, line in lines for t in text.split("\n")]
        js = f"{self.name}.js"
        lines.append((f"//# sourceMappingURL={js}.map", 0))
        code = "\n".join(text for text, _ in lines) + "\n"
        return JsModule(self.name, code, source_map(js, self.path, [line for _, line in lines]))

    def closure_factory(self, key: FunctionKey) -> Lines:
        """A lambda or inner def that setup made, as `__closure_<line>_<col>(cells,
        defaults)`: it returns the function, its free variables read from `cells` and its
        defaults from `defaults`, as the snapshot's {"__closure": ...} entry gives them."""
        info = self.env.function(key)
        line, col = closure_site(self.env, key)
        self.root = self.reach.root(key)
        # Its free variables: the cells setup left, and for an inner def reached without
        # being a root, every name of the functions around it (the snapshot sends the cells).
        cells = list(self.env.cells.get(key, {}))
        up = info.parent
        while up is not None:
            outer = self.env.function(up)
            cells += [x for x in [*outer.param_types, *outer.locals] if x not in cells]
            up = outer.parent
        parent = self.env.ctx_of(self.env.function(info.parent)) if info.parent else None
        ctx = Ctx(self.name, info, tuple(self.env.cells.get(key, {}).items()), parent)
        out: Lines = [(f"export function __closure_{line}_{col}($cells, $defaults) {{", line)]
        if cells:
            reads = ", ".join(f'{js_name(c)} = $cells["{c}"]' for c in cells)
            out.append((f"  let {reads};", line))
        out.append(("  let $f;", line))
        out += self.function(info.node, info, "closure", 1, factory=ctx)
        return out + [("  return $f;", line), ("}", info.node.end_lineno)]

    def exported(self, module: str, name: str) -> bool:
        sym = self.env.symbols[module].get(name)
        if sym is None:
            return False
        if sym.kind == "func":
            return self.reach.function(sym.target)
        if sym.kind == "const":
            return self.reach.global_(*sym.target)
        if sym.kind == "class":
            return self.reach.class_(sym.target)
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
                elif final is not None and final.kind in ("func", "const", "class"):
                    module, name = (
                        final.target.rpartition(".")[::2] if final.kind == "class" else final.target
                    )
                    if self.exported(module, name):
                        alias = "" if name == bound else f" as {js_name(bound)}"
                        out.append(f'import {{ {js_name(name)}{alias} }} from "./{module}.js";')
        return out

    # ---------------------------------------------------------- functions
    def function(
        self,
        n: ast.FunctionDef,
        info: FuncInfo,
        kind: str = "function",
        ind: int = 0,
        tail=(),
        factory: Ctx | None = None,
    ) -> Lines:
        """n as JS. kind: "function" (a module function, exported), "method", "getter" (a
        @property), "constructor", or "closure" (a def inside a function: an arrow bound to
        its name, its non-constant defaults evaluated once, where the def runs). A method's
        first parameter is not written: it is `this`. `tail` closes the body.

        With `factory` (the closure's Ctx as setup left it, its cells bound), a closure is
        the arrow `$f` and its positional defaults are read from `$defaults`, the values
        the snapshot carries."""
        a = n.args
        if a.vararg or a.kwarg:
            self.fail(n, "*args or **kwargs")
        for d in n.decorator_list:
            if kind == "getter" and ast.unparse(d) == "property":
                continue
            if not (kind == "function" and self.env.is_registration(d, self.name)):
                self.fail(n, f"the decorator @{ast.unparse(d)}")
        method = kind in ("method", "getter", "constructor")
        if method and not a.args:
            self.fail(n, f"the method {n.name} without a self parameter")
        if kind == "getter" and len(info.params) > 1:
            self.fail(n, f"the property {n.name} with parameters")
        params, captured = [], []
        positional = {x.arg for x in a.posonlyargs + a.args}
        for p in info.params[1:] if method else info.params:
            d = info.defaults.get(p)
            if d is None:
                params.append(js_name(p))
            elif factory is not None:
                if p not in positional:
                    self.fail(d, "a keyword-only default of a closure setup made")
                params.append(f"{js_name(p)} = $defaults[{len(captured)}]")
                captured.append(None)
            elif _constant(d):
                params.append(f"{js_name(p)} = {self.e(d)}")
            elif kind == "closure":
                captured.append(self.e(d))
                params.append(f"{js_name(p)} = $d{len(captured) - 1}")
            else:
                self.fail(d, f"the default `{ast.unparse(d)}`, which is not a constant,")
        sig = ", ".join(params)
        p = "  " * ind
        name = "$f" if factory is not None else js_name(n.name)
        if kind == "closure" and captured and factory is None:
            ds = ", ".join(f"$d{i}" for i in range(len(captured)))
            head = f"{name} = (({ds}) => ({sig}) => {{"
            close = f"}})({', '.join(captured)});"
        else:
            head = {
                "function": f"export function {name}({sig}) {{",
                "method": f"{n.name}({sig}) {{",
                "getter": f"get {n.name}() {{",
                "constructor": f"constructor({sig}) {{",
                "closure": f"{name} = ({sig}) => {{",
            }[kind]
            close = "};" if kind == "closure" else "}"
        outer = self.ctx
        if factory is not None:
            self.ctx = factory
        else:
            self.ctx = Ctx(self.name, info, (), outer if kind == "closure" else None)
        try:
            local = [js_name(v) for v in info.locals if v not in info.param_types]
            out = [(f"{p}{head}", n.lineno)]
            if local:
                out.append((f"{p}  let {', '.join(local)};", n.lineno))
            out += self.block(n.body, ind + 1)
            out += [(f"{p}  {t}", n.end_lineno) for t in tail]
            return out + [(f"{p}{close}", n.end_lineno)]
        finally:
            self.ctx = outer

    # ---------------------------------------------------------- classes
    def klass(self, n: ast.ClassDef, c: ClassInfo) -> Lines:
        """An app class. Its constructor is written for a dataclass or a reached __init__;
        objects setup made arrive through the snapshot, built without it. An instance of a
        class with __call__ is a function that carries the object (runtime callable_)."""
        for b in n.bases:
            if ast.unparse(b) != "object":
                self.fail(n, f"the class {n.name} with the base class {ast.unparse(b)}")
        if n.keywords:
            self.fail(n, f"the class {n.name} with keywords")
        for d in n.decorator_list:
            if not self.env.is_dataclass(d, self.name):
                self.fail(n, f"the decorator @{ast.unparse(d)}")
            given = {k.arg for k in d.keywords} if isinstance(d, ast.Call) else set()
            if given - {"eq", "frozen", "repr", "unsafe_hash", "slots"} or (
                isinstance(d, ast.Call) and d.args
            ):
                self.fail(n, f"the decorator @{ast.unparse(d)}")
        tail = [f"return {self.rt('callable_')}(this);"] if "__call__" in c.methods else []
        out: Lines = [(f"export class {js_name(n.name)} {{", n.lineno)]
        init = next(
            (s for s in n.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None
        )
        if init is not None and self.reach.function(c.methods["__init__"]):
            if tail and any(isinstance(st, ast.Return) for st in own_statements(init.body)):
                self.fail(init, "a return in the __init__ of a class with __call__")
            info = self.env.function(c.methods["__init__"])
            out += self.function(init, info, "constructor", 1, tail)
        elif c.dataclass:
            out += self.dataclass_init(n, c, tail)
        elif tail:
            out += [
                ("  constructor() {", n.lineno),
                (f"    {tail[0]}", n.lineno),
                ("  }", n.lineno),
            ]
        shared: Lines = []
        for st in n.body:
            if isinstance(st, ast.FunctionDef):
                if st.name == "constructor":
                    self.fail(st, "a method named constructor, which JS reserves,")
                key = c.methods[st.name]
                if st.name != "__init__" and self.reach.function(key):
                    kind = "getter" if st.name in c.properties else "method"
                    out += self.function(st, self.env.function(key), kind, 1)
            elif isinstance(st, (ast.AnnAssign, ast.Assign)):
                if isinstance(st, ast.AnnAssign) and (c.dataclass or st.value is None):
                    continue  # a field
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                    self.fail(st, "this class attribute")
                # A class attribute: every instance reads it through the prototype.
                value = self.e(st.value)
                shared.append(
                    (f"{js_name(n.name)}.prototype.{targets[0].id} = {value};", st.lineno)
                )
            elif not (
                isinstance(st, ast.Pass)
                or isinstance(st, ast.Expr)
                and isinstance(st.value, ast.Constant)
            ):
                self.fail(st, FRIENDLY.get(type(st).__name__, type(st).__name__))
        return out + [("}", n.end_lineno)] + shared

    def dataclass_init(self, n: ast.ClassDef, c: ClassInfo, tail) -> Lines:
        """The constructor @dataclass generates: one parameter per field, in order. JS
        evaluates a parameter's default on every call, so a default_factory is its call."""
        params, body = [], []
        for f in c.init_fields:
            name, v = f.target.id, f.value
            default = None
            if isinstance(v, ast.Call) and self.env.from_stdlib(
                v.func, self.name, "dataclasses", "field"
            ):
                kw = {k.arg: k.value for k in v.keywords}
                if v.args or set(kw) - {"default", "default_factory"}:
                    self.fail(v, f"`{ast.unparse(v)}`")
                if "default_factory" in kw:
                    default = self.factory(kw["default_factory"])
                v = kw.get("default")
            if v is not None:
                if not _constant(v):
                    self.fail(v, f"the field default `{ast.unparse(v)}`, which is not a constant,")
                default = self.e(v)
            params.append(js_name(name) if default is None else f"{js_name(name)} = {default}")
            body.append((f"    this.{name} = {js_name(name)};", f.lineno))
        if "__post_init__" in c.methods:
            body.append(("    this.__post_init__();", n.lineno))
        body += [(f"    {t}", n.lineno) for t in tail]
        return [(f"  constructor({', '.join(params)}) {{", n.lineno), *body, ("  }", n.lineno)]

    def factory(self, fac: ast.expr) -> str:
        """A default_factory as the expression that makes one value."""
        if isinstance(fac, ast.Name) and not self.is_local(fac.id):
            if self.env.resolve(self.ctx.module, fac.id) is None and fac.id in (
                "list",
                "dict",
                "set",
            ):
                return {"list": "[]", "dict": f"{self.rt('dict')}([])", "set": "new Set()"}[fac.id]
        if isinstance(fac, ast.Lambda) and not fac.args.args:
            return f"{self.e(fac)}()"
        cls = self.class_ref(fac)
        if cls is not None:
            return self.construct(ast.Call(fac, [], []), *cls)
        return f"{self.e(fac)}()"

    def class_ref(self, n: ast.expr) -> tuple[str, str] | None:
        """(JS expression, "module.Class") when n names an app class, else None."""
        if isinstance(n, ast.Name) and not self.is_local(n.id):
            sym = self.env.resolve(self.ctx.module, n.id)
            if sym is not None and sym.kind == "class":
                return js_name(n.id), sym.target
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            if not self.is_local(n.value.id):
                outer = self.env.resolve(self.ctx.module, n.value.id)
                if outer is not None and outer.kind == "module":
                    sym = self.env.resolve(outer.target, n.attr)
                    if sym is not None and sym.kind == "class":
                        return f"{js_name(n.value.id)}.{js_name(n.attr)}", sym.target
        return None

    def construct(self, n: ast.Call, callee: str, qualname: str) -> str:
        init = self.env.classes[qualname].methods.get("__init__")
        if init is None:
            if n.args or n.keywords:
                self.fail(n, f"arguments to {qualname}(), which has no __init__,")
            return f"new {callee}()"
        return self.place(n, self.env.function(init), f"new {callee}", skip=1)

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
            if self.component_write(n, t) or self.ty(t).kind == "field":
                return [(f"{p}{self.field_inplace(n)};", line)]
            if isinstance(t, ast.Subscript) and not (_simple(t.value) and _simple(t.slice)):
                self.fail(n, f"`{ast.unparse(n)}` on a computed target")
            if not isinstance(t, (ast.Name, ast.Attribute, ast.Subscript)):
                self.fail(n, "this augmented assignment")
            if self.ty(t).kind == "vec" and self.proxy_field(t):
                # pos/rot/scale read a copy: Python's += writes the sum back through the setter.
                if not _simple(t.value):
                    self.fail(n, f"`{ast.unparse(n)}` on a computed target")
                return [(f"{p}{self.e(t)} = {self.vector_inplace(n)};", line)]
            if self.ty(t).kind == "vec":
                return [(f"{p}{self.vector_inplace(n)};", line)]
            if self.ty(t).kind == "nd":
                value = ast.unparse(ast.BinOp(t, n.op, n.value))
                self.fail(
                    n,
                    f"`{ast.unparse(n)}`, a write into a numpy array (build a new one: `{ast.unparse(t)} = {value}`)",
                )
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
        if isinstance(n, ast.FunctionDef):
            info = self.env.function(
                (self.ctx.func.key[0], f"{self.ctx.func.key[1]}.<locals>.{n.name}")
            )
            return self.function(n, info, "closure", ind)
        if isinstance(n, ast.Break):
            return [(f"{p}break;", line)]
        if isinstance(n, ast.Continue):
            return [(f"{p}continue;", line)]
        self.fail(n, FRIENDLY.get(type(n).__name__, type(n).__name__))

    def proxy_field(self, t: ast.expr) -> bool:
        """t is pos, rot or scale of an entity handle's transform."""
        return isinstance(t, ast.Attribute) and any(
            m.kind == "obj" and m.name == "manifoldx.engine._TransformProxy"
            for m in self.ty(t.value).members
        )

    def shown(self, n: ast.stmt) -> str:
        """A write as an error quotes it: `q[Velocity].v += ...`."""
        if isinstance(n, ast.AugAssign):
            return f"`{ast.unparse(n.target)} {OP_TEXT.get(type(n.op), '?')}= ...`"
        return f"`{ast.unparse(n.targets[0] if isinstance(n, ast.Assign) else n.target)} = ...`"

    def component_write(self, n: ast.stmt, t: ast.expr) -> bool:
        """Fail a write a query cannot take: to a whole component (q[X] += ...), into a column
        of a field (q[X].f[:, j] += ...), or through an event handler's read-only query. True
        when t is a field view the write may go to."""
        if isinstance(t, ast.Subscript):
            inside = self.ty(t.value)
            if inside.kind == "query":
                acc = self.ty(t)
                fields = list(component_fields(acc.name.removeprefix("ro:"))) if acc.known else []
                op = OP_TEXT.get(type(n.op), "") if isinstance(n, ast.AugAssign) else ""
                self.fail(
                    n,
                    f"{self.shown(n)}, a write to a whole component (write a field: "
                    f"{ast.unparse(t)}.{fields[0] if fields else 'field'} {op}= ...)",
                )
            if inside.kind == "field":
                self.fail(n, f"{self.shown(n)}, a write into a column of a field view,")
        ft = self.ty(t)
        if isinstance(t, ast.Attribute) and self.ty(t.value).kind == "accessor" and not ft.known:
            self.fail(t, f"`{ast.unparse(t)}`, not a field of the component,")
        if ft.kind != "field":
            return False
        if readonly(ft):
            self.fail(
                n, f"{self.shown(n)}, a write through an event handler's query, which is read-only,"
            )
        return True

    def field_inplace(self, n: ast.AugAssign) -> str:
        """`f += v` (-=, *=) on a field view queues its new rows; `/=` takes the quotient then
        sets it, as Python falls back to __truediv__ and the accessor's setter."""
        t, v = n.target, self.ty(n.value)
        method = FIELD_IOPS.get(type(n.op))
        if method is None or (method == "idiv" and not isinstance(t, ast.Attribute)):
            self.fail(n, f"{self.shown(n)} on a field view")
        if not (v.is_numeric or (is_array(v) and len(v.members) == 1)):
            self.fail(n, f"{self.shown(n)} with a {v} on the right (a number or an array)")
        return f"{self.e(t)}.{method}({self.e(n.value)})"

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
        on_query = isinstance(t, ast.Subscript) or (
            isinstance(t, ast.Attribute) and self.ty(t.value).kind == "accessor"
        )
        if on_query and self.component_write(n, t):
            v = self.ty(value)
            if not (v.is_numeric or is_array(v) or v.all("list", "tuple")):
                self.fail(n, f"{self.shown(n)} of a {v} (a number, an array or a tuple)")
            return f"{self.e(t)}.set({self.e(value)})"
        if isinstance(t, ast.Subscript) and self.ty(t.value).kind == "nd":
            self.fail(
                n, f"`{ast.unparse(n)}`, a write through an index (build a new array with np.where)"
            )
        if isinstance(t, ast.Subscript):
            return self.setitem(t, self.e(value))
        if isinstance(t, ast.Attribute) and self.ty(t.value) == STATE:
            return f"{self.e(t.value)}.set({json.dumps(t.attr)}, {self.e(value)})"
        if isinstance(t, ast.Attribute):
            self.runtime_field(t, value)
        if isinstance(t, (ast.Tuple, ast.List)):
            return f"{self.target(t)} = {self.iterable(value, destructure=True)}"
        return f"{self.target(t)} = {self.e(value)}"

    def runtime_field(self, t: ast.Attribute, value: ast.expr) -> None:
        """A write of a camera's vector or number must keep its type: numpy would keep a
        float64 array, or a tuple, where the runtime rounds to float32 (get_forward, +=)."""
        v = self.ty(value)
        for m in self.ty(t.value).members:
            if m.kind == "obj" and m.name == "manifoldx.engine._TransformProxy":
                # The proxy's setters take what np.asarray takes: an array or numbers.
                flat = v.all("list", "tuple") and elem(v).is_numeric
                if not (flat or (is_array(v) and array_rank(v) == 1)):
                    self.fail(value, f"writing a {v} to `{ast.unparse(t)}` (an array or numbers)")
                continue
            field = RUNTIME_SURFACE.get(m.name, {}).get(t.attr) if m.kind == "obj" else None
            if field is None or field.kind not in ("vec", "num"):
                continue
            same = v.all("vec") and all(x.dtype == field.dtype for x in v.members)
            if field.kind == "vec" and not same:  # every vector field is float32
                self.fail(
                    value,
                    f"writing a {v} to `{ast.unparse(t)}`, a float32 vector "
                    "(build it with np.array(x, np.float32)),",
                )
            if field.kind == "num" and not v.all("num", "int"):
                self.fail(value, f"writing a {v} to `{ast.unparse(t)}`, a number,")

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
            if self.ty(t.value) == STATE:
                self.fail(t, f"`{ast.unparse(t)}` as a loop or unpacking target")
            return self.e(t)
        self.fail(t, f"assignment to `{ast.unparse(t)}`")

    def setitem(self, t: ast.Subscript, value: str) -> str:
        if isinstance(t.slice, ast.Slice):
            self.fail(t, "assignment to a slice")
        c = self.ty(t.value)
        obj, key = self.e(t.value), self.e(t.slice)
        if c.all("dict"):
            return f"{obj}.set({key}, {value})"
        if (
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
            return f"{self.e(it)}.keys()"
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
        if self.is_self(name):
            return "this"
        if self.is_local(name):
            return js_name(name)
        sym = self.env.symbols[self.ctx.module].get(name)
        if sym is not None:
            if sym.kind == "setup" and str(sym.target).startswith("from time import "):
                self.fail(n, f"`{sym.target}` {TIME_HINT}")
            if sym.kind == "setup":
                self.fail(n, f"`{name}` (bound by `{sym.target}`, which runs only at setup)")
            if sym.kind == "lib":
                self.fail(n, f"the module {name} as a value")
            if sym.kind == "class":
                self.fail(n, f"the class {name} as a value")
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
        if t == STATE:  # a key, as hot.State reads it; a dict's own attributes come first
            if n.attr in DICT_ATTRS:
                self.fail(n, f"`{ast.unparse(n)}`, a dict method used as a value,")
            return f"{self.rt('getitem')}({self.e(n.value)}, {json.dumps(n.attr)})"
        if not t.known:  # reach could not tell which class's field to snapshot
            self.fail(
                n,
                f"`{ast.unparse(n)}`, an attribute of a value of unknown type "
                f"(annotate `{ast.unparse(n.value)}`),",
            )
        if t.kind == "nd" and n.attr == "shape":
            return f"[...{self.e(n.value)}.shape]"
        if t.kind == "accessor":
            if not self.ty(n).known:
                self.fail(
                    n,
                    f"`{ast.unparse(n)}`, which is not a field of the component (Mesh and "
                    "Material ids are not read in frame code),",
                )
            return f"{self.e(n.value)}.field({json.dumps(n.attr)})"
        if t.kind == "field":
            if n.attr != "data":
                self.fail(n, f"`{ast.unparse(n)}`, of a field view (it has .data)")
            return f"{self.e(n.value)}.data"
        if not t.all("obj", "none"):
            self.fail(n, f"the attribute .{n.attr} of a {t}")
        if self.env.methods(t, n.attr):  # a bound method keeps its object, as in Python
            return f"{self.rt('bound')}({self.e(n.value)}, {json.dumps(n.attr)})"
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
        if "field" in (a_t.kind, b_t.kind):
            return self.field_op(n, a_t, b_t)
        arrays = [m.kind for m in a_t.members + b_t.members if m.kind in ("vec", "nd")]
        if "nd" in arrays or (arrays and isinstance(op, ND_ONLY_OPS)):
            return self.nd_op(n, a_t, b_t)
        if arrays:
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

    def field_op(self, n: ast.BinOp, a_t: Ty, b_t: Ty) -> str:
        """Arithmetic with a field view, as _FieldView's operators: a float32 (k, size) array."""
        if self.ty(n).kind != "nd":
            self.fail(
                n,
                f"`{ast.unparse(n)}` on a {a_t} and a {b_t} (a field view takes + - * / with a "
                "number, an array or a field on its right, or a number on its left)",
            )
        op = VECTOR_OPS[type(n.op)]
        a, b = self.e(n.left), self.e(n.right)
        if a_t.kind == "field":
            return f"{a}.{op}({b})"
        return f"{b}.{FIELD_ROPS[op]}({a})"

    def nd_op(self, n: ast.BinOp, a_t: Ty, b_t: Ty) -> str:
        """An operator on numpy arrays of any rank: $rt.nd's function with the dtype numpy 2
        computes in, which the page cannot tell from the values (2 and 2.0 are one number)."""
        t = self.ty(n)
        if isinstance(n.op, ast.Pow):
            return self.nd_pow(n, t)
        if isinstance(n.op, ast.MatMult):
            self.no_vector_in_matmul(n, a_t, b_t)
        if type(n.op) not in ND_OPS or t.kind != "nd":
            self.fail(n, f"`{ast.unparse(n)}` on a {a_t} and a {b_t}")
        a, b = self.e(n.left), self.e(n.right)
        return f'{self.rt("nd." + ND_OPS[type(n.op)])}({a}, {b}, "{t.dtype}")'

    def no_vector_in_matmul(self, n: ast.BinOp, a_t: Ty, b_t: Ty) -> None:
        """numpy multiplies a matrix by a vector through another kernel, whose rounding order
        the page cannot match; the same product with the vector as a column matches."""
        vector = [t.kind == "vec" or (t.kind == "nd" and t.rank == 1) for t in (a_t, b_t)]
        if vector[1] and not vector[0]:
            self.fail(
                n,
                f"`{ast.unparse(n)}`, a matrix times a vector, which numpy rounds in an order the "
                "page cannot match (write the vector as a column: m @ np.array([[x], [y], [z]]))",
            )
        if any(vector):
            self.fail(
                n,
                f"`{ast.unparse(n)}`, a vector in @, which numpy rounds in an order the page cannot "
                "match (write a vector as a row, np.array([[x, y, z]]), or a column)",
            )

    def nd_pow(self, n: ast.BinOp, t: Ty) -> str:
        """numpy computes a ** 2, a ** 0.5 and a ** -1 exactly, as square, sqrt and
        reciprocal; any other power goes through powf, which the page cannot match."""
        k = _const_number(n.right)
        if t.kind != "nd" or k not in (2, 0.5, -1) or (k == -1 and t.dtype == "i32"):
            a = ast.unparse(n.left)
            self.fail(
                n,
                f"`{ast.unparse(n)}`, a power numpy does not compute exactly in the page "
                f"(2, 0.5 and -1 are in; use {a} * {a} * {a})",
            )
        a, d = self.e(n.left), f'"{t.dtype}"'
        if k == 2:
            return f"{self.rt('nd.square')}({a}, {d})"
        if k == 0.5:
            return f"{self.rt('nd.sqrt')}({a}, {d})"
        return f"{self.rt('nd.div')}(1, {a}, {d})"

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
        if t.kind == "field":
            self.fail(
                n,
                f"`{ast.unparse(n)}` on a field view (negate its data: `-{ast.unparse(n.operand)}.data`)",
            )
        if t.kind == "nd":
            if isinstance(n.op, ast.USub) and t.dtype != "bool":
                return f"{self.rt('nd.neg')}({self.e(n.operand)})"
            if isinstance(n.op, ast.Invert) and t.dtype == "bool":
                return f"{self.rt('nd.not')}({self.e(n.operand)})"
            self.fail(n, f"`{ast.unparse(n)}` on a {t}")
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
        if "field" in (lt.kind, rt_.kind):
            if self.ty(n).kind != "nd" or len(n.ops) > 1:
                self.fail(
                    n,
                    f"`{ast.unparse(n)}` (a field view compares one column, on the left, with "
                    "< <= > >=: q[X].f[:, j] < x)",
                )
            return f'{a}.{ND_COMPARE[type(op)]}({b}, "{field_compare_dtype(rt_)}")'
        if not isinstance(op, (ast.In, ast.NotIn)) and any(
            m.kind in ("vec", "nd") for m in lt.members + rt_.members
        ):
            d = nd_compare_dtype(lt, rt_) if type(op) in ND_COMPARE else None
            if d is None or len(n.ops) > 1:
                self.fail(n, f"`{ast.unparse(n)}` on a {lt} and a {rt_}")
            return f'{self.rt("nd." + ND_COMPARE[type(op)])}({a}, {b}, "{d}")'

        if isinstance(op, (ast.Is, ast.IsNot)):
            if not (lt.all("none") or rt_.all("none")):
                self.fail(n, "`is` against anything but None")
            x = b if lt.all("none") else a
            return f"{x} {'==' if isinstance(op, ast.Is) else '!='} null"
        if isinstance(op, (ast.In, ast.NotIn)):
            return f"{'!' if isinstance(op, ast.NotIn) else ''}{self.rt('contains')}({b}, {a})"
        if isinstance(op, (ast.Eq, ast.NotEq)):
            neg = isinstance(op, ast.NotEq)
            if any(
                m.kind == "obj"
                and m.name in self.env.classes
                and self.env.classes[m.name].dataclass
                for m in lt.members + rt_.members
            ):
                self.fail(n, "== on dataclass instances, which compares their fields,")
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
        """An arrow. A default is evaluated once, where the lambda is made, as in Python
        (`lambda i=i: i` keeps the loop's value): non-constant ones go through an outer arrow."""
        a = n.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
            self.fail(n, "a lambda with *args or keyword-only parameters")
        first = len(a.args) - len(a.defaults)
        params, captured = [], []
        for i, x in enumerate(a.args):
            d = a.defaults[i - first] if i >= first else None
            if d is None:
                params.append(js_name(x.arg))
            elif _constant(d):
                params.append(f"{js_name(x.arg)} = {self.e(d)}")
            else:
                captured.append(self.e(d))
                params.append(f"{js_name(x.arg)} = $d{len(captured) - 1}")
        outer, self.ctx = self.ctx, self.ctx.bind(self.env.lambda_params(n, self.ctx))
        try:
            arrow = f"(({', '.join(params)}) => {self.e(n.body)})"
        finally:
            self.ctx = outer
        if not captured:
            return arrow
        ds = ", ".join(f"$d{i}" for i in range(len(captured)))
        return f"(({ds}) => {arrow})({', '.join(captured)})"

    def e_Subscript(self, n: ast.Subscript) -> str:
        t = self.ty(n.value)
        s = n.slice
        if t.kind == "query":
            if not self.ty(n).known:
                self.fail(n, f"`{ast.unparse(n)}`, a query of what is not a component,")
            name = self.env.component_name(s, self.ctx)
            return f"{self.e(n.value)}.get({json.dumps(name)})"
        if t.kind == "field":
            if not self.ty(n).known:
                self.fail(n, f"`{ast.unparse(n)}` (a field view takes [:, j], one column)")
            return f"{self.e(n.value)}.col({s.elts[1].value})"
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
        if t.kind == "nd" or (t.kind == "vec" and self.ty(s).kind == "nd"):
            if not self.ty(n).known:
                self.fail(
                    n,
                    f"`{ast.unparse(n)}`, an index of a numpy array outside the subset "
                    "(an int, [:, j], a bool mask or an index array),",
                )
            if isinstance(s, ast.Tuple):  # [:, j]
                return f"{self.rt('nd.getitem')}({self.e(n.value)}, [null, {self.e(s.elts[1])}])"
            return f"{self.rt('nd.getitem')}({self.e(n.value)}, {self.e(s)})"
        obj = self.e(n.value)
        if t.all("dict"):  # a missing key raises KeyError, as in Python
            return f"{self.rt('getitem')}({obj}, {self.e(s)})"
        if t.all("list", "tuple", "vec"):
            k = _const_int(s)
            return f"{obj}[{k}]" if k is not None and k >= 0 else f"{obj}.at({self.e(s)})"
        return f"{self.rt('getitem')}({obj}, {self.e(s)})"

    def loops(self, n, inner) -> str:
        """The for-heads and ifs of comprehension n around inner(), emitted in its scope."""
        heads = []
        outer = self.ctx
        try:
            for g in n.generators:
                if g.is_async:
                    self.fail(n, "an async comprehension")
                head, _ = self.for_head(g.target, g.iter, declare=True)
                self.ctx = self.env.comprehension([g], self.ctx)
                heads.append((head, [self.test(c) for c in g.ifs]))
            body = inner()
        finally:
            self.ctx = outer
        for head, conds in reversed(heads):
            for c in reversed(conds):
                body = f"if ({c}) {{ {body} }}"
            body = f"{head} {{ {body} }}"
        return body

    def comprehension(self, n, wrap: str = "") -> str:
        """A comprehension as an IIFE that fills an array, eagerly; a generator expression
        is one too (the subset has no lazy generators, except under next())."""
        body = self.loops(n, lambda: f"$r.push({self.e(n.elt)});")
        iife = f"(() => {{ const $r = []; {body} return $r; }})()"
        return f"new Set({iife})" if wrap == "set" else iife

    def next_(self, n: ast.Call) -> str:
        """next(x for x in xs if ...[, default]): the first item, computed lazily as
        Python's generator does; with no item, the default or StopIteration."""
        gen = n.args[0] if n.args else None
        if not isinstance(gen, ast.GeneratorExp) or len(n.args) > 2 or n.keywords:
            self.fail(n, "next() of anything but a generator expression")
        body = self.loops(gen, lambda: f"return {self.e(gen.elt)};")
        if len(n.args) == 1:
            return f"(() => {{ {body} return {self.rt('next_')}(); }})()"
        return f"(($d) => {{ {body} return {self.rt('next_')}($d); }})({self.e(n.args[1])})"

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
        component = self.env.component_name(f, self.ctx)
        if component is not None:
            return self.component(n, component)
        if isinstance(f, ast.Name) and not self.is_local(f.id):
            sym = self.env.symbols[self.ctx.module].get(f.id)
            if sym is None:
                if f.id in BUILTINS:
                    return self.builtin(f.id, n)
                self.fail(n, f"the undefined name {f.id}")
            final = self.env.resolve(self.ctx.module, f.id)
            if final is not None and final.kind == "func":
                return self.place(n, self.env.function(final.target), js_name(f.id))
            if final is not None and final.kind == "class":
                return self.construct(n, js_name(f.id), final.target)
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
                if target is not None and target.kind == "class":
                    return self.construct(n, self.e(f), target.target)
        if isinstance(f, ast.Attribute):
            lib = self.env.lib_name(f, self.ctx)
            if lib is not None:
                return self.lib_call(lib, n)
            if self.env.is_rotation(f, self.ctx):
                return self.rotation(n)
            methods = self.env.methods(self.ty(f.value), f.attr)
            if methods:
                sigs = {(tuple(m.params[1:]), tuple(sorted(m.defaults))) for m in methods}
                if n.keywords and len(sigs) > 1:
                    self.fail(n, f"keyword arguments to .{f.attr}(), whose classes differ in it,")
                return self.place(n, methods[0], f"{self.e(f.value)}.{f.attr}", skip=1)
            if f.attr in ("sum", "min", "max", "mean") and is_array(self.ty(f.value)):
                self.fail(n, f"`{ast.unparse(n)}` (write np.{f.attr}({ast.unparse(f.value)}))")
            if f.attr == "tolist" and not (n.args or n.keywords) and is_array(self.ty(f.value)):
                return f"{self.rt('nd.tolist')}({self.e(f.value)})"
            method = self.method(n, f)
            if method is not None:
                return method
        surface = {m.name for m in self.ty(f).members if m.kind == "fn"}
        name = surface.pop() if len(surface) == 1 else None
        if name in ("manifoldx.Engine.spawn", "manifoldx.Engine.destroy"):
            self.entity_call(n, name)
        allowed = RUNTIME_KEYWORDS.get(name, ())
        if any(k.arg not in allowed for k in n.keywords):
            self.fail(n, f"keyword arguments to `{ast.unparse(f)}`, whose signature is unknown,")
        args = [self.e(a) for a in n.args]
        if n.keywords:  # a runtime function's keywords: a final object literal
            args.append("{" + ", ".join(f"{k.arg}: {self.e(k.value)}" for k in n.keywords) + "}")
        return f"{self.e(f)}({', '.join(args)})"

    def entity_call(self, n: ast.Call, name: str) -> None:
        """engine.spawn takes components and n=, an int; engine.destroy one array of entity
        indices or a bool mask (or None)."""
        if name == "manifoldx.Engine.spawn":
            for a in n.args:
                if not self.ty(a).all("component"):
                    self.fail(
                        n,
                        f"engine.spawn() of `{ast.unparse(a)}`, not a component (Mesh, "
                        "Material, Transform or a Component subclass),",
                    )
            for k in n.keywords:
                if k.arg == "n" and not self.ty(k.value).all("int"):
                    self.fail(n, f"engine.spawn(n=...) of `{ast.unparse(k.value)}`, not an int,")
            return
        t = self.ty(n.args[0]) if len(n.args) == 1 and not n.keywords else None
        ok = t is not None and all(
            m.kind in ("nd", "none") or (m.kind in ("list", "tuple") and elem(m).all("int"))
            for m in t.members
        )
        if not (ok and t.members):
            self.fail(n, "engine.destroy() of anything but entity indices, a bool mask or None")

    def component(self, n: ast.Call, name: str) -> str:
        """A component's constructor as the runtime's record of its values, by field: Transform
        takes pos, rot and scale, Mesh a geometry the export saw, Material a material, and a
        Component subclass its fields, by keyword as its __init__ does."""
        what = f"{name}()"
        builtin = {
            "Transform": ["pos", "rot", "scale"],
            "Mesh": ["geometry"],
            "Material": ["material"],
        }
        params = builtin.get(name)
        if params is None:
            params = list(component_fields(name, self.env.component_classes.get(name)))
            if n.args:
                self.fail(n, f"positional arguments to {what}, which takes its fields by keyword,")
        if len(n.args) > len(params):
            self.fail(n, f"{what} with {len(n.args)} positional arguments")
        given = dict(zip(params, n.args))
        for k in n.keywords:
            if k.arg not in params or k.arg in given:
                self.fail(n, f"the keyword {k.arg} to {what}")
            given[k.arg] = k.value
        if name in ("Mesh", "Material") and len(given) != 1:
            self.fail(n, f"{what} without its {params[0]}")
        for x in given.values():
            t = self.ty(x)
            if name == "Mesh" and not (t.members and all(m == GEOMETRY for m in t.members)):
                if isinstance(x, ast.Call):
                    self.fail(
                        n,
                        f"`{ast.unparse(n)}`, a geometry built in frame code (build it at setup "
                        "and keep it in a global: meshes built at runtime are not in the page yet),",
                    )
                self.fail(
                    n,
                    f"Mesh() of a value that is not a geometry the export saw (`{ast.unparse(x)}`)",
                )
            if name == "Material" and not (
                t.all("obj") and all(is_material(m.name) for m in t.members)
            ):
                self.fail(
                    n,
                    f"Material() of a value that is not a material made at setup (`{ast.unparse(x)}`)",
                )
            values = t.is_numeric or (is_array(t) and len(t.members) == 1)
            values = values or (t.all("list", "tuple") and elem(t).is_numeric)
            if name not in ("Mesh", "Material") and not values:
                self.fail(n, f"{what} with `{ast.unparse(x)}`, a {t}, not numbers or an array,")
        pairs = ", ".join(f"[{json.dumps(k)}, {self.e(v)}]" for k, v in given.items())
        return f"{self.rt('component')}({json.dumps(name)}, new Map([{pairs}]))"

    def lib_call(self, name: str, n: ast.Call) -> str:
        """A call into math, numpy, textwrap or random."""
        what = f"{ast.unparse(n.func)}()"
        args, kw = n.args, {k.arg for k in n.keywords}
        self.no_field(n)
        if name in ("numpy.ones", "numpy.full") or (
            name == "numpy.zeros" and self.ty(n).kind != "vec"
        ):
            return self.nd_full(name, n, what)
        if name == "numpy.arange" and self.ty(n).kind == "nd":
            if kw - {"dtype"} or len(args) != 4 - len(kw):
                self.fail(n, f"{what} with a dtype and other than start, stop and step")
            for x in args[:3]:
                self.numeric(n, x, what)
            start, stop, step = (self.e(x) for x in args[:3])
            return f'{self.rt("nd.arange")}({start}, {stop}, {step}, "{self.ty(n).dtype}")'
        if name in ND_FUNCTIONS and not (
            name == "numpy.linalg.norm" and not kw and args and self.ty(args[0]).kind == "vec"
        ):
            return self.nd_call(name, n, what)
        if name == "numpy.eye":
            if kw - {"dtype"} or len(args) != 1 or not self.ty(n).known:
                self.fail(n, f"this call to {what}")
            self.numeric(n, args[0], what)
            return f'{self.rt("nd.eye")}({self.e(args[0])}, "{self.ty(n).dtype}")'
        if name == "numpy.array" and self.ty(n).kind == "nd":
            if kw - {"dtype"} or not 1 <= len(args) + len(kw) <= 2:
                self.fail(n, f"this call to {what}")
            return f'{self.rt("nd.array")}({self.iterable(args[0])}, "{self.ty(n).dtype}")'
        if name in ("numpy.array", "numpy.zeros"):
            if kw - {"dtype"} or not 1 <= len(args) + len(kw) <= 2:
                self.fail(n, f"this call to {what}")
            d = self.env.np_dtype(n, self.ctx)
            if d not in ("f32", "f64"):
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
        if name == "textwrap.wrap":
            if kw - {"width"} or not 1 <= len(args) + len(kw) <= 2:
                self.fail(n, f"{what} with options other than width")
            given = [k.value for k in n.keywords] + args[1:]
            if not self.ty(args[0]).all("str"):
                self.fail(n, f"{what} of anything but a str")
            if given and not self.ty(given[0]).all("int", "bool"):
                self.fail(n, f"{what} with a width that is not an int")
            width = f", {self.e(given[0])}" if given else ""
            return f"{self.rt('wrap')}({self.e(args[0])}{width})"
        if name == "time.time" and not args and not kw:  # the frame's clock
            return f"{self.rt('time_time')}()"
        if name.startswith("time."):
            self.fail(n, f"`{what}` {TIME_HINT}")
        if name == "random.Random":
            self.fail(n, f"`{ast.unparse(n)}`, a generator made in frame code (make it at setup)")
        if name == "random.seed":
            self.fail(n, f"`{ast.unparse(n)}`, a reseed in frame code (seed it at setup)")
        if kw:
            self.fail(n, f"keyword arguments to {what}")
        if name in RANDOM_FUNCTIONS:  # the module's generator, loaded from the snapshot
            return f"{self.rt('random')}.{name[7:]}({', '.join(self.e(x) for x in args)})"
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

    def no_field(self, n: ast.Call) -> None:
        """A field view is not an array: numpy functions and builtins (here or in a call
        among n's arguments) take its .data."""
        for c in ast.walk(n):
            if not isinstance(c, ast.Call):
                continue
            for x in [*c.args, *(k.value for k in c.keywords)]:
                if self.ty(x).kind in ("field", "accessor"):
                    inner = f"{ast.unparse(c.func)}()"
                    self.fail(
                        n,
                        f"`{ast.unparse(x)}`, a field view, in {inner} (pass "
                        f"`{ast.unparse(x)}.data`)",
                    )

    def nd_call(self, name: str, n: ast.Call, what: str) -> str:
        """A numpy function on arrays (ND_FUNCTIONS), given the dtype numpy computes in."""
        f = name.removeprefix("numpy.")
        t = self.ty(n)
        kw = {k.arg for k in n.keywords}
        axis_kw = f in ("sum", "mean", "min", "max", "linalg.norm", "stack")
        if not t.known or kw - ({"axis"} if axis_kw else set()):
            found = ", ".join(str(self.ty(a)) for a in n.args)
            self.fail(n, f"`{ast.unparse(n)}`, {what} of {found or 'nothing'},")
        args = [self.e(a) for a in n.args]
        d = t.dtype if t.kind == "nd" else {"f32": "f32", "f64": "f64"}.get(t.dtype, "i32")
        if f in ("sum", "mean", "min", "max", "linalg.norm", "stack"):
            axis = self.env.axis(n)
            if f == "stack" and axis is None:
                axis = 0
            fn = "norm" if f == "linalg.norm" else f
            return f'{self.rt("nd." + fn)}({args[0]}, {"null" if axis is None else axis}, "{d}")'
        if f in ("argmin", "argmax") or (f == "where" and len(args) == 1):
            return f"{self.rt('nd.' + f)}({args[0]})"
        return f'{self.rt("nd." + f)}({", ".join(args)}, "{d}")'

    def rotation(self, n: ast.Call) -> str:
        """Transform.rotation(x, y, z) of numbers: its quaternion, computed in float64 as numpy
        does for Python numbers, then float32."""
        what = "Transform.rotation()"
        given = dict(zip("xyz", n.args))
        ok = len(n.args) <= 3
        for k in n.keywords:
            ok = ok and k.arg in ("x", "y", "z") and k.arg not in given
            given[k.arg] = k.value
        if not ok:
            self.fail(n, f"this call to {what} (x, y and z, numbers; euler is not in)")
        for x in given.values():
            self.numeric(n, x, what)
            if precision(self.ty(x)) == "f32":
                self.fail(n, f"{what} of a numpy float32, which numpy computes in float32,")
        args = ", ".join(self.e(given[k]) if k in given else "0" for k in "xyz")
        return f"{self.rt('quat_euler')}({args})"

    def nd_full(self, name: str, n: ast.Call, what: str) -> str:
        """np.zeros, np.ones and np.full: a float vector for an int shape, else an NdArray."""
        args, kw = n.args, {k.arg for k in n.keywords}
        given = 2 if name == "numpy.full" else 1
        if kw - {"dtype"} or not given <= len(args) <= given + 1 - len(kw):
            self.fail(n, f"this call to {what}")
        t = self.ty(n)
        if t.kind not in ("vec", "nd"):
            self.fail(
                n,
                f"`{ast.unparse(n)}`, whose shape is not an int or a tuple of ints or whose dtype "
                "is not among float32, float64, bool_ and int32,",
            )
        fill = self.e(args[1]) if name == "numpy.full" else "1" if name == "numpy.ones" else "0"
        if t.kind == "vec":
            cls = "Float32Array" if t.dtype == "f32" else "Float64Array"
            return f"new {cls}({self.e(args[0])}).fill({fill})"
        return f'{self.rt("nd.full")}({self.e(args[0])}, {fill}, "{t.dtype}")'

    def method(self, n: ast.Call, f: ast.Attribute) -> str | None:
        """A method call on a typed builtin value; None for an app object's method."""
        t = self.ty(f.value)
        objects = t.all("obj", "none") and any(m.kind == "obj" for m in t.members)
        if objects or (t == STATE and f.attr not in DICT_ATTRS):
            return None  # an object's method (None ruled out by a test), or a State key's value
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
        if t.all("dict"):
            if f.attr == "get" and len(args) in (1, 2):
                return f"{self.rt('dget')}({obj}, {', '.join(args)})"
            if f.attr == "setdefault" and len(args) in (1, 2):
                return f"{self.rt('dsetdefault')}({obj}, {', '.join(args)})"
            if f.attr in ("keys", "values", "items") and not args:
                js = "entries" if f.attr == "items" else f.attr
                return f"Array.from({obj}.{js}())"
        if t.all("str") and f.attr in STR_METHODS:
            if f.attr in ("lower", "upper") and not args:
                return f"{obj}.to{f.attr.capitalize()}Case()"
            if f.attr == "strip" and not args:
                return f"{self.rt('strip')}({obj})"
            if f.attr in ("startswith", "endswith") and len(args) == 1:
                if not self.ty(n.args[0]).all("str"):
                    self.fail(n, f".{f.attr}() of anything but a str")
                return f"{obj}.{'startsWith' if f.attr == 'startswith' else 'endsWith'}({args[0]})"
            if f.attr == "join" and len(args) == 1:
                if not elem(self.ty(n.args[0])).all("str"):
                    self.fail(n, ".join() of items that are not all str")
                return f"Array.from({self.iterable(n.args[0])}).join({obj})"
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
        if t.all("set"):
            return self.set_method(n, f, obj, args)
        self.fail(n, f"the method {t}.{f.attr}()")

    def set_method(self, n: ast.Call, f: ast.Attribute, obj: str, args: list[str]) -> str:
        """add, discard, remove, clear and update, each None as in Python. A JS Set compares
        items by identity, so an item that is not a str, number, bool or None is refused."""
        items = [elem(self.ty(a)) for a in n.args] if f.attr == "update" else map(self.ty, n.args)
        for it in items:
            if it.known and not it.is_scalar:
                self.fail(n, f"`{ast.unparse(n)}`, .{f.attr}() of a {it},")
        if f.attr == "add" and len(args) == 1:
            return f"({obj}.add({args[0]}), null)"
        if f.attr == "discard" and len(args) == 1:
            return f"({obj}.delete({args[0]}), null)"
        if f.attr == "remove" and len(args) == 1:
            return f"{self.rt('sremove')}({obj}, {args[0]})"
        if f.attr == "clear" and not args:
            return f"({obj}.clear(), null)"
        if f.attr == "update":
            its = ", ".join(self.iterable(a) for a in n.args)
            return f"{self.rt('supdate')}({obj}{', ' if its else ''}{its})"
        self.fail(n, f"the method {self.ty(f.value)}.{f.attr}()")

    def place(self, n: ast.Call, info: FuncInfo, callee: str, skip: int = 0) -> str:
        """Arguments placed by the callee's signature; keywords become positions. `skip`
        leaves out a method's self."""
        label = key_label(info.key)
        a = info.node.args
        params = info.params[skip:]
        positional = len(a.posonlyargs) + len(a.args) - skip
        if len(n.args) > positional:
            self.fail(n, f"a call to {label} with {len(n.args)} positional arguments")
        slots: list[ast.expr | None] = [*n.args] + [None] * (len(params) - len(n.args))
        for k in n.keywords:
            if (
                k.arg not in params
                or params.index(k.arg) < len(n.args)
                or (k.arg in [x.arg for x in a.posonlyargs])
            ):
                self.fail(n, f"the keyword {k.arg} to {label}")
            slots[params.index(k.arg)] = k.value
        for p, s in zip(params, slots):
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
        self.no_field(n)
        if name in ("min", "max"):
            if set(kw) - {"key"} or not args:
                self.fail(n, f"this call to {name}()")
            key = self.e(kw["key"]) if "key" in kw else "null"
            if len(args) == 1:
                return f"{self.rt(name + '_')}({self.e(args[0])}, {key})"
            if "key" not in kw and all(self.ty(x).is_numeric for x in args):
                return f"Math.{name}({', '.join(self.e(x) for x in args)})"
            return f"{self.rt(name + '_')}({self.items(args)}, {key})"
        if name == "next":
            return self.next_(n)
        if kw:
            self.fail(n, f"keyword arguments to {name}()")
        if name == "hasattr" and len(args) == 2:
            return self.hasattr_(n)
        if name == "isinstance" and len(args) == 2:
            return self.isinstance_(n)
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
            if t.kind == "query":
                return f"{self.e(x)}.length"
            if t.kind == "nd":
                return f"{self.e(x)}.shape[0]"
            if t.all("list", "tuple", "vec"):
                return f"{self.e(x)}.length"
            if t.all("set"):
                return f"{self.e(x)}.size"
            if t.all("str"):  # code points, as Python counts
                return f"{self.rt('cplen')}({self.e(x)})"
            return f"{self.rt('len')}({self.e(x)})"
        if name == "str":
            return self.to_str(n, x)
        if name == "bool":
            return self.test(x)
        if name in ("any", "all"):
            return f"{self.rt(name + '_')}({self.e(x)})"
        if name == "abs" and t.kind == "nd":
            return f"{self.rt('nd.abs')}({self.e(x)})"
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

    def hasattr_(self, n: ast.Call) -> str:
        x, attr = n.args
        t = self.ty(x)
        if not t.known:
            self.fail(
                n,
                f"hasattr() of `{ast.unparse(x)}`, of unknown type (annotate `{ast.unparse(x)}`),",
            )
        if not t.all("obj", "none"):
            self.fail(n, f"hasattr() of a {t}")
        if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
            self.fail(n, "hasattr() with a computed name")
        return f"{self.rt('hasattr_')}({self.e(x)}, {json.dumps(attr.value)})"

    def isinstance_(self, n: ast.Call) -> str:
        """isinstance against the app's own classes, whose instances are built on the
        transpiled prototypes."""
        x, classes = n.args
        refs = []
        for c in classes.elts if isinstance(classes, ast.Tuple) else [classes]:
            ref = self.class_ref(c)
            if ref is None:
                self.fail(n, f"isinstance() against `{ast.unparse(c)}`, which is not an app class,")
            refs.append(ref[0])
        if len(refs) == 1:
            return f"({self.e(x)} instanceof {refs[0]})"
        return f"{self.rt('isinstance_')}({self.e(x)}, [{', '.join(refs)}])"

    def to_str(self, n: ast.expr, x: ast.expr, spec: str = "str()") -> str:
        """str(x), or x in an f-string with no format spec. JS cannot tell 3.0 from 3, so
        the type decides: str_ writes a float as Python's repr and an int as digits."""
        t = self.ty(x)
        if not t.known:
            self.fail(
                n, f"{spec} of `{ast.unparse(x)}`, of unknown type (annotate `{ast.unparse(x)}`),"
            )
        if t.all("str"):
            return self.e(x)
        if not t.all("str", "int", "bool", "none", "num"):
            self.fail(n, f"{spec} of a {t}")
        kinds = {m.kind for m in t.members}
        if {"num", "int"} <= kinds:
            self.fail(
                n,
                f"{spec} of `{ast.unparse(x)}`, which may be an int or a float, written "
                f"differently (annotate `{ast.unparse(x)}`),",
            )
        if spec == "str()" and any(m.kind == "num" and m.dtype == "f32" for m in t.members):
            self.fail(n, "str() of a numpy float32, which numpy writes its own way,")
        mode = "float" if "num" in kinds else "int"
        return f'{self.rt("str_")}({self.e(x)}, "{mode}")'

    def e_JoinedStr(self, n: ast.JoinedStr) -> str:
        parts = []
        for v in n.values:
            if isinstance(v, ast.Constant):
                parts.append(json.dumps(v.value))
                continue
            if v.conversion != -1:
                self.fail(v, f"the conversion !{chr(v.conversion)}")
            if v.format_spec is None:
                parts.append(self.to_str(v, v.value, "an f-string field"))
                continue
            if not all(isinstance(x, ast.Constant) for x in v.format_spec.values):
                self.fail(v, "a computed format spec")
            spec = "".join(x.value for x in v.format_spec.values)
            if not re.fullmatch(r"\.\d+f", spec):
                self.fail(v, f"the format spec :{spec}")
            self.numeric(v, v.value, f"the format :{spec}")
            parts.append(f"{self.rt('fmt')}({self.e(v.value)}, {json.dumps(spec)})")
        if not parts:
            return '""'
        return parts[0] if len(parts) == 1 else "(" + " + ".join(parts) + ")"
