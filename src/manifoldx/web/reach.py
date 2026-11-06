"""What the browser runs: the roots an engine holds, and what they reach through types.

The roots are the functions the engine calls every frame or on an event: its systems, its
event handlers (except startup, which runs at export, and manifoldx's own), and the getters
of the GUI's ValueDisplays. Reach follows calls and attribute reads through types: a method
is reached when reached code calls it on a value whose type allows its class, never by its
name alone. What reached code reads of each app class (`fields`) is what the snapshot
writes; everything not reached is setup and runs in Python at export.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

from manifoldx.gltf.engine_export import _internal, _label, _widgets

from .types import (
    NUM,
    Ctx,
    FuncInfo,
    FunctionKey,
    LiveTypes,
    TranspileError,
    TypeEnv,
    key_label,
    module_path,
    target_names,
)


@dataclass(frozen=True)
class Root:
    """A function the engine calls. kind: "system" (called with its query and dt),
    "handler" (with the event's payload; `event` names it) or "getter" (a ValueDisplay's,
    with nothing). `label` names it in errors and reports."""

    kind: str
    label: str
    func: object
    event: str | None = None

    @property
    def function(self):
        """The plain function: a bound method's __func__."""
        return getattr(self.func, "__func__", self.func)

    @property
    def owner(self):
        """A bound method's object, which the snapshot starts from; None otherwise."""
        return getattr(self.func, "__self__", None)

    @property
    def cells(self) -> dict[str, object]:
        """A closure's free variables and the values its cells hold, which the snapshot
        starts from; {} for a function that closes over nothing."""
        f = self.function
        code, closure = getattr(f, "__code__", None), getattr(f, "__closure__", None)
        if code is None or not closure:
            return {}
        return {name: c.cell_contents for name, c in zip(code.co_freevars, closure)}

    @property
    def defaults(self) -> tuple:
        """The parameter defaults the function was made with (a lambda's `i=i`)."""
        return getattr(self.function, "__defaults__", None) or ()


def find_roots(engine) -> list[Root]:
    """The engine's systems in order, its handlers by event (not startup, not manifoldx's
    own, the rule the glTF export's report uses) and each ValueDisplay's getter, walking the
    GUI's panels and their children."""
    from manifoldx.gui import ValueDisplay

    roots = [Root("system", f"system {_label(s.func)}", s.func) for s in engine.systems._systems]
    for event, handlers in engine._event_bus._handlers.items():
        if event == "startup":
            continue
        for h in handlers:
            if not _internal(h.func):
                roots.append(Root("handler", f'on("{event}") {_label(h.func)}', h.func, event))
    for panel in engine.gui._panels:
        for w in _widgets(panel):
            if isinstance(w, ValueDisplay):
                g = w._getter
                roots.append(Root("getter", f"ValueDisplay.getter {_label(g)}", g))
    return roots


class Reach:
    """The functions, classes, module globals and fields the roots reach, typed together:
    reach decides which call sites the types join, and types decide which methods a call
    reaches, so the two run to a fixed point.

    roots: Root objects, or "module.function" strings (a function of the sources, with no
    live owner); None reaches everything, as the parity tests' whole modules.

    functions: set of FunctionKey · classes: set of "module.Class" · globals: set of
    (module, name) · fields: {"module.Class": set of the field names reached code reads}
    · chain(key): the root's label, then each function from the root down to key."""

    def __init__(self, roots, env: TypeEnv):
        self.env = env
        self.everything = roots is None
        self.roots: list[Root | str] = list(roots or [])
        self.functions: set[FunctionKey] = set()
        self.classes: set[str] = set()
        self.globals: set[tuple[str, str]] = set()
        self.fields: dict[str, set[str]] = {}
        self._via: dict[object, FunctionKey | None] = {}  # what reached a thing first
        self._root: dict[FunctionKey, str] = {}  # a root function's label
        if self.everything:
            self.functions = set(env.functions)
            self.classes = set(env.classes)
            env.solve()
            return
        if env.live is None:
            env.live = LiveTypes(env.modules)
        self._order: list[FunctionKey] = []
        self._kinds: dict[FunctionKey, Root | str] = {}
        for r in self.roots:
            key = self._root_key(r)
            self._root[key] = r if isinstance(r, str) else r.label
            self._kinds[key] = r
            self._add(key, None)
        while True:
            env.solve(self._order, report=False)
            n = (len(self.functions), len(self.classes), len(self.globals))
            self._expand(check=False)
            if (len(self.functions), len(self.classes), len(self.globals)) == n:
                break
        self._expand(check=True)
        env.solve(self._order, root_of=self.root)

    # ---------------------------------------------------------- roots
    def _root_key(self, r: Root | str) -> FunctionKey:
        if isinstance(r, str):
            return self._label_key(r)
        f = r.function
        module, qualname = getattr(f, "__module__", None), getattr(f, "__qualname__", None)
        if module not in self.env.modules or qualname is None:
            raise TranspileError(
                "<engine>", 0, f"the root {r.label}, which is not a function of the app's modules,"
            )
        if f.__name__ == "<lambda>":
            key = self._lambda(module, qualname, f, r.label)
        else:
            key = (module, qualname)
            if key not in self.env.functions:
                raise TranspileError(
                    module_path(module),
                    f.__code__.co_firstlineno,
                    f"the root {r.label}, which is not a plain def or lambda,",
                )
        self._observe(r, key)
        return key

    def _label_key(self, label: str) -> FunctionKey:
        module, _, name = label.rpartition(".")
        while module and module not in self.env.modules:
            module, _, head = module.rpartition(".")
            name = f"{head}.{name}"
        if (module, name) not in self.env.functions:
            raise ValueError(f"no module-level function {label} in the sources")
        return (module, name)

    def _lambda(self, module: str, qualname: str, f, label: str) -> FunctionKey:
        """Find f's lambda in its module by line, and among several on that line by where
        its code's instructions sit."""
        code = f.__code__
        line = code.co_firstlineno
        found = [
            n
            for n in ast.walk(self.env.modules[module])
            if isinstance(n, ast.Lambda) and n.lineno == line
        ]
        if not found:
            raise TranspileError(module_path(module), line, f"the root {label}, not found,")
        spots = [(a, c) for a, _, c, _ in code.co_positions() if a is not None and c is not None]

        def inside(n: ast.Lambda) -> int:
            b = n.body
            lo, hi = (b.lineno, b.col_offset), (b.end_lineno, b.end_col_offset)
            return sum(lo <= s <= hi for s in spots)

        n = max(found, key=inside)
        return self.env.register_lambda(module, qualname, n).key

    def _observe(self, r: Root, key: FunctionKey) -> None:
        """Type a root from what it holds: its owner's fields, its cells, its defaults; a
        system's dt is a number."""
        live = self.env.live
        info = self.env.function(key)
        if r.owner is not None:
            live.observe(r.owner)
        cells = r.cells
        if cells:
            self.env.cells[key] = {k: live.observe(v) for k, v in cells.items()}
        namespace = r.function.__globals__
        for name, sym in self.env.symbols[key[0]].items():
            if sym.kind == "const" and name in namespace:
                live.observe_global(key[0], name, namespace[name])
        positional = info.params[: len(info.node.args.posonlyargs) + len(info.node.args.args)]
        if inspect.ismethod(r.func):
            positional = positional[1:]
        for p, d in zip(positional[len(positional) - len(r.defaults) :], r.defaults):
            if not info.declared[p].known:
                info.declared[p] = live.observe(d)
        if r.kind == "system" and len(positional) >= 2 and not info.declared[positional[1]].known:
            info.declared[positional[1]] = NUM

    # ---------------------------------------------------------- reach
    def _add(self, key: FunctionKey, via: FunctionKey | None) -> None:
        if key in self.functions:
            return
        self.functions.add(key)
        self._order.append(key)
        self._via[key] = via
        cls = self.env.function(key).cls
        if cls is not None:
            self._class(cls, key)

    def _class(self, name: str, via: FunctionKey) -> None:
        if name in self.env.classes and name not in self.classes:
            self.classes.add(name)
            self._via[("class", name)] = via

    def _symbol(self, sym, via: FunctionKey) -> None:
        if sym is None:
            return
        if sym.kind == "func":
            self._add(sym.target, via)
        elif sym.kind == "class":
            self._class(sym.target, via)
        elif sym.kind == "const" and sym.target not in self.globals:
            self.globals.add(sym.target)
            self._via[("global", *sym.target)] = via
            value = self._global_value(*sym.target)
            if value is not None:
                module = sym.target[0]
                for n in ast.walk(value):
                    if isinstance(n, ast.Name):
                        self._symbol(self.env.resolve(module, n.id), via)

    def _global_value(self, module: str, name: str) -> ast.expr | None:
        for n in self.env.modules[module].body:
            if isinstance(n, ast.Assign) and any(name in target_names(t) for t in n.targets):
                return n.value
            if isinstance(n, ast.AnnAssign) and name in target_names(n.target):
                return n.value
        return None

    def _expand(self, check: bool) -> None:
        """One walk over every reached function (the list grows as it goes)."""
        i = 0
        while i < len(self._order):
            key = self._order[i]
            i += 1
            info = self.env.function(key)
            if check:
                self._query(key, info)
            for n, ctx in self.env.walk(info):
                self._node(n, ctx, key, check)

    def _node(self, n: ast.AST, ctx: Ctx, key: FunctionKey, check: bool) -> None:
        env = self.env
        if isinstance(n, ast.Name) and not env.is_local(n.id, ctx):
            self._symbol(env.resolve(ctx.module, n.id), key)
        elif isinstance(n, ast.Attribute):
            if env.lib_name(n, ctx) is not None:
                return
            if isinstance(n.value, ast.Name) and not env.is_local(n.value.id, ctx):
                sym = env.resolve(ctx.module, n.value.id)
                if sym is not None and sym.kind == "module":
                    self._symbol(env.resolve(sym.target, n.attr), key)
                    return
            self._attribute(
                n, env.expr(n.value, ctx), n.attr, key, check, isinstance(n.ctx, ast.Load)
            )
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute):
            self._attribute(
                n.target, env.expr(n.target.value, ctx), n.target.attr, key, check, True
            )
        elif isinstance(n, ast.Call):
            for callee, _ in env.callees(n, ctx):
                self._add(callee.key, key)
            f = n.func
            if (
                isinstance(f, ast.Name)
                and f.id == "hasattr"
                and env.resolve(ctx.module, "hasattr") is None
                and len(n.args) == 2
                and isinstance(n.args[1], ast.Constant)
                and isinstance(n.args[1].value, str)
            ):
                self._attribute(n, env.expr(n.args[0], ctx), n.args[1].value, key, check, True)

    def _attribute(self, n, recv, attr: str, key: FunctionKey, check: bool, read: bool) -> None:
        """recv.attr: the method or property of each class recv may be, or the field."""
        for m in recv.members:
            if m.kind != "obj":
                continue
            c = self.env.classes.get(m.name)
            if c is None:
                if check and not (m.name or "").startswith("manifoldx."):
                    raise TranspileError(
                        module_path(key[0]),
                        n.lineno,
                        f"`{ast.unparse(n)}` on a {m.name}, a library object the browser "
                        "does not have,",
                        self.root(key),
                    )
                continue
            self._class(m.name, key)
            if attr in c.methods:
                self._add(c.methods[attr], key)
            elif read:
                self.fields.setdefault(m.name, set()).add(attr)
                for t in self.env.attribute(m, attr).members:
                    if t.kind == "obj":
                        self._class(t.name, key)

    def _query(self, key: FunctionKey, info: FuncInfo) -> None:
        """A system's first parameter is its component query, which the browser has not."""
        r = self._kinds.get(key)
        if not isinstance(r, Root) or r.kind != "system":
            return
        params = info.params[1:] if info.cls else info.params
        if not params:
            return
        q = params[0]
        for n, ctx in self.env.walk(info):
            if isinstance(n, ast.Name) and n.id == q:
                binder = ctx.binder(q)
                if (
                    binder is not None
                    and binder.func is info
                    and not any(k == q for k, _ in binder.extra)
                ):
                    raise TranspileError(
                        module_path(key[0]),
                        n.lineno,
                        f"`{q}`, the system's component query (component queries are not "
                        "supported in the browser yet),",
                        self.root(key),
                    )

    # ---------------------------------------------------------- reports
    def chain(self, key) -> list[str]:
        """How a function (or ("class", name), ("global", module, name)) was reached: the
        root's label, then each function after the root down to it."""
        out = []
        k = key if key in self.functions else self._via.get(key)
        while k is not None and k not in self._root:
            out.append(key_label(k))
            k = self._via.get(k)
        return ([self._root[k]] if k is not None else []) + out[::-1]

    def root(self, key) -> str | None:
        """chain(key) as one line, for an error's "reached from"."""
        if self.everything:
            return None
        return " -> ".join(self.chain(key)) or None

    # What the emitter asks.
    def function(self, key: FunctionKey) -> bool:
        return self.everything or key in self.functions

    def global_(self, module: str, name: str) -> bool:
        return self.everything or (module, name) in self.globals

    def class_(self, qualname: str) -> bool:
        return self.everything or qualname in self.classes
