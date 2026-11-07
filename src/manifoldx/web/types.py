"""The web transpiler's type model.

A `Ty` is what the transpiler knows about a value; it decides how an operator, a test or a
call becomes JavaScript (`[] or x` needs `or_`, `a + b` on lists concatenates, `len` of a
str counts code points). `TypeEnv` holds the app's modules and types every function's
locals and returns, from annotations first, then by inference over assignments.
Types come only from those sources, never from a name.

Later steps feed it more: live types from the objects setup built, and argument types
joined at call sites (`solve`).
"""

from __future__ import annotations

import ast
import fnmatch
import sys
import types
from dataclasses import dataclass, field

import numpy as np

KINDS = frozenset(
    {
        "num",
        "int",
        "bool",
        "str",
        "none",
        "list",
        "tuple",
        "dict",
        "set",
        "vec",
        "nd",
        "obj",
        "fn",
        "query",
        "accessor",
        "field",
        "component",
        "union",
        "unknown",
    }
)


@dataclass(frozen=True)
class Ty:
    """A type. `args` holds the element types: list[elem], set[elem], tuple[*items],
    dict[key, value], union[*members], fn[return] (empty when the return is unknown).
    `dtype` is a vec's "f64" or "f32", an nd's "f64", "f32", "bool" or "i32". An nd is a
    numpy array of `rank` dimensions (a bare vec is rank 1 of a float). A num carries one when it is a numpy scalar, which
    numpy keeps at its precision ("f32": np.float32, float32 arithmetic; "f64":
    np.float64); a Python float or int has none and takes the other operand's precision
    (NEP 50). `name` is an obj's class ("module.Qualname") or a
    fn's function ("module.qualname"), None for a lambda, or "manifoldx.hot.State" for
    engine.state's dict (STATE).

    A union with no members is the empty type: the elements of `[]`, or a variable before
    its first assignment. Joining it with anything gives the other side.
    """

    kind: str
    args: tuple[Ty, ...] = ()
    dtype: str | None = None
    name: str | None = None
    rank: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown type kind {self.kind!r}")
        if self.kind == "vec" and self.dtype not in ("f64", "f32"):
            raise ValueError(f"a vec's dtype is f64 or f32, not {self.dtype!r}")
        if self.kind == "nd" and (self.dtype not in ND_DTYPES or not self.rank):
            raise ValueError(f"an nd has a dtype among {ND_DTYPES} and a rank, not {self.dtype!r}")
        if self.kind == "num" and self.dtype not in (None, "f64", "f32"):
            raise ValueError(f"a num's dtype is None, f64 or f32, not {self.dtype!r}")
        if self.kind not in ("vec", "num", "nd") and self.dtype is not None:
            raise ValueError(f"a {self.kind} has no dtype")
        if self.kind != "nd" and self.rank is not None:
            raise ValueError(f"a {self.kind} has no rank")

    def __str__(self) -> str:
        k, a = self.kind, self.args
        if k == "union":
            return " | ".join(map(str, a)) if a else "never"
        if k in ("list", "set", "tuple", "dict"):
            return f"{k}[{', '.join(map(str, a))}]"
        if k == "vec" or (k == "num" and self.dtype):
            return f"{k}[{self.dtype}]"
        if k == "nd":
            return f"nd[{self.dtype}, {self.rank}]"
        if k in ("obj", "fn") and self.name:
            return f"{k} {self.name}"
        return k

    @property
    def members(self) -> tuple[Ty, ...]:
        """The members of a union, or the type itself."""
        return self.args if self.kind == "union" else (self,)

    def all(self, *kinds: str) -> bool:
        """Every member has one of `kinds` (False for the empty type)."""
        return bool(self.members) and all(m.kind in kinds for m in self.members)

    @property
    def known(self) -> bool:
        return self.kind != "unknown"

    @property
    def is_bool(self) -> bool:
        return self.all("bool")

    @property
    def is_numeric(self) -> bool:
        return self.all("num", "int", "bool")

    @property
    def is_scalar(self) -> bool:
        return self.all("num", "int", "bool", "str", "none")


ND_DTYPES = ("f64", "f32", "bool", "i32")
NUM, INT, BOOL, STR, NONE = Ty("num"), Ty("int"), Ty("bool"), Ty("str"), Ty("none")
UNKNOWN = Ty("unknown")
NEVER = Ty("union")
NUM32, NUM64 = Ty("num", dtype="f32"), Ty("num", dtype="f64")
# engine.state, a hot.State: a dict with str keys whose keys also read and write as
# attributes (`engine.state.cam`), which the transpiler turns into getitem and setitem.
STATE = Ty("dict", (STR, UNKNOWN), name="manifoldx.hot.State")


def list_of(elem: Ty) -> Ty:
    return Ty("list", (elem,))


def set_of(elem: Ty) -> Ty:
    return Ty("set", (elem,))


def tuple_of(*items: Ty) -> Ty:
    return Ty("tuple", tuple(items))


def dict_of(key: Ty, value: Ty) -> Ty:
    return Ty("dict", (key, value))


def vec(dtype: str = "f64") -> Ty:
    return Ty("vec", dtype=dtype)


def nd(dtype: str, rank: int) -> Ty:
    return Ty("nd", dtype=dtype, rank=rank)


def is_array(t: Ty) -> bool:
    """t is one numpy array: a vec or an nd."""
    return t.kind in ("vec", "nd")


def array_rank(t: Ty) -> int:
    """The rank of an operand: an nd's, 1 for a vec, 0 for a number."""
    return t.rank if t.kind == "nd" else 1 if t.kind == "vec" else 0


def nd_dtype(op: ast.AST, a: Ty, b: Ty) -> str | None:
    """The dtype numpy 2 gives `a op b` when an array is on a side (NEP 50), or None when
    the subset has no such operation: float32 with float32 or a Python number stays float32,
    float64 anywhere is float64, int32 with a Python int is int32 (but / gives float64), with
    a Python float or a float32 it is float64; a bool array takes no arithmetic, and a side of
    several types gives None."""
    kinds = []
    for t in (a, b):
        if len(t.members) != 1 or not t.known:
            return None
        if t.kind in ("vec", "nd"):
            kinds.append(("array", t.dtype))
        elif t.kind == "num" and t.dtype:
            kinds.append(("scalar", t.dtype))  # a numpy scalar keeps its dtype
        elif t.kind == "num":
            kinds.append(("py", "float"))
        elif t.kind in ("int", "bool"):
            kinds.append(("py", "int"))
        else:
            return None
    dtypes = {d for k, d in kinds if k != "py"}
    if not dtypes:  # two Python numbers, as np.where(m, 1.0, 0) sees them
        return "f64" if ("py", "float") in kinds else "i32"
    if "bool" in dtypes:
        return None
    if "f64" in dtypes:
        return "f64"
    if "i32" in dtypes:
        if "f32" in dtypes or ("py", "float") in kinds or isinstance(op, ast.Div):
            return "f64"
        return "i32"
    return "f32"


def nd_compare_dtype(a: Ty, b: Ty) -> str | None:
    """The dtype numpy compares a and b in (both cast to it), or None outside the subset: as
    arithmetic, with two bool arrays compared as bool."""
    if a.kind in ("vec", "nd") and b.kind in ("vec", "nd") and a.dtype == b.dtype == "bool":
        return "bool"
    return nd_dtype(ast.Add(), a, b)


def field_compare_dtype(b: Ty) -> str | None:
    """The dtype a float32 field's rows compare against b in, numpy 2's."""
    if b.kind == "field":
        return "f32"
    return nd_compare_dtype(nd("f32", 2), b)


def _vague(t: Ty) -> bool:
    """t is unknown or holds an unknown (list[unknown], the elements of a list of calls)."""
    return not t.known or any(_vague(a) for a in t.args)


def is_material(name: str | None) -> bool:
    """name ("manifoldx.resources.PhongMaterial") is a material class: the page builds its
    materials once from the glTF, so their fields do not change at runtime."""
    from manifoldx import resources

    module, _, cls = (name or "").rpartition(".")
    c = getattr(resources, cls, None) if module == "manifoldx.resources" else None
    return isinstance(c, type) and issubclass(c, resources.Material)


def obj(name: str) -> Ty:
    return Ty("obj", name=name)


# A system's component query, q[X] its accessor of component X, and an accessor's attribute
# a field view: columns [start, start + size) of X's rows for the query's entities. Each is
# read-only when the query is an event handler's. The name carries what the type knows.
def query(components, readonly: bool = False) -> Ty:
    return Ty("query", name=("ro:" if readonly else "") + ",".join(components))


def accessor(component: str, readonly: bool = False) -> Ty:
    return Ty("accessor", name=("ro:" if readonly else "") + component)


def field_view(component: str, name: str, start: int, size: int, readonly: bool = False) -> Ty:
    return Ty("field", name=f"{'ro:' if readonly else ''}{component}.{name}:{start}:{size}")


def readonly(t: Ty) -> bool:
    """A query, accessor or field of an event handler's query, which writes nothing."""
    return (t.name or "").startswith("ro:")


def query_components(t: Ty) -> list[str]:
    names = t.name.removeprefix("ro:")
    return names.split(",") if names else []


def field_parts(t: Ty) -> tuple[str, str, int, int]:
    """A field's (component, field name, start column, size)."""
    head, start, size = t.name.removeprefix("ro:").rsplit(":", 2)
    component, _, name = head.partition(".")
    return component, name, int(start), int(size)


# The fields a field view reads of each built-in component, under every spelling the
# accessor takes; a Component subclass's come from its _layout.
TRANSFORM_FIELDS = {
    "pos": (0, 3),
    "position": (0, 3),
    "rot": (3, 4),
    "rotation": (3, 4),
    "scale": (7, 3),
}


def component_fields(name: str, cls=None) -> dict[str, tuple[int, int]]:
    """{field: (col, size)} of the store's component `name` (its class `cls` when known): a
    built-in's, a Component subclass's _layout, an @engine.component class's column table."""
    from manifoldx.components import _COMPONENT_CLASSES, Material, Mesh
    from manifoldx.ecs import _COMPONENT_REGISTRY

    if name == "Transform":
        return dict(TRANSFORM_FIELDS)
    builtin = {"Mesh": Mesh, "Material": Material}.get(name) or cls or _COMPONENT_CLASSES.get(name)
    if builtin is not None and hasattr(builtin, "_layout"):
        return dict(builtin._layout)
    decorated = _COMPONENT_REGISTRY.get(name)
    return dict(getattr(decorated, "_component_start_idx", {}))


# A geometry: a dict with a 2-d "positions" array, or what has to_geometry() (a
# modeling.Mesh). It crosses by its id in the engine's geometry registry.
GEOMETRY = obj("manifoldx.Geometry")


def is_geometry(v) -> bool:
    if isinstance(v, dict):
        positions = v.get("positions")
        return isinstance(positions, np.ndarray) and positions.ndim == 2
    return not isinstance(v, type) and hasattr(v, "to_geometry")


def fn(name: str | None = None, returns: Ty | None = None) -> Ty:
    return Ty("fn", (returns,) if returns is not None else (), name=name)


def union(*tys: Ty) -> Ty:
    """The join of `tys`: flattened, without repeats; unknown absorbs everything.
    Two lists (sets, dicts) join their element types; tuples of one length join item-wise."""
    out: list[Ty] = []
    for t in tys:
        for m in t.members:
            if m.kind == "unknown":
                return UNKNOWN
            for i, o in enumerate(out):
                merged = _merge(o, m)
                if merged is not None:
                    out[i] = merged
                    break
            else:
                out.append(m)
    return out[0] if len(out) == 1 else Ty("union", tuple(out))


def _merge(a: Ty, b: Ty) -> Ty | None:
    """a and b as one type when they have the same shape, else None."""
    if a == b:
        return a
    if a.kind != b.kind or a.kind in ("vec", "nd", "obj"):
        return None
    if a.kind in ("list", "set", "dict") or (a.kind == "tuple" and len(a.args) == len(b.args)):
        return Ty(a.kind, tuple(union(x, y) for x, y in zip(a.args, b.args)))
    if a.kind == "fn" and a.name == b.name:
        return fn(a.name, union(*a.args, *b.args) if a.args or b.args else None)
    return None


def join(a: Ty, b: Ty) -> Ty:
    return union(a, b)


def precision(t: Ty) -> str | None:
    """The numpy precision arithmetic on t's numbers runs at: "f32" when some member is
    float32 and none float64, "f64" when some is float64 and none float32, None for Python
    numbers only, "mixed" when a member is each (the result depends on the value)."""
    found = {m.dtype for m in t.members if m.kind in ("num", "vec") and m.dtype}
    return "mixed" if len(found) > 1 else (found.pop() if found else None)


def arith_dtype(a: Ty, b: Ty) -> str | None:
    """The precision of a + b (or -, *, /) on numbers or vecs, numpy 2's rules: two float32
    sides or float32 against a Python number stay float32; float64 anywhere wins."""
    pa, pb = precision(a), precision(b)
    if "mixed" in (pa, pb):
        return "mixed"
    if "f64" in (pa, pb):
        return "f64"
    return "f32" if "f32" in (pa, pb) else None


def elem(t: Ty) -> Ty:
    """The type of the items iterating a value of type t yields."""
    out = []
    for m in t.members:
        if m.kind in ("list", "set"):
            out.append(m.args[0])
        elif m.kind == "tuple":
            out.append(union(*m.args))
        elif m.kind == "dict":
            out.append(m.args[0])
        elif m.kind == "str":
            out.append(STR)
        elif m.kind == "vec":
            out.append(Ty("num", dtype=m.dtype))
        elif m.kind == "nd":
            out.append(nd_item(m))
        elif m.kind == "query":  # iterating a query gives its entity indices
            out.append(INT)
        else:
            return UNKNOWN
    return union(*out)


def nd_item(t: Ty) -> Ty:
    """What indexing an nd's first axis by an int gives: an nd of one rank less, or a
    number of its dtype."""
    if t.rank > 1:
        return nd(t.dtype, t.rank - 1)
    return {"f32": NUM32, "f64": NUM64, "i32": INT, "bool": BOOL}[t.dtype]


def nd_list(t: Ty) -> Ty:
    """The type of an array's .tolist(): nested lists of Python numbers."""
    if t.kind == "vec" or t.rank == 1:
        return list_of({"i32": INT, "bool": BOOL}.get(t.dtype, NUM))
    return list_of(nd_list(nd(t.dtype, t.rank - 1)))


# ---------------------------------------------------------------- live types


def _library(module: str) -> bool:
    """A class from this module is a library's, not the app's: its fields are not walked."""
    root = module.partition(".")[0]
    if root in sys.stdlib_module_names or root in ("manifoldx", "numpy", "builtins"):
        return True
    path = getattr(sys.modules.get(module), "__file__", None) or ""
    return "site-packages" in path or "dist-packages" in path


def class_name(cls: type) -> str:
    """ "module.Qualname", by the shortest module that exports the class: a Path made at
    setup is a pathlib.PosixPath, not a pathlib._local.PosixPath."""
    parts = cls.__module__.split(".")
    for i in range(1, len(parts)):
        if getattr(sys.modules.get(".".join(parts[:i])), cls.__qualname__, None) is cls:
            return f"{'.'.join(parts[:i])}.{cls.__qualname__}"
    return f"{cls.__module__}.{cls.__qualname__}"


class LiveTypes:
    """The field types of the app's objects as setup left them: observe(obj) records the
    type of each field of obj's class, joined over every instance observed, and walks into
    the app objects the fields hold. `fields` maps "module.Qualname" to {field: Ty}.

    An object of a library class types as obj("module.Qualname") without its fields;
    `modules` (the app's module names) decides which classes are the app's, and without it
    every class outside the standard library, numpy, manifoldx and site-packages is."""

    def __init__(self, modules=None):
        self.modules = set(modules) if modules is not None else None
        self.fields: dict[str, dict[str, Ty]] = {}
        self.globals: dict[tuple[str, str], Ty] = {}
        self._seen: dict[int, object] = {}  # keeps what it saw alive, so ids stay unique

    def is_app(self, cls: type) -> bool:
        if self.modules is not None:
            return cls.__module__ in self.modules
        return not _library(cls.__module__)

    def observe(self, value) -> Ty:
        """value's type; for an app object, also record its class's fields."""
        return self.of(value)

    def observe_global(self, module: str, name: str, value) -> Ty:
        """Record the module global module.name as holding value."""
        t = self.of(value)
        self.globals[(module, name)] = join(self.globals.get((module, name), NEVER), t)
        return t

    def of(self, v) -> Ty:
        if v is None:
            return NONE
        if isinstance(v, (bool, np.bool_)):
            return BOOL
        if isinstance(v, np.floating):  # before float: np.float64 is a float
            return NUM32 if v.dtype == np.float32 else NUM64
        if isinstance(v, (int, np.integer)):
            return INT
        if isinstance(v, float):
            return NUM
        if isinstance(v, str):
            return STR
        if is_geometry(v):
            return GEOMETRY
        if isinstance(v, np.ndarray):
            if v.dtype == np.float32:
                return vec("f32")
            return vec("f64") if np.issubdtype(v.dtype, np.floating) else UNKNOWN
        if isinstance(v, list):
            return list_of(union(*map(self.of, v)))
        if isinstance(v, tuple):
            return tuple_of(*map(self.of, v))
        if isinstance(v, (set, frozenset)):
            return set_of(union(*map(self.of, v)))
        if isinstance(v, dict):
            return dict_of(union(*map(self.of, v)), union(*map(self.of, v.values())))
        if isinstance(v, (types.FunctionType, types.MethodType)):
            f = getattr(v, "__func__", v)
            return fn(None if f.__name__ == "<lambda>" else f"{f.__module__}.{f.__qualname__}")
        if isinstance(v, (type, types.ModuleType)) or callable(v) and not hasattr(v, "__dict__"):
            return UNKNOWN
        cls = type(v)
        name = class_name(cls)
        if self.is_app(cls) and id(v) not in self._seen:
            self._seen[id(v)] = v
            fields = self.fields.setdefault(name, {})
            for k, x in vars(v).items() if hasattr(v, "__dict__") else ():
                fields[k] = join(fields.get(k, NEVER), self.of(x))
        return obj(name)


# ---------------------------------------------------------------- the environment


class TranspileError(Exception):
    """A reached construct outside the subset, at path:line, and the root that reached it."""

    def __init__(self, path: str, line: int, what: str, root: str | None = None):
        super().__init__(path, line, what, root)
        self.path, self.line, self.what, self.root = path, line, what, root

    def __str__(self) -> str:
        where = f" (reached from {self.root})" if self.root else ""
        return f"{self.path}:{self.line}: {self.what} is outside the subset{where}"


def module_path(name: str) -> str:
    return name.replace(".", "/") + ".py"


FunctionKey = tuple[str, str]  # (module, qualname)


def key_label(key: FunctionKey) -> str:
    return f"{key[0]}.{key[1]}"


@dataclass(frozen=True)
class Symbol:
    """What a module-level name means.

    kind: "func" (target is a FunctionKey), "module" (an app module, target its name),
    "const" (a module global, target (module, name)), "import" (a name imported from an app
    module, target (module, name), resolved through that module), "lib" (math, numpy,
    textwrap: target the library), "setup" (a name bound by any other import: setup-only,
    an error when reached code names it; target the import's text), "class" (target
    "module.Qualname").
    """

    kind: str
    target: object


@dataclass
class FuncInfo:
    """A function's signature and what inference knows about its body.

    A method's key is (module, "Class.method") and `cls` its class ("module.Class"); its
    first parameter is typed as that class. A function defined inside another has the key
    (module, "outer.<locals>.inner") and `parent` the outer one's key."""

    key: FunctionKey
    node: ast.FunctionDef
    params: list[str]
    defaults: dict[str, ast.expr]
    param_types: dict[str, Ty]
    locals: dict[str, Ty] = field(default_factory=dict)
    returns: Ty = NEVER
    annotated: set[str] = field(default_factory=set)  # locals whose annotation fixes them
    cls: str | None = None
    parent: FunctionKey | None = None
    declared: dict[str, Ty] = field(default_factory=dict)  # param types before call sites
    declared_returns: Ty = UNKNOWN  # the return annotation, which wins over inference

    @property
    def self_name(self) -> str | None:
        """The name a method's instance goes by (its first parameter)."""
        return self.params[0] if self.cls and self.params else None


@dataclass
class ClassInfo:
    """An app class: its fields' types (from class-level annotations, then from every
    `self.x = ...` in its methods), its methods and properties, and for a dataclass the
    fields its generated __init__ takes, in order."""

    name: str  # "module.Qualname"
    node: ast.ClassDef
    dataclass: bool
    fields: dict[str, Ty] = field(default_factory=dict)
    annotated: set[str] = field(default_factory=set)
    init_fields: list[ast.AnnAssign] = field(default_factory=list)
    methods: dict[str, FunctionKey] = field(default_factory=dict)
    properties: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Ctx:
    """Where an expression is: its module, its function (None at module level), the names
    an enclosing comprehension or lambda binds, and the enclosing function's Ctx for a
    function defined inside another."""

    module: str
    func: FuncInfo | None = None
    extra: tuple[tuple[str, Ty], ...] = ()
    parent: Ctx | None = None

    def bind(self, names: dict[str, Ty]) -> Ctx:
        return Ctx(self.module, self.func, self.extra + tuple(names.items()), self.parent)

    def binder(self, name: str) -> Ctx | None:
        """The innermost Ctx whose function or bindings hold `name`, None for a global."""
        c = self
        while c is not None:
            if any(k == name for k, _ in c.extra):
                return c
            f = c.func
            if f is not None and (name in f.locals or name in f.param_types):
                return c
            c = c.parent
        return None


BUILTIN_RETURNS = {
    "len": INT,
    "bool": BOOL,
    "int": INT,
    "float": NUM,
    "any": BOOL,
    "all": BOOL,
    "hasattr": BOOL,
    "isinstance": BOOL,
    "str": STR,
    "range": list_of(INT),
}

# Methods of builtin values the subset has, and what they return.
STR_METHODS = {
    "lower": STR,
    "upper": STR,
    "strip": STR,
    "startswith": BOOL,
    "endswith": BOOL,
    "join": STR,
}

LIBS = {
    "math": "math",
    "numpy": "numpy",
    "random": "random",
    "textwrap": "textwrap",
    "time": "time",
}
# time.time() reads the frame's clock; frame code has no other time function.
TIME_HINT = "(frame code reads time.time(), the frame's clock, or engine.elapsed)"
# The numpy functions on arrays of any rank (TypeEnv.nd_function types them).
ND_FUNCTIONS = frozenset(
    f"numpy.{f}"
    for f in (
        "sin cos exp sqrt floor clip minimum maximum where sum mean min max argmin argmax "
        "stack column_stack"
    ).split()
) | {"numpy.linalg.norm"}
# Operators that make any numpy operand an nd, a vec included.
ND_ONLY_OPS = (ast.Pow, ast.BitAnd, ast.BitOr, ast.MatMult)
# The numpy dtypes an array constructor may name, by their dotted names.
NP_DTYPES = {
    "numpy.float32": "f32",
    "numpy.float64": "f64",
    "numpy.bool_": "bool",
    "numpy.bool": "bool",
    "numpy.int32": "i32",
}

# random.Random's methods the page has, and what they return; the module functions of the
# same names draw from the module's own generator ($rt.random).
RANDOM = obj("random.Random")
RANDOM_METHODS = {
    "random": NUM,
    "uniform": NUM,
    "randrange": INT,
    "randint": INT,
    "choice": UNKNOWN,
}
RANDOM_FUNCTIONS = frozenset(f"random.{m}" for m in RANDOM_METHODS)

# The manifoldx objects the headless runtime (js/manifoldx.js) implements, by class: each
# attribute's type, a method's as fn(name, return), by the names class_name gives the
# classes. What is not here types as unknown.
_INPUT, _CAMERA = "manifoldx.input.InputState", "manifoldx.Camera"
_HANDLE, _PROXY = "manifoldx.engine.EntityHandle", "manifoldx.engine._TransformProxy"
_GUI = "manifoldx.gui.widgets._GuiRoot"
_KEY_EVENT, _POINTER_EVENT = "manifoldx.input.KeyEvent", "manifoldx.input.PointerEvent"
_WHEEL_EVENT, _RESIZE_EVENT = "manifoldx.input.WheelEvent", "manifoldx.input.ResizeEvent"
_PANEL, _TEXT, _VALUE_DISPLAY = (
    "manifoldx.gui.Panel",
    "manifoldx.gui.Text",
    "manifoldx.gui.ValueDisplay",
)
_BUTTON, _SLIDER, _TOGGLE, _JOYSTICK, _IMAGE = (
    "manifoldx.gui.Button",
    "manifoldx.gui.Slider",
    "manifoldx.gui.Toggle",
    "manifoldx.gui.Joystick",
    "manifoldx.gui.Image",
)
RUNTIME_SURFACE = {
    "manifoldx.Engine": {
        "input": obj(_INPUT),
        "camera": obj(_CAMERA),
        "gui": obj(_GUI),
        "w": INT,
        "h": INT,
        "state": STATE,
        "set_material": fn("manifoldx.Engine.set_material", NONE),
        "log": fn("manifoldx.Engine.log", NONE),
        "spawn": fn("manifoldx.Engine.spawn", union(obj(_HANDLE), NONE)),
        "destroy": fn("manifoldx.Engine.destroy", NONE),
        "set_joints": fn("manifoldx.Engine.set_joints", NONE),
        "emit": fn("manifoldx.Engine.emit", NONE),
        "elapsed": NUM,
    },
    _INPUT: {
        "pressed_keys": set_of(STR),
        "modifiers": list_of(STR),
        "mouse_pos": tuple_of(NUM, NUM),
        "mouse_delta": tuple_of(NUM, NUM),
        "wheel_delta": tuple_of(NUM, NUM),
        "pressed_buttons": list_of(INT),
        "viewport_size": tuple_of(INT, INT),
        "pointer_captured": BOOL,
        **{
            m: fn(f"{_INPUT}.{m}", BOOL)
            for m in (
                "is_pressed",
                "just_pressed",
                "just_released",
                "is_mouse_pressed",
                "just_mouse_pressed",
                "just_mouse_released",
                "capture_pointer",
            )
        },
    },
    _CAMERA: {
        "position": vec("f32"),
        "target": vec("f32"),
        "up": vec("f32"),
        "fov": NUM,
        "near": NUM,
        "far": NUM,
        "get_forward": fn(f"{_CAMERA}.get_forward", vec("f32")),
        "project": fn(f"{_CAMERA}.project", tuple_of(nd("f64", 2), nd("f64", 1))),
    },
    _GUI: {"pointer_over_gui": BOOL},
    # An entity handle: its transform reads and writes the entity's row at once (pos in world
    # coordinates, float64), set_material writes its Material column, set_joints its pose,
    # destroy queues it.
    _HANDLE: {
        "index": INT,
        "transform": obj(_PROXY),
        "set_material": fn(f"{_HANDLE}.set_material", NONE),
        "set_joints": fn(f"{_HANDLE}.set_joints", NONE),
        "destroy": fn(f"{_HANDLE}.destroy", NONE),
    },
    _PROXY: {"pos": vec("f64"), "rot": vec("f32"), "scale": vec("f32")},
    # The input events' payloads (js/manifoldx.js's eventOf), read only; a tuple[X, ...] types
    # as a list of X, as its annotation does.
    _KEY_EVENT: {"key": STR, "modifiers": list_of(STR), "is_down": BOOL},
    _POINTER_EVENT: {
        "x": NUM,
        "y": NUM,
        "dx": NUM,
        "dy": NUM,
        "button": INT,
        "buttons": list_of(INT),
        "modifiers": list_of(STR),
        "phase": STR,
        "pointer_id": INT,
    },
    _WHEEL_EVENT: {
        "dx": NUM,
        "dy": NUM,
        "x": NUM,
        "y": NUM,
        "buttons": list_of(INT),
        "modifiers": list_of(STR),
    },
    _RESIZE_EVENT: {"width": INT, "height": INT, "pixel_ratio": NUM},
    # The widgets gui.js draws: the fields the page applies each frame when reached code
    # writes them (gui.js's sync), and nothing else.
    _PANEL: {"offset": tuple_of(NUM, NUM), "style_overrides": dict_of(STR, UNKNOWN)},
    _TEXT: {"text": STR, "style_overrides": dict_of(STR, UNKNOWN)},
    _VALUE_DISPLAY: {"style_overrides": dict_of(STR, UNKNOWN)},
    # The interactive widgets, whose state the runtime's port of their code keeps
    # (js/manifoldx.js's GuiBridge and WIDGET_HOOKS); was_clicked consumes the Button's latch.
    _BUTTON: {
        "name": STR,
        "label": STR,
        "was_clicked": fn(f"{_BUTTON}.was_clicked", BOOL),
        "style_overrides": dict_of(STR, UNKNOWN),
    },
    _SLIDER: {
        "name": STR,
        "value": NUM,
        "min": NUM,
        "max": NUM,
        "label": union(STR, NONE),
        "style_overrides": dict_of(STR, UNKNOWN),
    },
    _TOGGLE: {
        "name": STR,
        "value": BOOL,
        "label": STR,
        "style_overrides": dict_of(STR, UNKNOWN),
    },
    _JOYSTICK: {
        "name": STR,
        "value": tuple_of(NUM, NUM),
        "style_overrides": dict_of(STR, UNKNOWN),
    },
    _IMAGE: {"style_overrides": dict_of(STR, UNKNOWN)},
    # A random.Random made at setup: pyrt's Random draws as CPython's (choice types as the
    # element of its argument, TypeEnv.call).
    RANDOM.name: {m: fn(f"{RANDOM.name}.{m}", t) for m, t in RANDOM_METHODS.items()},
}

# The payload type of a handler of each engine event, for a payload parameter without an
# annotation the transpiler reads; a handler of another event annotates it.
EVENT_PAYLOADS = {
    "frame": dict_of(STR, NUM),
    "key_down": obj(_KEY_EVENT),
    "key_up": obj(_KEY_EVENT),
    "pointer_down": obj(_POINTER_EVENT),
    "pointer_up": obj(_POINTER_EVENT),
    "pointer_move": obj(_POINTER_EVENT),
    "wheel": obj(_WHEEL_EVENT),
    "resize": obj(_RESIZE_EVENT),
    # The widgets' events, by pattern (event_payload matches them).
    "gui:button:*:click": dict_of(STR, UNKNOWN),
    "gui:slider:*:change": dict_of(STR, NUM),
    "gui:slider:*:commit": dict_of(STR, NUM),
    "gui:toggle:*:change": dict_of(STR, BOOL),
    "gui:joystick:*:change": dict_of(STR, tuple_of(NUM, NUM)),
}


def event_payload(event: str) -> Ty | None:
    """The payload type of a handler of event: its EVENT_PAYLOADS entry, by name or pattern
    (a widget's event, "gui:slider:speed:change"), or None for an app's own event."""
    if event in EVENT_PAYLOADS:
        return EVENT_PAYLOADS[event]
    for pattern, t in EVENT_PAYLOADS.items():
        if "*" in pattern and fnmatch.fnmatchcase(event, pattern):
            return t
    return None


# The fields of manifoldx objects reached code may write: those the page applies each frame
# (gui.js's sync for the widgets, scene.js's sync for the camera). A write of any other field
# of a manifoldx object moves the desktop and not the page, so it fails the export.
WRITABLE = {
    _CAMERA: {"position", "target", "up", "fov", "near", "far"},
    _PROXY: {"pos", "rot", "scale"},
    _PANEL: {"offset", "style_overrides"},
    _TEXT: {"text", "style_overrides"},
    _VALUE_DISPLAY: {"style_overrides"},
    _BUTTON: {"label", "style_overrides"},
    _SLIDER: {"value", "min", "max", "style_overrides"},
    _TOGGLE: {"value", "style_overrides"},
    _JOYSTICK: {"value", "style_overrides"},
    _IMAGE: {"style_overrides"},
}

# The runtime-surface functions that take keywords, and which: the call passes them as a final
# object literal ({n: k}).
RUNTIME_KEYWORDS = {"manifoldx.Engine.spawn": ("n",)}

# How an error names an object of each manifoldx class; another goes by its class name.
RUNTIME_NAMES = {
    "manifoldx.Engine": "the engine",
    _INPUT: "the input",
    _CAMERA: "the camera",
    _GUI: "the GUI",
    _HANDLE: "an entity handle",
    _PROXY: "an entity's transform",
    _PANEL: "a Panel",
    _TEXT: "a Text",
    _VALUE_DISPLAY: "a ValueDisplay",
    _BUTTON: "a Button",
    _SLIDER: "a Slider",
    _TOGGLE: "a Toggle",
    _JOYSTICK: "a Joystick",
    _IMAGE: "an Image",
    _KEY_EVENT: "a KeyEvent",
    _POINTER_EVENT: "a PointerEvent",
    _WHEEL_EVENT: "a WheelEvent",
    _RESIZE_EVENT: "a ResizeEvent",
}

# math's functions the subset has, and its constants.
MATH_FUNCTIONS = frozenset(
    "sin cos tan asin acos atan atan2 sqrt exp log floor ceil fabs hypot radians degrees".split()
)
LIB_CONSTANTS = {
    "math.pi": "Math.PI",
    "math.e": "Math.E",
    "math.tau": "(2 * Math.PI)",
    "math.inf": "Infinity",
    "math.nan": "NaN",
    "numpy.pi": "Math.PI",
    "numpy.inf": "Infinity",
    "numpy.nan": "NaN",
}


class TypeEnv:
    """The app's modules, their symbols and functions, and the types inference found.

    expr(node, ctx) -> Ty         the type of an expression
    resolve(module, name) -> Symbol | None
    function(key) -> FuncInfo
    local(key, name) -> Ty
    solve(functions=None) -> None  type locals and returns to a fixed point
    """

    MAX_PASSES = 20

    def __init__(self, modules: dict[str, ast.Module], live=None):
        self.modules = modules
        self.live = live
        self.symbols: dict[str, dict[str, Symbol]] = {}
        self.globals: dict[tuple[str, str], Ty] = {}
        self.functions: dict[FunctionKey, FuncInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        # The app's Component subclasses, which are store layouts and not app classes (never
        # transpiled): {"module.Qualname": store name}, and each store name's live class.
        self.components: dict[str, str] = {}
        self.component_classes: dict[str, type] = {}
        # A closure root's free variables, typed from its live cells: {key: {name: Ty}}.
        self.cells: dict[FunctionKey, dict[str, Ty]] = {}
        for name, tree in modules.items():
            self.symbols[name] = self._scan(name, tree)
        # Signatures after every module's names are known, so an annotation may name a
        # class defined later or in another module.
        for name, tree in modules.items():
            for n in tree.body:
                if isinstance(n, ast.FunctionDef):
                    self._register(name, n, n.name)
                elif isinstance(n, ast.ClassDef):
                    self._class(name, n)

    # ---------------------------------------------------------- module scan
    def resolve_module(self, module: str, target: str | None, level: int) -> str | None:
        """The app module an import names, relative to `module`, or None."""
        if level:
            parts = module.split(".")[:-level] if level <= module.count(".") + 1 else None
            if parts is None:
                return None
            target = ".".join(parts + ([target] if target else []))
        return target if target in self.modules else None

    def _scan(self, module: str, tree: ast.Module) -> dict[str, Symbol]:
        syms: dict[str, Symbol] = {}
        for n in tree.body:
            if isinstance(n, ast.Import):
                for a in n.names:
                    bound = a.asname or a.name.split(".")[0]
                    if a.name in LIBS:
                        syms[bound] = Symbol("lib", a.name)
                    elif a.name in self.modules and (a.asname or "." not in a.name):
                        syms[bound] = Symbol("module", a.name)
                    else:
                        syms[bound] = Symbol("setup", f"import {a.name}")
            elif isinstance(n, ast.ImportFrom):
                src = self.resolve_module(module, n.module, n.level)
                for a in n.names:
                    bound = a.asname or a.name
                    sub = self.resolve_module(
                        module, f"{n.module}.{a.name}" if n.module else a.name, n.level
                    )
                    if src is not None:
                        syms[bound] = Symbol("import", (src, a.name))
                    elif sub is not None:
                        syms[bound] = Symbol("module", sub)
                    else:
                        syms[bound] = Symbol("setup", f"from {n.module} import {a.name}")
            elif isinstance(n, ast.FunctionDef):
                syms[n.name] = Symbol("func", (module, n.name))
            elif isinstance(n, ast.ClassDef):
                syms[n.name] = Symbol("class", f"{module}.{n.name}")
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for name in target_names(t):
                        syms[name] = Symbol("const", (module, name))
        return syms

    def _register(
        self,
        module: str,
        n: ast.FunctionDef,
        qualname: str,
        cls: str | None = None,
        parent: FunctionKey | None = None,
    ) -> FuncInfo:
        """A FuncInfo for n and for each function defined inside it."""
        key = (module, qualname)
        info = self._func_info(key, n, module)
        info.cls, info.parent = cls, parent
        if cls and n.args.args and n.args.args[0].annotation is None:
            info.param_types[n.args.args[0].arg] = obj(cls)
        info.declared = dict(info.param_types)
        self.functions[key] = info
        for st in own_statements(n.body):
            if isinstance(st, ast.FunctionDef):
                self._register(module, st, f"{qualname}.<locals>.{st.name}", parent=key)
        return info

    def _class(self, module: str, n: ast.ClassDef) -> None:
        name = f"{module}.{n.name}"
        if self._is_component(module, n):
            self.components[name] = n.name
            live = getattr(sys.modules.get(module), n.name, None)
            if isinstance(live, type):
                self.component_classes[n.name] = live
            return
        info = ClassInfo(name, n, any(self.is_dataclass(d, module) for d in n.decorator_list))
        self.classes[name] = info
        for st in n.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                if is_classvar(st.annotation):
                    continue
                t = self.annotation(st.annotation, module)
                if t.known:  # one the model cannot read (a bare `tuple`) leaves the live type
                    info.fields[st.target.id] = t
                    info.annotated.add(st.target.id)
                if info.dataclass:
                    info.init_fields.append(st)
            elif isinstance(st, ast.FunctionDef):
                f = self._register(module, st, f"{n.name}.{st.name}", cls=name)
                info.methods[st.name] = f.key
                if any(ast.unparse(d) == "property" for d in st.decorator_list):
                    info.properties.add(st.name)
        if info.dataclass and "__init__" not in info.methods:
            init = self._dataclass_init(n, info.init_fields)
            info.methods["__init__"] = self._register(
                module, init, f"{n.name}.__init__", cls=name
            ).key

    def _is_component(self, module: str, n: ast.ClassDef) -> bool:
        """n subclasses manifoldx.components.Component: its live class does, or a base names
        a library class that does."""
        from manifoldx.components import Component

        live = getattr(sys.modules.get(module), n.name, None)
        if isinstance(live, type):
            return issubclass(live, Component)
        return any(
            isinstance(c, type) and issubclass(c, Component)
            for c in (self.library_object(b, module) for b in n.bases)
        )

    def library_object(self, n: ast.expr, module: str):
        """The library object a setup-imported name (`Transform`, `mx.Transform`) holds, as
        setup imported it; None for anything else."""
        attrs = []
        while isinstance(n, ast.Attribute):
            attrs.append(n.attr)
            n = n.value
        if not isinstance(n, ast.Name):
            return None
        sym = self.symbols.get(module, {}).get(n.id)
        if sym is None or sym.kind != "setup":
            return None
        words = sym.target.split()
        if words[0] == "from":
            v = getattr(sys.modules.get(words[1]), words[3], None)
        else:
            v = sys.modules.get(words[1])
        for a in reversed(attrs):
            v = getattr(v, a, None)
        return v

    def component_name(self, n: ast.expr, ctx: Ctx) -> str | None:
        """The store name of the component n names in q[n]: Transform, Mesh, Material, a
        Component subclass (the app's or a library's), or a str; None for anything else."""
        from manifoldx.components import Component, Material, Mesh, Transform

        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.Name) and not self.is_local(n.id, ctx):
            sym = self.resolve(ctx.module, n.id)
            if sym is not None and sym.kind == "class":
                return self.components.get(sym.target)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            sym = self.resolve(ctx.module, n.value.id)
            if sym is not None and sym.kind == "module":
                inner = self.resolve(sym.target, n.attr)
                return (
                    self.components.get(inner.target) if inner and inner.kind == "class" else None
                )
        v = self.library_object(n, ctx.module)
        if v in (Transform, Mesh, Material) or (isinstance(v, type) and issubclass(v, Component)):
            if isinstance(v, type) and issubclass(v, Component):
                self.component_classes.setdefault(v.__name__, v)
            return v.__name__
        return None

    def component_field(self, recv: Ty, attr: str) -> Ty:
        """The field view recv.attr gives for an accessor recv, unknown when the component
        has no such field (or is Mesh or Material, whose ids frame code does not read)."""
        component = recv.name.removeprefix("ro:")
        if component in ("Mesh", "Material"):
            return UNKNOWN
        layout = component_fields(component, self.component_classes.get(component))
        if attr not in layout:
            return UNKNOWN
        start, size = layout[attr]
        return field_view(component, attr, start, size, readonly(recv))

    @staticmethod
    def _dataclass_init(n: ast.ClassDef, fields: list[ast.AnnAssign]) -> ast.FunctionDef:
        """The __init__ @dataclass generates, as a def: one parameter per field, whose
        default is the field's value (a constant or a field(...) call)."""
        args = [ast.arg("self")] + [ast.arg(f.target.id, f.annotation) for f in fields]
        defaults = [f.value for f in fields if f.value is not None]
        init = ast.FunctionDef(
            name="__init__",
            args=ast.arguments(
                posonlyargs=[], args=args, kwonlyargs=[], kw_defaults=[], defaults=defaults
            ),
            body=[],
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        return ast.copy_location(init, n)

    def is_dataclass(self, d: ast.expr, module: str) -> bool:
        """d is @dataclass or @dataclass(...), from the dataclasses module."""
        d = d.func if isinstance(d, ast.Call) else d
        return self.from_stdlib(d, module, "dataclasses", "dataclass")

    def is_registration(self, d: ast.expr, module: str) -> bool:
        """d is @engine.on(...) or @engine.system, engine a module global holding the Engine:
        both register the function at setup and return it unchanged, so the browser gets the
        plain function."""
        call = d.func if isinstance(d, ast.Call) else d
        if not (isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name)):
            return False
        if (call.attr, isinstance(d, ast.Call)) not in {("on", True), ("system", False)}:
            return False
        sym = self.resolve(module, call.value.id)
        return (
            sym is not None
            and sym.kind == "const"
            and self.globals.get(sym.target) == obj("manifoldx.Engine")
        )

    def from_stdlib(self, n: ast.expr, module: str, lib: str, name: str) -> bool:
        """n names `lib.name`, imported either way."""
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            sym = self.symbols.get(module, {}).get(n.value.id)
            return n.attr == name and sym is not None and sym.target == f"import {lib}"
        if isinstance(n, ast.Name):
            sym = self.symbols.get(module, {}).get(n.id)
            return sym is not None and sym.target == f"from {lib} import {name}"
        return False

    def ctx_of(self, info: FuncInfo) -> Ctx:
        """The Ctx of info's body: a function defined inside another sees that one's names;
        a closure whose live cells are known sees their types first."""
        parent = self.ctx_of(self.functions[info.parent]) if info.parent else None
        return Ctx(info.key[0], info, tuple(self.cells.get(info.key, {}).items()), parent)

    def register_lambda(self, module: str, qualname: str, n: ast.Lambda) -> FuncInfo:
        """A lambda as a function of its own (a GUI getter is one), keyed
        (module, "qualname@line:col"); its free variables come from `cells`."""
        body = ast.copy_location(ast.Return(n.body), n.body)
        node = ast.FunctionDef(
            name="<lambda>",
            args=n.args,
            body=[body],
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        key = (module, f"{qualname}@{n.lineno}:{n.col_offset}")
        info = self._func_info(key, ast.copy_location(node, n), module)
        info.declared = dict(info.param_types)
        self.functions[key] = info
        return info

    def _func_info(self, key: FunctionKey, n: ast.FunctionDef, module: str) -> FuncInfo:
        a = n.args
        params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
        positional = a.posonlyargs + a.args
        defaults = {
            p.arg: d for p, d in zip(positional[len(positional) - len(a.defaults) :], a.defaults)
        }
        defaults |= {p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None}
        types = {}
        for p in a.posonlyargs + a.args + a.kwonlyargs:
            types[p.arg] = self.annotation(p.annotation, module) if p.annotation else UNKNOWN
        returns = self.annotation(n.returns, module) if n.returns else UNKNOWN
        return FuncInfo(key, n, params, defaults, types, declared_returns=returns)

    def annotation(self, node: ast.expr, module: str) -> Ty:
        """The type an annotation names; unknown for what the model has no kind for."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.annotation(ast.parse(node.value, mode="eval").body, module)
        if isinstance(node, ast.Constant) and node.value is None:
            return NONE
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return union(self.annotation(node.left, module), self.annotation(node.right, module))
        if isinstance(node, ast.Name):
            simple = {"int": INT, "float": NUM, "bool": BOOL, "str": STR}
            if node.id in simple:
                return simple[node.id]
            sym = self.resolve(module, node.id)
            if sym is not None and sym.kind == "class":
                return obj(sym.target)
            if sym is not None and sym.kind == "setup":  # EntityHandle, from manifoldx.engine
                t = self.library_class(sym.target, None)
                return t if t.name in RUNTIME_SURFACE else UNKNOWN
            return UNKNOWN
        if isinstance(node, ast.Attribute) and ast.unparse(node) in ("np.ndarray", "numpy.ndarray"):
            return vec("f64")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            head = node.value.id
            if head == "tuple" and len(args) == 2 and _is_ellipsis(args[1]):
                # Of any length, as a tuple(...) call's result: a list of its items' type.
                return list_of(self.annotation(args[0], module))
            ts = [self.annotation(x, module) for x in args]
            if head == "list" and len(ts) == 1:
                return list_of(ts[0])
            if head == "set" and len(ts) == 1:
                return set_of(ts[0])
            if head == "dict" and len(ts) == 2:
                return dict_of(*ts)
            if head == "tuple":
                return tuple_of(*ts)
            if head == "Optional" and len(ts) == 1:
                return union(ts[0], NONE)
        return UNKNOWN

    # ---------------------------------------------------------- lookups
    def resolve(self, module: str, name: str) -> Symbol | None:
        """The module-level symbol `name` in `module`, following imports from app modules."""
        seen = set()
        sym = self.symbols.get(module, {}).get(name)
        while sym is not None and sym.kind == "import" and sym.target not in seen:
            seen.add(sym.target)
            src, attr = sym.target
            sym = self.symbols[src].get(attr)
            if sym is None and self.resolve_module(src, attr, 0) is None:
                sub = f"{src}.{attr}"
                sym = Symbol("module", sub) if sub in self.modules else None
        return sym

    def function(self, key: FunctionKey) -> FuncInfo:
        return self.functions[key]

    def local(self, key: FunctionKey, name: str) -> Ty:
        info = self.functions[key]
        return info.locals.get(name, info.param_types.get(name, UNKNOWN))

    @staticmethod
    def is_local(name: str, ctx: Ctx) -> bool:
        return ctx.binder(name) is not None

    def lib_name(self, n: ast.expr, ctx: Ctx) -> str | None:
        """The dotted name n spells inside a library module ("numpy.linalg.norm",
        "math.pi"), or None when n does not start at a library import."""
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if not isinstance(n, ast.Name) or self.is_local(n.id, ctx):
            return None
        sym = self.symbols.get(ctx.module, {}).get(n.id)
        if sym is None or sym.kind != "lib":
            return None
        return ".".join([sym.target, *reversed(parts)])

    def np_dtype(
        self, n: ast.Call, ctx: Ctx, at: int = 1, default: str | None = "f64"
    ) -> str | None:
        """The dtype a numpy constructor asks for, by keyword or as its positional argument
        `at`: `default` when none is given; "f32", "f64", "bool" or "i32" for np.float32,
        np.float64, np.bool_ (np.bool) and np.int32; None for any other."""
        given = [k.value for k in n.keywords if k.arg == "dtype"] + n.args[at : at + 1]
        if not given:
            return default
        return NP_DTYPES.get(self.lib_name(given[0], ctx) or "")

    def lib_call(self, name: str, n: ast.Call, ctx: Ctx) -> Ty:
        if name in ("numpy.zeros", "numpy.ones", "numpy.full") and n.args:
            if name == "numpy.full":
                fill = self.expr(n.args[1], ctx) if len(n.args) > 1 else UNKNOWN
                d = self.np_dtype(n, ctx, 2, "f64" if fill.all("num") else None)
            else:
                d = self.np_dtype(n, ctx)
            shape = self.expr(n.args[0], ctx)
            if d is None:
                return UNKNOWN
            if shape.all("tuple") and len(shape.members) == 1 and shape.args:
                return nd(d, len(shape.args))
            if shape.all("int"):
                return vec(d) if d in ("f32", "f64") else nd(d, 1)
            return UNKNOWN
        if name == "numpy.eye":
            d = self.np_dtype(n, ctx, len(n.args) + 1)
            return nd(d, 2) if d in ("f32", "f64", "i32") else UNKNOWN
        if name == "numpy.array" and n.args:
            given = any(k.arg == "dtype" for k in n.keywords) or len(n.args) > 1
            items = self.expr(n.args[0], ctx)
            row = elem(items)
            if items.all("list", "tuple") and len(row.members) == 1 and (
                (row.all("list", "tuple") and elem(row).is_numeric)
                or (is_array(row) and array_rank(row) == 1)
            ):  # rows: lists of numbers or rank-1 arrays, a matrix
                if given:
                    d = self.np_dtype(n, ctx)
                elif is_array(row):
                    d = row.dtype
                elif elem(row).all("int"):
                    d = "i32"
                else:  # Python numbers; a numpy scalar's dtype would decide numpy's
                    d = "f64" if all(m.dtype in (None, "f64") for m in elem(row).members) else None
                return nd(d, 2) if d else UNKNOWN
            if not given and items.all("list", "tuple") and elem(items).all("int"):
                return nd("i32", 1)  # numpy's int64, held as int32
            d = self.np_dtype(n, ctx)
            return vec(d) if d in ("f32", "f64") else nd(d, 1) if d else UNKNOWN
        if name in ND_FUNCTIONS:
            return self.nd_function(name, n, ctx)
        if name == "numpy.arange":
            given = any(k.arg == "dtype" for k in n.keywords) or len(n.args) > 3
            if not given:
                return vec("f64")
            d = self.np_dtype(n, ctx, 3)
            return nd(d, 1) if d in ("f32", "f64", "i32") else UNKNOWN
        if name in ("math.floor", "math.ceil"):
            return INT
        if name.startswith("math.") and name[5:] in MATH_FUNCTIONS:
            return NUM
        if name == "textwrap.wrap":
            return list_of(STR)
        if name == "random.choice" and n.args:
            return elem(self.expr(n.args[0], ctx))
        if name in RANDOM_FUNCTIONS:
            return RANDOM_METHODS[name[7:]]
        if name == "random.Random":
            return RANDOM
        if name == "time.time":
            return NUM
        return UNKNOWN

    @staticmethod
    def axis(n: ast.Call) -> int | None | str:
        """The axis a reduction names (keyword or second argument): None, 0 or 1, or "bad"
        for anything else."""
        given = [k.value for k in n.keywords if k.arg == "axis"] + n.args[1:2]
        if not given or (isinstance(given[0], ast.Constant) and given[0].value is None):
            return None
        v = given[0]
        return (
            v.value
            if isinstance(v, ast.Constant) and v.value in (0, 1) and type(v.value) is int
            else "bad"
        )

    def nd_function(self, name: str, n: ast.Call, ctx: Ctx) -> Ty:
        """The type of a numpy function of ND_FUNCTIONS on arrays; unknown outside the subset."""
        f = name.removeprefix("numpy.")
        args = [self.expr(a, ctx) for a in n.args]
        if args and any(not t.members for t in args):  # not typed yet (a call's return): wait
            return NEVER
        if not args or any(len(t.members) != 1 for t in args):
            return UNKNOWN
        a = args[0]
        if f in ("stack", "column_stack"):
            items = elem(a)
            if a.all("list", "tuple") and not items.members:
                return NEVER
            if not (a.all("list", "tuple") and len(items.members) == 1 and is_array(items)):
                return UNKNOWN
            rank = array_rank(items)
            axis = 1 if f == "column_stack" else self.axis(n)
            if axis == "bad" or (axis == 1 and rank != 1) or items.dtype == "bool":
                return UNKNOWN  # along axis 1, vectors only (as columns)
            return nd(items.dtype, rank + 1)
        if f == "where" and len(args) == 1:
            return (
                tuple_of(*[nd("i32", 1)] * a.rank)
                if a.kind == "nd" and a.dtype == "bool"
                else UNKNOWN
            )
        if f == "where" and len(args) == 3:
            if not (a.kind == "nd" and a.dtype == "bool"):
                return UNKNOWN
            d = nd_dtype(ast.Add(), args[1], args[2])
            return nd(d, max(map(array_rank, args))) if d else UNKNOWN
        if not is_array(a):
            return UNKNOWN
        rank = array_rank(a)
        if f in ("sin", "cos", "exp", "sqrt", "floor"):
            return nd(a.dtype, rank) if a.dtype in ("f32", "f64") else UNKNOWN
        if f in ("minimum", "maximum", "clip"):
            if len(args) != (3 if f == "clip" else 2):
                return UNKNOWN
            d = nd_dtype(ast.Add(), a, args[1])
            if f == "clip" and d:
                d = nd_dtype(ast.Add(), nd(d, rank), args[2])
            return nd(d, max(map(array_rank, args))) if d else UNKNOWN
        if len(args) > (1 if f == "linalg.norm" else 2) and f not in ("minimum", "maximum", "clip"):
            return UNKNOWN  # norm's second argument is ord, not the axis
        if f in ("argmin", "argmax"):
            return INT if len(args) == 1 and a.dtype != "bool" else UNKNOWN
        axis = self.axis(n)
        if axis == "bad" or (axis == 1 and rank != 2):
            return UNKNOWN
        if f == "linalg.norm":
            if axis is None:
                return Ty("num", dtype=a.dtype) if a.kind == "vec" else UNKNOWN
            out = "f64" if a.dtype in ("i32", "bool") else a.dtype
        elif f == "sum":
            out = "int" if a.dtype in ("i32", "bool") else a.dtype
        elif f == "mean":
            out = "f64" if a.dtype in ("i32", "bool") else a.dtype
        elif a.dtype == "bool":
            return UNKNOWN
        else:  # min, max
            out = "int" if a.dtype == "i32" else a.dtype
        if axis is None or rank == 1:
            return INT if out == "int" else Ty("num", dtype=out)
        return nd("i32" if out == "int" else out, 1)

    def name(self, name: str, ctx: Ctx) -> Ty:
        c = ctx.binder(name)
        if c is not None:
            for k, t in reversed(c.extra):
                if k == name:
                    return t
            return self.local(c.func.key, name)
        sym = self.resolve(ctx.module, name)
        if sym is None:
            return UNKNOWN
        if sym.kind == "const":
            return self.globals.get(sym.target, UNKNOWN)
        if sym.kind == "func":
            return fn(key_label(sym.target), self.functions[sym.target].returns)
        return UNKNOWN

    def attribute(self, recv: Ty, attr: str) -> Ty:
        """The type of recv.attr, joined over the app classes recv may be. A class without
        the attribute adds nothing (hasattr or isinstance guards that read)."""
        if not recv.members:
            return NEVER
        if recv.kind == "nd" and attr == "shape":
            return tuple_of(*[INT] * recv.rank)
        if recv.kind == "accessor":
            return self.component_field(recv, attr)
        if recv.kind == "field":
            return nd("f32", 2) if attr == "data" else UNKNOWN
        out = []
        for m in recv.members:
            if m.kind == "none":
                continue
            c = self.classes.get(m.name) if m.kind == "obj" else None
            if c is None:
                known = RUNTIME_SURFACE.get(m.name, {}) if m.kind == "obj" else {}
                if attr not in known:
                    return UNKNOWN
                out.append(known[attr])
                continue
            if attr in c.properties:
                out.append(self.functions[c.methods[attr]].returns)
            elif attr in c.methods:
                key = c.methods[attr]
                out.append(fn(key_label(key), self.functions[key].returns))
            elif attr in c.fields:
                out.append(c.fields[attr])
        return union(*out) if out else UNKNOWN

    def methods(self, recv: Ty, attr: str) -> list[FuncInfo] | None:
        """The methods recv.attr calls, one per app class recv may be; None when recv is
        not only app objects or a class has a field (not a method) by that name."""
        out = []
        for m in recv.members:
            if m.kind == "none":
                continue
            c = self.classes.get(m.name) if m.kind == "obj" else None
            if c is None or attr in c.fields or attr in c.properties:
                return None
            if attr in c.methods:
                out.append(self.functions[c.methods[attr]])
        return out or None

    def returns_of_call(self, t: Ty) -> Ty:
        """What calling a value of type t returns: a function's return, an object's
        __call__'s."""
        out = []
        for m in t.members:
            if m.kind == "fn" and m.args:
                out.append(m.args[0])
            elif (
                m.kind == "obj"
                and m.name in self.classes
                and "__call__" in (c := self.classes[m.name]).methods
            ):
                out.append(self.functions[c.methods["__call__"]].returns)
            else:
                return UNKNOWN
        return union(*out)

    # ---------------------------------------------------------- expressions
    def expr(self, n: ast.expr, ctx: Ctx) -> Ty:
        """The type of expression n in ctx."""
        if isinstance(n, ast.Constant):
            v = n.value
            if v is None:
                return NONE
            return {bool: BOOL, int: INT, float: NUM, str: STR}.get(type(v), UNKNOWN)
        if isinstance(n, ast.Name):
            return self.name(n.id, ctx)
        if isinstance(n, ast.List):
            return list_of(union(*(self.item(e, ctx) for e in n.elts)))
        if isinstance(n, ast.Set):
            return set_of(union(*(self.item(e, ctx) for e in n.elts)))
        if isinstance(n, ast.Tuple):
            if any(isinstance(e, ast.Starred) for e in n.elts):  # its length is not known
                return list_of(union(*(self.item(e, ctx) for e in n.elts)))
            return tuple_of(*(self.expr(e, ctx) for e in n.elts))
        if isinstance(n, ast.Dict):
            return dict_of(
                union(*(self.expr(k, ctx) for k in n.keys if k is not None)),
                union(*(self.expr(v, ctx) for v in n.values)),
            )
        if isinstance(n, (ast.ListComp, ast.GeneratorExp)):
            return list_of(self.expr(n.elt, self.comprehension(n.generators, ctx)))
        if isinstance(n, ast.SetComp):
            return set_of(self.expr(n.elt, self.comprehension(n.generators, ctx)))
        if isinstance(n, ast.BinOp):
            return self.binop(n.op, self.expr(n.left, ctx), self.expr(n.right, ctx))
        if isinstance(n, ast.UnaryOp):
            t = self.expr(n.operand, ctx)
            if isinstance(n.op, ast.Not):
                return BOOL
            if t.is_bool:
                return INT
            return t
        if isinstance(n, ast.Compare):
            if len(n.ops) == 1 and not isinstance(n.ops[0], (ast.In, ast.NotIn, ast.Is, ast.IsNot)):
                a, b = self.expr(n.left, ctx), self.expr(n.comparators[0], ctx)
                if "field" in (a.kind, b.kind):
                    return self.field_compare(n.ops[0], a, b)
                if any(is_array(m) for m in a.members + b.members):
                    rank = (
                        max(array_rank(a), array_rank(b))
                        if len(a.members) == len(b.members) == 1
                        else 0
                    )
                    return nd("bool", rank) if rank and nd_compare_dtype(a, b) else UNKNOWN
            return BOOL
        if isinstance(n, ast.BoolOp):
            return union(*(self.expr(v, ctx) for v in n.values))
        if isinstance(n, ast.IfExp):
            return union(self.expr(n.body, ctx), self.expr(n.orelse, ctx))
        if isinstance(n, ast.Subscript):
            return self.subscript(n, ctx)
        if isinstance(n, ast.Call):
            return self.call(n, ctx)
        if isinstance(n, ast.Lambda):
            return fn(None, self.expr(n.body, ctx.bind(self.lambda_params(n, ctx))))
        if isinstance(n, ast.Attribute):
            return self.attribute_expr(n, ctx)
        if isinstance(n, ast.JoinedStr):
            return STR
        return UNKNOWN

    @staticmethod
    def field_compare(op: ast.cmpop, a: Ty, b: Ty) -> Ty:
        """`field < x`: the entity indices whose one column holds, an int32 array; unknown for
        a field on the right, a wider field or == and !=."""
        if a.kind != "field" or field_parts(a)[3] != 1 or len(b.members) != 1:
            return UNKNOWN
        if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            return UNKNOWN
        return nd("i32", 1) if field_compare_dtype(b) else UNKNOWN

    def item(self, e: ast.expr, ctx: Ctx) -> Ty:
        """The type of what an element of a display adds: a starred one, its items."""
        return elem(self.expr(e.value, ctx)) if isinstance(e, ast.Starred) else self.expr(e, ctx)

    def lambda_params(self, n: ast.Lambda, ctx: Ctx) -> dict[str, Ty]:
        """A lambda's parameters: one with a default has the default's type, the others
        are unknown."""
        a = n.args
        out = {x.arg: UNKNOWN for x in a.args}
        for x, d in zip(a.args[len(a.args) - len(a.defaults) :], a.defaults):
            out[x.arg] = self.expr(d, ctx)
        return out

    def attribute_expr(self, n: ast.Attribute, ctx: Ctx) -> Ty:
        lib = self.lib_name(n, ctx)
        if lib is not None:
            return NUM if lib in LIB_CONSTANTS else UNKNOWN
        if isinstance(n.value, ast.Name) and not self.is_local(n.value.id, ctx):
            sym = self.resolve(ctx.module, n.value.id)
            if sym is not None and sym.kind == "module":
                return self.name(n.attr, Ctx(sym.target))
        return self.attribute(self.expr(n.value, ctx), n.attr)

    def comprehension(self, generators: list[ast.comprehension], ctx: Ctx) -> Ctx:
        """ctx with the names the generators bind."""
        for g in generators:
            names: dict[str, Ty] = {}
            bind_target(g.target, elem(self.expr(g.iter, ctx)), names)
            ctx = ctx.bind(names)
        return ctx

    @staticmethod
    def binop(op: ast.operator, a: Ty, b: Ty) -> Ty:
        if not (a.members and b.members):  # a side not typed yet (a call's return): wait
            return NEVER
        if not (a.known and b.known):
            return UNKNOWN
        if "field" in (a.kind, b.kind):
            return TypeEnv.field_binop(op, a, b)
        arrays = [m.kind for m in a.members + b.members if m.kind in ("vec", "nd")]
        if "nd" in arrays or (arrays and isinstance(op, ND_ONLY_OPS)):
            return TypeEnv.nd_binop(op, a, b)
        if a.is_numeric and b.is_numeric:
            d = arith_dtype(a, b)
            if d == "mixed":
                return UNKNOWN
            if d is not None:
                return Ty("num", dtype=d)
            if isinstance(op, ast.Div):
                return NUM
            if a.all("int", "bool") and b.all("int", "bool"):
                return union(INT, NUM) if isinstance(op, ast.Pow) else INT  # 2 ** -1 is 0.5
            return NUM
        if isinstance(op, ast.Add) and a.all("str") and b.all("str"):
            return STR
        if isinstance(op, ast.Add) and a.all("list") and b.all("list"):
            return union(a, b)
        if isinstance(op, ast.Add) and a.all("tuple") and b.all("tuple"):
            return (
                union(*a.members, *b.members)
                if len(a.members) > 1 or len(b.members) > 1
                else tuple_of(*a.args, *b.args)
            )
        if isinstance(op, ast.Mult):
            for s, k in ((a, b), (b, a)):
                if s.all("list", "str", "tuple") and k.all("int", "bool"):
                    return s
        if (a.kind == "vec" or b.kind == "vec") and all(
            t.kind == "vec" or t.is_numeric for t in (a, b)
        ):
            d = arith_dtype(a, b)
            return vec(d or "f64") if d != "mixed" else UNKNOWN
        return UNKNOWN

    @staticmethod
    def field_binop(op: ast.operator, a: Ty, b: Ty) -> Ty:
        """`a op b` with a field view on a side, as _FieldView computes it: the other side cast
        to float32 first, the result a float32 (k, size) array. A field on the right takes a
        number on the left (an array on the left would run numpy's operator, not the field's)."""
        if not isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div)) or len(a.members) != 1:
            return UNKNOWN
        other = b if a.kind == "field" else a
        if len(other.members) != 1:
            return UNKNOWN
        ok = other.is_numeric or (a.kind == "field" and (is_array(other) or other.kind == "field"))
        return nd("f32", 2) if ok else UNKNOWN

    @staticmethod
    def nd_binop(op: ast.operator, a: Ty, b: Ty) -> Ty:
        """`a op b` with a numpy array on a side and an nd among them (or **, & and | on
        any array): an nd of numpy 2's dtype and the larger rank."""
        rank = max(array_rank(a), array_rank(b)) if len(a.members) == len(b.members) == 1 else 0
        if not rank:
            return UNKNOWN
        if isinstance(op, (ast.BitAnd, ast.BitOr)):
            return nd("bool", rank) if a.dtype == b.dtype == "bool" else UNKNOWN
        if isinstance(op, ast.MatMult):  # matrices or stacks of them
            if min(array_rank(a), array_rank(b)) < 1 or "bool" in (a.dtype, b.dtype):
                return UNKNOWN
            if min(array_rank(a), array_rank(b)) == 1:  # typed, so the export reaches its hint
                d = nd_dtype(op, a, b)
                return nd(d, 1) if d else UNKNOWN
        elif not isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            return UNKNOWN
        d = nd_dtype(op, a, b)
        return nd(d, rank) if d else UNKNOWN

    @staticmethod
    def field_column(t: Ty, key: ast.expr) -> Ty:
        """field[:, j], j a constant column of the field: a one-column field view."""
        component, name, start, size = field_parts(t)
        if not (
            isinstance(key, ast.Tuple)
            and len(key.elts) == 2
            and isinstance(key.elts[0], ast.Slice)
            and not (key.elts[0].lower or key.elts[0].upper or key.elts[0].step)
            and isinstance(key.elts[1], ast.Constant)
            and type(key.elts[1].value) is int
            and 0 <= key.elts[1].value < size
        ):
            return UNKNOWN
        return field_view(component, name, start + key.elts[1].value, 1, readonly(t))

    def nd_subscript(self, t: Ty, key: ast.expr, ctx: Ctx) -> Ty:
        """a[key] of an nd: an int gives an item (an nd of one rank less, or a number), [:, j]
        a column, a bool mask the elements it selects (its rank is the array's) or the rows
        (rank 1), an index array the rows it names; every one a copy."""
        if (
            isinstance(key, ast.Tuple)
            and len(key.elts) == 2
            and isinstance(key.elts[0], ast.Slice)
            and not (key.elts[0].lower or key.elts[0].upper or key.elts[0].step)
            and self.expr(key.elts[1], ctx).all("int")
        ):
            return nd(t.dtype, 1) if t.rank == 2 else UNKNOWN
        k = self.expr(key, ctx)
        if k.all("int"):
            return nd_item(t)
        if k.kind == "nd" and k.rank == 1 and k.dtype == "i32":
            return t
        if k.kind == "nd" and k.dtype == "bool" and k.rank in (1, t.rank):
            return nd(t.dtype, 1 if k.rank == t.rank else t.rank)
        return UNKNOWN

    def subscript(self, n: ast.Subscript, ctx: Ctx) -> Ty:
        t = self.expr(n.value, ctx)
        if t.kind == "query":
            name = self.component_name(n.slice, ctx)
            return accessor(name, readonly(t)) if name is not None else UNKNOWN
        if t.kind == "field":
            return self.field_column(t, n.slice)
        if t.kind == "nd" or (t.kind == "vec" and self.expr(n.slice, ctx).kind == "nd"):
            return self.nd_subscript(nd(t.dtype, 1) if t.kind == "vec" else t, n.slice, ctx)
        if isinstance(n.slice, ast.Slice):
            return t
        out = []
        for m in t.members:
            if (
                m.kind == "tuple"
                and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, int)
                and -len(m.args) <= n.slice.value < len(m.args)
            ):
                out.append(m.args[n.slice.value])
            elif m.kind == "dict":
                out.append(m.args[1])
            elif m.kind in ("list", "tuple", "str", "vec"):
                out.append(elem(m))
            else:
                return UNKNOWN
        return union(*out)

    def call(self, n: ast.Call, ctx: Ctx) -> Ty:
        f = n.func
        component = self.component_name(f, ctx)
        if component is not None:
            return Ty("component", name=component)
        if isinstance(f, ast.Name):
            local = self.is_local(f.id, ctx)
            if local:
                return self.returns_of_call(self.name(f.id, ctx))
            sym = self.resolve(ctx.module, f.id)
            if sym is not None and sym.kind == "func":
                return self.functions[sym.target].returns
            if sym is not None and sym.kind == "class":
                return obj(sym.target)
            if sym is None:
                return self.builtin_call(f.id, n, ctx)
            if sym.kind == "setup":
                return self.library_class(sym.target, None)
            return UNKNOWN
        if isinstance(f, ast.Attribute):
            lib = self.lib_name(f, ctx)
            if lib is not None:
                return self.lib_call(lib, n, ctx)
            if self.is_rotation(f, ctx):
                return vec("f32")
            if isinstance(f.value, ast.Name):
                sym = self.resolve(ctx.module, f.value.id)
                if sym is not None and sym.kind == "module":
                    target = self.resolve(sym.target, f.attr)
                    if target is not None and target.kind == "func":
                        return self.functions[target.target].returns
                    if target is not None and target.kind == "class":
                        return obj(target.target)
                    return UNKNOWN
                if sym is not None and sym.kind == "setup":
                    return self.library_class(sym.target, f.attr)
            recv = self.expr(f.value, ctx)
            if recv.all("list") and f.attr == "pop":
                return elem(recv)
            if recv.all("list") and f.attr in ("append", "extend", "insert"):
                return NONE
            if recv.all("set") and f.attr in ("add", "discard", "remove", "clear", "update"):
                return NONE
            if recv.all("dict"):
                k, v = (union(*(m.args[i] for m in recv.members)) for i in (0, 1))
                if f.attr in ("get", "setdefault"):
                    other = self.expr(n.args[1], ctx) if len(n.args) > 1 else NONE
                    return union(v, other)
                return {
                    "keys": list_of(k),
                    "values": list_of(v),
                    "items": list_of(tuple_of(k, v)),
                }.get(f.attr, UNKNOWN)
            if recv.all("str") and f.attr in STR_METHODS:
                return STR_METHODS[f.attr]
            if (
                f.attr == "choice"
                and n.args
                and [m for m in recv.members if m.kind != "none"] == [RANDOM]
            ):
                return elem(self.expr(n.args[0], ctx))
            if len(recv.members) == 1 and is_array(recv) and f.attr == "tolist":
                return nd_list(recv)
            methods = self.methods(recv, f.attr)
            if methods:
                return union(*(m.returns for m in methods))
            return self.returns_of_call(self.attribute(recv, f.attr))
        return self.returns_of_call(self.expr(f, ctx))

    def is_rotation(self, f: ast.expr, ctx: Ctx) -> bool:
        """f names Transform.rotation, as setup imported it (Transform.rotation,
        mx.Transform.rotation)."""
        from manifoldx.components import Transform

        return (
            isinstance(f, ast.Attribute)
            and f.attr == "rotation"
            and self.library_object(f.value, ctx.module) is Transform
        )

    @staticmethod
    def library_class(imported: str, attr: str | None) -> Ty:
        """What calling a setup-only name makes: an object of a library class when the name
        is one (`from pathlib import Path`, or `import pathlib` and `pathlib.Path`), looked up
        among the modules setup already imported; unknown otherwise."""
        words = imported.split()
        if words[0] == "from":
            module, name = words[1], words[3]
            v = getattr(sys.modules.get(module), name, None)
        else:
            v = sys.modules.get(words[1])
        if attr is not None:
            v = getattr(v, attr, None)
        return obj(class_name(v)) if isinstance(v, type) else UNKNOWN

    def builtin_call(self, name: str, n: ast.Call, ctx: Ctx) -> Ty:
        args = [self.expr(a, ctx) for a in n.args]
        if name in ("min", "max"):
            return elem(args[0]) if len(args) == 1 else union(*args)
        if name == "abs" and args:
            return INT if args[0].is_bool else args[0]
        if name in ("list", "tuple"):  # a tuple of unknown length types as a list
            return list_of(elem(args[0]) if args else NEVER)
        if name == "set":
            return set_of(elem(args[0]) if args else NEVER)
        if name == "next" and args:
            return union(elem(args[0]), *args[1:])
        return BUILTIN_RETURNS.get(name, UNKNOWN)

    # ---------------------------------------------------------- inference
    def solve(self, functions=None, root_of=None, report: bool = True) -> None:
        """Type `functions` (every function when None): their parameters from the arguments
        their call sites in `functions` pass (and their defaults), their locals and returns,
        the app classes' fields and the module globals, to a fixed point. Annotations win,
        then live field types, then inference.

        Each pass starts over from annotations, live types and the parameter types the last
        pass found; an argument still unknown is left out of its parameter while the others
        settle (so recursion types), and counts again once they have. With `report`, the
        first operator whose operand stays unknown raises TranspileError asking for an
        annotation; `root_of(key)` names the root that reached a function, for the message."""
        keys = list(self.functions) if functions is None else list(functions)
        self._labels = {key_label(k): k for k in self.functions}
        self._sites: dict[tuple[FunctionKey, str], Ty] = {}
        for _ in range(self.MAX_PASSES):
            self._infer(keys)
            sites = self._site_types(keys, optimistic=True)
            if sites == self._sites:
                break
            self._sites = sites
        final = self._site_types(keys, optimistic=False)
        if final != self._sites:
            self._sites = final
            self._infer(keys)
        if report:
            self._report(keys, root_of)

    def _infer(self, keys: list[FunctionKey]) -> None:
        """Locals, returns, fields and globals over keys, from the current parameter types."""
        live = self.live.fields if self.live is not None else {}
        for c in self.classes.values():
            c.fields = {k: t for k, t in c.fields.items() if k in c.annotated}
            for k, t in live.get(c.name, {}).items():
                c.fields.setdefault(k, t)
        for info in self.functions.values():
            info.locals, info.annotated, info.returns = {}, set(), NEVER
            info.param_types = dict(info.declared)
            for p, t in info.declared.items():
                if not t.known and (info.key, p) in self._sites:
                    info.param_types[p] = self._sites[(info.key, p)]
        self.globals = dict(self.live.globals) if self.live is not None else {}
        for _ in range(self.MAX_PASSES):
            changed = self._globals_pass()
            for key in keys:
                changed |= self._function_pass(self.functions[key])
            if not changed:
                return

    def _site_types(self, keys, optimistic: bool) -> dict[tuple[FunctionKey, str], Ty]:
        """Each parameter's type as the join of what reached call sites pass it and its
        default. `optimistic` leaves unknown arguments out, and types a parameter whose
        arguments are all unknown as not typed yet (the empty type), so a cycle through it
        (a field fed by a call that reads the field) settles instead of staying unknown."""
        found: dict[tuple[FunctionKey, str], list[Ty]] = {}
        for key in keys:
            info = self.functions[key]
            for n, ctx in self.walk(info):
                if isinstance(n, ast.Call):
                    for callee, skip in self.callees(n, ctx):
                        self._bind_args(n, callee, skip, ctx, found)
            outer = self.ctx_of(self.functions[info.parent]) if info.parent else Ctx(key[0])
            for p, d in info.defaults.items():
                found.setdefault((key, p), []).append(self.expr(d, outer))
        out = {}
        for (key, p), ts in found.items():
            if self.functions[key].declared.get(p, UNKNOWN).known:
                continue
            known = [t for t in ts if t.known]
            out[(key, p)] = (union(*known) if known else NEVER) if optimistic else union(*ts)
        return out

    def _bind_args(self, call: ast.Call, info: FuncInfo, skip: int, ctx: Ctx, found) -> None:
        params = info.params[skip:]
        positional = len(info.node.args.posonlyargs) + len(info.node.args.args) - skip
        for i, a in enumerate(call.args):
            if isinstance(a, ast.Starred) or i >= positional:
                break
            found.setdefault((info.key, params[i]), []).append(self.expr(a, ctx))
        for k in call.keywords:
            if k.arg in params:
                found.setdefault((info.key, k.arg), []).append(self.expr(k.value, ctx))

    def callees(self, call: ast.Call, ctx: Ctx) -> list[tuple[FuncInfo, int]]:
        """The app functions `call` may run, each with how many leading parameters the call
        does not pass (1 for a method's self or a constructor's)."""
        f = call.func
        if isinstance(f, ast.Name) and not self.is_local(f.id, ctx):
            return self._symbol_callees(self.resolve(ctx.module, f.id))
        if isinstance(f, ast.Attribute):
            if self.lib_name(f, ctx) is not None:
                return []
            if isinstance(f.value, ast.Name) and not self.is_local(f.value.id, ctx):
                sym = self.resolve(ctx.module, f.value.id)
                if sym is not None and sym.kind == "module":
                    return self._symbol_callees(self.resolve(sym.target, f.attr))
            methods = self.methods(self.expr(f.value, ctx), f.attr)
            if methods:
                return [(m, 1) for m in methods]
        return self.value_callees(self.expr(f, ctx))

    def _symbol_callees(self, sym: Symbol | None) -> list[tuple[FuncInfo, int]]:
        if sym is not None and sym.kind == "func":
            return [(self.functions[sym.target], 0)]
        if sym is not None and sym.kind == "class" and sym.target in self.classes:
            init = self.classes[sym.target].methods.get("__init__")
            return [(self.functions[init], 1)] if init else []
        return []

    def value_callees(self, t: Ty) -> list[tuple[FuncInfo, int]]:
        """What calling a value of type t runs: a named function (a method's value is bound,
        so its self is not passed), an object's __call__."""
        out = []
        labels = getattr(self, "_labels", None) or {key_label(k): k for k in self.functions}
        for m in t.members:
            if m.kind == "fn" and m.name in labels:
                info = self.functions[labels[m.name]]
                out.append((info, 1 if info.cls else 0))
            elif m.kind == "obj" and m.name in self.classes:
                call = self.classes[m.name].methods.get("__call__")
                if call is not None:
                    out.append((self.functions[call], 1))
        return out

    def walk(self, info: FuncInfo):
        """(node, ctx) for every node of info's body, each in the Ctx it is read in: a
        lambda's body and a comprehension's parts see the names they bind. A def inside is
        its own function: only its defaults are walked here."""
        ctx = self.ctx_of(info)
        for st in info.node.body:
            yield from self._walk(st, ctx)

    def _walk(self, n: ast.AST, ctx: Ctx):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in [*n.args.defaults, *(d for d in n.args.kw_defaults if d is not None)]:
                yield from self._walk(d, ctx)
            return
        if isinstance(n, ast.ClassDef):
            return
        yield n, ctx
        if isinstance(n, ast.Lambda):
            for d in n.args.defaults:
                yield from self._walk(d, ctx)
            yield from self._walk(n.body, ctx.bind(self.lambda_params(n, ctx)))
            return
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            c = ctx
            for g in n.generators:
                yield from self._walk(g.iter, c)
                names: dict[str, Ty] = {}
                bind_target(g.target, elem(self.expr(g.iter, c)), names)
                c = c.bind(names)
                for test in g.ifs:
                    yield from self._walk(test, c)
            for part in (n.key, n.value) if isinstance(n, ast.DictComp) else (n.elt,):
                yield from self._walk(part, c)
            return
        for child in ast.iter_child_nodes(n):
            yield from self._walk(child, ctx)

    def _report(self, keys, root_of) -> None:
        """Raise for the first operator, in keys' order, whose operand type is unknown:
        + * % (and their augmented forms), == and != unless the other side is a known
        scalar, and an f-string field with no format spec."""
        for key in keys:
            for n, ctx in self.walk(self.functions[key]):
                bad = self._unknown_operand(n, ctx)
                if bad is not None:
                    raise TranspileError(
                        module_path(key[0]),
                        n.lineno,
                        bad,
                        root_of(key) if root_of else None,
                    )

    def _unknown_operand(self, n: ast.AST, ctx: Ctx) -> str | None:
        def unknown(x):
            return not self.expr(x, ctx).known

        def msg(x):
            return (
                f"`{ast.unparse(n)}` with an operand of unknown type (annotate `{ast.unparse(x)}`)"
            )

        if isinstance(n, (ast.BinOp, ast.AugAssign)) and isinstance(
            n.op, (ast.Add, ast.Mult, ast.Mod)
        ):
            left = n.left if isinstance(n, ast.BinOp) else n.target
            right = n.right if isinstance(n, ast.BinOp) else n.value
            for side in (left, right):
                if unknown(side):
                    return msg(side)
        if isinstance(n, ast.Compare):
            nodes = [n.left, *n.comparators]
            for i, op in enumerate(n.ops):
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    a, b = nodes[i], nodes[i + 1]
                    for x, other in ((a, b), (b, a)):
                        o = self.expr(other, ctx)
                        if unknown(x) and not (o.known and o.is_scalar):
                            return msg(x)
        if isinstance(n, ast.FormattedValue) and n.format_spec is None and unknown(n.value):
            x = ast.unparse(n.value)
            return f"the f-string field `{{{x}}}` of unknown type (annotate `{x}`)"
        return None

    def _globals_pass(self) -> bool:
        changed = False
        for module, tree in self.modules.items():
            ctx = Ctx(module)
            for n in tree.body:
                if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                    found: dict[str, Ty] = {}
                    if isinstance(n, ast.AnnAssign):
                        bind_target(n.target, self.annotation(n.annotation, module), found)
                    else:
                        for t in n.targets:
                            bind_target(t, self.expr(n.value, ctx), found)
                    live = self.live.globals if self.live is not None else {}
                    for name, t in found.items():
                        if _vague(t) and (module, name) in live:
                            continue  # setup's value crosses: what it holds is its type
                        new = join(self.globals.get((module, name), NEVER), t)
                        if new != self.globals.get((module, name)):
                            self.globals[(module, name)] = new
                            changed = True
        return changed

    def _function_pass(self, info: FuncInfo) -> bool:
        cls = self.classes.get(info.cls) if info.cls else None
        before = (dict(info.locals), info.returns, dict(cls.fields) if cls else None)
        ctx = self.ctx_of(info)

        def field_(target: ast.expr, t: Ty):
            """`self.x = v` in a method types the field x."""
            if (
                cls is not None
                and isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == info.self_name
                and target.attr not in cls.annotated
            ):
                cls.fields[target.attr] = join(cls.fields.get(target.attr, NEVER), t)

        def assign(names: dict[str, Ty]):
            for k, t in names.items():
                if k in info.annotated or k in info.param_types:
                    continue
                info.locals[k] = join(info.locals.get(k, NEVER), t)

        for st in own_statements(info.node.body):
            if isinstance(st, ast.Assign):
                t = self.expr(st.value, ctx)
                for target in st.targets:
                    found: dict[str, Ty] = {}
                    bind_target(target, t, found)
                    assign(found)
                    field_(target, t)
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                info.locals[st.target.id] = self.annotation(st.annotation, info.key[0])
                info.annotated.add(st.target.id)
            elif isinstance(st, ast.AnnAssign) and cls is not None:
                field_(st.target, self.annotation(st.annotation, info.key[0]))
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Attribute):
                old = self.expr(st.target, ctx)
                if old.kind not in ("vec", "nd"):
                    field_(st.target, self.binop(st.op, old, self.expr(st.value, ctx)))
            elif isinstance(st, ast.FunctionDef):
                inner = self.functions[(info.key[0], f"{info.key[1]}.<locals>.{st.name}")]
                assign({st.name: fn(key_label(inner.key), inner.returns)})
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                old = self.expr(st.target, ctx)
                if old.kind in ("vec", "nd", "field"):
                    continue  # numpy's += writes into the array (a field's queues): it stays
                assign({st.target.id: self.binop(st.op, old, self.expr(st.value, ctx))})
            elif isinstance(st, ast.For):
                found = {}
                bind_target(st.target, elem(self.expr(st.iter, ctx)), found)
                assign(found)
            elif isinstance(st, ast.Return) and not info.declared_returns.known:
                t = self.expr(st.value, ctx) if st.value is not None else NONE
                info.returns = join(info.returns, t)
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                self._mutation(st.value, ctx, assign)
        if info.declared_returns.known:
            info.returns = info.declared_returns
        elif not ends_in_return(info.node.body):
            info.returns = join(info.returns, NONE)
        return (info.locals, info.returns, dict(cls.fields) if cls else None) != before

    def _mutation(self, call: ast.Call, ctx: Ctx, assign) -> None:
        """`xs.append(v)` widens a local list's element type to v's, `s.add(v)` a set's."""
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and call.args:
            name = f.value.id
            recv = self.expr(f.value, ctx)
            if recv.all("list") and f.attr in ("append", "insert"):
                assign({name: list_of(self.expr(call.args[-1], ctx))})
            elif recv.all("list") and f.attr == "extend":
                assign({name: list_of(elem(self.expr(call.args[0], ctx)))})
            elif recv.all("set") and f.attr == "add":
                assign({name: set_of(self.expr(call.args[0], ctx))})
            elif recv.all("set") and f.attr == "update":
                assign({name: set_of(union(*(elem(self.expr(a, ctx)) for a in call.args)))})


def _is_ellipsis(n: ast.expr) -> bool:
    return isinstance(n, ast.Constant) and n.value is Ellipsis


def is_classvar(annotation: ast.expr) -> bool:
    head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return ast.unparse(head) in ("ClassVar", "typing.ClassVar")


def target_names(t: ast.expr) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [n for e in t.elts for n in target_names(e)]
    if isinstance(t, ast.Starred):
        return target_names(t.value)
    return []


def bind_target(t: ast.expr, ty: Ty, out: dict[str, Ty]) -> None:
    """The names an assignment target binds, with the types it gives them."""
    if isinstance(t, ast.Name):
        out[t.id] = join(out.get(t.id, NEVER), ty)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for i, e in enumerate(t.elts):
            items = [
                m.args[i] if m.kind == "tuple" and len(m.args) == len(t.elts) else elem(m)
                for m in ty.members
            ]
            bind_target(e, union(*items), out)  # nothing yet (a call's return): wait


def own_statements(body: list[ast.stmt]):
    """Every statement in body, entering blocks but not nested functions or classes."""
    for st in body:
        yield st
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for name in ("body", "orelse", "finalbody"):
            yield from own_statements(getattr(st, name, None) or [])
        for h in getattr(st, "handlers", None) or []:
            yield from own_statements(h.body)


def ends_in_return(body: list[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[-1], ast.Return)
