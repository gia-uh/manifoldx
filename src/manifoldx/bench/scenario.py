"""Scenarios and steps: what an app declares for `manifoldx bench run`.

A scenario is a function of an engine and its params that builds the app and returns it with a list
of steps. A step is a `StepCall`, a name and its arguments, resolved in `STEPS` when it runs. The
engine's own steps (manifoldx.bench.steps) get a `Ctx`; the steps an app registers with `@step` get
the app. A step function that is a generator takes one frame per `yield`; a plain one takes none.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator


@dataclass
class Scenario:
    """A named function `fn(engine, params) -> (app, steps)`, its default params and its sizes."""

    name: str
    fn: Callable
    params: dict = field(default_factory=dict)
    sizes: dict = field(default_factory=lambda: {"smoke": 30, "full": None})
    doc: str = ""


@dataclass
class StepCall:
    """One step of a scenario: the name of a registered step and its arguments."""

    name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class StepDef:
    name: str
    fn: Callable
    engine: bool  # True: an engine step, called with the Ctx; False: an app step, called with the app


SCENARIOS: dict[str, Scenario] = {}
STEPS: dict[str, StepDef] = {}


class StepTimeout(Exception):
    """A step waited longer than its max_s; the message names the step and its index."""


class Ctx:
    """What an engine step sees: the engine, the app, the run it records into, and its place."""

    def __init__(self, engine, app, run):
        self.engine, self.app, self.run = engine, app, run
        self.index, self.step_name = 0, ""
        self.dt = 1 / 60  # the last frame's dt; the runner keeps it current
        self.t = 0.0  # seconds of game time: the sum of dt over the frames so far

    def timeout(self, max_s: float):
        raise StepTimeout(f"step {self.step_name} #{self.index} did not finish in {max_s:g} s")


def scenario(name: str, *, params: dict | None = None, sizes: dict | None = None):
    """Register `fn(engine, params) -> (app, steps)` as scenario `name`."""

    def wrap(fn):
        SCENARIOS[name] = Scenario(name, fn, dict(params or {}), {"smoke": 30, "full": None, **(sizes or {})},
                                   (fn.__doc__ or "").strip().split("\n")[0])
        return fn

    return wrap


def step(name: str):
    """Register an app step: `fn(app, *args, **kwargs)`, plain or a generator yielding once a frame."""

    def wrap(fn):
        STEPS[name] = StepDef(name, fn, engine=False)

        def call(*args, **kwargs):
            return StepCall(name, args, kwargs)

        call.__name__, call.__doc__, call.fn = fn.__name__, fn.__doc__, fn
        return call

    return wrap


def engine_step(name: str):
    """Register one of the engine's steps; it is called with the Ctx."""

    def wrap(fn):
        STEPS[name] = StepDef(name, fn, engine=True)
        return fn

    return wrap


def iterate(call: StepCall, ctx: Ctx) -> Iterator[None]:
    """Run one step; yields once for each frame it needs rendered."""
    d = STEPS.get(call.name)
    if d is None:
        raise ValueError(f"no step named {call.name!r}; known: {', '.join(sorted(STEPS))}")
    ctx.step_name = call.name
    out = d.fn(ctx if d.engine else ctx.app, *call.args, **call.kwargs)
    if inspect.isgenerator(out):
        yield from out


_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


class Params:
    """A scenario's params: its defaults, with string overrides parsed to each default's type."""

    def __init__(self, defaults: dict, overrides: dict | None = None):
        values = dict(defaults)
        for k, v in (overrides or {}).items():
            if k not in defaults:
                raise ValueError(f"no param {k!r}; this scenario takes: {', '.join(sorted(defaults)) or 'none'}")
            values[k] = _parse(k, v, defaults[k])
        self.__dict__.update(values)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _parse(key, value, default):
    if not isinstance(value, str):
        return value
    if "," in value and not isinstance(default, str):
        raise ValueError(f"param {key}: one value only; sweeps over several values come in a later step")
    if isinstance(default, bool):
        if value.lower() in _TRUE | _FALSE:
            return value.lower() in _TRUE
        raise ValueError(f"param {key}: {value!r} is not a boolean")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _call_of(entry: dict) -> StepCall:
    ((name, a),) = entry.items()
    if isinstance(a, dict):
        return StepCall(name, (), dict(a))
    if isinstance(a, list):
        return StepCall(name, tuple(a), {})
    return StepCall(name, (a,), {})


def load_json(path: Path) -> Scenario:
    """A scenario from a JSON file: {"name", "app": "module:function", "params", "sizes", "steps"}.

    `app` is called as function(engine, params) and returns the app; each step is {name: args}, where
    args is a dict (keywords), a list (positional) or one value."""
    data = json.loads(Path(path).read_text())
    module, _, func = data["app"].partition(":")
    steps = [_call_of(e) for e in data.get("steps", [])]

    def fn(engine, p):
        make = getattr(importlib.import_module(module), func)
        return make(engine, p), list(steps)

    return Scenario(data.get("name", Path(path).stem), fn, dict(data.get("params", {})),
                    {"smoke": 30, "full": None, **data.get("sizes", {})}, data.get("doc", ""))
