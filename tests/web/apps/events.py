"""The event queue in the desktop's order: tick emits ping on frames 3 and 4; on_ping, run the
next frame, records the frame it saw and emits pong for the second; a host page's host:hello
is counted and answered with app:ack; on_frame keeps the frame
event's payload, and after_frame, registered after it, sees what on_frame wrote. One handler per
input event keeps its payload's fields, unannotated, so the payload types from its event."""

import manifoldx as mx

engine = mx.Engine("Events", width=320, height=200)
engine.state["k"] = 0
engine.state["pings"] = []
engine.state["pongs"] = 0
engine.state["frame"] = -1
engine.state["elapsed"] = -1.0
engine.state["dt"] = 0.0
engine.state["seen"] = []
engine.state["hellos"] = 0
for name in ("key_down", "key_up", "pointer_down", "pointer_up", "pointer_move", "wheel", "resize"):
    engine.state[name] = []
engine.state["mouse_at"] = [0.0, 0.0]


def tick(q, dt: float):
    k: int = engine.state["k"]
    if k == 3 or k == 4:
        engine.emit("ping", {"n": k})
    engine.state["k"] = k + 1


def on_ping(p: dict):
    f: int = engine.state["frame"]
    n: int = p["n"]
    pings: list[list[int]] = engine.state["pings"]
    pings.append([f, n])
    if n == 4:
        engine.emit("pong", {})


def on_pong(p: dict):
    pongs: int = engine.state["pongs"]
    engine.state["pongs"] = pongs + 1


def on_host_hello(p: dict):
    hellos: int = engine.state["hellos"]
    engine.state["hellos"] = hellos + 1
    n: int = p["n"]
    engine.emit("app:ack", {"n": n})


def on_frame(p: dict):
    engine.state["frame"] = p["frame"]
    engine.state["elapsed"] = p["elapsed"]
    engine.state["dt"] = p["dt"]


def after_frame(p: dict):
    f: int = engine.state["frame"]
    seen: list[int] = engine.state["seen"]
    seen.append(f)


def on_key_down(ev):
    engine.state["key_down"] = [ev.key, ev.modifiers, ev.is_down]


def on_key_up(ev):
    engine.state["key_up"] = [ev.key, ev.modifiers, ev.is_down]


def on_pointer_down(ev):
    engine.state["pointer_down"] = [ev.x, ev.y, ev.button, ev.buttons, ev.phase]
    engine.state["mouse_at"] = engine.input.mouse_pos


def on_pointer_up(ev):
    engine.state["pointer_up"] = [ev.x, ev.y, ev.button, ev.buttons, ev.phase, ev.pointer_id]


def on_pointer_move(ev):
    engine.state["pointer_move"] = [ev.x, ev.y, ev.dx, ev.dy, ev.modifiers, ev.pointer_id]


def on_wheel(ev):
    engine.state["wheel"] = [ev.dx, ev.dy, ev.x, ev.y, ev.buttons, ev.modifiers]


def on_resize(ev):
    engine.state["resize"] = [ev.width, ev.height, ev.pixel_ratio]


engine.system(tick)
engine.on("ping")(on_ping)
engine.on("pong")(on_pong)
engine.on("host:hello")(on_host_hello)
engine.on("frame")(on_frame)
engine.on("frame")(after_frame)
engine.on("key_down")(on_key_down)
engine.on("key_up")(on_key_up)
engine.on("pointer_down")(on_pointer_down)
engine.on("pointer_up")(on_pointer_up)
engine.on("pointer_move")(on_pointer_move)
engine.on("wheel")(on_wheel)
engine.on("resize")(on_resize)
engine.run()
