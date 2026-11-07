"""The event queue in the desktop's order: tick emits ping on frames 3 and 4; on_ping, run the
next frame, records the frame it saw and emits pong for the second; on_frame keeps the frame
event's payload, and after_frame, registered after it, sees what on_frame wrote."""

import manifoldx as mx

engine = mx.Engine("Events", width=320, height=200)
engine.state["k"] = 0
engine.state["pings"] = []
engine.state["pongs"] = 0
engine.state["frame"] = -1
engine.state["elapsed"] = -1.0
engine.state["dt"] = 0.0
engine.state["seen"] = []


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


def on_frame(p: dict):
    engine.state["frame"] = p["frame"]
    engine.state["elapsed"] = p["elapsed"]
    engine.state["dt"] = p["dt"]


def after_frame(p: dict):
    f: int = engine.state["frame"]
    seen: list[int] = engine.state["seen"]
    seen.append(f)


engine.system(tick)
engine.on("ping")(on_ping)
engine.on("pong")(on_pong)
engine.on("frame")(on_frame)
engine.on("frame")(after_frame)
engine.run()
