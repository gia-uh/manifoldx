"""A system that moves a root panel 5 px right each frame, rewrites its Text and widens it:
the page's DOM follows each write (gia-uh/manifoldx#147)."""

import manifoldx as mx
from manifoldx.gui import Panel, Text

engine = mx.Engine("Moving panel", width=320, height=200)


class Hud:
    def __init__(self):
        self.label = Text("frame 0")
        self.panel = Panel(children=[self.label], offset=(10, 20))
        self.n = 0


hud = Hud()
engine.gui.append(hud.panel)


def tick(query, dt: float):
    hud.n += 1
    x, y = hud.panel.offset
    hud.panel.offset = (x + 5, y)
    hud.label.text = f"frame {hud.n}"
    hud.panel.style_overrides["width"] = 100 + hud.n


engine.system(tick)
engine.run()
