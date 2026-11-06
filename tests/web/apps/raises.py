"""A system that reads an attribute of None on its sixth frame: the page's error panel names
this file and line."""

import manifoldx as mx

engine = mx.Engine("Raises", width=320, height=200)


class Spot:
    def __init__(self):
        self.x = 1.0


class Counter:
    def __init__(self):
        self.n = 0
        self.spot: Spot | None = None  # never set


counter = Counter()


def tick(query, dt: float):
    counter.n = counter.n + 1
    if counter.n > 5:
        engine.state["x"] = counter.spot.x  # the line the page names


engine.system(tick)

if __name__ == "__main__":
    engine.cli()
