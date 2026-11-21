"""Lines over the scene, a fixed width in screen pixels at any distance: roads on a map, a route, a path.

`engine.lines` holds layers of segments by key. Each segment runs from a to b in the render frame (the same
coordinates as a Transform's position), with a colour (linear RGBA) and a width in screen pixels. The lines pass
draws them after the scene and before the GUI, each layer in the order it was first set: a dark casing set before
a lighter fill draws under it. The depth test still applies, with each line brought `bias` of its distance toward
the camera, so a line lying on the ground is never cut by the ground under it while a hill in front still hides
it; the camera's vertical_scale stretches lines as it does the meshes. The edges fade over `feather` pixels.

    engine.lines.set("roads", a, b, color=(0.1, 0.1, 0.1, 1.0), width=3.0)   # a, b: (N, 3); color (4,) or (N, 4)
    engine.lines.remove("roads")
"""

from __future__ import annotations

import numpy as np

STRIDE = 12  # floats a segment takes on the GPU: a.xyz, width, b.xyz, 0, rgba


class Lines:
    def __init__(self) -> None:
        self.layers: dict[str, np.ndarray] = {}  # key -> (N, STRIDE) float32, packed
        self.versions: dict[str, int] = {}  # key -> a count bumped on each set, for the pass's uploads
        self.bias = 0.003  # share of a line's distance it is brought toward the camera for the depth test
        self.feather = 1.0  # screen pixels over which an edge fades
        self.visible = True

    def set(self, key: str, a, b, color, width) -> None:
        """Make the layer `key` the segments from a[i] to b[i] (render frame, (N, 3)), each `color` (linear RGBA,
        one (4,) for all or (N, 4)) and `width` screen pixels wide (a number or (N,))."""
        a = np.asarray(a, np.float32).reshape(-1, 3)
        b = np.asarray(b, np.float32).reshape(-1, 3)
        if len(a) != len(b):
            raise ValueError(f"a and b differ in length: {len(a)} and {len(b)}")
        n = len(a)
        packed = np.zeros((n, STRIDE), np.float32)
        packed[:, 0:3], packed[:, 4:7] = a, b
        packed[:, 3] = np.broadcast_to(np.asarray(width, np.float32), (n,))
        packed[:, 8:12] = np.broadcast_to(np.asarray(color, np.float32), (n, 4))
        self.layers[key] = packed
        self.versions[key] = self.versions.get(key, 0) + 1

    def remove(self, key: str) -> None:
        self.layers.pop(key, None)
        self.versions.pop(key, None)

    def clear(self) -> None:
        self.layers.clear()
        self.versions.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self.layers.values())
