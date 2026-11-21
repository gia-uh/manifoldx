"""Audio: the cost of mixing one block of n placed sources, as the sound process does 94 times a second."""

import numpy as np

from manifoldx.audio import Mixer, Placed, Pose

from benchmarks.harness import case, measure

BLOCK = 512


class _Tone:
    def __init__(self, f):
        self.f, self.i = f, 0

    def block(self, n):
        t = (self.i + np.arange(n)) / 48000
        self.i += n
        return (0.05 * np.sin(2 * np.pi * self.f * t)).astype(np.float32)


@case("audio", "mix_placed", sizes=[4, 8, 30], smoke=4)
def mix_placed(n):
    """One 512-sample block of n sources moving around a listener, through the buses and the limiter."""
    mixer, listener = Mixer(), Pose(0, 0, 0, 0, 0, 0)
    sources = [(Placed(_Tone(200 + 37 * i)), Pose(20 + 5 * i, 0, 10 * (i % 3 - 1), -8, 0, 0)) for i in range(n)]

    def step():
        for p, pose in sources:
            mixer["effects"].add(p.block(BLOCK, pose, listener))
        mixer.out(BLOCK)

    step()
    return measure(step)
