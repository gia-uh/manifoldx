"""Audio: a tone circling the listener at 20 m and 15 m/s, so its pan, its level and its Doppler shift can be heard.

    uv run --extra audio python examples/audio.py            # through the speakers, for 10 s
    uv run python examples/audio.py --out /tmp/circle.wav    # into a file instead
"""

import argparse
import math
import time

import numpy as np

from manifoldx.audio import Audio, pose_names


class Tone:
    """A soft square at 330 Hz: odd harmonics, so the air filter and the pan are easy to hear."""

    def __init__(self, ctl):
        self.i = 0

    def block(self, n):
        t = (self.i + np.arange(n)) / 48000
        self.i += n
        return (0.2 * np.sign(np.sin(2 * math.pi * 330 * t))).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", help="write a WAV instead of playing")
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    audio = Audio(backend="file" if a.out else "device", path=a.out, controls=pose_names("tone"))
    audio.source("tone", Tone, "effects", placed=("tone", 5.0, 300.0))
    r, w, t0 = 20.0, 15.0 / 20.0, time.monotonic()
    while (t := time.monotonic() - t0) < a.seconds:
        for k, v in zip(("x", "z", "vx", "vz"),
                        (r * math.cos(w * t), r * math.sin(w * t), -r * w * math.sin(w * t), r * w * math.cos(w * t))):
            audio.set(f"tone.{k}", v)
        audio.tick()
        time.sleep(1 / 60)
    print(audio.state())
    audio.close()


if __name__ == "__main__":
    main()
