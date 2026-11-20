"""The mixer: sources placed around a listener, summed into buses, then through a limiter.

Everything here is numpy on blocks of samples, so it runs and is tested without a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

RATE = 48_000
BUSES = ("engine", "effects", "ambience", "music", "ui")
CEILING = 10 ** (-1 / 20)  # the limiter's ceiling, -1 dBFS
SOUND = 343.0  # m/s
MAX_RADIAL = 60.0  # m/s: the most a source may approach or recede for Doppler
LOOKAHEAD = 240  # samples, 5 ms
RELEASE = 0.05  # seconds for the limiter's gain to rise from 0 to 1
RAMP = 0.05  # seconds for a bus's gain to reach a new value


@dataclass
class Pose:
    """A position and a velocity in metres and m/s, and a heading: at yaw 0 it faces +x, and a
    positive yaw turns it toward +z, which is its right."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw: float = 0.0


def _box(x: np.ndarray, k: int, tail: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """x through a moving average of k samples (a gentle low-pass), with the last k - 1 samples
    of the previous block as tail; returns the filtered block and the new tail."""
    if k <= 1:
        return x, x[-1:][:0] if tail is None else tail[:0]
    pre = np.zeros((k - 1,) + x.shape[1:], x.dtype) if tail is None or len(tail) != k - 1 else tail
    full = np.concatenate([pre, x])
    c = np.cumsum(full, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), c])
    y = (c[k:] - c[:-k]) / k
    return y.astype(np.float32), full[-(k - 1):]


def _box_len(cutoff: float) -> int:
    """The moving average whose first zero sits near twice the cutoff: open at 20 kHz."""
    return 1 if cutoff >= 19_999 else max(1, round(RATE / (2 * cutoff)))


class Placed:
    """A mono source heard from a pose: louder near, panned by bearing, duller far away and
    shifted in pitch by its speed toward the listener.

    ref: full level within ref metres, then ref / d. far: silent past it, fading over its last
    tenth. Doppler reads the velocities, never the positions, so a teleport changes no pitch.
    """

    def __init__(self, source, ref: float = 5.0, far: float = 300.0):
        self.source, self.ref, self.far = source, ref, far
        self._buf = np.zeros(0, np.float32)
        self._phase = 0.0  # where the next output sample reads in _buf
        self._air = None

    def _pull(self, need: int) -> None:
        if need > len(self._buf):
            x = np.asarray(self.source.block(need - len(self._buf)), np.float32)
            if x.ndim == 2:
                x = x.mean(axis=1)
            self._buf = np.concatenate([self._buf, x])

    def block(self, n: int, src: Pose, lis: Pose) -> np.ndarray:
        d = np.array([src.x - lis.x, src.y - lis.y, src.z - lis.z])
        dist = float(np.linalg.norm(d))
        u = -d / dist if dist > 1e-6 else np.zeros(3)  # from the source toward the listener
        rel = np.array([src.vx - lis.vx, src.vy - lis.vy, src.vz - lis.vz])
        v = float(np.clip(rel @ u, -MAX_RADIAL, MAX_RADIAL))
        ratio = SOUND / (SOUND - v)
        pos = self._phase + np.arange(n) * ratio
        self._pull(int(math.floor(pos[-1])) + 2)
        i = np.floor(pos).astype(int)
        f = (pos - i).astype(np.float32)
        x = self._buf[i] * (1 - f) + self._buf[i + 1] * f
        self._phase = pos[-1] + ratio
        drop = int(math.floor(self._phase))
        self._buf, self._phase = self._buf[drop:], self._phase - drop

        gain = 1.0 if dist <= self.ref else self.ref / dist
        if dist >= self.far:
            gain = 0.0
        elif dist > 0.9 * self.far:
            gain *= (self.far - dist) / (0.1 * self.far)
        cutoff = 20_000.0 if dist <= 50 else max(2000.0, 20_000.0 * 50 / dist)
        x, self._air = _box(x, _box_len(cutoff), self._air)
        right = np.array([-math.sin(lis.yaw), 0.0, math.cos(lis.yaw)])
        p = float(-u @ right) if dist > 1e-6 else 0.0
        a = (p + 1) * math.pi / 4
        return np.stack([x * (gain * math.cos(a)), x * (gain * math.sin(a))], 1).astype(np.float32)


class Bus:
    """A sum of blocks with a gain that moves to a new value over 50 ms, and a duck."""

    def __init__(self, name: str):
        self.name = name
        self.target = 1.0
        self._g = 1.0
        self._acc: np.ndarray | None = None
        self._duck = 1.0
        self._duck_left = 0  # samples the duck still holds
        self.level = -120.0  # dBFS of the last block, for the state

    def add(self, block: np.ndarray) -> None:
        b = np.asarray(block, np.float32)
        if b.ndim == 1:
            b = np.stack([b, b], 1)
        self._acc = b.copy() if self._acc is None else self._acc + b

    def duck(self, db: float, seconds: float) -> None:
        self._duck, self._duck_left = 10 ** (-db / 20), round(seconds * RATE)

    def take(self, n: int) -> np.ndarray:
        x = self._acc if self._acc is not None and len(self._acc) == n else np.zeros((n, 2), np.float32)
        self._acc = None
        want = self.target * (self._duck if self._duck_left > 0 else 1.0)
        self._duck_left = max(0, self._duck_left - n)
        new = self._g + (want - self._g) * min(1.0, n / (RAMP * RATE))
        g = np.linspace(self._g, new, n, endpoint=False, dtype=np.float32) if new != self._g else self._g
        self._g = new
        y = x * (g[:, None] if isinstance(g, np.ndarray) else g)
        r = float(np.sqrt(np.mean(np.square(y)))) if n else 0.0
        self.level = 20 * math.log10(r) if r > 1e-6 else -120.0
        return y


class Mixer:
    """Five buses summed, a master low-pass and a lookahead limiter at -1 dBFS."""

    def __init__(self, buses=BUSES):
        self.buses = {b: Bus(b) for b in buses}
        self._cut = 20_000.0
        self._air = None
        self._delay = np.zeros((LOOKAHEAD, 2), np.float32)
        self._lim = 1.0

    def __getitem__(self, bus: str) -> Bus:
        return self.buses[bus]

    def gain(self, bus: str, value: float) -> None:
        self.buses[bus].target = float(value)

    def duck(self, bus: str, db: float, seconds: float) -> None:
        self.buses[bus].duck(db, seconds)

    def lowpass(self, cutoff_hz: float) -> None:
        """Muffle everything (under water, say); 20000 opens it again."""
        self._cut = float(cutoff_hz)

    def out(self, n: int) -> np.ndarray:
        x = sum(b.take(n) for b in self.buses.values())
        x, self._air = _box(x, _box_len(self._cut), self._air)
        return self._limit(x)

    def _limit(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        full = np.concatenate([self._delay, x])  # what plays now is LOOKAHEAD samples old
        peak = sliding_window_view(np.abs(full).max(axis=1), LOOKAHEAD + 1).max(axis=1)
        need = np.minimum(1.0, CEILING / np.maximum(peak, 1e-9))
        r = 1.0 / (RELEASE * RATE)
        k = np.arange(n)
        g = np.minimum(np.minimum.accumulate(need - r * k) + r * k, self._lim + r * (k + 1))
        g = np.minimum(g, 1.0)
        self._lim = float(g[-1])
        self._delay = full[n:]
        return (full[:n] * g[:, None]).astype(np.float32)
