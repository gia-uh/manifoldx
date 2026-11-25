"""Public authoring inputs for the native 120 Hz vehicle fleet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace
from typing import NamedTuple, Sequence

import numpy as np

from .assists import ASSISTS, apply_assists
from .body import matrices
from .fleet import Vehicles
from .ground import FlatGround, ShapeGround
from .spec import VehicleSpec
from .state import pack


@dataclass(frozen=True)
class Heightfield:
    """Regular XZ samples. Heights are indexed as rows of Z and columns of X."""

    origin: tuple[float, float]
    spacing: tuple[float, float]
    heights: Sequence[Sequence[float]]
    grip: float = 1.0
    level: int = 0


@dataclass(frozen=True)
class GroundTriangle:
    """One XZ-projectable triangle with world XYZ vertices and a surface level ID."""

    vertices: Sequence[Sequence[float]]
    grip: float = 1.0
    level: int = 0


@dataclass(frozen=True)
class Obstacle:
    """Static XZ segment with a finite vertical interval and strength in [0, 1]."""

    ends: Sequence[Sequence[float]]
    y: tuple[float, float]
    strength: float = 1.0


@dataclass(frozen=True)
class GroundTile:
    """Geometry in one application-space cell, with a positive edit version."""

    key: tuple[int, int]
    version: int
    heightfields: Sequence[Heightfield] = ()
    triangles: Sequence[GroundTriangle] = ()
    obstacles: Sequence[Obstacle] = ()


@dataclass(frozen=True)
class TiledGround:
    """Geometry input copied on registration. A cell (i, j) starts at (i, j) × cell_size."""

    cell_size: float
    tiles: Sequence[GroundTile]


class GroundSample(NamedTuple):
    """Nearest surface under XZ, including its identity in the published tile version."""

    height: float
    normal: tuple[float, float, float]
    grip: float
    level: int
    tile: tuple[int, int]
    version: int


def _tile(tile):
    grids = []
    for grid in tile.heightfields:
        heights = np.asarray(grid.heights, dtype=float)
        if heights.ndim != 2:
            raise ValueError("heightfield needs a two-dimensional grid")
        nz, nx = heights.shape
        grids.append((grid.origin, grid.spacing, (nx, nz), heights.ravel().tolist(), grid.grip, grid.level))
    triangles = [(t.vertices, t.grip, t.level) for t in tile.triangles]
    obstacles = [(o.ends, o.y, o.strength) for o in tile.obstacles]
    return (tile.key, tile.version, grids, triangles, obstacles)


class NativeVehicles:
    """A Rust-owned fleet with window-independent stepping and no Python ground callbacks.

    Set per-frame controls with advance(). Use snapshot() for owned state readback
    and checkpoint()/restore() for replay. A Python Ground must be converted to
    TiledGround before use. FlatGround and ShapeGround supply built-in presets.
    """

    DT = Vehicles.DT
    MAX_FRAME = Vehicles.MAX_FRAME

    def __init__(
        self,
        specs: Sequence[VehicleSpec],
        ground: FlatGround | ShapeGround | TiledGround,
        *,
        contact_friction: float = 0.4,
    ):
        if type(ground) not in (FlatGround, ShapeGround, TiledGround):
            raise TypeError("native vehicles require FlatGround, ShapeGround or TiledGround")
        try:
            from manifoldx import _core_native
        except ImportError as error:
            raise RuntimeError("NativeVehicles requires the native ManifoldX wheel") from error
        if type(ground) is TiledGround:
            setup = {"terrain": _core_native._VehicleTerrain(ground.cell_size, [_tile(t) for t in ground.tiles])}
        elif type(ground) is FlatGround:
            setup = {"flat": (ground.level, ground._grip)}
        else:
            bridge = None if ground.bridge is None else tuple(ground.bridge)
            setup = {"profile": (
                list(zip(ground._px.tolist(), ground._py.tolist(), strict=True)),
                ground.kerbs.tolist(), ground.kerb_height, bridge, ground._grip,
            )}
        self._specs = list(specs)
        self._fleet = _core_native._VehicleFleet([asdict(s) for s in self._specs], contact_friction, **setup)

    def add(self, spec: VehicleSpec) -> int:
        """Append a vehicle and return its fleet index."""
        index = self._fleet.add(asdict(spec))
        self._specs.append(spec)
        return index

    def remove(self, index: int) -> None:
        """Remove a vehicle. Subsequent indices move down one."""
        self._fleet.remove(index)
        del self._specs[index]

    def place(self, index: int, position, yaw: float, *, velocity=(0.0, 0.0, 0.0)) -> None:
        """Place wheels near Y and set an optional initial world velocity in metres per second."""
        self._fleet.place(index, tuple(position), yaw, tuple(velocity))

    def update_ground(
        self, tiles: Sequence[GroundTile] = (), *, remove: Sequence[tuple[tuple[int, int], int]] = (),
    ) -> None:
        """Stage one validated tile batch for the next substep. Each removal names the exact version."""
        self._fleet.update_ground([_tile(tile) for tile in tiles], list(remove))

    def rebase(self, delta) -> None:
        """Stage a local origin shift for the next actual substep."""
        self._fleet.rebase(tuple(delta))

    def sample_ground(self, x: float, z: float, near: float) -> GroundSample | None:
        """Inspect the committed tile surface. A point outside loaded terrain returns None."""
        sample = self._fleet.sample(x, z, near)
        if sample is None:
            return None
        height, normal, grip, level, tile, version = sample
        return GroundSample(height, tuple(normal), grip, level, tile, version)

    def snapshot(self) -> list[dict]:
        """Copy the last completed state of each body, including wheels in spec order."""
        return self._fleet.snapshot()

    def checkpoint(self):
        """Save the whole fleet, including the accumulator, pending ground edits and interpolation endpoints."""
        return self._fleet.checkpoint(), tuple(self._specs)

    def restore(self, checkpoint) -> None:
        """Restore a checkpoint issued by this fleet."""
        state, specs = checkpoint
        self._fleet.restore(state)
        self._specs = list(specs)

    def advance(self, controls, dt: float, strengths=None) -> dict:
        """Advance a frame and return counts, contact impulses, solver time and time discarded by the cap.

        Controls are (steer, throttle, brake, handbrake) per body. Optional
        Python driving assists read one state batch and submit one correction
        per frame. The solver applies that batch to every substep in the frame.
        """
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("frame dt must be finite and non-negative")
        rows = np.asarray(controls, dtype=float)
        if not self._specs and rows.size == 0:
            rows = rows.reshape((0, 4))
        if rows.shape != (len(self._specs), 4):
            raise ValueError("expected one four-value controls row per vehicle")
        if not np.isfinite(rows).all() or np.any(rows[:, 0] < -1) or np.any(rows > 1) or np.any(rows[:, 1:] < 0):
            raise ValueError("controls must be finite: steer in [-1, 1], pedals in [0, 1]")
        corrections = None
        if strengths is not None and self._specs:
            rows, corrections = self._assists(rows, dt, strengths)
        return self._fleet.advance(dt, rows.tolist(), corrections)

    def _assists(self, controls, dt, strengths):
        strengths = np.asarray(strengths, dtype=float)
        if not np.all((strengths >= 0.0) & (strengths <= 1.0)):
            raise ValueError("assist strengths must lie in [0, 1]")
        snapshots = self.snapshot()
        packed = pack(self._specs)
        n, w = packed.wheel_mask.shape
        policy = SimpleNamespace(
            p=packed,
            DT=min(dt, self.MAX_FRAME),
            rotation=np.array([s["rotation"] for s in snapshots]),
            velocity=np.array([s["velocity"] for s in snapshots]),
            angular=np.array([s["angular"] for s in snapshots]),
            drive_cut=np.array([s["drive_cut"] for s in snapshots]),
            slip=np.zeros((n, w, 2)),
            omega=np.zeros((n, w)),
            brake_scale=np.ones((n, w)),
        )
        for i, spec in enumerate(self._specs):
            width = len(spec.wheels)
            for key in ("slip", "omega", "brake_scale"):
                getattr(policy, key)[i, :width] = snapshots[i][key]
        corrected = apply_assists(policy, controls, strengths)
        return corrected, [
            (float(policy.drive_cut[i]), policy.brake_scale[i, :len(s.wheels)].tolist(),
             policy.extra_brake[i, :len(s.wheels)].tolist(), policy.lit[i].tolist())
            for i, s in enumerate(self._specs)
        ]

    def pose(self, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate poses from the previous substep, using the fixed-clock residue by default."""
        rows = self._fleet.pose(alpha)
        return (np.asarray([p for p, _ in rows]).reshape((-1, 3)),
                np.asarray([q for _, q in rows]).reshape((-1, 4)))

    def model_transform(self, index: int, alpha: float | None = None):
        """Return the drawn model origin and body rotation for one vehicle."""
        position, rotation = self.pose(alpha)
        return position[index] + matrices(rotation[index:index+1])[0] @ self._specs[index].model_origin, rotation[index]

    def travel(self, index: int) -> np.ndarray:
        """Return the displacement of each wheel relative to its rest compression."""
        spec = self._specs[index]
        state = self.snapshot()[index]
        rest = [spec.mass * 9.81 / len(spec.wheels) / max(w.spring, 1.0) for w in spec.wheels]
        return np.asarray(state["compression"]) - rest

    def telemetry(self, index: int) -> dict:
        """Copy the HUD values of one vehicle, with forward and lateral speeds in car axes."""
        state = self.snapshot()[index]
        axes = matrices(np.asarray([state["rotation"]]))[0]
        speed, _, lateral = axes.T @ state["velocity"]
        return {
            "speed": float(speed), "lateral": float(lateral), "gear": state["gear"], "rpm": state["rpm"],
            "load": state["load"], "slip": state["slip"], "omega": state["omega"],
            "travel": state["compression"], "steer": state["steer_angle"], "impulse": state["impulse"],
            "lit": dict(zip(ASSISTS, state["lit"], strict=True)),
        }
