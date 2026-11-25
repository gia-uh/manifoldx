"""Check E3 through the installed public fleet API, with mutation controls."""
import argparse
import json
import math
import runpy
import time
from pathlib import Path

from manifoldx.vehicles import (
    FlatGround, GroundTile, GroundTriangle, Heightfield, NativeVehicles, Obstacle, TiledGround,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = runpy.run_path(str(ROOT / "examples" / "native_vehicles.py"))["public_spec"]
ZERO = [0.0, 0.0, 0.0, 0.0]
DT = 1 / 120


def rectangle(x0, z0, x1, z1, y, grip, level):
    a, b, c, d = (x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)
    return [GroundTriangle((a, b, c), grip, level), GroundTriangle((a, c, d), grip, level)]


def terrain(levels, walls=()):
    size = 16
    tiles = []
    for i in (-1, 0):
        for j in (-1, 0):
            triangles = [
                triangle
                for y, grip, level in levels
                for triangle in rectangle(i * size, j * size, (i + 1) * size, (j + 1) * size, y, grip, level)
            ]
            segments = [wall for wall in walls if j * size <= wall.ends[0][1] < (j + 1) * size and i == 0]
            tiles.append(GroundTile((i, j), 1, triangles=triangles, obstacles=segments))
    return TiledGround(size, tiles)


def close(actual, expected, absolute=1e-8, relative=0):
    return abs(actual - expected) <= absolute + relative * abs(expected)


def assert_translated_pose(actual, reference, shift, label):
    # Use the frozen f64 position and sign-aligned quaternion tolerances.
    actual_position = [x + dx for x, dx in zip(actual["position"], shift, strict=True)]
    assert all(close(x, y, 1e-8, 1e-9) for x, y in
               zip(actual_position, reference["position"], strict=True)), f"{label}: translated position differs"
    a, b = actual["rotation"], reference["rotation"]
    sign = -1 if sum(x * y for x, y in zip(a, b, strict=True)) < 0 else 1
    assert all(close(sign * x, y, 1e-9, 1e-9) for x, y in
               zip(a, b, strict=True)), f"{label}: orientation differs"


def check_levels(force_high=False):
    levels = [(-5, 0.5, 11), (0, 1, 22), (6, 0.25, 33)]
    fleet = NativeVehicles([SPEC()], terrain(levels))
    for (x, z), (y, grip, level) in zip(((1, 2), (2, 1), (2, 2)), levels, strict=True):
        sample = fleet.sample_ground(x, z, 6.5 if force_high else y + 0.5)
        assert (sample.level, sample.height, sample.grip) == (level, y, grip), "wrong native ground level"
        base = NativeVehicles([SPEC()], FlatGround(grip=grip))
        base.place(0, (0, 0.5, 0), 0)
        fleet.place(0, (0, y + 0.5, 0), 0)
        for _ in range(120):
            assert base.advance([ZERO], DT)["substeps"] == fleet.advance([ZERO], DT)["substeps"] == 1
        actual, reference = fleet.snapshot()[0], base.snapshot()[0]
        label = f"level {level}"
        assert_translated_pose(actual, reference, (0, -y, 0), label)
        assert actual["ground_y"] == [y] * 4
        assert [h - y for h in actual["ground_y"]] == reference["ground_y"], f"{label}: wheel floor differs"
        for field, absolute, relative in (("compression", 1e-8, 1e-9), ("load", 1e-3, 1e-8)):
            assert all(close(a, b, absolute, relative) for a, b in
                       zip(actual[field], reference[field], strict=True)), f"{label}: suspension {field} differs"
        print(f"{label}: 120 ticks, {grip=}, pose and suspension match the translated plane")


def check_grids_and_tiles():
    # Split the frozen analytic plane across four cells. Evaluate it
    # at each corner independently before handing over the grid.
    tiles = []
    for i in (-1, 0):
        for j in (-1, 0):
            x, z = 4 * i, 4 * j
            heights = [[0.25 * (u + 4) + 0.5 * (v + 4) for u in (x, x + 4)] for v in (z, z + 4)]
            tiles.append(GroundTile((i, j), 1, heightfields=[Heightfield((x, z), (4, 4), heights, 0.75, 44)]))
    fleet = NativeVehicles([], TiledGround(4, tiles))
    sample = fleet.sample_ground(1, -2, 2.25)
    normal = (-0.2182178902359924, 0.8728715609439696, -0.4364357804719848)
    assert sample.level == 44 and sample.grip == 0.75 and close(sample.height, 2.25)
    assert all(close(a, b, 1e-9) for a, b in zip(sample.normal, normal, strict=True))

    left = GroundTile((-1, 0), 1, triangles=rectangle(-16, 0, 0, 16, 0, 1, 51))
    right = GroundTile((0, 0), 2, triangles=rectangle(0, 0, 8, 16, 0, 1, 51)
                       + rectangle(8, 0, 16, 16, 0.125, 1, 52))
    ground = TiledGround(16, [left, right])
    road = NativeVehicles([SPEC()], ground)
    control = NativeVehicles([SPEC()], ground)
    for x, height in [(-0.125, 0), (0, 0), (0.125, 0), (7.875, 0), (8.125, 0.125)]:
        assert road.sample_ground(x, 1, 0).height == height
    road.place(0, (-4, 0, 1), 0)
    control.place(0, (-4, 0, 1), 0)
    positions = []
    for _ in range(360):
        assert road.advance([[0, 1, 0, 0]], DT)["substeps"] == control.advance([[0, 1, 0, 0]], DT)["substeps"] == 1
        positions.append(road.snapshot()[0])
    assert positions[-1]["position"][0] > 8.125
    assert any(0.125 in row["ground_y"] for row in positions), "wheels did not meet the kerb"

    for fleet in (road, control):
        fleet.update_ground(remove=[((0, 0), 2)])
    assert road.advance([ZERO], DT / 2)["substeps"] == control.advance([ZERO], DT / 2)["substeps"] == 0
    assert road.sample_ground(0.125, 1, 0).version == 2
    assert road.advance([ZERO], DT / 2)["substeps"] == control.advance([ZERO], DT / 2)["substeps"] == 1
    assert road.sample_ground(0.125, 1, 0) is None
    try:
        road.update_ground([right])
    except ValueError as error:
        assert "new positive version" in str(error)
    else:
        raise AssertionError("an unloaded tile version was reused")
    control_state = control.snapshot()
    assert road.snapshot() == control_state
    shift = (32, 2, 0)
    road.rebase(shift)
    assert road.snapshot() == control_state, "staged rebase moved the body before the substep"
    assert road.sample_ground(-0.125, 1, 0).version == 1
    assert road.advance([ZERO], DT)["substeps"] == control.advance([ZERO], DT)["substeps"] == 1
    assert_translated_pose(road.snapshot()[0], control.snapshot()[0], shift, "rebase publication tick 362")
    shifted = road.sample_ground(-32.125, 1, -2)
    assert (shifted.height, shifted.tile, shifted.version) == (-2, (-1, 0), 1)
    print("heightfield, kerb and cell edge: native normals, support, retire and relative pose at rebase tick 362")


def check_grip_and_callbacks():
    class PoisonGround:
        calls = 0

        def height(self, *_):
            self.calls += 1
            raise AssertionError("Python height was called")

        grip = height

    poison = PoisonGround()
    try:
        NativeVehicles([SPEC()], poison)
    except TypeError as error:
        assert "TiledGround" in str(error)
    else:
        raise AssertionError("Python ground was accepted")
    assert poison.calls == 0
    for grip in (0, 1):
        fleet = NativeVehicles([SPEC()], terrain([(0, grip, 11)]))
        fleet.place(0, (0, 0, 0), 0)
        for _ in range(360):
            fleet.advance([[0, 1, 0, 0]], DT)
        state = fleet.snapshot()[0]
        assert fleet.telemetry(0)["speed"] > 10 if grip else math.hypot(*state["velocity"][::2]) < 0.05
    for _ in range(240):
        fleet.advance([ZERO], DT)
    assert poison.calls == 0
    print("grip: uses the selected surface; unsupported Python callbacks rejected before use")


def check_impact(pair, control=False):
    # Put the whole segment in one tile, then rebase it to local z=-8..8.
    z = 8
    wall = Obstacle(((0, 0), (0, 16)), (-2, 2))
    grounds = ((0, 0, 11),)
    ground = terrain(grounds, [] if pair or control else [wall])
    free_ground = terrain(grounds)
    spec = SPEC(cd=0, rolling=0)
    count = 2 if pair else 1
    fleet = NativeVehicles([spec] * count, ground, contact_friction=0)
    free = NativeVehicles([spec] * count, free_ground, contact_friction=0)
    for sim in (fleet, free):
        sim.place(0, (-2.24, 0, z), 0, velocity=(8, 0, 0))
    if pair:
        fleet.place(1, (2.2, 0, z + (4 if control else 0)), 0)
        free.place(1, (2.2, 0, z + 4), 0)
    for sim in (fleet, free):
        sim.rebase((0, 0, z))
    reference = free.advance([ZERO] * count, DT)
    report = fleet.advance([ZERO] * count, DT)
    assert reference["contacts"] == [] and reference["impulse"] == 0
    assert report["bodies"] == count and report["substeps"] == 1
    assert len(report["contacts"]) == 1, "missing native pair impulse" if pair else "missing native wall impulse"
    a, b, normal, momentum = report["contacts"][0]
    assert (a, b) == (0, 1 if pair else None)
    assert all(close(x, y, 1e-9) for x, y in zip(normal, (-1, 0, 0), strict=True))
    initial = [s["velocity"][0] for s in free.snapshot()]
    final = [s["velocity"][0] for s in fleet.snapshot()]
    relative = initial[0] - initial[1] if pair else initial[0]
    reduced_mass = 600 if pair else 1200
    expected = reduced_mass * 1.2 * relative
    nominal = 5760 if pair else 11520
    assert close(expected, nominal, 1e-3, 1e-6)
    assert close(report["impulse"], expected, 1e-3, 1e-6)
    assert close(momentum[0], -expected, 1e-3, 1e-6)
    assert close(momentum[1], 0, 1e-9) and close(momentum[2], 0, 1e-9)
    assert close(1200 * (final[0] - initial[0]), -expected, 1e-3, 1e-6)
    if pair:
        assert close(1200 * sum(final), 1200 * sum(initial), 1e-3, 1e-6)
        assert close(1200 * (final[1] - initial[1]), expected, 1e-3, 1e-6)
    lost = 600 * (sum(v * v for v in initial) - sum(v * v for v in final))
    assert lost > 0 and close(lost, reduced_mass * (1 - 0.2 ** 2) * relative ** 2 / 2, 0.05, 1e-5)
    print(f"{'pair' if pair else 'wall'}: {expected=:.3f}, {nominal=}, {lost=:.3f}, {normal=}, {momentum=}")
    return {"free": reference, "contact": report, "initial_vx": initial, "final_vx": final, "lost_j": lost}


def check_obstacle_parameters():
    spec = SPEC(cd=0, rolling=0)
    for y, strength, expected in [((-2, 2), 0.25, 2880), ((6, 8), 1, 0), ((-2, 2), 0, 0)]:
        wall = Obstacle(((0, 0), (0, 16)), y, strength)
        fleet = NativeVehicles([spec], terrain([(0, 0, 11)], [wall]), contact_friction=0)
        fleet.place(0, (-2.24, 0, 8), 0, velocity=(8, 0, 0))
        report = fleet.advance([ZERO], DT)
        assert len(report["contacts"]) == int(expected > 0)
        assert close(report["impulse"], expected, 1e-3, 1e-6)
    print("obstacle strength and vertical extent: quarter impulse, overhead and zero impulse")


def check_load_and_replay():
    samples = []
    for size in (1, 10, 100):
        for repeat in range(3):
            fleet = NativeVehicles([SPEC()] * size, FlatGround())
            for i in range(size):
                fleet.place(i, (10 * i, 0, 0), 0)
            sequence = [(DT / 2, 0), (DT / 2, 1), (2 * DT, 2), (DT, 1)] * 60
            frames = []
            for dt, count in sequence:
                start = time.perf_counter_ns()
                report = fleet.advance([ZERO] * size, dt)
                wall_ns = time.perf_counter_ns() - start
                assert (report["substeps"], report["bodies"]) == (count, size)
                assert not report["contacts"] and report["chassis_contacts"] == 0 and report["impulse"] == 0
                frames.append(report | {"frame_ns": wall_ns})
            samples.append({"bodies": size, "repeat": repeat, "frames": frames})
            print(f"{size} bodies, {len(frames)} frames, {sum(s['substeps'] for s in frames)} fixed substeps")
    fleet = NativeVehicles([SPEC()], FlatGround())
    fleet.place(0, (0, 0, 0), 0)
    drive = [[0.3, 1, 0, 0]]
    for _ in range(30):
        fleet.advance(drive, DT)
    assert fleet.advance(drive, DT / 2)["substeps"] == 0
    checkpoint = fleet.checkpoint()

    def replay():
        assert fleet.advance(drive, DT / 2)["substeps"] == 1
        assert fleet.advance(drive, 2 * DT)["substeps"] == 2
        p0, _ = fleet.pose(0)
        p1, _ = fleet.pose(1)
        p, q = fleet.pose(0.5)
        assert all(close(v, (a + b) / 2) for v, a, b in zip(p[0], p0[0], p1[0], strict=True))
        assert close(math.hypot(*q[0]), 1, 1e-9)
        return fleet.snapshot(), p.tolist(), q.tolist()

    first = replay()
    fleet.restore(checkpoint)
    assert replay() == first
    capped = fleet.advance(drive, 0.2)
    assert capped["substeps"] == 6 and close(capped["dropped_seconds"], 0.15)
    print("checkpoint replay is exact; interpolation, residue and frame cap verified")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=["level", "wall", "pair"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.control == "level":
        check_levels(force_high=True)
    elif args.control:
        check_impact(pair=args.control == "pair", control=True)
    else:
        check_levels()
        check_grids_and_tiles()
        check_grip_and_callbacks()
        wall, pair = check_impact(pair=False), check_impact(pair=True)
        check_obstacle_parameters()
        loads = check_load_and_replay()
        if args.output:
            args.output.write_text(json.dumps({"wall": wall, "pair": pair, "loads": loads}, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
