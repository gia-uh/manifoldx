"""Native vehicle physics sample: one car, fixed 120 Hz, native ground.

Run with an installed native wheel, from any cwd:

    python -I examples/native_vehicles.py
    python -I examples/native_vehicles.py --negative-control

Python prepares one spec and a bounded control script. Rust owns the World,
steps the f64 solver and publishes the pose; the main thread runs the paced
loop while a separate Python producer only submits controls. The table reports
what is measured: fixed substeps and solver milliseconds per substep, never
FPS.
"""

import argparse
import importlib.machinery
import math
import threading
import time

import numpy as np
from manifoldx import _core_native as native
from manifoldx.vehicles import Engine, FlatGround, Gearbox, NativeVehicles, Tyre, VehicleSpec, Wheel

G = 9.81
A, B = 1.15, 1.45
TRACK = 1.5
RADIUS = 0.31


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def car_spec(mu=1.0):
    """The 1 200 kg test car: rear drive, front steer, 85 kW."""
    size = (4.4, 1.45, 1.75)
    corner = 1200.0 * G / 4.0
    spring = corner / G * (2.0 * math.pi * 1.4) ** 2
    damper = 2 * 0.35 * math.sqrt(spring * corner / G)
    peak = 85_000.0 / (5500.0 * 2.0 * math.pi / 60.0)

    def wheel(x, z, front):
        return {
            "mount": (x, -0.15, z),
            "radius": RADIUS,
            "travel": 0.2,
            "spring": spring,
            "damper": damper,
            "tyre": {"mu": mu},
            "driven": not front,
            "steered": front,
            "brake_share": 0.3 if front else 0.2,
            "handbrake": not front,
        }

    return {
        "mass": 1200.0,
        "inertia": (
            1200.0 / 12.0 * (size[1] ** 2 + size[2] ** 2),
            1200.0 / 12.0 * (size[0] ** 2 + size[2] ** 2),
            1200.0 / 12.0 * (size[0] ** 2 + size[1] ** 2),
        ),
        "size": size,
        "wheels": [
            wheel(A, -TRACK / 2, True),
            wheel(A, TRACK / 2, True),
            wheel(-B, -TRACK / 2, False),
            wheel(-B, TRACK / 2, False),
        ],
        "engine": {
            "idle_rpm": 850.0,
            "peak_torque_rpm": 3500.0,
            "peak_power_rpm": 5500.0,
            "idle_torque": 0.6 * peak,
            "peak_torque": 1.3 * peak,
            "power_torque": peak,
        },
        "gearbox": {"ratios": (3.6, 2.1, 1.4, 1.0, 0.8), "final": 3.9},
        "arb": (9000.0, 4000.0),
        "cd": 0.35,
        "area": 2.0,
        "rolling": 0.015,
        "max_steer": 0.6,
        "brake_torque": 5000.0,
    }


def public_spec(**overrides):
    """Construct the same reference car with the public authoring types."""
    config = car_spec()
    config["wheels"] = tuple(Wheel(**(w | {"tyre": Tyre(**w["tyre"])})) for w in config["wheels"])
    config["engine"] = Engine(**config["engine"])
    config["gearbox"] = Gearbox(**config["gearbox"])
    return VehicleSpec(**(config | overrides))


def public_fleet():
    fleet = NativeVehicles([public_spec()], FlatGround())
    fleet.place(0, (0.0, 0.0, 0.0), 0.0)
    stats = [fleet.advance([[0.0, 1.0, 0.0, 0.0]], 1 / 60) for _ in range(180)]
    state = fleet.snapshot()[0]
    require(sum(s["substeps"] for s in stats) == 360, "public fleet: wrong substep count")
    require(all(s["bodies"] == 1 for s in stats), "public fleet: wrong body count")
    require(len(state["wheel_centres"]) == 4, "public fleet: wrong wheel count")
    require(fleet.telemetry(0)["speed"] > 10.0, "public fleet: did not accelerate")
    print("public fleet: 360 fixed substeps, 1 body, 4 wheels")


ROAD = {
    "points": [(-50.0, 0.0), (20.0, 0.0), (40.0, 1.5), (60.0, 0.0), (400.0, 0.0)],
    "kerbs": [30.0, 70.0],
    "kerb_height": 0.15,
    "bridge": (90.0, 110.0, 3.0),
}


def spawn(world):
    world.register_component("position", "f32", 3)
    world.register_component("rotation", "f32", 4)
    world.register_component("velocity", "f32", 3)
    world.register_component("angular", "f32", 3)
    result = world.spawn(
        1,
        {
            "position": np.zeros((1, 3), np.float32),
            "rotation": np.array([[0.0, 0.0, 0.0, 1.0]], np.float32),
            "velocity": np.zeros((1, 3), np.float32),
            "angular": np.zeros((1, 3), np.float32),
        },
    )
    return np.frombuffer(result["handles"], dtype="<u8").reshape(-1, 2).copy()


def column(world, name, ids):
    result = world.read_component(name, ids)
    dtype = np.float32 if result["kind"] == "f32" else np.uint32
    return np.frombuffer(result["data"], dtype=dtype).reshape(-1, result["width"]).copy()


def wait_for_frame(runtime, target, deadline):
    """Wait for a completed frame while yielding to the native runner."""
    while time.monotonic() < deadline:
        time.sleep(0.005)
        phase = runtime.phase
        frames = runtime.stats()["frames"]
        if frames >= target:
            return frames
        if phase == "stopped":
            break
    raise AssertionError(f"native frames did not reach {target} while the control producer waited")


def controls_producer(runtime, events, admissions, failures):
    """Submit bounded controls at frame boundaries while the main thread runs."""
    deadline = time.monotonic() + 10.0
    try:
        for frame, controls in events:
            before = wait_for_frame(runtime, frame, deadline)
            result = runtime.set_controls(*controls)
            require(result["status"] == "admitted", f"control at frame {frame} was {result}")
            after = runtime.stats()["frames"]
            # The boundary at `after` may already have drained the queue.
            admissions.append((result["sequence"], before, after + 1))
            wait_for_frame(runtime, after + 1, min(deadline, time.monotonic() + 1.0))
    except Exception as error:
        failures.append(error)
        runtime.stop()


def run_case(name, spec, ground, place, frames, events=(), check=None, paced=False):
    """One fresh World and runtime; returns the owned pose, stats and check value."""
    sim = native._VehicleSim(spec)
    if ground[0] == "flat":
        sim.set_flat_ground(*ground[1:])
    else:
        sim.set_profile_ground(**ground[1])
    sim.place(*place)
    world = native._NativeWorld(1)
    ids = spawn(world)
    runtime = native.NativeVehicleRuntime(world, ids, sim)
    events = list(events)
    phase_ends = [frame for frame, _ in events[1:]] + [frames]
    expected_controls = len(events)
    admissions = []
    if events and events[0][0] == 0:
        admitted = runtime.set_controls(*events[0][1])
        require(admitted["status"] == "admitted", f"{name}: first controls {admitted}")
        admissions.append((admitted["sequence"], 0, 0))
        events = events[1:]
    failures = []
    producer = None
    if events:
        producer = threading.Thread(
            target=controls_producer,
            args=(runtime, events, admissions, failures),
            name="vehicle-controls",
            daemon=True,
        )
        producer.start()
    stats = runtime.run(frames, pace_hz=120 if paced else None)
    if producer is not None:
        producer.join(timeout=6)
        require(not producer.is_alive(), f"{name}: control producer leaked")
    if failures:
        raise AssertionError(f"{name}: {failures[0]}") from failures[0]
    state = runtime.snapshot()
    outcomes = []
    while (outcome := runtime.take_outcome()) is not None:
        outcomes.append(outcome)
    require(len(admissions) == len(outcomes) == expected_controls, f"{name}: missing or extra controls")
    for index, (outcome, (sequence, earliest, latest), phase_end) in enumerate(
        zip(outcomes, admissions, phase_ends), start=1
    ):
        require(outcome["sequence"] == sequence == index, f"{name}: out-of-order outcome {outcome}")
        require(outcome["status"] == "applied", f"{name}: unapplied control {outcome}")
        latest = min(latest, phase_end - 1)
        require(earliest <= outcome["frame"] <= latest, f"{name}: expected frames {earliest} to {latest}, got {outcome}")
    runtime.close()
    pose = {
        "position": column(world, "position", ids)[0],
        "rotation": column(world, "rotation", ids)[0],
        "velocity": column(world, "velocity", ids)[0],
        "angular": column(world, "angular", ids)[0],
    }
    norm = float(np.dot(pose["rotation"], pose["rotation"]))
    require(abs(norm - 1.0) < 1e-4, f"{name}: quaternion norm {norm}")
    require(np.isfinite(pose["position"]).all(), f"{name}: non-finite position")
    require(np.isfinite(pose["velocity"]).all(), f"{name}: non-finite velocity")
    require(stats["vehicle_steps"] == frames, f"{name}: {stats['vehicle_steps']} of {frames} steps")
    require(stats["publish_ns"] > 0, f"{name}: no pose publication measured")
    require(len(state["load"]) == 4, f"{name}: wheel count")
    require(all(value >= 0.0 for value in state["load"]), f"{name}: negative wheel load")
    if check is not None:
        check(name, pose, stats, state)
    return pose, stats, state


def forward_speed(pose):
    x, y, z, w = pose["rotation"]
    forward = (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w))
    return float(np.dot(pose["velocity"], forward))


def cases():
    rows = []

    def settle(name, pose, _stats, state):
        speed = float(np.hypot(pose["velocity"][0], pose["velocity"][2]))
        require(speed < 0.05, f"{name}: crept at {speed:.3f} m/s")
        require(min(state["load"]) > 0.0, f"{name}: a wheel carries no load")
        require(state["gear"] == 1, f"{name}: gear {state['gear']}")

    rows.append(
        (
            "settle",
            run_case("settle", car_spec(), ("flat", 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), 240, check=settle),
        )
    )

    def accelerate(name, pose, _stats, state):
        speed = forward_speed(pose)
        require(speed > 10.0, f"{name}: reached {speed:.3f} m/s")
        require(pose["position"][0] > 10.0, f"{name}: travelled {pose['position'][0]:.3f} m")
        require(state["gear"] >= 2, f"{name}: gear {state['gear']}")
        require(state["rpm"] > 800.0, f"{name}: rpm {state['rpm']:.0f}")
        require(max(state["load"]) > 1000.0, f"{name}: no loaded wheel")

    rows.append(
        (
            "accelerate",
            run_case(
                "accelerate",
                car_spec(),
                ("flat", 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0),
                360,
                events=[(0, (0.0, 1.0, 0.0, 0.0))],
                check=accelerate,
            ),
        )
    )

    def brake(name, pose, _stats, state):
        speed = abs(forward_speed(pose))
        require(speed < 1.0, f"{name}: still moving at {speed:.3f} m/s")
        require(pose["position"][0] > 5.0, f"{name}: never moved")

    rows.append(
        (
            "brake",
            run_case(
                "brake",
                car_spec(),
                ("flat", 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0),
                552,
                events=[(0, (0.0, 1.0, 0.0, 0.0)), (240, (0.0, 0.0, 1.0, 0.0)), (336, (0.0, 0.0, 0.0, 1.0))],
                check=brake,
                paced=True,
            ),
        )
    )

    def steer(name, pose, _stats, state):
        require(abs(pose["position"][2]) > 0.5, f"{name}: lateral {pose['position'][2]:.3f} m")
        require(abs(pose["angular"][1]) > 0.02, f"{name}: yaw {pose['angular'][1]:.3f} rad/s")
        require(abs(state["steer_angle"][0]) > 0.05, f"{name}: steer {state['steer_angle'][0]:.3f} rad")

    rows.append(
        (
            "steer",
            run_case(
                "steer",
                car_spec(),
                ("flat", 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0),
                360,
                events=[(0, (0.0, 1.0, 0.0, 0.0)), (180, (0.3, 1.0, 0.0, 0.0))],
                check=steer,
                paced=True,
            ),
        )
    )

    def coast(name, pose, _stats, _state):
        require(abs(pose["position"][2]) < 0.1, f"{name}: drifted {pose['position'][2]:.3f} m")
        require(forward_speed(pose) > 3.0, f"{name}: coasting speed lost")

    rows.append(
        (
            "coast",
            run_case(
                "coast",
                car_spec(),
                ("flat", 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0),
                420,
                events=[(0, (0.0, 1.0, 0.0, 0.0)), (180, (0.0, 0.0, 0.0, 0.0))],
                check=coast,
                paced=True,
            ),
        )
    )

    def slope(name, pose, _stats, _state):
        require(pose["velocity"][0] > 0.5, f"{name}: rolled at {pose['velocity'][0]:.3f} m/s")

    rows.append(
        (
            "slope",
            run_case(
                "slope",
                car_spec(),
                ("profile", {"points": [(-200.0, 10.0), (200.0, -10.0)]}),
                (0.0, 0.0, 0.0, 0.0),
                360,
                check=slope,
            ),
        )
    )

    def road(name, pose, _stats, state):
        require(pose["position"][0] > 50.0, f"{name}: reached {pose['position'][0]:.3f} m")
        require(state["gear"] >= 2, f"{name}: gear {state['gear']}")

    rows.append(
        (
            "profile/kerb/bridge-under",
            run_case(
                "profile/kerb/bridge-under",
                car_spec(),
                ("profile", ROAD),
                (0.0, 0.0, 0.0, 0.0),
                960,
                events=[(0, (0.0, 1.0, 0.0, 0.0))],
                check=road,
            ),
        )
    )

    def above(name, pose, _stats, state):
        require(pose["position"][1] > 5.0, f"{name}: settled at y {pose['position'][1]:.3f}")
        require(abs(state["ground_y"][0] - 6.0) < 1e-6, f"{name}: ground {state['ground_y'][0]:.3f}")

    rows.append(
        (
            "bridge-above",
            run_case(
                "bridge-above",
                car_spec(),
                ("profile", {"points": [(-100.0, 0.0), (100.0, 0.0)], "bridge": (5.0, 15.0, 6.0)}),
                (10.0, 6.0, 0.0, 0.0),
                120,
                check=above,
            ),
        )
    )

    def below(name, pose, _stats, state):
        require(pose["position"][1] < 1.0, f"{name}: settled at y {pose['position'][1]:.3f}")
        require(abs(state["ground_y"][0]) < 1e-6, f"{name}: ground {state['ground_y'][0]:.3f}")

    rows.append(
        (
            "bridge-below",
            run_case(
                "bridge-below",
                car_spec(),
                ("profile", {"points": [(-100.0, 0.0), (100.0, 0.0)], "bridge": (5.0, 15.0, 6.0)}),
                (10.0, 0.0, 0.0, 0.0),
                120,
                check=below,
            ),
        )
    )
    return rows


def check_wheel_inertia():
    spec = car_spec()
    spec["wheels"][2]["inertia"] = 0.0
    try:
        native._VehicleSim(spec)
    except ValueError as error:
        require(str(error) == "wheel 2 inertia must be finite and positive", str(error))
    else:
        raise AssertionError("native solver accepted zero wheel inertia")

    sim = native._VehicleSim(car_spec())
    sim.place(0.0, 0.0, 0.0, 0.0)
    sim.set_flat_ground(-10.0)
    sim.set_controls(0.0, 1.0, 0.0, 0.0)
    require(sim.step(1)["steps"] == 1, "expected one fixed substep")
    state = sim.snapshot()
    require(state["load"] == [0.0] * 4, "expected airborne wheels")
    for value in state["omega"] + state["wheel_turn"]:
        require(math.isfinite(value), "non-finite airborne wheel state")
    require(state["omega"][2] > 0.0 and state["wheel_turn"][2] > 0.0, "driven wheel did not spin")
    print("wheel inertia: zero rejected; airborne wheel spin finite after one fixed substep")


def negative_control():
    """The accelerate assertion must fail when tyre grip is removed."""
    pose, stats, _state = run_case(
        "negative zero grip",
        car_spec(mu=0.0),
        ("flat", 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
        360,
        events=[(0, (0.0, 1.0, 0.0, 0.0))],
    )
    speed = forward_speed(pose)
    if speed > 10.0:
        raise AssertionError(
            f"negative control failed: zero grip still met the acceleration threshold at {speed:.3f} m/s"
        )
    raise AssertionError(
        f"negative control: zero grip did not accelerate (speed {speed:.3f} m/s), "
        f"the force the assertion depends on was removed after {stats['vehicle_steps']} substeps"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--negative-control",
        action="store_true",
        help="remove tyre grip and require the acceleration assertion to fail",
    )
    args = parser.parse_args()
    require(
        any(native.__file__.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES),
        "compiled extension required",
    )
    if args.negative_control:
        negative_control()
        return
    check_wheel_inertia()
    public_fleet()
    rows = cases()
    print("| case | substeps | solver ms/substep | publish ms/frame | gear | rpm | pose (x, y, z) |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for name, (pose, stats, state) in rows:
        ms = stats["vehicle_ns"] / 1e6 / max(stats["vehicle_steps"], 1)
        publish = stats["publish_ns"] / 1e6 / max(stats["frames"], 1)
        print(
            f"| {name} | {stats['vehicle_steps']} | {ms:.4f} | {publish:.4f} | "
            f"{state['gear']} | {state['rpm']:.0f} | "
            f"({pose['position'][0]:.2f}, {pose['position'][1]:.2f}, {pose['position'][2]:.2f}) |"
        )
    total = sum(stats["vehicle_steps"] for _, (_, stats, _) in rows)
    solver = sum(stats["vehicle_ns"] for _, (_, stats, _) in rows)
    publish = sum(stats["publish_ns"] for _, (_, stats, _) in rows)
    paced = sum(stats["paced_ns"] for _, (_, stats, _) in rows)
    admitted = sum(stats["admitted"] for _, (_, stats, _) in rows)
    applied = sum(stats["applied"] for _, (_, stats, _) in rows)
    print(
        f"\nmeasured: one car, f64 solver, native flat/profile/kerb/bridge ground; "
        f"{total} fixed substeps; solver {solver / 1e6 / total:.4f} ms/substep; "
        f"pose publication {publish / 1e6 / total:.4f} ms/substep; "
        f"pacing {paced / 1e6:.1f} ms; controls {admitted} admitted / {applied} applied; not FPS."
    )


if __name__ == "__main__":
    main()
