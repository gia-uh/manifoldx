# Native vehicle physics

One upright four-wheel car, stepped in `f64` at a fixed `1/120 s`. The model is
the scalar Python model in `src/manifoldx/vehicles/single.py`: ray suspension
with spring, damper, bump stop and anti-roll bar; Pacejka longitudinal and
lateral slip with relaxation and combined-slip capping; a three-point engine
curve, automatic gearbox and reverse; an open differential, brakes, handbrake
and Ackermann steering; gravity, drag, quaternion body integration and body-box
ground contact.

The solver owns no clock, thread or queue. `_VehicleSim.step(substeps)` is the
CPU seam for tests and measurement. The production path attaches the solver to
the existing `Runtime`: `NativeVehicleRuntime.run(frames, pace_hz=...)` executes
one fixed substep per frame on the calling thread with Python detached, drains
bounded control commands from the existing FIFO queue and publishes the pose
into the existing `World`.

## Ground

Two native grounds, no callback crosses into Python inside a substep:

- `flat(level, grip)`;
- `profile(points, kerbs, kerb_height, bridge, grip)`: piecewise-linear height,
  square kerbs, and a bridge deck selected by the level nearest `near`.
  `place()` settles the car on the level nearest the requested height, so the
  same profile can put it above or below a bridge.

## Ownership

- The solver's `f64` state is authoritative. The World receives `f32`
  position, rotation, velocity and angular columns every frame.
- The runtime owns the vehicle entity and validates its generation-bearing
  handle before taking ownership. A destroyed or stale target rejects the
  publication; it cannot write into a reused slot.
- `_VehicleSim` is a setup source. `NativeVehicleRuntime` clones it once while
  the runtime is `Ready` and owns the active simulation; the source stays
  independent.
- Controls are ordinary bounded commands: same admission limits, credits,
  sequences and outcomes as `Update`, `Read` and `Destroy`. Non-finite or
  out-of-range controls are rejected before mutation.
- The runtime rejects a `dt` other than the solver's fixed step; a frame is one
  substep, not an arbitrary time slice.
- No Python buffer is acquired or released under a World lock, and no
  allocation or mutation happens before validation.

## Deliberately left for the next slice

Wall and segment contacts, vehicle-to-vehicle contacts, assists (traction,
ABS, stability), multi-vehicle fleets, full Contravia traffic and the public
Python vehicle API. `bootstrap` still reports the runtime as not implemented;
`MANIFOLDX_CORE=rust` is unchanged.

## Run it

With an installed wheel, from any directory:

```bash
python -I examples/native_vehicles.py
python -I examples/native_vehicles.py --negative-control
```

The first command runs named cases (settle, accelerate, brake, steer, coast,
slope, profile/kerb/bridge) and prints fixed substeps, solver ms/substep and
pose-publication ms/frame for one car. The negative control removes tyre grip
and requires the acceleration assertion to fail.

A measured run of the installed wheel (Linux x86_64, free-threaded CPython
3.14t) printed 3,492 fixed substeps, 0.0067 ms/substep for the solver and
0.0014 ms/substep for pose publication, with pacing reported separately. These
are measurements of this solver on that host, not an FPS claim.

`make native-check` covers formatting, clippy and the Rust tests, including the
physical acceptances, fixed-substep replay and the zero-grip negative control.
`scripts/gates/native-verify.sh` runs the installed-wheel sample, its negative
control and the missing-extension control.
