//! Python bindings for native vehicle physics.
//!
//! `_VehicleSim` builds and configures one f64 solver and can be handed to
//! `NativeVehicleRuntime`, which reuses the existing `Runtime` scheduler: each
//! run frame is one fixed 1/120 s substep, controls arrive as bounded FIFO
//! commands and the pose is written into the existing `World`. No Python runs
//! inside a substep.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use manifoldx_core::ecs::{EntityId, World};
use manifoldx_core::runtime::{Admission, Command, Control, Phase, Runtime, VehicleColumns};
use manifoldx_core::vehicles::{
    Controls, Engine, Gearbox, Ground, Sim, Tyre, VehicleSpec, Wheel, DT,
};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::runtime::stats_dict;
use super::{decode_handles_limited, PyNativeWorld};

fn field<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("vehicle spec is missing '{key}'")))
}

fn number(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    field(dict, key)?
        .extract::<f64>()
        .map_err(|_| PyTypeError::new_err(format!("vehicle spec '{key}' must be a float")))
}

fn optional_number(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    match dict.get_item(key)? {
        Some(value) => value
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err(format!("vehicle spec '{key}' must be a float"))),
        None => Ok(default),
    }
}

fn optional_flag(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match dict.get_item(key)? {
        Some(value) => value
            .extract::<bool>()
            .map_err(|_| PyTypeError::new_err(format!("vehicle spec '{key}' must be a bool"))),
        None => Ok(default),
    }
}

fn vector3(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<[f64; 3]> {
    let values = field(dict, key)?.extract::<Vec<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "vehicle spec '{key}' must be a list of three floats"
        ))
    })?;
    if values.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "vehicle spec '{key}' needs three values"
        )));
    }
    Ok([values[0], values[1], values[2]])
}

fn optional_vector3(dict: &Bound<'_, PyDict>, key: &str, default: [f64; 3]) -> PyResult<[f64; 3]> {
    match dict.get_item(key)? {
        Some(value) => {
            let values = value.extract::<Vec<f64>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "vehicle spec '{key}' must be a list of three floats"
                ))
            })?;
            if values.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "vehicle spec '{key}' needs three values"
                )));
            }
            Ok([values[0], values[1], values[2]])
        }
        None => Ok(default),
    }
}

fn tyre(value: Option<Bound<'_, PyAny>>) -> PyResult<Tyre> {
    let Some(value) = value else {
        return Ok(Tyre::default());
    };
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("wheel 'tyre' must be a dict"))?;
    let default = Tyre::default();
    Ok(Tyre {
        mu: optional_number(dict, "mu", default.mu)?,
        b_lat: optional_number(dict, "b_lat", default.b_lat)?,
        c_lat: optional_number(dict, "c_lat", default.c_lat)?,
        e_lat: optional_number(dict, "e_lat", default.e_lat)?,
        b_long: optional_number(dict, "b_long", default.b_long)?,
        c_long: optional_number(dict, "c_long", default.c_long)?,
        e_long: optional_number(dict, "e_long", default.e_long)?,
    })
}

fn wheel(value: &Bound<'_, PyAny>) -> PyResult<Wheel> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("each wheel must be a dict"))?;
    Ok(Wheel {
        mount: vector3(dict, "mount")?,
        radius: number(dict, "radius")?,
        travel: number(dict, "travel")?,
        spring: number(dict, "spring")?,
        damper: number(dict, "damper")?,
        tyre: tyre(dict.get_item("tyre")?)?,
        driven: optional_flag(dict, "driven", false)?,
        steered: optional_flag(dict, "steered", false)?,
        brake_share: optional_number(dict, "brake_share", 0.25)?,
        handbrake: optional_flag(dict, "handbrake", false)?,
        inertia: optional_number(dict, "inertia", 1.2)?,
    })
}

fn engine(value: &Bound<'_, PyAny>) -> PyResult<Engine> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("vehicle spec 'engine' must be a dict"))?;
    Ok(Engine {
        idle_rpm: number(dict, "idle_rpm")?,
        peak_torque_rpm: number(dict, "peak_torque_rpm")?,
        peak_power_rpm: number(dict, "peak_power_rpm")?,
        idle_torque: number(dict, "idle_torque")?,
        peak_torque: number(dict, "peak_torque")?,
        power_torque: number(dict, "power_torque")?,
        engine_brake: optional_number(dict, "engine_brake", 0.02)?,
    })
}

fn gearbox(value: &Bound<'_, PyAny>) -> PyResult<Gearbox> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("vehicle spec 'gearbox' must be a dict"))?;
    Ok(Gearbox {
        ratios: field(dict, "ratios")?.extract::<Vec<f64>>().map_err(|_| {
            PyTypeError::new_err("vehicle spec 'gearbox.ratios' must be a list of floats")
        })?,
        final_drive: number(dict, "final")?,
        reverse: optional_number(dict, "reverse", 3.5)?,
        shift_time: optional_number(dict, "shift_time", 0.3)?,
        efficiency: optional_number(dict, "efficiency", 0.85)?,
    })
}

fn parse_spec(value: &Bound<'_, PyDict>) -> PyResult<VehicleSpec> {
    let wheels = field(value, "wheels")?
        .extract::<Vec<Bound<'_, PyAny>>>()
        .map_err(|_| PyTypeError::new_err("vehicle spec 'wheels' must be a list of dicts"))?;
    let wheels = wheels.iter().map(wheel).collect::<PyResult<Vec<Wheel>>>()?;
    let spec = VehicleSpec {
        mass: number(value, "mass")?,
        inertia: vector3(value, "inertia")?,
        size: vector3(value, "size")?,
        wheels,
        engine: engine(&field(value, "engine")?)?,
        gearbox: gearbox(&field(value, "gearbox")?)?,
        arb: match value.get_item("arb")? {
            Some(value) => {
                let values = value.extract::<Vec<f64>>().map_err(|_| {
                    PyTypeError::new_err("vehicle spec 'arb' must be a list of two floats")
                })?;
                if values.len() != 2 {
                    return Err(PyValueError::new_err("vehicle spec 'arb' needs two values"));
                }
                [values[0], values[1]]
            }
            None => [0.0, 0.0],
        },
        cd: optional_number(value, "cd", 0.35)?,
        area: optional_number(value, "area", 2.0)?,
        rolling: optional_number(value, "rolling", 0.015)?,
        max_steer: optional_number(value, "max_steer", 0.6)?,
        brake_torque: optional_number(value, "brake_torque", 3000.0)?,
        model_origin: optional_vector3(value, "model_origin", [0.0, 0.0, 0.0])?,
    };
    spec.validate().map_err(PyValueError::new_err)?;
    Ok(spec)
}

fn list<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values)
}

fn sim_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn snapshot_dict<'py>(
    py: Python<'py>,
    state: &manifoldx_core::vehicles::Snapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let slip = state
        .slip
        .iter()
        .map(|pair| PyTuple::new(py, pair))
        .collect::<PyResult<Vec<_>>>()?;
    let out = PyDict::new(py);
    out.set_item("position", PyTuple::new(py, state.position)?)?;
    out.set_item("velocity", PyTuple::new(py, state.velocity)?)?;
    out.set_item("angular", PyTuple::new(py, state.angular)?)?;
    out.set_item("rotation", PyTuple::new(py, state.rotation)?)?;
    out.set_item("load", list(py, &state.load)?)?;
    out.set_item("compression", list(py, &state.compression)?)?;
    out.set_item("ground_y", list(py, &state.ground_y)?)?;
    out.set_item("omega", list(py, &state.omega)?)?;
    out.set_item("wheel_turn", list(py, &state.wheel_turn)?)?;
    out.set_item("slip", slip)?;
    out.set_item("steer_angle", list(py, &state.steer_angle)?)?;
    out.set_item("gear", state.gear)?;
    out.set_item("shift_timer", state.shift_timer)?;
    out.set_item("rpm", state.rpm)?;
    out.set_item("impulse", state.impulse)?;
    Ok(out)
}

/// A native one-vehicle solver. Configure it, then either step it directly or
/// hand it to `NativeVehicleRuntime`.
#[pyclass(name = "_VehicleSim", module = "manifoldx._core_native", frozen)]
pub struct PyVehicleSim {
    inner: Mutex<Option<Sim>>,
}

impl PyVehicleSim {
    fn clone_sim(&self) -> PyResult<Sim> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vehicle sim lock poisoned"))?;
        guard
            .as_ref()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("vehicle sim is unavailable"))
    }

    fn with<R>(&self, action: impl FnOnce(&mut Sim) -> PyResult<R>) -> PyResult<R> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vehicle sim lock poisoned"))?;
        let sim = guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("vehicle sim is unavailable"))?;
        action(sim)
    }
}

#[pymethods]
impl PyVehicleSim {
    #[new]
    fn new(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let sim = Sim::new(parse_spec(spec)?).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Mutex::new(Some(sim)),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vehicle sim lock poisoned"))?;
        Ok(match guard.as_ref() {
            Some(sim) => format!("_VehicleSim(wheels={})", sim.spec().wheels.len()),
            None => "_VehicleSim(unavailable)".into(),
        })
    }

    /// Let the wheels stand on the plane `y = level`.
    #[pyo3(signature = (level=0.0, grip=1.0))]
    fn set_flat_ground(&self, level: f64, grip: f64) -> PyResult<()> {
        let ground = Ground::flat(level, grip).map_err(PyValueError::new_err)?;
        self.with(|sim| {
            sim.set_ground(ground);
            Ok(())
        })
    }

    /// Let the wheels stand on a road along x: a piecewise-linear profile, square
    /// kerbs at each `kerbs` x and an optional bridge deck `(x0, x1, y)`.
    #[pyo3(signature = (points, kerbs=None, kerb_height=0.15, bridge=None, grip=1.0))]
    fn set_profile_ground(
        &self,
        points: Vec<(f64, f64)>,
        kerbs: Option<Vec<f64>>,
        kerb_height: f64,
        bridge: Option<(f64, f64, f64)>,
        grip: f64,
    ) -> PyResult<()> {
        let ground = Ground::profile(points, kerbs.unwrap_or_default(), kerb_height, bridge, grip)
            .map_err(PyValueError::new_err)?;
        self.with(|sim| {
            sim.set_ground(ground);
            Ok(())
        })
    }

    /// Put the car at rest at `(x, y, z)` facing `yaw`, on the level nearest `y`.
    /// Non-finite input is a `ValueError` and leaves the state unchanged.
    fn place(&self, x: f64, y: f64, z: f64, yaw: f64) -> PyResult<()> {
        self.with(|sim| sim.place(x, y, z, yaw).map_err(PyValueError::new_err))
    }

    /// Prepare bounded controls for the next direct steps.
    fn set_controls(&self, steer: f64, throttle: f64, brake: f64, handbrake: f64) -> PyResult<()> {
        let controls =
            Controls::new(steer, throttle, brake, handbrake).map_err(PyValueError::new_err)?;
        self.with(|sim| sim.set_controls(controls).map_err(PyValueError::new_err))
    }

    /// Run `substeps` fixed 1/120 s steps with Python detached.
    fn step<'py>(&self, py: Python<'py>, substeps: u32) -> PyResult<Bound<'py, PyDict>> {
        if substeps == 0 || substeps > 4096 {
            return Err(PyValueError::new_err("substeps must be between 1 and 4096"));
        }
        let result = py.detach(|| {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| "vehicle sim lock poisoned".to_string())?;
            let sim = guard
                .as_mut()
                .ok_or_else(|| "vehicle sim is unavailable".to_string())?;
            let start = Instant::now();
            let steps = sim.step_many(substeps).map_err(|error| error.to_string())?;
            Ok::<(u32, u64), String>((steps, start.elapsed().as_nanos() as u64))
        });
        let (steps, ns) = result.map_err(sim_error)?;
        let out = PyDict::new(py);
        out.set_item("steps", steps)?;
        out.set_item("ns", ns)?;
        out.set_item("ms_per_step", ns as f64 / 1e6 / steps as f64)?;
        Ok(out)
    }

    /// An owned copy of every solver output. The state is copied under one
    /// brief lock; Python objects are built after it is released.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.with(|sim| Ok(sim.snapshot()))?;
        snapshot_dict(py, &state)
    }

    /// The body's position and quaternion. Copied under one brief lock;
    /// Python objects are built after it is released.
    fn pose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (position, rotation) = self.with(|sim| Ok(sim.pose()))?;
        PyTuple::new(
            py,
            [
                PyTuple::new(py, position)?.into_any(),
                PyTuple::new(py, rotation)?.into_any(),
            ],
        )
    }
}

fn admission<'py>(py: Python<'py>, admitted: Admission) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    match admitted {
        Admission::Admitted(sequence) => {
            out.set_item("status", "admitted")?;
            out.set_item("sequence", sequence)?;
        }
        Admission::Full => {
            out.set_item("status", "full")?;
        }
        Admission::Stopped => {
            out.set_item("status", "stopped")?;
        }
        Admission::Error(error) => {
            out.set_item("status", "error")?;
            out.set_item("error", error)?;
        }
    }
    Ok(out)
}

/// The existing `Runtime`, with one native vehicle system attached. Each run
/// frame is one fixed 1/120 s solver step; controls are bounded asynchronous
/// commands and the pose lands in the World's `f32` columns.
#[pyclass(
    name = "NativeVehicleRuntime",
    module = "manifoldx._core_native",
    frozen
)]
pub struct PyVehicleRuntime {
    runtime: Mutex<Option<Runtime>>,
    owner: Arc<Mutex<Option<World>>>,
    control: Control,
}

#[pymethods]
impl PyVehicleRuntime {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (world, handles, sim, position="position", rotation="rotation", velocity="velocity", angular="angular", capacity=64, max_rows=4096, batches_per_frame=8))]
    fn new(
        py: Python<'_>,
        world: &PyNativeWorld,
        handles: &Bound<'_, PyAny>,
        sim: &PyVehicleSim,
        position: &str,
        rotation: &str,
        velocity: &str,
        angular: &str,
        capacity: usize,
        max_rows: usize,
        batches_per_frame: usize,
    ) -> PyResult<Self> {
        let ids: Vec<EntityId> = decode_handles_limited(py, handles, Some((1, 16)))?;
        if ids.len() != 1 {
            return Err(PyValueError::new_err(
                "a vehicle runtime needs exactly one entity handle",
            ));
        }
        if capacity == 0 || max_rows == 0 || batches_per_frame == 0 {
            return Err(PyValueError::new_err("runtime limits must be positive"));
        }
        let mut guard = world
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("World lock poisoned"))?;
        let columns = {
            let owned = guard
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("World is already owned by a runtime"))?;
            let resolve = |name: &str| -> PyResult<usize> {
                owned
                    .component_index(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))
            };
            let columns = VehicleColumns {
                position: resolve(position)?,
                rotation: resolve(rotation)?,
                velocity: resolve(velocity)?,
                angular: resolve(angular)?,
            };
            Runtime::validate_vehicle(owned, columns, ids[0]).map_err(PyValueError::new_err)?;
            columns
        };
        let sim = sim.clone_sim()?;
        let runtime = Runtime::with_vehicle(
            guard.take().expect("checked World ownership"),
            columns,
            sim,
            ids[0],
            capacity,
            max_rows,
            batches_per_frame,
        );
        let runtime = match runtime {
            Ok(runtime) => runtime,
            Err(boxed) => {
                let (world, error) = *boxed;
                *guard = Some(world);
                return Err(PyValueError::new_err(error));
            }
        };
        let control = runtime.control();
        drop(guard);
        Ok(Self {
            runtime: Mutex::new(Some(runtime)),
            owner: world.inner.clone(),
            control,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeVehicleRuntime(phase={}, vehicle_steps={})",
            self.phase(),
            self.control.stats().vehicle_steps
        )
    }

    /// Submit bounded controls without blocking; they apply at a frame boundary.
    #[pyo3(signature = (steer, throttle, brake, handbrake))]
    fn set_controls<'py>(
        &self,
        py: Python<'py>,
        steer: f64,
        throttle: f64,
        brake: f64,
        handbrake: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        Controls::new(steer, throttle, brake, handbrake).map_err(PyValueError::new_err)?;
        admission(
            py,
            self.control.submit(Command::Controls {
                steer,
                throttle,
                brake,
                handbrake,
            }),
        )
    }

    /// Execute `frames` fixed 1/120 s substeps on THIS thread, Python detached.
    #[pyo3(signature = (frames, *, pace_hz=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        frames: u64,
        pace_hz: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pace = match pace_hz {
            Some(hz) if hz.is_finite() && hz > 0.0 => Some(
                Duration::try_from_secs_f64(1.0 / hz)
                    .map_err(|_| PyValueError::new_err("invalid pace_hz"))?,
            ),
            Some(_) => return Err(PyValueError::new_err("pace_hz must be finite and positive")),
            None => None,
        };
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vehicle runtime lock poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("runtime is running or closed"))?;
        let result = py.detach(|| runtime.run(frames, DT as f32, pace));
        let mut slot = self
            .runtime
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vehicle runtime lock poisoned"))?;
        *slot = Some(runtime);
        drop(slot);
        let stats = result.map_err(PyRuntimeError::new_err)?;
        stats_dict(py, &stats)
    }

    fn stop(&self) {
        self.control.stop();
    }

    #[getter]
    fn phase(&self) -> &'static str {
        match self.control.phase() {
            Phase::Ready => "ready",
            Phase::Running => "running",
            Phase::Stopped => "stopped",
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_dict(py, &self.control.stats())
    }

    /// Owned final solver state of the active simulation, without callbacks.
    /// The state is copied under one brief lock; Python objects are built
    /// after it is released.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = {
            let guard = self
                .runtime
                .lock()
                .map_err(|_| PyRuntimeError::new_err("vehicle runtime lock poisoned"))?;
            let runtime = guard
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("runtime is running or closed"))?;
            runtime
                .vehicle_state()
                .ok_or_else(|| PyRuntimeError::new_err("runtime has no vehicle"))?
        };
        snapshot_dict(py, &state)
    }

    /// Take the next completion, freeing capacity.
    fn take_outcome<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(outcome) = self.control.take() else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("sequence", outcome.sequence)?;
        out.set_item("frame", outcome.frame)?;
        match outcome.result {
            Err(error) => {
                out.set_item(
                    "status",
                    if error == "cancelled on stop" {
                        "cancelled"
                    } else {
                        "rejected"
                    },
                )?;
                out.set_item("error", error)?;
            }
            Ok(_) => {
                out.set_item("status", "applied")?;
            }
        }
        Ok(Some(out))
    }

    /// Return World ownership; running execution must stop first.
    fn close(&self) -> PyResult<()> {
        let mut slot = self
            .runtime
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vehicle runtime lock poisoned"))?;
        let runtime = slot
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("runtime is running or closed"))?;
        *self
            .owner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("World lock poisoned"))? =
            Some(runtime.into_world());
        Ok(())
    }
}

impl Drop for PyVehicleRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.get_mut().unwrap().take() {
            *self.owner.lock().unwrap() = Some(runtime.into_world());
        }
    }
}
