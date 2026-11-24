//! Explicit production native execution API, independent of Engine bootstrap.
use super::*;
use manifoldx_core::ecs::ComponentSchema;
use manifoldx_core::runtime::{Admission, Command, Control, Phase, Runtime, Stats};
use std::time::Duration;

/// Owns the existing World until close/drop. run() stays on its calling thread.
#[pyclass(name = "NativeRuntime", module = "manifoldx._core_native", frozen)]
pub struct PyNativeRuntime {
    runtime: Mutex<Option<Runtime>>,
    owner: Arc<Mutex<Option<World>>>,
    control: Control,
    schemas: Vec<ComponentSchema>,
    max_rows: usize,
}
impl PyNativeRuntime {
    fn component(&self, name: &str) -> PyResult<(usize, &ComponentSchema)> {
        self.schemas
            .iter()
            .enumerate()
            .find(|(_, s)| s.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))
    }
    fn ids(
        &self,
        py: Python<'_>,
        handles: &Bound<'_, PyAny>,
        row_bytes: usize,
    ) -> PyResult<Vec<EntityId>> {
        decode_handles_limited(py, handles, Some((self.max_rows, row_bytes)))
    }

    fn admission<'py>(&self, py: Python<'py>, command: Command) -> PyResult<Bound<'py, PyDict>> {
        let admitted = self.control.submit(command);
        let d = PyDict::new(py);
        match admitted {
            Admission::Admitted(sequence) => {
                d.set_item("status", "admitted")?;
                d.set_item("sequence", sequence)?;
            }
            Admission::Full => {
                d.set_item("status", "full")?;
            }
            Admission::Stopped => {
                d.set_item("status", "stopped")?;
            }
            Admission::Error(error) => {
                d.set_item("status", "error")?;
                d.set_item("error", error)?;
            }
        }
        Ok(d)
    }
}
#[pymethods]
impl PyNativeRuntime {
    #[new]
    #[pyo3(signature = (world, position, velocity, *, capacity=64, max_rows=4096, batches_per_frame=8))]
    fn new(
        world: &PyNativeWorld,
        position: &str,
        velocity: &str,
        capacity: usize,
        max_rows: usize,
        batches_per_frame: usize,
    ) -> PyResult<Self> {
        let mut guard = world
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("World lock poisoned"))?;
        let owned = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("World is already owned by NativeRuntime"))?;
        let p = owned
            .component_index(position)
            .ok_or_else(|| PyValueError::new_err("unknown position column"))?;
        let v = owned
            .component_index(velocity)
            .ok_or_else(|| PyValueError::new_err("unknown velocity column"))?;
        Runtime::validate(owned, p, v, capacity, max_rows, batches_per_frame)
            .map_err(PyValueError::new_err)?;
        let schemas = owned.schemas().cloned().collect();
        let runtime = Runtime::new(
            guard.take().unwrap(),
            p,
            v,
            capacity,
            max_rows,
            batches_per_frame,
        )
        .unwrap_or_else(|_| unreachable!("validated World"));
        let control = runtime.control();
        drop(guard);
        Ok(Self {
            runtime: Mutex::new(Some(runtime)),
            owner: world.inner.clone(),
            control,
            schemas,
            max_rows,
        })
    }
    /// Nonblocking owned row batch; no buffer acquisition/release under any lock.
    fn submit_rows<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        handles: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (component, schema) = self.component(name)?;
        let ids = self.ids(py, handles, 17 + schema.width as usize * 4)?;
        let data = match schema.element {
            ElementType::F32 => ColumnData::F32(read_matrix::<f32>(
                py,
                data,
                ids.len(),
                schema.width as usize,
                schema.element,
                "command rows",
            )?),
            ElementType::U32 => ColumnData::U32(read_matrix::<u32>(
                py,
                data,
                ids.len(),
                schema.width as usize,
                schema.element,
                "command rows",
            )?),
        };
        self.admission(
            py,
            Command::Update {
                component,
                ids,
                data,
            },
        )
    }
    fn submit_destroy<'py>(
        &self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = self.ids(py, handles, 16)?;
        self.admission(py, Command::Destroy(ids))
    }
    fn request_read<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (component, schema) = self.component(name)?;
        let ids = self.ids(py, handles, 17 + schema.width as usize * 4)?;
        self.admission(py, Command::Read { component, ids })
    }
    /// Execute on THIS thread with Python detached. Single-shot, bounded frames.
    #[pyo3(signature = (frames, dt, *, pace_hz=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        frames: u64,
        dt: f32,
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
            .unwrap()
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("runtime is running or closed"))?;
        let result = py.detach(|| runtime.run(frames, dt, pace));
        *self.runtime.lock().unwrap() = Some(runtime);
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
    /// Take the next completion, freeing capacity. Returned bytes own their memory.
    fn take_outcome<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(outcome) = self.control.take() else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("sequence", outcome.sequence)?;
        d.set_item("frame", outcome.frame)?;
        match outcome.result {
            Err(error) => {
                d.set_item(
                    "status",
                    if error == "cancelled on stop" {
                        "cancelled"
                    } else {
                        "rejected"
                    },
                )?;
                d.set_item("error", error)?;
            }
            Ok(read) => {
                d.set_item("status", "applied")?;
                if let Some(read) = read {
                    d.set_item("handles", PyBytes::new(py, &handles_bytes(&read.ids)))?;
                    d.set_item(
                        "membership",
                        PyBytes::new(
                            py,
                            &read
                                .membership
                                .iter()
                                .map(|&x| u8::from(x))
                                .collect::<Vec<_>>(),
                        ),
                    )?;
                    d.set_item("data", PyBytes::new(py, &column_bytes(&read.data)))?;
                }
            }
        }
        Ok(Some(d))
    }
    /// Return ownership to the original World. Running execution must stop/return first.
    fn close(&self) -> PyResult<()> {
        let mut slot = self.runtime.lock().unwrap();
        let runtime = slot
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("runtime is running or closed"))?;
        *self.owner.lock().unwrap() = Some(runtime.into_world());
        Ok(())
    }
}
impl Drop for PyNativeRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.get_mut().unwrap().take() {
            *self.owner.lock().unwrap() = Some(runtime.into_world());
        }
    }
}
fn stats_dict<'py>(py: Python<'py>, s: &Stats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    macro_rules! fields { ($($name:ident),*) => { $(d.set_item(stringify!($name), s.$name)?;)* }; }
    fields!(
        frames,
        native_steps,
        rows_integrated,
        admitted,
        applied,
        rejected,
        cancelled,
        full,
        backlog,
        outcomes,
        bytes_in,
        bytes_out,
        readback_bytes,
        native_ns,
        frame_ns,
        max_frame_ns,
        paced_ns,
        wall_ns
    );
    d.set_item("first_error", s.first_error.as_deref())?;
    Ok(d)
}
