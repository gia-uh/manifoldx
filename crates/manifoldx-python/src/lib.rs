//! Python bindings for the ManifoldX runtime core.
//!
//! Phase 0 ships one extension, `manifoldx._core_native`, whose bootstrap
//! reports what the core crate actually is. Phase 1a adds the private
//! `_EntitySlots` binding so an installed wheel can exercise the native slot
//! allocator. It is a testable seam, not the public ECS or Engine runtime.
//!
//! The binding owns every returned value (slot lists, handle tuples and
//! snapshots); it retains no caller pointer. Calls lock one mutex, so a
//! free-threaded interpreter serializes mutations at this boundary.

use std::sync::Mutex;

use manifoldx_core::ecs::{EntityError, EntityId, EntitySlots, SlotState};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// What the runtime core really is, from the Rust build.
#[pyfunction]
fn bootstrap(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    let pyver = py.version_info();
    info.set_item("core", manifoldx_core::RUNTIME_NAME)?;
    info.set_item("version", manifoldx_core::VERSION)?;
    info.set_item("runtime", manifoldx_core::RUNTIME_STATUS)?;
    info.set_item("python", format!("{}.{}", pyver.major, pyver.minor))?;
    Ok(info)
}

/// A private fixed-capacity entity-slot allocator.
///
/// This is a testable native seam, not the public ECS. `spawn` returns slot
/// integers; `reserve` returns `(slot, generation)` handles for exact identity
/// checks. All state returned by `snapshot` is an owned copy.
#[pyclass(name = "_EntitySlots", module = "manifoldx._core_native", frozen)]
struct PyEntitySlots {
    inner: Mutex<EntitySlots>,
}

fn map_error(error: EntityError) -> PyErr {
    let message = error.to_string();
    match error {
        EntityError::Allocation { .. } => PyMemoryError::new_err(message),
        EntityError::SlotOutOfRange { .. } => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

fn to_handles(ids: &[EntityId]) -> Vec<(u32, u64)> {
    ids.iter().map(|id| (id.slot(), id.generation())).collect()
}

fn from_handles(handles: &[(u32, u64)]) -> Vec<EntityId> {
    handles
        .iter()
        .map(|(slot, generation)| EntityId::new(*slot, *generation))
        .collect()
}

#[pymethods]
impl PyEntitySlots {
    #[new]
    fn new(capacity: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(EntitySlots::new(capacity).map_err(map_error)?),
        })
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.lock().expect("entity slot lock");
        format!(
            "_EntitySlots(capacity={}, alive={}, reserved={}, high_water={})",
            inner.capacity(),
            inner.alive_count(),
            inner.reserved_count(),
            inner.high_water()
        )
    }

    /// Fixed capacity declared at construction.
    #[getter]
    fn capacity(&self) -> u32 {
        self.inner.lock().expect("entity slot lock").capacity()
    }

    /// Incremented by each successful mutation.
    #[getter]
    fn version(&self) -> u64 {
        self.inner.lock().expect("entity slot lock").version()
    }

    /// One past the highest slot ever committed live.
    #[getter]
    fn high_water(&self) -> u32 {
        self.inner.lock().expect("entity slot lock").high_water()
    }

    /// Number of live entities.
    #[getter]
    fn alive_count(&self) -> u32 {
        self.inner.lock().expect("entity slot lock").alive_count()
    }

    /// Number of reserved but not yet committed entities.
    #[getter]
    fn reserved_count(&self) -> u32 {
        self.inner
            .lock()
            .expect("entity slot lock")
            .reserved_count()
    }

    /// The state name of one slot.
    fn state(&self, slot: u32) -> PyResult<&'static str> {
        self.inner
            .lock()
            .expect("entity slot lock")
            .state_of(slot)
            .map(SlotState::as_str)
            .ok_or_else(|| PyValueError::new_err(format!("slot {slot} is outside capacity")))
    }

    /// The current `(slot, generation)` identity of one slot.
    fn current(&self, slot: u32) -> PyResult<(u32, u64)> {
        let id = self
            .inner
            .lock()
            .expect("entity slot lock")
            .current_identity(slot)
            .map_err(map_error)?;
        Ok((id.slot(), id.generation()))
    }

    /// Reserve `n` slots and return exact `(slot, generation)` handles.
    fn reserve(&self, n: u32) -> PyResult<Vec<(u32, u64)>> {
        let ids = self
            .inner
            .lock()
            .expect("entity slot lock")
            .reserve(n)
            .map_err(map_error)?;
        Ok(to_handles(&ids))
    }

    /// Make reserved handles live.
    fn commit(&self, handles: Vec<(u32, u64)>) -> PyResult<()> {
        let ids = from_handles(&handles);
        self.inner
            .lock()
            .expect("entity slot lock")
            .commit(&ids)
            .map_err(map_error)
    }

    /// Return reserved handles to the free set without making them live.
    fn cancel(&self, handles: Vec<(u32, u64)>) -> PyResult<()> {
        let ids = from_handles(&handles);
        self.inner
            .lock()
            .expect("entity slot lock")
            .cancel(&ids)
            .map_err(map_error)
    }

    /// Destroy live handles.
    fn destroy(&self, handles: Vec<(u32, u64)>) -> PyResult<()> {
        let ids = from_handles(&handles);
        self.inner
            .lock()
            .expect("entity slot lock")
            .destroy(&ids)
            .map_err(map_error)
    }

    /// Reserve and commit in one call; returns slot integers.
    fn spawn(&self, n: u32) -> PyResult<Vec<u32>> {
        let ids = self
            .inner
            .lock()
            .expect("entity slot lock")
            .spawn(n)
            .map_err(map_error)?;
        Ok(ids.iter().map(|id| id.slot()).collect())
    }

    /// An owned snapshot: slots, generations, states, free order and counters.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self.inner.lock().expect("entity slot lock").snapshot();
        let dict = PyDict::new(py);
        dict.set_item("capacity", snapshot.capacity)?;
        dict.set_item("slots", snapshot.slots)?;
        dict.set_item("generations", snapshot.generations)?;
        dict.set_item(
            "states",
            snapshot
                .states
                .iter()
                .map(|state| state.as_str())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "alive",
            snapshot
                .states
                .iter()
                .map(|state| *state == SlotState::Alive)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "reserved",
            snapshot
                .states
                .iter()
                .map(|state| *state == SlotState::Reserved)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("free_order", snapshot.free_order)?;
        dict.set_item("high_water", snapshot.high_water)?;
        dict.set_item("version", snapshot.version)?;
        Ok(dict)
    }
}

/// `manifoldx._core_native`
#[pymodule]
fn _core_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    m.add_class::<PyEntitySlots>()?;
    Ok(())
}
