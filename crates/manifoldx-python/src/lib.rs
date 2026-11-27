//! Python bindings for the ManifoldX runtime core.
//!
//! Phase 0 ships one extension, `manifoldx._core_native`, whose bootstrap
//! reports what the core crate actually is. Phase 1a adds the private
//! `_EntitySlots` binding so an installed wheel can exercise the native slot
//! allocator. Phase 1b adds the private `_NativeWorld` binding: explicit
//! component schemas, typed `f32`/`u32` rows, membership, validated bulk
//! updates and owned readback.
//!
//! This is a testable native seam, not the public ECS, the Engine runtime or a
//! compatibility backend. The binding owns every returned value; no pointer
//! into native storage is exported. NumPy inputs are copied into typed Rust
//! vectors with no World lock held, then committed after reacquiring it. Calls
//! lock one mutex only for schema resolution and the validated operation, so a
//! free-threaded interpreter serializes mutations here and buffer exporters
//! may safely reenter.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};
mod frames;
mod graphics;
mod gui;
mod kernel;
mod runtime;
mod stream_builder;
mod stream_cells;
mod vehicles;

use manifoldx_core::ecs::{
    ColumnData, ComponentPayload, ElementType, EntityError, EntityId, EntitySlots, SlotState,
    World, WorldError,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// What the runtime core really is, from the Rust build.
#[pyfunction]
fn bootstrap(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    let pyver = py.version_info();
    info.set_item("core", manifoldx_core::RUNTIME_NAME)?;
    info.set_item("version", manifoldx_core::VERSION)?;
    info.set_item("runtime", manifoldx_core::RUNTIME_STATUS)?;
    info.set_item("graphics", "native desktop base")?;
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

/// A private fixed-capacity component World.
///
/// Schemas are registered explicitly before the first entity is published.
/// Component payloads and updates accept only C-contiguous NumPy (or other
/// buffer) arrays of dtype `float32` or `uint32`, with the documented shapes.
/// Values are copied in; no writable native memory is returned. Readback is an
/// owned snapshot encoded as little-endian bytes.
#[pyclass(name = "_NativeWorld", module = "manifoldx._core_native", frozen)]
struct PyNativeWorld {
    inner: Arc<Mutex<Option<World>>>,
}

struct WorldGuard<'a>(MutexGuard<'a, Option<World>>);
impl Deref for WorldGuard<'_> {
    type Target = World;
    fn deref(&self) -> &World {
        self.0.as_ref().expect("checked World ownership")
    }
}
impl DerefMut for WorldGuard<'_> {
    fn deref_mut(&mut self) -> &mut World {
        self.0.as_mut().expect("checked World ownership")
    }
}
impl PyNativeWorld {
    fn lock(&self) -> PyResult<WorldGuard<'_>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("World lock poisoned"))?;
        if guard.is_none() {
            return Err(PyRuntimeError::new_err(
                "World is owned by NativeRuntime; close it first",
            ));
        }
        Ok(WorldGuard(guard))
    }
}

fn map_world_error(error: WorldError) -> PyErr {
    let message = error.to_string();
    match &error {
        WorldError::Entity(EntityError::Allocation { .. }) | WorldError::Allocation { .. } => {
            PyMemoryError::new_err(message)
        }
        WorldError::Entity(_) => PyRuntimeError::new_err(message),
        WorldError::PayloadKind { .. } => PyTypeError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn column_bytes(data: &ColumnData) -> Vec<u8> {
    match data {
        ColumnData::F32(values) => f32_bytes(values),
        ColumnData::U32(values) => u32_bytes(values),
    }
}

fn handles_bytes(ids: &[EntityId]) -> Vec<u8> {
    let mut values = Vec::with_capacity(ids.len() * 2);
    for id in ids {
        values.push(id.slot() as u64);
        values.push(id.generation());
    }
    u64_bytes(&values)
}

fn decode_handles(py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<Vec<EntityId>> {
    decode_handles_limited(py, object, None)
}

fn decode_handles_limited(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    limits: Option<(usize, usize)>,
) -> PyResult<Vec<EntityId>> {
    let buffer = PyBuffer::<u64>::get(object).map_err(|_| {
        PyTypeError::new_err("entity handles must be a C-contiguous uint64 buffer of shape (n, 2)")
    })?;
    if buffer.dimensions() != 2 || buffer.shape().get(1) != Some(&2) {
        return Err(PyValueError::new_err(format!(
            "entity handles must have shape (n, 2); got {:?}",
            buffer.shape()
        )));
    }
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err("entity handles must be C-contiguous"));
    }
    if let Some((max_rows, row_bytes)) = limits {
        let rows = buffer.shape()[0];
        if rows > max_rows
            || rows
                .checked_mul(row_bytes)
                .is_none_or(|bytes| bytes > manifoldx_core::runtime::MAX_COMMAND_BYTES)
        {
            return Err(PyValueError::new_err("command exceeds row or byte limit"));
        }
    }
    let flat = buffer.to_vec(py).map_err(|_| {
        PyTypeError::new_err("entity handles must be a C-contiguous uint64 buffer of shape (n, 2)")
    })?;
    let mut ids = Vec::with_capacity(flat.len() / 2);
    for pair in flat.as_chunks::<2>().0 {
        let slot = u32::try_from(pair[0]).map_err(|_| {
            PyValueError::new_err(format!("entity slot {} does not fit in u32", pair[0]))
        })?;
        ids.push(EntityId::new(slot, pair[1]));
    }
    Ok(ids)
}

fn read_matrix<'py, T: pyo3::buffer::Element>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    rows: usize,
    width: usize,
    element: ElementType,
    label: &str,
) -> PyResult<Vec<T>> {
    let buffer = PyBuffer::<T>::get(object).map_err(|_| {
        PyTypeError::new_err(format!(
            "{label} must be a buffer of dtype {}",
            element.as_str()
        ))
    })?;
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{label} must be C-contiguous"
        )));
    }
    if buffer.dimensions() != 2
        || buffer.shape().first() != Some(&rows)
        || buffer.shape().get(1) != Some(&width)
    {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape ({rows}, {width}); got {:?}",
            buffer.shape()
        )));
    }
    buffer.to_vec(py).map_err(|_| {
        PyTypeError::new_err(format!(
            "{label} must be a buffer of dtype {}",
            element.as_str()
        ))
    })
}

fn read_field<'py, T: pyo3::buffer::Element>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    rows: usize,
    element: ElementType,
    label: &str,
) -> PyResult<Vec<T>> {
    let buffer = PyBuffer::<T>::get(object).map_err(|_| {
        PyTypeError::new_err(format!(
            "{label} must be a buffer of dtype {}",
            element.as_str()
        ))
    })?;
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{label} must be C-contiguous"
        )));
    }
    if buffer.dimensions() != 1 || buffer.shape().first() != Some(&rows) {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape ({rows},); got {:?}",
            buffer.shape()
        )));
    }
    buffer.to_vec(py).map_err(|_| {
        PyTypeError::new_err(format!(
            "{label} must be a buffer of dtype {}",
            element.as_str()
        ))
    })
}

fn parse_element(kind: &str) -> PyResult<ElementType> {
    match kind {
        "f32" => Ok(ElementType::F32),
        "u32" => Ok(ElementType::U32),
        other => Err(PyValueError::new_err(format!(
            "unsupported component element type '{other}'; expected 'f32' or 'u32'"
        ))),
    }
}

struct ResolvedComponent {
    index: usize,
    name: String,
    element: ElementType,
    width: u32,
}

fn resolve_component(world: &World, name: &str) -> Option<ResolvedComponent> {
    let index = world.component_index(name)?;
    let schema = world.schema(index)?;
    Some(ResolvedComponent {
        index,
        name: schema.name.clone(),
        element: schema.element,
        width: schema.width,
    })
}

fn component_matches(world: &World, resolved: &ResolvedComponent) -> bool {
    world
        .schema(resolved.index)
        .map(|schema| {
            schema.name == resolved.name
                && schema.element == resolved.element
                && schema.width == resolved.width
        })
        .unwrap_or(false)
}

#[pymethods]
impl PyNativeWorld {
    #[new]
    fn new(capacity: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(
                World::new(capacity).map_err(map_world_error)?,
            ))),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let inner = self.lock()?;
        Ok(format!(
            "_NativeWorld(capacity={}, alive={}, schemas={}, published={})",
            inner.capacity(),
            inner.alive_count(),
            inner.schemas().count(),
            inner.published()
        ))
    }

    /// Fixed entity capacity.
    #[getter]
    fn capacity(&self) -> PyResult<u32> {
        Ok(self.lock()?.capacity())
    }

    /// Number of live entities.
    #[getter]
    fn alive_count(&self) -> PyResult<u32> {
        Ok(self.lock()?.alive_count())
    }

    /// One past the highest slot ever published.
    #[getter]
    fn high_water(&self) -> PyResult<u32> {
        Ok(self.lock()?.high_water())
    }

    /// Incremented by each successful schema or data mutation.
    #[getter]
    fn version(&self) -> PyResult<u64> {
        Ok(self.lock()?.version())
    }

    /// Whether an entity has ever been published.
    #[getter]
    fn published(&self) -> PyResult<bool> {
        Ok(self.lock()?.published())
    }

    /// Registered schemas as `[(name, kind, width), ...]`.
    #[getter]
    fn schemas(&self) -> PyResult<Vec<(String, &'static str, u32)>> {
        let schemas = {
            let world = self.lock()?;
            world
                .schemas()
                .map(|schema| (schema.name.clone(), schema.element.as_str(), schema.width))
                .collect()
        };
        Ok(schemas)
    }

    /// Register one component schema and return its index.
    ///
    /// `kind` is `"f32"` or `"u32"`; `width` is the fixed scalars per row and
    /// may be zero for a marker. A duplicate name, an empty name, an
    /// over-wide row or registration after the first published entity is
    /// rejected without changing live state.
    fn register_component(&self, name: &str, kind: &str, width: u32) -> PyResult<usize> {
        let element = parse_element(kind)?;
        let result = {
            let mut world = self.lock()?;
            world.register(name, element, width)
        };
        result.map_err(map_world_error)
    }

    /// Publish `count` entities with `{component_name: 2-D array}` payloads.
    ///
    /// Each payload is C-contiguous with dtype `float32` or `uint32` and shape
    /// `(count, width)`. A zero-width component accepts `None` or a zero-sized
    /// `(count, 0)` buffer. Returns owned handles (`uint64`, shape `(n, 2)`,
    /// little-endian) plus honest boundary byte counts.
    ///
    /// The World lock is held only to resolve a schema and again for the
    /// validated commit. No Python buffer is acquired or released under it, so
    /// an exporter's `__buffer__` or `__release_buffer__` may reenter.
    fn spawn<'py>(
        &self,
        py: Python<'py>,
        count: u32,
        payload: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut payloads = Vec::with_capacity(payload.len());
        let mut resolved = Vec::with_capacity(payload.len());
        let mut payload_in = 0usize;
        for (key, value) in payload {
            let name: String = key
                .extract()
                .map_err(|_| PyTypeError::new_err("component payload keys must be strings"))?;
            let component = {
                let world = self.lock()?;
                resolve_component(&world, &name)
            }
            .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
            let width = component.width as usize;
            let data = if value.is_none() {
                if width != 0 {
                    return Err(PyTypeError::new_err(format!(
                        "component '{name}' needs a {} payload of shape ({count}, {width})",
                        component.element.as_str()
                    )));
                }
                match component.element {
                    ElementType::F32 => ColumnData::F32(Vec::new()),
                    ElementType::U32 => ColumnData::U32(Vec::new()),
                }
            } else {
                let label = format!("component '{name}' payload");
                match component.element {
                    ElementType::F32 => ColumnData::F32(read_matrix::<f32>(
                        py,
                        &value,
                        count as usize,
                        width,
                        component.element,
                        &label,
                    )?),
                    ElementType::U32 => ColumnData::U32(read_matrix::<u32>(
                        py,
                        &value,
                        count as usize,
                        width,
                        component.element,
                        &label,
                    )?),
                }
            };
            payload_in += data.byte_len();
            payloads.push(ComponentPayload {
                component: component.index,
                data,
            });
            resolved.push(component);
        }

        let mut world = self.lock()?;
        for component in &resolved {
            if !component_matches(&world, component) {
                drop(world);
                return Err(PyRuntimeError::new_err(format!(
                    "component '{}' schema changed during submission",
                    component.name
                )));
            }
        }
        let result = world.spawn(count, &payloads);
        drop(world);
        let ids = result.map_err(map_world_error)?;

        let handles = handles_bytes(&ids);
        let dict = PyDict::new(py);
        dict.set_item("count", ids.len())?;
        dict.set_item("handles", PyBytes::new(py, &handles))?;
        dict.set_item("handles_in", 0usize)?;
        dict.set_item("payload_in", payload_in)?;
        dict.set_item("bytes_in", payload_in)?;
        dict.set_item("handles_out", handles.len())?;
        dict.set_item("bytes_out", handles.len())?;
        Ok(dict)
    }

    /// Overwrite whole member rows: `data` has shape `(len(handles), width)`.
    ///
    /// Every identity must be live and carry the component. The whole batch is
    /// validated before any row changes. Handles are decoded before the World
    /// lock is taken, and the commit is validated after reacquiring it.
    fn update_rows<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        handles: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let component = {
            let world = self.lock()?;
            resolve_component(&world, name)
        }
        .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let width = component.width as usize;
        let label = format!("component '{name}' rows");
        let payload = match component.element {
            ElementType::F32 => ColumnData::F32(read_matrix::<f32>(
                py,
                data,
                ids.len(),
                width,
                component.element,
                &label,
            )?),
            ElementType::U32 => ColumnData::U32(read_matrix::<u32>(
                py,
                data,
                ids.len(),
                width,
                component.element,
                &label,
            )?),
        };
        let payload_in = payload.byte_len();
        let handles_in = ids.len() * 16;

        let mut world = self.lock()?;
        if !component_matches(&world, &component) {
            drop(world);
            return Err(PyRuntimeError::new_err(format!(
                "component '{name}' schema changed during submission"
            )));
        }
        let result = world.update_rows(component.index, &ids, payload);
        drop(world);
        let rows = result.map_err(map_world_error)?;

        let dict = PyDict::new(py);
        dict.set_item("rows", rows)?;
        dict.set_item("handles_in", handles_in)?;
        dict.set_item("payload_in", payload_in)?;
        dict.set_item("bytes_in", handles_in + payload_in)?;
        Ok(dict)
    }

    /// Overwrite one field: `data` has shape `(len(handles),)`.
    ///
    /// The same lock discipline as `update_rows` applies: decode and copy
    /// Python buffers with no lock, then validate and commit after reacquiring.
    fn update_field<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        field: u32,
        handles: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let component = {
            let world = self.lock()?;
            resolve_component(&world, name)
        }
        .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let label = format!("component '{name}' field {field}");
        let payload = match component.element {
            ElementType::F32 => ColumnData::F32(read_field::<f32>(
                py,
                data,
                ids.len(),
                component.element,
                &label,
            )?),
            ElementType::U32 => ColumnData::U32(read_field::<u32>(
                py,
                data,
                ids.len(),
                component.element,
                &label,
            )?),
        };
        let payload_in = payload.byte_len();
        let handles_in = ids.len() * 16;

        let mut world = self.lock()?;
        if !component_matches(&world, &component) {
            drop(world);
            return Err(PyRuntimeError::new_err(format!(
                "component '{name}' schema changed during submission"
            )));
        }
        let result = world.update_field(component.index, field, &ids, payload);
        drop(world);
        let rows = result.map_err(map_world_error)?;

        let dict = PyDict::new(py);
        dict.set_item("rows", rows)?;
        dict.set_item("handles_in", handles_in)?;
        dict.set_item("payload_in", payload_in)?;
        dict.set_item("bytes_in", handles_in + payload_in)?;
        Ok(dict)
    }

    /// Bulk membership flags for `handles`, one `uint8` per entity.
    ///
    /// Handles are decoded with no lock; the query runs under one brief lock
    /// and the result is turned into Python objects after it is released.
    fn has<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let component = {
            let world = self.lock()?;
            resolve_component(&world, name)
        }
        .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let result = {
            let world = self.lock()?;
            world.memberships(component.index, &ids)
        };
        let flags = result.map_err(map_world_error)?;
        let handles_in = ids.len() * 16;
        let bytes: Vec<u8> = flags.iter().map(|flag| u8::from(*flag)).collect();

        let dict = PyDict::new(py);
        dict.set_item("count", flags.len())?;
        dict.set_item("flags", PyBytes::new(py, &bytes))?;
        dict.set_item("handles_in", handles_in)?;
        dict.set_item("payload_in", 0usize)?;
        dict.set_item("bytes_in", handles_in)?;
        dict.set_item("bytes_out", bytes.len())?;
        Ok(dict)
    }

    /// Destroy live entities and clear their rows and membership.
    ///
    /// Handles are decoded with no lock; the validated destroy runs under one
    /// brief lock and the response is built after it is released.
    fn destroy<'py>(
        &self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let handles_in = ids.len() * 16;
        let result = {
            let mut world = self.lock()?;
            world.destroy(&ids)
        };
        let count = result.map_err(map_world_error)?;

        let dict = PyDict::new(py);
        dict.set_item("entities", count)?;
        dict.set_item("handles_in", handles_in)?;
        dict.set_item("payload_in", 0usize)?;
        dict.set_item("bytes_in", handles_in)?;
        Ok(dict)
    }

    /// Owned readback of one component.
    ///
    /// `handles` selects and orders rows; `None` reads every live entity in
    /// ascending slot order. Returns the handles, membership flags (`uint8`)
    /// and values as little-endian bytes. Handles are decoded with no lock;
    /// the owned readout is copied under one lock, then Python objects are
    /// built after it is released.
    fn read_component<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        handles: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = match handles {
            Some(object) => Some(decode_handles(py, object)?),
            None => None,
        };
        let handles_in = ids.as_ref().map_or(0, |ids| ids.len() * 16);
        let component = {
            let world = self.lock()?;
            resolve_component(&world, name)
        }
        .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let result = {
            let world = self.lock()?;
            world.read_component(component.index, ids.as_deref())
        };
        let readout = result.map_err(map_world_error)?;

        let handles = handles_bytes(&readout.ids);
        let membership: Vec<u8> = readout
            .membership
            .iter()
            .map(|flag| u8::from(*flag))
            .collect();
        let data = column_bytes(&readout.data);
        let bytes_out = handles.len() + membership.len() + data.len();
        let dict = PyDict::new(py);
        dict.set_item("name", &readout.schema.name)?;
        dict.set_item("kind", readout.schema.element.as_str())?;
        dict.set_item("width", readout.schema.width)?;
        dict.set_item("version", readout.version)?;
        dict.set_item("count", readout.ids.len())?;
        dict.set_item("handles", PyBytes::new(py, &handles))?;
        dict.set_item("membership", PyBytes::new(py, &membership))?;
        dict.set_item("data", PyBytes::new(py, &data))?;
        dict.set_item("handles_in", handles_in)?;
        dict.set_item("payload_in", 0usize)?;
        dict.set_item("bytes_in", handles_in)?;
        dict.set_item("handles_out", handles.len())?;
        dict.set_item("membership_out", membership.len())?;
        dict.set_item("data_out", data.len())?;
        dict.set_item("bytes_out", bytes_out)?;
        Ok(dict)
    }

    /// An owned snapshot of entities, schemas and counters (not values).
    ///
    /// The snapshot is copied under one brief lock; the dictionary is built
    /// after the lock is released.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = {
            let world = self.lock()?;
            world.snapshot()
        };
        let handles = handles_bytes(&snapshot.entities);
        let schemas: Vec<(String, &'static str, u32)> = snapshot
            .schemas
            .iter()
            .map(|schema| (schema.name.clone(), schema.element.as_str(), schema.width))
            .collect();
        let dict = PyDict::new(py);
        dict.set_item("capacity", snapshot.capacity)?;
        dict.set_item("alive_count", snapshot.alive_count)?;
        dict.set_item("high_water", snapshot.high_water)?;
        dict.set_item("version", snapshot.version)?;
        dict.set_item("count", snapshot.entities.len())?;
        dict.set_item("handles", PyBytes::new(py, &handles))?;
        dict.set_item("schemas", schemas)?;
        dict.set_item("handles_in", 0usize)?;
        dict.set_item("payload_in", 0usize)?;
        dict.set_item("bytes_in", 0usize)?;
        dict.set_item("handles_out", handles.len())?;
        dict.set_item("bytes_out", handles.len())?;
        Ok(dict)
    }
}

/// `manifoldx._core_native`
#[pymodule]
fn _core_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    m.add_class::<PyEntitySlots>()?;
    m.add_class::<PyNativeWorld>()?;
    m.add_class::<runtime::PyNativeRuntime>()?;
    graphics::register(m)?;
    stream_builder::register(m)?;
    frames::register(m)?;
    kernel::register(m)?;
    m.add_class::<vehicles::PyVehicleSim>()?;
    m.add_class::<vehicles::PyVehicleFleet>()?;
    m.add_class::<vehicles::PyFleetCheckpoint>()?;
    m.add_class::<vehicles::PyVehicleTerrain>()?;
    m.add_class::<vehicles::PyVehicleRuntime>()?;
    Ok(())
}
