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
//! into native storage is exported. NumPy inputs are copied once at the
//! boundary into typed Rust vectors. Calls lock one mutex, so a free-threaded
//! interpreter serializes mutations here.

use std::sync::Mutex;

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
    inner: Mutex<World>,
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

#[pymethods]
impl PyNativeWorld {
    #[new]
    fn new(capacity: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(World::new(capacity).map_err(map_world_error)?),
        })
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.lock().expect("world lock");
        format!(
            "_NativeWorld(capacity={}, alive={}, schemas={}, published={})",
            inner.capacity(),
            inner.alive_count(),
            inner.schemas().count(),
            inner.published()
        )
    }

    /// Fixed entity capacity.
    #[getter]
    fn capacity(&self) -> u32 {
        self.inner.lock().expect("world lock").capacity()
    }

    /// Number of live entities.
    #[getter]
    fn alive_count(&self) -> u32 {
        self.inner.lock().expect("world lock").alive_count()
    }

    /// One past the highest slot ever published.
    #[getter]
    fn high_water(&self) -> u32 {
        self.inner.lock().expect("world lock").high_water()
    }

    /// Incremented by each successful schema or data mutation.
    #[getter]
    fn version(&self) -> u64 {
        self.inner.lock().expect("world lock").version()
    }

    /// Whether an entity has ever been published.
    #[getter]
    fn published(&self) -> bool {
        self.inner.lock().expect("world lock").published()
    }

    /// Registered schemas as `[(name, kind, width), ...]`.
    #[getter]
    fn schemas(&self) -> Vec<(String, &'static str, u32)> {
        self.inner
            .lock()
            .expect("world lock")
            .schemas()
            .map(|schema| (schema.name.clone(), schema.element.as_str(), schema.width))
            .collect()
    }

    /// Register one component schema and return its index.
    ///
    /// `kind` is `"f32"` or `"u32"`; `width` is the fixed scalars per row and
    /// may be zero for a marker. A duplicate name, an empty name, an
    /// over-wide row or registration after the first published entity is
    /// rejected without changing live state.
    fn register_component(&self, name: &str, kind: &str, width: u32) -> PyResult<usize> {
        let element = parse_element(kind)?;
        let mut world = self.inner.lock().expect("world lock");
        world
            .register(name, element, width)
            .map_err(map_world_error)
    }

    /// Publish `count` entities with `{component_name: 2-D array}` payloads.
    ///
    /// Each payload is C-contiguous with dtype `float32` or `uint32` and shape
    /// `(count, width)`. A zero-width component accepts `None` or a zero-sized
    /// `(count, 0)` buffer. Returns owned handles (`uint64`, shape `(n, 2)`,
    /// little-endian) plus the input byte count copied at this boundary.
    fn spawn<'py>(
        &self,
        py: Python<'py>,
        count: u32,
        payload: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut world = self.inner.lock().expect("world lock");
        let mut payloads = Vec::with_capacity(payload.len());
        let mut bytes_in = 0usize;
        for (key, value) in payload {
            let name: String = key
                .extract()
                .map_err(|_| PyTypeError::new_err("component payload keys must be strings"))?;
            let component = world
                .component_index(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
            let schema = world
                .schema(component)
                .expect("registered component has a schema");
            let element = schema.element;
            let width = schema.width as usize;
            let data = if value.is_none() {
                if width != 0 {
                    return Err(PyTypeError::new_err(format!(
                        "component '{name}' needs a {} payload of shape ({count}, {width})",
                        element.as_str()
                    )));
                }
                match element {
                    ElementType::F32 => ColumnData::F32(Vec::new()),
                    ElementType::U32 => ColumnData::U32(Vec::new()),
                }
            } else {
                let label = format!("component '{name}' payload");
                match element {
                    ElementType::F32 => ColumnData::F32(read_matrix::<f32>(
                        py,
                        &value,
                        count as usize,
                        width,
                        element,
                        &label,
                    )?),
                    ElementType::U32 => ColumnData::U32(read_matrix::<u32>(
                        py,
                        &value,
                        count as usize,
                        width,
                        element,
                        &label,
                    )?),
                }
            };
            bytes_in += data.byte_len();
            payloads.push(ComponentPayload { component, data });
        }
        let ids = world.spawn(count, &payloads).map_err(map_world_error)?;
        let handles = handles_bytes(&ids);
        let dict = PyDict::new(py);
        dict.set_item("count", ids.len())?;
        dict.set_item("handles", PyBytes::new(py, &handles))?;
        dict.set_item("bytes_in", bytes_in)?;
        dict.set_item("bytes_out", handles.len())?;
        Ok(dict)
    }

    /// Overwrite whole member rows: `data` has shape `(len(handles), width)`.
    ///
    /// Every identity must be live and carry the component. The whole batch is
    /// validated before any row changes.
    fn update_rows<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        handles: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let mut world = self.inner.lock().expect("world lock");
        let component = world
            .component_index(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let schema = world
            .schema(component)
            .expect("registered component has a schema");
        let element = schema.element;
        let width = schema.width as usize;
        let label = format!("component '{name}' rows");
        let payload = match element {
            ElementType::F32 => ColumnData::F32(read_matrix::<f32>(
                py,
                data,
                ids.len(),
                width,
                element,
                &label,
            )?),
            ElementType::U32 => ColumnData::U32(read_matrix::<u32>(
                py,
                data,
                ids.len(),
                width,
                element,
                &label,
            )?),
        };
        let bytes_in = payload.byte_len();
        let rows = world
            .update_rows(component, &ids, payload)
            .map_err(map_world_error)?;
        let dict = PyDict::new(py);
        dict.set_item("rows", rows)?;
        dict.set_item("bytes_in", bytes_in)?;
        Ok(dict)
    }

    /// Overwrite one field: `data` has shape `(len(handles),)`.
    fn update_field<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        field: u32,
        handles: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let mut world = self.inner.lock().expect("world lock");
        let component = world
            .component_index(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let schema = world
            .schema(component)
            .expect("registered component has a schema");
        let element = schema.element;
        let label = format!("component '{name}' field {field}");
        let payload = match element {
            ElementType::F32 => {
                ColumnData::F32(read_field::<f32>(py, data, ids.len(), element, &label)?)
            }
            ElementType::U32 => {
                ColumnData::U32(read_field::<u32>(py, data, ids.len(), element, &label)?)
            }
        };
        let bytes_in = payload.byte_len();
        let rows = world
            .update_field(component, field, &ids, payload)
            .map_err(map_world_error)?;
        let dict = PyDict::new(py);
        dict.set_item("rows", rows)?;
        dict.set_item("bytes_in", bytes_in)?;
        Ok(dict)
    }

    /// Bulk membership flags for `handles`, one `uint8` per entity.
    fn has<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let world = self.inner.lock().expect("world lock");
        let component = world
            .component_index(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let flags = world
            .memberships(component, &ids)
            .map_err(map_world_error)?;
        let bytes: Vec<u8> = flags.iter().map(|flag| u8::from(*flag)).collect();
        let dict = PyDict::new(py);
        dict.set_item("count", flags.len())?;
        dict.set_item("flags", PyBytes::new(py, &bytes))?;
        dict.set_item("bytes_out", bytes.len())?;
        Ok(dict)
    }

    /// Destroy live entities and clear their rows and membership.
    fn destroy<'py>(
        &self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = decode_handles(py, handles)?;
        let bytes_in = ids.len() * 16;
        let count = self
            .inner
            .lock()
            .expect("world lock")
            .destroy(&ids)
            .map_err(map_world_error)?;
        let dict = PyDict::new(py);
        dict.set_item("entities", count)?;
        dict.set_item("bytes_in", bytes_in)?;
        Ok(dict)
    }

    /// Owned readback of one component.
    ///
    /// `handles` selects and orders rows; `None` reads every live entity in
    /// ascending slot order. Returns the handles, membership flags (`uint8`)
    /// and values as little-endian bytes.
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
        let world = self.inner.lock().expect("world lock");
        let component = world
            .component_index(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown component '{name}'")))?;
        let readout = world
            .read_component(component, ids.as_deref())
            .map_err(map_world_error)?;
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
        dict.set_item("count", readout.ids.len())?;
        dict.set_item("handles", PyBytes::new(py, &handles))?;
        dict.set_item("membership", PyBytes::new(py, &membership))?;
        dict.set_item("data", PyBytes::new(py, &data))?;
        dict.set_item("bytes_out", bytes_out)?;
        Ok(dict)
    }

    /// An owned snapshot of entities, schemas and counters (not values).
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self.inner.lock().expect("world lock").snapshot();
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
    Ok(())
}
