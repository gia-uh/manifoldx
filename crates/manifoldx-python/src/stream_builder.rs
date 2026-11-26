//! Own the output of a trusted in-process v1 capsule before its call returns.

use std::ffi::c_void;
use std::slice;

use manifoldx_sdk::{BuildOutput, BuilderFn, CellRequest, BUILDER_NAME};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};

const MAX_BYTES: usize = 64 << 20;
const MAX_ERROR_BYTES: usize = 4096;

type SpawnRows = (u32, usize, Vec<u8>);

struct Mesh {
    material: u32,
    positions: Vec<f32>,
    normals: Vec<f32>,
    indices: Vec<u32>,
}

#[derive(Default)]
struct Output {
    meshes: Vec<Mesh>,
    rows: Vec<SpawnRows>,
    bytes: usize,
    error: Option<String>,
}

impl Output {
    fn charge(&mut self, bytes: Option<usize>) -> bool {
        let Some(total) = bytes.and_then(|size| self.bytes.checked_add(size)) else {
            self.error = Some("native cell output size overflows".into());
            return false;
        };
        if total > MAX_BYTES {
            self.error = Some("native cell output exceeds the byte limit".into());
            return false;
        }
        self.bytes = total;
        self.error.is_none()
    }
}

fn valid_ptr<T>(ptr: *const T) -> bool {
    !ptr.is_null() && ptr.is_aligned()
}

unsafe extern "C" fn mesh(
    context: *mut c_void,
    positions: *const f32,
    vertices: usize,
    normals: *const f32,
    indices: *const u32,
    count: usize,
    material: u32,
) -> u32 {
    let output = unsafe { &mut *context.cast::<Output>() };
    if vertices == 0
        || count == 0
        || !count.is_multiple_of(3)
        || !valid_ptr(positions)
        || !valid_ptr(normals)
        || !valid_ptr(indices)
    {
        output.error = Some("invalid native mesh buffer or triangle count".into());
        return u32::MAX;
    }
    let bytes = vertices
        .checked_mul(24)
        .and_then(|v| count.checked_mul(4)?.checked_add(v));
    if !output.charge(bytes) {
        return u32::MAX;
    }
    // The capsule owns valid allocations for the duration of this sink call.
    let positions = unsafe { slice::from_raw_parts(positions, vertices * 3) };
    let normals = unsafe { slice::from_raw_parts(normals, vertices * 3) };
    let indices = unsafe { slice::from_raw_parts(indices, count) };
    if positions.iter().chain(normals).any(|x| !x.is_finite())
        || indices.iter().any(|&index| index as usize >= vertices)
    {
        output.error = Some("invalid native vertex value or index".into());
        return u32::MAX;
    }
    let id = output.meshes.len() as u32;
    output.meshes.push(Mesh {
        material,
        positions: positions.to_vec(),
        normals: normals.to_vec(),
        indices: indices.to_vec(),
    });
    id
}

unsafe extern "C" fn spawn(
    context: *mut c_void,
    archetype: u32,
    rows: *const u8,
    count: usize,
    width: usize,
) {
    let output = unsafe { &mut *context.cast::<Output>() };
    if count == 0 {
        return;
    }
    if width == 0 || !valid_ptr(rows) {
        output.error = Some("invalid native row buffer or width".into());
        return;
    }
    let bytes = count.checked_mul(width);
    if output.charge(bytes) {
        let rows = unsafe { slice::from_raw_parts(rows, bytes.unwrap()) };
        output.rows.push((archetype, width, rows.to_vec()));
    }
}

unsafe extern "C" fn error(context: *mut c_void, message: *const u8, len: usize) {
    let output = unsafe { &mut *context.cast::<Output>() };
    let text = if len == 0 || !valid_ptr(message) {
        "native builder reported an error".into()
    } else {
        let bytes = unsafe { slice::from_raw_parts(message, len.min(MAX_ERROR_BYTES)) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    output.error.get_or_insert(text);
}

#[pyclass(name = "_StreamBuilder", module = "manifoldx._core_native", frozen)]
struct StreamBuilder {
    _capsule: Py<PyCapsule>,
    function: BuilderFn,
}

type MeshResult<'py> = (
    u32,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
);
type RowResult<'py> = (u32, usize, Bound<'py, PyBytes>);

#[pymethods]
impl StreamBuilder {
    #[new]
    fn new(capsule: Bound<'_, PyCapsule>) -> PyResult<Self> {
        let pointer = capsule.pointer_checked(Some(BUILDER_NAME)).map_err(|_| {
            let actual = capsule
                .name()
                .ok()
                .flatten()
                .map(|name| unsafe { name.as_cstr() }.to_string_lossy().into_owned())
                .unwrap_or_default();
            PyValueError::new_err(format!(
                "expected builder capsule manifoldx.builder.v1, got {actual:?}"
            ))
        })?;
        Ok(Self {
            _capsule: capsule.unbind(),
            function: unsafe { std::mem::transmute::<*mut c_void, BuilderFn>(pointer.as_ptr()) },
        })
    }

    fn build<'py>(
        &self,
        py: Python<'py>,
        x: f64,
        y: f64,
        size: f32,
        lod: u32,
        seed: u64,
    ) -> PyResult<(Vec<MeshResult<'py>>, Vec<RowResult<'py>>)> {
        if !x.is_finite() || !y.is_finite() || !size.is_finite() || size <= 0.0 {
            return Err(PyValueError::new_err(
                "cell request needs a finite origin and positive size",
            ));
        }
        let function = self.function;
        let (code, collected) = py.detach(move || {
            let mut collected = Output::default();
            let mut output = BuildOutput {
                ctx: (&mut collected as *mut Output).cast(),
                push_mesh: mesh,
                spawn,
                error,
            };
            let request = CellRequest {
                x,
                y,
                size,
                lod,
                seed,
            };
            let code = unsafe { function(&request, &mut output) };
            (code, collected)
        });
        if let Some(message) = collected.error {
            return Err(PyRuntimeError::new_err(format!(
                "cell builder failed ({code}): {message}"
            )));
        }
        if code != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "cell builder failed ({code})"
            )));
        }
        let meshes = collected
            .meshes
            .iter()
            .map(|mesh| {
                (
                    mesh.material,
                    as_bytes(py, &mesh.positions),
                    as_bytes(py, &mesh.normals),
                    as_bytes(py, &mesh.indices),
                )
            })
            .collect();
        let rows = collected
            .rows
            .iter()
            .map(|(id, size, bytes)| (*id, *size, PyBytes::new(py, bytes)))
            .collect();
        Ok((meshes, rows))
    }
}

fn as_bytes<'py, T>(py: Python<'py>, values: &[T]) -> Bound<'py, PyBytes> {
    // Only f32/u32 arrays reach this function.
    let bytes = unsafe {
        slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    PyBytes::new(py, bytes)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StreamBuilder>()
}
