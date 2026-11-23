//! The engine side of S-b: open a game's builder capsule and collect what it
//! pushes, so the spike can time it and compare its bytes with the Python builder.
//!
//! This is a stand-in for the core's streamer: the game and the engine are two
//! separate cdylibs, and only `#[repr(C)]` types and a function pointer cross.

use std::ffi::{c_void, CStr, CString};
use std::ptr::NonNull;
use std::slice;
use std::time::Instant;

use manifoldx_sdk::{BuildOutput, BuilderFn, CellRequest, BUILDER_NAME};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList, PyModule};

#[derive(Default)]
struct Mesh {
    positions: Vec<f32>,
    normals: Vec<f32>,
    indices: Vec<u32>,
    material: u32,
}

#[derive(Default)]
struct Collector {
    meshes: Vec<Mesh>,
    rows: Vec<u8>,
    row_bytes: usize,
    archetype: u32,
    error: Option<String>,
}

unsafe extern "C" fn push_mesh(
    ctx: *mut c_void,
    positions: *const f32,
    vertices: usize,
    normals: *const f32,
    indices: *const u32,
    index_count: usize,
    material: u32,
) -> u32 {
    let collector = unsafe { &mut *(ctx as *mut Collector) };
    let positions = if vertices == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(positions, vertices * 3) }.to_vec()
    };
    let normals = if vertices == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(normals, vertices * 3) }.to_vec()
    };
    let indices = if index_count == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(indices, index_count) }.to_vec()
    };
    collector.meshes.push(Mesh { positions, normals, indices, material });
    (collector.meshes.len() - 1) as u32
}

unsafe extern "C" fn spawn(
    ctx: *mut c_void,
    archetype: u32,
    rows: *const u8,
    row_count: usize,
    row_bytes: usize,
) {
    let collector = unsafe { &mut *(ctx as *mut Collector) };
    collector.archetype = archetype;
    collector.row_bytes = row_bytes;
    collector.rows = if row_count == 0 || row_bytes == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(rows, row_count * row_bytes) }.to_vec()
    };
}

unsafe extern "C" fn error(ctx: *mut c_void, message: *const u8, len: usize) {
    let collector = unsafe { &mut *(ctx as *mut Collector) };
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(message, len) }
    };
    collector.error = Some(String::from_utf8_lossy(bytes).into_owned());
}

fn bytes_of_f32(values: &[f32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

fn bytes_of_u32(values: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

#[pyclass]
struct NativeBuilder {
    function: BuilderFn,
}

#[pymethods]
impl NativeBuilder {
    /// Open a capsule once, at registration. A wrong name, a null pointer or a
    /// non-capsule fails here, never in a frame.
    #[new]
    fn new(builder: &Bound<'_, PyAny>) -> PyResult<Self> {
        let capsule = builder.cast::<PyCapsule>().map_err(|_| {
            PyTypeError::new_err(
                "builder must be a PyCapsule named manifoldx.builder.v1 (a Python callable is the other mode)",
            )
        })?;
        let pointer = capsule.pointer_checked(Some(BUILDER_NAME)).map_err(|_| {
            PyValueError::new_err(
                "builder capsule is not named manifoldx.builder.v1: it was compiled against another SDK version",
            )
        })?;
        let function: BuilderFn = unsafe { std::mem::transmute(pointer.as_ptr()) };
        Ok(Self { function })
    }

    fn build(
        &self,
        py: Python<'_>,
        x: f64,
        y: f64,
        size: f32,
        lod: u32,
        seed: u64,
    ) -> PyResult<Py<PyDict>> {
        let mut collector = Collector::default();
        let mut output = BuildOutput {
            ctx: &mut collector as *mut Collector as *mut c_void,
            push_mesh,
            spawn,
            error,
        };
        let request = CellRequest { x, y, size, lod, seed };

        let started = Instant::now();
        let code = unsafe { (self.function)(&request, &mut output) };
        let ms = started.elapsed().as_secs_f64() * 1000.0;

        if code != 0 {
            let message = collector.error.unwrap_or_else(|| "no message".to_string());
            return Err(PyRuntimeError::new_err(format!(
                "native builder failed (code {code}): {message}"
            )));
        }

        let result = PyDict::new(py);
        result.set_item("ms", ms)?;
        result.set_item("archetype", collector.archetype)?;
        result.set_item("row_bytes", collector.row_bytes)?;
        result.set_item("rows", PyBytes::new(py, &collector.rows))?;
        let meshes = PyList::empty(py);
        for mesh in &collector.meshes {
            let item = PyDict::new(py);
            item.set_item("positions", PyBytes::new(py, bytes_of_f32(&mesh.positions)))?;
            item.set_item("normals", PyBytes::new(py, bytes_of_f32(&mesh.normals)))?;
            item.set_item("indices", PyBytes::new(py, bytes_of_u32(&mesh.indices)))?;
            item.set_item("material", mesh.material)?;
            meshes.append(item)?;
        }
        result.set_item("meshes", meshes)?;
        Ok(result.unbind())
    }
}

/// A test helper: a capsule with any name and pointer, including a null pointer.
/// Only the spike's failure tests use it; the game never exports one.
#[pyfunction]
fn make_capsule(py: Python<'_>, name: &str, pointer: usize) -> PyResult<Py<PyCapsule>> {
    let name: &'static CStr = Box::leak(
        CString::new(name)
            .map_err(|_| PyValueError::new_err("capsule name must not contain a NUL"))?
            .into_boxed_c_str(),
    );
    let pointer = NonNull::new(pointer as *mut c_void)
        .ok_or_else(|| PyValueError::new_err("CPython refuses a capsule with a null pointer"))?;
    Ok(unsafe { PyCapsule::new_with_pointer(py, pointer, name) }?.unbind())
}

#[pymodule]
fn _spike_sb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeBuilder>()?;
    m.add_function(wrap_pyfunction!(make_capsule, m)?)?;
    Ok(())
}
