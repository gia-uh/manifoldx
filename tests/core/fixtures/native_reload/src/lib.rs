//! A tiny native cell builder for the reload test.
//!
//! `MANIFOLDX_NATIVE_VARIANT` picks the bytes at compile time: a positive
//! number is a valid build whose material and row carry that number (1, 2 and
//! later reloads use the same fixture); `abi` exports the same builder under
//! `manifoldx.builder.v2` so the loader must reject it. A request with a
//! negative x sleeps, so a swap can be raced against work already running.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Duration;

use manifoldx_sdk::{guarded, BuildOutput, CellRequest, Output};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyModule};

fn variant() -> u32 {
    match option_env!("MANIFOLDX_NATIVE_VARIANT") {
        Some("abi") => 0,
        Some(value) => value.parse().unwrap_or(1),
        None => 1,
    }
}

fn name() -> &'static std::ffi::CStr {
    if variant() == 0 {
        c"manifoldx.builder.v2"
    } else {
        manifoldx_sdk::BUILDER_NAME
    }
}

fn build_cell_impl(request: &CellRequest, out: &mut Output) -> Result<(), String> {
    if request.x < 0.0 {
        std::thread::sleep(Duration::from_millis(120));
    }
    let material = variant();
    let positions = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let normals = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let indices = [0u32, 1, 2];
    out.mesh(&positions, &normals, &indices, material);
    let mut row = Vec::with_capacity(12);
    row.extend_from_slice(&material.to_le_bytes());
    row.extend_from_slice(&0f32.to_le_bytes());
    row.extend_from_slice(&0u32.to_le_bytes());
    out.spawn(0, &row, 12);
    Ok(())
}

unsafe extern "C" fn build_cell(request: *const CellRequest, out: *mut BuildOutput) -> i32 {
    if request.is_null() || out.is_null() {
        return 3;
    }
    let request = unsafe { &*request };
    let mut output = unsafe { Output::new(out) };
    guarded(&mut output, |output| build_cell_impl(request, output))
}

#[pymodule]
fn native_reload(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let pointer =
        NonNull::new(build_cell as *mut c_void).expect("a function pointer is never null");
    let capsule = unsafe { PyCapsule::new_with_pointer(py, pointer, name()) }?;
    module.add("build_cell", capsule)?;
    module.add("sdk_version", "1")?;
    Ok(())
}
