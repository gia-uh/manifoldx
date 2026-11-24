//! Python bindings for the ManifoldX runtime core.
//!
//! Phase 0 ships one extension, `manifoldx._core_native`, whose bootstrap
//! reports what the core crate actually is. The runtime is not ported yet,
//! so nothing here claims otherwise. The name deliberately differs from the
//! `_native` module the spike loader in `manifoldx.native` looks for, so the
//! nbody/boids spikes keep working beside it.

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

/// `manifoldx._core_native`
#[pymodule]
fn _core_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    Ok(())
}
